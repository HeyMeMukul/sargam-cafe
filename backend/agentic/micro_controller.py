"""Micro-operation controller for evidence-cited score revision."""
from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .contracts import AgentTrace, ToolCall
from .memory import EpisodicMemory
from .operations import apply_operations
from .runtime_tools import build_runtime_registry
from .skill_registry import SkillRegistry


class MicroOperationAgent:
    def __init__(self, client: Any, model: str | None = None, max_tool_calls: int = 10, skill_dir: str | Path | None = None):
        self.client = client
        self.model = model or os.getenv("SARGAM_AGENTIC_MODEL", "gpt-5")
        self.max_tool_calls = max_tool_calls
        self.registry = build_runtime_registry()
        self.skills = SkillRegistry(skill_dir or Path(__file__).resolve().parents[1] / "skills")
        self.memory = EpisodicMemory()

    def _tool_definitions(self) -> list[dict[str, Any]]:
        from .controller import PianistAgent
        return PianistAgent(self.client, self.model, self.max_tool_calls)._tool_definitions()

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            for match in re.finditer(r"\{", text):
                try:
                    value, _ = json.JSONDecoder().raw_decode(text[match.start():])
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict) and "operations" in value:
                    return value
        return None

    @staticmethod
    def _compact(value: Any, max_chars: int = 16000) -> Any:
        text = json.dumps(value, ensure_ascii=False)
        if len(text) <= max_chars:
            return value
        if isinstance(value, dict):
            result = dict(value)
            if isinstance(result.get("frames"), list):
                result["frames"] = result["frames"][:100]
            return result
        return text[:max_chars]

    def run(self, audio_path: str) -> dict[str, Any]:
        manifest = self.registry.call("get_track_manifest", audio_path=audio_path)
        duration = float(manifest["duration"])
        baseline_result = self.registry.call("get_note_candidates", audio_path=audio_path, start=0.0, end=duration)
        baseline = baseline_result.get("events", [])
        for index, event in enumerate(baseline):
            event.setdefault("event_id", f"baseline-{index}")
            if not event.get("evidence_refs"):
                event["evidence_refs"] = [{
                    "tool": "get_note_candidates",
                    "artifact_id": baseline_result.get("artifact_id", "baseline"),
                    "feature": "baseline_event",
                    "start": event.get("start"),
                    "end": event.get("end"),
                    "confidence": event.get("pitch_confidence") or event.get("voicing_confidence"),
                    "summary": "baseline candidate; no mutation authority",
                }]
        trace = AgentTrace(
            trace_id=f"trace-{uuid.uuid4().hex[:12]}",
            audio_sha256=manifest["audio_sha256"],
            skill_citations=self.skills.citation_bundle("note boundary retrigger missing onset offset pitch pianist", limit=6),
        )
        skill_call = self.registry.call("retrieve_skills", query="note boundary retrigger missing onset offset pitch pianist", limit=6)
        trace.tool_calls.append(ToolCall(
            call_id="skill-bootstrap",
            tool="retrieve_skills",
            arguments={"query": "note boundary retrigger missing onset offset pitch pianist", "limit": 6},
            rationale="load relevant skills before proposing operations",
            result_artifact_id=skill_call.get("artifact_id"),
            status="completed",
        ))
        system = (
            "You are a conservative pianist score editor. You do not hear audio directly; use the tools. "
            "Do not return a replacement melody. Return only a JSON object with an operations array. "
            "Allowed operations are keep, insert_event, split_event, merge_events, retarget_pitch, shift_boundary. "
            "Every mutation must cite at least one targeted non-baseline tool artifact (inspect_pitch_window, "
            "inspect_boundary_window, get_harmonic_context, or align_lyrics_or_phonemes) and include a precise reason. "
            "Use keep when evidence is insufficient. Prefer zero mutations to an unsupported change. "
            "The baseline event IDs are authoritative identifiers. Do not use scale membership alone as evidence."
        )
        user = {
            "task": "Find missing repeated attacks and wrong boundary/pitch events in the supplied melody, but make only evidence-backed micro-edits.",
            "manifest": manifest,
            "baseline_events": baseline,
            "skills": trace.skill_citations,
            "required_output": {
                "operations": [
                    {"op": "keep", "reason": "no supported change", "evidence_refs": []},
                    {"op": "split_event", "event_id": "baseline-0", "split_time": 1.5, "reason": "...", "evidence_refs": [{"tool": "inspect_boundary_window", "artifact_id": "...", "feature": "..."}]}
                ],
                "state": "accepted|uncertain",
                "decision_reason": "...",
                "unresolved_questions": []
            }
        }
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ]
        for _ in range(self.max_tool_calls + 1):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self._tool_definitions(),
                tool_choice="auto",
                max_completion_tokens=4000,
                extra_body={"reasoning": {"effort": "medium"}},
            )
            message = response.choices[0].message
            calls = getattr(message, "tool_calls", None) or []
            if not calls:
                final = self._parse_json(message.content or "")
                if final is not None:
                    break
                messages.append({"role": "user", "content": "Return only the operations JSON now."})
                continue
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [{"id": call.id, "type": "function", "function": {"name": call.function.name, "arguments": call.function.arguments}} for call in calls],
            })
            for call in calls:
                args = json.loads(call.function.arguments or "{}")
                record = ToolCall(call.id, call.function.name, args, "agent-requested targeted evidence", status="planned")
                try:
                    result = self.registry.call(call.function.name, **args)
                    record.status = "completed"
                    record.result_artifact_id = result.get("artifact_id") if isinstance(result, dict) else None
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(self._compact(result), ensure_ascii=False)})
                except Exception as exc:
                    record.status = "failed"
                    record.error = f"{type(exc).__name__}: {exc}"
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps({"error": record.error})})
                trace.tool_calls.append(record)
        else:
            final = None
        operations = (final or {}).get("operations", []) if final else []
        promoted = False
        unresolved = list((final or {}).get("unresolved_questions", []))
        decision_reason = (final or {}).get("decision_reason", "")
        try:
            events = apply_operations(baseline, operations, duration)
            promoted = any(operation.get("op") != "keep" for operation in operations)
        except Exception as exc:
            events = baseline
            promoted = False
            unresolved.append(f"operations rejected: {type(exc).__name__}: {exc}")
            decision_reason = "baseline preserved because proposed operations failed validation"
        score = self.registry.call("score_hypothesis", audio_path=audio_path, events=events)
        audition = self.registry.call("audition_phrase", audio_path=audio_path, events=events, start=0.0, end=min(duration, 10.0))
        trace.tool_calls.extend([
            ToolCall(call_id="validation-score", tool="score_hypothesis", arguments={"audio_path": audio_path}, rationale="validate micro-operation result", result_artifact_id=score.get("artifact_id"), status="completed"),
            ToolCall(call_id="validation-audition", tool="audition_phrase", arguments={"audio_path": audio_path, "start": 0.0, "end": min(duration, 10.0)}, rationale="audition micro-operation result", result_artifact_id=audition.get("artifact_id"), status="completed"),
        ])
        trace_payload = {
            "trace_id": trace.trace_id,
            "audio_sha256": trace.audio_sha256,
            "skill_citations": trace.skill_citations,
            "tool_calls": [asdict(call) for call in trace.tool_calls],
            "operations": operations,
            "promoted": promoted,
        }
        memory = self.memory.write(trace_payload, outcome="accepted" if promoted else "uncertain", tags=["melody", "pianist", "micro-operations"])
        return {
            "trace": trace_payload,
            "operations": operations,
            "promoted": promoted,
            "events": events,
            "unresolved_questions": unresolved,
            "decision_reason": decision_reason,
            "validation": {"score": score, "audition": audition},
            "memory": {"memory_id": memory["memory_id"], "outcome": memory["outcome"]},
        }
