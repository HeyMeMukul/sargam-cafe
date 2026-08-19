"""Tool-using pianist controller.

The controller is intentionally audio-blind: it reasons over specialist-tool
artifacts and may ask for targeted observations. This makes every revision
traceable and prevents a text model from pretending it directly heard a note.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .contracts import AgentTrace, EvidenceRef, HypothesisVersion, NoteCandidate, ToolCall
from .runtime_tools import build_runtime_registry
from .skill_registry import SkillRegistry
from .memory import EpisodicMemory


FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis_id": {"type": "string"},
        "state": {"type": "string", "enum": ["candidate", "accepted", "uncertain"]},
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string"},
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "midi": {"type": ["integer", "null"]},
                    "raw_midi_float": {"type": ["number", "null"]},
                    "pitch_confidence": {"type": ["number", "null"]},
                    "voicing_confidence": {"type": ["number", "null"]},
                    "onset_confidence": {"type": ["number", "null"]},
                    "offset_confidence": {"type": ["number", "null"]},
                    "velocity": {"type": ["number", "null"]},
                    "articulation": {"type": "string"},
                    "render_role": {"type": "string"},
                    "state": {"type": "string"},
                    "alternatives": {"type": "array", "items": {"type": "object"}},
                    "evidence_refs": {"type": "array", "items": {"type": "object"}},
                    "agent_reason": {"type": "string"},
                },
                "required": ["event_id", "start", "end", "midi", "agent_reason", "evidence_refs"],
                "additionalProperties": True,
            },
        },
        "unresolved_questions": {"type": "array", "items": {"type": "string"}},
        "decision_reason": {"type": "string"},
    },
    "required": ["hypothesis_id", "state", "events", "unresolved_questions", "decision_reason"],
    "additionalProperties": False,
}


class PianistAgent:
    def __init__(self, client: Any, model: str | None = None, max_tool_calls: int = 10, skill_dir: str | Path | None = None):
        self.client = client
        self.model = model or os.getenv("SARGAM_AGENTIC_MODEL", "gpt-5")
        self.max_tool_calls = max_tool_calls
        self.registry = build_runtime_registry()
        self.skill_registry = SkillRegistry(skill_dir or Path(__file__).resolve().parents[1] / "skills")
        self.memory = EpisodicMemory()

    def _tool_definitions(self) -> list[dict[str, Any]]:
        definitions = []
        for spec in self.registry.describe():
            if spec["mutating"] or spec["name"] == "commit_hypothesis":
                continue
            properties = {key: {"type": "string"} for key in spec["required_arguments"] + spec["optional_arguments"]}
            for key in ("start", "end", "min_confidence"):
                if key in properties:
                    properties[key] = {"type": "number"}
            if "events" in properties:
                properties["events"] = {"type": "array", "items": {"type": "object"}}
            if "transcribers" in properties:
                properties["transcribers"] = {"type": "array", "items": {"type": "string"}}
            definitions.append({
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": spec["required_arguments"],
                        "additionalProperties": False,
                    },
                },
            })
        return definitions

    @staticmethod
    def _compact(value: Any, max_chars: int = 12000) -> Any:
        if isinstance(value, dict):
            copied = dict(value)
            if isinstance(copied.get("frames"), list) and len(copied["frames"]) > 80:
                copied["frames"] = copied["frames"][:80]
                copied["frames_truncated"] = True
            if isinstance(copied.get("strongest_local_frames"), list) and len(copied["strongest_local_frames"]) > 20:
                copied["strongest_local_frames"] = copied["strongest_local_frames"][:20]
            text = json.dumps(copied, ensure_ascii=False)
            if len(text) <= max_chars:
                return copied
            return {"artifact_id": copied.get("artifact_id"), "summary": text[:max_chars], "truncated": True}
        return value

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            pass
        for match in re.finditer(r"\{", text):
            try:
                value, _ = json.JSONDecoder().raw_decode(text[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "events" in value:
                return value
        return None

    def run(self, audio_path: str, duration: float | None = None) -> dict[str, Any]:
        manifest = self.registry.call("get_track_manifest", audio_path=audio_path)
        duration = duration or float(manifest["duration"])
        candidates = self.registry.call("get_note_candidates", audio_path=audio_path, start=0.0, end=duration)
        baseline_artifact = candidates.get("artifact_id", "baseline-candidates")
        for index, event in enumerate(candidates.get("events", [])):
            event.setdefault("event_id", f"baseline-{index}")
            if not event.get("evidence_refs"):
                event["evidence_refs"] = [{
                "tool": "get_note_candidates",
                "artifact_id": baseline_artifact,
                "feature": "baseline_event",
                "start": event.get("start"),
                "end": event.get("end"),
                "confidence": event.get("pitch_confidence") or event.get("voicing_confidence"),
                "summary": "production extractor candidate; preserved until targeted evidence supports revision",
                }]
        memories = self.memory.retrieve(manifest["audio_sha256"], tags=["melody", "pianist"], limit=3)
        skills = self.skill_registry.citation_bundle(
            "note boundary retrigger missing melody pitch onset offset audition pianist performance",
            limit=6,
        )
        trace = AgentTrace(trace_id=f"trace-{uuid.uuid4().hex[:12]}", audio_sha256=manifest["audio_sha256"], skill_citations=skills)
        bootstrap_skill = self.registry.call(
            "retrieve_skills",
            query="note boundary pitch onset offset repeated attack pianist evidence",
            limit=6,
        )
        trace.tool_calls.append(ToolCall(
            call_id="skill-bootstrap",
            tool="retrieve_skills",
            arguments={"query": "note boundary pitch onset offset repeated attack pianist evidence", "limit": 6},
            rationale="load task-relevant project skills before hypothesis formation",
            result_artifact_id=bootstrap_skill.get("artifact_id"),
            status="completed",
        ))
        system = (
            "You are the Sargam Pianist Controller. You do not hear audio directly. "
            "Use specialist tools to inspect evidence and make a reversible note hypothesis. "
            "Never invent a note from thaat, beat, genre, or familiarity. A missing event may be "
            "proposed only after a targeted tool observation supports its onset, pitch, and voicing. "
            "Every final event must include evidence_refs as objects with tool, artifact_id, feature, "
            "start, end, confidence, and summary. Every changed event must include a non-empty agent_reason. "
            "When evidence conflicts, query a narrower window or a second tool. Preserve alternatives. "
            "You have a bounded tool budget. At the end return JSON matching the supplied hypothesis contract."
        )
        user = {
            "task": "Transcribe the dominant sung melody as a sequence of piano note events and make it recognizable, then provide a faithful performance plan.",
            "track_manifest": manifest,
            "baseline_candidates": self._compact(candidates),
            "skill_citations": skills,
            "episodic_memories": [
                {"memory_id": memory.get("memory_id"), "outcome": memory.get("outcome"), "tags": memory.get("tags", [])}
                for memory in memories
            ],
            "skill_bootstrap_artifact": bootstrap_skill,
            "tool_policy": "Use tools for local disagreements, missing repeated attacks, and suspicious pitch/boundary events; do not request the whole waveform as prose. Before finalizing, use at least one targeted audio tool and preserve the baseline if evidence is insufficient.",
        }
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ]
        tool_calls_used = 0
        for _ in range(self.max_tool_calls + 1):
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "tools": self._tool_definitions(),
                "tool_choice": "auto",
                "max_completion_tokens": 5000,
                "extra_body": {"reasoning": {"effort": "medium"}},
            }
            response = self.client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []
            if not tool_calls:
                final = self._parse_json(message.content or "")
                if final is not None:
                    break
                messages.append({"role": "user", "content": "Return only the required JSON hypothesis object now. Do not include prose."})
                continue
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {"id": call.id, "type": "function", "function": {"name": call.function.name, "arguments": call.function.arguments}}
                    for call in tool_calls
                ],
            })
            for call in tool_calls:
                if tool_calls_used >= self.max_tool_calls:
                    break
                tool_calls_used += 1
                name = call.function.name
                args = json.loads(call.function.arguments or "{}")
                record = ToolCall(call.id, name, args, "agent-requested specialist observation", status="planned")
                try:
                    result = self.registry.call(name, **args)
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
        baseline_signature = [
            (round(float(event.get("start", 0)), 4), round(float(event.get("end", 0)), 4), event.get("midi"))
            for event in candidates.get("events", [])
        ]
        final_events = (final or {}).get("events", [])
        final_signature = [
            (round(float(event.get("start", 0)), 4), round(float(event.get("end", 0)), 4), event.get("midi"))
            for event in final_events
        ]
        changed_sequence = final_signature != baseline_signature
        nonbaseline_refs = [
            ref for event in final_events for ref in event.get("evidence_refs", [])
            if isinstance(ref, dict) and ref.get("tool") != "get_note_candidates"
        ]
        if changed_sequence and (not nonbaseline_refs or any(not event.get("agent_reason") for event in final_events)):
            final = {
                **(final or {}),
                "events": [],
                "unresolved_questions": list((final or {}).get("unresolved_questions", [])) + [
                    "sequence revision lacked targeted non-baseline evidence and per-event reasons; revision rejected"
                ],
            }
        unsupported_events = [event for event in (final or {}).get("events", []) if not event.get("evidence_refs")]
        if unsupported_events:
            final = {
                **(final or {}),
                "events": [],
                "unresolved_questions": list((final or {}).get("unresolved_questions", [])) + [
                    f"{len(unsupported_events)} agent events lacked evidence_refs; revision rejected"
                ],
            }
        if final is None or not final.get("events"):
            reason = "agent returned no usable evidence-cited event sequence within the tool budget"
            if final and final.get("decision_reason"):
                reason = final["decision_reason"]
            final = {
                "hypothesis_id": (final or {}).get("hypothesis_id", f"h-{uuid.uuid4().hex[:10]}"),
                "state": "uncertain",
                "events": candidates.get("events", []),
                "unresolved_questions": list((final or {}).get("unresolved_questions", [])) + [reason],
                "decision_reason": "fallback to baseline candidates; no unsupported mutation applied",
            }
        hypothesis = HypothesisVersion(
            hypothesis_id=str(final.get("hypothesis_id", f"h-{uuid.uuid4().hex[:10]}")),
            version=0,
            parent_version=None,
            state=final.get("state", "uncertain"),
            events=[],
            unresolved_questions=final.get("unresolved_questions", []),
            tool_calls=trace.tool_calls,
            decision_reason=final.get("decision_reason", ""),
        )
        for raw in final.get("events", []):
            try:
                event = NoteCandidate(
                    event_id=str(raw.get("event_id", f"event-{len(hypothesis.events)}")),
                    start=float(raw["start"]),
                    end=float(raw["end"]),
                    midi=int(raw["midi"]) if raw.get("midi") is not None else None,
                    raw_midi_float=raw.get("raw_midi_float"),
                    pitch_confidence=raw.get("pitch_confidence"),
                    voicing_confidence=raw.get("voicing_confidence"),
                    onset_confidence=raw.get("onset_confidence"),
                    offset_confidence=raw.get("offset_confidence"),
                    velocity=raw.get("velocity"),
                    articulation=raw.get("articulation", "normal"),
                    render_role=raw.get("render_role", "attack"),
                    state=raw.get("state", "candidate"),
                    alternatives=raw.get("alternatives", []),
                    evidence_refs=[
                        ref if isinstance(ref, EvidenceRef) else EvidenceRef(**ref)
                        for ref in raw.get("evidence_refs", [])
                        if isinstance(ref, (dict, EvidenceRef))
                    ],
                    agent_reason=raw.get("agent_reason", ""),
                )
                event.validate()
                hypothesis.events.append(event)
            except (KeyError, TypeError, ValueError):
                continue
        validation_score = self.registry.call(
            "score_hypothesis", audio_path=audio_path, events=[asdict(event) for event in hypothesis.events]
        )
        validation_audition = self.registry.call(
            "audition_phrase", audio_path=audio_path, events=[asdict(event) for event in hypothesis.events], start=0.0, end=min(duration, 10.0)
        )
        trace.tool_calls.extend([
            ToolCall(
                call_id="validation-score",
                tool="score_hypothesis",
                arguments={"audio_path": audio_path, "events": [asdict(event) for event in hypothesis.events]},
                rationale="promotion gate",
                result_artifact_id=validation_score.get("artifact_id"),
                status="completed",
            ),
            ToolCall(
                call_id="validation-audition",
                tool="audition_phrase",
                arguments={"audio_path": audio_path, "start": 0.0, "end": min(duration, 10.0)},
                rationale="promotion gate",
                result_artifact_id=validation_audition.get("artifact_id"),
                status="completed",
            ),
        ])
        trace.hypotheses.append(hypothesis)
        trace.final_hypothesis_id = hypothesis.hypothesis_id
        trace.validate()
        trace_payload = {
            "trace_id": trace.trace_id,
            "audio_sha256": trace.audio_sha256,
            "skill_citations": trace.skill_citations,
            "tool_calls": [asdict(call) for call in trace.tool_calls],
            "final_hypothesis_id": trace.final_hypothesis_id,
        }
        memory_record = self.memory.write(trace_payload, outcome=hypothesis.state, tags=["melody", "pianist"])
        return {
            "trace": {
                "trace_id": trace.trace_id,
                "audio_sha256": trace.audio_sha256,
                "skill_citations": trace.skill_citations,
                "tool_calls": [asdict(call) for call in trace.tool_calls],
                "final_hypothesis_id": trace.final_hypothesis_id,
            },
            "hypothesis": {
                "hypothesis_id": hypothesis.hypothesis_id,
                "state": hypothesis.state,
                "events": [asdict(event) for event in hypothesis.events],
                "unresolved_questions": hypothesis.unresolved_questions,
                "decision_reason": hypothesis.decision_reason,
            },
            "memory": {"memory_id": memory_record["memory_id"], "outcome": memory_record["outcome"]},
            "validation": {
                "score": validation_score,
                "audition": validation_audition,
            },
        }
