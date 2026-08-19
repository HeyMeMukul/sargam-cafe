"""Adapter from the legacy OpenCode subprocess runner to micro-operations."""
from __future__ import annotations

import json
import re
from typing import Any, Awaitable, Callable

from .operations import apply_operations


JSON_OBJECT_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def parse_operations(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    match = JSON_OBJECT_RE.search(text)
    candidates = [match.group(1)] if match else []
    candidates.append(text)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict) and "operations" in value:
                return value
        except json.JSONDecodeError:
            pass
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "operations" in value:
            return value
    return None


async def run_opencode_micro_agent(
    audio_path: str,
    duration: float,
    baseline: list[dict[str, Any]],
    run_agent_stream: Callable[..., Awaitable[str | None]],
    log_callback: Callable[[str], Awaitable[None]],
    model: str,
    max_tool_calls: int,
    cost_tracker: Any = None,
) -> dict[str, Any]:
    working_baseline = []
    for index, event in enumerate(baseline):
        item = dict(event)
        item.setdefault("event_id", f"baseline-{index}")
        working_baseline.append(item)
    prompt = (
        f"Use the sargam-pianist agent for this exact audio path: {audio_path}. "
        f"Track duration is {duration:.3f}s. Inspect the baseline through the typed tools and "
        f"return only the micro-operation JSON object. The baseline event IDs are: "
        f"{[event['event_id'] for event in working_baseline]}. "
        f"Do not return a replacement melody. Use at most {max_tool_calls} specialist calls."
    )
    cmd = [
        "opencode", "run", "--agent", "sargam-pianist", "--model", model,
        "--format", "json", "--log-level", "ERROR", "--auto", prompt,
    ]
    raw = await run_agent_stream(cmd, log_callback, cost_tracker=cost_tracker)
    parsed = parse_operations(raw)
    if not parsed:
        return {
            "promoted": False,
            "state": "uncertain",
            "operations": [],
            "events": baseline,
            "unresolved_questions": ["pianist agent returned no operations JSON"],
        }
    operations = parsed.get("operations") or []
    try:
        events = apply_operations(working_baseline, operations, duration)
        mutations = [operation for operation in operations if operation.get("op") != "keep"]
        promoted = parsed.get("state") == "accepted" and bool(mutations)
        if not promoted:
            events = baseline
        return {
            "promoted": promoted,
            "state": parsed.get("state", "uncertain"),
            "operations": operations,
            "events": events,
            "unresolved_questions": parsed.get("unresolved_questions", []),
            "decision_reason": parsed.get("decision_reason", ""),
        }
    except Exception as exc:
        await log_callback(f"[System] Pianist micro-operations rejected safely ({type(exc).__name__}).")
        return {
            "promoted": False,
            "state": "uncertain",
            "operations": operations,
            "events": baseline,
            "unresolved_questions": [f"operations rejected: {type(exc).__name__}: {exc}"],
            "decision_reason": "baseline preserved",
        }
