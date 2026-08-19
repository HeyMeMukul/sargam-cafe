"""Reversible, evidence-gated operations over a baseline note sequence."""
from __future__ import annotations

from copy import deepcopy
from typing import Any


MUTATIONS = {"insert_event", "split_event", "merge_events", "retarget_pitch", "shift_boundary"}


def _refs(operation: dict[str, Any]) -> list[dict[str, Any]]:
    refs = operation.get("evidence_refs") or []
    if not isinstance(refs, list) or not refs:
        raise ValueError("every score operation requires evidence_refs")
    if not any(isinstance(ref, dict) and ref.get("tool") != "get_note_candidates" for ref in refs):
        raise ValueError("score mutation requires targeted non-baseline evidence")
    return refs


def _reason(operation: dict[str, Any]) -> str:
    reason = str(operation.get("reason", "")).strip()
    if not reason:
        raise ValueError("every score operation requires a reason")
    return reason


def validate_operations(operations: list[dict[str, Any]], duration: float) -> None:
    for operation in operations:
        kind = operation.get("op")
        if kind == "keep":
            continue
        if kind not in MUTATIONS:
            raise ValueError(f"unsupported score operation: {kind}")
        _refs(operation)
        _reason(operation)
        if kind == "insert_event":
            event = operation.get("event") or {}
            if not 0 <= float(event.get("start", -1)) < float(event.get("end", -1)) <= duration:
                raise ValueError("insert_event timing is invalid")
            if event.get("midi") is None or not 0 <= int(event["midi"]) <= 127:
                raise ValueError("insert_event requires valid MIDI")
        elif kind == "split_event":
            split = float(operation.get("split_time", -1))
            if not 0 <= split <= duration:
                raise ValueError("split time is outside the track")
        elif kind == "merge_events":
            if len(operation.get("event_ids", [])) < 2:
                raise ValueError("merge_event requires at least two event IDs")
        elif kind == "retarget_pitch":
            midi = operation.get("new_midi")
            if midi is None or not 0 <= int(midi) <= 127:
                raise ValueError("retarget_pitch requires valid new_midi")
        elif kind == "shift_boundary":
            start = operation.get("new_start")
            end = operation.get("new_end")
            if start is None and end is None:
                raise ValueError("shift_boundary requires new_start or new_end")
            if start is not None and not 0 <= float(start) <= duration:
                raise ValueError("new_start is outside the track")
            if end is not None and not 0 <= float(end) <= duration:
                raise ValueError("new_end is outside the track")


def apply_operations(baseline: list[dict[str, Any]], operations: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    """Apply operations atomically; raise instead of returning a partial score."""
    validate_operations(operations, duration)
    events = deepcopy(baseline)
    for index, event in enumerate(events):
        event.setdefault("event_id", f"baseline-{index}")
    by_id = {event["event_id"]: event for event in events}
    for operation in operations:
        kind = operation.get("op")
        if kind == "keep":
            continue
        refs = operation["evidence_refs"]
        reason = operation["reason"]
        if kind == "insert_event":
            event = deepcopy(operation["event"])
            event.setdefault("event_id", f"agent-insert-{len(events)}")
            event["evidence_refs"] = refs
            event["agent_reason"] = reason
            events.append(event)
            by_id[event["event_id"]] = event
        elif kind == "split_event":
            target_id = operation.get("event_id")
            split = float(operation["split_time"])
            target = by_id.get(target_id)
            if target is None:
                derived = [
                    event for event in events
                    if event["event_id"].startswith(f"{target_id}-")
                    and float(event["start"]) < split < float(event["end"])
                ]
                if len(derived) == 1:
                    target = derived[0]
                else:
                    raise ValueError(f"split target not found: {target_id}")
            if not target["start"] < split < target["end"]:
                raise ValueError("split time must lie inside target event")
            first = deepcopy(target)
            second = deepcopy(target)
            target_base_id = target["event_id"]
            first["event_id"] = f"{target_base_id}-a"
            second["event_id"] = f"{target_base_id}-b"
            first["end"] = split
            second["start"] = split
            first["evidence_refs"] = refs
            second["evidence_refs"] = refs
            first["agent_reason"] = reason
            second["agent_reason"] = reason
            # A split is an articulation decision, not merely a boundary edit:
            # the second segment must receive a fresh piano attack. Without this
            # marker, the Song-like renderer's held-note collapse can merge the
            # two same-pitch segments back into one long, empty-sounding hold.
            second["retrigger"] = True
            second["articulation"] = "retrigger"
            events[events.index(target)] = first
            events.insert(events.index(first) + 1, second)
            by_id.pop(target_base_id, None)
            by_id[first["event_id"]] = first
            by_id[second["event_id"]] = second
        elif kind == "merge_events":
            ids = operation["event_ids"]
            targets = [by_id.get(event_id) for event_id in ids]
            if any(target is None for target in targets):
                raise ValueError("merge target not found")
            merged = deepcopy(targets[0])
            merged["event_id"] = f"agent-merge-{ids[0]}"
            merged["start"] = min(float(target["start"]) for target in targets)
            merged["end"] = max(float(target["end"]) for target in targets)
            merged["evidence_refs"] = refs
            merged["agent_reason"] = reason
            events = [event for event in events if event["event_id"] not in set(ids)]
            events.append(merged)
            by_id = {event["event_id"]: event for event in events}
        elif kind == "retarget_pitch":
            target = by_id.get(operation.get("event_id"))
            if target is None:
                raise ValueError("retarget target not found")
            target["midi"] = int(operation["new_midi"])
            target["evidence_refs"] = refs
            target["agent_reason"] = reason
        elif kind == "shift_boundary":
            target = by_id.get(operation.get("event_id"))
            if target is None:
                raise ValueError("boundary target not found")
            if operation.get("new_start") is not None:
                target["start"] = float(operation["new_start"])
            if operation.get("new_end") is not None:
                target["end"] = float(operation["new_end"])
            if target["end"] <= target["start"]:
                raise ValueError("shift creates non-positive event duration")
            target["evidence_refs"] = refs
            target["agent_reason"] = reason
    events.sort(key=lambda event: (float(event["start"]), float(event["end"])))
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    for event in events:
        if not event.get('note') and event.get('midi') is not None:
            midi = int(event['midi'])
            event['note'] = f"{pitch_classes[midi % 12]}{midi // 12 - 1}"
    return events
