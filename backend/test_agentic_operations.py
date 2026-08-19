#!/usr/bin/env python3
from agentic.operations import apply_operations, validate_operations

baseline = [
    {"event_id": "e1", "start": 1.0, "end": 2.0, "midi": 60},
    {"event_id": "e2", "start": 2.0, "end": 3.0, "midi": 62},
]
ref = [{"tool": "inspect_boundary_window", "artifact_id": "b1", "feature": "onset", "start": 1.5, "end": 1.6, "confidence": 0.9}]
result = apply_operations(baseline, [{
    "op": "split_event", "event_id": "e1", "split_time": 1.5,
    "evidence_refs": ref, "reason": "strong local retrigger evidence"
}], 3.0)
assert len(result) == 3
assert result[0]['end'] == 1.5 and result[1]['start'] == 1.5

inserted = apply_operations(baseline, [{
    "op": "insert_event", "event": {"start": 1.4, "end": 1.6, "midi": 61},
    "evidence_refs": ref, "reason": "stable pitch and onset evidence"
}], 3.0)
assert any(event['midi'] == 61 for event in inserted)

try:
    validate_operations([{
        "op": "retarget_pitch", "event_id": "e1", "new_midi": 61,
        "evidence_refs": [{"tool": "get_note_candidates", "artifact_id": "b", "feature": "baseline"}],
        "reason": "scale says so"
    }], 3.0)
except ValueError:
    pass
else:
    raise AssertionError('baseline-only pitch mutation was accepted')
print('agentic operations passed')
