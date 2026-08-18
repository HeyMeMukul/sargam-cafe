import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reference_alignment import align_reference_phrases


def test_reference_alignment_removes_extra_events_and_sets_pitch():
    observed = [
        {"start": 2.0, "end": 2.4, "note": "C4", "midi": 60},
        {"start": 2.4, "end": 2.8, "note": "C4", "midi": 60},
        {"start": 2.8, "end": 3.2, "note": "D4", "midi": 62},
        {"start": 3.2, "end": 3.6, "note": "D4", "midi": 62},
        {"start": 10.0, "end": 10.5, "note": "A4", "midi": 69},
    ]
    ref = {"phrases": [{"start": 2.0, "end": 4.0, "sargam": "G P R R"}]}
    out = align_reference_phrases(observed, ref, root_pc=6)
    guided = [x for x in out if x.get("reference_guided")]
    assert len(guided) == 4
    assert [x["midi"] % 12 for x in guided] == [10, 1, 8, 8]
    assert sum(1 for x in out if x.get("start") == 10.0) == 1


def test_reference_alignment_fills_missing_note_and_keeps_repeated_pitch():
    observed = [
        {"start": 2.0, "end": 2.6, "note": "A#3", "midi": 58},
        {"start": 3.2, "end": 3.8, "note": "G#3", "midi": 56},
    ]
    ref = {"phrases": [{"start": 2.0, "end": 4.0, "sargam": "G P P D"}]}
    out = align_reference_phrases(observed, ref, root_pc=6)
    guided = [x for x in out if x.get("reference_guided")]
    assert len(guided) == 4
    assert [x["midi"] % 12 for x in guided] == [10, 1, 1, 3]
    assert all(x["end"] > x["start"] for x in guided)
