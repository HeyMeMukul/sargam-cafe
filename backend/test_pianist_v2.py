"""Tests for pianist_v2 (deterministic, no GPU, no network)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from pianist_v2.ensemble import merge_rosvot_crepe
from pianist_v2.arranger import arrange_for_piano
from pianist_v2.performance import apply_performance
from pianist_v2.contracts import validate_score


def test_ensemble_prefers_rosvot_boundaries():
    crepe = [
        {"start": 0.0, "end": 0.5, "midi": 60, "pitch_confidence": 0.9, "velocity": 0.8},
        {"start": 0.5, "end": 1.0, "midi": 62, "pitch_confidence": 0.9, "velocity": 0.8},
    ]
    rosvot = [
        {"start": 0.0, "end": 0.48, "midi": 60, "velocity": 0.7},
        {"start": 0.52, "end": 1.0, "midi": 62, "velocity": 0.7},
    ]
    out = merge_rosvot_crepe(crepe, rosvot)
    assert len(out) == 2, out
    assert all(e["source_model"] == "rosvot+crepe_ensemble" for e in out)
    assert out[0]["end"] <= out[1]["start"]


def test_ensemble_empty_rosvot_returns_crepe():
    crepe = [{"start": 0.0, "end": 0.5, "midi": 60}]
    out = merge_rosvot_crepe(crepe, [])
    assert len(out) == 1 and out[0]["midi"] == 60


def test_arranger_adds_bass_and_keeps_melody():
    mel = [{"start": 0.0, "end": 0.5, "midi": 60, "note": "C4", "velocity": 0.7}]
    chords = [{"start": 0.0, "end": 2.0, "midis": [48, 52, 55]}]
    out = arrange_for_piano(mel, chords)
    roles = {e["role"] for e in out}
    assert "melody" in roles and "bass" in roles
    assert not validate_score([e for e in out if e.get("midi") is not None])


def test_performance_is_deterministic():
    ev = [{"start": 1.0, "end": 1.5, "midi": 60, "note": "C4", "velocity": 0.7,
           "phrase_pos": 0.5, "phrase_peak": True, "phrase_cadence": False, "phrase_id": 0}]
    a = apply_performance(ev, tempo=120, seed=7)
    b = apply_performance(ev, tempo=120, seed=7)
    assert a == b
    assert a[0]["velocity"] != 0.7  # arch applied


if __name__ == "__main__":
    test_ensemble_prefers_rosvot_boundaries()
    test_ensemble_empty_rosvot_returns_crepe()
    test_arranger_adds_bass_and_keeps_melody()
    test_performance_is_deterministic()
    print("pianist_v2 tests passed")
