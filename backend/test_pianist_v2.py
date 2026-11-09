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


def test_arranger_never_cuts_or_retransposes_melody():
    mel = [
        {"start": 0.0, "end": 0.5, "midi": 55, "note": "G3", "velocity": 0.7},  # < C4 -> octave lift only
        {"start": 0.6, "end": 1.0, "midi": 67, "note": "G4", "velocity": 0.75},
        {"start": 2.0, "end": 2.4, "midi": 72, "note": "C5", "velocity": 0.8},
    ]
    chords = [{"start": 0.0, "end": 3.0, "midis": [48, 52, 55]}]
    out = arrange_for_piano(mel, chords)
    melodies = [e for e in out if e["role"] == "melody"]
    assert len(melodies) == len(mel), out
    for orig, got in zip(mel, sorted(melodies, key=lambda e: e["start"])):
        # Timing/duration preserved exactly.
        assert got["start"] == orig["start"] and got["end"] == orig["end"], (orig, got)
        # Pitch: identical, or +12k octave lift if below C4 (pitch class kept).
        delta = got["midi"] - orig["midi"]
        assert delta % 12 == 0, (orig, got)
        if orig["midi"] >= 60:
            assert delta == 0, (orig, got)
        else:
            assert got["midi"] >= 60 and delta > 0, (orig, got)


def test_arranger_bass_harmony_fill_invariants():
    # Bass collision case: root-12 == 60 would hit the melody C4 exactly.
    mel = [{"start": 0.0, "end": 1.0, "midi": 60, "note": "C4", "velocity": 0.7}]
    chords = [{"start": 0.0, "end": 2.0, "midis": [72, 76, 79]}]  # root 72 -> bass 60 collides
    out = arrange_for_piano(mel, chords)
    basses = [e for e in out if e["role"] == "bass"]
    harmonies = [e for e in out if e["role"] == "harmony"]
    fills = [e for e in out if e["role"] == "fill"]
    assert basses, out
    for b in basses:
        for m in mel:
            overlaps = m["start"] < b["end"] and b["start"] < m["end"]
            if overlaps:
                assert abs(b["midi"] - (m["midi"] if m["midi"] >= 60 else m["midi"] + 12)) > 1, (b, m)
    # Harmony strictly quieter than every melody note.
    min_mel_vel = min(e["velocity"] for e in out if e["role"] == "melody")
    assert harmonies, out
    assert all(h["velocity"] < min_mel_vel for h in harmonies), out
    # At most one fill even with several gaps.
    mel2 = [
        {"start": 0.0, "end": 0.5, "midi": 60, "note": "C4", "velocity": 0.7},
        {"start": 1.5, "end": 2.0, "midi": 62, "note": "D4", "velocity": 0.7},
        {"start": 3.0, "end": 3.5, "midi": 64, "note": "E4", "velocity": 0.7},
        {"start": 4.5, "end": 5.0, "midi": 65, "note": "F4", "velocity": 0.7},
    ]
    out2 = arrange_for_piano(mel2, [])
    assert sum(1 for e in out2 if e["role"] == "fill") <= 1, out2


def test_performance_timing_velocity_bounds_and_determinism():
    ev = [
        {"start": 1.0, "end": 1.5, "midi": 60, "note": "C4", "velocity": 0.7,
         "role": "melody", "phrase_pos": 0.5, "phrase_peak": True, "phrase_cadence": False, "phrase_id": 0},
        {"start": 2.0, "end": 2.5, "midi": 62, "note": "D4", "velocity": 5.0,  # out-of-range input
         "role": "harmony", "phrase_pos": 0.5, "phrase_peak": False, "phrase_cadence": True, "phrase_id": 0},
        {"start": 0.005, "end": 0.5, "midi": 48, "note": "C3", "velocity": 0.01,  # near-zero start/vel
         "role": "bass", "phrase_pos": 0.9, "phrase_peak": False, "phrase_cadence": False, "phrase_id": 1},
    ]
    for tempo in (60, 120, 180):
        a = apply_performance(ev, tempo=tempo, seed=11)
        b = apply_performance(ev, tempo=tempo, seed=11)
        assert a == b  # deterministic by seed
        for orig in ev:
            match = next(g for g in a if g["midi"] == orig["midi"])
            assert abs(match["start"] - orig["start"]) <= 0.080 + 1e-9, (tempo, orig, match)
            assert 0.2 <= match["velocity"] <= 1.0, (tempo, match)
            assert match["start"] >= 0.0, match


def test_performance_pedal_only_at_phrase_boundaries():
    ev = [
        {"start": 0.0, "end": 0.5, "midi": 60, "note": "C4", "velocity": 0.7, "role": "melody",
         "phrase_pos": 0, "phrase_peak": False, "phrase_cadence": False, "phrase_id": 0},  # boundary: down
        {"start": 0.6, "end": 1.0, "midi": 62, "note": "D4", "velocity": 0.7, "role": "melody",
         "phrase_pos": 0.5, "phrase_peak": False, "phrase_cadence": False, "phrase_id": 0,
         "pedal_down": True, "pedal_up": True},  # mid-phrase strays must be stripped
        {"start": 1.1, "end": 1.6, "midi": 64, "note": "E4", "velocity": 0.7, "role": "melody",
         "phrase_pos": 0.9, "phrase_peak": False, "phrase_cadence": True, "phrase_id": 0},  # boundary: up
        {"start": 1.7, "end": 2.0, "midi": 65, "note": "F4", "velocity": 0.7, "role": "melody",
         "phrase_pos": 0.5, "phrase_peak": False, "phrase_cadence": False, "phrase_id": None},  # no phrase
    ]
    out = apply_performance(ev, tempo=120, seed=3)
    by_midi = {e["midi"]: e for e in out}
    assert by_midi[60].get("pedal_down") is True
    assert "pedal_up" not in by_midi[60]
    assert "pedal_down" not in by_midi[62] and "pedal_up" not in by_midi[62]
    assert by_midi[64].get("pedal_up") is True
    assert "pedal_down" not in by_midi[65] and "pedal_up" not in by_midi[65]


def test_ensemble_rest_preserved():
    """ROSVOT gaps are rests: merge must not fill them with CREPE ornaments."""
    crepe = [
        {"start": 0.0, "end": 0.5, "midi": 60, "pitch_confidence": 0.9,
         "voicing_confidence": 0.95, "velocity": 0.8},
        # Ornament-like CREPE evidence sitting inside the rest gap.
        {"start": 0.6, "end": 0.8, "midi": 64, "pitch_confidence": 0.9,
         "voicing_confidence": 0.95, "velocity": 0.8},
        {"start": 1.0, "end": 1.5, "midi": 62, "pitch_confidence": 0.9,
         "voicing_confidence": 0.95, "velocity": 0.8},
    ]
    rosvot = [
        {"start": 0.0, "end": 0.5, "midi": 60, "velocity": 0.7},
        {"start": 1.0, "end": 1.5, "midi": 62, "velocity": 0.7},
    ]
    out = merge_rosvot_crepe(crepe, rosvot)
    assert len(out) == 2, out
    # No output event may cover the rest interior (0.5, 1.0).
    for e in out:
        assert not (e["start"] < 0.75 < e["end"]), e
    assert out[0]["end"] <= out[1]["start"], out
    # Provenance present on every event.
    for e in out:
        assert e["source_model"] == "rosvot+crepe_ensemble", e
        assert "ensemble_method" in e and "crepe_support_frames" in e, e


def test_ensemble_pitch_correction_only_on_high_conf():
    """Decisive disagreement corrects pitch only when CREPE confidence is high."""

    def _crepe(conf):
        # 3 sustained frames at midi 64 -> median 64 vs ROSVOT 60.
        return [
            {"start": 0.0, "end": 0.2, "midi": 64, "pitch_confidence": conf,
             "voicing_confidence": 0.9, "velocity": 0.8},
            {"start": 0.2, "end": 0.4, "midi": 64, "pitch_confidence": conf,
             "voicing_confidence": 0.9, "velocity": 0.8},
            {"start": 0.4, "end": 0.6, "midi": 64, "pitch_confidence": conf,
             "voicing_confidence": 0.9, "velocity": 0.8},
        ]

    rosvot = [{"start": 0.0, "end": 0.6, "midi": 60, "velocity": 0.7}]
    hi = merge_rosvot_crepe(_crepe(0.9), rosvot)
    assert len(hi) == 1, hi
    assert hi[0]["midi"] == 64, hi
    assert hi[0]["pitch_corrected"] is True, hi
    assert hi[0]["pitch_corrected_by"] == "crepe_median", hi

    lo = merge_rosvot_crepe(_crepe(0.3), rosvot)
    assert len(lo) == 1, lo
    assert lo[0]["midi"] == 60, lo
    assert lo[0]["pitch_corrected"] is False, lo
    assert "pitch_corrected_by" not in lo, lo


def test_ensemble_overlap_clamp():
    """Overlapping ROSVOT intervals are clamped to a monotonic timeline."""
    crepe = [
        {"start": 0.0, "end": 0.5, "midi": 60, "pitch_confidence": 0.9,
         "voicing_confidence": 0.9, "velocity": 0.8},
        {"start": 0.5, "end": 1.0, "midi": 60, "pitch_confidence": 0.9,
         "voicing_confidence": 0.9, "velocity": 0.8},
        {"start": 1.0, "end": 1.5, "midi": 62, "pitch_confidence": 0.9,
         "voicing_confidence": 0.9, "velocity": 0.8},
    ]
    rosvot = [
        {"start": 0.0, "end": 1.0, "midi": 60, "velocity": 0.7},
        {"start": 0.5, "end": 1.5, "midi": 62, "velocity": 0.7},
    ]
    out = merge_rosvot_crepe(crepe, rosvot)
    assert len(out) == 2, out
    assert out[1]["start"] >= out[0]["end"], out
    assert out[1]["start"] == round(out[0]["end"], 3), out
    assert out[1]["overlap_clamped"] is True, out
    assert out[0]["overlap_clamped"] is False, out
    assert not validate_score([e for e in out if e.get("midi") is not None])


def test_ensemble_drops_unsupported_rosvot_when_voiced():
    """ROSVOT intervals with no CREPE support are dropped when CREPE is voiced."""
    crepe = [
        {"start": 0.0, "end": 0.5, "midi": 60, "pitch_confidence": 0.9,
         "voicing_confidence": 0.95, "velocity": 0.8},
        {"start": 0.5, "end": 1.0, "midi": 62, "pitch_confidence": 0.9,
         "voicing_confidence": 0.95, "velocity": 0.8},
        {"start": 1.0, "end": 1.5, "midi": 64, "pitch_confidence": 0.9,
         "voicing_confidence": 0.95, "velocity": 0.8},
    ]
    rosvot = [
        {"start": 0.0, "end": 0.5, "midi": 60, "velocity": 0.7},
        # Hallucinated: no CREPE frame overlaps [5.0, 5.5].
        {"start": 5.0, "end": 5.5, "midi": 67, "velocity": 0.7},
    ]
    out = merge_rosvot_crepe(crepe, rosvot)
    assert len(out) == 1, out
    assert out[0]["midi"] == 60, out
    assert out[0]["crepe_support_frames"] >= 1, out


if __name__ == "__main__":
    test_ensemble_prefers_rosvot_boundaries()
    test_ensemble_empty_rosvot_returns_crepe()
    test_ensemble_rest_preserved()
    test_ensemble_pitch_correction_only_on_high_conf()
    test_ensemble_overlap_clamp()
    test_ensemble_drops_unsupported_rosvot_when_voiced()
    test_arranger_adds_bass_and_keeps_melody()
    test_performance_is_deterministic()
    test_arranger_never_cuts_or_retransposes_melody()
    test_arranger_bass_harmony_fill_invariants()
    test_performance_timing_velocity_bounds_and_determinism()
    test_performance_pedal_only_at_phrase_boundaries()
    print("pianist_v2 tests passed")
