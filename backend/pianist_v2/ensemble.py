"""Ensemble decoder: ROSVOT boundaries + CREPE/RMVPE pitch evidence.

Problem it fixes: librosa onsets cut where nothing is sung (E1: 74/76 onsets
had no local pitch change) while real same-pitch re-attacks are missed.

Strategy (deterministic, no LLM):
- ROSVOT MIDI supplies note intervals/rests when available (trained boundary model).
- CREPE frame evidence corrects ROSVOT pitch when the sung median disagrees
  decisively (>0.75 semitone) with high voicing confidence.
- Voicing-aware merge: a ROSVOT interval with zero overlapping CREPE support
  is dropped when the global CREPE voiced fraction is high (>= 0.6). This
  removes hallucinated notes sung over silence without hurting low-coverage
  CREPE outputs (voiced fraction low -> keep ROSVOT).
- If ROSVOT is missing, return CREPE melody unchanged (v1 behavior).
- Gaps between ROSVOT intervals are rests by construction: never re-inject
  CREPE ornaments into them.
- Provenance is preserved in `source_model` (+ `ensemble_method` and
  per-event CREPE support fields) so eval can ablate.
"""
from __future__ import annotations

import statistics

# A CREPE frame counts as voiced when its voicing (or pitch) confidence
# clears this threshold.
VOICED_CONF_THRESHOLD = 0.5
# Drop unsupported ROSVOT intervals only when CREPE is confidently voiced
# this often globally — otherwise CREPE may simply have missed the note.
VOICED_FRACTION_DROP_THRESHOLD = 0.6
# Pitch-correction gate (unchanged from v1): decisive + high-conf + sustained.
PITCH_CORRECTION_MIN_SEMITONES = 1
PITCH_CORRECTION_MIN_CONF = 0.7
PITCH_CORRECTION_MIN_FRAMES = 3


def _frame_conf(e: dict) -> float:
    """Voicing confidence for a CREPE frame (falls back to pitch_confidence)."""
    v = e.get("voicing_confidence", None)
    if v is None:
        v = e.get("pitch_confidence", 0)
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


def _crepe_voiced_fraction(crepe: list[dict]) -> float:
    """Fraction of CREPE frames with midi that look voiced."""
    if not crepe:
        return 0.0
    voiced = sum(1 for e in crepe if e.get("midi") is not None and _frame_conf(e) >= VOICED_CONF_THRESHOLD)
    return voiced / len(crepe)


def _overlapping(crepe: list[dict], start: float, end: float) -> list[dict]:
    return [c for c in crepe if c.get("end", 0) > start and c.get("start", 0) < end]


def _crepe_median_in_window(crepe: list[dict], start: float, end: float):
    mids = [e["midi"] for e in crepe if e["end"] > start and e["start"] < end and e.get("midi") is not None]
    confs = [float(e.get("pitch_confidence", 0) or 0) for e in crepe if e["end"] > start and e["start"] < end]
    if not mids:
        return None, 0.0
    return int(round(statistics.median(mids))), (sum(confs) / len(confs)) if confs else 0.0


def _note_fields(midi: int) -> dict:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    m = int(midi)
    return {
        "note": f"{names[m % 12]}{m // 12 - 1}",
        "octave": m // 12 - 1,
        "pitch_class": names[m % 12],
    }


def merge_rosvot_crepe(crepe: list[dict], rosvot: list[dict] | None) -> list[dict]:
    """Merge ROSVOT intervals with CREPE pitch evidence."""
    crepe = list(crepe or [])
    voiced_frac = _crepe_voiced_fraction(crepe)

    if not rosvot:
        out = [dict(e) for e in crepe]
        for e in out:
            e.setdefault("source_model", e.get("source_model", "crepe"))
            # Provenance for every event, including passthrough.
            e.setdefault("ensemble_method", "crepe_passthrough")
            e.setdefault("pitch_corrected", False)
            e.setdefault("overlap_clamped", False)
            if "crepe_support_frames" not in e:
                e["crepe_support_frames"] = 1 if e.get("midi") is not None else 0
            if "crepe_median_midi" not in e:
                e["crepe_median_midi"] = e.get("midi")
            if "crepe_mean_confidence" not in e:
                try:
                    e["crepe_mean_confidence"] = round(float(e.get("pitch_confidence", 0) or 0), 3)
                except (TypeError, ValueError):
                    e["crepe_mean_confidence"] = 0.0
            e["crepe_voiced_fraction"] = round(voiced_frac, 3)
        return sorted(out, key=lambda x: (x["start"], x["midi"] if x.get("midi") is not None else -1))

    merged: list[dict] = []
    for r in rosvot:
        r = dict(r)
        try:
            s, e = float(r["start"]), float(r["end"])
        except (TypeError, ValueError):
            continue
        if not s < e or r.get("midi") is None:
            continue
        window = _overlapping(crepe, s, e)
        support = sum(1 for c in window if c.get("midi") is not None)
        med, conf = _crepe_median_in_window(crepe, s, e)

        # (1) Voicing-aware merge: drop ROSVOT intervals with no CREPE
        # support when CREPE is confidently voiced elsewhere. When CREPE
        # coverage is sparse (low voiced fraction) keep ROSVOT — CREPE may
        # simply have missed the note.
        if support == 0 and voiced_frac >= VOICED_FRACTION_DROP_THRESHOLD:
            continue

        # Correct ROSVOT pitch only on decisive, high-confidence disagreement.
        corrected = False
        if med is not None and abs(med - int(r["midi"])) >= PITCH_CORRECTION_MIN_SEMITONES and conf >= PITCH_CORRECTION_MIN_CONF:
            # Require sustained disagreement, not a single CREPE frame.
            if len(window) >= PITCH_CORRECTION_MIN_FRAMES:
                r["midi"] = med
                r["pitch_corrected_by"] = "crepe_median"
                r["pitch_correction_confidence"] = round(conf, 3)
                corrected = True
        # Carry useful CREPE dynamics when ROSVOT velocity is flat/default.
        if r.get("velocity") in (None, 0.7):
            vels = [c.get("velocity") for c in crepe if c["end"] > s and c["start"] < e and c.get("velocity") is not None]
            if vels:
                r["velocity"] = round(float(sum(vels) / len(vels)), 3)
        r["source_model"] = "rosvot+crepe_ensemble"
        # (3) Provenance fields for every event.
        r["ensemble_method"] = "rosvot_boundaries+crepe_pitch"
        r["crepe_support_frames"] = support
        r["crepe_median_midi"] = med
        r["crepe_mean_confidence"] = round(conf, 3)
        r["crepe_voiced_fraction"] = round(voiced_frac, 3)
        r["pitch_corrected"] = corrected
        r["overlap_clamped"] = False
        # Normalize note naming fields if caller only gave midi.
        if "note" not in r or "octave" not in r:
            r.setdefault("note", _note_fields(int(r["midi"]))["note"])
            r.setdefault("octave", _note_fields(int(r["midi"]))["octave"])
            r.setdefault("pitch_class", _note_fields(int(r["midi"]))["pitch_class"])
        merged.append(r)

    # (2) Gap/rest preservation: gaps between ROSVOT intervals are rests by
    # construction. Do not re-inject CREPE-only ornaments into them.
    merged.sort(key=lambda x: (x["start"], x["midi"]))

    # Clamp overlaps so the timeline is monotonic (same contract as agent_runner).
    out: list[dict] = []
    for seg in merged:
        if out and seg["start"] < out[-1]["end"]:
            original = float(seg["start"])
            seg["start"] = round(float(out[-1]["end"]), 3)
            seg["overlap_clamped"] = True
            seg["original_start"] = round(original, 3)
            if seg["start"] >= seg["end"]:
                continue
        out.append(seg)
    return out
