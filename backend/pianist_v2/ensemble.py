"""Ensemble decoder: ROSVOT boundaries + CREPE/RMVPE pitch evidence.

Problem it fixes: librosa onsets cut where nothing is sung (E1: 74/76 onsets
had no local pitch change) while real same-pitch re-attacks are missed.

Strategy (deterministic, no LLM):
- ROSVOT MIDI supplies note intervals/rests when available (trained boundary model).
- CREPE frame evidence corrects ROSVOT pitch when the sung median disagrees
  decisively (>0.75 semitone) with high voicing confidence.
- If ROSVOT is missing, return CREPE melody unchanged (v1 behavior).
- Provenance is preserved in `source_model` so eval can ablate.
"""
from __future__ import annotations

import statistics


def _crepe_median_in_window(crepe: list[dict], start: float, end: float):
    mids = [e["midi"] for e in crepe if e["end"] > start and e["start"] < end and e.get("midi") is not None]
    confs = [float(e.get("pitch_confidence", 0) or 0) for e in crepe if e["end"] > start and e["start"] < end]
    if not mids:
        return None, 0.0
    return int(round(statistics.median(mids))), (sum(confs) / len(confs)) if confs else 0.0


def merge_rosvot_crepe(crepe: list[dict], rosvot: list[dict] | None) -> list[dict]:
    """Merge ROSVOT intervals with CREPE pitch evidence."""
    if not rosvot:
        out = [dict(e) for e in crepe]
        for e in out:
            e.setdefault("source_model", e.get("source_model", "crepe"))
        return sorted(out, key=lambda x: (x["start"], x["midi"]))

    merged: list[dict] = []
    for r in rosvot:
        r = dict(r)
        try:
            s, e = float(r["start"]), float(r["end"])
        except (TypeError, ValueError):
            continue
        if not s < e or r.get("midi") is None:
            continue
        med, conf = _crepe_median_in_window(crepe, s, e)
        # Correct ROSVOT pitch only on decisive, high-confidence disagreement.
        if med is not None and abs(med - int(r["midi"])) >= 1 and conf >= 0.7:
            # Require sustained disagreement, not a single CREPE frame.
            window = [c for c in crepe if c["end"] > s and c["start"] < e]
            if len(window) >= 3:
                r["midi"] = med
                r["pitch_corrected_by"] = "crepe_median"
                r["pitch_correction_confidence"] = round(conf, 3)
        # Carry useful CREPE dynamics when ROSVOT velocity is flat/default.
        if r.get("velocity") in (None, 0.7):
            vels = [c.get("velocity") for c in crepe if c["end"] > s and c["start"] < e and c.get("velocity") is not None]
            if vels:
                r["velocity"] = round(float(sum(vels) / len(vels)), 3)
        r["source_model"] = "rosvot+crepe_ensemble"
        # Normalize note naming fields if caller only gave midi.
        if "note" not in r or "octave" not in r:
            names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            m = int(r["midi"])
            r.setdefault("note", f"{names[m % 12]}{m // 12 - 1}")
            r.setdefault("octave", m // 12 - 1)
            r.setdefault("pitch_class", names[m % 12])
        merged.append(r)

    # Keep CREPE-only ornaments/evidence that fall entirely inside long ROSVOT gaps?
    # No — gaps are rests by construction. Do not re-inject notes into rests.
    merged.sort(key=lambda x: (x["start"], x["midi"]))

    # Clamp overlaps so the timeline is monotonic (same contract as agent_runner).
    out: list[dict] = []
    for seg in merged:
        if out and seg["start"] < out[-1]["end"]:
            seg["start"] = round(out[-1]["end"], 3)
            if seg["start"] >= seg["end"]:
                continue
        out.append(seg)
    return out
