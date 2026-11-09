"""Phrase-arch performance: timing + dynamics + pedal from the score itself.

Replaces arbitrary slider-only Human Touch with a deterministic default:
- Timing: slight stretch at phrase ends (cadence breath), rush mid-phrase peak.
- Dynamics: arch peaking near phrase_peak, softer at phrase start/cadence.
- Articulation: legato by default; short `kan` stays detached.
- Pedal: down at phrase start, up at cadence (frontend already emulates this;
  here we emit explicit `pedal` markers the frontend can honor).

Deterministic given `seed`; frontend sliders remain as post-scale overrides.
"""
from __future__ import annotations

import random


def apply_performance(events: list[dict], tempo: float | None = None, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    beat = 60.0 / float(tempo or 120.0)
    out: list[dict] = []
    for e in events:
        e = dict(e)
        pos = float(e.get("phrase_pos", 0.5))
        is_peak = bool(e.get("phrase_peak"))
        is_cad = bool(e.get("phrase_cadence"))
        role = e.get("role", "melody")
        # Timing: ±6% beat-scaled arch; cadence breath +25ms, peak -10ms.
        dt = beat * 0.06 * (0.5 - abs(pos - 0.5)) * 2
        if is_cad:
            dt += 0.025
        if is_peak:
            dt -= 0.010
        # Humanize deterministically: ±8ms so repeated renders match.
        dt += (rng.random() - 0.5) * 0.016
        e["start"] = round(float(e["start"]) + dt, 3)
        # Dynamics: arch 0.85..1.1 around base velocity; melody only.
        if role == "melody":
            arch = 0.9 + 0.2 * (1.0 - abs(pos - 0.55) * 2)
            if is_peak:
                arch += 0.05
            if is_cad:
                arch -= 0.08
            e["velocity"] = round(max(0.2, min(1.0, float(e.get("velocity", 0.7)) * arch)), 3)
        # Pedal markers at phrase boundaries (frontend buildPedalPlan honors these).
        if e.get("phrase_id") is not None:
            if pos == 0:
                e["pedal_down"] = True
            if is_cad:
                e["pedal_up"] = True
        out.append(e)
    out.sort(key=lambda x: x["start"])
    return out
