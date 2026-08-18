"""Conservative frame-lattice decoder for singing note candidates.

This module deliberately does not replace the legacy extractor by itself. It
consumes frame evidence and returns note events only when a global path makes
that choice cheaper than a rest, a sustain, or an onset-gated pitch change.
The caller can keep the legacy path as the default and enable this decoder
behind a feature flag during evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class DecoderConfig:
    min_note_seconds: float = 0.025
    min_stable_frames: int = 2
    pitch_sigma_semitones: float = 0.65
    voiced_weight: float = 2.0
    pitch_weight: float = 2.6
    onset_weight: float = 1.5
    rest_weight: float = 1.8
    pitch_change_penalty: float = 4.0
    unsupported_change_penalty: float = 7.5
    rest_transition_penalty: float = 1.0
    attack_threshold: float = 0.22
    voicing_gap_threshold: float = 0.35
    min_unonset_pitch_change: float = 1.5
    max_pitch_jump: int = 12


def _safe01(values: Sequence[float], default: float = 0.0) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return x
    finite = np.isfinite(x)
    if not finite.any():
        return np.full_like(x, default)
    out = np.nan_to_num(x, nan=default, posinf=1.0, neginf=0.0)
    lo = float(np.percentile(out[finite], 5))
    hi = float(np.percentile(out[finite], 95))
    if hi <= lo + 1e-9:
        return np.clip(out, 0.0, 1.0)
    return np.clip((out - lo) / (hi - lo), 0.0, 1.0)


def _pitch_emission(midi: float, candidate: int, sigma: float) -> float:
    if not np.isfinite(midi):
        return -8.0
    delta = float(midi) - float(candidate)
    return -0.5 * (delta / max(sigma, 1e-6)) ** 2


def _candidate_set(midi: Sequence[float], alternatives: Iterable[Sequence[tuple[int, float]]] | None,
                   min_midi: int, max_midi: int) -> list[int]:
    candidates: set[int] = set()
    for value in np.asarray(midi, dtype=np.float64):
        if np.isfinite(value):
            base = int(round(float(value)))
            for p in (base - 1, base, base + 1):
                if min_midi <= p <= max_midi:
                    candidates.add(p)
    if alternatives is not None:
        for frame in alternatives:
            for pitch, _prob in frame:
                p = int(round(pitch))
                if min_midi <= p <= max_midi:
                    candidates.add(p)
    return sorted(candidates)


def decode_frame_lattice(
    times: Sequence[float],
    midi: Sequence[float],
    voicing: Sequence[float | bool],
    onset: Sequence[float],
    rms: Sequence[float] | None = None,
    alternatives: Iterable[Sequence[tuple[int, float]]] | None = None,
    config: DecoderConfig | None = None,
    min_midi: int = 48,
    max_midi: int = 83,
) -> list[dict]:
    """Decode frame evidence into ordered note events.

    `voicing` and `onset` may be probabilities or arbitrary nonnegative
    strengths; both are robustly normalized to [0, 1]. `alternatives` is an
    optional per-frame sequence of `(MIDI, probability)` candidates. The
    output is deliberately free of ornaments: articulation must be established
    by a later, separate performance layer.
    """
    cfg = config or DecoderConfig()
    t = np.asarray(times, dtype=np.float64)
    f0 = np.asarray(midi, dtype=np.float64)
    v = _safe01(voicing)
    o = _safe01(onset)
    energy = _safe01(rms if rms is not None else np.ones_like(t), default=0.5)
    n = min(len(t), len(f0), len(v), len(o), len(energy))
    if n < 2:
        return []
    t, f0, v, o, energy = t[:n], f0[:n], v[:n], o[:n], energy[:n]
    dt = float(np.median(np.diff(t))) if n > 1 else 0.005
    candidates = _candidate_set(f0, alternatives, min_midi, max_midi)
    if not candidates:
        return []

    # State 0 is REST; states 1..P are stable/attack-capable pitch states.
    pitches = np.asarray(candidates, dtype=np.int32)
    states = len(pitches) + 1
    neg_inf = -1e18
    dp = np.full((n, states), neg_inf, dtype=np.float64)
    back = np.full((n, states), -1, dtype=np.int32)

    def emission(frame: int, state: int) -> float:
        if state == 0:
            return cfg.rest_weight * (1.0 - v[frame]) + 0.45 * (1.0 - o[frame]) + 0.25 * (1.0 - energy[frame])
        p = int(pitches[state - 1])
        return (cfg.pitch_weight * _pitch_emission(f0[frame], p, cfg.pitch_sigma_semitones)
                + cfg.voiced_weight * v[frame]
                + cfg.onset_weight * o[frame])

    dp[0, 0] = emission(0, 0)
    for s in range(1, states):
        # Starting on an attack without onset evidence is possible but costly;
        # voiced evidence still prevents the decoder from erasing a real note.
        dp[0, s] = emission(0, s) - cfg.pitch_change_penalty * (1.0 - o[0])

    for frame in range(1, n):
        for state in range(states):
            best_score = neg_inf
            best_prev = -1
            for prev in range(states):
                if prev == 0 and state == 0:
                    penalty = 0.0
                elif prev == 0 and state != 0:
                    penalty = cfg.pitch_change_penalty * (1.0 - o[frame])
                elif prev != 0 and state == 0:
                    penalty = cfg.rest_transition_penalty * v[frame]
                elif prev == state:
                    penalty = 0.0
                else:
                    jump = abs(int(pitches[prev - 1]) - int(pitches[state - 1]))
                    if jump > cfg.max_pitch_jump:
                        continue
                    onset_supported = o[frame] >= cfg.attack_threshold
                    voice_gap = v[frame] <= cfg.voicing_gap_threshold
                    # A pitch change without an onset or release is normally
                    # tracker drift/vibrato. Allow it only when the displacement
                    # is substantial; small changes must wait for attack evidence.
                    if not onset_supported and not voice_gap and jump < cfg.min_unonset_pitch_change:
                        continue
                    supported = onset_supported or voice_gap
                    penalty = cfg.pitch_change_penalty if supported else cfg.unsupported_change_penalty
                    penalty += 0.12 * jump
                score = dp[frame - 1, prev] - penalty
                if score > best_score:
                    best_score = score
                    best_prev = prev
            dp[frame, state] = best_score + emission(frame, state)
            back[frame, state] = best_prev

    path = np.zeros(n, dtype=np.int32)
    path[-1] = int(np.argmax(dp[-1]))
    for frame in range(n - 1, 0, -1):
        path[frame - 1] = back[frame, path[frame]] if back[frame, path[frame]] >= 0 else 0

    # Convert path runs to notes. Rests are explicit gaps in the returned
    # timeline; they are never filled by duration accumulation.
    events: list[dict] = []
    start = None
    active_pitch = None
    for i, state in enumerate(path):
        pitch = None if state == 0 else int(pitches[state - 1])
        changed = pitch != active_pitch
        if changed:
            if active_pitch is not None and start is not None:
                end = float(t[i])
                if end - start >= cfg.min_note_seconds:
                    events.append({
                        'start': round(float(start), 4),
                        'end': round(end, 4),
                        'midi': int(active_pitch),
                        'decoder': 'frame_viterbi',
                    })
            active_pitch = pitch
            start = float(t[i]) if pitch is not None else None
    if active_pitch is not None and start is not None:
        end = float(t[-1] + dt)
        if end - start >= cfg.min_note_seconds:
            events.append({
                'start': round(float(start), 4),
                'end': round(end, 4),
                'midi': int(active_pitch),
                'decoder': 'frame_viterbi',
            })
    return events
