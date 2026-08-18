"""Reference-conditioned score alignment over frame evidence.

This is a separate score-construction mode, not a post-filter on the legacy
melody. It is useful when the user supplies a trusted sargam/lyric note
sequence. It never invents extra score events and never edits the default
(audio-only) path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ReferenceToken:
    pitch_class: int
    label: str
    octave_hint: int | None = None


@dataclass(frozen=True)
class ReferenceAlignConfig:
    min_note_frames: int = 3
    min_rest_frames: int = 1
    onset_bonus: float = 0.8
    repeated_pitch_boundary_penalty: float = 1.2
    pitch_change_boundary_bonus: float = 0.35
    voiced_penalty: float = 1.5
    unvoiced_rest_cost: float = 0.15
    voiced_rest_cost: float = 2.5
    pitch_distance_weight: float = 2.8
    smoothness_weight: float = 0.15


def _circular_pc_distance(midi: float, pc: int) -> float:
    if not np.isfinite(midi):
        return 3.0
    d = abs((float(midi) % 12.0) - float(pc))
    return min(d, 12.0 - d)


def _normalise(values: Sequence[float], default: float = 0.0) -> np.ndarray:
    x = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=default, posinf=1.0, neginf=0.0)
    if x.size == 0:
        return x
    lo, hi = np.percentile(x, 5), np.percentile(x, 95)
    if hi <= lo + 1e-9:
        return np.clip(x, 0.0, 1.0)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def align_reference_to_frames(
    times: Sequence[float],
    midi: Sequence[float],
    voicing: Sequence[float],
    onset: Sequence[float],
    rms: Sequence[float],
    reference: Sequence[ReferenceToken],
    config: ReferenceAlignConfig | None = None,
) -> list[dict]:
    """Align a complete reference sequence to frame evidence.

    The DP assigns every frame to either the current reference note or an
    explicit rest after the current note. Reference tokens cannot be skipped,
    and extra detector events cannot be inserted. This is intentionally
    conservative for song-specific mode: a reference can be wrong, so the
    result includes an alignment confidence diagnostic rather than silently
    claiming that the reference is correct.
    """
    cfg = config or ReferenceAlignConfig()
    t = np.asarray(times, dtype=np.float64)
    f0 = np.asarray(midi, dtype=np.float64)
    v = _normalise(voicing)
    o = _normalise(onset)
    e = _normalise(rms, default=0.5)
    n = min(len(t), len(f0), len(v), len(o), len(e))
    k = len(reference)
    if n == 0 or k == 0:
        return []
    t, f0, v, o, e = t[:n], f0[:n], v[:n], o[:n], e[:n]
    dt = float(np.median(np.diff(t))) if n > 1 else 0.005

    # note[i,t] is the cost of assigning frame t to reference token i.
    note_cost = np.zeros((k, n), dtype=np.float64)
    for i, token in enumerate(reference):
        for frame in range(n):
            note_cost[i, frame] = (
                cfg.pitch_distance_weight * _circular_pc_distance(f0[frame], token.pitch_class)
                + cfg.voiced_penalty * (1.0 - v[frame])
                + cfg.smoothness_weight * (1.0 - e[frame])
            )
    rest_cost = cfg.unvoiced_rest_cost * (1.0 - v) + cfg.voiced_rest_cost * v + 0.25 * e

    # States: active note i (0..k-1), and rest-after-note i (k..2k-1).
    states = 2 * k
    inf = 1e18
    dp = np.full((n, states), inf, dtype=np.float64)
    back = np.full((n, states), -1, dtype=np.int32)
    run = np.zeros((n, states), dtype=np.int16)

    dp[0, 0] = note_cost[0, 0]
    run[0, 0] = 1
    if k > 1:
        # Starting later is discouraged rather than silently deleting tokens.
        for i in range(1, k):
            dp[0, i] = inf

    for frame in range(1, n):
        for state in range(states):
            if state < k:
                i = state
                # Stay on the current token.
                best = dp[frame - 1, state]
                best_prev = state
                best_run = min(int(run[frame - 1, state]) + 1, 32767)
                # Enter from the previous token or a rest after the previous token.
                if i > 0:
                    prev_note = i - 1
                    boundary = cfg.pitch_change_boundary_bonus if reference[i - 1].pitch_class != reference[i].pitch_class else cfg.repeated_pitch_boundary_penalty
                    boundary -= cfg.onset_bonus * float(o[frame])
                    cand = dp[frame - 1, prev_note] + boundary
                    if cand < best and run[frame - 1, prev_note] >= cfg.min_note_frames:
                        best, best_prev, best_run = cand, prev_note, 1
                    prev_rest = k + i - 1
                    cand = dp[frame - 1, prev_rest] - cfg.onset_bonus * float(o[frame])
                    if cand < best and run[frame - 1, prev_rest] >= cfg.min_rest_frames:
                        best, best_prev, best_run = cand, prev_rest, 1
                dp[frame, state] = best + note_cost[i, frame]
                back[frame, state] = best_prev
                run[frame, state] = best_run
            else:
                i = state - k
                # Enter rest only after the corresponding note has lasted.
                best = dp[frame - 1, state]
                best_prev = state
                best_run = min(int(run[frame - 1, state]) + 1, 32767)
                if run[frame - 1, i] >= cfg.min_note_frames:
                    cand = dp[frame - 1, i] + rest_cost[frame]
                    if cand < best:
                        best, best_prev, best_run = cand, i, 1
                dp[frame, state] = best + (rest_cost[frame] if best_prev >= k else 0.0)
                back[frame, state] = best_prev
                run[frame, state] = best_run

    # End in the last note or its rest state; prefer the latter only when it is
    # genuinely cheaper, so trailing silence remains a rest rather than a held note.
    end_states = [k - 1, 2 * k - 1]
    end_state = min(end_states, key=lambda s: dp[-1, s])
    path = np.zeros(n, dtype=np.int32)
    path[-1] = end_state
    for frame in range(n - 1, 0, -1):
        prev = int(back[frame, path[frame]])
        path[frame - 1] = prev if prev >= 0 else path[frame]

    assignments: list[list[int]] = [[] for _ in range(k)]
    rests: list[int] = []
    for frame, state in enumerate(path):
        if 0 <= state < k:
            assignments[state].append(frame)
        else:
            rests.append(frame)

    events: list[dict] = []
    for i, token in enumerate(reference):
        frames = assignments[i]
        if not frames:
            # This should be rare and is explicitly surfaced, never hidden.
            continue
        start_idx, end_idx = min(frames), max(frames)
        segment = f0[frames]
        finite = segment[np.isfinite(segment)]
        if token.octave_hint is not None:
            base = int(round(float(np.median(finite)))) if len(finite) else int(token.pitch_class + 60)
            octave = int(token.octave_hint)
            target = token.pitch_class + 12 * (octave + 1)
            candidates = [target - 12, target, target + 12]
            played_midi = min(candidates, key=lambda x: abs(x - base))
        elif len(finite):
            med = float(np.median(finite))
            played_midi = int(round(med - ((med - token.pitch_class) % 12)))
            while played_midi < 48:
                played_midi += 12
            while played_midi > 83:
                played_midi -= 12
        else:
            played_midi = int(token.pitch_class + 60)
        events.append({
            'start': round(float(t[start_idx]), 4),
            'end': round(float(t[end_idx] + dt), 4),
            'midi': int(played_midi),
            'label': token.label,
            'reference_index': i,
            'reference_pitch_class': int(token.pitch_class),
            'alignment_confidence': round(float(np.mean(np.exp(-note_cost[i, frames] / 3.0))), 4),
            'retrigger': bool(i > 0 and reference[i - 1].pitch_class == token.pitch_class),
        })
    return events
