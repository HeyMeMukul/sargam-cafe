#!/usr/bin/env python3
"""Strict ordered note evaluator for Sargam Cafe.

Unlike the historical best-window phrase scorer, this aligns the complete
reference and estimate in order with dynamic programming. It reports missing
and extra events explicitly and never substitutes a best local subsequence for
whole-phrase accuracy.

Usage:
    python evaluation/strict_sequence.py reference.json estimate.json
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Note:
    start: float
    end: float
    midi: int


def load_notes(path: str | Path) -> list[Note]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("melody", data)
        if isinstance(data, dict):
            data = data.get("melody", [])
    out = []
    for item in data:
        if item.get("midi") is None or item.get("note") in ("-", "rest"):
            continue
        start = float(item.get("start", 0.0))
        end = float(item.get("end", start))
        if end <= start:
            continue
        out.append(Note(start, end, int(round(float(item["midi"])))) )
    return sorted(out, key=lambda n: (n.start, n.end, n.midi))


def overlap(a: Note, b: Note) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def match_cost(ref: Note, est: Note, onset_tolerance: float) -> float:
    pitch_cost = abs(ref.midi - est.midi) * 2.0
    onset_cost = min(abs(ref.start - est.start) / max(onset_tolerance, 1e-9), 4.0)
    duration_cost = min(abs((ref.end - ref.start) - (est.end - est.start)), 2.0)
    overlap_bonus = min(overlap(ref, est), 1.0)
    return pitch_cost + onset_cost + duration_cost - overlap_bonus


def align(reference: list[Note], estimate: list[Note], onset_tolerance: float = 0.15):
    """Global ordered alignment; returns (cost, operations).

    Operations are tuples `(kind, ref_index_or_none, est_index_or_none)` where
    kind is `match`, `missing`, or `extra`.
    """
    n, m = len(reference), len(estimate)
    gap = 4.0
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * gap
        back[i][0] = ("missing", i - 1, None)
    for j in range(1, m + 1):
        dp[0][j] = j * gap
        back[0][j] = ("extra", None, j - 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = [
                (dp[i - 1][j - 1] + match_cost(reference[i - 1], estimate[j - 1], onset_tolerance), "match", i - 1, j - 1),
                (dp[i - 1][j] + gap, "missing", i - 1, None),
                (dp[i][j - 1] + gap, "extra", None, j - 1),
            ]
            best = min(options, key=lambda x: x[0])
            dp[i][j] = best[0]
            back[i][j] = best[1:]
    ops = []
    i, j = n, m
    while i or j:
        op = back[i][j]
        if op is None:
            break
        kind, ri, ej = op
        ops.append((kind, ri, ej))
        if kind == "match":
            i -= 1; j -= 1
        elif kind == "missing":
            i -= 1
        else:
            j -= 1
    return dp[n][m], list(reversed(ops))


def evaluate(reference: list[Note], estimate: list[Note], onset_tolerance: float = 0.15) -> dict:
    _, ops = align(reference, estimate, onset_tolerance)
    matched = []
    missing = []
    extra = []
    for kind, ri, ei in ops:
        if kind == "match":
            matched.append((reference[ri], estimate[ei]))
        elif kind == "missing":
            missing.append(reference[ri])
        else:
            extra.append(estimate[ei])
    exact = [
        (r, e) for r, e in matched
        if r.midi == e.midi and abs(r.start - e.start) <= onset_tolerance
    ]
    onset_errors = [e.start - r.start for r, e in matched]
    return {
        "reference_count": len(reference),
        "estimate_count": len(estimate),
        "matched_alignment_count": len(matched),
        "exact_pitch_onset_count": len(exact),
        "recall": len(exact) / len(reference) if reference else 1.0,
        "precision": len(exact) / len(estimate) if estimate else 0.0,
        "f1": (2 * len(exact) / (len(reference) + len(estimate))) if reference or estimate else 1.0,
        "missing_count": len(missing),
        "extra_count": len(extra),
        "pitch_mismatch_count": sum(1 for r, e in matched if r.midi != e.midi),
        "mean_abs_onset_error": sum(abs(x) for x in onset_errors) / len(onset_errors) if onset_errors else None,
        "matched": [{"ref_midi": r.midi, "est_midi": e.midi, "ref_start": r.start, "est_start": e.start} for r, e in matched],
        "missing": [{"midi": n.midi, "start": n.start, "end": n.end} for n in missing],
        "extra": [{"midi": n.midi, "start": n.start, "end": n.end} for n in extra],
    }


def main() -> int:
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        return 2
    tolerance = float(sys.argv[3]) if len(sys.argv) == 4 else 0.15
    result = evaluate(load_notes(sys.argv[1]), load_notes(sys.argv[2]), tolerance)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
