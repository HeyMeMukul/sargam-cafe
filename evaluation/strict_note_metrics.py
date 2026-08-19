#!/usr/bin/env python3
"""Strict ordered onset/pitch/offset metrics for singing-note ground truth.

Usage:
    python evaluation/strict_note_metrics.py reference.json estimate.json [onset_tol] [offset_tol]

The reference and estimate may be arrays of note objects or objects containing
`melody`. Alignment remains globally ordered; metrics never use a best local
subsequence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from strict_sequence import align, load_notes


def evaluate(reference, estimate, onset_tolerance=0.05, offset_tolerance=0.10):
    _, ops = align(reference, estimate, onset_tolerance)
    matched = []
    missing = []
    extra = []
    for kind, ri, ei in ops:
        if kind == 'match':
            matched.append((reference[ri], estimate[ei]))
        elif kind == 'missing':
            missing.append(reference[ri])
        else:
            extra.append(estimate[ei])

    def pitch_onset(r, e):
        return r.midi == e.midi and abs(r.start - e.start) <= onset_tolerance

    def pitch_onset_offset(r, e):
        return pitch_onset(r, e) and abs(r.end - e.end) <= offset_tolerance

    correct_pitch_onset = [(r, e) for r, e in matched if pitch_onset(r, e)]
    correct_onset_pitch_offset = [(r, e) for r, e in matched if pitch_onset_offset(r, e)]
    onset_errors = [e.start - r.start for r, e in matched]
    offset_errors = [e.end - r.end for r, e in matched]

    def f1(count, ref_count, est_count):
        return 2 * count / (ref_count + est_count) if ref_count or est_count else 1.0

    return {
        'reference_count': len(reference),
        'estimate_count': len(estimate),
        'aligned_count': len(matched),
        'missing_count': len(missing),
        'extra_count': len(extra),
        'pitch_mismatch_count': sum(r.midi != e.midi for r, e in matched),
        'correct_pitch_onset_count': len(correct_pitch_onset),
        'correct_onset_pitch_offset_count': len(correct_onset_pitch_offset),
        'onset_recall': len(correct_pitch_onset) / len(reference) if reference else 1.0,
        'onset_precision': len(correct_pitch_onset) / len(estimate) if estimate else 0.0,
        'onset_pitch_f1': f1(len(correct_pitch_onset), len(reference), len(estimate)),
        'onset_pitch_offset_recall': len(correct_onset_pitch_offset) / len(reference) if reference else 1.0,
        'onset_pitch_offset_precision': len(correct_onset_pitch_offset) / len(estimate) if estimate else 0.0,
        'onset_pitch_offset_f1': f1(len(correct_onset_pitch_offset), len(reference), len(estimate)),
        'mean_abs_onset_error': sum(abs(x) for x in onset_errors) / len(onset_errors) if onset_errors else None,
        'mean_abs_offset_error': sum(abs(x) for x in offset_errors) / len(offset_errors) if offset_errors else None,
        'matched': [
            {
                'ref_midi': r.midi,
                'est_midi': e.midi,
                'ref_start': r.start,
                'est_start': e.start,
                'ref_end': r.end,
                'est_end': e.end,
                'pitch_onset_correct': pitch_onset(r, e),
                'onset_pitch_offset_correct': pitch_onset_offset(r, e),
            }
            for r, e in matched
        ],
        'missing': [{'midi': n.midi, 'start': n.start, 'end': n.end} for n in missing],
        'extra': [{'midi': n.midi, 'start': n.start, 'end': n.end} for n in extra],
    }


def main():
    if len(sys.argv) not in (3, 4, 5):
        print(__doc__)
        return 2
    onset_tolerance = float(sys.argv[3]) if len(sys.argv) >= 4 else 0.05
    offset_tolerance = float(sys.argv[4]) if len(sys.argv) >= 5 else 0.10
    result = evaluate(
        load_notes(Path(sys.argv[1])),
        load_notes(Path(sys.argv[2])),
        onset_tolerance,
        offset_tolerance,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
