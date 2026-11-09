#!/usr/bin/env python3
"""Strict ordered eval gate for SargamKing (M1/M5 entry point).

Compares an estimate score against an adjudicated ground-truth reference with
globally-ordered alignment (evaluation/strict_sequence.py). It reports missing
and extra events explicitly and never substitutes a best local subsequence for
whole-phrase accuracy.

Fails closed:
  * unreadable/invalid reference or estimate -> exit 2 (usage/IO) or 1 (fail)
  * empty reference or empty estimate        -> exit 1 with reason
  * any threshold miss                        -> exit 1 with reason

Usage:
    python evaluation/gate_v2.py --reference REF.json --estimate EST.json
        [--onset-tol 0.05] [--offset-tol 0.10]
        [--min-f1 0.0] [--max-missing-rate 1.0] [--max-extra-rate 1.0]
        [--report-out report.json]

Reference and estimate accept the same shapes as strict_sequence.load_notes:
a JSON array of note objects or an object containing `melody`.

Exit codes: 0 pass, 1 gate fail, 2 usage/IO error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strict_sequence as ss


GATE_VERSION = "gate_v2/1.0"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Strict ordered eval gate (fail-closed).")
    ap.add_argument("--reference", required=True, help="Adjudicated ground-truth JSON")
    ap.add_argument("--estimate", required=True, help="Estimate score JSON")
    ap.add_argument("--onset-tol", type=float, default=0.05)
    ap.add_argument("--offset-tol", type=float, default=0.10)
    ap.add_argument("--min-f1", type=float, default=0.0,
                    help="Minimum onset+pitch+offset F1 to pass")
    ap.add_argument("--max-missing-rate", type=float, default=1.0,
                    help="Maximum missing/reference ratio to pass")
    ap.add_argument("--max-extra-rate", type=float, default=1.0,
                    help="Maximum extra/estimate ratio to pass")
    ap.add_argument("--report-out", default=None, help="Write JSON report to this path")
    return ap


def evaluate(ref_notes: list[ss.Note], est_notes: list[ss.Note],
             onset_tol: float, offset_tol: float) -> dict:
    _, ops = ss.align(ref_notes, est_notes, onset_tol)
    matched, missing, extra = [], [], []
    for kind, ri, ei in ops:
        if kind == "match":
            matched.append((ref_notes[ri], est_notes[ei]))
        elif kind == "missing":
            missing.append(ref_notes[ri])
        else:
            extra.append(est_notes[ei])

    def onset_pitch_ok(r: ss.Note, e: ss.Note) -> bool:
        return r.midi == e.midi and abs(r.start - e.start) <= onset_tol

    def full_ok(r: ss.Note, e: ss.Note) -> bool:
        return onset_pitch_ok(r, e) and abs(r.end - e.end) <= offset_tol

    n_full = sum(1 for r, e in matched if full_ok(r, e))
    n_ref, n_est = len(ref_notes), len(est_notes)
    f1 = (2 * n_full / (n_ref + n_est)) if (n_ref or n_est) else 1.0
    return {
        "reference_count": n_ref,
        "estimate_count": n_est,
        "aligned_count": len(matched),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "correct_onset_pitch_offset_count": n_full,
        "onset_pitch_offset_f1": round(f1, 4),
        "missing_rate": round(len(missing) / n_ref, 4) if n_ref else 0.0,
        "extra_rate": round(len(extra) / n_est, 4) if n_est else 0.0,
        "mean_abs_onset_error": (
            round(sum(abs(e.start - r.start) for r, e in matched) / len(matched), 4)
            if matched else None
        ),
    }


def main() -> int:
    args = build_parser().parse_args()
    try:
        ref_notes = ss.load_notes(args.reference)
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        print(json.dumps({"pass": False, "reason": f"bad reference: {exc}"}, indent=2))
        return 2
    try:
        est_notes = ss.load_notes(args.estimate)
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        print(json.dumps({"pass": False, "reason": f"bad estimate: {exc}"}, indent=2))
        return 2

    if not ref_notes:
        report = {"pass": False, "gate": GATE_VERSION, "reason": "empty reference"}
        print(json.dumps(report, indent=2))
        return 1
    if not est_notes:
        report = {
            "pass": False, "gate": GATE_VERSION,
            "reference_count": len(ref_notes), "estimate_count": 0,
            "reason": "empty estimate",
        }
        print(json.dumps(report, indent=2))
        return 1

    metrics = evaluate(ref_notes, est_notes, args.onset_tol, args.offset_tol)
    failures = []
    if metrics["onset_pitch_offset_f1"] < args.min_f1:
        failures.append(
            f"f1 {metrics['onset_pitch_offset_f1']} < --min-f1 {args.min_f1}")
    if metrics["missing_rate"] > args.max_missing_rate:
        failures.append(
            f"missing_rate {metrics['missing_rate']} > {args.max_missing_rate}")
    if metrics["extra_rate"] > args.max_extra_rate:
        failures.append(
            f"extra_rate {metrics['extra_rate']} > {args.max_extra_rate}")

    report = {
        "pass": not failures,
        "gate": GATE_VERSION,
        "onset_tol": args.onset_tol,
        "offset_tol": args.offset_tol,
        "thresholds": {
            "min_f1": args.min_f1,
            "max_missing_rate": args.max_missing_rate,
            "max_extra_rate": args.max_extra_rate,
        },
        **metrics,
    }
    if failures:
        report["reason"] = "; ".join(failures)
    if args.report_out:
        Path(args.report_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
