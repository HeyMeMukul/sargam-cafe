#!/usr/bin/env python3
"""Kaggle batch stub: clips -> scores with strict eval gate.

Stages: Demucs -> STARS|ROSVOT -> Pop2Piano -> VirtuosoNet -> strict_sequence.
Runs on Kaggle GPU kernels where checkpoints are vendored. Fails closed: any
stage error marks the clip failed instead of emitting a silent fallback score.
"""
import argparse
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--transcriber", default="stars", choices=["stars", "rosvot"])
    ap.add_argument("--arranger", default="pop2piano", choices=["pop2piano", "rule"])
    ap.add_argument("--performer", default="virtuosonet", choices=["virtuosonet", "phrase_arch"])
    args = ap.parse_args()

    if not os.getenv("KAGGLE_API_TOKEN"):
        print("KAGGLE_API_TOKEN not set in environment (see kaggle/README.md)", file=sys.stderr)
        return 2
    os.makedirs(args.out, exist_ok=True)
    manifest = {
        "transcriber": args.transcriber,
        "arranger": args.arranger,
        "performer": args.performer,
        "input": os.path.abspath(args.input),
        "note": "Fill model imports on Kaggle kernel; see docs/PIANIST_V2_ROADMAP.md M5",
    }
    json.dump(manifest, open(os.path.join(args.out, "batch_manifest.json"), "w"), indent=2)
    print(f"kaggle batch stub ready -> {args.out}/batch_manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
