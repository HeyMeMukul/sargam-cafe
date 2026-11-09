#!/usr/bin/env python3
"""Validate a ground-truth annotation file against annotations/schema.json rules.

Usage: python evaluation/annotations/validate.py evaluation/annotations/<song>.json
Exits non-zero on any violation (missing fields, bad order, overlap).
"""
import json
import sys


def main(path: str) -> int:
    data = json.load(open(path, encoding="utf-8"))
    if not isinstance(data, list) or not data:
        print("annotation must be a non-empty JSON array")
        return 1
    prev_end = -1e9
    for i, e in enumerate(data):
        for k in ("start", "end", "midi"):
            if k not in e:
                print(f"event {i}: missing {k}")
                return 1
        s, en, m = float(e["start"]), float(e["end"]), int(e["midi"])
        if not s < en:
            print(f"event {i}: non-positive duration")
            return 1
        if s < prev_end - 1e-6:
            print(f"event {i}: out of order / overlap (start {s} < prev end {prev_end})")
            return 1
        if not 21 <= m <= 108:
            print(f"event {i}: midi {m} outside piano range")
            return 1
        prev_end = en
    print(f"annotation OK: {len(data)} events in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
