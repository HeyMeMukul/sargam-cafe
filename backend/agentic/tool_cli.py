#!/usr/bin/env python3
"""CLI bridge for OpenCode agents to call typed Sargam specialist tools."""
from __future__ import annotations

import argparse
import json
import os

from .runtime_tools import build_runtime_registry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tool", choices=[
        "retrieve_skills", "get_track_manifest", "get_note_candidates",
        "inspect_pitch_window", "inspect_boundary_window", "score_hypothesis",
        "audition_phrase",
    ])
    parser.add_argument("--audio", dest="audio_path")
    parser.add_argument("--start", type=float)
    parser.add_argument("--end", type=float)
    parser.add_argument("--query")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--transcribers", nargs="*")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--resolution", default="coarse")
    parser.add_argument("--events-json")
    parser.add_argument("--reference-path")
    args = parser.parse_args()
    registry = build_runtime_registry()
    if args.tool == "retrieve_skills":
        result = registry.call("retrieve_skills", query=args.query or "melody transcription", limit=args.limit)
    elif args.tool == "get_track_manifest":
        result = registry.call("get_track_manifest", audio_path=args.audio_path)
    elif args.tool == "get_note_candidates":
        result = registry.call("get_note_candidates", audio_path=args.audio_path, start=args.start, end=args.end, transcribers=args.transcribers, min_confidence=args.min_confidence)
    elif args.tool == "inspect_pitch_window":
        result = registry.call("inspect_pitch_window", audio_path=args.audio_path, start=args.start, end=args.end, resolution=args.resolution)
    elif args.tool == "inspect_boundary_window":
        result = registry.call("inspect_boundary_window", audio_path=args.audio_path, start=args.start, end=args.end, resolution=args.resolution)
    elif args.tool == "score_hypothesis":
        result = registry.call("score_hypothesis", audio_path=args.audio_path, events=json.loads(args.events_json or "[]"), reference_path=args.reference_path)
    else:
        result = registry.call("audition_phrase", audio_path=args.audio_path, events=json.loads(args.events_json or "[]"), start=args.start, end=args.end)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
