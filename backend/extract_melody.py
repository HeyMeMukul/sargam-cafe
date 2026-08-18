#!/usr/bin/env python3
"""Extract the full-song melody as a Sargam timeline.

Robust approach for polyphonic tracks (drums, chords, vocals all present):
  - HPSS to isolate harmonic content (removes drums/percussion)
  - CQT chromagram, per-frame dominant pitch class
  - Temporal median smoothing to suppress chord-tone flicker
  - Merge holds into segments >= a minimum duration

Each segment carries the dominant note name and its Sargam label relative to
the discovered Root. This is a "scale-following" melody line, not a note-perfect
AMT transcription.

Usage:
    python extract_melody.py <audio_filepath> <root_note> [--out out.json]

Output (stdout or --out):
    {
      "root": "D#",
      "root_note": "D#4",
      "duration": 323.45,
      "melody": [
        {"start": 0.5, "end": 2.1, "note": "D#", "sargam": "Sa"},
        ...
      ],
      "sargam_counts": { "Sa": 40, "Re": 12, ... }
    }
"""
import argparse
import json
import sys

import librosa
import numpy as np
import warnings

warnings.filterwarnings('ignore', module='librosa')

NOTE_TO_PC = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
    'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
}
PC_TO_NOTE = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

INTERVAL_TO_SARGAM = {
    0: 'Sa', 1: 're', 2: 'Re', 3: 'ga', 4: 'Ga', 5: 'Ma',
    6: 'ma', 7: 'Pa', 8: 'dha', 9: 'Dha', 10: 'ni', 11: 'Ni',
}


def normalize_root(root: str) -> int:
    name = root.strip()
    while name and name[-1].isdigit():
        name = name[:-1]
    if name not in NOTE_TO_PC:
        raise ValueError(f"Unknown root note: {root}")
    return NOTE_TO_PC[name]


def main():
    parser = argparse.ArgumentParser(description="Extract full-song Sargam melody.")
    parser.add_argument("audio_filepath", help="Path to the audio file")
    parser.add_argument("root_note", help="Root/Sa note, e.g. D# or D#4")
    parser.add_argument("--smooth", type=float, default=0.6,
                        help="Median smoothing window in seconds (default 0.6)")
    parser.add_argument("--minseg", type=float, default=1.0,
                        help="Minimum segment length in seconds (default 1.0)")
    parser.add_argument("--out", help="Optional output file path")
    args = parser.parse_args()

    try:
        root_pc = normalize_root(args.root_note)
    except ValueError as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

    try:
        y, sr = librosa.load(args.audio_filepath, sr=22050, mono=True)
    except Exception as e:
        print(json.dumps({"error": f"Failed to load audio: {str(e)}"}))
        sys.exit(1)

    duration = librosa.get_duration(y=y, sr=sr)

    # 1. Harmonic separation — remove drums/percussion energy
    y_harmonic, _ = librosa.effects.hpss(y)

    # 2. CQT chromagram (12 pitch classes over time)
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    times = librosa.times_like(chroma)

    # 3. Dominant pitch class per frame, then temporal median smoothing
    dom = np.argmax(chroma, axis=0)
    win = int(round(args.smooth * (chroma.shape[1] / times[-1])))
    dom_smooth = np.zeros_like(dom)
    for i in range(len(dom)):
        lo = max(0, i - win // 2)
        hi = min(len(dom), i + win // 2 + 1)
        dom_smooth[i] = np.median(dom[lo:hi])

    # 4. Merge consecutive same-pc frames into segments (min duration filter)
    segments = []
    cur, start = int(dom_smooth[0]), 0
    for i in range(1, len(dom_smooth) + 1):
        if i == len(dom_smooth) or int(dom_smooth[i]) != cur:
            end_t = times[min(i, len(dom_smooth) - 1)]
            if end_t - times[start] >= args.minseg:
                note_name = PC_TO_NOTE[cur]
                interval = (cur - root_pc) % 12
                segments.append({
                    "start": round(float(times[start]), 2),
                    "end": round(float(end_t), 2),
                    "note": note_name,
                    "sargam": INTERVAL_TO_SARGAM[interval],
                })
            if i < len(dom_smooth):
                cur = int(dom_smooth[i])
            start = i

    sargam_counts = {}
    for seg in segments:
        sargam_counts[seg["sargam"]] = sargam_counts.get(seg["sargam"], 0) + 1

    result = {
        "root": PC_TO_NOTE[root_pc],
        "root_note": args.root_note,
        "duration": round(duration, 2),
        "melody": segments,
        "sargam_counts": sargam_counts,
    }

    output = json.dumps(result)
    if args.out:
        with open(args.out, "w") as f:
            f.write(output)
        print(f"Wrote {len(segments)} melody segments to {args.out}")
    else:
        print(output)


if __name__ == "__main__":
    main()