#!/usr/bin/env python3
"""CLI tool for the Piano Master Agent.

Loads the audio snippet ONCE and returns salience scores for a batch of notes.
Called by the opencode agent via its bash tool.

Usage:
    python test_notes.py <audio_filepath> <start_time> [note1 note2 ...]

Output:
    Single JSON object on stdout:
    {
      "C4": 0.42,
      "C#4": 0.11,
      ...
    }
"""
import argparse
import json
import sys

import librosa
import numpy as np
import warnings

warnings.filterwarnings('ignore', module='librosa')

NOTE_TO_BIN = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
    'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
}


def salience_for_snippet(y, sr, target_bin):
    """Compute the normalized salience score for a pitch class in a loaded snippet."""
    y_harmonic, _ = librosa.effects.hpss(y)
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    mean_chroma = np.mean(chroma, axis=1)
    target_energy = mean_chroma[target_bin]
    total_energy = np.sum(mean_chroma)
    if total_energy == 0:
        return 0.0
    score = float(target_energy / total_energy)
    return min(1.0, score * 3.0)


def main():
    parser = argparse.ArgumentParser(description="Test notes against an audio snippet.")
    parser.add_argument("audio_filepath", help="Path to the audio file")
    parser.add_argument("start_time", type=float, help="Time in seconds for the snippet")
    parser.add_argument("notes", nargs="+", help="Notes to test, e.g. C4 C#4 D4")
    parser.add_argument("--duration", type=float, default=1.0, help="Snippet duration (default 1.0s)")
    parser.add_argument("--step", type=float, default=0.0,
                        help="If >0, scan forward from start_time in --step increments to the end of the file")
    parser.add_argument("--end", type=float, default=None,
                        help="Optional upper bound (seconds) for --step scanning; default: end of file")
    args = parser.parse_args()

    valid_notes = []
    for note in args.notes:
        base_note = ''.join(ch for ch in note if not ch.isdigit())
        if base_note not in NOTE_TO_BIN:
            print(json.dumps({"error": f"Invalid note: {note}"}))
            sys.exit(1)
        valid_notes.append((note, base_note))

    try:
        y, sr = librosa.load(args.audio_filepath, sr=22050, mono=True)
    except Exception as e:
        print(json.dumps({"error": f"Failed to load audio: {str(e)}"}))
        sys.exit(1)

    duration = librosa.get_duration(y=y, sr=sr)

    y_harmonic, _ = librosa.effects.hpss(y)
    win = int(round(args.duration * sr))

    def salience_at(t, bins):
        lo = int(round(t * sr))
        hi = min(len(y), lo + win)
        if hi <= lo:
            return {b: 0.0 for b in bins}
        snippet = y_harmonic[lo:hi]
        chroma = librosa.feature.chroma_cqt(y=snippet, sr=sr)
        mean_chroma = np.mean(chroma, axis=1)
        total_energy = np.sum(mean_chroma)
        out = {}
        for b in bins:
            if total_energy == 0:
                out[b] = 0.0
            else:
                out[b] = min(1.0, float(mean_chroma[b] / total_energy) * 3.0)
        return out

    targets = list(zip(valid_notes, [NOTE_TO_BIN[bn] for _, bn in valid_notes]))
    results = {}

    if args.step > 0:
        # Scan mode: report the winning note for each timestamp window
        t = args.start_time
        upper = args.end if args.end is not None else duration
        frames = []
        while t < min(upper, duration):
            scores = salience_at(t, [b for _, b in targets])
            best = max(targets, key=lambda nt: scores[nt[1]])
            best_note = best[0][0]
            frames.append({
                "time": round(t, 2),
                "note": best_note,
                "score": round(scores[best[1]], 3),
                "scores": {note: round(scores[b], 3) for (note, _), b in targets},
            })
            t += args.step
        print(json.dumps({"duration": round(duration, 2), "frames": frames}))
        return

    scores = salience_at(args.start_time, [b for _, b in targets])
    for note, _ in valid_notes:
        results[note] = round(scores[NOTE_TO_BIN[''.join(ch for ch in note if not ch.isdigit())]], 3)

    print(json.dumps(results))


if __name__ == "__main__":
    main()