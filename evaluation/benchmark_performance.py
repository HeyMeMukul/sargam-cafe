#!/usr/bin/env python3
"""Sargam Cafe PERFORMANCE benchmark (audit item P9).

Beyond transcription note-F1, a "good" extracted melody must also be playable:
timing should be steady (no runaway gaps), durations should be consistent,
phrases should be continuous, and dynamics should have a usable range. This
script measures those performance qualities directly from an extracted melody
JSON (the vocal_melody.py output format — a dict with a "melody" list of note
objects carrying start, end, midi, velocity, pitch_confidence, phrase_id, ...).

Reference annotation format (JSON list), same as benchmark.py:
    [
      {"start": 0.5, "end": 1.2, "midi": 68},
      ...
    ]

Usage:
    python3 evaluation/benchmark_performance.py <extracted.json> [reference.json]
"""
import json
import sys

import numpy as np


def load_melody(path):
    d = json.load(open(path))
    mel = d.get("melody", d) if isinstance(d, dict) else d
    if isinstance(mel, dict) and "melody" in mel:
        mel = mel["melody"]
    return mel


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    mel_path = sys.argv[1]
    ref_path = sys.argv[2] if len(sys.argv) > 2 else None

    mel = load_melody(mel_path)

    print("=== Sargam Cafe performance benchmark ===")
    print(f"extracted notes: {len(mel)}")

    starts = np.array([float(s.get("start", 0.0)) for s in mel])
    ends = np.array([float(s.get("end", s.get("start", 0.0))) for s in mel])
    mids = [s.get("midi") for s in mel]
    vels = [s.get("velocity") for s in mel]
    confs = [s.get("pitch_confidence") for s in mel]

    # 1. note count + confidence coverage
    n_conf = sum(1 for c in confs if c is not None)
    coverage = n_conf / len(mel) if mel else float("nan")
    print("\n[Note count & confidence coverage]")
    print(f"  notes: {len(mel)}")
    print(f"  pitch_confidence present: {n_conf}/{len(mel)} "
          f"({fmt(coverage * 100, 1)}%)")

    # 2. timing deviation: inter-onset gaps vs median gap (beat-grid proxy)
    print("\n[Timing deviation (inter-onset gaps)]")
    if len(starts) < 2:
        print("  (fewer than 2 notes — skipping)")
    else:
        gaps = np.diff(starts)
        med_gap = float(np.median(gaps))
        gap_dev = np.abs(gaps - med_gap)
        print(f"  median inter-onset gap: {fmt(med_gap)}s "
              f"(~{fmt(60.0 / med_gap, 1)} bpm if on the beat)")
        print(f"  gap mean: {fmt(np.mean(gaps))}s  std: {fmt(np.std(gaps))}s")
        print(f"  mean abs deviation from median gap: {fmt(np.mean(gap_dev))}s")
        # fitted beat grid: gaps quantized to nearest integer multiple of the
        # median gap — the residual is the grid-timing error.
        if med_gap > 0:
            multiples = np.round(gaps / med_gap)
            multiples = np.clip(multiples, 1, None)
            grid_residual = np.abs(gaps - multiples * med_gap)
            print(f"  fitted-grid residual (gaps vs nearest beat multiple): "
                  f"mean {fmt(np.mean(grid_residual))}s, "
                  f"max {fmt(np.max(grid_residual))}s")

    # 3. duration error vs median duration
    print("\n[Duration error]")
    durs = ends - starts
    durs = durs[durs > 0]
    if len(durs) == 0:
        print("  (no positive durations)")
    else:
        med_dur = float(np.median(durs))
        dur_err = np.abs(durs - med_dur)
        print(f"  median duration: {fmt(med_dur)}s")
        print(f"  mean duration: {fmt(np.mean(durs))}s  std: {fmt(np.std(durs))}s")
        print(f"  mean abs duration error vs median: {fmt(np.mean(dur_err))}s")

    # 4. phrase continuity: fraction of notes whose gap < 1.5x median gap
    print("\n[Phrase continuity]")
    if len(starts) < 2:
        print("  (fewer than 2 notes — skipping)")
    else:
        med_gap = float(np.median(gaps))
        if med_gap > 0:
            n_cont = int(np.sum(gaps < 1.5 * med_gap))
            frac_cont = n_cont / len(gaps)
            print(f"  gaps < 1.5x median gap: {n_cont}/{len(gaps)} "
                  f"({fmt(frac_cont * 100, 1)}%)")
        else:
            print("  median gap is 0 — cannot assess continuity")
        # also flag the largest breaks explicitly
        big = gaps[gaps >= 1.5 * med_gap] if med_gap > 0 else gaps
        print(f"  unnatural breaks (>= 1.5x median): {len(big)}")
        n_phrases = len({s.get("phrase_id") for s in mel if s.get("phrase_id") is not None})
        if n_phrases:
            print(f"  distinct phrase_id values: {n_phrases}")

    # 5. velocity curve stats
    print("\n[Velocity / dynamics]")
    vels_num = np.array([v for v in vels if v is not None], dtype=float)
    if len(vels_num) == 0:
        print("  (no velocity data)")
    else:
        vmin, vmax = float(np.min(vels_num)), float(np.max(vels_num))
        print(f"  min: {fmt(vmin)}  max: {fmt(vmax)}  mean: {fmt(np.mean(vels_num))}")
        print(f"  dynamic range (max - min): {fmt(vmax - vmin)}")

    # 6. note-count ratio vs reference
    if ref_path is not None:
        print("\n[Note-count ratio vs reference]")
        try:
            ref = json.load(open(ref_path))
            ref_list = ref.get("melody", ref) if isinstance(ref, dict) else ref
            if isinstance(ref_list, dict) and "melody" in ref_list:
                ref_list = ref_list["melody"]
            n_ref = len(ref_list)
            ratio = len(mel) / n_ref if n_ref else float("nan")
            print(f"  reference notes: {n_ref}")
            print(f"  extracted notes: {len(mel)}")
            print(f"  note-count ratio (est/ref): {fmt(ratio, 2)}")
        except Exception as e:
            print(f"  (could not read reference: {e})")


if __name__ == "__main__":
    main()
