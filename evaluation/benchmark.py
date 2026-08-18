#!/usr/bin/env python3
"""Sargam Cafe transcription benchmark (accuracy review §2.8).

Evaluates an extracted melody JSON against a reference annotation using
mir_eval: onset F1, note F1 (with/without offsets), octave-error rate, and
note-count ratio. Also reports basic self-consistency stats (out-of-scale,
confidence coverage) so a patch can be measured rather than judged by ear.

Reference annotation format (JSON list):
    [
      {"start": 0.5, "end": 1.2, "midi": 68},
      ...
    ]

Usage:
    python evaluation/benchmark.py <extracted_melody.json> [reference.json]
"""
import json
import sys

import numpy as np


def load_extracted(path):
    d = json.load(open(path))
    mel = d.get("melody", d) if isinstance(d, dict) else d
    if isinstance(mel, dict) and "melody" in mel:
        mel = mel["melody"]
    times = []
    freqs = []
    for s in mel:
        if s.get("midi") is None:
            continue
        t0 = float(s.get("start", 0))
        t1 = float(s.get("end", t0 + 0.1))
        f = 440.0 * 2 ** ((int(s["midi"]) - 69) / 12.0)
        times.append([t0, t1])
        freqs.append(f)
    return np.array(times), np.array(freqs), mel


def load_reference(path):
    ann = json.load(open(path))
    times = []
    freqs = []
    for s in ann:
        t0 = float(s.get("start", 0))
        t1 = float(s.get("end", t0 + 0.1))
        f = 440.0 * 2 ** ((int(s["midi"]) - 69) / 12.0)
        times.append([t0, t1])
        freqs.append(f)
    return np.array(times), np.array(freqs)


def octave_error_rate(ref_times, ref_freqs, est_times, est_freqs):
    """Fraction of estimated notes whose pitch is within ~12 semitones of the
    nearest reference pitch but in the wrong octave."""
    if len(ref_freqs) == 0 or len(est_freqs) == 0:
        return float("nan")
    err = 0.0
    count = 0
    for (t0, t1), f in zip(est_times, est_freqs):
        mid = (t0 + t1) / 2.0
        # nearest reference by onset
        ref_mid = (ref_times[:, 0] + ref_times[:, 1]) / 2.0
        idx = int(np.argmin(np.abs(ref_mid - mid)))
        rf = ref_freqs[idx]
        semis = 12 * np.log2(f / rf)
        if abs(semis) > 0.5 and abs(abs(semis) - 12) < 1.5:
            err += 1
        count += 1
    return err / count if count else float("nan")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    est_path = sys.argv[1]
    ref_path = sys.argv[2] if len(sys.argv) > 2 else None

    est_times, est_freqs, mel = load_extracted(est_path)

    print("=== Sargam Cafe benchmark ===")
    print(f"extracted notes: {len(mel)}")

    # self-consistency: out-of-scale + confidence coverage
    scale_iv = {0, 2, 4, 5, 7, 9, 11}  # Bilawal default; pass root via CLI ideally
    root = None
    import os
    # root guess from first note pitch-class 0 -> Sa is rare; skip root check here
    n_oos = 0
    n_conf = 0
    for s in mel:
        if s.get("midi") is not None:
            if s.get("out_of_scale_candidate"):
                n_oos += 1
        if s.get("pitch_confidence") is not None:
            n_conf += 1
    print(f"flagged out-of-scale: {n_oos}")
    print(f"notes with pitch_confidence: {n_conf}/{len(mel)}")

    if ref_path is None:
        print("(no reference annotation provided — skipping mir_eval metrics)")
        return

    import mir_eval
    ref_times, ref_freqs = load_reference(ref_path)

    # onset-only F1
    try:
        onset_p, onset_r, onset_f = mir_eval.transcription.onset_precision_recall_f1(
            ref_times, est_times)
        print(f"\nonet F1: {onset_f:.3f} (P {onset_p:.3f} R {onset_r:.3f})")
    except Exception as e:
        print("onset F1 error:", e)

    # note F1 without offsets (returns P, R, F1, avg_overlap)
    try:
        res = mir_eval.transcription.precision_recall_f1_overlap(
            ref_times, ref_freqs, est_times, est_freqs)
        p, r, f = res[0], res[1], res[2]
        print(f"note F1 (overlap): {f:.3f} (P {p:.3f} R {r:.3f})")
    except Exception as e:
        print("note F1 error:", e)

    # note F1 with offsets
    try:
        p2, r2, f2 = mir_eval.transcription.offset_precision_recall_f1(
            ref_times, est_times, offset_ratio=0.2, offset_min_tolerance=0.05)
        print(f"note F1 (offset): {f2:.3f} (P {p2:.3f} R {r2:.3f})")
    except Exception as e:
        print("note F1(offset) error:", e)

    oer = octave_error_rate(ref_times, ref_freqs, est_times, est_freqs)
    print(f"octave-error rate (approx): {oer:.3f}")

    # note-count ratio
    if len(ref_freqs) > 0:
        ratio = len(est_freqs) / len(ref_freqs)
        print(f"note-count ratio (est/ref): {ratio:.2f}")


if __name__ == "__main__":
    main()
