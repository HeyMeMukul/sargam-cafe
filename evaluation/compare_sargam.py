#!/usr/bin/env python3
"""Compare the extracted melody against a known sargam reference (Tum Se Hi).

Builds a reference annotation from sargam lines + the known verse start time,
then aligns the extracted pitch-class sequence against it and reports a
pitch-match accuracy per phrase (robust to octave and timing offsets via a
best-match sliding window).

Usage:
    python evaluation/compare_sargam.py <extracted_melody.json> <verse_start_s>
"""
import json
import re
import sys

SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
ROOT_PC = 6  # F#
DEG = {'S': 0, 'R': 2, 'G': 4, 'M': 5, 'P': 7, 'D': 9, 'N': 11,
       'n': 10, 'm': 6, 'r': 1, 'g': 3, 'd': 8}
PC = {'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11,'E#':5,'B#':0}

# The user's sargam, in song order (verse starts at verse_start_s in the clip).
# Only the FIRST 4 phrases are actually played in the 30s audio.
SARGAM = [
    "G P R R G P P D",           # Na hai ye panaa
    "D S' P P D G G",            # Na khona hi hainn
    "S R G P R R G P P D",       # Tere na hone Jaanee
    "D S' P P D G G",            # Kyun hona hi hainn
]

def sargam_to_note_names(phrase):
    out = []
    for t in phrase.split():
        up = t.count("'")
        deg = t[0]
        off = DEG.get(deg, 0) + 12 * up
        pc = (ROOT_PC + off) % 12
        octv = 4 + (off // 12)
        out.append(SHARP[pc] + str(octv))
    return out

def note_pc(note):
    m = re.match(r'^([A-G][#b]?)(-?\d+)?$', note.strip())
    if not m:
        return None
    return PC.get(m.group(1))

def load_extracted(path):
    d = json.load(open(path))
    mel = d.get('melody', d) if isinstance(d, dict) else d
    if isinstance(mel, dict) and 'melody' in mel:
        mel = mel['melody']
    return mel

def best_match_accuracy(ext_pcs, ref_pcs):
    """Sliding-window best alignment: accuracy of ref pcs against a contiguous
    window of ext pcs (handles octave via pitch-class)."""
    best = 0.0
    n = len(ref_pcs)
    for i in range(max(0, len(ext_pcs) - n + 1)):
        win = ext_pcs[i:i + n]
        hits = sum(1 for a, b in zip(win, ref_pcs) if a == b)
        acc = hits / n
        if acc > best:
            best = acc
    return best

def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    ext_path = sys.argv[1]
    verse_start = float(sys.argv[2])
    mel = load_extracted(ext_path)

    # build reference pitch-class sequence
    ref_notes = []
    ref_names = []
    for phrase in SARGAM:
        for n in sargam_to_note_names(phrase):
            ref_notes.append(note_pc(n))
            ref_names.append(n)
    ref_pcs = [p for p in ref_notes]

    # extracted pitch classes at/after verse_start
    ext_notes = [s['note'] for s in mel if s['start'] >= verse_start - 0.5]
    ext_pcs = [note_pc(n) for n in ext_notes]
    ext_pcs = [p for p in ext_pcs if p is not None]

    print(f"verse_start={verse_start}s | ref notes: {len(ref_pcs)} | ext notes(from {verse_start}s): {len(ext_pcs)}")
    print(f"\nKnown reference (first 40): {' '.join([SHARP[p] for p in ref_pcs[:40]])}")
    print(f"Extracted  (first 40):      {' '.join([SHARP[p] for p in ext_pcs[:40]])}")

    acc = best_match_accuracy(ext_pcs, ref_pcs)
    print(f"\nBest pitch-class match accuracy (whole melody): {acc*100:.1f}%")

    # per-phrase
    print("\nPer-phrase match:")
    idx = 0
    for phrase in SARGAM:
        names = sargam_to_note_names(phrase)
        lp = [note_pc(n) for n in names]
        sub_acc = best_match_accuracy(ext_pcs, lp)
        print(f"  {phrase:28} ({names[0]}...): {sub_acc*100:.0f}%")
        idx += len(lp)

if __name__ == '__main__':
    main()
