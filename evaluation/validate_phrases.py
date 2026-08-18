#!/usr/bin/env python3
"""Validate the GENERAL extraction strategy against known phrase boundaries.

The reference (sargam) is used ONLY to MEASURE accuracy of whatever extraction
is passed in — never to guide extraction. This lets us iterate on the general
strategy and see if accuracy improves across the phrases.

Usage:
    python evaluation/validate_phrases.py <extracted.json>
"""
import json
import re
import sys

SHARP = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
ROOT_PC = 6  # F# for Tum Se Hi
DEG = {'S':0,'R':2,'G':4,'M':5,'P':7,'D':9,'N':11,'n':10,'m':6}
PC = {'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11,'E#':5,'B#':0}

# phrases in the 30s clip with user-provided start times
PHRASES = [
    (2.0,  "G P R R G P P D"),          # Na hai ye panaa
    (9.0,  "D S' P P D G G"),           # Na khona hi hainn
    (15.0, "S R G P R R G P P D"),      # Tere na hone Jaanee
    (24.0, "D S' P P D G G"),           # Kyun hona hi hainn
]

def sargam_pcs(phrase):
    out = []
    for t in phrase.split():
        up = t.count("'"); deg = t[0]
        off = DEG.get(deg, 0) + 12 * up
        pc = (ROOT_PC + off) % 12
        out.append(pc)
    return out

def note_pc(note):
    m = re.match(r'^([A-G][#b]?)', note)
    return PC.get(m.group(1)) if m else None

def load_extracted(path):
    d = json.load(open(path))
    mel = d.get('melody', d) if isinstance(d, dict) else d
    if isinstance(mel, dict) and 'melody' in mel:
        mel = mel['melody']
    return mel

def best_window_acc(ext_pcs, ref_pcs):
    best = 0.0
    n = len(ref_pcs)
    for i in range(len(ext_pcs) - n + 1):
        win = ext_pcs[i:i+n]
        acc = sum(1 for a,b in zip(win, ref_pcs) if a==b) / n
        best = max(best, acc)
    return best

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    mel = load_extracted(sys.argv[1])

    # next phrase start as the end bound of the current one
    bounds = [s for s, _ in PHRASES] + [30.0]
    total_hits = total_notes = 0
    print("Per-phrase accuracy of the GENERAL extraction strategy:")
    for i, (start, phrase) in enumerate(PHRASES):
        end = bounds[i+1]
        ref_pcs = sargam_pcs(phrase)
        ext = [s for s in mel if s['start'] >= start and s['start'] < end]
        ext_pcs = [note_pc(s['note']) for s in ext]
        ext_pcs = [p for p in ext_pcs if p is not None]
        acc = best_window_acc(ext_pcs, ref_pcs)
        # matched count scaled by acc * n_ref
        matched = round(acc * len(ref_pcs))
        total_hits += matched
        total_notes += len(ref_pcs)
        print(f"  phrase {i+1} ({start}-{end}s): {acc*100:.0f}%  ({matched}/{len(ref_pcs)} notes, ext had {len(ext_pcs)})")
    print(f"\n  OVERALL pitch accuracy: {total_hits/total_notes*100:.1f}%  ({total_hits}/{total_notes})")

if __name__ == '__main__':
    main()
