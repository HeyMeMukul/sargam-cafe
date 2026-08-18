#!/usr/bin/env python3
"""Build a reference annotation for the Sargam Cafe benchmark.

Takes a simple, paste-able list of known notes and writes the mir_eval
reference JSON that `benchmark.py` consumes.

Input format (a text file, one note per line):
    <note_name> <start_seconds> <end_seconds>
    e.g.
    D4 0.00 0.60
    E4 0.62 1.10
    F4 1.10 1.50

Note names may include octave and sharps/flats (D4, F#4, Bb3, E#4). Bare names
default to octave 4.

Usage:
    python evaluation/make_reference.py notes.txt evaluation/references/song.json
"""
import json
import re
import sys

PC = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'Fb': 4,
      'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10,
      'Bb': 10, 'B': 11, 'Cb': 11, 'E#': 5, 'B#': 0}


def note_to_midi(name):
    m = re.match(r'^([A-G][#b]?)(-?\d+)?$', name.strip())
    if not m:
        raise ValueError(f"bad note: {name}")
    pc = PC[m.group(1)]
    octave = int(m.group(2)) if m.group(2) else 4
    return (octave + 1) * 12 + pc


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    refs = []
    with open(src) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            note = parts[0]
            start = float(parts[1])
            end = float(parts[2]) if len(parts) > 2 else start + 0.5
            refs.append({"start": round(start, 3), "end": round(end, 3),
                         "midi": note_to_midi(note)})
    with open(dst, 'w') as f:
        json.dump(refs, f, indent=2)
    print(f"wrote {len(refs)} reference notes -> {dst}")


if __name__ == "__main__":
    main()
