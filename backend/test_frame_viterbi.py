#!/usr/bin/env python3
"""Dependency-light tests for the isolated frame lattice."""
import numpy as np

from frame_viterbi import DecoderConfig, decode_frame_lattice


def main():
    times = np.arange(0.0, 0.30, 0.01)
    midi = np.array([60.0] * 8 + [np.nan] * 5 + [62.0] * 17)
    voicing = np.array([0.9] * 8 + [0.05] * 5 + [0.9] * 17)
    onset = np.array([0.9] + [0.05] * 7 + [0.0] * 5 + [0.85] + [0.05] * 16)
    rms = np.array([0.6] * 8 + [0.02] * 5 + [0.65] * 17)
    notes = decode_frame_lattice(
        times,
        midi,
        voicing,
        onset,
        rms,
        config=DecoderConfig(attack_threshold=0.35),
    )
    assert len(notes) == 2, notes
    assert [n['midi'] for n in notes] == [60, 62], notes
    assert notes[0]['start'] == 0.0
    assert notes[1]['start'] >= 0.13
    assert notes[1]['end'] > notes[1]['start']
    print('frame lattice regression passed')


if __name__ == '__main__':
    main()
