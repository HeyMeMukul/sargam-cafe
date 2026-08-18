#!/usr/bin/env python3
"""Dependency-light tests for reference-conditioned score alignment."""
import numpy as np

from reference_score_alignment import ReferenceToken, align_reference_to_frames


def main():
    times = np.arange(0.0, 0.30, 0.01)
    midi = np.array([60.0] * 8 + [np.nan] * 5 + [60.0] * 17)
    voicing = np.array([0.9] * 8 + [0.03] * 5 + [0.9] * 17)
    onset = np.array([0.9] + [0.05] * 7 + [0.0] * 5 + [0.9] + [0.05] * 16)
    rms = np.array([0.6] * 8 + [0.01] * 5 + [0.6] * 17)
    ref = [ReferenceToken(0, 'S'), ReferenceToken(0, 'S')]
    events = align_reference_to_frames(times, midi, voicing, onset, rms, ref)
    assert len(events) == 2, events
    assert [e['reference_index'] for e in events] == [0, 1]
    assert events[1]['retrigger'] is True
    assert events[1]['start'] >= 0.13, events
    assert events[1]['end'] > events[1]['start']
    print('reference score alignment regression passed')


if __name__ == '__main__':
    main()
