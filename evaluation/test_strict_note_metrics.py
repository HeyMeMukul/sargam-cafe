#!/usr/bin/env python3
from strict_note_metrics import evaluate
from strict_sequence import Note

reference = [Note(0.0, 0.5, 61), Note(0.5, 1.0, 61), Note(1.0, 1.5, 63)]
estimate = [Note(0.01, 0.48, 61), Note(1.0, 1.7, 62)]
result = evaluate(reference, estimate, onset_tolerance=0.05, offset_tolerance=0.10)
assert result['reference_count'] == 3
assert result['estimate_count'] == 2
assert result['missing_count'] == 1
assert result['extra_count'] == 0
assert result['correct_pitch_onset_count'] == 1
assert result['correct_onset_pitch_offset_count'] == 1
assert result['pitch_mismatch_count'] == 1
assert result['onset_pitch_offset_f1'] == 0.4
print('strict note metrics contract passed')
