# Sargam Cafe — Evaluation / Regression Suite

Objective, measurable checks for the transcription pipeline (accuracy review §2.8).
A patch should only be merged if it improves these metrics on held-out clips.

## Requirements
- `mir_eval` (in `backend/requirements.txt`)

## How to run
```bash
# Self-consistency only (no reference yet):
python evaluation/benchmark.py /path/to/extracted_melody.json

# Full evaluation against a reference annotation:
python evaluation/benchmark.py /path/to/extracted_melody.json /path/to/reference.json
```

`extracted_melody.json` is the JSON emitted by `backend/vocal_melody.py`
(e.g. `{"root": "F#", "melody": [{"start":..,"end":..,"midi":..}, ...]}`).

## Reference annotation format
A JSON array of notes, one per sung event:
```json
[
  {"start": 0.5, "end": 1.2, "midi": 68},
  {"start": 1.4, "end": 2.0, "midi": 66}
]
```
Store these in `evaluation/references/<song>.json`. Annotate at least 10-20
clips across singers, keys, tempos and production styles; expand to 50+.

## Metrics reported
- **Onset F1** — did we find the right note start times?
- **Note F1 (overlap)** — pitch+time match without strict offsets.
- **Note F1 (offset)** — stricter onset/pitch tolerance.
- **Octave-error rate** — approx fraction of notes a 12-semitone octave off.
- **Note-count ratio** — est/reference count (under-count = missing notes).
- Self-consistency: flagged out-of-scale count, confidence coverage.

## Regression rule
Do not merge a change that improves scale membership but lowers note F1,
onset F1, or raises the octave-error rate on held-out songs. A wrong in-scale
note is still wrong.

## Baseline outputs
Keep the current outputs under `evaluation/baselines/` so you can compare raw
extraction vs reviewed vs scale-constrained vs each new candidate model.
