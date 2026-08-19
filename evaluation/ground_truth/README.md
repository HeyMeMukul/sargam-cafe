# Sargam Cafe Boundary-Aware Annotation Workflow

## Purpose

This workflow creates a small, high-quality evaluation set for singing-note transcription. It is intentionally independent of the production decoder. The annotations are the acceptance target, not a copy of the model output.

## Clip selection

Annotate five clips of 10–20 seconds each. Include the supplied Tum Se Hi excerpt, two additional excerpts with clear syllabic melodies, one legato or ornament-heavy excerpt, and one faster or syncopated excerpt. Keep the clips from the same application domain: polyphonic commercial audio with a dominant singing melody.

## Required annotation fields

Each musical event is one object with the following fields:

```json
{
  "start": 2.410,
  "end": 2.840,
  "midi": 61,
  "pitch_cents": 0.0,
  "articulation": "attack|retrigger|legato|rest",
  "ornament": "none|meend|gamak|kan|trill",
  "confidence": "high|medium|low",
  "annotator_note": "optional explanation"
}
```

`start` and `end` are the musical note boundaries in seconds. `midi` is the nearest stable semitone of the sung note, while `pitch_cents` records a sustained deviation when it is musically meaningful. A same-pitch repeated syllable must be represented as two events with separate starts, not as one extended event. A glide should remain one note with an ornament marker unless the target pitch is held as a new stable note.

## Annotation procedure

Use a separated vocal stem and the original mix together. First listen to the phrase at normal speed. Then inspect a spectrogram and F0 overlay at 0.25×–0.5× speed. Mark the perceptual onset where the new sung pitch begins, not merely the consonant’s broadband energy spike. Mark the offset where the stable note ends or a rest begins. For uncertain boundaries, mark `confidence: "low"` and add a note; do not silently force a decision.

A second pass must listen to the original mix to catch attacks that are weak in the separated stem. A second annotator or a later blind review should resolve disagreements. Store both raw annotations and the adjudicated annotation; never overwrite the raw record.

## Acceptance metrics

Use strict ordered alignment with a one-to-one event mapping. Report exact pitch accuracy, onset tolerance at 50 ms and 100 ms, offset tolerance at 100 ms, Correct Onset and Pitch, Correct Onset/Pitch/Offset, missing events, extra events, same-pitch retrigger accuracy, and ornament false-attack rate. Do not use scale membership as a substitute for pitch correctness.

A decoder candidate may be promoted only if it improves the adjudicated set’s primary metric without causing a material regression in extra-event rate. The initial promotion gate is:

| Metric | Initial gate |
|---|---:|
| Correct Onset + Pitch + Offset | improve over baseline on at least 4 of 5 clips |
| Ordered event F1 | improve on the aggregate set |
| Missing-event rate | decrease by at least 15% relative |
| Extra-event rate | not increase by more than 5% relative |
| Same-pitch retrigger F1 | not decrease |
| Browser completion | 3 consecutive clean runs |

These are engineering gates, not claims of human-level perfection. The thresholds should be tightened after the first adjudicated set exists.

## File layout

```text
evaluation/ground_truth/
  README.md
  tum_se_hi_clip_01.raw.json
  tum_se_hi_clip_01.adjudicated.json
  manifest.json
```

The manifest records audio SHA-256, clip start/end, annotator identity, annotation date, and adjudication status. Every evaluation report must include the manifest hash so results cannot silently change when an annotation is edited.

## Important limitation

The user-provided sargam phrases are useful diagnostic sequence references but do not contain verified onset and offset labels. They must remain separate from this ground-truth set and must not be used alone to claim 100% accuracy.
