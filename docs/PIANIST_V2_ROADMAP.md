# SargamKing Pianist v2 — Real AI Pianist Roadmap

Goal: vocal audio → professional piano performance (not literal MIDI typing).

## Why v1 fails (accepted)
1. Librosa onsets ≠ sung boundaries (E1: 74/76 onsets had no pitch change).
2. `median pitch per window` collapses meend/gamak; RMS ≠ touch; beat-snap kills rubato.
3. Scale-snapping creates wrong-but-in-scale notes.
4. Basic-Pitch-on-vocals is a category error; ROSVOT demoted for wrong reasons.
5. LLM reviewers are evidence-gated to `keep` — they cost tokens, change nothing.
6. Frontend fakes pedal/rubato with sliders; sampler can't bend meend.
7. No ground-truth annotations → no iteration signal.
8. No arrangement layer (melody+bass+voicing), no performance model.

## v2 pipeline (all deterministic-first, learned where it matters)
```
audio
 → Demucs stems (cache by content hash)
 → F0 ensemble: CREPE + RMVPE (frame agreement, voicing intersection)
 → Boundaries: ROSVOT MIDI (primary) + pitch-persistence + onset-energy (secondary)
 → Ensemble merge (this repo: backend/pianist_v2/ensemble.py)
 → Key/chords (Krumhansl + chroma templates, unchanged)
 → Arrangement (backend/pianist_v2/arranger.py): melody + bass + inner voices + fills
 → Performance (backend/pianist_v2/performance.py): phrase-arch timing/dynamics/pedal
 → Render contracts (__PIANO_SCORE__ + __SOLO__ compatible)
 → Eval gate (strict_sequence + annotation schema)
 → Kaggle batch (kaggle/): Demucs→STARS/ROSVOT→Pop2Piano→VirtuosoNet
```

## Model choices (researched, Kaggle-fit)
- Transcription boundaries: **STARS bilingual** primary (lyrics-aware when available), **ROSVOT** fallback/robust (vendored in `ROSVOT/`). RMVPE as F0 front-end (already inside both).
- Arrangement: **Pop2Piano** (`sweetcocoa/pop2piano`, HF transformers) for cover-style piano; rule-based arranger here as offline-safe default.
- Performance: **VirtuosoNet** (MusicXML → expressive MIDI) on Kaggle; deterministic phrase-arch here as default.
- Do NOT use: VOCANO (torch1.6/apex, stale), MIDI-DDSP (no piano, TF-only), PerformanceRNN (generative, not a renderer).

## Milestones (one by one)
- [x] M0: maps + roadmap (this doc) + v2 skeleton
- [ ] M1: annotation schema + 10-clip ground truth + strict gate in CI
- [ ] M2: ensemble decoder (ROSVOT boundaries + CREPE/RMVPE pitch) + E1 regression
- [ ] M3: rule-based arranger (bass/inner/fills) behind `SARGAM_ARRANGER=v2`
- [ ] M4: phrase-arch performance (timing/dynamics/pedal) behind `SARGAM_PERFORMANCE=v2`
- [ ] M5: Kaggle batch: vendor STARS + pop2piano + virtuosonet, run 50-clip eval
- [ ] M6: promote ensemble+arranger to default when strict F1 improves on held-out set
- [ ] M7: neural render (DDSP-piano / VST) to replace sampler bends

## Flags (all safe, fallback to v1)
- `SARGAM_ENSEMBLE=on` — use pianist_v2 ensemble instead of raw CREPE melody
- `SARGAM_ARRANGER=v2` — arrange for piano instead of literal vocal copy
- `SARGAM_PERFORMANCE=v2` — phrase-arch performance instead of slider-only Human Touch
- `SARGAM_PIANIST_V2=on` — all three above at once

## Kaggle token
`.creds.sh` contains `KAGGLE_API_TOKEN` (verified present, never committed). `kaggle/run_batch.py` reads it from env only.
