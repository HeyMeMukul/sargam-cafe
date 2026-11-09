# v2 vs v3 decoder — Vocadito 5-clip benchmark (2026-11-09)

Method: `backend/vocal_melody.py` (v2 = commit e32e23a flat-150ms, v3 = c3d4690 weak-onset + 60/120ms + retrigger + beat soft-prior), root C / Bilawal (MIDI absolute), references = A1 notes via `map_to_schema.py`, metrics = `strict_note_metrics.evaluate` (musical = 0.4P+0.3O+0.2D+0.1R).

| clip (lang) | ref | v2 est | v3 est | v2 musical | v3 musical | v2 opoF1 | v3 opoF1 | v2 rep | v3 rep |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 Tagalog | 59 | 62 | 64 | 0.580 | 0.646 | 0.198 | 0.293 | 1.000 | 1.000 |
| 2 Spanish | 76 | 60 | 68 | 0.390 | 0.407 | 0.265 | 0.236 | 0.292 | 0.292 |
| 6 English | 73 | 69 | 87 | 0.455 | 0.502 | 0.239 | 0.300 | 0.333 | 0.417 |
| 10 | 35 | 14 | 18 | 0.253 | 0.267 | 0.204 | 0.189 | 0.500 | 0.500 |
| 20 | 31 | 29 | 31 | 0.373 | 0.437 | 0.167 | 0.194 | 0.250 | 0.500 |
| mean | — | — | — | 0.410 | 0.452 | 0.215 | 0.242 | — | — |

v3 wins 5/5 musical, 3/5 opoF1. Offsets remain the weakest axis (0.19–0.30).
Clip 10 under-segments in both (recall 0.17 v3) — needs boundary-recall work (M2 next).
MIR-1K proxy skipped for scoring (no human onsets). ST500 long-clip run pending.
