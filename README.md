# Sargam Cafe

AI ear-training & song transcription — upload any song and get a human-like piano performance.

## What it does
- Detects the **key / scale** (Hindustani thaat + Western scale) deterministically
- Extracts the **sung vocal melody** with octaves, rhythm, dynamics and Indian-classical ornaments (meend, gamak, kan-swar, sustain)
- Detects the **chord progression** for left-hand backing
- Renders an **expressive piano solo** (phrase-aware rubato, dynamic arcs, articulation, sustain pedal) with a "Human Touch" engine
- Play **melody / chords / with-song** layers independently

## Architecture
- **Frontend**: Vanilla JS + Vite + Tone.js (multi-sampled piano)
- **Backend**: FastAPI (uvicorn) + WebSocket, opencode AI agents
- **ML/audio**: Demucs (vocal separation), torchcrepe (pitch tracking), librosa, Basic-Pitch (fallback)
- **AI agents**: main agent + parallel section reviewers on the DeepSeek free model

## Quick start
```bash
# Backend (port 8000)
cd backend
./venv/bin/python -m uvicorn main:app --port 8000

# Frontend
npm install
npm run dev
```

## Evaluation
`evaluation/` contains a mir_eval-based benchmark (`benchmark.py`) and a reference-annotation helper (`make_reference.py`) to measure transcription accuracy (onset F1, note F1, octave-error rate, note-count ratio).
