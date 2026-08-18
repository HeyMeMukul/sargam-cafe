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

## Installation

### Prerequisites
- **Python 3.10+**
- **Node.js 18+** (for the Vite frontend)
- **ffmpeg** (required by Demucs for audio decoding):
  ```bash
  # Debian/Ubuntu
  sudo apt install ffmpeg
  # macOS
  brew install ffmpeg
  ```
- **CUDA GPU (optional but recommended)** — the vocal-melody pipeline runs torchcrepe on GPU when available and falls back to CPU otherwise.

### 1. Backend
The backend has many dependencies, including heavyweight ML libraries (torch, torchaudio, demucs, torchcrepe). Install them **in this order** so torch matches your machine's CUDA/CPU:

```bash
cd backend

# 1a) Create + activate a virtualenv
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 1b) Install torch/torchaudio FIRST from pytorch.org so it matches your CUDA
#     (CPU-only example; pick your CUDA version on https://pytorch.org/get-started)
pip install torch torchaudio

# 1c) Install everything else from requirements.txt
pip install -r requirements.txt

# 1d) Run the backend (port 8000)
./venv/bin/python -m uvicorn main:app --port 8000
```

> **Note on requirements.txt:** the base list (fastapi, librosa, numpy, basic-pitch, mir-eval, etc.) is at the top. The heavy ML deps (torch, torchaudio, `demucs==4.1.0`, `torchcrepe==0.0.24`) are listed at the bottom with a note — **install torch/torchaudio separately first** (step 1b) to avoid downloading a mismatched binary, then `pip install -r requirements.txt` pulls in the rest.

### 2. Frontend
```bash
# from the project root
npm install
npm run dev        # dev server
npm run build      # production build (dist/)
```

### 3. AI agents (optional)
The transcription uses the `opencode` CLI (main agent + section reviewers on a free DeepSeek model). Install it separately if you want agent-driven transcription:
```bash
# see https://opencode.ai/docs for your platform
# ensure `opencode` is on your PATH
```
Without it, the deterministic audio pipeline (key → chords → melody) still runs; only the LLM validation subagents are skipped.

## Usage
1. Start the backend and frontend.
2. Open the app in the browser, click **Load Track**, pick a song.
3. The pipeline runs: key detection → chord detection → vocal melody extraction → phrase-aware "Human Touch" rendering.
4. Use the sidebar to toggle **Chords / Melody / With Song** layers and the **Human Touch** sliders (Expression, Rubato, Cadence breath, Pedal, Melody prominence, Seed).

## Evaluation
`evaluation/` contains a mir_eval-based benchmark (`benchmark.py`) and reference-annotation helpers (`make_reference.py`, `validate_phrases.py`) to measure transcription accuracy (onset F1, note F1, octave-error rate, note-count ratio).

## Optional reference-guided accuracy mode

Audio-only singing transcription cannot promise 100% song-specific note accuracy when the source contains accompaniment, breath noise, octave ambiguity, or repeated syllables. For a known song, the new opt-in mode accepts a JSON annotation containing lyric/sargam phrases and aligns it to the detector timeline with monotonic dynamic programming. It does not silently rewrite ordinary runs: guided events carry `reference_guided`, `reference_token`, `source_pitch`, and `review_flags` provenance, while detector-only events outside annotated windows remain available.

Set `SARGAM_REFERENCE_FILE` to an annotation file before starting the backend. A ready-to-run example for the supplied clip is `backend/reference_examples_tum_se_hi.json`. Its four annotated phrases score **100.0% (32/32 pitch classes)** on the repository’s fixed-window evaluator, compared with 71.9% for the frozen CREPE baseline and 65.6% for the fresh ROSVOT run. This is a measured song-specific mode, not a claim that arbitrary unannotated songs are automatically perfect.

```bash
export SARGAM_REFERENCE_FILE=/absolute/path/to/backend/reference_examples_tum_se_hi.json
export SARGAM_TRANSCRIBER=rosvot   # or auto/crepe, depending on the installed models
```

Repeated reference tokens are emitted as explicit retriggers so the piano scheduler does not collapse repeated syllables into one held note. Remove `SARGAM_REFERENCE_FILE` to return to fully audio-derived transcription.
