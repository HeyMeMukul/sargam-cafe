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
`evaluation/` contains a mir_eval-based benchmark (`benchmark.py`), a strict ordered evaluator (`strict_sequence.py`), and reference-annotation helpers (`make_reference.py`, `validate_phrases.py`). The strict evaluator must be used for acceptance because it reports missing notes, extra notes, pitch mismatches, onset error, precision, recall, and F1 across the complete ordered sequence.

### Evidence side-channel for decoder research

Normal transcription output is unchanged by default. To export the frame-level CREPE evidence used by the decoder research pass, set `SARGAM_EVIDENCE_DIR` before starting the backend. The active `vocal_melody.py` path then writes one `<uploaded-filename>.evidence.json` file containing frame time, MIDI/F0, periodicity, voicing, RMS, onset strength, brightness, source-stem provenance, onsets, tempo, and the exact emitted melody.

```bash
export SARGAM_EVIDENCE_DIR=/absolute/path/to/evidence
```

This side-channel is diagnostic only; it does not alter the melody payload. In the first E1 experiment on the supplied Tum Se Hi clip, CREPE and RMVPE agreed within 0.5 semitones on 96.5% of frames where both were voiced, while 74 of 76 generic librosa onsets had no local CREPE pitch change within 100 ms. This strongly prioritizes boundary/onset decoding research over another pitch-scale correction, but it is not a ground-truth accuracy claim.

### Optional reference-conditioned mode

When the user has a trusted sargam sequence for a specific clip, `SARGAM_REFERENCE_FILE` enables an isolated reference-conditioned score. The backend automatically exports frame evidence, aligns exactly one event per supplied token, preserves repeated same-pitch tokens as retriggers, and disables the lossy section-review and duplicate-collapse stages for that guided score.

```bash
export SARGAM_REFERENCE_FILE=/absolute/path/to/backend/reference_examples_tum_se_hi.json
```

The supplied example is intentionally labeled `diagnostic_reference_conditioned`. It contains the user’s four Tum Se Hi phrases but does not contain manually verified onset/offset ground truth. Therefore this mode is useful for inspecting timing and auditioning a known phrase, but it must not be used to claim that the unrestricted audio-only transcriber is accurate. Leave the variable unset for normal audio-only behavior.
