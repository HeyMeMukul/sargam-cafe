# Kaggle batch pipeline (GPU) — STARS / ROSVOT / Pop2Piano / VirtuosoNet

Never commit API keys. The runner reads `KAGGLE_API_TOKEN` (and `KAGGLE_USERNAME`)
from the environment only. `.creds.sh` on your machine exports them; keep that
file local (it is git-ignored).

## Vendor checkpoints once (Kaggle dataset or /kaggle/input)
- STARS: `gwx314/STARS` + HF `checkpoints/stars_bilingual` + `rmvpe`
- ROSVOT: `RickyL-2000/ROSVOT` `checkpoints.zip` (rosvot/rwbd/rmvpe, M4Singer)
- Pop2Piano: `sweetcocoa/pop2piano` (research-only, no license declared)
- VirtuosoNet: `jdasam/virtuosoNet` + `dasaem/virtuosonet` (all-rights-reserved default)

## Run
```bash
export KAGGLE_API_TOKEN=...   # from local .creds.sh, never commit
export KAGGLE_USERNAME=...
python kaggle/run_batch.py --input /path/to/clips --out /path/to/scores \
  --transcriber stars --arranger pop2piano --performer virtuosonet
```

`run_batch.py` here is a contract stub: stage order, JSON schemas, and strict
eval gating. Fill model-specific imports on the Kaggle kernel where GPU + ckpts
exist. Local `backend/pianist_v2` remains the CPU-safe default.
