# Annotation Guide — `evaluation/annotations/`

Ground-truth note annotations for the M1 eval set. Templates (`.template.json`)
contain **no ground-truth pitches** — only format examples. Real annotations
live in sibling `<song>.json` files as a bare JSON array.

## Clips (durations via `ffprobe format=duration`)

| Template | Audio | Duration |
|---|---|---:|
| `alfaaz_hamza_malik_zain_ali.template.json` | `backend/uploads/Alfaaz_-_Hamza_Malik_Zain_Ali_Lyrics.wav` | 185.05 s |
| `ambarsariya_fukrey.template.json` | `backend/uploads/Ambarsariya Fukrey 320 Kbps.mp3` | 248.24 s |
| `pal_pal_dil_ke_paas.template.json` | `backend/uploads/Pal Pal Dil Ke Paas(KoshalWorld.Com).mp3` | 314.75 s |
| `saathiyaa.template.json` | `backend/uploads/Saathiyaa_320(KoshalWorld.Com).mp3` | 310.78 s |
| `tera_hone_laga_hoon_remix.template.json` | `backend/uploads/Tera Hone Laga Hoon Remix ... 130bpm-440hz.mp3` | 286.04 s |
| `tum_se_hi_full.template.json` | `backend/uploads/Tum Se Hi Jab We Met 128 Kbps.mp3` | 323.50 s |
| `raga_of_revenge_full.template.json` | `backend/uploads/YTDown_YouTube_Raga-of-Revenge ... 120bpm-440hz.mp3` | 131.34 s |
| `tum_se_hi_excerpt_xbstsos0.template.json` | `backend/uploads/tum-se-hi-jab-we-met-128-kbps-t6kdsjrq_xbStSoS0.mp3` (byte-identical copy at `public/tum-se-hi-jab-we-met-128-kbps-t6kdsjrq_xbStSoS0.mp3`) | 29.73 s |
| `tum_se_hi_excerpt_t6kdsjrq.template.json` | `backend/uploads/tum-se-hi-jab-we-met-128-kbps_T6KdsjrQ.mp3` | 93.23 s |
| `raga_of_revenge_excerpt_jbtfo6oc.template.json` | `backend/uploads/ytdown-youtube-raga-of-revenge ... _JbTfo6oc.mp3` | 32.03 s |

## How to annotate

Each sung note is one event:

```json
[
  {"start": 0.5, "end": 1.2, "midi": 68},
  {"start": 1.4, "end": 2.0, "midi": 66}
]
```

- **onset (`start`, seconds):** the perceptual start of the new sung pitch
  (not the consonant's broadband energy spike). Listen at normal speed, then
  refine boundaries at 0.25×–0.5× with a spectrogram/F0 overlay.
- **offset (`end`, seconds):** where the stable note ends or a rest begins.
  Must satisfy `end > start`.
- **midi (integer 21–108):** nearest stable semitone of the sung note.
- Optional: `sargam` (syllable), `phrase_id` (integer phrase index).

Rules (enforced by `validate.py` and `gate_v2.py`):

1. **Time order:** events sorted by `start`.
2. **No overlap:** every `start >= previous end` (tolerance 1 µs).
3. **No merged retriggers:** a same-pitch repeated syllable is TWO events with
   separate starts, never one extended event.
4. **Glides:** one event unless the target pitch is held as a new stable note.
5. **Uncertainty:** annotate your best guess; record low confidence in an
   `annotator_note` field rather than silently forcing a boundary.

## Workflow

```bash
# 1. Copy the template's "annotation" array into <song>.json and fill it in.
#    (Never commit pitches into a *.template.json file.)

# 2. Validate structure/order/overlap (exit non-zero on violation):
python evaluation/annotations/validate.py evaluation/annotations/<song>.json

# 3. Score an extracted melody against it (strict ordered alignment):
python evaluation/gate_v2.py evaluation/annotations/<song>.json extracted_melody.json

# 4. Regression check vs a recorded baseline:
python evaluation/gate_v2.py evaluation/annotations/<song>.json new_melody.json \
    --baseline evaluation/baselines/<song>.gate.json
```

The gate rejects `*.template.json` inputs outright (exit 1): placeholder
examples must never score as ground truth.

## Schema

Field definitions live in `schema.json`. Structural validation lives in
`validate.py`. Metric semantics (strict ordered sequence + onset/pitch/offset)
live in `../strict_sequence.py` and `../strict_note_metrics.py`.
