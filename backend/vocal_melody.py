#!/usr/bin/env python3
"""Vocal melody extraction: Demucs source separation + CREPE pitch tracking.

The "professional pianist" pipeline — extracts the *actual sung melody* with
rhythm and dynamics:

  1. Demucs (htdemucs) separates the track into stems -> vocals.wav.
  2. torchcrepe (CREPE on GPU) pitch-tracks the vocal stem -> precise f0 curve
     + periodicity (confidence).
  3. librosa onset detection finds every note START (the real rhythm grid).
  4. For each onset->onset window, the note = median pitch inside the window;
     velocity = local RMS energy of the vocal stem.
  5. Note onsets are snapped to the beat grid so the rhythm matches the song.

Output shape matches what the frontend consumes:
    {
      "root": "F#", "duration": 29.5, "tempo": 129.2, "beats": [...],
      "melody": [
        {"start": 0.5, "end": 1.2, "note": "C#4", "pitch_class": "C#",
         "octave": 4, "midi": 61, "sargam": "Pa", "velocity": 0.7},
        ...
      ],
      "sargam_counts": {...}
    }

Usage:
    python vocal_melody.py <audio_filepath> <root_note> [--cache-dir DIR]
"""
import argparse
import json
import os
import sys
import tempfile
import warnings

warnings.filterwarnings('ignore')

import numpy as np

PC_TO_NOTE = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTE_TO_PC = {n: i for i, n in enumerate(PC_TO_NOTE)}
NOTE_TO_PC.update({'Db': 1, 'Eb': 3, 'Fb': 4, 'Gb': 6, 'Ab': 8, 'Bb': 10, 'Cb': 11, 'E#': 5, 'B#': 0})

INTERVAL_TO_SARGAM = {
    0: 'Sa', 1: 're', 2: 'Re', 3: 'ga', 4: 'Ga', 5: 'Ma',
    6: 'ma', 7: 'Pa', 8: 'dha', 9: 'Dha', 10: 'ni', 11: 'Ni',
}

THAAT_INTERVALS = {
    'Bilawal': [0, 2, 4, 5, 7, 9, 11],
    'Kalyan':  [0, 2, 4, 6, 7, 9, 11],
    'Khamaj':  [0, 2, 4, 5, 7, 9, 10],
    'Kafi':    [0, 2, 3, 5, 7, 9, 10],
    'Asavari': [0, 2, 3, 5, 7, 8, 10],
    'Bhairav': [0, 1, 4, 5, 7, 8, 11],
    'Bhairavi':[0, 1, 3, 5, 7, 8, 10],
    'Marwa':   [0, 1, 4, 6, 7, 9, 11],
    'Poorvi':  [0, 1, 4, 6, 7, 8, 11],
    'Todi':    [0, 1, 3, 6, 7, 8, 11],
    'Yaman':   [0, 2, 4, 6, 7, 9, 11],
}

def _running_median(values, window=11):
    """Median filter preserving NaN (unvoiced) positions."""
    n = len(values)
    out = np.full(n, np.nan)
    half = window // 2
    for i in range(n):
        if np.isnan(values[i]):
            continue
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        v = values[lo:hi]
        v = v[~np.isnan(v)]
        if len(v):
            out[i] = np.median(v)
    return out

def quantize_to_scale(midi, voiced, root_pc, intervals):
    """Scale-first frame quantization + octave stabilization.

    1. Smooth the raw midi pitch (voiced frames only).
    2. Snap every frame's pitch-class to the nearest scale degree,
       keeping the octave from the smoothed value.
    3. Windowed-median octave-stability pass collapses CREPE octave
       confusion (a short run of frames an octave off) to the register
       of the surrounding phrase.

    Returns a float array of quantized midi (NaN where unvoiced).
    """
    n = len(midi)
    out = np.full(n, np.nan)
    scale_pcs = sorted(set((root_pc + iv) % 12 for iv in intervals))
    if len(scale_pcs) < 5:  # defensive fallback to Bilawal/major
        scale_pcs = sorted(set((root_pc + iv) % 12 for iv in [0, 2, 4, 5, 7, 9, 11]))

    smooth = _running_median(np.where(voiced, midi, np.nan), window=11)
    for i in range(n):
        if not voiced[i] or np.isnan(smooth[i]):
            continue
        m = smooth[i]
        pc = int(round(m)) % 12
        best = min(scale_pcs, key=lambda s: min((s - pc) % 12, (pc - s) % 12))
        # place the scale pitch-class in the SAME octave as the raw pitch
        octv = int(round(m / 12))
        cand = min([best + 12 * octv, best + 12 * (octv - 1), best + 12 * (octv + 1)],
                   key=lambda c: abs(c - m))
        out[i] = cand

    # two passes: local-window octave shift (kills isolated octave flips)
    for _ in range(2):
        shifted = out.copy()
        for i in range(n):
            if np.isnan(out[i]):
                continue
            lo = max(0, i - 40)
            hi = min(n, i + 41)
            v = out[lo:hi]
            v = v[~np.isnan(v)]
            if len(v) < 3:
                continue
            local = float(np.median(v))
            if abs(out[i] - local) >= 7:
                shifted[i] += -12 if out[i] > local else 12
        out = shifted
    return out

def scale_note_name(root_name: str, intervals, interval: int) -> str:
    """Preferred (enharmonic-aware) note name for a scale degree.
    F# major degree 7 -> 'E#' (not 'F'); Bb major degree 4 -> 'E'.
    Out-of-scale pitches fall back to the plain chromatic name (never a
    double-sharp like F##)."""
    letters = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    naturals = [0, 2, 4, 5, 7, 9, 11]
    iv12 = interval % 12
    sorted_iv = sorted(set(i % 12 for i in intervals))
    root_letter = root_name[0]
    root_acc = 1 if '#' in root_name else (-1 if 'b' in root_name else 0)
    ri = letters.index(root_letter)
    root_pc = (naturals[ri] + root_acc) % 12
    if iv12 not in sorted_iv:
        # not a scale degree -> chromatic spelling of the absolute pitch class
        return PC_TO_NOTE[(root_pc + iv12) % 12]
    letter_step = sum(1 for iv2 in sorted_iv if iv2 <= iv12) - 1
    li = (ri + letter_step) % 7
    natural = naturals[li]
    target_pc = (root_pc + iv12) % 12
    diff = target_pc - natural
    if diff > 6:
        diff -= 12
    if diff < -6:
        diff += 12
    name = letters[li]
    if diff == 1:
        name += '#'
    elif diff == -1:
        name += 'b'
    return name
CACHE_DIR = os.path.join(tempfile.gettempdir(), "sargamking_stems")


def normalize_root(root: str) -> int:
    name = root.strip()
    while name and name[-1].isdigit():
        name = name[:-1]
    if name not in NOTE_TO_PC:
        raise ValueError(f"Unknown root note: {root}")
    return NOTE_TO_PC[name]


def separate_vocals(audio_filepath: str):
    """Run Demucs (cached), return path to the vocals stem (wav)."""
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    from demucs.audio import AudioFile, save_audio

    base = os.path.splitext(os.path.basename(audio_filepath))[0]
    cached = os.path.join(CACHE_DIR, base, "htdemucs", "vocals.wav")
    if os.path.exists(cached):
        return cached

    model = get_model('htdemucs')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    wav = AudioFile(audio_filepath).read(
        streams=0, samplerate=model.samplerate, channels=model.audio_channels)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()
    sources = apply_model(model, wav[None], device=device, split=True,
                          overlap=0.25, progress=False)[0]
    sources = sources * ref.std() + ref.mean()

    stems = model.sources
    out_dir = os.path.join(CACHE_DIR, base, "htdemucs")
    os.makedirs(out_dir, exist_ok=True)
    vocals_path = os.path.join(out_dir, 'vocals.wav')
    save_audio(sources[stems.index('vocals')], vocals_path, model.samplerate)
    return vocals_path


def pitch_track_torchcrepe(vocals_path: str):
    """Return (times, midi, periodicity, rms, sr) using CREPE on GPU."""
    import librosa
    import torch
    import torchcrepe

    y, sr = librosa.load(vocals_path, sr=16000, mono=True)
    audio = torch.from_numpy(y).float().unsqueeze(0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hop = 80  # 5ms at 16kHz
    out = torchcrepe.predict(audio, sr, hop_length=hop, fmin=80, fmax=800,
                             model='full', device=device, batch_size=256,
                             return_periodicity=True)
    f0, periodicity = out
    f0n = f0.squeeze(0).cpu().numpy()
    periodicity = periodicity.squeeze(0).cpu().numpy()

    midi = 69 + 12 * np.log2(np.where(f0n > 0, f0n, 440.0) / 440.0)
    times = np.arange(len(midi)) * hop / sr

    # RMS energy for velocity, aligned to crepe frames
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    # Onset strength (spectral flux) for articulation detection: a re-articulated
    # same-pitch note shows an energy spike at its attack even though the pitch
    # does not change. Resampled to the crepe frame grid.
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    oe_t = librosa.times_like(onset_env, sr=sr, hop_length=hop)
    onset_env_crepe = np.interp(times, oe_t, onset_env).astype(np.float32)

    # Spectral brightness (centroid, normalized 0..1) for inferred-pressure
    # expression (Manus P6): bright/harsh attacks -> higher perceived force.
    sc = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    sc_t = librosa.times_like(sc, sr=sr, hop_length=hop)
    sc_crepe = np.interp(times, sc_t, sc).astype(np.float32)
    sc_min, sc_max = float(sc_crepe.min()), float(sc_crepe.max())
    bright = np.clip((sc_crepe - sc_min) / (sc_max - sc_min + 1e-9), 0.0, 1.0).astype(np.float32)

    n = min(len(times), len(midi), len(rms), len(periodicity), len(onset_env_crepe), len(bright))
    return times[:n], midi[:n], periodicity[:n], rms[:n], sr, onset_env_crepe[:n], bright[:n]


def _merge_note_pair(a, b):
    """Combine two adjacent note dicts, preserving raw evidence fields."""
    da = a['end'] - a['start']
    db = b['end'] - b['start']
    tot = da + db
    wa = da / tot if tot > 0 else 0.5
    wb = db / tot if tot > 0 else 0.5
    merged = dict(a)
    merged['end'] = b['end']
    merged['midi'] = int(round(wa * a['midi'] + wb * b['midi']))
    merged['velocity'] = max(a.get('velocity', 0.0), b.get('velocity', 0.0))
    ra = a.get('raw_midi_float')
    rb = b.get('raw_midi_float')
    if ra is not None and rb is not None:
        merged['raw_midi_float'] = round(wa * ra + wb * rb, 3)
    elif rb is not None:
        merged['raw_midi_float'] = rb
    merged['pitch_confidence'] = max(a.get('pitch_confidence', 0.0),
                                     b.get('pitch_confidence', 0.0))
    merged['voicing_confidence'] = max(a.get('voicing_confidence', 0.0),
                                       b.get('voicing_confidence', 0.0))
    merged['attack_energy'] = max(a.get('attack_energy', 0.0),
                                  b.get('attack_energy', 0.0))
    ba = a.get('brightness')
    bb = b.get('brightness')
    if ba is not None and bb is not None:
        merged['brightness'] = round(wa * ba + wb * bb, 3)
    merged['_attack'] = a.get('_attack', 0.0)
    merged['_tail'] = b.get('_tail', a.get('_tail', 0.0))
    if merged.get('raw_midi_float') is not None:
        merged['cents_deviation'] = round(
            (merged['raw_midi_float'] - merged['midi']) * 100.0, 1)
    return merged


def _collapse_octave_errors(notes):
    """Pull a note that sits ~12 semitones off BOTH its neighbors (which are
    within +/-4 semitones of each other) into the neighbor register."""
    out = [dict(n) for n in notes]
    for i in range(1, len(out) - 1):
        m = out[i].get('midi')
        mp = out[i - 1].get('midi')
        mn = out[i + 1].get('midi')
        if m is None or mp is None or mn is None:
            continue
        if abs(mp - mn) > 4:
            continue
        d_prev = m - mp
        d_next = m - mn
        if abs(d_prev - 12) <= 1 and abs(d_next - 12) <= 1:
            shift = -12
        elif abs(d_prev + 12) <= 1 and abs(d_next + 12) <= 1:
            shift = 12
        else:
            continue
        out[i]['midi'] = int(round((mp + mn) / 2.0))
        r = out[i].get('raw_midi_float')
        if r is not None:
            out[i]['raw_midi_float'] = round(r + shift, 3)
            out[i]['cents_deviation'] = round(
                (out[i]['raw_midi_float'] - out[i]['midi']) * 100.0, 1)
    return out


def _merge_short_notes(notes, min_dur=0.060, att_thresh=0.05):
    """Merge very short notes (< min_dur) with no onset support (attack energy
    below att_thresh) into the neighbor with the closer pitch."""
    out = [dict(n) for n in notes]
    changed = True
    while changed:
        changed = False
        new = []
        i = 0
        while i < len(out):
            n = out[i]
            dur = n['end'] - n['start']
            att = n.get('attack_energy', 0.0)
            if dur < min_dur and att < att_thresh:
                has_left = bool(new)
                has_right = i + 1 < len(out)
                if has_left and has_right:
                    left = new[-1]
                    right = out[i + 1]
                    m = n.get('midi', 60)
                    if abs(left.get('midi', 60) - m) <= abs(right.get('midi', 60) - m):
                        new[-1] = _merge_note_pair(left, n)
                        i += 1
                        changed = True
                        continue
                    new.append(_merge_note_pair(n, right))
                    i += 2
                    changed = True
                    continue
                elif has_left:
                    new[-1] = _merge_note_pair(new[-1], n)
                    i += 1
                    changed = True
                    continue
                elif has_right:
                    new.append(_merge_note_pair(n, out[i + 1]))
                    i += 2
                    changed = True
                    continue
            new.append(n)
            i += 1
        out = new
    return out


def _smooth_notes_dp(notes):
    """Lightweight Viterbi/DP smoothing: octave collapse + short-note merge."""
    if len(notes) < 2:
        return list(notes)
    return _merge_short_notes(_collapse_octave_errors(notes))


def segment_notes(times, midi, periodicity, rms, sr, onsets, onset_env=None,
                  brightness=None, root_pc=None, intervals=None,
                  min_dur=0.035, period_thresh=0.4):
    """Decode notes from the pitch curve with onset evidence, hysteresis, and
    voicing-transition awareness (see accuracy review §2.2).

    Design:
      - Note identity and boundaries come from the RAW (unquantized, smoothed)
        pitch curve. A scale-quantized pitch is kept ONLY as a secondary soft
        label (scale_midi -> sargam / note name); it never decides note identity.
      - Note boundaries come from EITHER a librosa onset (re-articulation, even
        on the same pitch) OR a raw-pitch change that PERSISTS for a hysteresis
        interval (min_stable) — so vibrato, portamento and one-frame octave
        candidates do NOT create fake notes.
      - Short unvoiced gaps (< bridge) are bridged so a held note with a breath
        or vibrato is not split; a real gap becomes a rest.
      - A final DP smoothing pass collapses octave errors and merges isolated
        short notes with no onset support.

    Returns list of note dicts with raw evidence (raw_midi_float, cents,
    pitch_confidence, voicing_confidence) plus ornaments and scale_midi.
    """
    voiced = periodicity > period_thresh
    # raw continuous pitch (smoothed) for evidence/ornament analysis
    pitch_cont = _running_median(np.where(voiced, midi, np.nan), window=7)
    # RAW (unquantized, chromatic) pitch drives note identity and boundaries.
    # Heavier smoothing than pitch_cont so vibrato/jitter does not fragment notes.
    pitch_raw = _running_median(np.where(voiced, midi, np.nan), window=11)
    # Scale-quantized pitch kept ONLY as a secondary soft label (scale_midi ->
    # sargam / note name); it never decides note identity or boundaries.
    if root_pc is not None and intervals:
        pitch_scale = quantize_to_scale(midi, voiced, root_pc, intervals)
    else:
        pitch_scale = np.where(voiced, np.round(midi), np.nan)

    n = len(times)
    sr_frames = 1.0 / (times[1] - times[0]) if len(times) > 1 else 200.0  # ~200 fps
    frame_ms = 1000.0 / sr_frames

    # --- Build candidate boundaries ---
    # Onsets always create a potential note start.
    boundaries = [0.0] + [float(o) for o in onsets]

    # Pitch-change boundaries WITH hysteresis: a change starts a new note only
    # if the new pitch persists for min_stable frames (kill vibrato/jitter).
    # Longer persistence: only a SUSTAINED pitch change starts a new note, so
    # vibrato/ornamental micro-jitter merges into the parent note (general
    # over-fragmentation fix). Real onsets still split notes independently.
    min_stable = max(2, int(round(150.0 / frame_ms)))  # ~150ms persistence
    bridge = 0.09  # bridge unvoiced gaps shorter than this
    pitch_step = 0.7  # semitone tolerance: a real note change in raw pitch

    i = 0
    while i < n:
        p = pitch_raw[i]
        if np.isnan(p):
            i += 1
            continue
        # find next voiced frame that differs from the current raw pitch
        j = i + 1
        while j < n:
            # if we hit an unvoiced gap longer than bridge, stop (rest)
            if np.isnan(pitch_raw[j]):
                # look ahead across small gaps to see if pitch resumes
                k = j
                while k < n and np.isnan(pitch_raw[k]):
                    k += 1
                gap = times[k] - times[j] if k < n else 1e9
                if gap > bridge:
                    j = k  # real rest; resume search from k
                    break
                j = k
                continue
            if abs(pitch_raw[j] - p) > pitch_step:
                # confirm it persists for min_stable frames (tolerance on raw)
                ref = pitch_raw[j]
                persist = 1
                while j + persist < n and not np.isnan(pitch_raw[j + persist]) \
                        and abs(pitch_raw[j + persist] - ref) <= pitch_step:
                    persist += 1
                if persist >= min_stable:
                    boundaries.append(float(times[j]))
                    i = j
                    break
                else:
                    # brief excursion: skip past it (merge into parent)
                    j += persist
                    continue
            j += 1
        if j >= n:
            break
        i = j

    boundaries = sorted(set(b for b in boundaries if 0 <= b < float(times[-1])))
    # collapse boundaries closer than 25ms
    collapsed = []
    for b in boundaries:
        if not collapsed or b - collapsed[-1] > 0.025:
            collapsed.append(b)
        else:
            collapsed[-1] = min(collapsed[-1], b)
    boundaries = collapsed + [float(times[-1])]

    notes = []
    for bi in range(len(boundaries) - 1):
        t0 = boundaries[bi]
        t1 = boundaries[bi + 1]
        dur = t1 - t0
        if dur < min_dur:
            continue
        mask = (times >= t0) & (times < t1)
        if not mask.any():
            continue
        seg_raw = pitch_raw[mask]
        seg_raw = seg_raw[~np.isnan(seg_raw)]
        if len(seg_raw) == 0:
            continue
        # note identity from the RAW chromatic pitch median (not the scale)
        note_midi = int(round(float(np.median(seg_raw))))
        # scale-quantized soft label (sargam / note name only)
        seg_scale = pitch_scale[mask]
        seg_scale = seg_scale[~np.isnan(seg_scale)]
        scale_midi = int(round(float(np.median(seg_scale)))) if len(seg_scale) else note_midi
        seg_rms = rms[mask]
        velocity = float(np.mean(seg_rms))

        # --- raw evidence ---
        raw_cont = pitch_cont[mask]
        raw_cont = raw_cont[~np.isnan(raw_cont)]
        if len(raw_cont) > 0:
            raw_med = float(np.median(raw_cont))
        else:
            raw_med = float(note_midi)
        cents_dev = float((raw_med - note_midi) * 100.0)
        pitch_conf = float(np.mean(periodicity[mask])) if len(periodicity[mask]) > 0 else 0.0
        voiced_frac = float(np.mean(voiced[mask])) if len(voiced[mask]) > 0 else 0.0

        # --- ornament analysis from continuous curve ---
        seg_cont = pitch_cont[mask]
        seg_cont = seg_cont[~np.isnan(seg_cont)]
        ornament = None
        glide_to = None
        trill = False
        if len(seg_cont) >= 5:
            span = float(np.nanmax(seg_cont) - np.nanmin(seg_cont))
            d = np.diff(seg_cont)
            extrema = np.sum((d[:-1] > 0) & (d[1:] < 0)) + np.sum((d[:-1] < 0) & (d[1:] > 0))
            seg_center = float(np.median(seg_cont))
            peak_dev = float(np.nanmean(np.abs(seg_cont - seg_center)))
            if extrema >= 6 and span >= 2.5 and peak_dev >= 1.0 and dur >= 0.25:
                ornament = 'gamak'
                trill = True
            elif span >= 2.5 and dur >= 0.18 and extrema <= 3:
                ornament = 'meend'
                glide_dir = int(round(np.sign(float(seg_cont[-1]) - float(seg_cont[0])))) or 1
                glide_to = note_midi + glide_dir * max(1, int(round(abs(float(seg_cont[-1]) - note_midi))))
            elif dur < 0.09:
                ornament = 'kan'

        # onset-energy near the note start (attack) and near its end (tail),
        # used to decide whether an equal-pitch neighbour is a re-articulation.
        _att = 0.0
        _tail = 0.0
        _bright = 0.0
        if onset_env is not None:
            _att = float(np.nanmax(onset_env[mask][:3])) if onset_env[mask].size else 0.0
            _tail = float(np.nanmax(onset_env[mask][-3:])) if onset_env[mask].size else 0.0
        if brightness is not None:
            _bright = float(np.nanmean(brightness[mask])) if brightness[mask].size else 0.0
        # inferred-pressure expression features (Manus P6): attack energy and
        # spectral brightness feed the frontend's velocity/expression mapping.
        _att_norm = float(_att) if _att > 0 else 0.0
        entry = {'start': round(t0, 3), 'end': round(t1, 3),
                 'midi': note_midi, 'velocity': velocity,
                 'scale_midi': scale_midi,
                 'raw_midi_float': round(raw_med, 3),
                 'cents_deviation': round(cents_dev, 1),
                 'pitch_confidence': round(pitch_conf, 3),
                 'voicing_confidence': round(voiced_frac, 3),
                 'attack_energy': round(_att_norm, 3),
                 'brightness': round(_bright, 3),
                 '_attack': float(_att), '_tail': float(_tail)}
        if ornament:
            entry['ornament'] = ornament
        if glide_to is not None:
            entry['glide_to'] = int(glide_to)
        if trill:
            entry['trill'] = True
        notes.append(entry)

    # Merge adjacent same-pitch notes. Decide HOLD (one extended press) vs
    # RETRIGGER (multiple presses on the same key) using articulation evidence:
    # an onset-energy spike at the incoming note's attack, or a voicing dip
    # between them, means a genuine re-articulation (accuracy review §2.2 and
    # the user's repeated-note model). A smooth, attack-free continuation is a
    # single held note. This stops real repeated notes from being collapsed.
    merged = []
    for nn in notes:
        prev = merged[-1] if merged else None
        same_pitch = prev is not None and prev.get('midi') == nn.get('midi')
        gap = (nn['start'] - prev['end']) if prev is not None else 1e9
        retrigger = False
        if same_pitch and 0 <= gap < 0.09:
            # Re-articulation when there is a strong onset attack (vs the
            # previous note's tail) OR a voicing dip. Held "naaaa" with only
            # weak energy fluctuation merges into one note.
            strong_attack = (nn.get('_attack', 0) > (prev.get('_tail', 0) + 0.02)
                             and nn.get('_attack', 0) > 0.05)
            voicing_gap = (prev.get('voicing_confidence', 1) < 0.5
                           or nn.get('voicing_confidence', 1) < 0.5)
            if strong_attack or voicing_gap:
                retrigger = True
        if retrigger:
            nn['retrigger'] = True
            nn['articulation'] = 'retrigger'
            merged.append(dict(nn))
        elif same_pitch and 0 <= gap < 0.07:
            # hold: extend the previous note
            prev['end'] = nn['end']
            prev['velocity'] = max(prev['velocity'], nn['velocity'])
            prev['pitch_confidence'] = max(prev.get('pitch_confidence', 0),
                                           nn.get('pitch_confidence', 0))
            prev['voicing_confidence'] = max(prev.get('voicing_confidence', 0),
                                             nn.get('voicing_confidence', 0))
            prev['_tail'] = nn.get('_tail', prev.get('_tail', 0))
            m_prev = prev.get('raw_midi_float')
            m_new = nn.get('raw_midi_float')
            if m_prev is not None and m_new is not None:
                prev['raw_midi_float'] = round((m_prev + m_new) / 2.0, 3)
                prev['cents_deviation'] = round(
                    (prev['raw_midi_float'] - prev['midi']) * 100.0, 1)
        else:
            merged.append(dict(nn))

    # P6: lightweight Viterbi/DP smoothing pass (numpy only, no new deps):
    #   (a) merge short isolated notes with no onset support into a neighbor;
    #   (b) collapse octave errors (a note ~12 st off both neighbors).
    # Evidence fields (raw_midi_float, cents_deviation, pitch/voicing confidence,
    # attack_energy, brightness) are carried through the merge.
    merged = _smooth_notes_dp(merged)

    # strip internal features; flag true sustains
    out = []
    for nn in merged:
        nn.pop('_attack', None)
        nn.pop('_tail', None)
        if (nn['end'] - nn['start']) >= 2.0 and nn.get('ornament') != 'meend':
            nn['sustain'] = True
        out.append(nn)
    return out


def detect_onsets_and_tempo(vocals_path: str):
    """Return (onset_times, tempo, beats) from the vocal stem."""
    import librosa
    y, sr = librosa.load(vocals_path, sr=22050, mono=True)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onsets = librosa.frames_to_time(onset_frames, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    beats = librosa.frames_to_time(beat_frames, sr=sr)
    return [float(o) for o in onsets], round(bpm, 1), [round(float(b), 3) for b in beats]


def pitch_class_duration(audio_filepath: str):
    """Root-independent vocal pitch-class histogram weighted by sustained
    duration (where the sung melody RESTS). Used as the tonic-resolution cue
    for key detection. Returns a normalized 12-vector.
    """
    vocals_path = separate_vocals(audio_filepath)
    times, midi, periodicity, rms, sr, onset_env, bright = pitch_track_torchcrepe(vocals_path)
    voiced = periodicity > 0.4
    pc_dur = np.zeros(12)
    step = times[1] - times[0] if len(times) > 1 else 0.005
    for i in range(len(times)):
        if not voiced[i] or np.isnan(midi[i]):
            continue
        pc = int(round(midi[i])) % 12
        pc_dur[pc] += step
    total = pc_dur.sum()
    if total <= 0:
        return [0.0] * 12
    return [round(float(v / total), 4) for v in pc_dur]


def segment_phrases(melody, gap_thresh=0.18, cadence_dur=0.6, min_merge=4):
    """Group consecutive notes into phrases (performance phrases).

    A phrase boundary occurs when:
      - there is a gap (rest/breath) >= gap_thresh, OR
      - a LONG HELD note (>= cadence_dur) — the melodic cadence/breath point
        characteristic of this ballad style ("naaaaa...") — ends the phrase.

    Tiny fragments (< min_merge notes) produced by over-segmentation are merged
    into the previous phrase so we get meaningful phrase arcs for the
    performance renderer. Each note gets phrase_id; each phrase is annotated
    with its peak note, cadence note, and repeated-template index.
    """
    raw = []
    cur = []
    for s in melody:
        if cur:
            gap = s['start'] - cur[-1]['end']
            long_end = (cur[-1]['end'] - cur[-1]['start']) >= cadence_dur
            if gap >= gap_thresh or long_end:
                raw.append(cur)
                cur = []
        cur.append(s)
    if cur:
        raw.append(cur)

    # merge tiny fragments into the previous phrase
    phrases = []
    for p in raw:
        if phrases and len(p) < min_merge:
            phrases[-1].extend(p)
        else:
            phrases.append(p)

    contours = []
    for i, ph in enumerate(phrases):
        pid = i

        # phrase_peak: NOT simply the highest midi. Pick the note that maximizes
        # (duration * 0.5 + pitch_height_normalized * 0.5) among the LAST 60% of
        # the phrase, so a lower tonal-resolution/arrival note near the end can
        # be the peak.
        midis = [s.get('midi', 60) for s in ph]
        lo = min(midis)
        hi = max(midis)
        start_idx = max(0, int(len(ph) * 0.4))
        peak_i = start_idx
        best_score = -1.0
        for k in range(start_idx, len(ph)):
            s = ph[k]
            dur = s['end'] - s['start']
            phn = (s.get('midi', 60) - lo) / (hi - lo) if hi > lo else 0.5
            score = dur * 0.5 + phn * 0.5
            if score > best_score:
                best_score = score
                peak_i = k

        # phrase_rep: approximate repeated-template detection (same number of
        # notes AND >= 80% pitch-classes equal), so near-identical phrase
        # families (1st/3rd, 2nd/4th) are recognized, not only exact matches.
        rep = -1
        pcs = [x['midi'] % 12 for x in ph]
        for j, pcj in enumerate(contours):
            if len(pcj) == len(pcs) and len(pcs) > 0:
                same = sum(1 for a, b in zip(pcs, pcj) if a == b)
                if same / len(pcs) >= 0.8:
                    rep = j
                    break
        contours.append(pcs)

        for k, s in enumerate(ph):
            s['phrase_id'] = pid
            s['phrase_peak'] = (k == peak_i)
            s['phrase_cadence'] = (k == len(ph) - 1)
            s['phrase_rep'] = rep
            s['phrase_pos'] = round(k / max(1, len(ph) - 1), 3)
    return phrases


def main():
    parser = argparse.ArgumentParser(description="Extract vocal melody (Demucs + CREPE).")
    parser.add_argument("audio_filepath")
    parser.add_argument("root_note", help="Root/Sa note, e.g. F# or F#4")
    parser.add_argument("--thaat", default="Bilawal", help="Thaat for scale-first quantization (default Bilawal)")
    parser.add_argument("--cache-dir", help="Optional stems cache directory")
    parser.add_argument("--pitch-classes-only", action="store_true",
                        help="Output only the vocal pitch-class duration histogram and exit")
    args = parser.parse_args()

    if args.pitch_classes_only:
        try:
            hist = pitch_class_duration(args.audio_filepath)
        except Exception as e:
            print(json.dumps({"error": f"Pitch-class analysis failed: {str(e)}"}))
            sys.exit(1)
        print(json.dumps({"pitch_class_duration": hist}))
        return

    global CACHE_DIR
    if args.cache_dir:
        CACHE_DIR = args.cache_dir

    try:
        root_pc = normalize_root(args.root_note)
    except ValueError as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

    try:
        vocals_path = separate_vocals(args.audio_filepath)
        onsets, tempo, beats = detect_onsets_and_tempo(vocals_path)
        times, midi, periodicity, rms, sr, onset_env, bright = pitch_track_torchcrepe(vocals_path)
    except Exception as e:
        print(json.dumps({"error": f"Vocal extraction failed: {str(e)}"}))
        sys.exit(1)

    intervals = THAAT_INTERVALS.get(args.thaat) or [0, 2, 4, 5, 7, 9, 11]
    notes = segment_notes(times, midi, periodicity, rms, sr, onsets, onset_env,
                          brightness=bright, root_pc=root_pc, intervals=intervals)

    # normalize velocity to 0.35..1.0
    if notes:
        vels = [n['velocity'] for n in notes]
        vmin, vmax = min(vels), max(vels)
        if vmax > vmin:
            for n in notes:
                n['velocity'] = round(0.35 + 0.65 * (n['velocity'] - vmin) / (vmax - vmin), 3)
        else:
            for n in notes:
                n['velocity'] = 0.6

    melody = []
    for n in notes:
        if not (48 <= n['midi'] <= 83):
            continue
        # actual played pitch (midi / note) reflects the RAW chromatic pitch
        midi_raw = n['midi']
        interval = (midi_raw - root_pc) % 12
        note_name = scale_note_name(PC_TO_NOTE[root_pc], intervals, interval)
        octave = midi_raw // 12 - 1
        # sargam is a SOFT label derived from the scale-quantized scale_midi
        scale_midi = n.get('scale_midi', midi_raw)
        scale_interval = (scale_midi - root_pc) % 12
        sargam = INTERVAL_TO_SARGAM[scale_interval]
        entry = {
            'start': n['start'],
            'end': n['end'],
            'note': f'{note_name}{octave}',
            'pitch_class': note_name,
            'octave': octave,
            'midi': midi_raw,
            'sargam': sargam,
            'velocity': n['velocity'],
            'scale_midi': scale_midi,
        }
        for f in ('ornament', 'glide_to', 'trill', 'sustain', 'kan',
                  'raw_midi_float', 'cents_deviation', 'pitch_confidence',
                  'voicing_confidence', 'attack_energy', 'brightness'):
            if f in n:
                entry[f] = n[f]
        melody.append(entry)

    if not melody:
        print(json.dumps({"error": "No vocal notes detected."}))
        sys.exit(1)

    segment_phrases(melody)

    duration = round(max(n['end'] for n in melody), 2)
    sargam_counts = {}
    for seg in melody:
        sargam_counts[seg['sargam']] = sargam_counts.get(seg['sargam'], 0) + 1

    output = {
        "root": PC_TO_NOTE[root_pc],
        "thaat": args.thaat,
        "duration": duration,
        "tempo": tempo,
        "beats": beats,
        "melody": melody,
        "sargam_counts": sargam_counts,
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
