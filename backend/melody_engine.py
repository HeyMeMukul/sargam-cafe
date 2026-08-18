#!/usr/bin/env python3
"""Note-perfect melody extraction using Spotify's Basic-Pitch (ONNX backend).

Basic-Pitch transcribes the full track into polyphonic note events
(start, end, MIDI pitch, amplitude). We then extract a *smooth monophonic
melody line*:

  1. Filter notes to a vocal register band (C3..B5) above a minimum amplitude.
  2. Frame-level tracking with OCTAVE-CONTINUITY: at each frame we pick the
     active note that best continues the running melody (nearest pitch within
     a fifth), falling back to the loudest note on a new phrase onset. This
     avoids the harmonic octave-jump problem that plagues "highest note"
     extraction and yields a singable, phrase-like line.
  3. Tempo (BPM) + beat times are detected with librosa so the agent knows the
     song's speed and rhythmic grid.

Output shape matches what the frontend already consumes:
    {
      "root": "D#",
      "duration": 320.44,
      "tempo": 96.0,
      "beats": [0.0, 0.62, 1.25, ...],
      "melody": [
        {"start": 3.6, "end": 4.5, "note": "G#4", "pitch_class": "G#",
         "octave": 4, "midi": 68, "sargam": "Ma"},
        ...
      ],
      "sargam_counts": { "Sa": 40, "Re": 12, ... }
    }

Usage:
    python melody_engine.py <audio_filepath> <root_note> [--min-amp 0.3]
"""
import argparse
import json
import sys
import warnings

warnings.filterwarnings('ignore')

import numpy as np

NOTE_TO_PC = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
    'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
}
PC_TO_NOTE = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

INTERVAL_TO_SARGAM = {
    0: 'Sa', 1: 're', 2: 'Re', 3: 'ga', 4: 'Ga', 5: 'Ma',
    6: 'ma', 7: 'Pa', 8: 'dha', 9: 'Dha', 10: 'ni', 11: 'Ni',
}

# Melody register: notes in this MIDI band are candidate melody notes.
MELODY_LO, MELODY_HI = 48, 83  # C3 .. B5
# Max pitch step (semitones) for continuity before we consider it a new phrase.
CONTINUITY_STEP = 7  # a perfect fifth


def normalize_root(root: str) -> int:
    name = root.strip()
    while name and name[-1].isdigit():
        name = name[:-1]
    if name not in NOTE_TO_PC:
        raise ValueError(f"Unknown root note: {root}")
    return NOTE_TO_PC[name]


def midi_to_name(midi: int):
    pc = midi % 12
    octave = midi // 12 - 1
    return PC_TO_NOTE[pc], octave


def detect_tempo(audio_filepath: str):
    """Return (bpm, beat_times) using librosa beat tracking, or (None, [])."""
    import librosa
    try:
        y, sr = librosa.load(audio_filepath, sr=22050, mono=True)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(np.atleast_1d(tempo)[0])
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        return round(bpm, 1), [round(float(t), 2) for t in beat_times]
    except Exception:
        return None, []


def extract_melody(audio_filepath: str, root_pc: int, min_amp: float,
                   smooth: float, min_dur: float):
    import contextlib
    import io
    from basic_pitch.inference import predict

    # Basic-Pitch prints a "Predicting MIDI ..." progress line to stdout,
    # which would corrupt our JSON output. Silence it.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model_output, midi_data, note_events = predict(audio_filepath)

    # (start, end, midi, amplitude, bends) -> filter weak/out-of-register notes
    notes = []
    for ev in note_events:
        start = float(ev[0])
        end = float(ev[1])
        midi = int(ev[2])
        amp = float(ev[3])
        if amp < min_amp or end - start < 0.08:
            continue
        if not (MELODY_LO <= midi <= MELODY_HI):
            continue
        notes.append((start, end, midi, amp))

    if not notes:
        return None

    notes.sort(key=lambda n: n[0])
    duration = max(n[1] for n in notes)

    # Build a frame-level melody with octave-continuity tracking.
    step = 0.1
    n_frames = int(duration / step) + 1

    # Precompute, for each frame, the set of active (midi, amp) sorted by amp.
    # Use an event sweep: maintain a sorted structure keyed by frame.
    frame_best_midi = np.full(n_frames, -1.0)
    frame_best_amp = np.full(n_frames, -1.0)

    # For continuity we need the running melody pitch; process frames in order.
    running = None  # current melody midi
    running_amp = -1.0
    running_end = -1.0

    # We'll build an interval list per note and sweep.
    # Simpler: for each frame, gather active notes via a pointer sweep.
    active = []  # list of (end, midi, amp)
    note_ptr = 0

    frame_midi = np.full(n_frames, -1.0)

    for i in range(n_frames):
        t = i * step
        # add notes that start at/before t
        while note_ptr < len(notes) and notes[note_ptr][0] <= t:
            s, e, m, a = notes[note_ptr]
            active.append([e, m, a])
            note_ptr += 1
        # remove expired
        active = [n for n in active if n[0] > t]

        if not active:
            running = None
            running_amp = -1.0
            frame_midi[i] = -1.0
            continue

        # pick the melody note at this frame
        if running is None:
            # phrase onset: loudest note in register
            best = max(active, key=lambda n: n[2])
            running = best[1]
            running_amp = best[2]
        else:
            # continuity: prefer notes within CONTINUITY_STEP of running pitch
            near = [n for n in active if abs(n[1] - running) <= CONTINUITY_STEP]
            if near:
                best = max(near, key=lambda n: n[2])
            else:
                # no near note -> new phrase (loudest active)
                best = max(active, key=lambda n: n[2])
            running = best[1]
            running_amp = best[2]

        frame_midi[i] = running

    # light median smoothing (removes single-frame flicker without smearing
    # phrase boundaries)
    win = max(1, int(round(smooth / step)))
    smoothed = np.array(frame_midi, dtype=np.float64)
    for i in range(n_frames):
        lo = max(0, i - win // 2)
        hi = min(n_frames, i + win // 2 + 1)
        vals = frame_midi[lo:hi]
        vals = vals[vals >= 0]
        smoothed[i] = np.median(vals) if len(vals) else -1.0

    # merge consecutive frames with same midi into segments
    raw_segments = []
    cur = None
    for i in range(n_frames):
        m = smoothed[i]
        if m < 0:
            if cur is not None:
                cur['end'] = round(i * step, 2)
                raw_segments.append(cur)
                cur = None
            continue
        m = int(round(m))
        if cur is None or cur['midi'] != m:
            if cur is not None:
                cur['end'] = round(i * step, 2)
                raw_segments.append(cur)
            cur = {'start': round(i * step, 2), 'midi': m}
    if cur is not None:
        cur['end'] = round(n_frames * step, 2)
        raw_segments.append(cur)

    # filter segments by min duration
    out = []
    for seg in raw_segments:
        if seg['end'] - seg['start'] < min_dur:
            continue
        note_name, octave = midi_to_name(seg['midi'])
        interval = (seg['midi'] - root_pc) % 12
        out.append({
            'start': round(seg['start'], 2),
            'end': round(seg['end'], 2),
            'note': f'{note_name}{octave}',
            'pitch_class': note_name,
            'octave': octave,
            'midi': seg['midi'],
            'sargam': INTERVAL_TO_SARGAM[interval],
        })

    return out, duration


def main():
    parser = argparse.ArgumentParser(description="Extract note-perfect Sargam melody with Basic-Pitch.")
    parser.add_argument("audio_filepath")
    parser.add_argument("root_note", help="Root/Sa note, e.g. D# or D#4")
    parser.add_argument("--min-amp", type=float, default=0.3)
    parser.add_argument("--smooth", type=float, default=0.2)
    parser.add_argument("--min-dur", type=float, default=0.12)
    args = parser.parse_args()

    try:
        root_pc = normalize_root(args.root_note)
    except ValueError as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

    result = extract_melody(args.audio_filepath, root_pc, args.min_amp,
                            args.smooth, args.min_dur)
    if result is None:
        print(json.dumps({"error": "No notes detected."}))
        sys.exit(1)

    segments, duration = result
    tempo, beats = detect_tempo(args.audio_filepath)

    sargam_counts = {}
    for seg in segments:
        sargam_counts[seg['sargam']] = sargam_counts.get(seg['sargam'], 0) + 1

    output = {
        "root": PC_TO_NOTE[root_pc],
        "duration": round(duration, 2),
        "tempo": tempo,
        "beats": beats,
        "melody": segments,
        "sargam_counts": sargam_counts,
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
