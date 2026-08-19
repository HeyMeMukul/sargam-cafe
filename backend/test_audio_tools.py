#!/usr/bin/env python3
import hashlib
import json
import os
import shutil
from pathlib import Path

AUDIO = '/home/ubuntu/upload/tum-se-hi-jab-we-met-128-kbps-t6kdsjrq_xbStSoS0.mp3'
CACHE = Path('/home/ubuntu/work/agentic_evidence_cache')
os.environ['SARGAM_AGENTIC_EVIDENCE_DIR'] = str(CACHE)

from agentic.audio_tools import (
    _cache_dir,
    audition_phrase,
    get_note_candidates,
    get_track_manifest,
    inspect_boundary_window,
    inspect_pitch_window,
    score_hypothesis,
)

cache = _cache_dir(AUDIO)
shutil.copy('/home/ubuntu/work/clean_production_evidence.json', cache / 'production.evidence.json')
shutil.copy('/home/ubuntu/work/bridge_on_output.json', cache / 'production.melody.json')
manifest = get_track_manifest(AUDIO)
assert manifest['duration'] > 29
pitch = inspect_pitch_window(AUDIO, 2.0, 3.0)
assert pitch['summary']['frame_count'] > 0
boundary = inspect_boundary_window(AUDIO, 2.0, 3.0)
assert boundary['candidate_onsets']
candidates = get_note_candidates(AUDIO, 2.0, 9.0)
assert candidates['events']
audition = audition_phrase(AUDIO, candidates['events'], 2.0, 9.0)
assert audition['audio_rendered'] is False
score = score_hypothesis(AUDIO, candidates['events'])
assert score['event_count'] == len(candidates['events'])
print('audio tools passed:', json.dumps({
    'manifest_duration': manifest['duration'],
    'pitch_frames': pitch['summary']['frame_count'],
    'boundary_onsets': len(boundary['candidate_onsets']),
    'candidate_events': len(candidates['events']),
    'audition_artifact': audition['artifact_path'],
}))
