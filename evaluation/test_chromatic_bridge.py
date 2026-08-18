#!/usr/bin/env python3
import importlib.util
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / 'backend' / 'vocal_melody.py'
spec = importlib.util.spec_from_file_location('vocal_melody_bridge_test', SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

scale = [0, 2, 4, 5, 7, 9, 11]

def event(midi, start, end, raw=None):
    return {'midi': midi, 'start': start, 'end': end,
            'raw_midi_float': float(midi if raw is None else raw)}

# C# -> D -> D# in F# major: D is a measured short chromatic bridge.
source = [event(61, 0.0, 0.5), event(62, 0.5, 0.7, 61.6), event(63, 0.7, 1.2)]
clean = mod._collapse_chromatic_bridges(source, 6, scale)
assert [x['midi'] for x in clean] == [61, 63]
assert clean[0]['end'] == 0.7
assert clean[0]['ornament'] == 'meend'
assert clean[0]['glide_to'] == 63

# A genuine chromatic event with a large leap must remain untouched.
source = [event(61, 0.0, 0.5), event(66, 0.5, 0.7, 65.8), event(63, 0.7, 1.2)]
clean = mod._collapse_chromatic_bridges(source, 6, scale)
assert [x['midi'] for x in clean] == [61, 66, 63]

# A long chromatic note must remain a separate event.
source = [event(61, 0.0, 0.5), event(62, 0.5, 0.9, 61.8), event(63, 0.9, 1.2)]
clean = mod._collapse_chromatic_bridges(source, 6, scale)
assert [x['midi'] for x in clean] == [61, 62, 63]

print('chromatic bridge contract passed')
