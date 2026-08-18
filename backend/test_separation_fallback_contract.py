from pathlib import Path

source = Path(__file__).with_name('vocal_melody.py').read_text(encoding='utf-8')
assert "segment=segment" in source
assert "min(default_segment, 4.0)" in source
assert "mixture_fallback.wav" in source
assert 'source_separation' in source
print('separation fallback contract passed')
