#!/usr/bin/env python3
from pathlib import Path

from agentic.contracts import AgentTrace, EvidenceRef, HypothesisVersion, NoteCandidate, ToolCall
from agentic.skill_registry import SkillRegistry
from agentic.tool_registry import default_registry
from agentic.runtime_tools import build_runtime_registry


ref = EvidenceRef('inspect_pitch_window', 'pitch-1', 'midi_crepe', 1.0, 1.5, 0.91, 'stable pitch')
event = NoteCandidate('n1', 1.0, 1.5, 61, raw_midi_float=60.8, pitch_confidence=0.91, voicing_confidence=0.95, evidence_refs=[ref])
hypothesis = HypothesisVersion('h1', 0, None, 'candidate', [event], ['check boundary'], [ToolCall('c1', 'inspect_pitch_window', {'audio_path':'x','start':1.0,'end':1.5}, 'verify pitch')])
trace = AgentTrace('trace-1', 'audio-sha', hypotheses=[hypothesis], final_hypothesis_id='h1')
trace.validate()

skills = SkillRegistry(Path(__file__).resolve().parent / 'skills')
assert len(skills) >= 8
retrieved = skills.retrieve('short repeated note onset pitch boundary', limit=3)
assert retrieved
assert any('Music_Flow_Engine' in item.skill_id or 'Pitch_Skill' in item.skill_id for item in retrieved)

registry = default_registry()
registry.validate_call('inspect_pitch_window', {'audio_path':'x','start':0.0,'end':1.0})
try:
    registry.validate_call('inspect_pitch_window', {'audio_path':'x','start':0.0})
except ValueError:
    pass
else:
    raise AssertionError('missing required tool argument was not rejected')
try:
    registry.validate_call('inspect_pitch_window', {'audio_path':'x','start':0.0,'end':1.0,'bad':True})
except ValueError:
    pass
else:
    raise AssertionError('unknown tool argument was not rejected')
runtime = build_runtime_registry()
manifest = runtime.call('get_track_manifest', audio_path='/home/ubuntu/upload/tum-se-hi-jab-we-met-128-kbps-t6kdsjrq_xbStSoS0.mp3')
assert manifest['duration'] > 29
print('agentic contracts passed')
