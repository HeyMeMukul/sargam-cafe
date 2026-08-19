# Sargam Pianist Agent

The agentic layer is an **evidence-to-hypothesis controller**, not an LLM-only audio decoder. Specialist audio tools provide pitch, voicing, onset, boundary, candidate-note, and audition artifacts. The OpenCode `sargam-pianist` controller retrieves project skills, requests targeted observations, and returns reversible micro-operations over the baseline score.

## Runtime modes

The browser backend keeps the deterministic CREPE/ROSVOT path as the default. Set `SARGAM_PIANIST_AGENT=shadow` to run the pianist controller after extraction, expose a compact proposal summary in the transcription payload, and preserve the baseline score for playback. Set `SARGAM_PIANIST_AGENT=on` only for an explicit experiment; the controller may replace the baseline only when it returns an accepted operation set that passes typed validation. Any parse, evidence, timing, or operation failure falls back to the baseline.

```bash
export SARGAM_PIANIST_AGENT=shadow
export SARGAM_AGENTIC_MODEL=opencode/deepseek-v4-flash-free
export SARGAM_AGENTIC_MAX_TOOL_CALLS=8
# Optional hard bounds for the OpenCode subprocess
export SARGAM_AGENTIC_MAX_REQUESTS=20
export SARGAM_AGENTIC_TIMEOUT_SECONDS=180
```

The runtime telemetry reports `pianist_agent=off|shadow|on`. The payload contains an `agentic` object with mode, state, promotion status, mutation types, and unresolved questions. Shadow mode is the recommended first browser test.

## Tool bridge

The OpenCode agent calls tools through `python3 -m backend.agentic.tool_cli`. The available operations include skill retrieval, track manifest, candidate notes, pitch-window inspection, boundary-window inspection, hypothesis scoring, and phrase audition. Each artifact has an ID and the controller must cite non-baseline artifact IDs for every mutation.

## Promotion contract

The controller may return only `keep`, `insert_event`, `split_event`, `merge_events`, `retarget_pitch`, or `shift_boundary`. A mutation requires targeted evidence and a reason. Scale membership, genre familiarity, beat-grid proximity, and “recognizability” are not sufficient evidence. The operation engine applies changes atomically and maintains event IDs, raw evidence references, and MIDI-to-note rendering fields.

A note event has two distinct articulation meanings: an unchanged event is one continuous sustain with its measured `start` and `end`; an evidence-backed `split_event` creates a second segment marked `retrigger=true` and `articulation="retrigger"`, which causes a fresh piano attack. The agent must support a retrigger with both targeted onset/energy evidence and targeted voiced stable-pitch evidence. Internal spectral peaks alone do not authorize repetition. The second segment marker also prevents the Song-like renderer from collapsing a validated same-pitch re-attack back into one long hold.

The agentic layer currently writes an episodic JSONL trace under `SARGAM_AGENTIC_MEMORY_DIR` or `/tmp/sargam_agentic_memory`. These traces are memory, not model training. Fine-tuning or reinforcement learning requires a separate annotated dataset and held-out evaluation.
