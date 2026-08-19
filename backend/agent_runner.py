"""Runs the Piano Master Agent via the opencode CLI (GO subscription).

Pipeline:
  Phase A: `piano-master` agent finds Root/Thaat/Scale by hit-and-trial.
  Phase B: Basic-Pitch (note-perfect) extracts the full melody line.
  Phase C: the song is split into length-proportional sections; each section
           is validated/cleaned by a `melody-searcher` subagent (free DeepSeek
           Flash). The media player is driven (seek/loop/play) per section so
           the piano "plays along" as the subagent works.
  Phase D: all sections are merged by timestamp and the complete melody is
           sent to the UI, then performed solo on the piano.

Agent messages + piano-note + media commands are forwarded over the WebSocket.
"""
import asyncio
import json
import os
import tempfile
import re
import subprocess

try:
    from agentic.opencode_adapter import run_opencode_micro_agent
except ImportError:
    run_opencode_micro_agent = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(BACKEND_DIR, "venv", "bin", "python3")
EXTRACT_SCRIPT = os.path.join(BACKEND_DIR, "extract_melody.py")
MELODY_SCRIPT = os.path.join(BACKEND_DIR, "melody_engine.py")
VOCAL_SCRIPT = os.path.join(BACKEND_DIR, "vocal_melody.py")
ROSVOT_SCRIPT = os.path.join(BACKEND_DIR, "rosvot_adapter.py")
ROSVOT_DIR = os.getenv("SARGAM_ROSVOT_DIR", os.path.join(PROJECT_ROOT, "third_party", "ROSVOT"))
ROSVOT_CKPT_DIR = os.getenv("SARGAM_ROSVOT_CKPT_DIR", os.path.join(ROSVOT_DIR, "checkpoints"))
ROSVOT_PYTHON = os.getenv("SARGAM_ROSVOT_PYTHON", "").strip()
ROSVOT_READY = (
    os.path.isdir(ROSVOT_DIR)
    and os.path.isfile(os.path.join(ROSVOT_CKPT_DIR, "rosvot", "model.pt"))
    and os.path.isfile(os.path.join(ROSVOT_CKPT_DIR, "rwbd", "model.pt"))
)
TRANSCRIBER_MODE = os.getenv("SARGAM_TRANSCRIBER", "auto").strip().lower()
KEY_AGENT_MODE = os.getenv("SARGAM_KEY_AGENT", "auto").strip().lower()
KEY_AGENT_MIN_CONF = float(os.getenv("SARGAM_KEY_AGENT_MIN_CONF", "0.65"))
CHORD_SCRIPT = os.path.join(BACKEND_DIR, "chord_detection.py")
KEY_SCRIPT = os.path.join(BACKEND_DIR, "key_detection.py")
TEST_SCRIPT = os.path.join(BACKEND_DIR, "test_notes.py")
THEORY_JSON = os.path.join(BACKEND_DIR, "skills", "Music_Theory_Engine.json")

AGENT_NAME = "piano-master"
MODEL = "opencode/deepseek-v4-flash-free"
SECTION_AGENT = "melody-searcher"
SECTION_MODEL = "opencode/deepseek-v4-flash-free"

# Section transcription settings (length-proportional)
# Free DeepSeek Flash subagents cost nothing, so we go WIDE: many small
# sections in parallel. This gives finer-grained validation and keeps each
# subagent's prompt small (which also makes it faster and less error-prone).
TARGET_SECTION_LEN = 6.0    # aim for ~6s per subagent section
MAX_SECTIONS = 32           # upper bound on parallel sections
MAX_CONCURRENCY = 10        # subagents running at once
SECTION_REVIEW_MODE = os.getenv("SARGAM_SECTION_REVIEW", "off").strip().lower()
# Optional research side-channel; unset by default so normal runs are unchanged.
EVIDENCE_DIR = os.getenv("SARGAM_EVIDENCE_DIR", "").strip()
REFERENCE_FILE = os.getenv("SARGAM_REFERENCE_FILE", "").strip()
CHROMATIC_BRIDGE_MODE = os.getenv("SARGAM_CHROMATIC_BRIDGE", "on").strip().lower()
PIANIST_AGENT_MODE = os.getenv("SARGAM_PIANIST_AGENT", "off").strip().lower()  # off | shadow | on
PIANIST_AGENT_MODEL = os.getenv("SARGAM_AGENTIC_MODEL", "gpt-5-mini").strip()
PIANIST_AGENT_MAX_TOOL_CALLS = int(os.getenv("SARGAM_AGENTIC_MAX_TOOL_CALLS", "8"))

# Matches note names like C4, C#4, D#5, F#3, B2 inside a bash command
NOTE_RE = re.compile(r"\b([A-G](?:#|b)?[0-9])\b")

# Matches the final structured JSON block from the agent's last message
FINAL_JSON_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
SECTION_JSON_RE = re.compile(r"```json\s*(\[.*?\])\s*```", re.DOTALL)

PC_TO_NOTE = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTE_TO_PC = {n: i for i, n in enumerate(PC_TO_NOTE)}
NOTE_TO_PC.update({'Db': 1, 'Eb': 3, 'Fb': 4, 'Gb': 6, 'Ab': 8, 'Bb': 10, 'Cb': 11, 'E#': 5, 'B#': 0})

INTERVAL_TO_SARGAM = {
    0: 'Sa', 1: 're', 2: 'Re', 3: 'ga', 4: 'Ga', 5: 'Ma',
    6: 'ma', 7: 'Pa', 8: 'dha', 9: 'Dha', 10: 'ni', 11: 'Ni',
}


def normalize_root(root: str) -> int:
    name = root.strip()
    while name and name[-1].isdigit():
        name = name[:-1]
    return NOTE_TO_PC.get(name, 0)


def thaat_intervals(thaat: str):
    try:
        with open(THEORY_JSON, "r") as f:
            theory = json.load(f)
        return theory["thaats"].get(thaat, {}).get("intervals_semitones")
    except Exception:
        return None


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

def scale_notes_for(root: str, thaat: str):
    """Return the 7 scale note names (octave 4) for a root+thaat,
    using enharmonic-aware names (F# major 7th shows as E#, not F)."""
    intervals = thaat_intervals(thaat)
    if not intervals:
        intervals = [0, 2, 4, 5, 7, 9, 11]
    root_pc = normalize_root(root)
    return [f"{scale_note_name(root, intervals, iv)}4" for iv in intervals]


def sargam_for_note(note_name: str, root_pc: int):
    name = note_name.strip()
    while name and name[-1].isdigit():
        name = name[:-1]
    if name not in NOTE_TO_PC:
        return None
    return INTERVAL_TO_SARGAM[(NOTE_TO_PC[name] - root_pc) % 12]


def midi_to_note(midi: int):
    return f"{PC_TO_NOTE[midi % 12]}{midi // 12 - 1}"


def _reference_tokens(labels, root_pc: int):
    deg = {'S': 0, 'R': 2, 'G': 4, 'M': 5, 'P': 7, 'D': 9, 'N': 11,
           'r': 1, 'g': 3, 'm': 6, 'd': 8, 'n': 10}
    out = []
    for item in labels:
        if isinstance(item, str):
            label, pitch_class, octave_hint = item, None, None
        else:
            label = str(item.get('label') or item.get('sargam') or 'S')
            pitch_class = item.get('pitch_class')
            octave_hint = item.get('octave_hint')
        if pitch_class is None:
            symbol = label[0]
            if symbol not in deg:
                raise ValueError(f'Unknown sargam token: {label}')
            pitch_class = (root_pc + deg[symbol] + (12 if "'" in label else 0)) % 12
        out.append({'label': label, 'pitch_class': int(pitch_class), 'octave_hint': octave_hint})
    return out


def reference_tokens_from_file(path: str, root_pc: int):
    """Load explicit sargam reference tokens without mutating audio-only mode."""
    with open(path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return _reference_tokens(data, root_pc)
    if data.get('tokens'):
        return _reference_tokens(data['tokens'], root_pc)
    phrases = data.get('sargam_phrases') or []
    return _reference_tokens(' '.join(phrases).split(), root_pc)


def reference_groups_from_file(path: str, root_pc: int):
    with open(path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    groups = data.get('phrase_groups') if isinstance(data, dict) else None
    if not groups:
        return [(reference_tokens_from_file(path, root_pc), None, None)]
    return [(_reference_tokens(group.get('tokens') or str(group.get('sargam', '')).split(), root_pc),
             float(group['start']) if group.get('start') is not None else None,
             float(group['end']) if group.get('end') is not None else None)
            for group in groups]


def apply_reference_alignment(melody, evidence_path: str, reference_path: str, root_pc: int):
    """Replace only the optional guided score with a fixed-token alignment."""
    from reference_score_alignment import ReferenceToken, align_reference_to_frames
    with open(evidence_path, 'r', encoding='utf-8') as fh:
        evidence = json.load(fh)
    frames = evidence.get('frames') or []
    if not frames:
        raise ValueError('Evidence sidecar has no frames')
    all_times = [x.get('t', 0.0) for x in frames]
    all_midi = [x.get('midi_crepe') for x in frames]
    all_voicing = [x.get('periodicity', 0.0) for x in frames]
    all_onset = [x.get('onset_strength', 0.0) for x in frames]
    all_rms = [x.get('rms', 0.0) for x in frames]
    aligned = []
    reference_offset = 0
    for tokens, window_start, window_end in reference_groups_from_file(reference_path, root_pc):
        ref = [ReferenceToken(x['pitch_class'], x['label'], x.get('octave_hint')) for x in tokens]
        indices = list(range(len(frames)))
        if window_start is not None:
            indices = [i for i in indices if all_times[i] >= window_start]
        if window_end is not None:
            indices = [i for i in indices if all_times[i] < window_end]
        if len(indices) < max(2, len(ref)):
            raise ValueError(f'Reference window has too few frames for {len(ref)} tokens')
        local = align_reference_to_frames(
            [all_times[i] for i in indices], [all_midi[i] for i in indices],
            [all_voicing[i] for i in indices], [all_onset[i] for i in indices],
            [all_rms[i] for i in indices], ref
        )
        if len(local) != len(ref):
            raise ValueError(f'Reference alignment emitted {len(local)} of {len(ref)} tokens')
        for item in local:
            item['reference_index'] += reference_offset
        aligned.extend(local)
        reference_offset += len(ref)
    out = []
    for item in aligned:
        m = int(item['midi'])
        iv = (m - root_pc) % 12
        nm = scale_note_name(PC_TO_NOTE[root_pc], [0, 2, 4, 5, 7, 9, 11], iv)
        out.append({
            'start': item['start'], 'end': item['end'], 'note': f'{nm}{m // 12 - 1}',
            'pitch_class': nm, 'octave': m // 12 - 1, 'midi': m,
            'sargam': item['label'], 'velocity': 0.65,
            'pitch_confidence': item['alignment_confidence'],
            'voicing_confidence': item['alignment_confidence'],
            'source_model': 'crepe+reference_score_alignment',
            'reference_index': item['reference_index'],
            'reference_pitch_class': item['reference_pitch_class'],
            'retrigger': item.get('retrigger', False),
            'articulation': 'retrigger' if item.get('retrigger') else 'normal',
            'reference_alignment': True,
        })
    guided = dict(melody)
    guided['melody'] = out
    guided['transcriber'] = 'crepe+reference_score_alignment'
    guided['reference_alignment'] = True
    guided['reference_file'] = os.path.basename(reference_path)
    return guided


async def run_script(script: str, *args):
    """Run a venv python script, return its parsed JSON stdout (or None)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            VENV_PYTHON, script, *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        text = stdout.decode(errors="replace").strip()
        # Some tools print progress lines before the JSON; parse the last
        # balanced JSON object in the stream.
        return parse_last_json(text)
    except Exception:
        return None


def parse_last_json(text: str):
    """Parse the last complete JSON value in a stream that may contain noise."""
    if not text:
        return None
    # try whole text first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # fall back to scanning for the last balanced {...}
    start = text.rfind('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


async def play_final_scale(root: str, thaat: str, log_callback):
    """Play the discovered scale's notes on the piano (Sa Re Ga Ma Pa Dha Ni Sa)."""
    intervals = thaat_intervals(thaat)
    if not intervals:
        await log_callback("[System] Final performance: playing the root note.")
        await log_callback(f"__PLAY_NOTE__:{root}4")
        await asyncio.sleep(0.6)
        return

    root_pc = normalize_root(root)
    await log_callback(f"[System] Final performance: {thaat} scale (Sa = {root})")
    for iv in intervals:
        pc = (root_pc + iv) % 12
        await log_callback(f"__PLAY_NOTE__:{PC_TO_NOTE[pc]}4")
        await asyncio.sleep(0.45)
    await log_callback(f"__PLAY_NOTE__:{root}5")
    await asyncio.sleep(0.7)
    for iv in reversed(intervals):
        pc = (root_pc + iv) % 12
        await log_callback(f"__PLAY_NOTE__:{PC_TO_NOTE[pc]}4")
        await asyncio.sleep(0.45)
    await log_callback("[System] Final performance complete.")


def parse_final_json(agent_text: str):
    """Extract {root, thaat, western_scale} from the agent's final output."""
    match = FINAL_JSON_RE.search(agent_text)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    root = (data.get("root") or "").strip()
    if not root:
        return None
    return {
        "root": root,
        "thaat": (data.get("thaat") or "Unknown").strip(),
        "western_scale": (data.get("western_scale") or "").strip(),
    }


def parse_section_json(agent_text: str):
    """Extract the JSON array of melody segments from a section subagent."""
    match = SECTION_JSON_RE.search(agent_text)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    return data


class CostTracker:
    """Accumulates tokens + cost across every opencode run in a session."""

    def __init__(self):
        self.cost = 0.0
        self.tokens = {"input": 0, "output": 0, "reasoning": 0,
                       "cache_read": 0, "cache_write": 0, "total": 0}
        self.requests = 0

    def add(self, tokens, cost):
        if isinstance(tokens, dict):
            self.tokens["input"] += int(tokens.get("input") or 0)
            self.tokens["output"] += int(tokens.get("output") or 0)
            self.tokens["reasoning"] += int(tokens.get("reasoning") or 0)
            cache = tokens.get("cache") or {}
            self.tokens["cache_read"] += int(cache.get("read") or 0)
            self.tokens["cache_write"] += int(cache.get("write") or 0)
            self.tokens["total"] += int(tokens.get("total") or 0)
        if isinstance(cost, (int, float)):
            self.cost += float(cost)
        self.requests += 1

    def snapshot(self):
        return {
            "cost": round(self.cost, 6),
            "tokens": dict(self.tokens),
            "requests": self.requests,
        }


async def run_agent_stream(cmd, log_callback, collect_text=True, cost_tracker=None):
    """Run an opencode CLI subprocess, forwarding events. Returns collected text parts."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=PROJECT_ROOT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        await log_callback("[Fatal Error] opencode CLI not found in PATH.")
        return None

    text_parts = []

    async def read_stream(stream, is_stdout: bool):
        while True:
            line = await stream.readline()
            if not line:
                break
            text = line.decode(errors="replace").strip()
            if not text:
                continue
            if is_stdout:
                await handle_event(text, log_callback, text_parts, cost_tracker)
            else:
                await log_callback(f"[System] {text}")

    await asyncio.gather(
        read_stream(proc.stdout, True),
        read_stream(proc.stderr, False),
    )
    await proc.wait()
    return "".join(text_parts)


async def run_section_subagent(audio_filepath: str, start: float, end: float,
                               root: str, thaat: str, scale_notes,
                               slice_segments, log_callback, semaphore,
                               section_idx: int, total_sections: int,
                               cost_tracker=None, tempo=None, beats=None):
    """Validate/clean one section's Basic-Pitch melody with a free-Flash subagent."""
    async with semaphore:
        await log_callback(
            f"[System] Section {section_idx}/{total_sections}: "
            f"validating {start:.0f}s–{end:.0f}s..."
        )

        # Section reviewers read the uploaded file directly. Do not issue seek/
        # loop/play commands here: gather() runs reviewers concurrently, so those
        # media commands race and leave the browser audio in an arbitrary section.
        # The UI receives one authoritative pause before the final payload instead.

        # Beat times within this section (for rhythm validation)
        section_beats = []
        if beats:
            section_beats = [round(b, 2) for b in beats if start <= b <= end]

        slice_json = json.dumps(slice_segments)
        prompt = (
            f"You are an EVIDENCE REVIEWER for one section of an audio-derived "
            f"melody transcription: {audio_filepath}\n"
            f"Section window: {start:.1f}s to {end:.1f}s.\n"
            f"Root (Sa) label: {root} | Thaat label: {thaat}\n"
            f"Scale notes (octave 4, labeling only): {' '.join(scale_notes)}\n"
            f"Tempo (BPM): {tempo if tempo else 'unknown'}\n"
            f"Beat times in this section (candidate grid, not ground truth): "
            f"{section_beats if section_beats else 'unknown'}\n\n"
            f"Below is the machine-extracted melody for your section. The raw "
            f"audio-derived event stream is AUTHORITATIVE by default. Apply the "
            f"revised knowledge skills in backend/skills/ as SOFT PRIORS and data "
            f"contracts — they do NOT authorize automatic deletion or scale "
            f"correction. Read: Music_Flow_Engine.json, Pitch_Skill.json, "
            f"Beat_Skill.json, Rhythm_Skill.json, Piano_Skill.json, "
            f"Scale_Identification.json, Music_Theory_Engine.json.\n"
            f"{slice_json}\n\n"
            f"Review policy:\n"
            f"1. PRESERVE the raw event stream — keep every valid field, every note, "
            f"raw pitch, confidence, velocity, ornaments, and timing.\n"
            f"2. Do NOT delete notes by duration alone (a short event may be a grace "
            f"note, rapid syllable, ornament, or valid articulation).\n"
            f"3. Do NOT snap every onset — preserve syncopation, pickups, triplets, "
            f"swing and rubato unless audio evidence contradicts them.\n"
            f"4. Do NOT force a scale — passing/borrowed/chromatic tones may be valid. "
            f"Keep measured pitch; flag scale disagreement instead of replacing it.\n"
            f"5. Correct an octave ONLY with evidence (short, low-confidence excursion "
            f"returning to the surrounding register). A sustained/high-confidence "
            f"register change must be preserved.\n"
            f"6. Do NOT infer ornaments from length/genre — preserve extractor "
            f"ornament fields.\n"
            f"7. Do NOT alter velocity without evidence.\n"
            f"8. Do NOT use stepwise contour as a deletion rule.\n\n"
            f"Evidence hierarchy (highest first): high-confidence audio pitch/onset/"
            f"offset/voicing > pitch-curve continuity > local onset/energy > beat/"
            f"harmonic context > global key/thaat > stylistic expectation.\n\n"
            f"Allowed corrections: ONLY clear localized artifacts, recorded via "
            f"review_flags and review_reason (e.g. one-frame low-confidence octave "
            f"flip, invalid negative duration, duplicated section-boundary event). "
            f"If ambiguous, retain the event and flag it (possible_octave_error, "
            f"possible_rearticulation, possible_chromatic_tone, timing_uncertain). "
            f"Do not silently mutate.\n\n"
            f"Output ONLY a JSON array (wrapped in ```json) of the same events in the "
            f"same section, preserving order, count, timing, pitch, velocity, "
            f"confidence and ornament fields, covering {start:.1f}s–{end:.1f}s. You "
            f"may add review_flags, review_reason, review_confidence. No prose after "
            f"the JSON."
        )

        cmd = [
            "opencode", "run",
            "--agent", SECTION_AGENT,
            "--model", SECTION_MODEL,
            "--format", "json",
            "--log-level", "ERROR",
            "--auto",
            prompt,
        ]

        try:
            text = await run_agent_stream(cmd, log_callback, cost_tracker=cost_tracker)
        except Exception as exc:
            await log_callback(
                f"[System] Section {section_idx}/{total_sections}: reviewer exception; "
                f"using raw extraction ({type(exc).__name__})."
            )
            text = ""
        segments = parse_section_json(text or "")
        if segments:
            await log_callback(
                f"[System] Section {section_idx}/{total_sections}: {len(segments)} segments.")
        else:
            # fall back to the raw Basic-Pitch slice
            segments = slice_segments
            await log_callback(
                f"[System] Section {section_idx}/{total_sections}: subagent failed, "
                f"using raw extraction ({len(segments)} segments).")
        return segments


async def transcribe_audio_agentic(audio_filepath: str, log_callback):
    """Run the full transcription pipeline."""
    if not os.path.exists(audio_filepath):
        await log_callback(f"[Error] File {audio_filepath} not found on server.")
        return

    await log_callback("[System] Initializing Piano Master Agent...")
    await log_callback(
        f"[System] Runtime config: transcriber={TRANSCRIBER_MODE or 'auto'}, "
        f"key_agent={KEY_AGENT_MODE or 'auto'}, section_review={SECTION_REVIEW_MODE or 'off'}, "
        f"reference={'on' if REFERENCE_FILE else 'off'}, "
        f"decoder=chromatic_bridge_{CHROMATIC_BRIDGE_MODE or 'on'}, "
        f"pianist_agent={PIANIST_AGENT_MODE or 'off'}, "
        f"rosvot={'ready' if ROSVOT_READY else 'unavailable'}, "
        f"rosvot_python={'custom' if ROSVOT_PYTHON else 'backend'}"
    )
    cost_tracker = CostTracker()

    # --- Phase 0: vocal pitch-class duration (root-independent tonic cue) ---
    # Extracting the isolated vocal melody's pitch-class REST histogram is the
    # strongest cue for resolving relative-key ambiguity (e.g. D minor vs A
    # phrygian). This also warms the Demucs stem cache for the melody pass.
    pc_file = os.path.join(tempfile.gettempdir(), "sargamking_pcd.json")
    pcd = await run_script(VOCAL_SCRIPT, audio_filepath, "C", "--thaat", "Bilawal",
                           "--pitch-classes-only")
    if pcd and pcd.get("pitch_class_duration"):
        with open(pc_file, "w") as f:
            json.dump({"pitch_class_duration": pcd["pitch_class_duration"]}, f)
        await log_callback("[System] Vocal pitch-class emphasis computed.")
    else:
        await log_callback("[System] Pitch-class emphasis unavailable; using chroma only.")

    # --- Phase 0b: deterministic key detection (Krumhansl + melodic cue) ---
    key_cmd = [KEY_SCRIPT, audio_filepath]
    if os.path.exists(pc_file):
        key_cmd += ["--melodic-duration-json", pc_file]
    key = await run_script(*key_cmd)

    # The deterministic detector is cheap, repeatable and already analyzes the
    # whole track. For a confident result, do not launch the expensive hit-and-
    # trial LLM: it adds hundreds of thousands of tokens, emits raw chatter, and
    # can waste time on invalid paths/timestamps without improving note extraction.
    key_conf = float(key.get("confidence") or 0.0) if key else 0.0
    use_key_directly = bool(
        key and key.get("root") and
        KEY_AGENT_MODE != "always" and
        (KEY_AGENT_MODE == "never" or key_conf >= KEY_AGENT_MIN_CONF)
    )
    if key and key.get("root"):
        await log_callback(
            f"[System] Key detector: {key['root']} {key['mode']} "
            f"(thaat {key['thaat']}, conf {key.get('confidence', '?')})"
        )

    final = None
    if use_key_directly:
        final = {
            "root": key["root"],
            "thaat": key["thaat"],
            "western_scale": key["western_scale"],
        }
        await log_callback(
            f"[System] Deterministic key accepted at confidence {key_conf:.3f}; "
            "skipping hit-and-trial key agent."
        )
    else:
        key_hint = ""
        if key and key.get("root"):
            key_hint = (
                f"\n\nIMPORTANT: Use this exact absolute audio path verbatim in every "
                f"command: {audio_filepath}\n"
                f"A deterministic Krumhansl-Schmuckler key detector analyzing the "
                f"whole track determined {key['root']} {key['mode']} (thaat "
                f"{key['thaat']}, confidence {key.get('confidence', '?')}). "
                "Treat it as a strong prior and override only with decisive, "
                "multi-window contradictory evidence."
            )
        cmd = [
            "opencode", "run",
            "--agent", AGENT_NAME,
            "--model", MODEL,
            "--format", "json",
            "--log-level", "ERROR",
            "--auto",
            f"Transcribe this track using the exact path above: {audio_filepath}{key_hint}",
        ]
        agent_text = await run_agent_stream(cmd, log_callback, cost_tracker=cost_tracker)
        final = parse_final_json(agent_text or "")
        if not final and key and key.get("root"):
            await log_callback("[System] Agent produced no output; using key detector result.")
            final = {
                "root": key["root"],
                "thaat": key["thaat"],
                "western_scale": key["western_scale"],
            }
    if not final:
        await log_callback("[System] Agent finished without structured output.")
        return

    await log_callback(f"[System] Root discovered: {final['root']} | Thaat: {final['thaat']}")
    root_pc = normalize_root(final["root"])

    # --- Phase B1: chord detection (chords first) ---
    await log_callback("[System] Detecting chord progression...")
    chord_data = await run_script(CHORD_SCRIPT, audio_filepath, final["root"],
                                  f"--thaat", final["thaat"])
    chords = (chord_data or {}).get("chords") or []
    await log_callback(f"[System] Chords detected: {len(chords)}")

    # --- Phase B2: dedicated singing-note extraction with safe fallback ---
    melody = None
    if TRANSCRIBER_MODE in {"auto", "rosvot"} and not REFERENCE_FILE:
        await log_callback("[System] Trying ROSVOT note-level singing transcription...")
        melody = await run_script(ROSVOT_SCRIPT, audio_filepath, final["root"],
                                  "--thaat", final["thaat"])
        if melody and melody.get("melody"):
            await log_callback(f"[System] ROSVOT extracted {len(melody['melody'])} primary note events.")
        else:
            await log_callback("[System] ROSVOT unavailable or failed; falling back to CREPE note extraction.")
    if REFERENCE_FILE:
        await log_callback("[System] Reference mode requires frame evidence; using CREPE extraction path.")
    if not melody or not melody.get("melody"):
        await log_callback("[System] Extracting vocal melody (Demucs source separation + pitch tracking)...")
        vocal_args = [audio_filepath, final["root"], "--thaat", final["thaat"]]
        evidence_path = None
        evidence_root = EVIDENCE_DIR or (tempfile.gettempdir() if REFERENCE_FILE else "")
        if evidence_root:
            os.makedirs(evidence_root, exist_ok=True)
            evidence_path = os.path.join(
                evidence_root,
                os.path.basename(audio_filepath) + ".evidence.json",
            )
            vocal_args.extend(["--evidence-out", evidence_path])
        melody = await run_script(VOCAL_SCRIPT, *vocal_args)
        if evidence_path and os.path.exists(evidence_path):
            await log_callback(f"[System] Evidence side-channel written: {evidence_path}")
        if melody and melody.get("melody") and REFERENCE_FILE:
            try:
                melody = apply_reference_alignment(
                    melody, evidence_path, REFERENCE_FILE, root_pc
                )
                await log_callback(
                    f"[System] Reference-conditioned score selected: {len(melody['melody'])} tokens."
                )
            except Exception as exc:
                await log_callback(
                    f"[System] Reference mode unavailable ({exc}); preserving audio-only score."
                )
    if not melody or not melody.get("melody"):
        await log_callback("[System] Vocal extraction failed; falling back to Basic-Pitch...")
        melody = await run_script(MELODY_SCRIPT, audio_filepath, final["root"])
        if not melody or not melody.get("melody"):
            await log_callback("[System] Basic-Pitch failed; falling back to chroma extraction...")
            melody = await run_script(EXTRACT_SCRIPT, audio_filepath, final["root"])
            if not melody:
                await log_callback("[System] Melody extraction failed.")
                await log_callback("[System] Transcription session completed.")
                return

    full_segments = melody["melody"]
    duration = float(melody.get("duration") or 0)
    tempo = melody.get("tempo")
    beats = melody.get("beats") or []
    guided_mode = bool(melody.get("reference_alignment"))
    agentic_shadow = None
    if PIANIST_AGENT_MODE in {"shadow", "on"} and run_opencode_micro_agent is not None and not guided_mode:
        await log_callback(f"[System] Running evidence-gated pianist agent ({PIANIST_AGENT_MODE})...")
        try:
            agentic_shadow = await run_opencode_micro_agent(
                audio_filepath,
                duration,
                full_segments,
                run_agent_stream,
                log_callback,
                PIANIST_AGENT_MODEL,
                PIANIST_AGENT_MAX_TOOL_CALLS,
                cost_tracker,
            )
            if agentic_shadow.get("promoted") and PIANIST_AGENT_MODE == "on":
                full_segments = agentic_shadow.get("events") or full_segments
                melody = dict(melody)
                melody["melody"] = full_segments
                melody["transcriber"] = "agentic_pianist"
                melody["agentic_promoted"] = True
                await log_callback(
                    f"[System] Pianist agent promoted {len(full_segments)} evidence-gated events."
                )
            else:
                await log_callback(
                    f"[System] Pianist agent kept baseline (promoted={bool(agentic_shadow.get('promoted'))})."
                )
        except Exception as exc:
            await log_callback(
                f"[System] Pianist agent unavailable; preserving baseline ({type(exc).__name__})."
            )
    tempo = melody.get("tempo")
    await log_callback(f"[System] Melody extracted: {len(full_segments)} notes over {duration:.0f}s."
                       + (f" Tempo: {tempo} BPM" if tempo else ""))

    scale_notes = scale_notes_for(final["root"], final["thaat"])

    # --- Phase C: optional length-proportional LLM review ---
    # The extractor already carries pitch/onset/voicing evidence. The live
    # audit showed that five reviewers preserved the same events while consuming
    # hundreds of thousands of tokens and flooding the UI with JSON/prose.
    if full_segments and duration > 0 and SECTION_REVIEW_MODE in {"on", "always"} and not guided_mode:
        n_sections = min(MAX_SECTIONS, max(1, int(duration / TARGET_SECTION_LEN) + (1 if duration % TARGET_SECTION_LEN else 0)))
        section_len = duration / n_sections
        sections = [(i * section_len, min((i + 1) * section_len, duration))
                    for i in range(n_sections)]

        await log_callback(f"[System] Splitting into {n_sections} sections for validation subagents...")

        # Pre-slice the Basic-Pitch melody per section
        slices = [[] for _ in sections]
        for seg in full_segments:
            s, e = seg["start"], seg["end"]
            mid = (s + e) / 2
            idx = min(int(mid / section_len), n_sections - 1)
            if idx < 0:
                idx = 0
            slices[idx].append(seg)

        semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
        try:
            section_results = await asyncio.gather(*[
                run_section_subagent(audio_filepath, s, e, final["root"], final["thaat"],
                                     scale_notes, slices[i], log_callback, semaphore,
                                     i + 1, n_sections, cost_tracker, tempo, beats)
                for i, (s, e) in enumerate(sections)
            ])
        except Exception as exc:
            await log_callback(
                f"[System] Section review aborted safely ({type(exc).__name__}); "
                "preserving the raw extractor score."
            )
            section_results = None

        # --- Phase D: merge by timestamp ---
        merged = []
        if section_results is not None:
            for segs in section_results:
                merged.extend(segs)
        else:
            merged = full_segments
    else:
        reason = "reference-conditioned score" if guided_mode else "extractor evidence"
        await log_callback(
            f"[System] Skipping LLM section review (SARGAM_SECTION_REVIEW={SECTION_REVIEW_MODE or 'off'}); "
            f"preserving {reason} for deterministic rendering."
        )
        merged = full_segments

    # Stop looping the player
    await log_callback("__MEDIA__:{\"action\": \"pause\"}")

    # Normalize + sort merged segments
    cleaned = []
    for seg in merged:
        try:
            start = float(seg.get("start"))
            end = float(seg.get("end"))
            note = (seg.get("note") or "").strip()
        except (TypeError, ValueError):
            continue
        if start >= end or not note or note == "-":
            continue
        sargam = seg.get("sargam") or sargam_for_note(note, root_pc)
        if not sargam:
            sargam = "other"
        entry = {
            "start": round(start, 2),
            "end": round(end, 2),
            "note": note,
            "sargam": sargam,
        }
        # preserve velocity (dynamics) if present (from vocal_melody)
        if seg.get("velocity") is not None:
            entry["velocity"] = round(float(seg["velocity"]), 3)
        # preserve octave/midi if present (from Basic-Pitch) for piano-roll layout
        midi = seg.get("midi")
        if midi is None:
            nm = note
            while nm and nm[-1].isdigit():
                nm = nm[:-1]
            if nm in NOTE_TO_PC:
                octv = note[len(nm):]
                if octv.isdigit():
                    midi = (int(octv) + 1) * 12 + NOTE_TO_PC[nm]
        if midi is not None:
            entry["midi"] = int(midi)
            # scale-consistent enharmonic naming (E# not F in F# major)
            intervals_n = thaat_intervals(final["thaat"]) or [0, 2, 4, 5, 7, 9, 11]
            iv = (int(midi) - root_pc) % 12
            nm_s = scale_note_name(final["root"], intervals_n, iv)
            entry["note"] = f"{nm_s}{int(midi) // 12 - 1}"
            entry["pitch_class"] = nm_s
            entry["sargam"] = INTERVAL_TO_SARGAM[iv]
        octave = seg.get("octave")
        if octave is not None:
            entry["octave"] = int(octave)
        # preserve ornament/expression fields
        for f in ('ornament', 'glide_to', 'trill', 'sustain', 'kan',
                  'retrigger', 'articulation', 'grace_note', 'grace_duration'):
            if seg.get(f) is not None:
                entry[f] = seg[f]
        # preserve reviewer metadata + raw evidence (audit trail) through merge
        for f in ('review_flags', 'review_reason', 'review_confidence',
                  'source_model', 'render_role', 'model_threshold', 'timing_source',
                  'out_of_scale_candidate', 'octave_error_candidate',
                  'raw_midi_float', 'cents_deviation', 'pitch_confidence',
                  'voicing_confidence', 'onset_confidence', 'offset_confidence',
                  'reference_alignment', 'reference_index', 'reference_pitch_class',
                  'attack_energy', 'brightness'):
            if seg.get(f) is not None:
                entry[f] = seg[f]
        # preserve phrase/performance metadata
        for f in ('phrase_id', 'phrase_peak', 'phrase_cadence', 'phrase_rep',
                  'phrase_pos'):
            if seg.get(f) is not None:
                entry[f] = seg[f]
        cleaned.append(entry)

    cleaned.sort(key=lambda s: s["start"])

    # Re-attach velocity (dynamics) from the authoritative vocal extraction,
    # since subagents often drop the velocity field when returning cleaned notes.
    vel_by_time = {}
    vel_by_note = {}
    for seg in full_segments:
        if seg.get("velocity") is not None:
            vel_by_time[(seg.get("note"), round(float(seg.get("start", 0)), 1))] = seg["velocity"]
            vel_by_note.setdefault(seg.get("note"), seg["velocity"])
    for entry in cleaned:
        if entry.get("velocity") is None:
            v = vel_by_time.get((entry.get("note"), round(entry.get("start", 0), 1)))
            if v is None:
                v = vel_by_note.get(entry.get("note"))
            if v is not None:
                entry["velocity"] = round(float(v), 3)

    # Deduplicate ordinary section-boundary duplicates only. A guided
    # reference is already an ordered token stream; repeated same-pitch tokens
    # must remain explicit retriggers.
    cleaned.sort(key=lambda s: s["start"])
    deduped = list(cleaned) if guided_mode else []
    for seg in ([] if guided_mode else cleaned):
        m = seg.get("midi")
        if deduped and m is not None and deduped[-1].get("midi") == m                 and abs(seg["start"] - deduped[-1]["start"]) < 0.06:
            # keep the event with higher confidence (or longer), merge flags
            keep = seg if (seg.get("pitch_confidence") or 0) >= (deduped[-1].get("pitch_confidence") or 0) else deduped[-1]
            for f in ('review_flags', 'review_reason'):
                if seg.get(f) and f not in keep:
                    keep[f] = seg[f]
            deduped[-1] = keep
            deduped[-1]['end'] = max(deduped[-1]['end'], seg['end'])
            deduped[-1]['start'] = min(deduped[-1]['start'], seg['start'])
            continue
        deduped.append(seg)
    cleaned = deduped

    # Clamp overlaps so the timeline is monotonic
    final_cleaned = []
    for seg in cleaned:
        if final_cleaned and seg["start"] < final_cleaned[-1]["end"]:
            seg["start"] = round(final_cleaned[-1]["end"], 2)
            if seg["start"] >= seg["end"]:
                continue
        final_cleaned.append(seg)

    if not final_cleaned:
        await log_callback("[System] No melody produced.")
        await log_callback("[System] Transcription session completed.")
        return

    # --- Review pass (NON-MUTATING): flag rather than silently rewrite ---
    # Per the accuracy review (Priority 1), the raw audio-derived pitch is
    # authoritative. We never force a note into the scale or collapse an octave
    # automatically. Instead we mark candidates so the UI/user can decide.
    intervals = thaat_intervals(final["thaat"]) or [0, 2, 4, 5, 7, 9, 11]
    scale_pcs = set((root_pc + iv) % 12 for iv in intervals)
    flagged = []
    for i, seg in enumerate(final_cleaned):
        seg = dict(seg)
        midi = seg.get("midi")
        flags = list(seg.get("review_flags") or [])
        # 1) flag out-of-scale candidates (do NOT change the measured pitch)
        if midi is not None and (midi % 12) not in scale_pcs:
            if "possible_chromatic_tone" not in flags:
                flags.append("possible_chromatic_tone")
            seg["out_of_scale_candidate"] = True
        # 2) flag (do NOT collapse) clear isolated octave jumps >= 2 octaves from
        #    both neighbours, which is almost certainly a pitch-tracking error
        if i > 0 and i < len(final_cleaned) - 1:
            pm = final_cleaned[i - 1].get("midi")
            nm2 = final_cleaned[i + 1].get("midi")
            if pm is not None and nm2 is not None and midi is not None:
                if abs(pm - nm2) <= 4 and abs(midi - pm) >= 24 and abs(midi - nm2) >= 24:
                    if "possible_octave_error" not in flags:
                        flags.append("possible_octave_error")
                    seg["octave_error_candidate"] = True
        if flags:
            seg["review_flags"] = flags
        flagged.append(seg)
    final_cleaned = flagged

    sargam_counts = {}
    for seg in final_cleaned:
        sargam_counts[seg["sargam"]] = sargam_counts.get(seg["sargam"], 0) + 1

    transcriber = melody.get("transcriber") or next(
        (seg.get("source_model") for seg in final_cleaned if seg.get("source_model")),
        "crepe",
    )
    model_threshold = melody.get("model_threshold")
    agentic_summary = None
    if isinstance(agentic_shadow, dict):
        agentic_summary = {
            "mode": PIANIST_AGENT_MODE,
            "state": agentic_shadow.get("state", "uncertain"),
            "promoted": bool(agentic_shadow.get("promoted")),
            "operation_count": len(agentic_shadow.get("operations") or []),
            "mutations": [
                operation.get("op") for operation in (agentic_shadow.get("operations") or [])
                if operation.get("op") != "keep"
            ],
            "unresolved_questions": agentic_shadow.get("unresolved_questions") or [],
        }

    melody_out = {
        "root": PC_TO_NOTE[root_pc],
        "root_note": f"{final['root']}4",
        "duration": round(duration, 2),
        "tempo": tempo,
        "beats": beats,
        "melody": final_cleaned,
        "sargam_counts": sargam_counts,
        "transcriber": transcriber,
        "model_threshold": model_threshold,
        "reference_alignment": bool(melody.get("reference_alignment")),
        "reference_file": melody.get("reference_file"),
        "source_separation": melody.get("source_separation"),
        "source_separation_warning": melody.get("source_separation_warning"),
        "agentic": agentic_summary,
    }

    payload = {
        "root": final["root"],
        "thaat": final["thaat"],
        "western_scale": final["western_scale"],
        "tempo": tempo,
        "transcriber": transcriber,
        "model_threshold": model_threshold,
        "reference_alignment": bool(melody.get("reference_alignment")),
        "reference_file": melody.get("reference_file"),
        "source_separation": melody.get("source_separation"),
        "source_separation_warning": melody.get("source_separation_warning"),
        "agentic": agentic_summary,
        "chords": chords,
        "melody": melody_out,
    }
    await log_callback(f"__TRANSCRIPTION__:{json.dumps(payload)}")
    await log_callback(f"[System] Transcription session completed. ({len(final_cleaned)} notes)")

    # Final cost summary
    await log_callback(f"__COST__:{json.dumps(cost_tracker.snapshot())}")

    # Phase D: final solo performance of the complete melody on the piano
    await play_final_melody(final_cleaned, log_callback)


async def play_final_melody(segments, log_callback):
    """Send the complete melody to the browser as ONE batched, sample-accurate
    event stream (accuracy review §2.4/§2.5).

    Instead of emitting one note at a time with backend sleeps, we send a single
    __SOLO__ message with a documented timeline contract. The browser schedules
    every event with a look-ahead Tone.js/Web Audio scheduler using the absolute
    start time and TRUE duration, so timing does not drift and overlapping
    events start correctly.
    """
    await log_callback("[System] Now performing the complete melody...")
    t0 = segments[0]["start"] if segments else 0.0
    events = []
    for seg in segments:
        note = seg.get("note")
        if not note or note == "-":
            continue
        events.append({
            "note": note,
            "midi": seg.get("midi"),
            "start": round(float(seg["start"]) - t0, 4),   # relative to solo start
            "duration": round(max(0.08, float(seg["end"]) - float(seg["start"])), 4),
            "velocity": round(float(seg.get("velocity") or 0.8), 3),
            "ornament": seg.get("ornament"),
            "glide_to": seg.get("glide_to"),
            "trill": seg.get("trill"),
            "sustain": seg.get("sustain"),
            "retrigger": seg.get("retrigger"),
            "articulation": seg.get("articulation"),
            "grace_note": seg.get("grace_note"),
            "grace_duration": seg.get("grace_duration"),
        })
    solo = {
        "timeline_origin": "relative_to_solo_start",  # all `start` values are seconds from 0
        "unit": "seconds",
        "audio_duration": round((segments[-1]["end"] - t0) if segments else 0, 4),
        "events": events,
    }
    await log_callback(f"__SOLO__:{json.dumps(solo)}")
    await log_callback("[System] Solo performance complete.")


async def handle_event(raw_line: str, log_callback, agent_text_parts, cost_tracker=None):
    """Parse a single JSON event from opencode's stream and forward it to the UI."""
    try:
        event = json.loads(raw_line)
    except json.JSONDecodeError:
        return

    etype = event.get("type")
    part = event.get("part", {})

    if etype == "text":
        text = (part.get("text") or "").strip()
        if text:
            agent_text_parts.append(text)
            await log_callback(f"[Agent] {text}")

    elif etype == "tool_use":
        # When the agent runs test_notes.py, extract the notes it's testing
        # and physically play them on the piano.
        if part.get("type") == "tool" and part.get("tool") == "bash":
            command = part.get("state", {}).get("input", {}).get("command", "")
            if "test_notes.py" in command:
                notes = NOTE_RE.findall(command)
                for note in notes:
                    await log_callback(f"__PLAY_NOTE__:{note}")
                await log_callback(f"[System] Agent testing notes: {', '.join(notes)}")

    elif etype == "step_finish":
        reason = part.get("reason")
        # accumulate cost/token usage and push a live usage update
        if cost_tracker is not None:
            cost_tracker.add(part.get("tokens"), part.get("cost"))
            await log_callback(f"__COST__:{json.dumps(cost_tracker.snapshot())}")
        if reason == "stop":
            await log_callback("[System] Agent finished its transcription.")
