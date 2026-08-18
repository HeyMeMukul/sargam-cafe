import * as Tone from 'tone';

// --- PIANO CONFIGURATION ---
const startOctave = 2;
const endOctave = 5; // 4 octaves (C2 to B5)
const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

const notes = [];
for (let oct = startOctave; oct <= endOctave; oct++) {
  noteNames.forEach(name => {
    notes.push({
      note: `${name}${oct}`,
      type: name.includes('#') ? 'black' : 'white'
    });
  });
}
// Add final C note
notes.push({ note: `C${endOctave + 1}`, type: 'white' });

const totalWhiteKeys = notes.filter(n => n.type === 'white').length;

const pianoContainer = document.getElementById('piano-keyboard');
const loadingStatus = document.getElementById('loading-status');
const keyElements = {};
let synth = null;

// Initialize Sampler (Zero-lag, high quality)
async function initSynth() {
  await Tone.start();
  if (!synth) {
    return new Promise((resolve) => {
      synth = new Tone.Sampler({
        urls: {
          "A0": "A0.mp3",
          "C1": "C1.mp3",
          "D#1": "Ds1.mp3",
          "F#1": "Fs1.mp3",
          "A1": "A1.mp3",
          "C2": "C2.mp3",
          "D#2": "Ds2.mp3",
          "F#2": "Fs2.mp3",
          "A2": "A2.mp3",
          "C3": "C3.mp3",
          "D#3": "Ds3.mp3",
          "F#3": "Fs3.mp3",
          "A3": "A3.mp3",
          "C4": "C4.mp3",
          "D#4": "Ds4.mp3",
          "F#4": "Fs4.mp3",
          "A4": "A4.mp3",
          "C5": "C5.mp3",
          "D#5": "Ds5.mp3",
          "F#5": "Fs5.mp3",
          "A5": "A5.mp3",
          "C6": "C6.mp3",
          "D#6": "Ds6.mp3",
          "F#6": "Fs6.mp3",
          "A6": "A6.mp3",
          "C7": "C7.mp3",
          "D#7": "Ds7.mp3",
          "F#7": "Fs7.mp3",
          "A7": "A7.mp3",
          "C8": "C8.mp3"
        },
        baseUrl: "https://tonejs.github.io/audio/salamander/",
        onload: () => {
          loadingStatus.textContent = "Piano Ready";
          loadingStatus.classList.add('ready');
          // Room ambience: subtle reverb + gentle feedback delay so the piano
          // sounds like a real instrument in a space, not a dry sample.
          const reverb = new Tone.Reverb({ decay: 2.2, wet: 0.18 });
          const delay = new Tone.FeedbackDelay({ delayTime: 0.25, feedback: 0.15, wet: 0.08 });
          synth.chain(reverb, delay, Tone.Destination);
          resolve();
        }
      });
    });
  }
  return Promise.resolve();
}

// Render Keyboard
function renderPiano() {
  let whiteKeyCount = 0;
  
  notes.forEach((n) => {
    const key = document.createElement('div');
    key.className = `key ${n.type}`;
    key.dataset.note = n.note;

    if (n.type === 'white' && n.note.startsWith('C')) {
      key.innerText = n.note;
    }

    if (n.type === 'black') {
      key.style.width = `calc(100% / ${totalWhiteKeys} * 0.6)`;
      key.style.left = `calc((100% / ${totalWhiteKeys}) * ${whiteKeyCount})`;
    } else {
      whiteKeyCount++;
    }

    // Pointer Events: mouse + touch + multi-touch + drag (P4)
    key.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      try { key.setPointerCapture(e.pointerId); } catch (err) {}
      const id = playNote(n.note);
      key.__pt = id; // remember the voice id for this pointer
    });
    key.addEventListener('pointerup', (e) => {
      stopNote(n.note, key.__pt);
      key.__pt = null;
    });
    key.addEventListener('pointercancel', (e) => {
      stopNote(n.note, key.__pt);
      key.__pt = null;
    });
    key.addEventListener('pointerleave', () => { /* note held while dragging */ });

    keyElements[n.note] = key;
    pianoContainer.appendChild(key);
  });
}

// --- ONE renderer: the Tone.Sampler, used by EVERY path (P1) ---
const ENHARMONIC = { 'E#': 'F', 'B#': 'C', 'Cb': 'B', 'Fb': 'E' };
function pianoNoteName(note) {
  // Scale-degree names like E#4 map to the actual piano sample F4.
  const m = note.match(/^([A-G][#b]?)(-?\d+)$/);
  if (!m) return note;
  return (ENHARMONIC[m[1]] || m[1]) + m[2];
}

// Shared audio/UI event bus (P2): every audio event publishes here and the
// keyboard, live-note display and piano-roll subscribe. No subsystem invents
// its own timing.
const noteEventBus = {
  listeners: new Set(),
  on(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); },
  emit(ev) { this.listeners.forEach(fn => { try { fn(ev); } catch (e) {} }); }
};

// Reference-counted key state (P2): overlapping voices on the same pitch keep
// the key lit until ALL voices end. No single timeout is the source of truth.
const activeVoices = new Map(); // note -> Set of voiceIds
let voiceCounter = 0;

function uiNoteOn(note, voiceId) {
  const set = activeVoices.get(note) || new Set();
  set.add(voiceId);
  activeVoices.set(note, set);
  const key = keyElements[note];
  if (key) { key.classList.add('active'); }
}
function uiNoteOff(note, voiceId) {
  const set = activeVoices.get(note);
  if (!set) return;
  set.delete(voiceId);
  if (!set.size) { activeVoices.delete(note); keyElements[note]?.classList.remove('active'); }
}

// Low-latency sampler attack/release entry (manual keys, MIDI).
function playNote(note) {
  const key = pianoNoteName(note);
  if (synth && synth.loaded && keyElements[key]) {
    const id = ++voiceCounter;
    synth.triggerAttack(key, Tone.now(), 0.85);
    uiNoteOn(key, id);
    noteEventBus.emit({ type: 'noteOn', id, note: key, velocity: 0.85 });
    return id;
  }
  return null;
}
function stopNote(note, voiceId = null) {
  const key = pianoNoteName(note);
  if (!synth || !synth.loaded || !keyElements[key]) return;
  // Release only this pointer's visual voice. Do not release an overlapping
  // scheduled voice on the same pitch until the final reference disappears.
  if (voiceId !== null && voiceId !== undefined) uiNoteOff(key, voiceId);
  if (!activeVoices.has(key)) synth.triggerRelease(key);
}

// THE single renderer for scheduled/automated playback (solo, with-song,
// chords, ornaments). True duration + velocity + reference-counted visuals.
window.agentPlayNote = (note, duration = '4n', velocity = 0.8) => {
  if (!synth || !synth.loaded) return;
  let full = note;
  if (!/[0-9]$/.test(note)) full = `${note}4`;
  const key = pianoNoteName(full);
  if (!keyElements[key]) return;
  // Preserve the lower dynamic range: a 0.2 floor makes quiet left-hand
  // accompaniment as loud as soft melody notes and defeats prominence.
  const v = Math.max(0.04, Math.min(1, velocity || 0.8));
  const id = ++voiceCounter;
  const durMs = typeof duration === 'number' ? Math.max(40, duration * 1000) : 500;
  synth.triggerAttackRelease(key, duration, Tone.now(), v);
  uiNoteOn(key, id);
  keyElements[key].style.filter = `brightness(${1 + (v - 0.5) * 0.8})`;
  noteEventBus.emit({ type: 'noteOn', id, note: key, velocity: v });
  // release is scheduled at the true end; visuals cleared only when THIS voice ends
  setTimeout(() => {
    uiNoteOff(key, id);
    if (!activeVoices.has(key)) keyElements[key].style.filter = '';
  }, Math.min(8000, durMs));
};

renderPiano();


// --- AUDIO PLAYER LOGIC ---
const audioUpload = document.getElementById('audio-upload');
const audioPlayer = document.getElementById('audio-player');
const simBtn = document.getElementById('simulate-ai-btn');
const logBox = document.getElementById('ai-log');

const playPauseBtn = document.getElementById('play-pause-btn');
const progressContainer = document.getElementById('progress-container');
const progressBar = document.getElementById('progress-bar');
const currentTimeDisplay = document.getElementById('current-time');
const totalTimeDisplay = document.getElementById('total-time');
const volumeSlider = document.getElementById('volume-slider');
// Keep the media element and the visible slider synchronized from first load.
audioPlayer.volume = Math.max(0, Math.min(1, Number(volumeSlider.value || 0.8)));

// Auto-init synth on first click anywhere (browser policy)
document.body.addEventListener('click', () => {
  initSynth();
}, { once: true });


function formatTime(seconds) {
  if (isNaN(seconds)) return "0:00";
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
}

// Media-control loop region (set by the agent to play a section in a loop)
let loopRegion = null;

audioPlayer.addEventListener('loadedmetadata', () => {
  totalTimeDisplay.textContent = formatTime(audioPlayer.duration);
});

audioPlayer.addEventListener('timeupdate', () => {
  currentTimeDisplay.textContent = formatTime(audioPlayer.currentTime);
  const progressPercent = (audioPlayer.currentTime / audioPlayer.duration) * 100;
  progressBar.style.width = `${progressPercent}%`;
  // enforce loop region if active
  if (loopRegion && audioPlayer.currentTime >= loopRegion.end) {
    audioPlayer.currentTime = loopRegion.start;
  }
});

volumeSlider.addEventListener('input', (e) => {
  audioPlayer.volume = e.target.value;
});

audioPlayer.addEventListener('ended', () => {
  playPauseBtn.innerHTML = '<svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" stroke-width="1.5" fill="none" stroke-linecap="round" stroke-linejoin="round" class="play-icon"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>';
  melodyPlayEnabled = false;
  chordPlayEnabled = false;
  lastChordIndex = -1;
  playNotationBtn.classList.remove('active');
  playNotationBtn.textContent = 'Play';
});

playPauseBtn.addEventListener('click', () => {
  if (audioPlayer.paused && audioPlayer.src) {
    audioPlayer.play();
    playPauseBtn.innerHTML = '<svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" stroke-width="1.5" fill="none" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>';
  } else if (!audioPlayer.paused) {
    audioPlayer.pause();
    playPauseBtn.innerHTML = '<svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" stroke-width="1.5" fill="none" stroke-linecap="round" stroke-linejoin="round" class="play-icon"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>';
  }
});

progressContainer.addEventListener('click', (e) => {
  if (!audioPlayer.src || isNaN(audioPlayer.duration)) return;
  const rect = progressContainer.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const width = rect.width;
  const clickPercent = clickX / width;
  audioPlayer.currentTime = clickPercent * audioPlayer.duration;
});


// --- REAL AI INTEGRATION ---
let currentFilename = "";

// Transcription state
let transcriptionData = null;
let currentSegmentIndex = -1;

const transcriptionStatus = document.getElementById('transcription-status');
const summaryBox = document.getElementById('transcription-summary');
const summaryRoot = document.getElementById('summary-root');
const summaryThaat = document.getElementById('summary-thaat');
const summaryScale = document.getElementById('summary-scale');
const summaryTempo = document.getElementById('summary-tempo');
const summaryTranscriber = document.getElementById('summary-transcriber');
const sargamStrip = document.getElementById('sargam-strip');
const liveSargam = document.getElementById('live-sargam');
const liveNote = document.getElementById('live-note');
const liveSargamName = document.getElementById('live-sargam-name');
const pianoRoll = document.getElementById('piano-roll');
const rollTrack = document.getElementById('roll-track');
const rollCursor = document.getElementById('roll-cursor');
const playNotationBtn = document.getElementById('play-notation-btn');
const layerChordsBtn = document.getElementById('layer-chords');
const layerMelodyBtn = document.getElementById('layer-melody');
const layerSongBtn = document.getElementById('layer-song');

let melodyPlayEnabled = false;
let chordPlayEnabled = false;
let notationTimers = [];
let lastChordIndex = -1;
let transcriptionCleanup = null;
let soloPlaybackActive = false;
let soloPlaybackEndTimer = null;

// --- Cost / usage status bar ---
const sessionStatus = document.getElementById('session-status');
const statusDot = document.getElementById('status-dot');
const costModel = document.getElementById('cost-model');
const costValue = document.getElementById('cost-value');
const costTokens = document.getElementById('cost-tokens');
const costRequests = document.getElementById('cost-requests');

function fmtTokens(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'k';
  return String(n);
}

function handleCost(msg) {
  try {
    const data = JSON.parse(msg.slice("__COST__:".length));
    if (data.cost !== undefined) costValue.textContent = '$' + data.cost.toFixed(6);
    const t = data.tokens || {};
    costTokens.textContent = fmtTokens(t.total || 0);
    costRequests.textContent = String(data.requests || 0);
  } catch (err) {
    // ignore malformed cost updates
  }
}

function setStatus(state) {
  statusDot.className = 'status-dot ' + state;
  sessionStatus.textContent = state === 'working' ? 'Agent working...' :
    state === 'done' ? 'Complete' : 'Idle';
}

function stopNotationPlayback() {
  notationTimers.forEach(clearTimeout);
  notationTimers = [];
  if (soloPlaybackEndTimer !== null) {
    clearTimeout(soloPlaybackEndTimer);
    soloPlaybackEndTimer = null;
  }
  soloPlaybackActive = false;
  allNotesOff();
}

// Safety: release every held note on the synth (all-notes-off).
function allNotesOff() {
  try {
    if (synth && synth.loaded) {
      Object.keys(keyElements).forEach(k => synth.triggerRelease(k));
      synth.releaseAll();
    }
  } catch (e) { /* ignore */ }
  document.querySelectorAll('.key.active').forEach(k => k.classList.remove('active'));
}


// =====================================================================
// HUMAN TOUCH ENGINE — phrase-aware expressive performance rendering
// (Manus "Performance Rendering" design). Never random per-note jitter:
// timing, velocity, articulation and pedal are CORRELATED within phrases.
// =====================================================================
const HT_STATE = {
  mode: 'faithful',        // 'faithful' | 'songlike'
  expression: 0.8,         // overall dynamic size
  rubato: 0.5,             // phrase-correlated timing deviation
  breath: 0.12,            // extra release at phrase cadence (s)
  pedal: 0.4,              // sustain amount at phrase/harmony boundaries
  prominence: 12,          // melody velocity lift over accompaniment
  seed: 42,
};

// Deterministic seeded PRNG (mulberry32) so humanization is reproducible.
function mulberry32(a) {
  return function() {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

// Smooth phrase timing curve: lean into the peak, stretch at cadence.
function phraseTimingOffset(pos, isPeak, isCadence, tempo) {
  const rub = HT_STATE.rubato * (HT_STATE.mode === 'songlike' ? 1 : 0.25);
  const beat = 60 / (tempo || 120);
  // anchors in seconds, scaled by beat & rubato
  let off = 0;
  if (pos < 0.25) off = 0.004;                 // room to enter
  else if (pos < 0.6) off = -0.006;            // lean into phrase
  if (isPeak) off = 0.006;                     // let the peak breathe
  if (pos > 0.75 && !isPeak) off = 0.012;      // cadence preparation
  if (isCadence) off = 0.02;                   // resolution stretch
  return off * beat * (1 + rub * 6);
}

// Velocity envelope: phrase dynamics + contour + inferred audio pressure (P6).
// attack_energy (loud attack) and brightness (bright/harsh) from the audio feed
// the expression formula; never map raw RMS directly.
function phraseVelocity(baseVel, pos, isPeak, isCadence, tempo, seedRnd, attack, bright) {
  // Expression is always audible. At the neutral value 0.8 it is unity;
  // Faithful mode keeps the measured contour but still permits a controlled
  // master dynamic change instead of making the slider a no-op.
  const exp = HT_STATE.expression;
  const master = 0.78 + 0.275 * exp;
  if (HT_STATE.mode === 'faithful') {
    return Math.max(0.12, Math.min(1, baseVel * master));
  }
  // phrase envelope: swell toward peak, release at cadence
  let env = 1.0;
  env += 0.18 * Math.sin(Math.min(1, pos) * Math.PI) * exp;   // arch
  if (isPeak) env += 0.14 * exp;
  if (isCadence) env -= 0.06 * exp;                            // gentle release
  // inferred pressure from the source audio (attack sharpness + brightness)
  let pressure = 0.0;
  if (attack > 0) pressure += 0.45 * Math.min(1, attack / 3.0);
  if (bright > 0) pressure += 0.15 * bright;
  env += pressure * exp;
  // articulation detail: small correlated wobble per phrase position
  const wobble = (seedRnd() - 0.5) * 0.06 * exp;
  const v = baseVel * env + wobble;
  return Math.max(0.25, Math.min(1, v));
}

// Articulation ratio from the gap to the next note.
function articulationRatio(gapToNext, tempo) {
  const beat = 60 / (tempo || 120);
  const interv = Math.max(0.1, gapToNext);
  if (interv < 0.35 * beat) return 1.0;      // legato (overlap)
  if (interv < 0.8 * beat) return 0.9;       // non-legato
  return 0.75;                               // separated
}

// Collapse adjacent same-pitch evidence for playback only. The transcription
// remains untouched for audit/UI; a held sung vowel should not retrigger the
// same piano key five times just because a section boundary or onset detector
// split it into multiple records.
function collapseHeldMelody(melody) {
  const out = [];
  for (const source of melody || []) {
    const s = { ...source };
    const prev = out[out.length - 1];
    const prevMidi = prev?.midi ?? parseNote(prev?.note)?.midi;
    const midi = s.midi ?? parseNote(s.note)?.midi;
    const flags = s.review_flags || [];
    const explicitRetrigger = s.retrigger || flags.includes('possible_rearticulation');
    const gap = prev ? (Number(s.start) - Number(prev.end)) : Infinity;
    const attack = Number(s.attack_energy || 0);
    const prevAttack = Number(prev?.attack_energy || 0);
    // A zero-gap tracker split is normally one sung hold. Preserve a real
    // retrigger only when it has an explicit flag, a very strong attack, or a
    // separated onset; this prevents long vowels from sounding like tapping.
    const strongRetrigger = attack >= 0.9 || (gap > 0.02 && attack > prevAttack + 0.35);
    if (prev && midi !== undefined && prevMidi === midi && gap <= 0.03 && !explicitRetrigger && !strongRetrigger) {
      prev.end = Math.max(Number(prev.end), Number(s.end));
      prev.velocity = Math.max(Number(prev.velocity || 0), Number(s.velocity || 0));
      prev.attack_energy = Math.max(Number(prev.attack_energy || 0), Number(s.attack_energy || 0));
      prev.pitch_confidence = Math.max(Number(prev.pitch_confidence || 0), Number(s.pitch_confidence || 0));
      continue;
    }
    out.push(s);
  }
  return out;
}

// Short kan detections are grace evidence, not independent melody notes.
// Attach them to the following target for optional Song-like rendering and omit
// the standalone attack in the default Faithful path.
function collapseRenderOrnaments(melody) {
  const out = [];
  for (let i = 0; i < (melody || []).length; i++) {
    const s = { ...melody[i] };
    const next = melody[i + 1];
    const shortKan = s.ornament === 'kan' && (Number(s.end) - Number(s.start)) <= 0.16;
    const close = next && Number(next.start) - Number(s.end) <= 0.08;
    if (shortKan && close) {
      const target = { ...next, graceNote: s.note, graceDuration: Math.min(0.06, Number(s.end) - Number(s.start)) };
      out.push(target);
      i += 1;
      continue;
    }
    out.push(s);
  }
  return out;
}

// Compute a full performance plan for the melody notes.
function buildPerformancePlan(melody, tempo) {
  melody = collapseRenderOrnaments(collapseHeldMelody(melody));
  const seedRnd = mulberry32(HT_STATE.seed);
  const plan = [];
  const byPhrase = {};
  melody.forEach(s => {
    const pid = s.phrase_id !== undefined ? s.phrase_id : 0;
    (byPhrase[pid] = byPhrase[pid] || []).push(s);
  });
  for (const pid in byPhrase) {
    const notes = byPhrase[pid];
    notes.forEach((s, k) => {
      const pos = s.phrase_pos !== undefined ? s.phrase_pos : k / Math.max(1, notes.length - 1);
      const isPeak = !!s.phrase_peak;
      const isCadence = !!s.phrase_cadence;
      const tOff = phraseTimingOffset(pos, isPeak, isCadence, tempo);
      // Melody velocity is the REFERENCE: prominence only lowers accompaniment,
      // never the melody (creates the loudness gap between hands).
      let vel = phraseVelocity(s.velocity || 0.7, pos, isPeak, isCadence, tempo, seedRnd,
                               s.attack_energy, s.brightness);
      // Rendering gate only: retain uncertain events in the transcript, but do
      // not turn a very short, weakly voiced pitch-tracker artifact into a full
      // piano attack. Real notes with strong pitch/voicing evidence still play.
      const uncertain = (s.pitch_confidence !== undefined && s.pitch_confidence < 0.35) &&
                        (s.voicing_confidence === undefined || s.voicing_confidence < 0.65);
      const measuredDur = Math.max(0, Number(s.end) - Number(s.start));
      const render = !(uncertain && measuredDur < 0.28);
      if (uncertain) vel *= render ? 0.35 : 0.08;
      // cadence breath: extend release at phrase end
      let dur = s.end - s.start;
      if (isCadence) dur += HT_STATE.breath * (HT_STATE.mode === 'songlike' ? 1 : 0.2);
      // Pedal emulation: no real sustain pedal API on the Sampler, so extend the
      // phrase-cadence note to let it ring (sustain tail) when pedal is engaged.
      if (isCadence && HT_STATE.pedal > 0.05) dur += HT_STATE.pedal * 0.3;
      const next = notes[k + 1];
      const gapToNext = next ? next.start - s.end : (s.end - s.start);
      const art = articulationRatio(gapToNext, tempo);
      plan.push({
        note: s.note, midi: s.midi,
        scoreStart: s.start, scoreEnd: s.end,
        performanceStart: s.start + tOff,
        performanceEnd: s.end + tOff,
        duration: Math.max(0.08, dur),
        velocity: vel,
        render,
        articulation: art,
        phraseId: pid, isPeak, isCadence, sargam: s.sargam,
        ornament: s.ornament, glide_to: s.glide_to, trill: s.trill, sustain: s.sustain,
        graceNote: s.graceNote, graceDuration: s.graceDuration,
      });
    });
  }
  return plan.sort((a, b) => a.performanceStart - b.performanceStart);
}

// Sustain pedal: press at each phrase start, release at phrase cadence.
function buildPedalPlan(plan, tempo) {
  const groups = [];
  const byPhrase = {};
  plan.forEach(p => (byPhrase[p.phraseId] = byPhrase[p.phraseId] || []).push(p));
  for (const pid in byPhrase) {
    const notes = byPhrase[pid];
    if (!notes.length) continue;
    // Pedal scales the sustain tail of each phrase group (matches the note extension).
    const sustainTail = HT_STATE.pedal * 0.3;
    groups.push({ down: notes[0].performanceStart, up: notes[notes.length - 1].performanceEnd + sustainTail });
  }
  return groups.filter(g => HT_STATE.pedal > 0.05 && HT_STATE.mode === 'songlike');
}

// Pre-compute the plan whenever transcription is ready.
function refreshPerformancePlan() {
  if (!transcriptionData) return;
  performancePlan = buildPerformancePlan(transcriptionData.melody.melody || [], transcriptionData.tempo || 120);
  pedalPlan = buildPedalPlan(performancePlan, transcriptionData.tempo || 120);
}
let performancePlan = [];
let pedalPlan = [];

// Touch curve: remap a 0..1 velocity through soft/neutral/bright.
function touchCurve(v, curve) {
  // neutral identity by default; soft pulls down, bright pushes up
  if (curve === 'soft') return Math.pow(v, 1.4);
  if (curve === 'bright') return Math.pow(v, 0.7);
  return v;
}

// Convert a MIDI note number to a note name like "G#4"
const PC_ARR = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
function midiToNote(midi) {
  return `${PC_ARR[midi % 12]}${Math.floor(midi / 12) - 1}`;
}

// Scale pitch-classes for the detected thaat (from transcriptionData root/thaat).
// Thaat interval sets (semitones from Sa), used for scale-aware ornament neighbours.
const THAAT_INTERVALS = {
  'Bilawal': [0,2,4,5,7,9,11], 'Kalyan': [0,2,4,6,7,9,11], 'Khamaj': [0,2,4,5,7,9,10],
  'Kafi': [0,2,3,5,7,9,10], 'Asavari': [0,2,3,5,7,8,10], 'Bhairav': [0,1,4,5,7,8,11],
  'Bhairavi': [0,1,3,5,7,8,10], 'Marwa': [0,1,4,6,7,9,11], 'Poorvi': [0,1,4,6,7,8,11],
  'Todi': [0,1,3,6,7,8,11], 'Yaman': [0,2,4,6,7,9,11]
};
function scalePcs() {
  const root = transcriptionData && transcriptionData.root;
  const thaat = transcriptionData && transcriptionData.thaat;
  const rootPc = root && NOTE_TO_PC[root.replace(/\d/g, '')] !== undefined
    ? NOTE_TO_PC[root.replace(/\d/g, '')] : 0;
  const ivs = THAAT_INTERVALS[thaat] || [0,2,4,5,7,9,11];
  return new Set(ivs.map(iv => (rootPc + iv) % 12));
}
// Nearest scale pitch-class neighbour (above or below) for a midi note.
function scaleNeighbour(midi, dir) {
  const pcs = scalePcs();
  const pc = ((midi % 12) + 12) % 12;
  let best = null, bestDist = Infinity;
  for (const s of pcs) {
    let d = (s - pc) % 12;
    if (d > 6) d -= 12;
    if (dir > 0 && d <= 0) d += 12;
    if (dir < 0 && d >= 0) d -= 12;
    if (Math.abs(d) > 0 && Math.abs(d) < Math.abs(bestDist)) {
      bestDist = d; best = s;
    }
  }
  return best !== null ? midi + bestDist : midi + (dir > 0 ? 2 : -1);
}

// Schedule the ornamented rendering of a single note at `when` (seconds from now).
// - meend : fast, quiet grace-note run ending ON the target (never simultaneous)
// - gamak/trill : rapid alternation with the SCALE-AWARE neighbouring note
// - kan : a short SCALE-AWARE grace note just before the main note
function playOrnamentedNote(seg, when, dur, vel) {
  const note = seg.note;
  if (!note || note === '-' || seg.render === false) return;
  if (seg.graceNote && HT_STATE.mode === 'songlike') {
    notationTimers.push(setTimeout(() => {
      window.agentPlayNote(seg.graceNote, seg.graceDuration || 0.05, vel * 0.35);
    }, Math.max(0, when * 1000 - 60)));
    notationTimers.push(setTimeout(() => {
      window.agentPlayNote(note, dur, vel);
    }, Math.max(0, when * 1000)));
    return;
  }
  // The extractor’s ornament labels are hypotheses, not guaranteed extra
  // attacks. Faithful mode renders the measured melody only; Song-like mode
  // opts into ornament playback. This prevents false kan/gamak/meend labels
  // from sounding like repeated key tapping in the default path.
  if (HT_STATE.mode !== 'songlike' && seg.ornament === 'kan') return;
  if (HT_STATE.mode !== 'songlike' || !seg.ornament) {
    notationTimers.push(setTimeout(() => {
      window.agentPlayNote(note, dur, vel);
    }, Math.max(0, when * 1000)));
    return;
  }
  const noteMidi = parseNote(note)?.midi;
  const at = () => when * 1000;

  if (seg.ornament === 'meend' && seg.glide_to !== undefined && noteMidi) {
    // vocal_melody.py defines glide_to as the pitch-curve destination. Travel
    // from the measured note toward that destination; do not reverse it.
    const targetMidi = Number(seg.glide_to);
    const steps = Math.abs(targetMidi - noteMidi);
    if (Number.isFinite(targetMidi) && steps >= 1 && steps <= 12) {
      const dir = Math.sign(targetMidi - noteMidi);
      const passDur = Math.max(0.04, dur / (steps + 1));
      for (let s = 0; s <= steps; s++) {
        const passMidi = noteMidi + dir * s;
        const passNote = midiToNote(passMidi);
        const isLast = (s === steps);
        notationTimers.push(setTimeout(() => {
          window.agentPlayNote(passNote, passDur, isLast ? vel : vel * 0.3);
        }, at() + s * passDur * 1000));
      }
    } else {
      notationTimers.push(setTimeout(() => {
        window.agentPlayNote(note, dur, vel);
      }, at()));
    }
  } else if (seg.trill && noteMidi) {
    // gamak: alternate note with its SCALE-AWARE upper neighbour at the tempo
    const neighbor = scaleNeighbour(noteMidi, 1);
    const trillDur = Math.min(0.22, dur / 2);
    let t = 0;
    while (t < dur) {
      const mainNote = (Math.floor(t / trillDur) % 2 === 0) ? note : midiToNote(neighbor);
      notationTimers.push(setTimeout(() => {
        window.agentPlayNote(mainNote, trillDur * 0.8, vel * 0.8);
      }, at() + t * 1000));
      t += trillDur;
    }
  } else if (seg.ornament === 'kan' && noteMidi) {
    // kan: a short SCALE-AWARE grace note (below) before the main note
    const grace = midiToNote(scaleNeighbour(noteMidi, -1));
    notationTimers.push(setTimeout(() => {
      window.agentPlayNote(grace, 0.05, 0.3);
    }, at() - 60));
    notationTimers.push(setTimeout(() => {
      window.agentPlayNote(note, dur, vel);
    }, at()));
  } else {
    notationTimers.push(setTimeout(() => {
      window.agentPlayNote(note, dur, vel);
    }, at()));
  }
}

function playNotation(data, opts = {}) {
  stopNotationPlayback();
  if (!data) return;
  const playChords = opts.chords !== false;
  const playMelody = opts.melody !== false;
  const melody = data.melody.melody || [];
  const chords = data.chords || [];
  let delay = 0;
  let prevEnd = 0;

  // Chord backing (left hand): sustained chords under the melody
  if (playChords && chords.length) {
    chords.forEach(c => {
      const when = c.start;
      const dur = Math.max(0.5, c.end - c.start);
      notationTimers.push(setTimeout(() => {
        (c.midis || []).forEach(m => window.agentPlayNote(midiToNote(m), dur, Math.max(0.04, 0.22 - (HT_STATE.prominence / 30) * 0.17)));
      }, when * 1000));
    });
  }

  if (!playMelody) return delay * 1000;
  // route through the Human Touch performance plan (phrase timing/velocity)
  let plan = performancePlan;
  if (!plan || !plan.length) {
    plan = melody.map(seg => ({
      note: seg.note, performanceStart: seg.start, duration: seg.end - seg.start,
      velocity: seg.velocity || 0.8, ornament: seg.ornament, glide_to: seg.glide_to,
      trill: seg.trill, sustain: seg.sustain,
    }));
  }
  let t = 0;
  let prev = 0;
  plan.forEach((p) => {
    t += Math.max(0, p.performanceStart - prev);
    const seg = {
      note: p.note, ornament: p.ornament, glide_to: p.glide_to, trill: p.trill,
      graceNote: p.graceNote, graceDuration: p.graceDuration, render: p.render,
    };
    playOrnamentedNote(seg, t, p.duration, p.velocity);
    t += p.duration;
    prev = p.performanceStart + p.duration;
  });
  return t * 1000;
}

const SARGAM_COLORS = ['sa', 're', 'ga', 'ma', 'pa', 'dha', 'ni', 'other'];
const SARGAM_ORDER = ['Sa', 're', 'Re', 'ga', 'Ga', 'Ma', 'ma', 'Pa', 'dha', 'Dha', 'ni', 'Ni'];

function sargamClass(sargam) {
  const norm = (sargam || '').toLowerCase();
  if (SARGAM_COLORS.includes(norm)) return norm;
  return 'other';
}

// Parse a note like "G#4" -> {pitchClass:"G#", octave:4, midi:68}
const NOTE_TO_PC = { C:0, 'C#':1, 'Db':1, D:2, 'D#':3, 'Eb':3, E:4, 'Fb':4, F:5, 'F#':6, 'Gb':6,
  G:7, 'G#':8, 'Ab':8, A:9, 'A#':10, 'Bb':10, B:11, 'Cb':11, 'E#':5, 'B#':0 };
function parseNote(note) {
  if (!note) return null;
  const m = note.match(/^([A-G][#b]?)(-?\d+)$/);
  if (!m) return null;
  const octave = parseInt(m[2], 10);
  const pc = NOTE_TO_PC[m[1]];
  if (pc === undefined) return null;
  return { pitchClass: m[1], octave, midi: (octave + 1) * 12 + pc };
}

const htPanel = document.getElementById('human-touch');
const htModeBtns = document.querySelectorAll('.ht-mode-btn');
const htExpression = document.getElementById('ht-expression');
const htRubato = document.getElementById('ht-rubato');
const htBreath = document.getElementById('ht-breath');
const htPedal = document.getElementById('ht-pedal');
const htProminence = document.getElementById('ht-prominence');
const htSeed = document.getElementById('ht-seed');

function setupHumanTouch() {
  if (!htPanel) return;
  htPanel.hidden = false;
  htModeBtns.forEach(b => b.onclick = () => {
    htModeBtns.forEach(x => x.classList.toggle('active', x === b));
    HT_STATE.mode = b.id === 'ht-songlike' ? 'songlike' : 'faithful';
    refreshPerformancePlan();
  });
  const bind = (el, key) => el.oninput = () => {
    HT_STATE[key] = parseFloat(el.value);
    // update the live value label next to the slider
    const val = el.nextElementSibling;
    if (val && val.classList.contains('ht-value')) val.textContent = el.value;
    refreshPerformancePlan();
  };
  bind(htExpression, 'expression');
  bind(htRubato, 'rubato');
  bind(htBreath, 'breath');
  bind(htPedal, 'pedal');
  bind(htProminence, 'prominence');
  htSeed.onchange = () => { HT_STATE.seed = parseInt(htSeed.value) || 0; refreshPerformancePlan(); };

  // Preview phrase: play the first phrase of the current performance plan so
  // the user hears the Human Touch settings immediately.
  const htPreview = document.getElementById('ht-preview');
  if (htPreview) htPreview.onclick = () => {
    if (!performancePlan || !performancePlan.length) return;
    stopNotationPlayback();
    const minPid = Math.min(...performancePlan.map(p => p.phraseId));
    const phrase = performancePlan.filter(p => p.phraseId === minPid && p.render !== false);
    const origin = phrase.length ? phrase[0].performanceStart : 0;
    phrase.forEach(p => {
      if (!p.note || p.note === '-') return;
      const timer = setTimeout(() => {
        window.agentPlayNote(p.note, p.duration, p.velocity);
      }, Math.max(0, (p.performanceStart - origin) * 1000));
      notationTimers.push(timer);
    });
  };
}

const midiEnable = document.getElementById('midi-enable');
const midiStatus = document.getElementById('midi-status');
const midiDiag = document.getElementById('midi-diag');
let midiAccess = null;

function setupMidi() {
  if (!midiEnable) return;
  midiEnable.onclick = async () => {
    if (midiAccess) { midiStatus.textContent = 'MIDI active'; return; }
    try {
      if (!navigator.requestMIDIAccess) {
        midiStatus.textContent = 'Web MIDI not supported in this browser';
        return;
      }
      midiAccess = await navigator.requestMIDIAccess();
      midiAccess.onstatechange = refreshMidiDevices;
      refreshMidiDevices();
      const inputs = midiAccess.inputs.values();
      for (const input of inputs) {
        input.onmidimessage = onMidiMessage;
      }
    } catch (e) {
      midiStatus.textContent = 'MIDI access denied';
    }
  };
}

function refreshMidiDevices() {
  const n = midiAccess ? midiAccess.inputs.size : 0;
  if (n > 0) { midiStatus.textContent = n + ' MIDI device(s) active'; midiStatus.classList.add('on'); }
  else { midiStatus.textContent = 'No MIDI device'; midiStatus.classList.remove('on'); }
}

const CC_SUSTAIN = 64;
let midiHeld = {};      // midi note -> velocity
let midiPedal = false;

function onMidiMessage(e) {
  if (!e.data || e.data.length < 2) return;
  const status = e.data[0] & 0xF0;
  const channel = e.data[0] & 0x0F;
  const d1 = e.data[1];
  const d2 = e.data[2] !== undefined ? e.data[2] : 0;
  const NOTE_ON = 0x90, NOTE_OFF = 0x80, POLY_PRESSURE = 0xA0,
        CHANNEL_PRESSURE = 0xD0, CONTROL_CHANGE = 0xB0;

  if ((status === NOTE_ON && d2 > 0) || (status === NOTE_OFF) || (status === NOTE_ON && d2 === 0)) {
    const isOn = (status === NOTE_ON && d2 > 0);
    const midi = d1;
    const vel = isOn ? d2 / 127 : 0;
    if (isOn) {
      midiHeld[midi] = vel;
      const note = midiToNote(midi);
      // route real velocity through the Human Touch piano
      window.agentPlayNote(note, 0.6, vel);
      updateMidiDiag('velocity', (vel * 100).toFixed(0) + '%');
    } else {
      delete midiHeld[midi];
    }
  } else if (status === CHANNEL_PRESSURE || status === POLY_PRESSURE) {
    // aftertouch -> route to expression/brightness
    const p = d2 / 127;
    HT_STATE.expression = 0.4 + p; // map pressure to expression
    updateMidiDiag('aftertouch', (p * 100).toFixed(0) + '%');
  } else if (status === CONTROL_CHANGE && d1 === CC_SUSTAIN) {
    midiPedal = d2 >= 64;
    // The Sampler sustains naturally (no real pedal API): release all on pedal-up.
    if (!midiPedal && synth && synth.loaded) {
      try { synth.releaseAll(); } catch (err) {}
    }
    updateMidiDiag('pedal', midiPedal ? 'down' : 'up');
  }
}

function updateMidiDiag(key, val) {
  if (!midiDiag) return;
  const parts = {};
  const m = midiDiag.textContent.match(/(velocity|aftertouch|pedal):\s*[^&]*/g) || [];
  m.forEach(s => { const [k, v] = s.split(': '); parts[k.trim()] = v; });
  parts[key] = val;
  midiDiag.textContent = `velocity: ${parts.velocity || '–'} &nbsp; aftertouch: ${parts.aftertouch || '–'} &nbsp; pedal: ${parts.pedal || '–'}`;
}

function renderTranscription(data) {
  // A second transcription replaces the previous event handlers and scheduler;
  // otherwise every subsequent run multiplies timeupdate/chord callbacks.
  if (transcriptionCleanup) {
    transcriptionCleanup();
    transcriptionCleanup = null;
  }
  summaryRoot.textContent = data.root;
  summaryThaat.textContent = data.thaat;
  summaryScale.textContent = data.western_scale;
  summaryTempo.textContent = data.tempo ? `${data.tempo} BPM` : '—';
  const engine = data.transcriber || data.melody.transcriber || 'unknown';
  summaryTranscriber.textContent = engine === 'rosvot' ? 'ROSVOT' : engine === 'crepe' ? 'CREPE' : engine;
  summaryBox.hidden = false;

  // Sargam strip
  const counts = data.melody.sargam_counts || {};
  sargamStrip.innerHTML = '';
  const chipLabels = {
    Sa: 'Sa', re: 're♭', Re: 'Re', ga: 'ga♭', Ga: 'Ga',
    Ma: 'Ma', ma: 'ma♯', Pa: 'Pa', dha: 'dha♭', Dha: 'Dha',
    ni: 'ni♭', Ni: 'Ni', other: 'chromatic',
  };
  const present = SARGAM_ORDER.filter(s => counts[s] > 0);
  if (counts.other > 0) present.push('other');
  (present.length ? present : ['Sa']).forEach(s => {
    const chip = document.createElement('span');
    chip.className = 'sargam-chip' + (s === 'Sa' ? ' highlight' : '');
    chip.textContent = chipLabels[s] || s;
    chip.title = `${counts[s] || 0} event${counts[s] === 1 ? '' : 's'}`;
    chip.dataset.sargam = s;
    sargamStrip.appendChild(chip);
  });

  // --- Vertical piano-roll ---
  const melody = data.melody.melody || [];
  renderPianoRoll(melody);

  liveSargam.hidden = false;
  playNotationBtn.disabled = false;
  layerChordsBtn.disabled = false;
  layerMelodyBtn.disabled = false;
  layerSongBtn.disabled = false;
  setupHumanTouch();
  setupMidi();
  refreshPerformancePlan();

  // Highlight the active sargam chip
  const allChips = [...sargamStrip.querySelectorAll('.sargam-chip')];
  function updateChips(activeSargam) {
    allChips.forEach(c => {
      c.classList.toggle('highlight', c.dataset.sargam === activeSargam);
    });
  }

  // Drive the live display + roll cursor from the audio player's time
  function updateLiveFromTime() {
    if (!transcriptionData) return;
    const t = audioPlayer.currentTime;
    const rawMel = transcriptionData.melody.melody || [];
    const mel = (performancePlan && performancePlan.length)
      ? performancePlan.filter(p => p.render !== false).map(p => ({
          start: p.performanceStart, end: p.performanceEnd, note: p.note, sargam: p.sargam || ''
        }))
      : rawMel;
    let idx = -1;
    for (let i = 0; i < mel.length; i++) {
      if (t >= mel[i].start && t < mel[i].end) { idx = i; break; }
    }
    if (idx !== -1) {
      const seg = mel[idx];
      liveNote.textContent = seg.note;
      liveSargamName.textContent = seg.sargam;
      liveSargam.classList.add('playing');
      updateChips(seg.sargam);
      if (idx !== currentSegmentIndex) {
        currentSegmentIndex = idx;
        // Note playback is handled by the look-ahead scheduler; here we only
        // drive the visual "now playing" highlight.
      }
    } else {
      liveNote.textContent = '-';
      liveSargamName.textContent = '-';
      liveSargam.classList.remove('playing');
      currentSegmentIndex = -1;
    }
    // Chord backing (left hand) synced with the audio
    const ch = transcriptionData.chords || [];
    if (chordPlayEnabled && ch.length && !withSongSchedulerActive) {
      let ci = -1;
      for (let i = 0; i < ch.length; i++) {
        if (t >= ch[i].start && t < ch[i].end) { ci = i; break; }
      }
      if (ci !== lastChordIndex && ci !== -1) {
        lastChordIndex = ci;
        const c = ch[ci];
        const dur = Math.max(0.5, c.end - c.start);
        (c.midis || []).forEach(m => window.agentPlayNote(midiToNote(m), dur, Math.max(0.04, 0.22 - (HT_STATE.prominence / 30) * 0.17)));
      }
      if (ci === -1 && t > 0) lastChordIndex = -1; // outside all chords -> reset
    }
    // cursor position in the piano-roll
    const dur = audioPlayer.duration || transcriptionData.melody.duration || 1;
    const pct = (t / dur) * 100;
    rollCursor.style.left = `${pct}%`;
  }

  // --- Look-ahead scheduler for WITH-SONG playback ---
  // Fires each note/chord exactly ONCE at its true onset (driven by the audio
  // clock via requestAnimationFrame) and lets it ring its TRUE duration. This
  // replaces unreliable `timeupdate` triggering so long sustains are not cut
  // short (accuracy review §2.4/§2.5). `updateLiveFromTime` is kept for the
  // visual "now playing" highlight only.
  let melTriggered = -1;   // last melody index already fired
  let chordTriggered = -1; // last chord index already fired
  let rafId = null;
  let withSongSchedulerActive = false;

  function stopWithSongScheduler() {
    if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; }
    withSongSchedulerActive = false;
    melTriggered = -1;
    chordTriggered = -1;
  }

  function startWithSongScheduler(melody, chords, wantMelody, wantChords) {
    stopWithSongScheduler();
    withSongSchedulerActive = true;
    const endsAt = (audioPlayer.duration || 0);
    function tick() {
      rafId = requestAnimationFrame(tick);
      const t = audioPlayer.currentTime;
      if (audioPlayer.paused) return;
      // Read the latest plan every frame so Human Touch slider changes affect
      // notes that have not fired yet, rather than a stale captured array.
      const plan = performancePlan;
      // Read the current layer toggles every frame so clicking CHORDS or
      // MELODY during With Song playback takes effect immediately.
      const currentWantMelody = layerMelodyBtn.classList.contains('active');
      const currentWantChords = layerChordsBtn.classList.contains('active');
      // fire melody notes through the Human Touch plan
      if (currentWantMelody && plan.length) {
        for (let i = melTriggered + 1; i < plan.length; i++) {
          const p = plan[i];
          if (t >= p.performanceStart) {
            melTriggered = i;
            if (p.note && p.note !== '-') {
              if (p.render !== false) window.agentPlayNote(p.note, p.duration, p.velocity);
            }
          } else break;
        }
      }
      // fire chord changes once (left hand, softer than melody)
      if (currentWantChords && chords.length) {
        for (let i = chordTriggered + 1; i < chords.length; i++) {
          const c = chords[i];
          if (t >= c.start) {
            chordTriggered = i;
            const dur = Math.max(0.5, c.end - c.start);
            const vel = Math.max(0.04, 0.22 - (HT_STATE.prominence / 30) * 0.17);
            (c.midis || []).forEach(m => window.agentPlayNote(midiToNote(m), dur, vel));
          } else break;
        }
      }
      if (t >= endsAt) stopWithSongScheduler();
    }
    rafId = requestAnimationFrame(tick);
  }

  audioPlayer.addEventListener('timeupdate', updateLiveFromTime);
  audioPlayer.addEventListener('pause', stopWithSongScheduler);
  transcriptionCleanup = () => {
    audioPlayer.removeEventListener('timeupdate', updateLiveFromTime);
    audioPlayer.removeEventListener('pause', stopWithSongScheduler);
    stopWithSongScheduler();
  };
  updateLiveFromTime();

  // Layer toggles: Chords / Melody / With Song
  layerChordsBtn.onclick = () => {
    layerChordsBtn.classList.toggle('active');
    chordPlayEnabled = layerChordsBtn.classList.contains('active');
  };
  layerMelodyBtn.onclick = () => {
    layerMelodyBtn.classList.toggle('active');
    melodyPlayEnabled = layerMelodyBtn.classList.contains('active');
  };
  layerSongBtn.onclick = () => {
    layerSongBtn.classList.toggle('active');
    if (!layerSongBtn.classList.contains('active')) {
      // turning With Song off stops the audio-synced layers
      melodyPlayEnabled = false;
      chordPlayEnabled = false;
      audioPlayer.pause();
    }
  };

  // Play: plays the selected layer combination, with or without the song
  playNotationBtn.onclick = () => {
    const isOn = playNotationBtn.classList.contains('active');
    const wantChords = layerChordsBtn.classList.contains('active');
    const wantMelody = layerMelodyBtn.classList.contains('active');
    const withSong = layerSongBtn.classList.contains('active');

    if (isOn) {
      // stop current playback
      playNotationBtn.classList.remove('active');
      playNotationBtn.textContent = 'Play';
      stopNotationPlayback();
      stopWithSongScheduler();
      melodyPlayEnabled = false;
      chordPlayEnabled = false;
      lastChordIndex = -1;
      audioPlayer.pause();
      return;
    }

    playNotationBtn.classList.add('active');
    playNotationBtn.textContent = 'Stop';
    if (withSong) {
      // audio-synced: notes/chords fire via the look-ahead scheduler
      melodyPlayEnabled = wantMelody;
      chordPlayEnabled = wantChords;
      lastChordIndex = -1;
      currentSegmentIndex = -1;
      audioPlayer.currentTime = 0;
      audioPlayer.play();
      const mel = transcriptionData.melody.melody || [];
      const ch = transcriptionData.chords || [];
      startWithSongScheduler(mel, ch, wantMelody, wantChords);
      updateLiveFromTime();
    } else {
      // standalone: schedule the selected layers on timers
      stopNotationPlayback();
      playNotation(transcriptionData, { chords: wantChords, melody: wantMelody });
    }
  };
}

function renderPianoRoll(melody) {
  rollTrack.innerHTML = '';
  if (!melody.length) return;

  // Determine pitch range from the melody (clamp to the piano's C2..B5 range)
  const minMidi = 36; // C2
  const maxMidi = 83; // B5
  const usedMin = Math.max(minMidi, Math.min(...melody.map(s => parseNote(s.note)?.midi ?? 60)));
  const usedMax = Math.min(maxMidi, Math.max(...melody.map(s => parseNote(s.note)?.midi ?? 60)));

  const PX_PER_SEC = 24;
  const ROW_H = 12;

  const duration = audioPlayer.duration || melody[melody.length - 1].end || 1;
  const trackWidth = duration * PX_PER_SEC;
  rollTrack.style.width = `${trackWidth}px`;
  rollTrack.style.height = `${(usedMax - usedMin + 1) * ROW_H}px`;

  // subtle row grid lines
  for (let m = usedMin; m <= usedMax; m++) {
    const row = document.createElement('div');
    row.className = 'roll-row' + (NOTE_TO_PC[parseInt(m) % 12 === 0 ? 'C' : ''] !== undefined && m % 12 === 0 ? ' octave-line' : '');
    row.style.bottom = `${(m - usedMin) * ROW_H}px`;
    row.style.height = `${ROW_H}px`;
    row.style.width = `${trackWidth}px`;
    rollTrack.appendChild(row);
  }

  melody.forEach((seg, i) => {
    const p = parseNote(seg.note);
    if (!p || p.midi < usedMin || p.midi > usedMax) return;
    const el = document.createElement('div');
    el.className = `roll-note ${sargamClass(seg.sargam)}`;
    el.dataset.index = i;
    el.title = `${seg.note} (${seg.sargam}) · ${seg.start}s–${seg.end}s`;
    el.style.left = `${seg.start * PX_PER_SEC}px`;
    el.style.width = `${Math.max(8, (seg.end - seg.start) * PX_PER_SEC)}px`;
    el.style.bottom = `${(p.midi - usedMin) * ROW_H + 1}px`;
    el.style.height = `${ROW_H - 2}px`;
    el.textContent = seg.note;
    el.addEventListener('click', () => {
      audioPlayer.currentTime = seg.start;
    });
    rollTrack.appendChild(el);
  });

  pianoRoll.hidden = false;
  pianoRoll.classList.add('has-data');
}

audioUpload.addEventListener('change', async function(e) {
  const file = e.target.files[0];
  if (file) {
    // Reset any previous transcription
    transcriptionData = null;
    currentSegmentIndex = -1;
    melodyPlayEnabled = false;
    chordPlayEnabled = false;
    lastChordIndex = -1;
    loopRegion = null;
    playNotationBtn.disabled = true;
    playNotationBtn.classList.remove('active');
    playNotationBtn.textContent = 'Play';
    layerChordsBtn.disabled = true;
    layerMelodyBtn.disabled = true;
    layerSongBtn.disabled = true;
    layerChordsBtn.classList.add('active');
    layerMelodyBtn.classList.add('active');
    layerSongBtn.classList.add('active');
    stopNotationPlayback();
    summaryBox.hidden = true;
    pianoRoll.hidden = true;
    pianoRoll.classList.remove('has-data');
    liveSargam.hidden = true;
    if (htPanel) htPanel.hidden = true;
    transcriptionStatus.textContent = 'Waiting for agent';
    transcriptionStatus.classList.remove('ready');

    const url = URL.createObjectURL(file);
    audioPlayer.src = url;
    
    logBox.innerHTML = `Track loaded locally: ${file.name}.<br>> Uploading to Agent...`;
    
    const formData = new FormData();
    formData.append("file", file);
    
    try {
      const res = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      currentFilename = data.filename;
      logBox.innerHTML += `<br>> Upload successful. Ready for Agent initialization.`;
      simBtn.disabled = false;
    } catch (err) {
      logBox.innerHTML += `<br>> [Fatal Error] Could not connect to backend on port 8000. Is it running?`;
    }
  }
});

function handleMediaControl(msg) {
  try {
    const cmd = JSON.parse(msg.slice("__MEDIA__:".length));
    if (!cmd || !cmd.action) return;
    switch (cmd.action) {
      case 'seek':
        if (isFinite(cmd.time)) audioPlayer.currentTime = cmd.time;
        break;
      case 'play':
        if (audioPlayer.src && audioPlayer.paused) {
          audioPlayer.play();
          playPauseBtn.innerHTML = '<svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" stroke-width="1.5" fill="none" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>';
        }
        break;
      case 'pause':
        audioPlayer.pause();
        playPauseBtn.innerHTML = '<svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" stroke-width="1.5" fill="none" stroke-linecap="round" stroke-linejoin="round" class="play-icon"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>';
        break;
      case 'loop':
        if (isFinite(cmd.start) && isFinite(cmd.end)) {
          loopRegion = { start: cmd.start, end: cmd.end };
        }
        break;
      case 'unloop':
        loopRegion = null;
        break;
    }
  } catch (err) {
    logBox.innerHTML += `<br>> [Error] Bad media control message: ${err.message}`;
  }
}

simBtn.addEventListener('click', () => {
  if (!synth || !synth.loaded) {
    logBox.innerHTML += `<br>> [Error] Piano samples are still loading. Please wait.`;
    return;
  }
  if (!currentFilename) return;

  if (audioPlayer.paused) {
    audioPlayer.play();
    playPauseBtn.innerHTML = '<svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" stroke-width="1.5" fill="none" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>';
  }
  
  simBtn.disabled = true;
  transcriptionStatus.textContent = 'Agent working...';
  setStatus('working');
  logBox.innerHTML += `<br>> [System] Connecting to Piano Master Agent...`;
  
  const ws = new WebSocket("ws://localhost:8000/ws/agent");
  
  ws.onopen = () => {
    ws.send(currentFilename);
  };
  
  // --- Sample-accurate solo scheduling (look-ahead scheduler) ---
  // Receives a __SOLO__ batch with a timeline contract and schedules every
  // event with Tone.js, so timing does not drift and true durations hold.
  function scheduleSolo(solo) {
    stopNotationPlayback();
    const events = solo.events || [];
    if (!events.length) return;
    // Use the same Human Touch renderer as standalone playback. The backend
    // contract is relative to zero; the performance plan is source-absolute,
    // so normalize it once and add a small lead-in for browser scheduling.
    let evList = events;
    if (performancePlan && performancePlan.length) {
      const origin = performancePlan[0].performanceStart;
      evList = performancePlan.map(p => ({
        note: p.note, start: p.performanceStart - origin, duration: p.duration,
        velocity: p.velocity, ornament: p.ornament, glide_to: p.glide_to,
        trill: p.trill, graceNote: p.graceNote, graceDuration: p.graceDuration,
        render: p.render,
      }));
    }
    evList.forEach(ev => {
      if (!ev.note || ev.note === '-' || ev.render === false) return;
      playOrnamentedNote(
        ev,
        0.15 + Number(ev.start || 0),
        Math.max(0.05, Number(ev.duration || 0.5)),
        Number(ev.velocity || 0.8),
      );
    });
    soloPlaybackActive = evList.some(ev => ev.render !== false);
    if (soloPlaybackActive) {
      const total = Math.max(0, ...evList.map(ev => Number(ev.start || 0) + Number(ev.duration || 0)));
      soloPlaybackEndTimer = setTimeout(() => {
        soloPlaybackActive = false;
        soloPlaybackEndTimer = null;
      }, (total + 0.5) * 1000);
    }
  }

  // Schedule a single note with Tone.js at the current audio clock (look-ahead).
  function scheduleToneNote(note, dur, vel) {
    if (!synth || !synth.loaded) return;
    const key = pianoNoteName(note);
    if (!keyElements[key]) return;
    const v = Math.max(0.04, Math.min(1, vel || 0.8));
    synth.triggerAttackRelease(key, Math.max(0.05, dur), Tone.now(), v);
    keyElements[key].classList.add('active');
    setTimeout(() => {
      keyElements[key].classList.remove('active');
    }, Math.max(60, dur * 1000));
  }

  ws.onmessage = (event) => {
    const msg = event.data;
    
    if (msg.startsWith("__TRANSCRIPTION__:")) {
      try {
        const payload = JSON.parse(msg.slice("__TRANSCRIPTION__:".length));
        transcriptionData = payload;
        renderTranscription(payload);
        transcriptionStatus.textContent = 'Transcription ready';
        transcriptionStatus.classList.add('ready');
      } catch (err) {
        logBox.innerHTML += `<br>> [Error] Could not parse transcription payload: ${err.message}`;
      }
      return;
    }

    if (msg.startsWith("__MEDIA__:")) {
      handleMediaControl(msg);
      return;
    }

    if (msg.startsWith("__COST__:")) {
      handleCost(msg);
      return;
    }

    if (msg.startsWith("__SOLO__:")) {
      try {
        const solo = JSON.parse(msg.slice("__SOLO__:".length));
        scheduleSolo(solo);
      } catch (err) {
        logBox.innerHTML += `<br>> [Error] Could not parse solo payload: ${err.message}`;
      }
      return;
    }

    if (msg.startsWith("__PLAY_NOTE__:__")) {
        const note = msg.split("__")[2];
        window.agentPlayNote(note, '4n', 0.8);
    } else if (msg.startsWith("__PLAY_NOTE__:")){
        // New full contract:  __PLAY_NOTE__:<note>:<start>:<duration>:<velocity>
        // Legacy fallback:     __PLAY_NOTE__:<note>[:<velocity>]  -> '4n'
        const parts = msg.split(":");
        const note = parts[1];
        if (parts.length >= 5) {
            const start = parseFloat(parts[2]);
            const dur = parseFloat(parts[3]);
            const vel = parseFloat(parts[4]);
            const d = (isNaN(dur) || dur <= 0) ? '4n' : Math.max(0.05, dur);
            window.agentPlayNote(note, d, isNaN(vel) ? 0.8 : vel);
        } else {
            const vel = parts[2] !== undefined ? parseFloat(parts[2]) : 0.8;
            window.agentPlayNote(note, '4n', isNaN(vel) ? 0.8 : vel);
        }
    } else {
        logBox.innerHTML += `<br>${msg}`;
        logBox.scrollTop = logBox.scrollHeight;
    }
  };
  
  ws.onclose = () => {
    logBox.innerHTML += `<br>> [System] Connection closed.`;
    // The websocket closes immediately after sending the batched solo. Do not
    // cancel that scheduled performance merely because transport is complete.
    if (!soloPlaybackActive) stopNotationPlayback();
    melodyPlayEnabled = false;
    chordPlayEnabled = false;
    simBtn.disabled = false;
    loopRegion = null;
    setStatus(transcriptionData ? 'done' : 'idle');
    if (!transcriptionData) {
      transcriptionStatus.textContent = 'Agent failed';
    }
  };
});
