import fs from 'node:fs';

const source = fs.readFileSync(new URL('../main.js', import.meta.url), 'utf8');
const start = source.indexOf('function playOrnamentedNote');
const end = source.indexOf('\nfunction playNotation', start);
if (start < 0 || end < 0) throw new Error('playOrnamentedNote contract not found');
const body = source.slice(start, end);
if (body.includes("HT_STATE.mode !== 'songlike' && seg.ornament === 'kan') return")) {
  throw new Error('Faithful mode still drops kan events');
}
if (!body.includes("if (HT_STATE.mode !== 'songlike' || !seg.ornament)")) {
  throw new Error('Faithful one-to-one fallback is missing');
}
if (!source.includes('Tone.Time(duration).toSeconds()')) {
  throw new Error('Tone duration is not used for visual release timing');
}
if (!source.includes('(data.melody.melody || []).find(s => s.source_model)')) {
  throw new Error('transcriber provenance fallback is missing');
}
console.log('faithful renderer contract passed');
