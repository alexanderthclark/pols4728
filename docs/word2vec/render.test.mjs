import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import vm from 'node:vm';
import {
  vocabulary as vocab, initial, forward, step, interpolateModel,
} from './model.js';

const require = createRequire(import.meta.url);
const bundledKatex = require('./vendor/katex/katex.min.js');
const formulas = new Set();
const katex = {
  renderToString(tex, options) {
    // A missing String.raw / escaped backslash can make KaTeX silently typeset
    // command names as letters, rather than reporting a malformed expression.
    assert.doesNotMatch(tex, /(^|[^\\A-Za-z])(?:cdot|heta|longrightarrow)(?=$|[^A-Za-z])/,
      `An unescaped math command reached KaTeX: ${JSON.stringify(tex)}`);
    assert.doesNotMatch(tex, /[\t\r\n\f\b]/,
      `A JavaScript escape reached a single-line formula: ${JSON.stringify(tex)}`);
    formulas.add(tex);
    return bundledKatex.renderToString(tex, options);
  },
};

// Exercise the production render functions without emulating a browser. The
// DOM wiring starts at markers; only two earlier initializers need removal.
const source = readFileSync(new URL('./app.js', import.meta.url), 'utf8');
assert.ok(source.includes('\nconst markers='), 'The render harness boundary moved.');
const renderSource = source.split('\nconst markers=')[0]
  .replace(/^import[^\n]*\n/m, '')
  .replace(/^const story=document[^\n]*\n/m, '')
  .replace(/^const reduced=matchMedia[^\n]*\n/m, '');
const context = vm.createContext({ katex, vocab, initial, forward, step, interpolateModel });
vm.runInContext(`${renderSource}\nglobalThis.harness={scenes,render,latex,first,corpus,final};`, context);
const { scenes, render, latex, first, corpus, final } = context.harness;
assert.equal(scenes.length, 15);

const attrs = tag => Object.fromEntries(
  [...tag.matchAll(/([\w-]+)="([^"]*)"/g)].map(([, key, value]) => [key, value]),
);
const number = (values, key) => {
  const value = Number(values[key]);
  assert.ok(Number.isFinite(value), `Non-finite SVG attribute ${key}=${values[key]}`);
  return value;
};
const within = (x, y, width, height, label) => {
  assert.ok(x >= 0 && x <= width && y >= 0 && y <= height,
    `${label}: (${x}, ${y}) lies outside 0..${width}, 0..${height}`);
};

function checkPathEndpoints(d, width, height, label) {
  // The lesson uses M, L, H, V, and A for diagram paths. This checks actual
  // path positions, not negative relative displacements in bracket outlines.
  const tokens = d.match(/[A-Za-z]|[-+]?(?:\d*\.\d+|\d+)(?:e[-+]?\d+)?/gi) || [];
  let x = 0, y = 0, command;
  const arity = { M: 2, L: 2, H: 1, V: 1, A: 7 };
  for (let index = 0; index < tokens.length;) {
    if (/^[A-Za-z]$/.test(tokens[index])) command = tokens[index++];
    const upper = command?.toUpperCase();
    assert.ok(upper in arity, `Unsupported diagram path command ${command}`);
    const values = tokens.slice(index, index + arity[upper]).map(Number);
    assert.equal(values.length, arity[upper]);
    assert.ok(values.every(Number.isFinite));
    index += arity[upper];
    const relative = command === command.toLowerCase();
    if (upper === 'M' || upper === 'L') {
      x = values[0] + (relative ? x : 0);
      y = values[1] + (relative ? y : 0);
    } else if (upper === 'H') x = values[0] + (relative ? x : 0);
    else if (upper === 'V') y = values[0] + (relative ? y : 0);
    else {
      assert.ok(values[0] >= 0 && values[1] >= 0, 'Arc radii must be nonnegative.');
      x = values[5] + (relative ? x : 0);
      y = values[6] + (relative ? y : 0);
    }
    within(x, y, width, height, label);
  }
}

function checkGeometry(markup, narrow, label) {
  const width = narrow ? 400 : 1000, height = narrow ? 450 : 420;
  assert.doesNotMatch(markup, /(?:NaN|Infinity|undefined)/, label);
  // Math is laid out inside centered foreignObjects. Their padded left margins
  // may extend past the viewport even when the actual label is comfortably in it.
  for (const [tag] of markup.matchAll(/<foreignObject\b[^>]*>/g)) {
    const a = attrs(tag);
    within(number(a, 'x') + number(a, 'width') / 2,
      number(a, 'y') + number(a, 'height') / 2, width, height, `${label} math center`);
  }
  // Ignore KaTeX's own internal SVG paths; they have their own coordinate system.
  const diagram = markup.replace(/<foreignObject\b[\s\S]*?<\/foreignObject>/g, '');
  for (const [tag, kind] of diagram.matchAll(/<(line|rect|text|path)\b[^>]*>/g)) {
    const a = attrs(tag);
    if (kind === 'line') {
      within(number(a, 'x1'), number(a, 'y1'), width, height, `${label} line start`);
      within(number(a, 'x2'), number(a, 'y2'), width, height, `${label} line end`);
    } else if (kind === 'rect') {
      const x = number(a, 'x'), y = number(a, 'y');
      const w = number(a, 'width'), h = number(a, 'height');
      assert.ok(w >= 0 && h >= 0, `${label} negative rectangle dimension`);
      within(x, y, width, height, `${label} rectangle start`);
      within(x + w, y + h, width, height, `${label} rectangle end`);
    } else if (kind === 'text') {
      within(number(a, 'x'), number(a, 'y'), width, height, `${label} text anchor`);
    } else checkPathEndpoints(a.d, width, height, `${label} path endpoint`);
  }
}

let renderedStates = 0;
for (const [i, scene] of scenes.entries()) {
  assert.doesNotThrow(() => latex(scene.eq), `Scene ${i + 1} equation`);
  for (const narrow of [false, true]) {
    for (const t of [0, 0.25, 0.5, 1]) {
      const label = `scene ${i + 1}, phase ${t}, ${narrow ? 'narrow' : 'wide'}`;
      const markup = render(i, t, narrow);
      assert.ok(markup.length > 0, label);
      checkGeometry(markup, narrow, label);
      renderedStates += 1;
    }
  }
}

// Equations outside the scroll stage must use the same working math renderer.
const html = readFileSync(new URL('./index.html', import.meta.url), 'utf8');
for (const [, equation] of html.matchAll(/data-math="([^"]*)"/g)) latex(equation);

// Test the numerical/visual integration, rather than reimplementing SGD here.
const annotations = markup => [...markup.matchAll(/<annotation encoding="application\/x-tex">([\s\S]*?)<\/annotation>/g)].map(match => match[1]);
for (const narrow of [false, true]) {
  const before = render(9, 0, narrow), after = render(9, 1, narrow);
  assert.notEqual(before, after, 'The update scene must change with scroll phase.');
  const math = annotations(after);
  const updatedH = forward(first).h;
  assert.ok(math.includes(updatedH[0].toFixed(2)) && math.includes(updatedH[1].toFixed(2)),
    'The update scene must display the model’s updated input column.');
  assert.ok(after.includes(forward(first).p[2].toFixed(3)),
    'The update scene must display the recomputed probability.');
  assert.notEqual(render(10, 0, narrow), render(10, 1, narrow),
    'The corpus scene must evolve across its complete sequence.');
  assert.notEqual(render(12, 0, narrow), render(12, 1, narrow),
    'The similarity scene must show its angle change.');
  assert.notEqual(render(13, 0, narrow), render(13, 1, narrow),
    'The analogy scene must translate its difference vector.');
}
assert.ok(corpus.length > 2);
assert.ok(final.Win.flat().some((value, index) => value !== initial.Win.flat()[index]));
assert.ok([...formulas].some(tex => tex === String.raw`\cdot`));
assert.ok([...formulas].some(tex => tex === String.raw`\theta`));
assert.ok([...formulas].some(tex => tex.includes(String.raw`\longrightarrow`)));

console.log(`Render checks passed: ${renderedStates} scene/phase/layout states, ${formulas.size} distinct KaTeX formulas, diagram bounds, scroll-dependent geometry, and exact model integration.`);
console.log('These checks do not replace browser review of scrolling, text extents, keyboard access, or reduced motion.');
