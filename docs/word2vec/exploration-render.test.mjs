import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import vm from 'node:vm';
import * as math from './math.js';
import * as projection from './projection.js';

const require = createRequire(import.meta.url);
const bundledKatex = require('./vendor/katex/katex.min.js');
const formulas = new Set();
const katex = {
  renderToString(tex, options) {
    assert.doesNotMatch(tex, /(^|[^\\A-Za-z])(?:mathrm|mathsf|cdot|heta|longrightarrow|lVert|rVert)(?=$|[^A-Za-z])/,
      `An unescaped math command reached KaTeX: ${JSON.stringify(tex)}`);
    assert.doesNotMatch(tex, /[\t\r\n\f\b]/,
      `A JavaScript control escape reached a formula: ${JSON.stringify(tex)}`);
    const markup = bundledKatex.renderToString(tex, options);
    formulas.add(tex);
    return markup;
  },
};

// Load production rendering directly, stopping before DOM initialization. Both
// imports are supplied as real module exports; no browser behavior is emulated.
const source = readFileSync(new URL('./exploration.js', import.meta.url), 'utf8');
assert.ok(source.includes('\nconst markers='), 'The render harness boundary moved.');
const renderSource = source.split('\nconst markers=')[0].replace(/^import[^\n]*\n/gm, '');
const context = vm.createContext({ ...math, ...projection, V: math.vectors, katex });
vm.runInContext(`${renderSource}\nglobalThis.harness={scenes,render,latex,qWater,qRaw,qUnit,ranking};`, context);
const { scenes, render, latex, qWater, qRaw, qUnit, ranking } = context.harness;
assert.equal(scenes.length, 10);
const close = (actual, expected, tolerance = 1e-12) => {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);
};

// Independently reconstruct the displayed queries from raw fixture values.
const manualUnit = vector => {
  const length = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
  return vector.map(value => value / length);
};
const manualSum = (a, b) => a.map((value, i) => value + b[i]);
const manualCosine = (a, b) => {
  const ua = manualUnit(a), ub = manualUnit(b);
  return ua.reduce((sum, value, i) => sum + value * ub[i], 0);
};
const { water, dirt, rain } = math.vectors;
for (const [actual, expected] of [
  [qWater, manualSum(manualUnit(water), manualUnit(dirt))],
  [qRaw, [2.6, 1.6]],
  [qUnit, manualSum(manualUnit(dirt), manualUnit(rain))],
]) actual.forEach((value, i) => close(value, expected[i]));
assert.equal(ranking[0].word, 'puddle');
ranking.forEach(({ word, score }) => close(score, manualCosine(math.vectors[word], qWater)));
assert.ok(ranking.every(({ word }) => word !== 'water' && word !== 'dirt'));

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

function checkPath(d, width, height, label) {
  const tokens = d.match(/[A-Za-z]|[-+]?(?:\d*\.\d+|\d+)(?:e[-+]?\d+)?/gi) || [];
  let x = 0, y = 0, startX = 0, startY = 0, command;
  const arity = { M: 2, L: 2, H: 1, V: 1, A: 7, Z: 0 };
  for (let index = 0; index < tokens.length;) {
    if (/^[A-Za-z]$/.test(tokens[index])) command = tokens[index++];
    const upper = command?.toUpperCase();
    assert.ok(upper in arity, `Unsupported diagram path command ${command}`);
    if (upper === 'Z') {
      x = startX; y = startY;
      within(x, y, width, height, label);
      continue;
    }
    const values = tokens.slice(index, index + arity[upper]).map(Number);
    assert.equal(values.length, arity[upper]);
    assert.ok(values.every(Number.isFinite));
    index += arity[upper];
    const relative = command === command.toLowerCase();
    if (upper === 'M' || upper === 'L') {
      x = values[0] + (relative ? x : 0);
      y = values[1] + (relative ? y : 0);
      if (upper === 'M') { startX = x; startY = y; }
    } else if (upper === 'H') x = values[0] + (relative ? x : 0);
    else if (upper === 'V') y = values[0] + (relative ? y : 0);
    else {
      assert.ok(values[0] >= 0 && values[1] >= 0);
      x = values[5] + (relative ? x : 0);
      y = values[6] + (relative ? y : 0);
    }
    within(x, y, width, height, label);
  }
}

function checkGeometry(markup, narrow, label) {
  const width = narrow ? 400 : 1000, height = narrow ? 460 : 440;
  assert.doesNotMatch(markup, /(?:NaN|Infinity|undefined)/, label);
  // Centered math objects contain generous padding; check label centers rather
  // than treating their harmless outside padding as actual off-screen text.
  for (const [tag] of markup.matchAll(/<foreignObject\b[^>]*>/g)) {
    const a = attrs(tag);
    within(number(a, 'x') + number(a, 'width') / 2,
      number(a, 'y') + number(a, 'height') / 2, width, height, `${label} math label`);
  }
  const diagram = markup.replace(/<foreignObject\b[\s\S]*?<\/foreignObject>/g, '');
  for (const [tag, kind] of diagram.matchAll(/<(line|rect|circle|text|path)\b[^>]*>/g)) {
    const a = attrs(tag);
    if (kind === 'line') {
      within(number(a, 'x1'), number(a, 'y1'), width, height, `${label} line start`);
      within(number(a, 'x2'), number(a, 'y2'), width, height, `${label} line end`);
    } else if (kind === 'rect') {
      const x = number(a, 'x'), y = number(a, 'y'), w = number(a, 'width'), h = number(a, 'height');
      assert.ok(w >= 0 && h >= 0, `${label} negative rectangle dimension`);
      within(x, y, width, height, `${label} rectangle start`);
      within(x + w, y + h, width, height, `${label} rectangle end`);
    } else if (kind === 'circle') {
      const x = number(a, 'cx'), y = number(a, 'cy'), r = number(a, 'r');
      assert.ok(r >= 0, `${label} negative circle radius`);
      within(x - r, y - r, width, height, `${label} circle start`);
      within(x + r, y + r, width, height, `${label} circle end`);
    } else if (kind === 'text') within(number(a, 'x'), number(a, 'y'), width, height, `${label} text anchor`);
    else checkPath(a.d, width, height, `${label} path`);
  }
}

const errors = [];
let states = 0;
for (const [i, scene] of scenes.entries()) {
  try { latex(scene.eq); } catch (error) { errors.push(`scene ${i + 1} equation: ${error.message}`); }
  for (const narrow of [false, true]) {
    for (const t of [0, 0.25, 0.5, 1]) {
      const label = `scene ${i + 1}, phase ${t}, ${narrow ? 'narrow' : 'wide'}`;
      try {
        const markup = render(i, t, narrow);
        assert.ok(markup.length > 0);
        checkGeometry(markup, narrow, label);
      } catch (error) { errors.push(`${label}: ${error.message}`); }
      states += 1;
    }
  }
}
assert.deepEqual(errors, [], `Render failures:\n${errors.join('\n')}`);

for (const narrow of [false, true]) {
  for (const scene of [1, 2, 3, 4, 5, 7, 8, 9]) {
    assert.notEqual(render(scene, 0, narrow), render(scene, 1, narrow),
      `Scene ${scene + 1} needs scroll-dependent diagram changes.`);
  }
  const rankedMarkup = render(3, 1, narrow);
  assert.ok(rankedMarkup.includes(ranking[0].word));
  assert.ok(rankedMarkup.includes(ranking[0].score.toFixed(3)));
  const productMarkup = render(6, 1, narrow);
  const row = math.rankCosmul([water, dirt], [], math.candidates, ['water', 'dirt'])[0];
  assert.ok(productMarkup.includes(row.word) && productMarkup.includes(row.score.toFixed(3)));
}

function plottedPositions(markup) {
  const positions = [];
  for (const [tag, kind] of markup.matchAll(/<(circle|rect|path)\b[^>]*>/g)) {
    const a = attrs(tag);
    if (kind === 'circle') positions.push([number(a, 'cx'), number(a, 'cy')]);
    else if (kind === 'rect') positions.push([
      number(a, 'x') + number(a, 'width') / 2,
      number(a, 'y') + number(a, 'height') / 2,
    ]);
    else {
      // The third group uses triangles whose first point is six units above
      // the fitted position; this is a marker convention, not a model formula.
      const match = a.d.match(/^M([-\d.e+]+) ([-\d.e+]+)l6 11h-12z$/);
      assert.ok(match, 'Unexpected synthetic-map marker.');
      positions.push([Number(match[1]), Number(match[2]) + 6]);
    }
  }
  return positions;
}

// Verify the actual SVG positions preserve all map distances during rotation,
// and that the preceding reveal finishes at that same fitted arrangement.
for (const narrow of [false, true]) {
  const revealEnd = plottedPositions(render(8, 1, narrow));
  const rotationStart = plottedPositions(render(9, 0, narrow));
  const rotationEnd = plottedPositions(render(9, 1, narrow));
  assert.equal(rotationStart.length, projection.points.length);
  assert.equal(revealEnd.length, rotationStart.length);
  revealEnd.forEach((point, i) => point.forEach((value, j) => close(value, rotationStart[i][j], 1e-10)));
  assert.equal(rotationEnd.length, rotationStart.length);
  for (let i = 0; i < rotationStart.length; i += 1) {
    for (let j = i + 1; j < rotationStart.length; j += 1) {
      const distance = values => Math.hypot(values[i][0] - values[j][0], values[i][1] - values[j][1]);
      close(distance(rotationStart), distance(rotationEnd), 1e-10);
    }
  }
}

// Projection positions and declared source dimensions must match the imported
// fit. No rerun or synthetic reimplementation of t-SNE is needed for this check.
assert.equal(projection.points.length, projection.projectionMetadata.pointCount);
assert.equal(projection.projectionMetadata.synthetic, true);
for (const point of projection.points) {
  assert.equal(point.vector.length, projection.projectionMetadata.sourceDimensions);
  assert.equal(point.position.length, 2);
  assert.ok(point.position.every(Number.isFinite));
  close(Math.sqrt(point.vector.reduce((sum, value) => sum + value * value, 0)), 1);
}

console.log(`Companion render checks passed: ${states} scene/phase/layout states, ${formulas.size} KaTeX formulas, independent query values, exact score displays, geometry bounds, and projection integration.`);
console.log('Browser scrolling, glyph extents, keyboard navigation, and reduced motion remain separate visual checks.');
