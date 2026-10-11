import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
const math = require('../docs/trees/model.js');
const story = require('../docs/trees/story.js');
const { stump1D, grown1D, second1D, stumpX, stumpY, conditional } = math.examples;
const intervalValues = segments => segments.map(({ lo, hi, value }) => [lo, hi, value]);
const area = cell => cell.bounds.reduce((size, [lo, hi]) => size * (hi - lo), 1);
const interaction = predict => predict([1, 1]) - predict([-1, 1]) - predict([1, -1]) + predict([-1, -1]);

test('threshold equality follows the left branch at every depth', () => {
  assert.equal(math.evaluate(grown1D, [-0.5]), 0);
  assert.equal(math.evaluate(grown1D, [-0.5 + Number.EPSILON]), 1);
  assert.equal(math.evaluate(grown1D, [0.75]), 1);
  assert.equal(math.evaluate(grown1D, [0.75 + Number.EPSILON]), 3);
  assert.equal(math.evaluate(conditional, [0, 1]), 0);
  assert.equal(math.evaluate(conditional, [1, 0]), 0);
  assert.equal(math.evaluate(conditional, [Number.EPSILON, Number.EPSILON]), 2);
  assert.equal(math.leafAt(grown1D, [0.75]).id, 'g1-middle');
  assert.deepEqual(math.pathNodes(grown1D, [0.75]).map(node => node.id), ['g1-root', 'g1-right', 'g1-middle']);
});

test('a conditional split produces three rectangles and stays inside its parent', () => {
  const cells = math.regions(conditional);
  assert.deepEqual(cells, [
    { bounds: [[-2, 0], [-2, 2]], value: 0, leafId: 'c-left', path: ['c-root', 'c-left'] },
    { bounds: [[0, 2], [-2, 0]], value: 0, leafId: 'c-low', path: ['c-root', 'c-right', 'c-low'] },
    { bounds: [[0, 2], [0, 2]], value: 2, leafId: 'c-high', path: ['c-root', 'c-right', 'c-high'] }
  ]);
  assert.equal(cells.reduce((sum, cell) => sum + area(cell), 0), 16);
  // The x1 stump introduces no new line: its split already exists. A global
  // grid implementation would incorrectly divide the left-hand rectangle.
  const overlay = math.refinement([stumpX, conditional]);
  assert.equal(overlay.length, 3);
  assert.deepEqual(overlay.map(cell => [cell.bounds, cell.value, cell.contributions]), [
    [[[-2, 0], [-2, 2]], 0, [0, 0]],
    [[[0, 2], [-2, 0]], 1, [1, 0]],
    [[[0, 2], [0, 2]], 3, [1, 2]]
  ]);
});

test('crossing stumps make four cells whose values obey an additive restriction', () => {
  const cells = math.refinement([stumpX, stumpY]);
  assert.equal(cells.length, 4);
  assert.equal(cells.reduce((sum, cell) => sum + area(cell), 0), 16);
  const sum = point => math.evaluate(stumpX, point) + math.evaluate(stumpY, point);
  assert.deepEqual([sum([-1, -1]), sum([1, -1]), sum([-1, 1]), sum([1, 1])], [0, 1, 2, 3]);
  assert.equal(interaction(sum), 0);
  assert.equal(interaction(point => math.evaluate(conditional, point)), 2);
  // Check several points strictly inside every cell, independently of the
  // clipping algorithm, and check that those interiors do not overlap.
  for (const cell of cells) {
    for (const fraction of [0.1, 0.5, 0.9]) {
      const point = cell.bounds.map(([lo, hi]) => lo + fraction * (hi - lo));
      assert.equal(cell.value, sum(point));
      assert.deepEqual(cell.contributions, [math.evaluate(stumpX, point), math.evaluate(stumpY, point)]);
      assert.equal(cells.filter(other => other.bounds.every(([lo, hi], j) => point[j] > lo && point[j] < hi)).length, 1);
    }
  }
});

test('a sum of two one-dimensional stumps equals a grown tree, including at both jumps', () => {
  const expected = [[-2, -0.5, 0], [-0.5, 0.75, 1], [0.75, 2, 3]];
  assert.deepEqual(intervalValues(math.segments1D(grown1D)), expected);
  assert.deepEqual(intervalValues(math.sumSegments1D([stump1D, second1D])), expected);
  for (const x of [-2, -0.5, -0.5 + Number.EPSILON, 0, 0.75, 0.75 + Number.EPSILON, 2]) {
    assert.equal(math.evaluate(stump1D, [x]) + math.evaluate(second1D, [x]), math.evaluate(grown1D, [x]));
  }
});

test('opposite jumps cancel while the common refinement retains component boundaries', () => {
  const opposite = {
    id: 'opposite', feature: 0, threshold: -0.5,
    left: { id: 'opposite-left', value: 0 }, right: { id: 'opposite-right', value: -1 }
  };
  assert.deepEqual(intervalValues(math.sumSegments1D([stump1D, opposite])), [[-2, -0.5, 0], [-0.5, 2, 0]]);
  for (const x of [-2, -0.5, -0.5 + Number.EPSILON, 2]) {
    assert.equal(math.evaluate(stump1D, [x]) + math.evaluate(opposite, [x]), 0);
  }
  assert.deepEqual(intervalValues(math.sumSegments1D([])), [[-2, 2, 0]]);
});

test('clipping omits zero-width drawing regions and preserves inputs', () => {
  const bounds = [[0, 1], [-1, 1]];
  const before = JSON.stringify(conditional);
  const cells = math.regions(conditional, bounds);
  assert.equal(cells.length, 2);
  assert.equal(cells.reduce((sum, cell) => sum + area(cell), 0), 2);
  // Membership at a plot edge remains a prediction question, even when the
  // corresponding leaf has no positive area within the visible domain.
  assert.equal(math.leafAt(conditional, [0, 1]).id, 'c-left');
  cells[0].bounds[0][0] = -10;
  assert.deepEqual(bounds, [[0, 1], [-1, 1]]);
  assert.equal(JSON.stringify(conditional), before);
  assert.deepEqual(intervalValues(math.segments1D(grown1D, 0.8, 1.5)), [[0.8, 1.5, 3]]);
  assert.throws(() => math.evaluate(stumpY, [1]), RangeError);
  assert.throws(() => math.segments1D(stumpY), RangeError);
  assert.throws(() => math.regions(stumpX, [[2, -2]]), RangeError);
});

const attributes = tag => Object.fromEntries(Array.from(tag.matchAll(/([\w-]+)="([^"]*)"/g), match => [match[1], match[2]]));

test('all eight scenes render finite geometry at both layouts and interaction limits', () => {
  const states = [
    { x: -2, y: -2, addition: 0, interaction: false },
    { x: 2, y: 2, addition: 1, interaction: true },
    { x: -0.5, y: 0, addition: 0.35, interaction: false },
    { x: 0.75, y: 0, addition: 0, interaction: true },
    { x: 0, y: 0, addition: 1, interaction: false }
  ];
  for (const compact of [false, true]) {
    for (let scene = 0; scene < 8; scene++) {
      for (const state of states) {
        const svg = story.renderScene(scene, state, compact);
        assert.doesNotMatch(svg, /NaN|Infinity|undefined/);
        assert.match(svg, compact ? /viewBox="0 0 420 280"/ : /viewBox="0 0 780 510"/);
        assert.match(svg, /<title id="scene-title">[^<]+<\/title>/);
        assert.match(svg, /<desc id="scene-description">[^<]+<\/desc>/);
        for (const [, value] of svg.matchAll(/\b(?:x|y|x1|y1|x2|y2|cx|cy|width|height|r)="([^"]+)"/g)) {
          assert.ok(Number.isFinite(Number(value)), `Scene ${scene} contains invalid geometry: ${value}`);
        }
        for (const tag of svg.match(/<(?:rect|circle)\b[^>]*>/g) || []) {
          const attr = attributes(tag);
          for (const key of ['width', 'height', 'r']) {
            if (key in attr) assert.ok(Number(attr[key]) > 0, `Scene ${scene} has nonpositive ${key}`);
          }
        }
      }
    }
  }
});

test('the rendered conditional boundary ends at its parent split in both layouts', () => {
  assert.deepEqual(story.cuts(conditional), [
    { feature: 0, threshold: 0, bounds: [[-2, 2], [-2, 2]], depth: 0 },
    { feature: 1, threshold: 0, bounds: [[0, 2], [-2, 2]], depth: 1 }
  ]);
  for (const compact of [false, true]) {
    const svg = story.renderScene(4, {}, compact);
    const cuts = (svg.match(/<line\b[^>]*>/g) || []).map(attributes)
      .filter(attr => attr.class?.includes('partition-cut'));
    assert.equal(cuts.length, 2);
    const horizontal = cuts.find(attr => attr.y1 === attr.y2);
    assert.ok(horizontal);
    const plot = (svg.match(/<rect\b[^>]*>/g) || []).map(attributes)
      .find(attr => attr['data-plot'] === '2');
    assert.ok(plot);
    const [x, y, width, height] = plot['data-box'].split(',').map(Number);
    assert.deepEqual([horizontal.x1, horizontal.y1, horizontal.x2, horizontal.y2].map(Number),
      [x + width / 2, y + height / 2, x + width, y + height / 2]);
  }
});

test('zero contribution leaves no apparent hole at the cancelled step boundary', () => {
  for (const addition of [0, 1]) {
    const svg = story.renderScene(2, { addition, x: -1 });
    const plots = (svg.match(/<rect\b[^>]*>/g) || []).map(attributes)
      .filter(attr => attr['data-plot'] === '1');
    const circles = (svg.match(/<circle\b[^>]*>/g) || []).map(attributes);
    // At x = .75, the second plot and the sum have a real jump only when
    // the second tree contributes. A closed marker remains in either case.
    for (const plot of plots.slice(1)) {
      const [x, y, width, height] = plot['data-box'].split(',').map(Number);
      const boundaryX = x + (0.75 + 2) / 4 * width;
      const markers = circles.filter(circle => Number(circle.cx) === boundaryX &&
        Number(circle.cy) >= y && Number(circle.cy) <= y + height);
      assert.equal(markers.filter(circle => circle.fill === 'white').length, addition);
      assert.equal(markers.filter(circle => circle.fill !== 'white').length, 1);
    }
  }
});

test('readouts honor threshold membership, weighted addition, and conditional interactions', () => {
  const base = { x: 1, y: 1, addition: 0.25, interaction: false };
  assert.equal(story.readout(0, { ...base, x: -0.5 }), 'x = −0.5  →  prediction = 0');
  assert.equal(story.readout(1, { ...base, x: 0.75 }), 'x = 0.75  →  prediction = 1');
  assert.equal(story.readout(2, base), '1 + 0.25 × 2 = 1.5');
  assert.equal(story.readout(3, { ...base, x: 0 }), '(x₁, x₂) = (0, 1)  →  T₁(x) = 0');
  assert.equal(story.readout(4, { ...base, y: 0 }), '(x₁, x₂) = (1, 0)  →  U(x) = 0');
  assert.equal(story.readout(5, base), 'T₁ + λT₂ = 1 + 0.25 × 2 = 1.5');
  assert.equal(story.readout(6, base), '1 + 2 = 3 · x₁ jump: 1 below, 1 above');
  assert.equal(story.readout(6, { ...base, interaction: true }), '1 + 2 + 2 = 5 · x₁ jump: 1 below, 3 above');
  assert.equal(story.readout(6, { ...base, y: 0, interaction: true }), '1 + 0 + 0 = 1 · x₁ jump: 1 below, 3 above');
});

test('the interaction toggle raises only the upper-right ensemble prediction', () => {
  function displayedCells(interaction) {
    const svg = story.renderScene(6, { interaction });
    return Array.from(svg.matchAll(/<text\b[^>]*class="cell-value"[^>]*>([^<]+)<\/text>/g), match => Number(match[1]));
  }
  assert.deepEqual(displayedCells(false), [0, 0, 2, 0, 2, 1, 3]);
  assert.deepEqual(displayedCells(true), [0, 0, 2, 0, 2, 1, 5]);
});
