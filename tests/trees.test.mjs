import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
const math = require('../docs/trees/model.js');
const { stump1D, grown1D, second1D, stumpX, stumpY, recursive2D, matched2D, conditional } = math.examples;
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

test('the same two-split budget creates three local regions or four overlay cells', () => {
  assert.equal(math.splitCount(recursive2D), 2);
  assert.equal(math.splitCount(stumpX) + math.splitCount(stumpY), 2);
  const recursive = math.regions(recursive2D);
  const ensemble = math.refinement([stumpX, stumpY]);
  assert.equal(recursive.length, 3);
  assert.equal(ensemble.length, 4);
  assert.deepEqual(recursive.map(({ bounds, value }) => ({ bounds, value })), [
    { bounds: [[-2, 0], [-2, 2]], value: 0 },
    { bounds: [[0, 2], [-2, 0]], value: 1 },
    { bounds: [[0, 2], [0, 2]], value: 3 }
  ]);
  assert.deepEqual([[-1, -1], [1, -1], [-1, 1], [1, 1]].map(point => math.evaluate(recursive2D, point)), [0, 1, 0, 3]);
  // The additional ensemble cell is meaningful: it has prediction 2, which
  // differs from all three predictions of this two-split recursive tree.
  assert.deepEqual(ensemble.map(cell => cell.value).sort((a, b) => a - b), [0, 1, 2, 3]);
});

test('one tree with three splits matches the two stumps across quadrants and boundaries', () => {
  assert.equal(math.splitCount(matched2D), 3);
  assert.equal(math.regions(matched2D).length, 4);
  const coordinates = [-2, -1, 0, Number.EPSILON, 1, 2];
  for (const x of coordinates) {
    for (const y of coordinates) {
      assert.equal(math.evaluate(matched2D, [x, y]), math.evaluate(stumpX, [x, y]) + math.evaluate(stumpY, [x, y]));
    }
  }
  assert.equal(interaction(point => math.evaluate(matched2D, point)), 0);
  assert.equal(interaction(point => math.evaluate(recursive2D, point)), 2);
});

test('a sum of two one-dimensional stumps equals a grown tree, including at both jumps', () => {
  // The 2-D three-versus-four comparison is not a universal region-count
  // advantage: here both constructions use two splits and three intervals.
  assert.equal(math.splitCount(grown1D), 2);
  assert.equal(math.splitCount(stump1D) + math.splitCount(second1D), 2);
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
