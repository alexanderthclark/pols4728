import assert from 'node:assert/strict';
import test from 'node:test';
import { electoralResults } from '../docs/shapley/electoral-college.mjs';
import { ELECTORAL_TILES } from '../docs/shapley/electoral-tile-data.mjs';
import { prepareElectoralMap } from '../docs/shapley/electoral-map-geometry.mjs';

const EPSILON = 1e-8;
const expected = new Map(electoralResults.map(state => [state.abbreviation, state]));

function closeTo(actual, target, tolerance = EPSILON, context = '') {
  assert.ok(Math.abs(actual - target) <= tolerance,
    `${context}: ${actual} should equal ${target} within ${tolerance}`);
}

// These checks derive counts and topology from cells, independently of any
// metadata or perimeter generation used by the visualization.
function assertFourConnected(cells, context) {
  const occupied = new Set(cells.map(([x, y]) => `${x},${y}`));
  const stack = [cells[0]];
  const visited = new Set();
  while (stack.length) {
    const [x, y] = stack.pop();
    const key = `${x},${y}`;
    if (!occupied.has(key) || visited.has(key)) continue;
    visited.add(key);
    stack.push([x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]);
  }
  assert.equal(visited.size, occupied.size, `${context}: state cells are edge-connected`);
}

function assertNoGridHoles(cells, context) {
  const occupied = new Set(cells.map(([x, y]) => `${x},${y}`));
  const xs = cells.map(cell => cell[0]), ys = cells.map(cell => cell[1]);
  const left = Math.min(...xs) - 1, right = Math.max(...xs) + 1;
  const top = Math.min(...ys) - 1, bottom = Math.max(...ys) + 1;
  const stack = [[left, top]], exterior = new Set();
  while (stack.length) {
    const [x, y] = stack.pop();
    if (x < left || x > right || y < top || y > bottom) continue;
    const key = `${x},${y}`;
    if (occupied.has(key) || exterior.has(key)) continue;
    exterior.add(key);
    stack.push([x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]);
  }
  for (let x = left; x <= right; x++) {
    for (let y = top; y <= bottom; y++) {
      const key = `${x},${y}`;
      assert.ok(occupied.has(key) || exterior.has(key), `${context}: enclosed empty cell at ${key}`);
    }
  }
}

function rectangleOverlap(a, b) {
  return Math.max(0, Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x)) *
    Math.max(0, Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y));
}

function ringArea(ring) {
  let twice = 0;
  for (let i = 0; i < ring.length; i++) {
    const a = ring[i], b = ring[(i + 1) % ring.length];
    twice += a[0] * b[1] - b[0] * a[1];
  }
  return Math.abs(twice / 2);
}

// Orthogonal polygon intersection from independent scan-line integration.
// Boundary contact has zero area and remains legal between neighboring states.
function polygonOverlap(a, b) {
  const ys = [...new Set([...a, ...b].flat().map(point => point[1]))].sort((x, y) => x - y);
  const intervals = (rings, y) => {
    const xs = [];
    for (const ring of rings) {
      for (let i = 0; i < ring.length; i++) {
        const p = ring[i], q = ring[(i + 1) % ring.length];
        if ((p[1] > y) !== (q[1] > y)) {
          xs.push(p[0] + (y - p[1]) * (q[0] - p[0]) / (q[1] - p[1]));
        }
      }
    }
    xs.sort((x, y) => x - y);
    assert.equal(xs.length % 2, 0);
    return Array.from({ length: xs.length / 2 }, (_, i) => [xs[2 * i], xs[2 * i + 1]]);
  };
  let area = 0;
  for (let i = 1; i < ys.length; i++) {
    const low = ys[i - 1], high = ys[i], y = (low + high) / 2;
    const aa = intervals(a, y), bb = intervals(b, y);
    for (const [a0, a1] of aa) for (const [b0, b1] of bb) {
      area += Math.max(0, Math.min(a1, b1) - Math.max(a0, b0)) * (high - low);
    }
  }
  return area;
}

function assertRectanglesConnected(rectangles, context) {
  const positive = rectangles.filter(rect => rect.width * rect.height > EPSILON);
  const neighbors = (a, b) => {
    const sharedVertical = Math.abs(a.x + a.width - b.x) < EPSILON || Math.abs(b.x + b.width - a.x) < EPSILON;
    const sharedHorizontal = Math.abs(a.y + a.height - b.y) < EPSILON || Math.abs(b.y + b.height - a.y) < EPSILON;
    const xOverlap = Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x);
    const yOverlap = Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y);
    return (sharedVertical && yOverlap > EPSILON) || (sharedHorizontal && xOverlap > EPSILON);
  };
  const visited = new Set(), stack = [0];
  while (stack.length) {
    const index = stack.pop();
    if (visited.has(index)) continue;
    visited.add(index);
    positive.forEach((rect, next) => {
      if (!visited.has(next) && neighbors(positive[index], rect)) stack.push(next);
    });
  }
  assert.equal(visited.size, positive.length, `${context}: rectangles remain edge-connected`);
}

function assertNoRectangleOverlaps(rectangles, context) {
  for (let i = 0; i < rectangles.length; i++) {
    for (let j = 0; j < i; j++) {
      assert.ok(rectangleOverlap(rectangles[i], rectangles[j]) < EPSILON,
        `${context}: overlapping rectangles ${rectangles[i].state}:${i} and ${rectangles[j].state}:${j}`);
    }
  }
}

const rectanglesFor = (states, key) => states.flatMap(state => state[key].map(rectangle => ({ ...rectangle, state: state.abbreviation })));
const interpolateRectangle = (a, b, t) => Object.fromEntries(['x', 'y', 'width', 'height'].map(key => [key, a[key] + (b[key] - a[key]) * t]));

test('electoral tiles contain exactly one equal square for each of the 538 actual electors', () => {
  assert.equal(ELECTORAL_TILES.states.length, 51);
  assert.equal(new Set(ELECTORAL_TILES.states.map(state => state.abbreviation)).size, 51);
  const occupied = new Set();
  for (const state of ELECTORAL_TILES.states) {
    const value = expected.get(state.abbreviation);
    assert.ok(value, state.abbreviation);
    assert.equal(state.votes, value.votes, state.abbreviation);
    closeTo(state.adjustedVotes, 538 * value.share, EPSILON, `${state.abbreviation} adjusted vote data`);
    assert.equal(state.cells.length, value.votes, `${state.abbreviation}: actual electoral cells`);
    for (const [x, y] of state.cells) {
      assert.ok(Number.isInteger(x) && Number.isInteger(y), `${state.abbreviation}: integer grid position`);
      assert.ok(x >= 0 && y >= 0 && x < ELECTORAL_TILES.width && y < ELECTORAL_TILES.height,
        `${state.abbreviation}: cell fits grid`);
      const key = `${x},${y}`;
      assert.ok(!occupied.has(key), `${state.abbreviation}: duplicate occupied grid cell ${key}`);
      occupied.add(key);
    }
    assertFourConnected(state.cells, state.abbreviation);
    assertNoGridHoles(state.cells, state.abbreviation);
    const actualSquares = state.rectangles.filter(rect => rect.width * rect.height > EPSILON);
    assert.equal(actualSquares.length, value.votes, `${state.abbreviation}: rendered actual square count`);
    for (const rect of actualSquares) {
      closeTo(rect.width, 1, EPSILON, `${state.abbreviation}: square width`);
      closeTo(rect.height, 1, EPSILON, `${state.abbreviation}: square height`);
      assert.ok(state.cells.some(([x, y]) => Math.abs(rect.x - x) < EPSILON && Math.abs(rect.y - y) < EPSILON),
        `${state.abbreviation}: rendered square matches its listed cell`);
    }
  }
  assert.equal(occupied.size, 538);
  const allRectangles = rectanglesFor(ELECTORAL_TILES.states, 'rectangles');
  assert.equal(allRectangles.filter(rect => rect.width * rect.height < EPSILON).length, 9);
  assertNoRectangleOverlaps(allRectangles, 'actual electoral cells');
});

test('fractional boundary cells encode exact Shapley-adjusted vote areas in a fixed unit', () => {
  let total = 0, fullCount = 0, partialCount = 0;
  for (const state of ELECTORAL_TILES.states) {
    assert.equal(state.rectangles.length, state.adjustedRectangles.length, `${state.abbreviation}: stable animation slots`);
    const target = 538 * expected.get(state.abbreviation).share;
    let area = 0, statePartialCount = 0;
    for (const rect of state.adjustedRectangles) {
      assert.ok(Object.values(rect).every(Number.isFinite), `${state.abbreviation}: finite rectangle`);
      assert.ok(rect.width > 0 && rect.height > 0 && rect.width <= 1 + EPSILON && rect.height <= 1 + EPSILON,
        `${state.abbreviation}: each cell is at most one vote`);
      assert.ok(Math.abs(rect.width - 1) < EPSILON || Math.abs(rect.height - 1) < EPSILON,
        `${state.abbreviation}: fractions clip only one dimension`);
      const cellArea = rect.width * rect.height;
      area += cellArea;
      if (cellArea < 1 - EPSILON) { statePartialCount++; partialCount++; }
      else fullCount++;
    }
    closeTo(area, target, EPSILON, `${state.abbreviation}: sum of rendered adjusted cell areas`);
    assert.equal(statePartialCount, 1, `${state.abbreviation}: exactly one fractional boundary cell`);
    closeTo(state.adjustedRectangles.filter(rect => rect.width * rect.height < 1 - EPSILON)[0].width *
      state.adjustedRectangles.filter(rect => rect.width * rect.height < 1 - EPSILON)[0].height,
      target - Math.floor(target), EPSILON, `${state.abbreviation}: fractional remainder`);
    assertRectanglesConnected(state.adjustedRectangles, state.abbreviation);
    total += area;
  }
  assert.equal(fullCount, 496);
  assert.equal(partialCount, 51);
  closeTo(total, 538, EPSILON, 'national adjusted total');
  assertNoRectangleOverlaps(rectanglesFor(ELECTORAL_TILES.states, 'adjustedRectangles'), 'adjusted cells');
});

test('polyomino perimeters have the same vote areas as their independently counted squares', () => {
  for (const state of ELECTORAL_TILES.states) {
    for (const [key, target] of [['rings', state.votes], ['adjustedRings', 538 * expected.get(state.abbreviation).share]]) {
      assert.ok(state[key].length > 0, `${state.abbreviation}: ${key} exists`);
      for (const ring of state[key]) {
        assert.ok(ring.length >= 3);
        for (let i = 0; i < ring.length; i++) {
          const a = ring[i], b = ring[(i + 1) % ring.length];
          assert.ok(a.length === 2 && a.every(Number.isFinite));
          assert.ok(Math.abs(a[0] - b[0]) < EPSILON || Math.abs(a[1] - b[1]) < EPSILON,
            `${state.abbreviation}: ${key} has square-cell edges`);
        }
      }
      closeTo(state[key].reduce((sum, ring) => sum + ringArea(ring), 0), target, EPSILON,
        `${state.abbreviation}: ${key} perimeter area`);
    }
  }
  for (const key of ['rings', 'adjustedRings']) {
    for (let i = 0; i < ELECTORAL_TILES.states.length; i++) {
      for (let j = 0; j < i; j++) {
        const a = ELECTORAL_TILES.states[i], b = ELECTORAL_TILES.states[j];
        closeTo(polygonOverlap(a[key], b[key]), 0, EPSILON, `${a.abbreviation}/${b.abbreviation}: ${key} overlap`);
      }
    }
  }
});

test('the electoral-to-adjusted cell morph preserves 538 votes and prevents overlap throughout', () => {
  for (let frame = 0; frame <= 40; frame++) {
    const t = frame / 40;
    let total = 0;
    const allRectangles = [];
    for (const state of ELECTORAL_TILES.states) {
      const rectangles = state.rectangles.map((rect, index) => interpolateRectangle(rect, state.adjustedRectangles[index], t));
      const area = rectangles.reduce((sum, rect) => sum + rect.width * rect.height, 0);
      closeTo(area, state.votes + (538 * expected.get(state.abbreviation).share - state.votes) * t,
        EPSILON, `${state.abbreviation}: area at fraction ${t}`);
      allRectangles.push(...rectangles.map(rect => ({ ...rect, state: state.abbreviation })));
      total += area;
    }
    closeTo(total, 538, EPSILON, `national area at fraction ${t}`);
    assertNoRectangleOverlaps(allRectangles, `morph fraction ${t}`);
  }
});

test('the geometry actually rendered uses one fixed area unit for both cartogram phases', () => {
  const prepared = prepareElectoralMap(electoralResults);
  assert.equal(prepared.states.length, 51);
  assert.ok(prepared.unit > 0 && Number.isFinite(prepared.unit));
  const unitArea = prepared.unit ** 2;
  for (const state of prepared.states) {
    const target = expected.get(state.abbreviation);
    assert.ok(target, state.abbreviation);
    assert.equal(state.votes, target.votes);
    closeTo(state.adjustedVotes, 538 * target.share, EPSILON, `${state.abbreviation}: prepared adjusted votes`);
    assert.equal(state.cells.filter(cell => cell.electoral[2] * cell.electoral[3] > EPSILON).length, target.votes,
      `${state.abbreviation}: rendered positive actual cells`);
    for (const [phase, votes] of [['electoral', target.votes], ['adjusted', 538 * target.share]]) {
      const area = state.cells.reduce((sum, cell) => sum + cell[phase][2] * cell[phase][3], 0);
      closeTo(area / unitArea, votes, EPSILON, `${state.abbreviation}: actual ${phase} cell area in vote units`);
      closeTo(state[phase].reduce((sum, ring) => sum + ringArea(ring), 0) / unitArea,
        votes, 1e-7, `${state.abbreviation}: actual ${phase} perimeter area in vote units`);
      for (const cell of state.cells) {
        const [x, y, w, h] = cell[phase];
        assert.ok([x, y, w, h].every(Number.isFinite));
        assert.ok(w >= 0 && h >= 0);
        assert.ok(x >= -EPSILON && y >= -EPSILON && x + w <= prepared.width + EPSILON && y + h <= prepared.height + EPSILON,
          `${state.abbreviation}: ${phase} cell lies within the fixed viewport`);
      }
    }
  }
  for (let frame = 0; frame <= 40; frame++) {
    const t = frame / 40;
    const rectangles = prepared.states.flatMap(state => state.cells.map(cell => {
      const r = cell.electoral.map((value, d) => value + (cell.adjusted[d] - value) * t);
      return { x: r[0], y: r[1], width: r[2], height: r[3], state: state.abbreviation };
    }));
    closeTo(rectangles.reduce((sum, r) => sum + r.width * r.height, 0) / unitArea,
      538, EPSILON, `actual rendered national area at fraction ${t}`);
    assertNoRectangleOverlaps(rectangles, `actual rendered morph fraction ${t}`);
  }
});
