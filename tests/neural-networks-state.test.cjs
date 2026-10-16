'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const M = require('../docs/neural-networks/math.js');
const S = require('../docs/neural-networks/state.js');

function close(actual, expected, tolerance = 1e-10, message = '') {
  assert.ok(Number.isFinite(actual), `Nonfinite value: ${actual}. ${message}`);
  assert.ok(Number.isFinite(expected), `Nonfinite expected value: ${expected}. ${message}`);
  assert.ok(Math.abs(actual - expected) <= tolerance,
    `${actual} differs from ${expected} by more than ${tolerance}. ${message}`);
}

function grid(lo, hi, intervals = 200) {
  return Array.from({ length: intervals + 1 }, (_, i) => lo + (hi - lo) * i / intervals);
}

function allNumbersFinite(value, path = 'frame') {
  if (typeof value === 'number') {
    assert.ok(Number.isFinite(value), `${path} must be finite, received ${value}`);
  } else if (Array.isArray(value)) {
    value.forEach((entry, i) => allNumbersFinite(entry, `${path}[${i}]`));
  } else if (value && typeof value === 'object') {
    Object.entries(value).forEach(([key, entry]) => allNumbersFinite(entry, `${path}.${key}`));
  }
}

function sampledError(model, intervals = 10000) {
  let error = 0;
  for (const x of grid(0, M.TAU, intervals)) {
    error = Math.max(error, Math.abs(M.evaluate(model, x) - Math.sin(x)));
  }
  return error;
}

test('the scroll story contains exactly 16 distinct, valid scenes', () => {
  assert.equal(S.scenes.length, 16);
  assert.equal(new Set(S.scenes).size, 16);
  assert.equal(S.scenes[0], 'affine');
  assert.equal(S.scenes.at(-1), 'universal');
  assert.throws(() => S.frame('missing-scene', 0.5), RangeError);

  for (const scene of S.scenes) {
    for (const progress of [-1, 0, 0.125, 0.5, 0.875, 1, 2, NaN]) {
      const frame = S.frame(scene, progress);
      assert.equal(frame.scene, scene);
      assert.ok(frame.progress >= 0 && frame.progress <= 1);
      assert.ok(frame.caption.length > 0, `${scene} needs a caption`);
      assert.ok(frame.equation.length > 0, `${scene} needs an equation`);
      allNumbersFinite(frame);
      if (frame.model) {
        const inputs = frame.kind === 'sine' ? grid(0, M.TAU) : grid(-2, 2);
        for (const x of inputs) {
          assert.ok(Number.isFinite(M.evaluate(frame.model, x)), `${scene} output at ${x}`);
          allNumbersFinite(M.contributions(frame.model, x), `${scene}.contributions`);
        }
      } else if (frame.kind === 'depth') {
        for (const x of grid(0, 1)) {
          assert.ok(Number.isFinite(M.deepValue(x, frame.depth)));
          allNumbersFinite(M.deepActivations(x, frame.depth), `${scene}.activations`);
        }
      }
    }
  }
});

test('the actual coarse sine curve has five visible joints and width six', () => {
  const frame = S.frame('sine-coarse', 0);
  const model = frame.model;
  assert.equal(frame.width, 6);
  assert.equal(model.width, 6);
  assert.equal(model.knots.length, 7);
  assert.deepEqual(model.knots, [...S.initialKnots]);
  assert.equal(model.neurons[0].baseline, true);
  assert.equal(model.neurons[0].t, -1);
  assert.equal(model.neurons.slice(1).filter(unit => Math.abs(unit.a) > 1e-10).length, 5);

  for (let i = 0; i < model.knots.length; i += 1) {
    close(M.evaluate(model, model.knots[i]), Math.sin(model.knots[i]));
  }
  for (let i = 0; i < model.knots.length - 1; i += 1) {
    const left = model.knots[i], right = model.knots[i + 1];
    for (const fraction of [0.1, 0.37, 0.75]) {
      const x = left + fraction * (right - left);
      close(M.evaluate(model, x), Math.sin(left) + fraction * (Math.sin(right) - Math.sin(left)));
    }
  }
});

test('the three-neuron hill has slopes 0, +1, -1, 0 and continuous scene boundaries', () => {
  const full = S.frame('hill-flat', 1).model;
  assert.equal(full.width, 3);
  const epsilon = 1e-4;
  for (const [x, slope] of [[-1.5, 0], [-0.5, 1], [0.5, -1], [1.5, 0]]) {
    close((M.evaluate(full, x + epsilon) - M.evaluate(full, x - epsilon)) / (2 * epsilon), slope);
  }
  for (const [x, y] of [[-2, 0], [-1, 0], [-0.5, 0.5], [0, 1], [0.5, 0.5], [1, 0], [2, 0]]) {
    close(M.evaluate(full, x), y);
  }
  const boundaries = [
    [S.frame('hill-rise', 1), S.frame('hill-fall', 0)],
    [S.frame('hill-fall', 1), S.frame('hill-flat', 0)],
    [S.frame('hill-flat', 1), S.frame('shallow', 0)],
  ];
  for (const [before, after] of boundaries) {
    for (const x of grid(-2, 2)) close(M.evaluate(before.model, x), M.evaluate(after.model, x));
  }
});

test('depth one, two, three, and five have 1, 3, 7, and 31 genuine joints', () => {
  for (const [scene, depth, jointCount] of [
    ['depth-one', 1, 1], ['depth-two', 2, 3], ['depth-three', 3, 7], ['depth-five', 5, 31],
  ]) {
    const frame = S.frame(scene, 1);
    assert.equal(frame.depth, depth);
    assert.equal(M.deepJoints(depth).length, jointCount);
    const segments = 2 ** depth;
    for (let k = 0; k <= segments; k += 1) {
      close(M.deepValue(k / segments, depth), k % 2);
    }
    for (let k = 0; k < segments; k += 1) {
      const left = (k + 0.2) / segments, right = (k + 0.8) / segments;
      close((M.deepValue(right, depth) - M.deepValue(left, depth)) / (right - left),
        (k % 2 ? -1 : 1) * segments);
      const layers = M.deepActivations(left, depth);
      assert.equal(layers.length, depth);
      assert.ok(layers.every(layer => layer.activations.length === 2));
      close(layers.at(-1).output, M.deepValue(left, depth));
    }
  }
  assert.equal(S.frame('depth-five', 0).depth, 3);
  assert.equal(S.frame('depth-five', 0.5).depth, 4);
  assert.equal(S.frame('depth-five', 1).depth, 5);
});

test('inserting one sine joint starts at the old curve and ends exactly on sine', () => {
  const coarse = S.frame('sine-coarse', 1).model;
  const start = S.frame('sine-joint', 0);
  const end = S.frame('sine-joint', 1);
  assert.equal(start.width, coarse.width + 1);
  assert.equal(end.width, coarse.width + 1);
  assert.equal(start.model.width, coarse.width + 1);
  assert.equal(end.model.width, coarse.width + 1);
  assert.equal(start.x, S.jointX);
  assert.equal(end.x, S.jointX);

  const inserted = end.model.neurons.filter(unit => !coarse.neurons.some(old => Math.abs(old.t - unit.t) < 1e-10));
  assert.equal(inserted.length, 1);
  close(inserted[0].t, S.jointX);
  close(start.model.neurons.find(unit => Math.abs(unit.t - S.jointX) < 1e-10).a, 0);
  assert.ok(Math.abs(M.evaluate(coarse, S.jointX) - Math.sin(S.jointX)) > 0.1);
  close(M.evaluate(end.model, S.jointX), Math.sin(S.jointX));
  const changedExisting = coarse.neurons.filter(old => {
    const updated = end.model.neurons.find(unit => Math.abs(old.t - unit.t) < 1e-10);
    return updated && Math.abs(updated.a - old.a) > 1e-8;
  });
  assert.ok(changedExisting.length > 0, 'The construction must update existing output weights too.');

  for (const x of grid(0, M.TAU)) close(M.evaluate(start.model, x), M.evaluate(coarse, x));
  for (const progress of [0, 0.1, 0.4, 0.75, 1]) {
    const frame = S.frame('sine-joint', progress);
    const signedError = M.evaluate(frame.model, S.jointX) - Math.sin(S.jointX);
    close(signedError, (1 - progress) * (M.evaluate(coarse, S.jointX) - Math.sin(S.jointX)));
    for (const x of S.initialKnots) close(M.evaluate(frame.model, x), Math.sin(x));
  }
});

test('the widening scene progresses from width 7 to 26 with decreasing actual error', () => {
  let previous = S.approximation(6);
  let previousError = sampledError(previous.model);
  for (let width = 7; width <= 26; width += 1) {
    const progress = width === 26 ? 1 : (width - 7 + 0.25) / 20;
    const frame = S.frame('sine-wide', progress);
    const current = S.approximation(width);
    assert.equal(frame.width, width);
    assert.equal(frame.model.width, width);
    assert.equal(current.model.width, width);
    assert.equal(current.knots.length, previous.knots.length + 1);
    for (const x of previous.knots) assert.ok(current.knots.includes(x));
    const additions = current.knots.filter(x => !previous.knots.includes(x));
    assert.equal(additions.length, 1);
    close(additions[0], M.maxError(previous.knots).x);
    close(M.evaluate(current.model, additions[0]), Math.sin(additions[0]));
    close(current.error, M.maxError(current.knots).error);
    const actualError = sampledError(current.model);
    assert.ok(actualError < previousError, `Maximum error must decrease at width ${width}.`);
    previous = current;
    previousError = actualError;
  }
  assert.equal(S.frame('sine-wide', 0).width, 7);
  assert.equal(S.frame('sine-wide', 1).width, 26);
  assert.equal(S.frame('universal', 0).width, 26);
  assert.ok(previousError < 0.01);
});

test('weight blending is linear in function space, including outside the plotted domain', () => {
  for (const [beforeWidth, afterWidth] of [[6, 7], [7, 26]]) {
    const before = S.approximation(beforeWidth).model;
    const after = S.approximation(afterWidth).model;
    for (const alpha of [-0.5, 0, 0.1, 0.4, 0.75, 1, 1.5]) {
      const p = Math.max(0, Math.min(1, alpha));
      const blended = S.blend(before, after, alpha);
      assert.equal(blended.width, after.width);
      allNumbersFinite(blended, 'blended');
      for (const x of grid(-2, M.TAU + 2, 400)) {
        close(M.evaluate(blended, x),
          (1 - p) * M.evaluate(before, x) + p * M.evaluate(after, x));
      }
    }
  }
});
