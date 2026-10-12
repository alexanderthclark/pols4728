import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import test from 'node:test';
import vm from 'node:vm';

const require = createRequire(import.meta.url);
const example = require('../docs/boosting/worked-example.js');
const math = require('../docs/shared/boosting-math.js');
const close = (actual, expected, tolerance = 1e-11) => {
  assert.ok(Number.isFinite(actual), `Expected a finite value, received ${actual}`);
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} should equal ${expected}`);
};
const vectorClose = (actual, expected) => {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => close(value, expected[index]));
};
const moved = (prediction, direction, alpha) => prediction.map((value, index) => value + alpha * direction[index]);

test('the appended example uses the four source observations and their best constant prediction', () => {
  assert.deepEqual(example.x, [1, 2, 3, 4]);
  assert.deepEqual(example.y, [0, 2, 1, 5]);
  assert.equal(example.baseline, 2);
  const path = example.build();
  assert.equal(path.length, 3);
  assert.equal(path[0].round, 0);
  assert.deepEqual(path[0].predictions, [2, 2, 2, 2]);
  assert.deepEqual(path[0].gradient, [2, 0, 1, -3]);
  assert.deepEqual(path[0].residual, [-2, 0, -1, 3]);
  assert.equal(path[0].tree, null);
  assert.equal(path[0].directionalDerivative, null);
  assert.equal(path[0].loss, 7);
  assert.equal(example.loss([2, 2, 2, 2]), 7);
  for (const constant of [1, 1.5, 2.5, 3]) {
    assert.ok(example.loss(Array(4).fill(constant)) > path[0].loss);
  }
});

test('the source candidate tables distinguish stump-fitting SSE from model loss', () => {
  const path = example.build();
  const expected = [
    [[1.5, -2, 2 / 3, 26 / 3], [2.5, -1, 1, 10], [3.5, -1, 3, 2]],
    [[1.5, -1, 1 / 3, 2 / 3], [2.5, 0, 0, 2], [3.5, 0, 0, 2]]
  ];
  expected.forEach((rows, index) => {
    const state = path[index + 1];
    assert.equal(state.candidates.length, 3);
    state.candidates.forEach((row, candidate) => {
      vectorClose([row.threshold, row.left, row.right, row.sse], rows[candidate]);
    });
    const best = state.candidates.reduce((winner, row) => row.sse < winner.sse ? row : winner);
    assert.equal(best.threshold, state.tree.threshold);
    vectorClose(best.values, state.correction);
    close(state.tree.sse, 2 * state.loss);
  });
});

test('the two full source updates yield loss 7 → 1 → 1/3 and recomputed gradients', () => {
  const [initial, first, second] = example.build();
  vectorClose(first.previous, initial.predictions);
  vectorClose(first.correction, [-1, -1, -1, 3]);
  vectorClose(first.predictions, [1, 1, 1, 5]);
  vectorClose(first.residual, [-1, 1, 0, 0]);
  vectorClose(first.gradient, [1, -1, 0, 0]);
  vectorClose(second.correction, [-1, 1 / 3, 1 / 3, 1 / 3]);
  vectorClose(second.predictions, [0, 4 / 3, 4 / 3, 16 / 3]);
  vectorClose(second.residual, [0, 2 / 3, -1 / 3, -1 / 3]);
  vectorClose(second.gradient, [0, -2 / 3, 1 / 3, 1 / 3]);
  close(first.loss, 1);
  close(second.loss, 1 / 3);
  close(first.directionalDerivative, -12);
  close(second.directionalDerivative, -4 / 3);
  // An aggregate improvement does not require every observation to improve.
  assert.equal(initial.predictions[1], example.y[1]);
  assert.notEqual(first.predictions[1], example.y[1]);
  assert.equal(first.predictions[3], example.y[3]);
  assert.notEqual(second.predictions[3], example.y[3]);
});

test('gradient coordinates differentiate predictions rather than predictor values', () => {
  const initial = example.build()[0];
  const originalX = example.x.slice(), originalY = example.y.slice();
  const epsilon = 1e-5;
  initial.gradient.forEach((entry, index) => {
    const plus = initial.predictions.slice(), minus = initial.predictions.slice();
    plus[index] += epsilon;
    minus[index] -= epsilon;
    close((example.loss(plus) - example.loss(minus)) / (2 * epsilon), entry, 1e-8);
  });
  assert.deepEqual(example.x, originalX);
  assert.deepEqual(example.y, originalY);
  assert.ok(example.loss([2, 2, 2, 2.001]) < initial.loss);
  assert.ok(example.loss([2, 2, 2, 1.999]) > initial.loss);
});

test('directional derivatives and whole loss curves justify descent and the step-size bound', () => {
  const path = example.build(), epsilon = 1e-5;
  const quadratics = [[7, -12, 6], [1, -4 / 3, 2 / 3]];
  path.slice(1).forEach((state, index) => {
    const [constant, linear, quadratic] = quadratics[index];
    close(state.directionalDerivative, linear);
    close((example.loss(moved(state.previous, state.correction, epsilon)) -
      example.loss(moved(state.previous, state.correction, -epsilon))) / (2 * epsilon), linear, 1e-8);
    for (const alpha of [0, 0.25, 0.5, 1, 1.5, 2, 2.5]) {
      close(example.loss(moved(state.previous, state.correction, alpha)), constant + linear * alpha + quadratic * alpha * alpha);
    }
    close(example.loss(moved(state.previous, state.correction, 2)), state.beforeLoss);
    assert.ok(example.loss(moved(state.previous, state.correction, 2.5)) > state.beforeLoss);
  });
});

test('half-step shrinkage refits the second tree and resolves its exact SSE tie by sorted threshold', () => {
  const [initial, first, second] = example.build(0.5);
  vectorClose(first.predictions, [1.5, 1.5, 1.5, 3.5]);
  vectorClose(first.residual, [-1.5, 0.5, -0.5, 1.5]);
  vectorClose(first.step, [-0.5, -0.5, -0.5, 1.5]);
  close(first.loss, 5 / 2);
  const expected = [[1.5, -1.5, 0.5, 2], [2.5, -0.5, 0.5, 4], [3.5, -0.5, 1.5, 2]];
  second.candidates.forEach((row, index) => vectorClose([row.threshold, row.left, row.right, row.sse], expected[index]));
  const tied = second.candidates.filter(row => row.sse === second.tree.sse);
  assert.deepEqual(tied.map(row => row.threshold), [1.5, 3.5]);
  assert.equal(second.tree.threshold, 1.5);
  vectorClose(second.correction, [-1.5, 0.5, 0.5, 0.5]);
  vectorClose(second.predictions, [3 / 4, 7 / 4, 7 / 4, 15 / 4]);
  vectorClose(second.residual, [-3 / 4, 1 / 4, -3 / 4, 5 / 4]);
  close(second.loss, 11 / 8);
  close(second.directionalDerivative, -3);
  assert.equal(initial.loss, 7);
  const alternative = moved(first.predictions, tied[1].values, 0.5);
  vectorClose(alternative, [5 / 4, 5 / 4, 5 / 4, 17 / 4]);
  close(example.loss(alternative), 11 / 8);
  assert.notDeepEqual(second.correction, example.build(1)[2].correction);
});

test('every requested round fits fresh residuals and scales the complete correction', () => {
  for (const rate of [0, 0.1, 0.25, 0.5, 1]) {
    const path = example.build(rate, 8);
    assert.equal(path.length, 9);
    for (let index = 1; index < path.length; index++) {
      const before = path[index - 1], state = path[index];
      vectorClose(state.previous, before.predictions);
      vectorClose(state.predictions, moved(before.predictions, state.correction, rate));
      vectorClose(state.residual, example.y.map((value, i) => value - state.predictions[i]));
      vectorClose(state.gradient, state.residual.map(value => -value));
      close(state.tree.sse, Math.min(...state.candidates.map(row => row.sse)));
      close(math.dot(before.residual, state.correction), math.dot(state.correction, state.correction));
      close(before.loss - state.loss, (rate - rate * rate / 2) * math.dot(state.correction, state.correction));
      if (rate > 0) assert.ok(state.loss < before.loss);
      else close(state.loss, before.loss);
    }
  }
  const full = example.build(1, 8);
  assert.deepEqual(full.slice(1).map(state => state.tree.threshold), [3.5, 1.5, 2.5, 1.5, 2.5, 1.5, 2.5, 1.5]);
  full.slice(1).forEach((state, index) => close(state.loss, 1 / 3 ** index));
  assert.ok(full[8].loss > 0);
});

test('immutable builds and fractional labels preserve the example across independent rate choices', () => {
  const path = example.build();
  const before = JSON.stringify(path);
  assert.ok(Object.isFrozen(path));
  for (const state of path) {
    assert.ok(Object.isFrozen(state));
    for (const key of ['predictions', 'residual', 'gradient', 'correction', 'step', 'candidates']) {
      assert.ok(Object.isFrozen(state[key]));
    }
    for (const row of state.candidates) assert.ok(Object.isFrozen(row) && Object.isFrozen(row.values));
    if (state.tree) assert.ok(Object.isFrozen(state.tree) && Object.isFrozen(state.tree.values));
  }
  assert.throws(() => { path[1].tree.values[0] = 100; }, TypeError);
  assert.throws(() => { path[2].candidates[0].left = 100; }, TypeError);
  example.build(0.5);
  assert.equal(JSON.stringify(path), before);
  for (const [value, label] of [[0, '0'], [-1, '-1'], [-1 / 3, '-1/3'], [4 / 3, '4/3'],
    [16 / 3, '16/3'], [26 / 3, '26/3'], [11 / 8, '11/8'], [1 / 2187, '1/2187']]) {
    assert.equal(example.formatFraction(value), label);
  }
});

test('the standalone UMD module loads the unchanged shared math and exposes its browser API', () => {
  const context = vm.createContext({ window: { BoostingMath: math } });
  const source = readFileSync(new URL('../docs/boosting/worked-example.js', import.meta.url), 'utf8');
  vm.runInContext(source, context);
  assert.equal(context.window.WorkedBoostingExample.baseline, 2);
  close(context.window.WorkedBoostingExample.build()[2].loss, 1 / 3);
  assert.throws(() => vm.runInContext(source, vm.createContext({ window: {} })), /Load BoostingMath/);
});

test('invalid rates, rounds, predictions, and labels are rejected without changing valid builds', () => {
  for (const rate of [-0.1, 1.1, NaN, Infinity]) assert.throws(() => example.build(rate), RangeError);
  for (const rounds of [-1, 1.5, NaN, Infinity]) assert.throws(() => example.build(1, rounds), RangeError);
  assert.equal(example.build(1, 0).length, 1);
  assert.throws(() => example.loss([2, 2]), RangeError);
  assert.throws(() => example.loss([2, 2, 2, NaN]), TypeError);
  assert.throws(() => example.formatFraction(Infinity), TypeError);
});
