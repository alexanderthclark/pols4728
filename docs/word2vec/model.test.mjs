import assert from 'node:assert/strict';
import {
  vocabulary, initial, forward, step, interpolateModel, trainingPath,
} from './model.js';

const close = (actual, expected, tolerance = 1e-9) => {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);
};
const loss = model => -Math.log(forward(model).p[2]);

assert.equal(vocabulary.length, 6);
assert.equal(initial.Win.length, 2);
assert.ok(initial.Win.every(row => row.length === vocabulary.length));
assert.equal(initial.Wout.length, vocabulary.length);
assert.ok(initial.Wout.every(row => row.length === initial.Win.length));
assert.equal(initial.b.length, vocabulary.length);

const first = forward(initial);
assert.deepEqual(first.h, [0.6, 0.4]);
[0.34, 0, 0.48, -0.10, -0.04, 0.26].forEach((value, j) => close(first.z[j], value));
assert.equal(first.p.length, vocabulary.length);
assert.ok(first.p.every(value => value > 0 && value < 1));
close(first.p.reduce((sum, value) => sum + value, 0), 1);

const before = structuredClone(initial);
const eta = 0.3;
const updated = step(initial, 1, 2, eta);
assert.deepEqual(initial, before, 'An SGD step must not mutate its input.');
assert.ok(loss(updated) < loss(initial), 'The first step must reduce observed-pair loss.');
for (let d = 0; d < initial.Win.length; d += 1) {
  for (let c = 0; c < vocabulary.length; c += 1) {
    if (c === 1) assert.notEqual(updated.Win[d][c], initial.Win[d][c]);
    else assert.equal(updated.Win[d][c], initial.Win[d][c]);
  }
}
assert.ok(updated.Wout.every((row, j) => row.some((value, d) => value !== initial.Wout[j][d])));
assert.deepEqual(updated.b, [0, 0, 0, 0, 0, 0], 'The minimal model omits biases.');

function finiteDifferenceGradient(key, index, innerIndex) {
  const epsilon = 1e-6;
  const plus = structuredClone(initial);
  const minus = structuredClone(initial);
  if (innerIndex === undefined) {
    plus[key][index] += epsilon;
    minus[key][index] -= epsilon;
  } else {
    plus[key][index][innerIndex] += epsilon;
    minus[key][index][innerIndex] -= epsilon;
  }
  return (loss(plus) - loss(minus)) / (2 * epsilon);
}

// Numerical differentiation verifies the SGD implementation independently of
// its analytical gradient, including the target and non-target output rows.
for (let d = 0; d < 2; d += 1) {
  close((initial.Win[d][1] - updated.Win[d][1]) / eta,
    finiteDifferenceGradient('Win', d, 1), 1e-7);
}
for (let j = 0; j < vocabulary.length; j += 1) {
  for (let d = 0; d < 2; d += 1) {
    close((initial.Wout[j][d] - updated.Wout[j][d]) / eta,
      finiteDifferenceGradient('Wout', j, d), 1e-7);
  }
}

assert.deepEqual(interpolateModel(initial, updated, 0), initial);
const interpolatedEnd = interpolateModel(initial, updated, 1);
for (const key of ['Win', 'Wout', 'b']) {
  const actual = interpolatedEnd[key].flat();
  updated[key].flat().forEach((value, index) => close(actual[index], value));
}
close(interpolateModel(initial, updated, 0.5).Win[0][1],
  (initial.Win[0][1] + updated.Win[0][1]) / 2);
assert.equal(trainingPath.length, 17);
assert.deepEqual(trainingPath[0], initial);
assert.deepEqual(trainingPath[1], updated);
for (let i = 1; i < trainingPath.length; i += 1) {
  assert.ok(loss(trainingPath[i]) < loss(trainingPath[i - 1]));
  const { p } = forward(trainingPath[i]);
  close(p.reduce((sum, value) => sum + value, 0), 1);
  assert.ok(p.every(Number.isFinite));
}

// A common score shift must leave the distribution unchanged.
const shifted = structuredClone(initial);
shifted.Wout = shifted.Wout.map(row => row.map(value => value + 10000));
forward(shifted).p.forEach((value, j) => close(value, first.p[j]));

console.log('Word2vec model checks passed: dimensions, exact forward values, finite-difference gradients, immutable SGD, selected-column updates, loss descent, and stable softmax.');
console.log(JSON.stringify({
  initial: { ...first, loss: loss(initial) },
  firstUpdate: { model: updated, ...forward(updated), loss: loss(updated) },
  final: { ...forward(trainingPath.at(-1)), loss: loss(trainingPath.at(-1)) },
}, null, 2));
