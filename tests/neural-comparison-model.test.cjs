'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const M = require('../docs/neural-networks/comparison-model.js');

const sigmoid = z => 1 / (1 + Math.exp(-z));
const close = (actual, expected, tolerance = 1e-11) =>
  assert.ok(Math.abs(actual - expected) < tolerance, `${actual} differs from ${expected}`);

// Recompute from the documented parameters, independently of exported helpers.
const affine = (bias, weights, x) => weights.reduce((sum, w, i) => sum + w * x[i], bias);
const forward = (model, x) => sigmoid(model.outputBias + model.outputWeights.reduce((sum, w, j) =>
  sum + w * sigmoid(affine(model.hiddenBias[j], model.hiddenWeights[j], x)), 0));

test('comparison functions are exact ordinary logistic and one-hidden-layer sigmoid models', () => {
  assert.equal(M.network1d.hiddenWeights.length, 2);
  assert.equal(M.network2d.hiddenWeights.length, 4);
  assert.match(M.provenance.description, /Fixed illustrative.*not a fit/);
  for (let i = 0; i <= 20; i++) {
    const x = i / 20;
    close(M.logisticProbability1d(x), sigmoid(affine(M.logistic1d.bias, M.logistic1d.weights, [x])));
    close(M.networkProbability1d(x), forward(M.network1d, [x]));
    for (let j = 0; j <= 20; j++) {
      const x2 = j / 20;
      close(M.logisticProbability2d(x, x2), sigmoid(affine(M.logistic2d.bias, M.logistic2d.weights, [x, x2])));
      close(M.networkProbability2d(x, x2), forward(M.network2d, [x, x2]));
    }
  }
});

test('one-input logistic is monotone while two hidden sigmoid units make a hump', () => {
  close(M.logisticProbability1d(0.3), 0.5);
  for (let i = 0; i < 100; i++) {
    assert.ok(M.logisticProbability1d(i / 100) < M.logisticProbability1d((i + 1) / 100));
  }
  assert.ok(M.networkProbability1d(0.1) < 0.01);
  assert.ok(M.networkProbability1d(0.5) > 0.99);
  assert.ok(M.networkProbability1d(0.9) < 0.01);
  for (let i = 0; i < 50; i++) {
    const x = i / 100;
    assert.ok(M.networkProbability1d(x) < M.networkProbability1d(x + 0.01));
    close(M.networkProbability1d(x), M.networkProbability1d(1 - x));
  }
});

test('two-input logistic has a straight boundary and its score is affine', () => {
  for (const x1 of [0.2, 0.4, 0.6, 0.8]) {
    const x2 = (0.82 - x1) / 0.65;
    close(M.logisticScore2d(x1, x2), 0);
    close(M.logisticProbability2d(x1, x2), 0.5);
  }
  close(M.logisticScore2d(0.45, 0.55),
    (M.logisticScore2d(0.2, 0.8) + M.logisticScore2d(0.7, 0.3)) / 2);
});

test('four sigmoid units produce a smooth closed nonlinear region from their own scores', () => {
  assert.ok(M.networkScore2d(0.5, 0.5) > 0);
  // The positive component is enclosed: every edge of the displayed unit square
  // is negative, while the middle is positive.
  for (let i = 0; i <= 100; i++) {
    const t = i / 100;
    assert.ok(M.networkScore2d(0, t) < 0);
    assert.ok(M.networkScore2d(1, t) < 0);
    assert.ok(M.networkScore2d(t, 0) < 0);
    assert.ok(M.networkScore2d(t, 1) < 0);
    close(M.networkScore2d(t, 0.3), M.networkScore2d(1 - t, 0.3));
  }
  // Locate actual zero-score intersections with rays from the center. Their
  // different radii rule out a substituted circular contour, and four zeroes
  // on non-collinear axes rule out an affine classifier.
  const radiusAt = angle => {
    let lo = 0, hi = 0.5;
    for (let i = 0; i < 60; i++) {
      const radius = (lo + hi) / 2;
      if (M.networkScore2d(0.5 + radius * Math.cos(angle), 0.5 + radius * Math.sin(angle)) > 0) lo = radius;
      else hi = radius;
    }
    return (lo + hi) / 2;
  };
  const axisRadius = radiusAt(0), diagonalRadius = radiusAt(Math.PI / 4);
  assert.ok(axisRadius > 0.23 && axisRadius < 0.39);
  assert.ok(diagonalRadius > 0.23 && diagonalRadius < 0.39);
  assert.ok(Math.abs(axisRadius - diagonalRadius) > 0.001);
  for (let i = 0; i < 16; i++) {
    const angle = 2 * Math.PI * i / 16, radius = radiusAt(angle);
    close(M.networkScore2d(0.5 + radius * Math.cos(angle), 0.5 + radius * Math.sin(angle)), 0);
  }
});

test('the fixed binary observations separate under the networks and expose affine limitations', () => {
  assert.equal(new Set(M.points1d.map(p => p.id)).size, M.points1d.length);
  assert.equal(new Set(M.points2d.map(p => p.id)).size, M.points2d.length);
  assert.ok(M.points1d.every(p => p.x >= 0 && p.x <= 1 && [0, 1].includes(p.y)));
  assert.ok(M.points2d.every(p => [p.x1, p.x2].every(x => x >= 0 && x <= 1) && [0, 1].includes(p.y)));
  for (const point of M.points1d) assert.ok((2 * point.y - 1) * M.networkScore1d(point.x) > 2);
  for (const point of M.points2d) assert.ok((2 * point.y - 1) * M.networkScore2d(point.x1, point.x2) > 1.5);
  assert.ok(M.points1d.some(p => Number(M.logisticProbability1d(p.x) >= 0.5) !== p.y));
  assert.ok(M.points2d.some(p => Number(M.logisticProbability2d(p.x1, p.x2) >= 0.5) !== p.y));

  // Exact convex-hull witnesses: a positive observation lies between two
  // negative observations. No affine score can strictly separate either set.
  const left = M.points1d.find(p => p.x === 0.04), right = M.points1d.find(p => p.x === 0.96);
  const middle = M.points1d.find(p => p.x === 0.5);
  assert.equal(left.y, 0); assert.equal(right.y, 0); assert.equal(middle.y, 1);
  close((left.x + right.x) / 2, middle.x);
  const center = M.points2d[0], outer = M.points2d.filter(p => p.y === 0);
  close((outer[0].x1 + outer[12].x1) / 2, center.x1);
  close((outer[0].x2 + outer[12].x2) / 2, center.x2);
  assert.equal(center.y, 1);
});
