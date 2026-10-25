'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const D = require('../docs/neural-networks/classification-data.js');

const dot = (a, b) => a.reduce((sum, value, j) => sum + value * b[j], 0);
const sigmoid = value => 1 / (1 + Math.exp(-value));
const close = (a, b, tolerance = 1e-10) => {
  assert.ok(Number.isFinite(a) && Number.isFinite(b));
  assert.ok(Math.abs(a - b) <= tolerance, `${a} differs from ${b}`);
};

// Deliberately independent of the renderer and its model helpers: these tests
// recompute the predictions from the frozen learned parameters.
function forward(x) {
  const preactivations = D.hiddenWeights.map((weights, j) => dot(weights, x) + D.hiddenBias[j]);
  const hidden = preactivations.map(value => Math.max(0, value));
  const score = D.outputBias + dot(D.outputWeights, hidden);
  return { preactivations, hidden, score, probability: sigmoid(score) };
}

test('the teaching artifact contains a fitted two-input, two-ReLU, one-sigmoid network', () => {
  assert.deepEqual(D.provenance.architecture, [2, 2, 1]);
  assert.equal(D.provenance.hiddenActivation, 'ReLU');
  assert.equal(D.provenance.outputActivation, 'sigmoid');
  assert.equal(D.hiddenWeights.length, 2);
  assert.equal(D.hiddenBias.length, 2);
  assert.equal(D.outputWeights.length, 2);
  assert.ok(D.hiddenWeights.every(row => row.length === 2 && row.every(Number.isFinite)));
  assert.ok([...D.hiddenBias, ...D.outputWeights, D.outputBias].every(Number.isFinite));
  assert.equal(D.provenance.optimizerConverged, true);
  assert.match(D.provenance.training, /Joint.*binary cross-entropy/);
  assert.match(D.provenance.description, /New constructed binary example/);
  assert.match(D.provenance.interpretation, /Training accuracy only/);
});

test('every displayed observation is correctly classified by the actual logistic head', () => {
  assert.equal(D.inputs.length, 82);
  assert.equal(D.labels.length, D.inputs.length);
  assert.equal(D.labels.filter(label => label === 0).length, 41);
  assert.equal(D.labels.filter(label => label === 1).length, 41);
  assert.ok(D.inputs.every(x => x.length === 2 && x.every(Number.isFinite)));
  const predictions = D.inputs.map(forward);
  const margins = predictions.map((prediction, index) => {
    assert.equal(Number(prediction.probability >= 0.5), D.labels[index]);
    assert.ok(prediction.probability > 0 && prediction.probability < 1,
      'Perfect training classification does not mean probabilities equal zero or one.');
    return (2 * D.labels[index] - 1) * prediction.score;
  });
  assert.ok(Math.min(...margins) > 5, 'All observations are comfortably away from the boundary.');
  close(Math.min(...margins), D.metrics.minSignedLogit);
  close(Math.min(...margins) / Math.hypot(...D.outputWeights), D.metrics.minGeometricMargin);
  close(predictions.reduce((sum, prediction, i) =>
    sum - Math.log(D.labels[i] ? prediction.probability : 1 - prediction.probability), 0) / predictions.length,
  D.metrics.meanCrossEntropy);
  assert.equal(D.metrics.correct, 82);
  assert.equal(D.metrics.trainingCount, 82);
  assert.equal(D.metrics.accuracy, 1);
});

test('both learned units change activation state and the representation is genuinely two-dimensional', () => {
  const predictions = D.inputs.map(forward);
  for (let j = 0; j < 2; j++) {
    assert.ok(predictions.some(p => p.preactivations[j] < 0));
    assert.ok(predictions.some(p => p.preactivations[j] > 0));
    assert.ok(Math.abs(D.outputWeights[j]) > 1);
  }
  const hidden = predictions.map(p => p.hidden);
  for (let i = 0; i < hidden.length; i++) {
    for (let j = i + 1; j < hidden.length; j++) {
      assert.ok(Math.hypot(hidden[i][0] - hidden[j][0], hidden[i][1] - hidden[j][1]) > 1e-6,
        'Moving observations retain distinct endpoint coordinates.');
    }
  }
  const mean = [0, 1].map(j => hidden.reduce((sum, h) => sum + h[j], 0) / hidden.length);
  const centered = hidden.map(h => h.map((value, j) => value - mean[j]));
  const xx = centered.reduce((sum, h) => sum + h[0] ** 2, 0);
  const yy = centered.reduce((sum, h) => sum + h[1] ** 2, 0);
  const xy = centered.reduce((sum, h) => sum + h[0] * h[1], 0);
  assert.ok(xx * yy - xy * xy > 1);
  assert.equal(D.metrics.hiddenCoordinateRank, 2);
});

test('the original observations cannot be perfectly separated by any affine score', () => {
  // An explicit intersection of the two convex hulls proves the claim without
  // relying on a particular failed logistic-regression fit or optimizer.
  const points = [0, 1].map(label => {
    const witness = D.nonseparabilityWitness[`class${label}`];
    close(witness.weights.reduce((sum, value) => sum + value, 0), 1);
    assert.ok(witness.weights.every(weight => weight > 0));
    witness.indices.forEach(index => assert.equal(D.labels[index], label));
    return [0, 1].map(j => witness.indices.reduce((sum, index, i) =>
      sum + witness.weights[i] * D.inputs[index][j], 0));
  });
  points[0].forEach((value, j) => {
    close(value, points[1][j], 1e-11);
    close(value, D.nonseparabilityWitness.intersection[j], 1e-11);
  });
  assert.equal(D.metrics.originalLinearSeparable, false);
});

test('the head has a straight 0.5-probability boundary in the learned coordinates', () => {
  const [a, b] = D.outputWeights;
  // Three widely spaced points on the displayed head's affine zero set.
  for (const h1 of [0, 2, 4]) {
    const h2 = -(D.outputBias + a * h1) / b;
    close(sigmoid(D.outputBias + a * h1 + b * h2), 0.5);
    assert.ok(sigmoid(D.outputBias + a * h1 + b * (h2 - 0.1)) > 0.5);
    assert.ok(sigmoid(D.outputBias + a * h1 + b * (h2 + 0.1)) < 0.5);
  }
});
