'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const D = require('../docs/neural-networks/width-data.js');

const widths = [2, 4, 8, 16, 32, 64, 128, 256];
const close = (actual, expected, tolerance = 1e-10) => {
  assert.ok(Number.isFinite(actual) && Number.isFinite(expected));
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} differs from ${expected}`);
};

// Independently reconstruct the affine/ReLU/sigmoid network from its exported
// raw-coordinate parameters. Do not call the renderer or generator helpers.
function forward(model, input) {
  const hidden = model.hiddenWeights.map((weights, j) =>
    Math.max(0, weights.reduce((sum, weight, k) => sum + weight * input[k], model.hiddenBias[j])));
  const score = hidden.reduce((sum, value, j) => sum + model.outputWeights[j] * value, model.outputBias);
  return { hidden, score, probability: 1 / (1 + Math.exp(-score)), label: Number(score >= 0) };
}

function classificationError(model, inputs, labels) {
  return inputs.reduce((mistakes, input, i) => mistakes + Number(forward(model, input).label !== labels[i]), 0) / inputs.length;
}

test('every displayed width is a finite two-input ReLU network with an affine sigmoid head', () => {
  assert.deepEqual(D.models.map(model => model.width), widths);
  for (const model of D.models) {
    assert.equal(model.hiddenWeights.length, model.width);
    assert.equal(model.hiddenBias.length, model.width);
    assert.equal(model.outputWeights.length, model.width);
    assert.ok(model.hiddenWeights.every(row => row.length === 2 && row.every(Number.isFinite)));
    assert.ok([...model.hiddenBias, ...model.outputWeights, model.outputBias].every(Number.isFinite));
    for (const input of [[0, 0], [Math.PI / 2, -.5], [Math.PI, .9]]) {
      const prediction = forward(model, input);
      assert.equal(prediction.hidden.length, model.width);
      assert.ok(prediction.hidden.every(value => Number.isFinite(value) && value >= 0));
      assert.ok(Number.isFinite(prediction.score));
      assert.ok(prediction.probability >= 0 && prediction.probability <= 1);
      assert.equal(Number(prediction.probability >= .5), prediction.label);
    }
  }
});

test('the fixed noisy sample preserves the two class curves and is perfectly classified at every displayed width', () => {
  assert.equal(D.inputs.length, 82);
  assert.equal(D.labels.length, 82);
  assert.equal(D.labels.filter(label => label === 0).length, 41);
  assert.equal(D.labels.filter(label => label === 1).length, 41);
  const residuals = [];
  for (let i = 0; i < D.inputs.length; i++) {
    const input = D.inputs[i];
    assert.equal(input.length, 2);
    assert.ok(input.every(Number.isFinite));
    close(input[0], (i % 41) * Math.PI / 40, 1e-11);
    residuals.push(input[1] + Math.sin(input[0]) ** 2 - .9 * D.labels[i]);
  }
  assert.ok(residuals.some(value => value > .1) && residuals.some(value => value < -.1),
    'The wider-network experiment must contain the new noisy observations.');
  for (const model of D.models) {
    close(classificationError(model, D.inputs, D.labels), model.trainingError);
    assert.equal(model.trainingError, 0);
    assert.equal(model.trainingMistakes, 0);
  }
});

test('displayed test observations are distinct from training observations and retain valid classifications', () => {
  assert.ok(D.testInputs.length >= 20);
  assert.equal(D.testInputs.length, D.testLabels.length);
  assert.equal(D.testInputs.length, D.testIndices.length);
  assert.equal(D.testCount, 40000);
  assert.deepEqual(D.testIndices, Array.from({ length: 160 }, (_, i) => i * 250),
    'The visible subset is selected by fixed indices, never by prediction outcomes.');
  assert.ok(D.testLabels.includes(0) && D.testLabels.includes(1));
  for (let i = 0; i < D.testInputs.length; i++) {
    const input = D.testInputs[i];
    assert.equal(input.length, 2);
    assert.ok(input.every(Number.isFinite));
    assert.ok([0, 1].includes(D.testLabels[i]));
    assert.ok(D.inputs.every(trainingInput => Math.hypot(input[0] - trainingInput[0], input[1] - trainingInput[1]) > 1e-9));
  }
});

test('test-point error halos agree with independent predictions at every displayed width', () => {
  const context = {
    NNWidthData: D,
    katex: { renderToString: tex => `<span>${tex}</span>` },
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../docs/neural-networks/width-views.js'), 'utf8'), context);
  D.models.forEach((model, modelIndex) => {
    const svg = { innerHTML: '', setAttribute() {}, removeAttribute() {} };
    context.NNWidthViews.render(svg, { modelIndex, noiseMix: 1, testOpacity: 1, showErrors: true });
    const actualMistakes = [...svg.innerHTML.matchAll(/data-test-error="(\d+)"/g)].map(match => Number(match[1]));
    const expectedMistakes = D.testInputs.flatMap((input, i) => forward(model, input).label === D.testLabels[i] ? [] : [i]);
    assert.deepEqual(actualMistakes, expectedMistakes);
    assert.equal([...svg.innerHTML.matchAll(/data-test-point-id="\d+"/g)].length, D.testInputs.length);
    assert.equal([...svg.innerHTML.matchAll(/data-width-point-id="\d+"/g)].length, D.inputs.length);
  });
});

test('every drawn contour segment is on the actual fitted zero-logit boundary', () => {
  for (const model of D.models) {
    assert.ok(model.boundaries.length > 0, `Width ${model.width} has no fitted contour.`);
    for (const line of model.boundaries) {
      assert.ok(line.length >= 2);
      for (let i = 0; i < line.length; i++) {
        const point = line[i];
        assert.equal(point.length, 2);
        point.forEach((value, j) => {
          assert.ok(value >= D.domain[j][0] - 1e-10 && value <= D.domain[j][1] + 1e-10);
        });
        close(forward(model, point).score, 0, 1e-5);
        if (i === 0) continue;
        const previous = line[i - 1];
        for (const t of [.25, .5, .75]) {
          const between = point.map((value, j) => previous[j] + t * (value - previous[j]));
          close(forward(model, between).score, 0, 1e-5);
        }
        const dx = point[0] - previous[0], dy = point[1] - previous[1];
        const length = Math.hypot(dx, dy);
        if (length > 1e-5) {
          const middle = point.map((value, j) => (previous[j] + value) / 2);
          const offset = Math.min(1e-5, length / 100);
          const a = forward(model, [middle[0] + offset * dy / length, middle[1] - offset * dx / length]).score;
          const b = forward(model, [middle[0] - offset * dy / length, middle[1] + offset * dx / length]).score;
          assert.ok(a * b < 0, `Width ${model.width} contour must separate the two predicted classes.`);
        }
      }
    }
  }
});

test('drawn contours include every independently found crossing through the plotted domain', () => {
  // Independently slice the network vertically. ReLU breakpoints make the
  // score affine on each slice interval, so all zeroes can be found exactly.
  // This checks completeness rather than merely checking the saved vertices.
  const [xDomain, yDomain] = D.domain;
  for (const model of D.models) {
    for (let scan = 0; scan < 37; scan++) {
      const x1 = xDomain[0] + (scan + .37) / 37 * (xDomain[1] - xDomain[0]);
      const cuts = [yDomain[0], yDomain[1]];
      model.hiddenWeights.forEach((weights, j) => {
        if (Math.abs(weights[1]) < 1e-12) return;
        const cut = -(weights[0] * x1 + model.hiddenBias[j]) / weights[1];
        if (cut > yDomain[0] && cut < yDomain[1]) cuts.push(cut);
      });
      cuts.sort((a, b) => a - b);
      const scores = cuts.map(x2 => forward(model, [x1, x2]).score);
      const roots = [];
      for (let j = 1; j < cuts.length; j++) {
        if (scores[j - 1] * scores[j] < 0) {
          roots.push(cuts[j - 1] - scores[j - 1] * (cuts[j] - cuts[j - 1]) / (scores[j] - scores[j - 1]));
        }
      }
      const drawn = [];
      for (const line of model.boundaries) {
        for (let j = 1; j < line.length; j++) {
          const a = line[j - 1], b = line[j];
          if ((a[0] - x1) * (b[0] - x1) <= 0 && Math.abs(b[0] - a[0]) > 1e-12) {
            drawn.push(a[1] + (x1 - a[0]) * (b[1] - a[1]) / (b[0] - a[0]));
          }
        }
      }
      for (const root of roots) {
        assert.ok(drawn.some(value => Math.abs(value - root) < 1e-8),
          `Width ${model.width} omits a fitted crossing at (${x1}, ${root}).`);
      }
      for (const crossing of drawn) {
        assert.ok(roots.some(value => Math.abs(value - crossing) < 1e-8),
          `Width ${model.width} adds a crossing outside the fitted zero set.`);
      }
    }
  }
});

test('the measured holdout curve preserves its actual nonmonotonic errors and endpoint counts', () => {
  // Exact counts on the independently generated 40,000-point holdout. The
  // generator's --check independently regenerates that full sample; the SVG
  // displays only a small subset and must not supply these full-sample rates.
  const mistakes = [378, 414, 396, 314, 312, 297, 274, 281];
  D.models.forEach((model, i) => {
    assert.equal(model.testMistakes, mistakes[i]);
    close(model.testError, mistakes[i] / D.testCount, 1e-12);
  });
  assert.ok(D.models[1].testError > D.models[0].testError);
  assert.ok(D.models.at(-1).testError > D.models.at(-2).testError);
  assert.ok(D.models.at(-1).testError < D.models[0].testError);
});

test('the confirmation report retains all widths and the narrow-network optimization failure', () => {
  const rows = D.confirmationSummary.rows;
  assert.equal(rows.length, 24);
  for (const seed of [6400, 6401, 6402]) {
    assert.deepEqual(rows.filter(row => row.trainingSeed === seed).map(row => row.width), widths);
  }
  const narrowFailure = rows.find(row => row.trainingSeed === 6400 && row.width === 2);
  assert.equal(narrowFailure.trainingMistakes, 8);
  close(narrowFailure.trainingError, 8 / 82, 1e-12);
  assert.equal(narrowFailure.testMistakes, 6003);
  for (const [seed, narrowMistakes, wideMistakes] of [[6401, 653, 521], [6402, 331, 320]]) {
    const narrow = rows.find(row => row.trainingSeed === seed && row.width === 2);
    const wide = rows.find(row => row.trainingSeed === seed && row.width === 256);
    assert.equal(narrow.trainingMistakes, 0);
    assert.equal(wide.trainingMistakes, 0);
    assert.equal(narrow.testMistakes, narrowMistakes);
    assert.equal(wide.testMistakes, wideMistakes);
    assert.ok(wide.testBCE > narrow.testBCE,
      'Improved classification on these two samples must not be described as improved probability calibration.');
  }
});
