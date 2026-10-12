import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import test from 'node:test';
import vm from 'node:vm';

const require = createRequire(import.meta.url);
const example = require('../docs/boosting/loss-example.js');
const worked = require('../docs/boosting/worked-example.js');
const close = (actual, expected, tolerance = 1e-10) => {
  assert.ok(Number.isFinite(actual), `Expected a finite value, received ${actual}`);
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} should equal ${expected}`);
};
const vectorClose = (actual, expected, tolerance) => {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => close(value, expected[index], tolerance));
};
const correctionLoss = (residuals, correction, options) => residuals.reduce((sum, value) => sum + example.pointLoss(value, correction, options), 0);

test('squared-error paths agree with each other and the existing worked calculation', () => {
  for (const rate of [0.25, 0.5, 1]) {
    const model = example.build({ dataset: 'worked', loss: 'squared', rate, rounds: 24 });
    const reference = worked.build(rate, 24);
    close(model.initial, 2);
    assert.deepEqual(model.gradient, model.residual);
    model.gradient.forEach((state, index) => {
      vectorClose(state.predictions, reference[index].predictions);
      vectorClose(state.targets, reference[index].residual);
      close(state.loss, reference[index].loss);
      if (index > 0) assert.equal(state.tree.threshold, reference[index].tree.threshold);
    });
  }
  const model = example.build({ loss: 'squared', dataset: 'outlier' });
  assert.deepEqual(model.gradient, model.residual);
});

test('negative gradients differentiate the selected original loss with respect to predictions', () => {
  for (const options of [
    { loss: 'squared' }, { loss: 'huber', delta: 0.5 }, { loss: 'huber', delta: 2 },
    { loss: 'expectile', tau: 0.2 }, { loss: 'expectile', tau: 0.5 }, { loss: 'expectile', tau: 0.8 }
  ]) {
    for (const [outcome, prediction] of [[0, 2], [1, 0.75], [5, 2], [3, 3]]) {
      const epsilon = 1e-5;
      const derivative = (example.pointLoss(outcome, prediction + epsilon, options) -
        example.pointLoss(outcome, prediction - epsilon, options)) / (2 * epsilon);
      close(-derivative, example.negativeGradient(outcome, prediction, options), 1e-5);
    }
  }
  close(example.negativeGradient(15, 2, { loss: 'huber', delta: 1 }), 1);
  close(example.negativeGradient(15, 2, { loss: 'expectile', tau: 0.8 }), 20.8);
  close(example.negativeGradient(0, 2, { loss: 'expectile', tau: 0.8 }), -0.8);
});

test('Huber corrections minimize original loss rather than averaging clipped targets', () => {
  const options = { loss: 'huber', delta: 1 };
  const cases = [
    { values: [-2, -2, -1], minimizer: -5 / 3 },
    { values: [0, 1, 2, 13], minimizer: 1.5 },
    { values: [-5, 5], minimizer: 0 },
    { values: [7], minimizer: 7 },
    { values: [0, 2, 1, 5], minimizer: 1.5 }
  ];
  for (const { values, minimizer } of cases) {
    const correction = example.minimizeResiduals(values, options);
    close(correction, minimizer);
    close(values.reduce((sum, value) => sum + example.negativeGradient(value, correction, options), 0), 0);
    for (const candidate of [correction - 3, correction - 0.2, correction + 0.2, correction + 3]) {
      assert.ok(correctionLoss(values, correction, options) <= correctionLoss(values, candidate, options) + 1e-10);
    }
  }
  // This loss has a flat optimum: every value between −4 and +4 minimizes it.
  close(correctionLoss([-5, 5], -4, options), correctionLoss([-5, 5], 4, options));
  assert.notEqual(example.minimizeResiduals([-2, -2, -1], options), -1);
});

test('asymmetric squared-error corrections are expectiles with the specified loss scale', () => {
  close(example.pointLoss(2, 0, { loss: 'expectile', tau: 0.8 }), 3.2);
  close(example.pointLoss(0, 2, { loss: 'expectile', tau: 0.8 }), 0.8);
  close(example.minimizeResiduals([0, 10], { loss: 'expectile', tau: 0.8 }), 8);
  close(example.minimizeResiduals([0, 10], { loss: 'expectile', tau: 0.2 }), 2);
  close(example.minimizeResiduals([0, 2, 1, 5], { loss: 'expectile', tau: 0.8 }), 23 / 7);
  for (const tau of [0.2, 0.5, 0.8]) {
    for (const values of [[0, 10], [-2, -2, -1], [0, 1, 2, 13], [4, 4, 4], [7]]) {
      const options = { loss: 'expectile', tau };
      const correction = example.minimizeResiduals(values, options);
      close(values.reduce((sum, value) => sum + example.negativeGradient(value, correction, options), 0), 0);
      for (const offset of [-2, -0.1, 0.1, 2]) {
        assert.ok(correctionLoss(values, correction, options) <= correctionLoss(values, correction + offset, options) + 1e-10);
      }
    }
  }
});

test('τ = 1/2 reproduces half-squared loss, gradients, constants, splits, and both paths', () => {
  for (const dataset of ['worked', 'outlier']) {
    for (const rate of [0.25, 0.5, 1]) {
      const squared = example.build({ dataset, loss: 'squared', rate });
      const symmetric = example.build({ dataset, loss: 'expectile', tau: 0.5, rate });
      assert.equal(squared.initial, symmetric.initial);
      assert.deepEqual(squared.gradient, symmetric.gradient);
      assert.deepEqual(squared.residual, symmetric.residual);
    }
  }
});

test('reflecting outcomes and replacing τ by 1 − τ reflects the whole expectile path', () => {
  const y = [0, 2, 1, 5], reflectedY = y.map(value => -value);
  for (const rate of [0.25, 0.5, 1]) {
    const upper = example.build({ loss: 'expectile', tau: 0.8, dataset: 'worked', rate });
    const lower = example.build({ loss: 'expectile', tau: 0.2, x: [1, 2, 3, 4], y: reflectedY, rate });
    close(lower.initial, -upper.initial);
    for (const method of ['gradient', 'residual']) {
      upper[method].forEach((state, index) => {
        vectorClose(lower[method][index].predictions, state.predictions.map(value => -value));
        vectorClose(lower[method][index].targets, state.targets.map(value => -value));
        close(lower[method][index].loss, state.loss);
        if (index > 0) assert.equal(lower[method][index].tree.threshold, state.tree.threshold);
      });
    }
  }
});

test('Huber changes the selected split even on the original four observations', () => {
  const model = example.build({ dataset: 'worked', loss: 'huber', delta: 1, rate: 1, rounds: 1 });
  close(model.initial, 1.5);
  vectorClose(model.gradient[0].targets, [-1, 0.5, -0.5, 1]);
  assert.equal(model.gradient[1].tree.threshold, 1.5);
  assert.equal(model.residual[1].tree.threshold, 3.5);
  // The mathematically tied target fits choose the first sorted threshold.
  close(model.gradient[1].candidates[0].sse, model.gradient[1].candidates[2].sse);
  assert.equal(model.gradient[1].residualTree.threshold, 3.5);
  vectorClose(model.gradient[0].predictions, model.residual[0].predictions);
});

test('the unusually high outcome changes the selected split and calibrated leaf values', () => {
  const model = example.build({ dataset: 'outlier', loss: 'huber', delta: 1, rate: 1, rounds: 1 });
  const next = model.gradient[1];
  close(model.initial, 2);
  vectorClose(model.gradient[0].targets, [-1, -1, -1, 0, 1, 1, 1]);
  assert.equal(next.tree.threshold, 3.5);
  assert.equal(model.residual[1].tree.threshold, 6.5);
  close(next.tree.targetLeft, -1); close(next.tree.targetRight, 0.75);
  close(next.tree.left, -5 / 3); close(next.tree.right, 1.5);
  const tiedCuts = next.candidates.filter(row => Math.abs(row.sse - next.tree.sse) < 1e-12).map(row => row.threshold);
  assert.deepEqual(tiedCuts, [3.5, 4.5]);
  assert.ok(next.candidates.find(row => row.threshold === 6.5).sse > next.tree.sse);
  assert.equal(next.residualTree.threshold, 6.5);
  assert.equal(next.calibratedResidualTree.threshold, 6.5);
  close(next.residualTree.left, -1 / 3);
  close(next.calibratedResidualTree.left, -0.5);
  // Gradient fitting is a local surrogate. It need not beat every other
  // partition's training loss on a particular round.
  assert.ok(model.residual[1].loss < next.loss);
});

test('gradient paths refit, recalibrate, and reduce selected loss for every supported control setting', () => {
  for (const dataset of ['worked', 'outlier']) {
    for (const loss of ['squared', 'huber', 'expectile']) {
      const parameters = loss === 'huber' ? [0.5, 1, 2] : loss === 'expectile' ? [0.2, 0.5, 0.8] : [1];
      for (const parameter of parameters) {
        for (const rate of [0.25, 0.5, 1]) {
          const model = example.build({ dataset, loss, delta: loss === 'huber' ? parameter : 1,
            tau: loss === 'expectile' ? parameter : 0.8, rate, rounds: 24 });
          for (let index = 1; index < model.gradient.length; index++) {
            const before = model.gradient[index - 1], state = model.gradient[index];
            assert.ok(state.loss <= before.loss + 1e-9, `${dataset}/${loss}/${parameter}/${rate}/${index}`);
            vectorClose(state.previous, before.predictions);
            vectorClose(state.predictions, before.predictions.map((value, i) => value + rate * state.tree.values[i]));
            vectorClose(state.targets, model.y.map((value, i) => example.negativeGradient(value, state.predictions[i], model.options)));
            assert.deepEqual(state.candidates, example.candidates(before.targets, model.x));
            for (const side of ['left', 'right']) {
              const residuals = before.residual.filter((_, i) => side === 'left' ? model.x[i] <= state.tree.threshold : model.x[i] > state.tree.threshold);
              close(state.tree[side], example.minimizeResiduals(residuals, model.options));
            }
            close(state.directionalDerivative, -before.targets.reduce((sum, value, i) => sum + value * state.correction[i], 0));
          }
          assert.ok(model.gradient[24].loss < model.gradient[0].loss);
        }
      }
    }
  }
});

test('a residual-only update can leave the selected optimum and stall at the wrong constant', () => {
  const model = example.build({ loss: 'expectile', tau: 0.8, x: [1, 1], y: [0, 10], rate: 1, rounds: 3 });
  close(model.initial, 8);
  model.gradient.forEach(state => { vectorClose(state.predictions, [8, 8]); close(state.loss, 16); });
  model.residual.slice(1).forEach(state => { vectorClose(state.predictions, [5, 5]); close(state.loss, 25); });
  vectorClose(model.residual[2].tree.values, [0, 0]);
  assert.ok(example.negativeGradient(0, 5, model.options) + example.negativeGradient(10, 5, model.options) > 0);
  assert.equal(model.gradient[1].tree.threshold, null);
});

test('the identical-input Huber witness stays at its selected optimum while residuals fit the mean', () => {
  const model = example.build({ dataset: 'shared', loss: 'huber', delta: 1, rate: 1, rounds: 3 });
  vectorClose(model.x, [1, 1, 1]);
  vectorClose(model.y, [0, 0, 10]);
  close(model.initial, 0.5);
  model.gradient.forEach(state => {
    vectorClose(state.predictions, [0.5, 0.5, 0.5]);
    close(state.loss, 9.25);
  });
  model.residual.slice(1).forEach(state => {
    vectorClose(state.predictions, [10 / 3, 10 / 3, 10 / 3]);
    close(state.loss, 71 / 6);
  });
  const fitted = model.gradient[1];
  assert.equal(fitted.tree.threshold, null);
  assert.equal(fitted.candidates.length, 1);
  close(fitted.tree.left, 0); close(fitted.tree.right, 0);
  close(fitted.residualTree.left, 17 / 6);
  assert.ok(model.residual[1].loss > model.residual[0].loss);
  for (const rate of [0.25, 0.5, 1]) {
    const path = example.build({ dataset: 'shared', loss: 'huber', rate }).residual;
    close(path[24].predictions[0], 10 / 3 + (0.5 - 10 / 3) * (1 - rate) ** 24);
  }
});

test('paths and model metadata are immutable and zero shrinkage preserves both starting models', () => {
  const model = example.build({ rate: 0 });
  const original = JSON.stringify(model);
  assert.ok(Object.isFrozen(model) && Object.isFrozen(model.options));
  for (const method of ['gradient', 'residual']) {
    assert.ok(Object.isFrozen(model[method]));
    for (const state of model[method]) {
      close(state.loss, model[method][0].loss);
      assert.ok(Object.isFrozen(state));
      for (const key of ['predictions', 'residual', 'targets', 'correction', 'step', 'candidates', 'residualCandidates']) assert.ok(Object.isFrozen(state[key]));
      for (const row of [...state.candidates, ...state.residualCandidates]) assert.ok(Object.isFrozen(row) && Object.isFrozen(row.values));
      if (state.tree) assert.ok(Object.isFrozen(state.tree) && Object.isFrozen(state.tree.values));
    }
  }
  assert.throws(() => { model.gradient[1].tree.left = 99; }, TypeError);
  example.build({ loss: 'squared', dataset: 'worked' });
  assert.equal(JSON.stringify(model), original);
  assert.equal(example.build({ rounds: 0 }).gradient.length, 1);
});

test('the independent UMD module exposes its browser API without a shared math dependency', () => {
  const context = vm.createContext({ window: {} });
  const source = readFileSync(new URL('../docs/boosting/loss-example.js', import.meta.url), 'utf8');
  vm.runInContext(source, context);
  close(context.window.LossBoostingExample.build({ rounds: 1 }).initial, 2);
  assert.equal(context.window.LossBoostingExample.build({ rounds: 1 }).gradient[1].tree.threshold, 3.5);
});

test('invalid losses, parameters, data, rates, and rounds are rejected', () => {
  for (const options of [null, [], 'huber']) assert.throws(() => example.build(options), TypeError);
  for (const options of [{ loss: 'absolute' }, { loss: 'toString' }, { delta: 0 }, { delta: -1 },
    { delta: Infinity }, { tau: 0 }, { tau: 1 }, { tau: NaN }, { rate: -0.1 }, { rate: 1.1 },
    { rate: Infinity }, { rounds: -1 }, { rounds: 1.5 }, { rounds: NaN }, { dataset: 'unknown' }]) {
    assert.throws(() => example.build(options), RangeError);
  }
  for (const values of [[], [NaN], [Infinity], 'values']) assert.throws(() => example.minimizeResiduals(values), TypeError);
  assert.throws(() => example.pointLoss(0, Infinity), TypeError);
  assert.throws(() => example.negativeGradient(NaN, 0), TypeError);
  assert.throws(() => example.candidates([1, 2], [1]), RangeError);
  assert.throws(() => example.build({ x: [1], y: [0, 1] }), RangeError);
  assert.throws(() => example.build({ x: [1] }), TypeError);
});
