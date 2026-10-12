(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.LossBoostingExample = api;
})(typeof window !== 'undefined' ? window : null, function () {
  'use strict';

  const freezeVector = values => Object.freeze(values.slice());
  const x = freezeVector([1, 2, 3, 4]);
  const datasets = Object.freeze({
    worked: Object.freeze({
      id: 'worked', label: 'Four-observation example',
      x, y: freezeVector([0, 2, 1, 5])
    }),
    outlier: Object.freeze({
      id: 'outlier', label: 'One unusually high outcome',
      x: freezeVector([1, 2, 3, 4, 5, 6, 7]),
      y: freezeVector([0, 0, 1, 2, 3, 4, 15])
    }),
    shared: Object.freeze({
      id: 'shared', label: 'Identical inputs',
      x: freezeVector([1, 1, 1]), y: freezeVector([0, 0, 10])
    })
  });
  const losses = Object.freeze({
    squared: Object.freeze({ id: 'squared', label: 'Squared error', name: 'Half-squared error',
      formula: '½(y − F)²', target: 'y − F' }),
    huber: Object.freeze({ id: 'huber', label: 'Huber', name: 'Huber loss',
      formula: '½r² if |r| ≤ δ; δ(|r| − δ/2) otherwise', target: 'clip(y − F, −δ, δ)' }),
    expectile: Object.freeze({ id: 'expectile', label: 'Asymmetric squared error', name: 'Asymmetric squared error',
      formula: 'τr² if r ≥ 0; (1 − τ)r² otherwise', target: '2τr if r ≥ 0; 2(1 − τ)r otherwise' })
  });

  function vector(values, name) {
    if (!Array.isArray(values) || values.length === 0 || !values.every(Number.isFinite)) {
      throw new TypeError(name + ' must be a nonempty array of finite numbers.');
    }
  }

  function lossOptions(options) {
    if (options === undefined) options = {};
    if (!options || typeof options !== 'object' || Array.isArray(options)) {
      throw new TypeError('Options must be an object.');
    }
    const { loss = 'squared', delta = 1, tau = 0.8 } = options;
    if (!Object.prototype.hasOwnProperty.call(losses, loss)) throw new RangeError('Unknown loss.');
    if (!Number.isFinite(delta) || delta <= 0) throw new RangeError('Huber δ must be finite and positive.');
    if (!Number.isFinite(tau) || tau <= 0 || tau >= 1) throw new RangeError('Expectile τ must lie strictly between 0 and 1.');
    return { loss, delta, tau };
  }

  function clip(value, bound) {
    return Math.max(-bound, Math.min(bound, value));
  }

  function lossAtResidual(residual, options) {
    if (options.loss === 'squared') return residual * residual / 2;
    if (options.loss === 'huber') {
      const magnitude = Math.abs(residual);
      return magnitude <= options.delta ? residual * residual / 2 : options.delta * (magnitude - options.delta / 2);
    }
    return (residual >= 0 ? options.tau : 1 - options.tau) * residual * residual;
  }

  function targetAtResidual(residual, options) {
    if (options.loss === 'squared') return residual;
    if (options.loss === 'huber') return clip(residual, options.delta);
    return 2 * (residual >= 0 ? options.tau : 1 - options.tau) * residual;
  }

  function finitePair(y, prediction) {
    if (!Number.isFinite(y) || !Number.isFinite(prediction)) throw new TypeError('Outcome and prediction must be finite.');
  }

  function pointLoss(y, prediction, options) {
    finitePair(y, prediction);
    return lossAtResidual(y - prediction, lossOptions(options));
  }

  function negativeGradient(y, prediction, options) {
    finitePair(y, prediction);
    return targetAtResidual(y - prediction, lossOptions(options));
  }

  // Minimize the ORIGINAL loss in a constant correction c, with r = y − F.
  // For Huber the score is piecewise linear, so interpolation at its knots
  // finds a minimizer without treating clipped targets as final leaf values.
  function minimizeResiduals(residuals, options) {
    vector(residuals, 'Residuals');
    const normalized = lossOptions(options);
    if (normalized.loss === 'squared' || normalized.loss === 'expectile' && normalized.tau === 0.5) {
      return residuals.reduce((sum, value) => sum + value, 0) / residuals.length;
    }
    if (normalized.loss === 'expectile') {
      const sorted = residuals.slice().sort((a, b) => a - b);
      // Each interval fixes which observations are over- and underpredicted.
      // Its weighted mean is exact if it lies in that interval.
      for (let countBelow = 0; countBelow <= sorted.length; countBelow++) {
        let numerator = 0, denominator = 0;
        sorted.forEach((value, index) => {
          const weight = index < countBelow ? 1 - normalized.tau : normalized.tau;
          numerator += weight * value;
          denominator += weight;
        });
        const candidate = numerator / denominator;
        const low = countBelow === 0 ? -Infinity : sorted[countBelow - 1];
        const high = countBelow === sorted.length ? Infinity : sorted[countBelow];
        if (candidate >= low && candidate <= high) return candidate;
      }
      throw new RangeError('Could not compute a finite expectile correction.');
    }

    const knots = Array.from(new Set(residuals.flatMap(value => [value - normalized.delta, value + normalized.delta]))).sort((a, b) => a - b);
    const scores = knots.map(correction => residuals.reduce((sum, value) => sum + clip(value - correction, normalized.delta), 0));
    const tolerance = 1e-12 * Math.max(1, residuals.length * normalized.delta);
    const roots = [];
    scores.forEach((score, index) => {
      if (Math.abs(score) <= tolerance) roots.push(knots[index]);
      if (index === scores.length - 1) return;
      const next = scores[index + 1];
      if (score > tolerance && next < -tolerance) {
        roots.push(knots[index] + score * (knots[index + 1] - knots[index]) / (score - next));
      }
    });
    if (roots.length === 0) throw new RangeError('Could not compute a finite Huber correction.');
    // Huber may have a whole interval of minimizers. Its midpoint is a stable
    // symmetric convention; it does not imply that the optimum is unique.
    return (Math.min(...roots) + Math.max(...roots)) / 2;
  }

  function candidates(targets, inputs = x) {
    vector(targets, 'Targets');
    vector(inputs, 'Inputs');
    if (inputs.length !== targets.length) throw new RangeError('Inputs and targets must have equal lengths.');
    const unique = Array.from(new Set(inputs)).sort((a, b) => a - b);
    const thresholds = unique.length === 1 ? [null] : unique.slice(0, -1).map((value, index) => value / 2 + unique[index + 1] / 2);
    return Object.freeze(thresholds.map(threshold => {
      const leftTargets = targets.filter((_, index) => threshold === null || inputs[index] <= threshold);
      const rightTargets = threshold === null ? leftTargets : targets.filter((_, index) => inputs[index] > threshold);
      const left = leftTargets.reduce((sum, value) => sum + value, 0) / leftTargets.length;
      const right = rightTargets.reduce((sum, value) => sum + value, 0) / rightTargets.length;
      const values = freezeVector(inputs.map(value => threshold === null || value <= threshold ? left : right));
      const sse = targets.reduce((sum, value, index) => sum + (value - values[index]) ** 2, 0);
      return Object.freeze({ threshold, left, right, values, sse, targetLeft: left, targetRight: right });
    }));
  }

  function bestCandidate(rows) {
    return rows.reduce((best, row) => {
      // Mathematically tied fits retain the first sorted threshold, even if
      // floating-point sums differ by a few ulps.
      const tolerance = 32 * Number.EPSILON * Math.max(best.sse, row.sse);
      return row.sse < best.sse - tolerance ? row : best;
    });
  }

  function calibratedTree(candidate, residuals, inputs, options, optimizeLeaves) {
    const inLeft = index => candidate.threshold === null || inputs[index] <= candidate.threshold;
    const leftResiduals = residuals.filter((_, index) => inLeft(index));
    const rightResiduals = candidate.threshold === null ? leftResiduals : residuals.filter((_, index) => !inLeft(index));
    const left = optimizeLeaves ? minimizeResiduals(leftResiduals, options) : candidate.left;
    const right = optimizeLeaves ? minimizeResiduals(rightResiduals, options) : candidate.right;
    const values = freezeVector(inputs.map((_, index) => inLeft(index) ? left : right));
    return Object.freeze({
      threshold: candidate.threshold, left, right, values, sse: candidate.sse,
      targetLeft: candidate.left, targetRight: candidate.right, targetValues: candidate.values
    });
  }

  function build(options) {
    if (options === undefined) options = {};
    if (!options || typeof options !== 'object' || Array.isArray(options)) throw new TypeError('Options must be an object.');
    const normalized = lossOptions({ ...options, loss: options.loss === undefined ? 'huber' : options.loss });
    const { rate = 0.5, rounds = 24, dataset = 'outlier' } = options;
    if (!Number.isFinite(rate) || rate < 0 || rate > 1) throw new RangeError('Learning rate must lie between 0 and 1.');
    if (!Number.isInteger(rounds) || rounds < 0) throw new RangeError('Rounds must be a nonnegative integer.');
    if (!Object.prototype.hasOwnProperty.call(datasets, dataset)) throw new RangeError('Unknown dataset.');
    const custom = options.x !== undefined || options.y !== undefined;
    const inputs = custom ? options.x : datasets[dataset].x;
    const outcomes = custom ? options.y : datasets[dataset].y;
    vector(inputs, 'Inputs'); vector(outcomes, 'Outcomes');
    if (inputs.length !== outcomes.length) throw new RangeError('Inputs and outcomes must have equal lengths.');
    const frozenX = freezeVector(inputs), frozenY = freezeVector(outcomes);
    const initial = minimizeResiduals(outcomes, normalized);
    const fullOptions = Object.freeze({ ...normalized, rate, rounds, dataset: custom ? 'custom' : dataset });
    const totalLoss = predictions => outcomes.reduce((sum, value, index) => sum + lossAtResidual(value - predictions[index], normalized), 0);

    function path(useGradient) {
      const predictions = freezeVector(outcomes.map(() => initial));
      const residual = freezeVector(outcomes.map(value => value - initial));
      const targets = freezeVector(residual.map(value => useGradient ? targetAtResidual(value, normalized) : value));
      const zero = freezeVector(outcomes.map(() => 0));
      const states = [Object.freeze({ round: 0, predictions, residual, targets, loss: totalLoss(predictions),
        tree: null, candidates: Object.freeze([]), previous: null, beforeLoss: totalLoss(predictions),
        correction: zero, step: zero, directionalDerivative: null, calibratedResidualTree: null,
        residualCandidates: Object.freeze([]), residualTree: null })];
      for (let round = 1; round <= rounds; round++) {
        const before = states[round - 1];
        const rows = candidates(before.targets, inputs);
        const tree = calibratedTree(bestCandidate(rows), before.residual, inputs, normalized, useGradient);
        const correction = tree.values;
        const step = freezeVector(correction.map(value => rate * value));
        const updated = freezeVector(before.predictions.map((value, index) => value + step[index]));
        const updatedResidual = freezeVector(outcomes.map((value, index) => value - updated[index]));
        const updatedTargets = freezeVector(updatedResidual.map(value => useGradient ? targetAtResidual(value, normalized) : value));
        const residualCandidates = candidates(before.residual, inputs);
        const rawCandidate = bestCandidate(residualCandidates);
        const residualTree = calibratedTree(rawCandidate, before.residual, inputs, normalized, false);
        const calibratedResidualTree = calibratedTree(rawCandidate, before.residual, inputs, normalized, true);
        states.push(Object.freeze({
          round, predictions: updated, residual: updatedResidual, targets: updatedTargets,
          loss: totalLoss(updated), tree, candidates: rows,
          previous: before.predictions, beforeLoss: before.loss, correction, step,
          directionalDerivative: -before.residual.reduce((sum, value, index) => sum + targetAtResidual(value, normalized) * correction[index], 0),
          calibratedResidualTree, residualCandidates, residualTree
        }));
      }
      return Object.freeze(states);
    }
    return Object.freeze({ x: frozenX, y: frozenY, initial, options: fullOptions,
      dataset: fullOptions.dataset, gradient: path(true), residual: path(false) });
  }

  return Object.freeze({ x, datasets, losses, pointLoss, negativeGradient, minimizeResiduals, candidates, build });
});
