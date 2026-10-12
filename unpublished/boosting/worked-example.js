(function (root, factory) {
  'use strict';
  const math = root && root.BoostingMath || (typeof require === 'function' ? require('../../docs/shared/boosting-math.js') : null);
  const api = factory(math);
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.WorkedBoostingExample = api;
})(typeof window !== 'undefined' ? window : null, function (math) {
  'use strict';
  if (!math) throw new Error('Load BoostingMath before the worked boosting example.');

  // Section 1.4.3 of the notes: one predictor and four observed responses.
  // This mean-initialized calculation supplements the earlier zero-initialized
  // three-observation geometry; it does not replace that construction.
  const x = Object.freeze([1, 2, 3, 4]);
  const y = Object.freeze([0, 2, 1, 5]);
  const baseline = y.reduce((sum, value) => sum + value, 0) / y.length;

  function loss(predictions) {
    return math.loss(y, predictions);
  }

  // Reconstruct the small rational values in the example for legible labels.
  // If no short fraction approximates a value, retain a concise decimal.
  function formatFraction(value) {
    if (!Number.isFinite(value)) throw new TypeError('The value must be finite.');
    if (value === 0) return '0';
    const magnitude = Math.abs(value);
    const tolerance = 1e-10 * Math.max(1, magnitude);
    let remainder = magnitude;
    let numeratorBefore = 0, numerator = 1;
    let denominatorBefore = 1, denominator = 0;
    for (let iteration = 0; iteration < 32; iteration++) {
      const integer = Math.floor(remainder);
      const nextNumerator = integer * numerator + numeratorBefore;
      const nextDenominator = integer * denominator + denominatorBefore;
      if (!Number.isSafeInteger(nextNumerator) || nextDenominator > 10000) break;
      numeratorBefore = numerator;
      numerator = nextNumerator;
      denominatorBefore = denominator;
      denominator = nextDenominator;
      if (Math.abs(magnitude - numerator / denominator) <= tolerance) {
        const sign = value < 0 && numerator !== 0 ? '-' : '';
        return sign + numerator + (denominator === 1 ? '' : '/' + denominator);
      }
      const fractionalPart = remainder - integer;
      if (fractionalPart === 0) break;
      remainder = 1 / fractionalPart;
    }
    return String(Number(value.toPrecision(6)));
  }

  function candidates(residual) {
    return Object.freeze(x.slice(0, -1).map((input, index) => {
      const threshold = input / 2 + x[index + 1] / 2;
      const leftResiduals = residual.slice(0, index + 1);
      const rightResiduals = residual.slice(index + 1);
      const left = leftResiduals.reduce((sum, value) => sum + value, 0) / leftResiduals.length;
      const right = rightResiduals.reduce((sum, value) => sum + value, 0) / rightResiduals.length;
      const values = Object.freeze(x.map(value => value <= threshold ? left : right));
      return Object.freeze({ threshold, left, right, values, sse: 2 * math.loss(residual, values) });
    }));
  }

  // A state's predictions, residual, and gradient describe its current model.
  // For round > 0, its tree/candidates describe the fit to the PREVIOUS state's
  // residuals. Its directional derivative is evaluated before that update.
  // Tree values are unscaled; step is the correction after learning-rate scaling.
  function build(rate = 1, rounds = 2) {
    if (!Number.isFinite(rate) || rate < 0 || rate > 1) {
      throw new RangeError('Learning rate must be finite and between 0 and 1.');
    }
    if (!Number.isInteger(rounds) || rounds < 0) {
      throw new RangeError('Rounds must be a nonnegative integer.');
    }

    const initial = Object.freeze(y.map(() => baseline));
    const zero = Object.freeze(y.map(() => 0));
    const initialLoss = loss(initial);
    const states = [Object.freeze({
      round: 0,
      rate,
      predictions: initial,
      residual: Object.freeze(math.sub(y, initial)),
      gradient: Object.freeze(math.sub(initial, y)),
      tree: null,
      loss: initialLoss,
      candidates: Object.freeze([]),
      directionalDerivative: null,
      previous: null,
      beforeLoss: initialLoss,
      correction: zero,
      step: zero
    })];

    for (let round = 1; round <= rounds; round++) {
      const before = states[round - 1];
      const fitted = math.fitStump(x, before.residual);
      const correction = Object.freeze(fitted.values.slice());
      const tree = Object.freeze({ ...fitted, values: correction });
      const step = Object.freeze(math.scale(correction, rate));
      const predictions = Object.freeze(math.add(before.predictions, step));
      states.push(Object.freeze({
        round,
        rate,
        predictions,
        residual: Object.freeze(math.sub(y, predictions)),
        gradient: Object.freeze(math.sub(predictions, y)),
        tree,
        loss: loss(predictions),
        candidates: candidates(before.residual),
        directionalDerivative: math.dot(before.gradient, correction),
        previous: before.predictions,
        beforeLoss: before.loss,
        correction,
        step
      }));
    }
    return Object.freeze(states);
  }

  return { x, y, baseline, loss, formatFraction, build };
});
