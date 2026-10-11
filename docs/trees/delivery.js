(function (root, factory) {
  'use strict';
  const math = root && root.BoostingMath || (typeof require === 'function' ? require('../boosting/model.js') : null);
  const api = factory(math);
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.DeliveryExample = api;
})(typeof window !== 'undefined' ? window : null, function (math) {
  'use strict';
  if (!math) throw new Error('Load BoostingMath before the delivery example.');

  // Synthetic delivery distances (km) and times (minutes). Every update uses
  // learning rate 1 so the fitted leaf means can be followed arithmetically.
  const x = Object.freeze([1, 3, 4, 6, 7, 9, 10, 12]);
  const y = Object.freeze([6, 6, 11, 11, 11, 17, 17, 17]);
  const baseline = y.reduce((sum, value) => sum + value, 0) / y.length;

  function mse(predictions, observations = y) {
    return 2 * math.loss(observations, predictions) / observations.length;
  }

  const rounds = [];
  let current = y.map(() => baseline);
  for (let number = 1; number <= 2; number++) {
    const previous = current.slice();
    const residual = math.sub(y, previous);
    const fitted = math.fitStump(x, residual);
    const correction = fitted.values.slice();
    const prediction = math.add(previous, correction);
    rounds.push(Object.freeze({
      number,
      previous: Object.freeze(previous),
      residual: Object.freeze(residual),
      threshold: fitted.threshold,
      left: fitted.left,
      right: fitted.right,
      correction: Object.freeze(correction),
      prediction: Object.freeze(prediction),
      mse: mse(prediction)
    }));
    current = prediction;
  }
  Object.freeze(rounds);

  // A partial update adds progress times the current stump to all previous
  // completed updates. Input values between or beyond the observed distances
  // follow the same threshold rules; this is not a generalization claim.
  function predict(round, input, progress = 1) {
    if (!Number.isInteger(round) || round < 0 || round > rounds.length) {
      throw new RangeError('Round must be 0, 1, or 2.');
    }
    if (!Number.isFinite(input)) throw new TypeError('The input must be finite.');
    if (!Number.isFinite(progress) || progress < 0 || progress > 1) {
      throw new RangeError('Update progress must lie between 0 and 1.');
    }
    let value = baseline;
    for (let index = 0; index < round; index++) {
      const stump = rounds[index];
      const weight = index === round - 1 ? progress : 1;
      value += weight * (input <= stump.threshold ? stump.left : stump.right);
    }
    return value;
  }

  return { x, y, baseline, rounds, mse, predict };
});
