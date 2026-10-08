(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.BoostingMath = api;
})(typeof window !== 'undefined' ? window : null, function () {
  'use strict';

  function vector(values, name) {
    if (!Array.isArray(values) || values.length === 0 || !values.every(Number.isFinite)) {
      throw new TypeError(name + ' must be a nonempty array of finite numbers.');
    }
  }

  function pair(a, b) {
    vector(a, 'First vector');
    vector(b, 'Second vector');
    if (a.length !== b.length) throw new RangeError('Vector lengths must agree.');
  }

  function dot(a, b) {
    pair(a, b);
    return a.reduce((sum, value, i) => sum + value * b[i], 0);
  }

  function add(a, b) {
    pair(a, b);
    return a.map((value, i) => value + b[i]);
  }

  function sub(a, b) {
    pair(a, b);
    return a.map((value, i) => value - b[i]);
  }

  function scale(a, s) {
    vector(a, 'Vector');
    if (!Number.isFinite(s)) throw new TypeError('Scale must be finite.');
    return a.map(value => value * s);
  }

  function norm(a) {
    vector(a, 'Vector');
    return Math.hypot.apply(null, a);
  }

  // Half the sum of squared errors, with no division by sample size.
  function loss(target, pred) {
    const residual = sub(target, pred);
    return 0.5 * dot(residual, residual);
  }

  // A one-feature regression stump. The leaf predictions are residual means;
  // each distinct adjacent feature pair supplies a candidate threshold.
  // Enumerating thresholds in ascending order makes exact ties deterministic.
  function fitStump(x, residual) {
    pair(x, residual);
    const unique = Array.from(new Set(x)).sort((a, b) => a - b);
    if (unique.length === 1) {
      const mean = residual.reduce((sum, value) => sum + value, 0) / residual.length;
      const values = residual.map(() => mean);
      return { values, threshold: null, left: mean, right: mean, sse: 2 * loss(residual, values) };
    }
    let best = null;
    for (let k = 0; k < unique.length - 1; k++) {
      const threshold = unique[k] / 2 + unique[k + 1] / 2;
      let leftSum = 0;
      let leftCount = 0;
      let rightSum = 0;
      let rightCount = 0;
      x.forEach((value, i) => {
        if (value <= threshold) {
          leftSum += residual[i];
          leftCount++;
        } else {
          rightSum += residual[i];
          rightCount++;
        }
      });
      const left = leftSum / leftCount;
      const right = rightSum / rightCount;
      const values = x.map(value => value <= threshold ? left : right);
      const sse = 2 * loss(residual, values);
      if (best === null || sse < best.sse) best = { values, threshold, left, right, sse };
    }
    return best;
  }

  // Zero initialization is deliberate: it mirrors the introductory geometry.
  // Each round refits to the CURRENT residual, then scales the entire tree.
  function buildPath(options) {
    const { rate = 0.5, rounds = 24, target = [1, -1, 1], x = [1, 2, 3] } = options || {};
    pair(x, target);
    if (!Number.isFinite(rate) || rate < 0) throw new RangeError('Rate must be finite and nonnegative.');
    if (!Number.isInteger(rounds) || rounds < 0) throw new RangeError('Rounds must be a nonnegative integer.');
    const initial = target.map(() => 0);
    const states = [{ round: 0, pred: initial, residual: target.slice(), loss: loss(target, initial), tree: null }];
    for (let round = 1; round <= rounds; round++) {
      const previous = states[round - 1];
      const tree = fitStump(x, previous.residual);
      const pred = add(previous.pred, scale(tree.values, rate));
      states.push({
        round, pred, residual: sub(target, pred), loss: loss(target, pred),
        tree, correction: tree.values.slice()
      });
    }
    return states;
  }

  // The angle of a zero vector is undefined; return null for display purposes.
  function angleDegrees(a, b) {
    pair(a, b);
    const denominator = norm(a) * norm(b);
    if (denominator === 0) return null;
    const cosine = Math.max(-1, Math.min(1, dot(a, b) / denominator));
    return Math.acos(cosine) * 180 / Math.PI;
  }

  // Exact 2-D geometry. An acute direction improves squared-error loss iff
  // 0 < relativeStep < 2*cos(theta). The endpoint itself gives equal loss.
  function stepGeometry(angle, relativeStep, residualLength) {
    if (residualLength === undefined) residualLength = 1;
    if (!Number.isFinite(angle) || angle < 0 || angle > 180) {
      throw new RangeError('Angle must lie between 0 and 180 degrees.');
    }
    if (!Number.isFinite(relativeStep) || relativeStep < 0) {
      throw new RangeError('Relative step must be finite and nonnegative.');
    }
    if (!Number.isFinite(residualLength) || residualLength <= 0) {
      throw new RangeError('Residual length must be finite and positive.');
    }
    const radians = angle * Math.PI / 180;
    // Set the geometrically exact right angle explicitly rather than retaining
    // the tiny positive floating-point value returned by cos(pi/2).
    const cosine = angle === 90 ? 0 : Math.cos(radians);
    const sine = angle === 0 || angle === 180 ? 0 : Math.sin(radians);
    const length = relativeStep * residualLength;
    const step = [length * cosine, length * sine];
    const beforeLoss = 0.5 * residualLength * residualLength;
    const afterLoss = loss([residualLength, 0], step);
    return {
      step, beforeLoss, afterLoss, improvement: beforeLoss - afterLoss,
      maxRelativeStep: Math.max(0, 2 * cosine)
    };
  }

  return { dot, add, sub, scale, norm, loss, fitStump, buildPath, angleDegrees, stepGeometry };
});
