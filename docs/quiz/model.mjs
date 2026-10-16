/** Quantities used by the quiz explainer; no display code or dependencies. */
export const LIMIT = 730;
export const YEAR = 365;

const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

function nonnegative(value, name) {
  if (!Number.isFinite(value) || value < 0) throw new RangeError(`${name} must be finite and nonnegative.`);
}

/** Standard normal CDF, evaluated by its convergent central series. */
export function normalCDF(z) {
  if (Number.isNaN(z)) return NaN;
  if (z === -Infinity) return 0;
  if (z === Infinity) return 1;
  // Beyond nine standard deviations the tail is below double precision.
  if (z < -9) return 0;
  if (z > 9) return 1;
  const x = Math.abs(z);
  let term = x;
  let sum = x;
  for (let denominator = 3; denominator < 1001; denominator += 2) {
    term *= x * x / denominator;
    const next = sum + term;
    if (next === sum) break;
    sum = next;
  }
  const upper = clamp(0.5 + Math.exp(-x * x / 2) * sum / Math.sqrt(2 * Math.PI), 0, 1);
  return z < 0 ? 1 - upper : upper;
}

/** Floor first, then restrict the report to the three feasible ages. */
export function coarsePrediction(prediction) {
  return clamp(YEAR * Math.floor(prediction / YEAR), 0, LIMIT);
}

export function randomModelSelection(candidateCount = 32768, validationCount = 10) {
  if (!Number.isSafeInteger(candidateCount) || candidateCount < 1) {
    throw new RangeError('candidateCount must be a positive integer.');
  }
  if (!Number.isInteger(validationCount) || validationCount < 0 || validationCount >= 20) {
    throw new RangeError('validationCount must be an integer from 0 to 19, leaving an untouched test set.');
  }
  const singlePerfectProbability = 2 ** -validationCount;
  const testCount = 20 - validationCount;
  return {
    singlePerfectProbability,
    perfectProbability: -Math.expm1(candidateCount * Math.log1p(-singlePerfectProbability)),
    expectedPerfect: candidateCount * singlePerfectProbability,
    testCount,
    testAccuracy: 0.5,
    testSD: Math.sqrt(0.25 / testCount),
  };
}

/** Independent Box–Muller draws. Carnie 3 coarsens firstSlopes, not a new draw. */
export function sampleCarnies(random = Math.random, multiplier = 1, count = 10) {
  nonnegative(multiplier, 'multiplier');
  if (!Number.isInteger(count) || count < 0) throw new RangeError('count must be a nonnegative integer.');
  if (typeof random !== 'function') throw new TypeError('random must be a function.');
  function uniform() {
    const draw = random();
    if (!Number.isFinite(draw) || draw < 0 || draw >= 1) throw new RangeError('random must return numbers in [0, 1).');
    return draw;
  }
  function normal() {
    return Math.sqrt(-2 * Math.log(1 - uniform())) * Math.cos(2 * Math.PI * uniform());
  }
  const firstSlopes = [];
  const secondSlopes = [];
  for (let i = 0; i < count; i++) {
    firstSlopes.push(1 + Math.sqrt(multiplier / (100 * LIMIT)) * normal());
    secondSlopes.push(1 + Math.sqrt(multiplier / 100) * normal());
  }
  return { firstSlopes, secondSlopes };
}

/** Prediction moments across training sets, at one fixed population x. */
export function carnieMoments(x, multiplier = 1) {
  nonnegative(multiplier, 'multiplier');
  if (!Number.isFinite(x) || x < 0 || x > LIMIT) throw new RangeError(`x must be between 0 and ${LIMIT}.`);
  const firstVariance = multiplier * x * x / (100 * LIMIT);
  const first = { mean: x, bias: 0, biasSquared: 0, variance: firstVariance };
  const second = { mean: x, bias: 0, biasSquared: 0, variance: multiplier * x * x / 100 };
  let pAtLeast365;
  let pAtLeast730;
  if (firstVariance === 0) {
    pAtLeast365 = Number(x >= YEAR);
    pAtLeast730 = Number(x >= LIMIT);
  } else {
    const sd = Math.sqrt(firstVariance);
    pAtLeast365 = normalCDF((x - YEAR) / sd);
    pAtLeast730 = normalCDF((x - LIMIT) / sd);
  }
  // P(report >= 365) and P(report >= 730) determine all three masses.
  const mean = YEAR * (pAtLeast365 + pAtLeast730);
  const bias = mean - x;
  const secondMoment = YEAR * YEAR * (pAtLeast365 + 3 * pAtLeast730);
  const third = {
    mean,
    bias,
    biasSquared: bias * bias,
    variance: Math.max(0, secondMoment - mean * mean),
    pAtLeast365,
    pAtLeast730,
  };
  return [first, second, third];
}

// Integrate both moments together on the normalized uniform support [0, 1].
// Segment boundaries include both reporting thresholds, so narrow transitions
// remain visible to the adaptive rule even when the training noise is small.
function integratedCoarseMoments(multiplier) {
  if (multiplier === 0) return [YEAR * YEAR / 3, 0];
  const evaluate = (t) => {
    const { biasSquared, variance } = carnieMoments(LIMIT * t, multiplier)[2];
    return [biasSquared, variance];
  };
  const simpson = (a, b, fa, fm, fb) => fa.map((value, i) => (b - a) * (value + 4 * fm[i] + fb[i]) / 6);
  function adaptive(a, b, fa, fm, fb, whole, tolerance, remaining) {
    const midpoint = (a + b) / 2;
    const fl = evaluate((a + midpoint) / 2);
    const fr = evaluate((midpoint + b) / 2);
    const left = simpson(a, midpoint, fa, fl, fm);
    const right = simpson(midpoint, b, fm, fr, fb);
    const delta = whole.map((value, i) => left[i] + right[i] - value);
    if (remaining === 0 || delta.every(value => Math.abs(value) <= 15 * tolerance)) {
      return delta.map((value, i) => left[i] + right[i] + value / 15);
    }
    const first = adaptive(a, midpoint, fa, fl, fm, left, tolerance / 2, remaining - 1);
    const second = adaptive(midpoint, b, fm, fr, fb, right, tolerance / 2, remaining - 1);
    return first.map((value, i) => value + second[i]);
  }
  const result = [0, 0];
  for (let segment = 0; segment < 8; segment++) {
    const a = segment / 8;
    const b = (segment + 1) / 8;
    const fa = evaluate(a);
    const fm = evaluate((a + b) / 2);
    const fb = evaluate(b);
    const piece = adaptive(a, b, fa, fm, fb, simpson(a, b, fa, fm, fb), 1e-7, 24);
    result[0] += piece[0];
    result[1] += piece[1];
  }
  return result;
}

/** Population-average decomposition; meanMSE excludes new-outcome noise. */
export function integratedCarnies(multiplier = 1) {
  nonnegative(multiplier, 'multiplier');
  const noise = multiplier * LIMIT / 200;
  const [coarseBiasSquared, coarseVariance] = integratedCoarseMoments(multiplier);
  return [
    [0, multiplier * LIMIT / 300],
    [0, multiplier * LIMIT * LIMIT / 300],
    [coarseBiasSquared, coarseVariance],
  ].map(([biasSquared, variance]) => ({
    biasSquared,
    variance,
    noise,
    meanMSE: biasSquared + variance,
    mse: biasSquared + variance + noise,
  }));
}

export function tutoringRows() {
  return [
    { group: 'Low', count: 12, D: 0, Y: 50, yhat: 52.5, ehat: 0.25 },
    { group: 'Low', count: 4, D: 1, Y: 60, yhat: 52.5, ehat: 0.25 },
    { group: 'High', count: 4, D: 0, Y: 90, yhat: 97.5, ehat: 0.75 },
    { group: 'High', count: 12, D: 1, Y: 100, yhat: 97.5, ehat: 0.75 },
  ].map(row => ({ ...row, R: row.Y - row.yhat, T: row.D - row.ehat }));
}

export function tutoringSummary() {
  const rows = tutoringRows();
  const groupMean = (field, treatment) => {
    const group = rows.filter(row => row.D === treatment);
    return group.reduce((sum, row) => sum + row.count * row[field], 0)
      / group.reduce((sum, row) => sum + row.count, 0);
  };
  const numerator = rows.reduce((sum, row) => sum + row.count * row.T * row.R, 0);
  const denominator = rows.reduce((sum, row) => sum + row.count * row.T * row.T, 0);
  return {
    rawDifference: groupMean('Y', 1) - groupMean('Y', 0),
    ate: 10,
    outcomeOnly: groupMean('R', 1) - groupMean('R', 0),
    numerator,
    denominator,
    both: numerator / denominator,
  };
}
