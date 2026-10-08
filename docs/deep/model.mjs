// Original folding construction in Prince's h / beta / Omega notation.
// Parameter counts use standard dense weight/bias slots, including repeated
// values and zeros. These chosen coefficients are not a training result.
export const DEFAULT_THRESHOLD = 0.5;
export const THRESHOLD_RANGE = Object.freeze([0.35, 0.65]);
export const INPUT_DOMAIN = Object.freeze([-0.25, 3.25]);
export const FIRST_LAYER_KNOTS = Object.freeze([0, 1, 2]);
export const OUTPUT_WEIGHTS = Object.freeze([1, -2, 2]);
export const SURFACE_MODES = Object.freeze([
  'h11', 'h12', 'h13', 'fold', 'shallow', 'preactivation',
  'unit1', 'unit2', 'unit3', 'combined',
]);

export const relu = value => Math.max(0, value);

export function parameterCount(widths) {
  if (!Array.isArray(widths) || widths.length < 2 || !widths.every(width => Number.isInteger(width) && width > 0)) {
    throw new RangeError('widths must list positive integer layer sizes, including input and output');
  }
  return widths.slice(1).reduce((sum, width, index) => sum + (widths[index] + 1) * width, 0);
}

export const PARAMETER_COUNTS = Object.freeze({
  deep: parameterCount([1, 3, 3, 1]),
  shallowMatch: parameterCount([1, 10, 1]),
  shallowSameBudgetWidth: 7,
  shallowSameBudget: parameterCount([1, 7, 1]),
});

export function validateThreshold(threshold = DEFAULT_THRESHOLD) {
  if (!Number.isFinite(threshold) || threshold < THRESHOLD_RANGE[0] || threshold > THRESHOLD_RANGE[1]) {
    throw new RangeError('threshold must be between ' + THRESHOLD_RANGE[0] + ' and ' + THRESHOLD_RANGE[1]);
  }
  return threshold;
}

export function networkParameters({ threshold = DEFAULT_THRESHOLD } = {}) {
  validateThreshold(threshold);
  return {
    beta0: [0, -1, -2],
    omega0: [[1], [1], [1]],
    beta1: [-0.2, -threshold, -0.8],
    omega1: [[1, -2, 2], [1, -2, 2], [1, -2, 2]],
    beta2: 0,
    omega2: [...OUTPUT_WEIGHTS],
  };
}

export function foldedValue(q, { threshold = DEFAULT_THRESHOLD } = {}) {
  if (!Number.isFinite(q)) throw new TypeError('The folded coordinate must be a finite number');
  validateThreshold(threshold);
  return relu(q - 0.2) - 2 * relu(q - threshold) + 2 * relu(q - 0.8);
}

// Branch locations on the three nonconstant pieces in [0, 3]. Endpoint
// locations may repeat. q=0 also has the constant x<=0 branch outside this list.
export function foldedPreimages(q) {
  if (!Number.isFinite(q) || q < 0 || q > 1) {
    throw new RangeError('q must be between 0 and 1 for the three folded branches');
  }
  return [q, 2 - q, 2 + q];
}

export function evaluateNetwork(x, { threshold = DEFAULT_THRESHOLD } = {}) {
  if (!Number.isFinite(x)) throw new TypeError('The input must be a finite number');
  validateThreshold(threshold);
  const firstPreactivation = [x, x - 1, x - 2];
  const h1 = firstPreactivation.map(relu);
  const fold = h1[0] - 2 * h1[1] + 2 * h1[2];
  const secondPreactivation = [fold - 0.2, fold - threshold, fold - 0.8];
  const h2 = secondPreactivation.map(relu);
  const combined = h2.reduce((sum, value, index) => sum + OUTPUT_WEIGHTS[index] * value, 0);
  return {
    x, firstPreactivation, h1, secondPreactivation, h2,
    h11: h1[0], h12: h1[1], h13: h1[2],
    fold, q: fold, shallow: fold, preactivation: secondPreactivation[1],
    unit1: h2[0], unit2: h2[1], unit3: h2[2], combined, y: combined,
  };
}

export function surfaceValue(mode, x, options = {}) {
  if (!SURFACE_MODES.includes(mode)) throw new RangeError('Unknown curve mode: ' + mode);
  return evaluateNetwork(x, options)[mode];
}

export function preactivationRoots({ threshold = DEFAULT_THRESHOLD } = {}) {
  validateThreshold(threshold);
  return [0.2, threshold, 0.8].map(foldedPreimages);
}

// Exact affine coefficients on a fixed input interval.
export function affineSurface(mode, activeFirst, activeSecond, { threshold = DEFAULT_THRESHOLD } = {}) {
  if (!SURFACE_MODES.includes(mode)) throw new RangeError('Unknown curve mode: ' + mode);
  const parameters = networkParameters({ threshold });
  const first = parameters.beta0.map((bias, index) => ({
    slope: activeFirst[index] ? 1 : 0,
    intercept: activeFirst[index] ? bias : 0,
  }));
  const foldWeights = parameters.omega1[0];
  const fold = {
    slope: foldWeights.reduce((sum, weight, index) => sum + weight * first[index].slope, 0),
    intercept: foldWeights.reduce((sum, weight, index) => sum + weight * first[index].intercept, 0),
  };
  const second = parameters.beta1.map(bias => ({ slope: fold.slope, intercept: fold.intercept + bias }));
  if (mode.startsWith('h1')) return first[Number(mode.at(-1)) - 1];
  if (mode === 'fold' || mode === 'shallow') return fold;
  if (mode === 'preactivation') return second[1];
  if (mode.startsWith('unit')) {
    const index = Number(mode.at(-1)) - 1;
    return activeSecond[index] ? second[index] : { slope: 0, intercept: 0 };
  }
  return {
    slope: second.reduce((sum, item, index) => sum + (activeSecond[index] ? OUTPUT_WEIGHTS[index] * item.slope : 0), 0),
    intercept: second.reduce((sum, item, index) => sum + (activeSecond[index] ? OUTPUT_WEIGHTS[index] * item.intercept : 0), 0),
  };
}
