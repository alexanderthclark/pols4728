// Original illustrative construction using Prince's h / beta / Omega notation.
// These weights are chosen, not trained. Adding the layer changes the number
// of units and parameters; this is not an equal-parameter comparison.
export const DEFAULT_THRESHOLD = 0.3;
export const THRESHOLD_RANGE = Object.freeze([0.2, 0.8]);
export const INPUT_DOMAIN = Object.freeze([-0.25, 3.25]);
export const FIRST_LAYER_KNOTS = Object.freeze([0, 1, 2]);
export const OUTPUT_WEIGHTS = Object.freeze([1, 0.3, 0.3]);
export const SURFACE_MODES = Object.freeze([
  'h11', 'h12', 'h13', 'shallow', 'preactivation',
  'unit1', 'unit2', 'unit3', 'combined',
]);

export const relu = value => Math.max(0, value);

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
    beta1: [-0.5, -threshold, -0.6],
    omega1: [[1, -2, 2], [1, -3, 4], [1, -1.5, 1]],
    beta2: 0,
    omega2: [...OUTPUT_WEIGHTS],
  };
}

export function evaluateNetwork(x, { threshold = DEFAULT_THRESHOLD } = {}) {
  if (!Number.isFinite(x)) throw new TypeError('The input must be a finite number');
  validateThreshold(threshold);
  const firstPreactivation = [x, x - 1, x - 2];
  const h1 = firstPreactivation.map(relu);
  const secondPreactivation = [
    -0.5 + h1[0] - 2 * h1[1] + 2 * h1[2],
    -threshold + h1[0] - 3 * h1[1] + 4 * h1[2],
    -0.6 + h1[0] - 1.5 * h1[1] + h1[2],
  ];
  const h2 = secondPreactivation.map(relu);
  const combined = h2.reduce((sum, value, index) => sum + OUTPUT_WEIGHTS[index] * value, 0);
  return {
    x, firstPreactivation, h1, secondPreactivation, h2,
    h11: h1[0], h12: h1[1], h13: h1[2],
    shallow: secondPreactivation[0], preactivation: secondPreactivation[0],
    unit1: h2[0], unit2: h2[1], unit3: h2[2], combined, y: combined,
  };
}

export function surfaceValue(mode, x, options = {}) {
  if (!SURFACE_MODES.includes(mode)) throw new RangeError('Unknown curve mode: ' + mode);
  return evaluateNetwork(x, options)[mode];
}

// Full roots on the real line, irrespective of the viewing domain. Every root
// lies in an open first-layer interval for the allowed threshold range.
export function preactivationRoots({ threshold = DEFAULT_THRESHOLD } = {}) {
  validateThreshold(threshold);
  return [
    [0.5, 1.5, 2.5],
    [threshold, (3 - threshold) / 2, (5 + threshold) / 2],
    [0.6, 1.8, 2.2],
  ];
}

// Exact affine coefficients on an interval with fixed activation patterns.
// Geometry determines each pattern in the interval's interior.
export function affineSurface(mode, activeFirst, activeSecond, { threshold = DEFAULT_THRESHOLD } = {}) {
  if (!SURFACE_MODES.includes(mode)) throw new RangeError('Unknown curve mode: ' + mode);
  const parameters = networkParameters({ threshold });
  const first = parameters.beta0.map((bias, index) => ({
    slope: activeFirst[index] ? 1 : 0,
    intercept: activeFirst[index] ? bias : 0,
  }));
  const second = parameters.omega1.map((row, unit) => ({
    slope: row.reduce((sum, weight, index) => sum + weight * first[index].slope, 0),
    intercept: parameters.beta1[unit]
      + row.reduce((sum, weight, index) => sum + weight * first[index].intercept, 0),
  }));
  if (mode.startsWith('h1')) return first[Number(mode.at(-1)) - 1];
  if (mode === 'shallow' || mode === 'preactivation') return second[0];
  if (mode.startsWith('unit')) {
    const index = Number(mode.at(-1)) - 1;
    return activeSecond[index] ? second[index] : { slope: 0, intercept: 0 };
  }
  return {
    slope: second.reduce((sum, item, index) =>
      sum + (activeSecond[index] ? OUTPUT_WEIGHTS[index] * item.slope : 0), 0),
    intercept: second.reduce((sum, item, index) =>
      sum + (activeSecond[index] ? OUTPUT_WEIGHTS[index] * item.intercept : 0), 0),
  };
}
