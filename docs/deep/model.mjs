// This is an illustrative construction, not a trained or parameter-matched
// comparison. The first-layer features stay fixed when the layer is added.
export const DEFAULT_PINCH = 2;
export const PINCH_RANGE = Object.freeze([0.5, 3]);
export const INPUT_EXTENT = 1.6;
export const OUTPUT_WEIGHTS = Object.freeze([1, 0.3, 0.3]);
export const SURFACE_MODES = Object.freeze([
  'h11', 'h12', 'h13', 'shallow', 'preactivation',
  'unit1', 'unit2', 'unit3', 'combined',
]);

export const relu = value => Math.max(0, value);

export function validatePinch(pinch = DEFAULT_PINCH) {
  if (!Number.isFinite(pinch) || pinch < PINCH_RANGE[0] || pinch > PINCH_RANGE[1]) {
    throw new RangeError(`pinch must be between ${PINCH_RANGE[0]} and ${PINCH_RANGE[1]}`);
  }
  return pinch;
}

export function networkParameters({ pinch = DEFAULT_PINCH } = {}) {
  validatePinch(pinch);
  return {
    beta0: [0, 0, 0],
    omega0: [[1, 0], [0, 1], [-1, -1]],
    beta1: [1, 1, 1],
    omega1: [[-1, -1, -1], [-pinch, -1, -1], [-1, -2, -1]],
    beta2: 0,
    omega2: [...OUTPUT_WEIGHTS],
  };
}

export function evaluateNetwork(x1, x2, { pinch = DEFAULT_PINCH } = {}) {
  if (!Number.isFinite(x1) || !Number.isFinite(x2)) {
    throw new TypeError('The two inputs must be finite numbers');
  }
  validatePinch(pinch);
  const firstPreactivation = [x1, x2, -x1 - x2];
  const h1 = firstPreactivation.map(relu);
  const secondPreactivation = [
    1 - h1[0] - h1[1] - h1[2],
    1 - pinch * h1[0] - h1[1] - h1[2],
    1 - h1[0] - 2 * h1[1] - h1[2],
  ];
  const h2 = secondPreactivation.map(relu);
  const combined = h2.reduce((sum, value, index) => sum + OUTPUT_WEIGHTS[index] * value, 0);
  return {
    x: [x1, x2], firstPreactivation, h1, secondPreactivation, h2,
    h11: h1[0], h12: h1[1], h13: h1[2],
    shallow: secondPreactivation[0], preactivation: secondPreactivation[0],
    unit1: h2[0], unit2: h2[1], unit3: h2[2], combined, y: combined,
  };
}

export function surfaceValue(mode, x1, x2, options = {}) {
  if (!SURFACE_MODES.includes(mode)) throw new RangeError(`Unknown surface mode: ${mode}`);
  return evaluateNetwork(x1, x2, options)[mode];
}

// Within any of the six input cones and any fixed second-layer activation
// pattern, every displayed surface is exactly affine. Geometry uses these
// coefficients directly instead of approximating a curved or gridded surface.
export function affineSurface(mode, activeFirst, activeSecond, { pinch = DEFAULT_PINCH } = {}) {
  if (!SURFACE_MODES.includes(mode)) throw new RangeError(`Unknown surface mode: ${mode}`);
  const { omega0, omega1 } = networkParameters({ pinch });
  const first = omega0.map((row, index) => activeFirst[index] ? [...row] : [0, 0]);
  const second = omega1.map(row => ({
    gradient: [0, 1].map(axis => row.reduce((sum, weight, index) => sum + weight * first[index][axis], 0)),
    intercept: 1,
  }));
  if (mode.startsWith('h1')) return { gradient: first[Number(mode.at(-1)) - 1], intercept: 0 };
  if (mode === 'shallow' || mode === 'preactivation') return second[0];
  if (mode.startsWith('unit')) {
    const index = Number(mode.at(-1)) - 1;
    return activeSecond[index] ? second[index] : { gradient: [0, 0], intercept: 0 };
  }
  return {
    gradient: [0, 1].map(axis => second.reduce((sum, item, index) =>
      sum + (activeSecond[index] ? OUTPUT_WEIGHTS[index] * item.gradient[axis] : 0), 0)),
    intercept: second.reduce((sum, item, index) =>
      sum + (activeSecond[index] ? OUTPUT_WEIGHTS[index] * item.intercept : 0), 0),
  };
}
