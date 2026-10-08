import {
  affineSurface, DEFAULT_THRESHOLD, evaluateNetwork, FIRST_LAYER_KNOTS,
  foldedValue, INPUT_DOMAIN, networkParameters, OUTPUT_WEIGHTS,
  preactivationRoots, SURFACE_MODES, surfaceValue,
} from './model.mjs';

const EPSILON = 1e-10;

function validateDomain(domain) {
  if (!Array.isArray(domain) || domain.length !== 2 || !domain.every(Number.isFinite) || domain[0] >= domain[1]) {
    throw new RangeError('domain must contain two increasing finite endpoints');
  }
}
function uniqueSorted(values) {
  // Preserve stated knots when an analytically generated root differs from
  // one only by floating-point rounding.
  const unique = [];
  for (const value of values) {
    if (!unique.some(existing => Math.abs(value - existing) <= EPSILON)) unique.push(value);
  }
  return unique.sort((left, right) => left - right);
}
function inDomain(value, domain) {
  return value >= domain[0] - EPSILON && value <= domain[1] + EPSILON;
}

function exactCurve(candidates, domain, coefficientsAt, valueAt) {
  validateDomain(domain);
  const initial = uniqueSorted([...domain, ...candidates].filter(value => inDomain(value, domain))
    .map(value => Math.max(domain[0], Math.min(domain[1], value))));
  // An affine output may cross zero between activation roots, especially when
  // output weights have different signs. Add those exact roots as samples.
  const roots = initial.slice(0, -1).flatMap((left, index) => {
    const right = initial[index + 1];
    const coefficients = coefficientsAt(left + (right - left) / 2);
    if (Math.abs(coefficients.slope) <= EPSILON) return [];
    const root = -coefficients.intercept / coefficients.slope;
    return root >= left - EPSILON && root <= right + EPSILON ? [Math.max(left, Math.min(right, root))] : [];
  });
  const locations = uniqueSorted([...initial, ...roots]);
  const points = locations.map(x => [x, valueAt(x)]);
  const segments = locations.slice(0, -1).map((left, index) => {
    const right = locations[index + 1];
    return { left, right, ...coefficientsAt(left + (right - left) / 2) };
  });
  const bends = points.slice(1, -1).flatMap(([x, y], index) => {
    const leftSlope = segments[index].slope, rightSlope = segments[index + 1].slope;
    return Math.abs(rightSlope - leftSlope) > EPSILON ? [{ x, y, leftSlope, rightSlope }] : [];
  });
  // This compatibility field includes crossings and isolated zero contacts,
  // plus boundaries of flat zero regions; it excludes flat-zero interiors.
  const zeroCrossings = points.flatMap(([x, y], index) => {
    if (Math.abs(y) > EPSILON) return [];
    const adjacent = [segments[index - 1], segments[index]].filter(Boolean);
    return adjacent.some(segment => Math.abs(segment.slope) > EPSILON) ? [x] : [];
  });
  return { domain: [...domain], points, segments, bends, zeroCrossings };
}

export function curveGeometry(mode, { threshold = DEFAULT_THRESHOLD, domain = INPUT_DOMAIN } = {}) {
  if (!SURFACE_MODES.includes(mode)) throw new RangeError('Unknown curve mode: ' + mode);
  const secondPreactivationRoots = preactivationRoots({ threshold });
  let relevantRoots = [];
  if (mode === 'preactivation') relevantRoots = secondPreactivationRoots[1];
  else if (mode.startsWith('unit')) relevantRoots = secondPreactivationRoots[Number(mode.at(-1)) - 1];
  else if (mode === 'combined') relevantRoots = secondPreactivationRoots.flat();
  const geometry = exactCurve([...FIRST_LAYER_KNOTS, ...relevantRoots], domain, x => {
    const evaluation = evaluateNetwork(x, { threshold });
    return affineSurface(
      mode,
      evaluation.firstPreactivation.map(value => value > 0),
      evaluation.secondPreactivation.map(value => value > 0),
      { threshold },
    );
  }, x => surfaceValue(mode, x, { threshold }));
  return {
    mode, threshold, ...geometry,
    firstLayerKnots: [...FIRST_LAYER_KNOTS], secondPreactivationRoots,
  };
}

export function foldedCurveGeometry({ threshold = DEFAULT_THRESHOLD, domain = [0, 1] } = {}) {
  const thresholds = networkParameters({ threshold }).beta1.map(value => -value);
  const geometry = exactCurve(thresholds, domain, q => {
    const active = thresholds.map(value => q > value);
    return {
      slope: OUTPUT_WEIGHTS.reduce((sum, weight, index) => sum + (active[index] ? weight : 0), 0),
      intercept: OUTPUT_WEIGHTS.reduce((sum, weight, index) => sum - (active[index] ? weight * thresholds[index] : 0), 0),
    };
  }, q => foldedValue(q, { threshold }));
  return {
    mode: 'folded', coordinate: 'q', threshold, thresholds, ...geometry,
    firstLayerKnots: [], secondPreactivationRoots: thresholds.map(value => [value]),
  };
}
