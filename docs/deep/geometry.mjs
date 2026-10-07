import {
  affineSurface, DEFAULT_THRESHOLD, evaluateNetwork, FIRST_LAYER_KNOTS,
  INPUT_DOMAIN, preactivationRoots, SURFACE_MODES, surfaceValue,
} from './model.mjs';

const EPSILON = 1e-10;

function uniqueSorted(values) {
  return values.sort((left, right) => left - right).filter((value, index, sorted) =>
    index === 0 || Math.abs(value - sorted[index - 1]) > EPSILON);
}

export function curveGeometry(mode, { threshold = DEFAULT_THRESHOLD, domain = INPUT_DOMAIN } = {}) {
  if (!SURFACE_MODES.includes(mode)) throw new RangeError('Unknown curve mode: ' + mode);
  if (!Array.isArray(domain) || domain.length !== 2 || !domain.every(Number.isFinite) || domain[0] >= domain[1]) {
    throw new RangeError('domain must contain two increasing finite endpoints');
  }
  const secondPreactivationRoots = preactivationRoots({ threshold });
  let relevantRoots = [];
  if (mode === 'shallow' || mode === 'preactivation') relevantRoots = secondPreactivationRoots[0];
  else if (mode.startsWith('unit')) relevantRoots = secondPreactivationRoots[Number(mode.at(-1)) - 1];
  else if (mode === 'combined') relevantRoots = secondPreactivationRoots.flat();

  const locations = uniqueSorted([
    ...domain,
    ...FIRST_LAYER_KNOTS,
    ...relevantRoots,
  ].filter(value => value >= domain[0] - EPSILON && value <= domain[1] + EPSILON)
    .map(value => Math.max(domain[0], Math.min(domain[1], value))));
  const points = locations.map(x => [x, surfaceValue(mode, x, { threshold })]);
  const segments = locations.slice(0, -1).map((left, index) => {
    const right = locations[index + 1];
    const evaluation = evaluateNetwork(left + (right - left) / 2, { threshold });
    const coefficients = affineSurface(
      mode,
      evaluation.firstPreactivation.map(value => value > 0),
      evaluation.secondPreactivation.map(value => value > 0),
      { threshold },
    );
    return { left, right, ...coefficients };
  });
  const bends = points.slice(1, -1).flatMap(([x, y], index) => {
    const leftSlope = segments[index].slope;
    const rightSlope = segments[index + 1].slope;
    return Math.abs(rightSlope - leftSlope) > EPSILON ? [{ x, y, leftSlope, rightSlope }] : [];
  });

  // Zero transitions are crossings for a signed preactivation/readout, and
  // positive-to-zero boundaries for a clipped response. A wholly flat zero
  // interval supplies no additional crossings at its internal sample points.
  const zeroCrossings = points.flatMap(([x, y], index) => {
    if (Math.abs(y) > EPSILON) return [];
    const adjacent = [segments[index - 1], segments[index]].filter(Boolean);
    return adjacent.some(segment => Math.abs(segment.slope) > EPSILON) ? [x] : [];
  });
  return {
    mode, threshold, domain: [...domain], points, bends, zeroCrossings, segments,
    firstLayerKnots: [...FIRST_LAYER_KNOTS],
    secondPreactivationRoots,
  };
}
