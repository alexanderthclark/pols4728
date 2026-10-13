// Deterministic synthetic data. The two centered raw score columns are
// orthogonal; scaling makes their sample variances 9 and 1. A 45° rotation
// therefore gives sample covariance [[5, 4], [4, 5]] in the original basis.
const rawScores = [
  [-4, 0],
  [-3, -1], [-3, 1],
  [-2, -1.5], [-2, 0], [-2, 1.5],
  [-1, -2], [-1, 0], [-1, 2],
  [0, -2], [0, -1], [0, -.5], [0, .5], [0, 1], [0, 2],
  [1, -2], [1, 0], [1, 2],
  [2, -1.5], [2, 0], [2, 1.5],
  [3, -1], [3, 1],
  [4, 0],
];

export const PRINCIPAL_ANGLE = 45;
export const EIGENVALUES = Object.freeze([9, 1]);
export const LINE_POINTS = Object.freeze(
  [-2, -1, 0, 1, 2].map(value => Object.freeze([value, value])),
);
export const THREE_D_EIGENVALUES = Object.freeze([9, 1, .25]);
// Matrix rows index original variables; columns are the three new axes.
export const THREE_D_BASIS = Object.freeze([
  [1 / Math.sqrt(3), 1 / Math.sqrt(2), 1 / Math.sqrt(6)],
  [1 / Math.sqrt(3), -1 / Math.sqrt(2), 1 / Math.sqrt(6)],
  [1 / Math.sqrt(3), 0, -2 / Math.sqrt(6)],
].map(row => Object.freeze(row)));

export function direction(angleDegrees) {
  const radians = angleDegrees * Math.PI / 180;
  return [Math.cos(radians), Math.sin(radians)];
}

export function perpendicular(u) {
  return [-u[1], u[0]];
}

export function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1];
}

export function dotND(a, b) {
  return a.reduce((sum, value, index) => sum + value * b[index], 0);
}

const sumSquares = column => rawScores.reduce((sum, row) => sum + row[column] ** 2, 0);
const primaryScale = Math.sqrt(EIGENVALUES[0] * (rawScores.length - 1) / sumSquares(0));
const secondaryScale = Math.sqrt(EIGENVALUES[1] * (rawScores.length - 1) / sumSquares(1));
const principalDirection = direction(PRINCIPAL_ANGLE);
const secondDirection = perpendicular(principalDirection);

export const POINTS = Object.freeze(rawScores.map(([first, second]) => Object.freeze([
  first * primaryScale * principalDirection[0] + second * secondaryScale * secondDirection[0],
  first * primaryScale * principalDirection[1] + second * secondaryScale * secondDirection[1],
])));

// Squared primary scores, centered, are orthogonal to both existing raw
// columns by their symmetry. They add a small third spread without randomness.
const primarySecondMoment = sumSquares(0) / rawScores.length;
const rawThirdScores = rawScores.map(([first]) => first ** 2 - primarySecondMoment);
const thirdScale = Math.sqrt(THREE_D_EIGENVALUES[2] * (rawScores.length - 1)
  / rawThirdScores.reduce((sum, value) => sum + value ** 2, 0));
export const THREE_D_POINTS = Object.freeze(reconstructBasis(
  rawScores.map(([first, second], index) => [
    first * primaryScale, second * secondaryScale, rawThirdScores[index] * thirdScale,
  ]), THREE_D_BASIS,
).map(row => Object.freeze(row)));

// Removing a unit direction subtracts its rank-one reconstruction from each
// observation. The remaining rows all lie in the perpendicular subspace.
export function deflate(points, u) {
  return points.map(point => {
    const score = dotND(point, u);
    return point.map((value, index) => value - score * u[index]);
  });
}

const threeDPrimaryDirection = THREE_D_BASIS.map(row => row[0]);
const residualFirstAxis = THREE_D_BASIS.map(row => row[1]);
const residualSecondAxis = THREE_D_BASIS.map(row => row[2]);

export const THREE_D_RESIDUALS = Object.freeze(
  deflate(THREE_D_POINTS, threeDPrimaryDirection).map(row => Object.freeze(row)),
);
// Coordinates in the remaining plane. Its first axis will become PC2 and its
// second will become PC3; searching within this plane is the same 2D problem.
export const THREE_D_RESIDUAL_COORDINATES = Object.freeze(
  THREE_D_RESIDUALS.map(row => Object.freeze([
    dotND(row, residualFirstAxis), dotND(row, residualSecondAxis),
  ])),
);

export function residualDirection(angleDegrees) {
  const [first, second] = direction(angleDegrees);
  return residualFirstAxis.map((value, index) =>
    first * value + second * residualSecondAxis[index]);
}

export function residualSummary(angleDegrees) {
  const u = residualDirection(angleDegrees);
  const v = residualDirection(angleDegrees + 90);
  const scores = THREE_D_RESIDUALS.map(point => dotND(point, u));
  const otherScores = THREE_D_RESIDUALS.map(point => dotND(point, v));
  const variance = sampleVariance(scores);
  const otherVariance = sampleVariance(otherScores);
  return {
    u, v, scores, otherScores, variance, otherVariance,
    totalVariance: variance + otherVariance,
    crossCovariance: sampleCovariance(scores, otherScores),
  };
}

export function project(points, u) {
  return points.map(point => dot(point, u));
}

const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;

function sampleCovariance(first, second) {
  if (first.length < 2) throw new RangeError('Sample covariance requires at least two observations.');
  const firstMean = mean(first);
  const secondMean = mean(second);
  return first.reduce((sum, value, index) =>
    sum + (value - firstMean) * (second[index] - secondMean), 0) / (first.length - 1);
}

export function sampleVariance(scores) {
  return sampleCovariance(scores, scores);
}

export function covariance(points) {
  const first = points.map(point => point[0]);
  const second = points.map(point => point[1]);
  const cross = sampleCovariance(first, second);
  return [[sampleVariance(first), cross], [cross, sampleVariance(second)]];
}

export const SAMPLE_COVARIANCE = Object.freeze(covariance(POINTS).map(row => Object.freeze(row)));

export function quadraticVariance(S, u) {
  return u[0] * (S[0][0] * u[0] + S[0][1] * u[1])
    + u[1] * (S[1][0] * u[0] + S[1][1] * u[1]);
}

export function coordinates(points, angleDegrees) {
  const u = direction(angleDegrees);
  const v = perpendicular(u);
  return points.map(point => [dot(point, u), dot(point, v)]);
}

// For a unit direction, keep the coordinate along u and discard the other one.
export function reconstruct(scores, u) {
  return scores.map(score => [score * u[0], score * u[1]]);
}

// Row observations multiplied by the basis matrix yield new coordinates.
export function changeBasis(points, basis) {
  return points.map(point => basis[0].map((_, column) =>
    point.reduce((sum, value, row) => sum + value * basis[row][column], 0)));
}

// An orthonormal basis inverts by transposition; omit later coordinates to
// reconstruct from only the requested number of components.
export function reconstructBasis(scores, basis, components = basis.length) {
  return scores.map(score => basis.map(row => row.slice(0, components)
    .reduce((sum, loading, column) => sum + loading * score[column], 0)));
}

// Expose the SVD factors for centered observations whose known orthonormal
// principal axes are supplied as basis columns. This is a teaching example,
// not a general SVD solver. Nonzero orthogonal score columns normalized to
// length one give P, and their lengths give the singular values d:
// X = P diag(d) Qᵀ, with Q = basis and scores = XQ = P diag(d).
export function svdExample(points = THREE_D_POINTS, basis = THREE_D_BASIS) {
  const scores = changeBasis(points, basis);
  const singularValues = basis[0].map((_, column) => Math.sqrt(
    scores.reduce((sum, row) => sum + row[column] ** 2, 0),
  ));
  const P = scores.map(row => row.map((value, column) =>
    singularValues[column] === 0 ? 0 : value / singularValues[column]));
  return {P, singularValues, Q: basis, scores};
}

export function pcaSummary(angleDegrees, points = POINTS) {
  const u = direction(angleDegrees);
  const v = perpendicular(u);
  const scores = project(points, u);
  const otherScores = project(points, v);
  const variance = sampleVariance(scores);
  const otherVariance = sampleVariance(otherScores);
  const S = covariance(points);
  const totalVariance = S[0][0] + S[1][1];
  const reconstruction = reconstruct(scores, u);
  const meanSquaredError = points.reduce((sum, point, index) =>
    sum + (point[0] - reconstruction[index][0]) ** 2
      + (point[1] - reconstruction[index][1]) ** 2, 0) / points.length;
  return {
    u, v, scores, otherScores, variance, otherVariance,
    crossCovariance: sampleCovariance(scores, otherScores),
    totalVariance,
    retainedFraction: variance / totalVariance,
    reconstruction,
    meanSquaredError,
  };
}
