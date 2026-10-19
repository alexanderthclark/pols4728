import assert from 'node:assert/strict';
import test from 'node:test';
import {
  POINTS, SAMPLE_COVARIANCE, PRINCIPAL_ANGLE, EIGENVALUES,
  direction, perpendicular, dot, project, sampleVariance, covariance,
  quadraticVariance, coordinates, reconstruct, pcaSummary,
  LINE_POINTS, THREE_D_POINTS, THREE_D_BASIS, THREE_D_EIGENVALUES,
  changeBasis, reconstructBasis, dotND, deflate,
  THREE_D_RESIDUALS, THREE_D_RESIDUAL_COORDINATES,
  residualDirection, residualSummary, svdExample,
} from '../docs/pca/model.mjs';

const close = (actual, expected, label = '') => assert.ok(
  Number.isFinite(actual) && Math.abs(actual - expected) < 1e-10,
  `${label}: ${actual} differs from ${expected}`,
);
const average = values => values.reduce((sum, value) => sum + value, 0) / values.length;
const manualCovariance = (first, second) => {
  const a = average(first);
  const b = average(second);
  return first.reduce((sum, value, index) => sum + (value - a) * (second[index] - b), 0)
    / (first.length - 1);
};
const squaredDistance = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;

test('24 centered observations have the declared sample covariance', () => {
  assert.equal(POINTS.length, 24);
  assert.equal(PRINCIPAL_ANGLE, 45);
  assert.deepEqual(EIGENVALUES, [9, 1]);
  const columns = [0, 1].map(column => POINTS.map(point => point[column]));
  columns.forEach(column => close(average(column), 0, 'column mean'));
  const expected = [[5, 4], [4, 5]];
  const calculated = covariance(POINTS);
  for (let row = 0; row < 2; row++) {
    for (let column = 0; column < 2; column++) {
      close(manualCovariance(columns[row], columns[column]), expected[row][column], 'independent covariance');
      close(calculated[row][column], expected[row][column], 'calculated covariance');
      close(SAMPLE_COVARIANCE[row][column], expected[row][column], 'exported covariance');
    }
  }
});

test('variance and covariance center their input and use n − 1', () => {
  close(sampleVariance([1, 2, 6]), 7);
  close(sampleVariance([101, 102, 106]), 7);
  assert.deepEqual(covariance([[3, 8], [5, 4], [7, 6]]), [[4, -2], [-2, 4]]);
  assert.throws(() => sampleVariance([1]), RangeError);
});

test('projected sample variance agrees with uᵀSu at arbitrary angles', () => {
  for (const angle of [-120, 0, 17, 45, 90, 135, 225]) {
    const u = direction(angle);
    const v = perpendicular(u);
    close(dot(u, u), 1, 'unit direction');
    close(dot(v, v), 1, 'unit perpendicular');
    close(dot(u, v), 0, 'orthogonality');
    const expectedVariance = 5 + 4 * Math.sin(2 * angle * Math.PI / 180);
    const summary = pcaSummary(angle);
    close(sampleVariance(project(POINTS, u)), expectedVariance, 'projected variance');
    close(quadraticVariance([[5, 4], [4, 5]], u), expectedVariance, 'quadratic variance');
    close(summary.variance, expectedVariance, 'summary variance');
    close(summary.otherVariance, 10 - expectedVariance, 'perpendicular variance');
    close(summary.crossCovariance, 4 * Math.cos(2 * angle * Math.PI / 180), 'cross covariance');
    close(summary.totalVariance, 10, 'invariant total variance');
  }
});

test('unit length is necessary for comparing directional spread', () => {
  const unit = direction(45);
  const doubled = unit.map(value => 2 * value);
  close(sampleVariance(project(POINTS, unit)), 9);
  close(sampleVariance(project(POINTS, doubled)), 36);
  close(quadraticVariance(SAMPLE_COVARIANCE, doubled), 36);
  close(quadraticVariance(SAMPLE_COVARIANCE, [0, 0]), 0);
});

test('the eigenvector directions give the global maximum and minimum', () => {
  close(pcaSummary(45).variance, 9);
  close(pcaSummary(135).variance, 1);
  close(pcaSummary(0).variance, 5);
  close(pcaSummary(90).variance, 5);
  for (let angle = 0; angle < 180; angle += .5) {
    const spread = pcaSummary(angle).variance;
    assert.ok(spread >= 1 - 1e-10 && spread <= 9 + 1e-10);
  }
  close(pcaSummary(225).variance, pcaSummary(45).variance, 'opposite directions span the same axis');
});

test('an orthonormal change of basis preserves geometry and is invertible', () => {
  for (const angle of [0, 23, 45, 107, -60]) {
    const changed = coordinates(POINTS, angle);
    const u = direction(angle);
    const v = perpendicular(u);
    for (let index = 0; index < POINTS.length; index++) {
      const next = (index + 7) % POINTS.length;
      close(squaredDistance(changed[index], changed[next]), squaredDistance(POINTS[index], POINTS[next]), 'distance');
      close(dot(changed[index], changed[next]), dot(POINTS[index], POINTS[next]), 'dot product');
      close(changed[index][0] * u[0] + changed[index][1] * v[0], POINTS[index][0], 'recovered x₁');
      close(changed[index][0] * u[1] + changed[index][1] * v[1], POINTS[index][1], 'recovered x₂');
    }
  }
});

test('principal scores are uncorrelated, without requiring independence', () => {
  const summary = pcaSummary(45);
  close(manualCovariance(summary.scores, summary.otherScores), 0);
  close(summary.crossCovariance, 0);
  assert.ok(Math.abs(pcaSummary(20).crossCovariance) > 1);
  // In this discrete distribution, every observation at an extreme primary
  // score has secondary score zero, but secondary scores are not always zero.
  // Those unequal conditional and marginal probabilities disprove independence.
  const maximum = Math.max(...summary.scores.map(Math.abs));
  const extreme = summary.scores.map((score, index) => ({score, other: summary.otherScores[index]}))
    .filter(row => Math.abs(Math.abs(row.score) - maximum) < 1e-10);
  const zeroFraction = summary.otherScores.filter(score => Math.abs(score) < 1e-10).length / POINTS.length;
  assert.equal(extreme.length, 2);
  assert.ok(extreme.every(row => Math.abs(row.other) < 1e-10));
  close(zeroFraction, .25);
});

test('one-dimensional reconstruction drops exactly the perpendicular spread', () => {
  for (const angle of [0, 20, 45, 90, 135]) {
    const summary = pcaSummary(angle);
    const fitted = reconstruct(project(POINTS, summary.u), summary.u);
    let sumSquaredError = 0;
    POINTS.forEach((point, index) => {
      close(fitted[index][0], summary.reconstruction[index][0]);
      close(fitted[index][1], summary.reconstruction[index][1]);
      const residual = [point[0] - fitted[index][0], point[1] - fitted[index][1]];
      close(dot(residual, summary.u), 0, 'residual orthogonal to retained axis');
      close(squaredDistance(point, fitted[index]), summary.otherScores[index] ** 2, 'discarded squared score');
      sumSquaredError += squaredDistance(point, fitted[index]);
    });
    close(summary.meanSquaredError, sumSquaredError / POINTS.length);
    close(sumSquaredError / (POINTS.length - 1), summary.otherVariance, 'sample variance of discarded score');
    close(summary.meanSquaredError, (POINTS.length - 1) / POINTS.length * summary.otherVariance, 'n versus n − 1');
    close(summary.retainedFraction, summary.variance / 10);
  }
  close(pcaSummary(45).retainedFraction, .9);
  close(pcaSummary(45).meanSquaredError, 23 / 24);
  assert.ok(pcaSummary(45).meanSquaredError < pcaSummary(0).meanSquaredError);
  assert.ok(pcaSummary(45).meanSquaredError < pcaSummary(135).meanSquaredError);
});

test('the exact x₁ = x₂ example needs only one component and loses nothing', () => {
  assert.deepEqual(LINE_POINTS, [[-2, -2], [-1, -1], [0, 0], [1, 1], [2, 2]]);
  assert.deepEqual(covariance(LINE_POINTS), [[2.5, 2.5], [2.5, 2.5]]);
  const summary = pcaSummary(45, LINE_POINTS);
  close(summary.variance, 5);
  close(summary.otherVariance, 0);
  close(summary.totalVariance, 5);
  close(summary.retainedFraction, 1);
  close(summary.meanSquaredError, 0);
  LINE_POINTS.forEach((point, index) => {
    close(summary.scores[index], Math.sqrt(2) * point[0]);
    close(summary.otherScores[index], 0);
    close(summary.reconstruction[index][0], point[0]);
    close(summary.reconstruction[index][1], point[1]);
  });
  close(pcaSummary(0, LINE_POINTS).retainedFraction, .5);
  close(pcaSummary(0, LINE_POINTS).meanSquaredError, 2);
});

test('the three-dimensional orthonormal basis is invertible and preserves distances', () => {
  assert.equal(THREE_D_POINTS.length, 24);
  assert.deepEqual(THREE_D_EIGENVALUES, [9, 1, .25]);
  const u1 = THREE_D_BASIS.map(row => row[0]);
  const u2 = THREE_D_BASIS.map(row => row[1]);
  const u3 = THREE_D_BASIS.map(row => row[2]);
  u1.forEach(value => close(value, 1 / Math.sqrt(3)));
  u2.forEach((value, index) => close(value, [1, -1, 0][index] / Math.sqrt(2)));
  u3.forEach((value, index) => close(value, [1, 1, -2][index] / Math.sqrt(6)));
  const axes = [u1, u2, u3];
  axes.forEach((axis, index) => axes.forEach((other, second) => {
    close(axis.reduce((sum, value, row) => sum + value * other[row], 0), index === second ? 1 : 0);
  }));
  const scores = changeBasis(THREE_D_POINTS, THREE_D_BASIS);
  const recovered = reconstructBasis(scores, THREE_D_BASIS);
  const distance3D = (a, b) => a.reduce((sum, value, index) => sum + (value - b[index]) ** 2, 0);
  THREE_D_POINTS.forEach((point, index) => {
    point.forEach((value, column) => close(recovered[index][column], value, '3D inverse'));
    const next = (index + 7) % THREE_D_POINTS.length;
    close(distance3D(point, THREE_D_POINTS[next]), distance3D(scores[index], scores[next]), '3D distance');
  });
});

test('three-dimensional principal scores have variances 9, 1, and .25, and zero cross covariance', () => {
  const scores = changeBasis(THREE_D_POINTS, THREE_D_BASIS);
  const columns = [0, 1, 2].map(column => scores.map(point => point[column]));
  for (let row = 0; row < 3; row++) {
    close(average(THREE_D_POINTS.map(point => point[row])), 0, 'centered original coordinate');
    close(average(columns[row]), 0, 'centered score');
    for (let column = 0; column < 3; column++) {
      close(manualCovariance(columns[row], columns[column]), row === column ? [9, 1, .25][row] : 0);
    }
  }
});

test('two components retain 40/41 of 3D variance and discard squared error 5.75', () => {
  const scores = changeBasis(THREE_D_POINTS, THREE_D_BASIS);
  const fitted = reconstructBasis(scores, THREE_D_BASIS, 2);
  const spread = [0, 1, 2].map(column => manualCovariance(
    scores.map(row => row[column]), scores.map(row => row[column]),
  ));
  close((spread[0] + spread[1]) / spread.reduce((sum, value) => sum + value, 0), 40 / 41);
  const sumSquaredError = THREE_D_POINTS.reduce((sum, point, index) => sum
    + point.reduce((rowSum, value, column) => rowSum + (value - fitted[index][column]) ** 2, 0), 0);
  close(sumSquaredError, 5.75);
  close(sumSquaredError, (THREE_D_POINTS.length - 1) * spread[2]);
  THREE_D_POINTS.forEach((point, index) => {
    const residual = point.map((value, column) => value - fitted[index][column]);
    close(residual.reduce((sum, value) => sum + value ** 2, 0), scores[index][2] ** 2);
  });
});

test('deflation subtracts the rank-one PC1 reconstruction and leaves a centered perpendicular plane', () => {
  const u1 = THREE_D_BASIS.map(row => row[0]);
  const residuals = deflate(THREE_D_POINTS, u1);
  assert.deepEqual(residuals, THREE_D_RESIDUALS);
  residuals.forEach((residual, index) => {
    close(dotND(residual, u1), 0, 'residual perpendicular to PC1');
    const score = THREE_D_POINTS[index].reduce((sum, value, column) =>
      sum + value * u1[column], 0);
    residual.forEach((value, column) => {
      close(value + score * u1[column], THREE_D_POINTS[index][column], 'PC1 part plus residual');
    });
  });
  [0, 1, 2].forEach(column => close(average(residuals.map(row => row[column])), 0, 'residual mean'));
  // The helper works for unit directions in any number of coordinates.
  assert.deepEqual(deflate([[2, 3, 4, 5]], [0, 1, 0, 0]), [[2, 0, 4, 5]]);
});

test('removing PC1 leaves exactly the original PC2 and PC3 scores', () => {
  const residualScores = changeBasis(THREE_D_RESIDUALS, THREE_D_BASIS);
  const originalScores = changeBasis(THREE_D_POINTS, THREE_D_BASIS);
  residualScores.forEach((row, index) => {
    close(row[0], 0, 'zero PC1 coordinate');
    close(row[1], originalScores[index][1], 'unchanged PC2 coordinate');
    close(row[2], originalScores[index][2], 'unchanged PC3 coordinate');
    close(THREE_D_RESIDUAL_COORDINATES[index][0], row[1], 'plane first coordinate');
    close(THREE_D_RESIDUAL_COORDINATES[index][1], row[2], 'plane second coordinate');
  });
  const columns = [0, 1, 2].map(column => residualScores.map(row => row[column]));
  for (let row = 0; row < 3; row++) {
    for (let column = 0; column < 3; column++) {
      close(manualCovariance(columns[row], columns[column]), row === column ? [0, 1, .25][row] : 0,
        'rank-two residual covariance');
    }
  }
});

test('searching the residual plane finds PC2 and preserves its total variance', () => {
  const u1 = THREE_D_BASIS.map(row => row[0]);
  for (let angle = -180; angle <= 180; angle += 2) {
    const summary = residualSummary(angle);
    const radians = angle * Math.PI / 180;
    const expectedVariance = Math.cos(radians) ** 2 + .25 * Math.sin(radians) ** 2;
    close(dotND(summary.u, summary.u), 1, 'unit residual direction');
    close(dotND(summary.u, u1), 0, 'direction perpendicular to PC1');
    close(dotND(summary.u, summary.v), 0, 'perpendicular residual axes');
    close(summary.variance, expectedVariance, 'directional plane variance');
    close(summary.otherVariance, 1.25 - expectedVariance, 'other plane variance');
    close(summary.totalVariance, 1.25, 'remaining total variance');
    close(summary.crossCovariance, -.75 * Math.cos(radians) * Math.sin(radians), 'plane cross covariance');
    assert.ok(summary.variance >= .25 - 1e-10 && summary.variance <= 1 + 1e-10);
    summary.scores.forEach((score, index) => close(score,
      THREE_D_POINTS[index].reduce((sum, value, column) => sum + value * summary.u[column], 0),
      'projecting residual or original has the same score'));
  }
  const pc2 = residualDirection(0);
  pc2.forEach((value, index) => close(value, THREE_D_BASIS[index][1]));
  close(residualSummary(0).variance, 1, 'PC2 is residual maximum');
  close(residualSummary(90).variance, .25, 'PC3 is residual minimum');
  const remainingLine = deflate(THREE_D_RESIDUALS, pc2);
  const remainingScores = changeBasis(remainingLine, THREE_D_BASIS);
  remainingScores.forEach(row => {
    close(row[0], 0, 'PC1 removed');
    close(row[1], 0, 'PC2 removed');
  });
  close(sampleVariance(remainingScores.map(row => row[2])), .25, 'only PC3 remains');
});

test('SVD singular values encode the principal score variances', () => {
  const {P, singularValues, Q, scores} = svdExample();
  assert.equal(P.length, THREE_D_POINTS.length);
  assert.equal(P[0].length, 3);
  assert.equal(Q, THREE_D_BASIS);
  singularValues.forEach((value, column) => {
    close(value ** 2 / (THREE_D_POINTS.length - 1), THREE_D_EIGENVALUES[column], 'd² / (n − 1)');
    scores.forEach((row, index) => close(P[index][column] * value, row[column], 'scores = P diag(d)'));
  });
  for (let first = 0; first < 3; first++) {
    close(average(P.map(row => row[first])), 0, 'centered left singular vector');
    for (let second = 0; second < 3; second++) {
      const columnDot = P.reduce((sum, row) => sum + row[first] * row[second], 0);
      close(columnDot, first === second ? 1 : 0, 'orthonormal columns of P');
    }
  }
});

test('SVD reconstructs the cloud as three rank-one pieces, with deflation removing the first', () => {
  const {P, singularValues, Q} = svdExample();
  THREE_D_POINTS.forEach((point, index) => point.forEach((value, coordinate) => {
    const pieces = singularValues.map((singularValue, component) =>
      singularValue * P[index][component] * Q[coordinate][component]);
    close(pieces.reduce((sum, piece) => sum + piece, 0), value, 'X = P diag(d) Qᵀ');
    close(pieces[1] + pieces[2], THREE_D_RESIDUALS[index][coordinate], 'residual contains later pieces');
  }));
});
