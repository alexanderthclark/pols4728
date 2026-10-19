import assert from 'node:assert/strict';
import test from 'node:test';
import {
  THREE_D_POINTS, THREE_D_RESIDUALS, THREE_D_BASIS,
  THREE_D_RESIDUAL_COORDINATES, coordinates, dotND, sampleVariance,
} from '../docs/pca/model.mjs';
import {
  RESIDUAL_BASIS, residualCamera, projectWithCamera,
  projectResidualPoint, residualPointAt, assembleComponents,
} from '../docs/pca/storyboard-geometry.mjs';

const close = (actual, expected, label = '') => assert.ok(
  Number.isFinite(actual) && Math.abs(actual - expected) < 1e-9,
  `${label}: ${actual} differs from ${expected}`,
);
const closePoint = (actual, expected, label = '') => actual.forEach((value, index) =>
  close(value, expected[index], `${label} coordinate ${index}`));
const distanceSquared = (first, second) => first.reduce((sum, value, index) =>
  sum + (value - second[index]) ** 2, 0);
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const axes = [0, 1, 2].map(column => THREE_D_BASIS.map(row => row[column]));

test('the residual camera starts at the existing 3D view, with no initial jump', () => {
  const pitch = 22 * Math.PI / 180;
  for (const cameraDegrees of [-60, 0, 25, 90]) {
    const yaw = cameraDegrees * Math.PI / 180;
    const state = residualCamera(0, cameraDegrees);
    closePoint(state.center, [360, 215], 'initial center');
    close(state.scale, 31, 'initial scale');
    THREE_D_RESIDUALS.forEach(point => {
      const [x, y, z] = point;
      const expected = [
        360 + 31 * (x * Math.cos(yaw) - y * Math.sin(yaw)),
        215 - 31 * ((x * Math.sin(yaw) + y * Math.cos(yaw)) * Math.sin(pitch)
          + z * Math.cos(pitch)),
      ];
      closePoint(projectResidualPoint(point, 0, cameraDegrees), expected, 'legacy projection');
    });
  }
});

test('turning the residual plane uses a rigid orthonormal camera throughout', () => {
  const arbitraryPoints = [[2, -5, 4], [-3, 1, .7], [0, 0, 0], [1, 1, 1]];
  for (const cameraDegrees of [-60, 0, 25, 90]) {
    let handedness;
    for (let frame = 0; frame <= 40; frame++) {
      const state = residualCamera(frame / 40, cameraDegrees);
      const frameAxes = [state.horizontal, state.vertical, state.normal];
      frameAxes.forEach((axis, index) => frameAxes.forEach((other, second) =>
        close(dotND(axis, other), index === second ? 1 : 0, 'orthonormal camera')));
      const determinant = dotND(cross(frameAxes[0], frameAxes[1]), frameAxes[2]);
      close(Math.abs(determinant), 1, 'camera handedness');
      handedness ??= Math.sign(determinant);
      close(determinant, handedness, 'camera never becomes a reflection');
      const cameraCoordinates = arbitraryPoints.map(point => frameAxes.map(axis => dotND(point, axis)));
      arbitraryPoints.forEach((point, index) => {
        close(dotND(cameraCoordinates[index], cameraCoordinates[index]), dotND(point, point), 'preserved length');
        const next = (index + 1) % arbitraryPoints.length;
        close(distanceSquared(cameraCoordinates[index], cameraCoordinates[next]),
          distanceSquared(point, arbitraryPoints[next]), 'preserved 3D distance');
      });
    }
  }
});

test('the face-on endpoint matches the 2D search coordinates exactly', () => {
  const expectedCoordinates = coordinates(THREE_D_RESIDUAL_COORDINATES, 30);
  const state = residualCamera(1);
  closePoint(state.center, [360, 179], 'face-on center');
  close(state.scale, 57, 'face-on scale');
  closePoint(state.horizontal, RESIDUAL_BASIS.a, 'horizontal plane direction');
  closePoint(state.vertical, RESIDUAL_BASIS.b, 'vertical plane direction');
  THREE_D_RESIDUALS.forEach((point, index) => {
    const [a, b] = expectedCoordinates[index];
    closePoint(projectResidualPoint(point, 1), [360 + 57 * a, 179 - 57 * b], '2D endpoint');
    closePoint(projectWithCamera(point, state), [360 + 57 * a, 179 - 57 * b], 'shared projection');
    for (const cameraDegrees of [-60, 0, 90]) {
      closePoint(projectResidualPoint(point, 1, cameraDegrees), [360 + 57 * a, 179 - 57 * b], 'same endpoint after camera changes');
    }
  });
});

test('the face-on projection preserves every residual distance up to its declared scale', () => {
  const projected = THREE_D_RESIDUALS.map(point => projectResidualPoint(point, 1));
  THREE_D_RESIDUALS.forEach((point, index) => {
    for (let other = index + 1; other < THREE_D_RESIDUALS.length; other++) {
      close(distanceSquared(projected[index], projected[other]) / 57 ** 2,
        distanceSquared(point, THREE_D_RESIDUALS[other]), 'undistorted residual plane');
    }
  });
});

test('subtracting PC2 travels only along PC2 and ends at the original PC3 contribution', () => {
  for (const amount of [0, .2, .5, .8, 1]) {
    THREE_D_RESIDUALS.forEach((point, index) => {
      const next = residualPointAt(point, amount);
      const originalPC3Score = dotND(THREE_D_POINTS[index], axes[2]);
      close(dotND(next, axes[0]), 0, 'PC1 stays removed');
      close(dotND(next, axes[1]), (1 - amount) * dotND(point, axes[1]), 'PC2 coordinate');
      close(dotND(next, axes[2]), originalPC3Score, 'PC3 coordinate preserved');
      if (amount === 0) closePoint(next, point, 'initial residual');
      if (amount === 1) closePoint(next, axes[2].map(value => value * originalPC3Score), 'PC3 line endpoint');
    });
    const remaining = THREE_D_RESIDUALS.map(point => residualPointAt(point, amount));
    const variancePC2 = sampleVariance(remaining.map(point => dotND(point, axes[1])));
    const variancePC3 = sampleVariance(remaining.map(point => dotND(point, axes[2])));
    close(variancePC2, (1 - amount) ** 2, 'squared decrease of PC2 variance');
    close(variancePC3, .25, 'unchanged PC3 variance');
  }
});

test('camera motion and subtraction clamp at their endpoints and never mutate observations', () => {
  const pointsBefore = structuredClone(THREE_D_POINTS);
  const residualsBefore = structuredClone(THREE_D_RESIDUALS);
  const mutablePoint = [...THREE_D_RESIDUALS[3]];
  const pointBefore = [...mutablePoint];
  for (const amount of [-10, -.1, 0, .5, 1, 1.1, 10]) {
    const clamped = Math.max(0, Math.min(1, amount));
    assert.deepEqual(residualCamera(amount), residualCamera(clamped));
    closePoint(projectResidualPoint(mutablePoint, amount), projectResidualPoint(mutablePoint, clamped));
    closePoint(residualPointAt(mutablePoint, amount), residualPointAt(mutablePoint, clamped));
  }
  assert.deepEqual(mutablePoint, pointBefore);
  assert.deepEqual(THREE_D_POINTS, pointsBefore);
  assert.deepEqual(THREE_D_RESIDUALS, residualsBefore);
});

test('component assembly starts at the origin and reaches each successive reconstruction', () => {
  const endpoints = [
    { progress: 0, weights: [0, 0, 0], active: 0 },
    { progress: 1 / 3, weights: [1, 0, 0], active: 1 },
    { progress: 2 / 3, weights: [1, 1, 0], active: 2 },
    { progress: 1, weights: [1, 1, 1], active: 2 },
  ];
  for (const expected of endpoints) {
    const assembled = assembleComponents(THREE_D_POINTS, THREE_D_BASIS, expected.progress);
    assert.deepEqual(assembled.weights, expected.weights);
    assert.equal(assembled.active, expected.active);
    assembled.points.forEach((point, i) => {
      const expectedPoint = [0, 1, 2].map(coordinate => axes.reduce((sum, axis, component) =>
        sum + expected.weights[component] * dotND(THREE_D_POINTS[i], axis) * axis[coordinate], 0));
      closePoint(point, expectedPoint, 'sum of completed rank-one contributions');
      if (expected.progress === 1) closePoint(point, THREE_D_POINTS[i], 'full reconstruction');
    });
  }
});

test('each assembly phase moves only along its component and pauses before the next', () => {
  for (let component = 0; component < 3; component++) {
    const start = assembleComponents(THREE_D_POINTS, THREE_D_BASIS, component / 3);
    const halfway = assembleComponents(THREE_D_POINTS, THREE_D_BASIS, (component + .35) / 3);
    const held = assembleComponents(THREE_D_POINTS, THREE_D_BASIS, (component + .8) / 3);
    const heldLater = assembleComponents(THREE_D_POINTS, THREE_D_BASIS, (component + .95) / 3);
    assert.equal(halfway.active, component);
    halfway.weights.forEach((weight, j) => close(weight, j < component ? 1 : j === component ? .5 : 0, 'smoothstep midpoint'));
    halfway.points.forEach((point, i) => {
      const movement = point.map((value, j) => value - start.points[i][j]);
      axes.forEach((axis, j) => close(dotND(movement, axis),
        j === component ? .5 * dotND(THREE_D_POINTS[i], axis) : 0, 'only the active direction changes'));
      closePoint(held.points[i], heldLater.points[i], 'phase holds its completed contribution');
    });
  }
});

test('assembly variance is the sum of squared component weights times original score variances', () => {
  const bases = [THREE_D_BASIS, [[0, 1, 0], [0, 0, 1], [1, 0, 0]]];
  for (const basis of bases) {
    const columns = [0, 1, 2].map(j => basis.map(row => row[j]));
    const originalVariances = columns.map(axis => sampleVariance(THREE_D_POINTS.map(point => dotND(point, axis))));
    for (let frame = 0; frame <= 60; frame++) {
      const assembled = assembleComponents(THREE_D_POINTS, basis, frame / 60);
      const variance = [0, 1, 2].reduce((sum, j) =>
        sum + sampleVariance(assembled.points.map(point => point[j])), 0);
      const expected = assembled.weights.reduce((sum, weight, j) =>
        sum + weight ** 2 * originalVariances[j], 0);
      close(variance, expected, 'variance budget throughout the assembly');
    }
  }
});

test('assembly clamps progress and never mutates points or basis', () => {
  const points = structuredClone(THREE_D_POINTS);
  const basis = structuredClone(THREE_D_BASIS);
  const pointsBefore = structuredClone(points);
  const basisBefore = structuredClone(basis);
  for (const amount of [-10, -.1, 0, .5, 1, 1.1, 10]) {
    assert.deepEqual(assembleComponents(points, basis, amount), assembleComponents(points, basis, Math.max(0, Math.min(1, amount))));
  }
  assert.deepEqual(points, pointsBefore);
  assert.deepEqual(basis, basisBefore);
});
