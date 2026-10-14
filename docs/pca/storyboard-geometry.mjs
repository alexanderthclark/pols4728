import { THREE_D_BASIS, dotND } from './model.mjs';

// The remaining plane uses a reference basis 30° away from PC2 and PC3.
// Its coordinates therefore agree with coordinates(residualScores, 30).
const pc1 = THREE_D_BASIS.map(row => row[0]);
const pc2 = THREE_D_BASIS.map(row => row[1]);
const pc3 = THREE_D_BASIS.map(row => row[2]);
const cosine = Math.cos(Math.PI / 6);
const sine = Math.sin(Math.PI / 6);
export const RESIDUAL_BASIS = Object.freeze({
  a: Object.freeze(pc2.map((value, j) => cosine * value + sine * pc3[j])),
  b: Object.freeze(pc2.map((value, j) => -sine * value + cosine * pc3[j])),
  normal: Object.freeze(pc1),
});

const clamp = value => Math.min(1, Math.max(0, value));
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];

// Quaternion operations rotate a complete orthonormal camera frame. Mixing
// screen coordinates directly would deform the plane between its endpoints.
function quaternionFromFrame(horizontal, vertical, normal) {
  const matrix = horizontal.map((_, i) => [horizontal[i], vertical[i], normal[i]]);
  const trace = matrix[0][0] + matrix[1][1] + matrix[2][2];
  if (trace > 0) {
    const s = 2 * Math.sqrt(trace + 1);
    return [
      (matrix[2][1] - matrix[1][2]) / s,
      (matrix[0][2] - matrix[2][0]) / s,
      (matrix[1][0] - matrix[0][1]) / s,
      s / 4,
    ];
  }
  const i = matrix[1][1] > matrix[0][0]
    ? (matrix[2][2] > matrix[1][1] ? 2 : 1)
    : (matrix[2][2] > matrix[0][0] ? 2 : 0);
  const j = (i + 1) % 3;
  const k = (i + 2) % 3;
  const s = 2 * Math.sqrt(1 + matrix[i][i] - matrix[j][j] - matrix[k][k]);
  const result = [0, 0, 0, (matrix[k][j] - matrix[j][k]) / s];
  result[i] = s / 4;
  result[j] = (matrix[j][i] + matrix[i][j]) / s;
  result[k] = (matrix[k][i] + matrix[i][k]) / s;
  return result;
}

function interpolateQuaternion(first, last, amount) {
  let product = dotND(first, last);
  const destination = product < 0 ? last.map(value => -value) : last;
  product = Math.abs(product);
  if (product > .9995) {
    const mixed = first.map((value, j) => value + amount * (destination[j] - value));
    const length = Math.hypot(...mixed);
    return mixed.map(value => value / length);
  }
  const angle = Math.acos(Math.min(1, product));
  const denominator = Math.sin(angle);
  const firstWeight = Math.sin((1 - amount) * angle) / denominator;
  const lastWeight = Math.sin(amount * angle) / denominator;
  return first.map((value, j) => value * firstWeight + destination[j] * lastWeight);
}

function frameFromQuaternion([x, y, z, w]) {
  return {
    horizontal: [1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)],
    vertical: [2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)],
    normal: [2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)],
  };
}

/** An orthographic camera, with one common scale for both screen axes. */
export function residualCamera(progress, cameraDegrees = 25) {
  const amount = clamp(progress);
  const yaw = cameraDegrees * Math.PI / 180;
  const pitch = 22 * Math.PI / 180;
  const horizontal = [Math.cos(yaw), -Math.sin(yaw), 0];
  const vertical = [Math.sin(yaw) * Math.sin(pitch), Math.cos(yaw) * Math.sin(pitch), Math.cos(pitch)];
  const normal = cross(horizontal, vertical);
  let frame;
  if (amount === 0) frame = { horizontal, vertical, normal };
  else if (amount === 1) frame = {
    horizontal: [...RESIDUAL_BASIS.a],
    vertical: [...RESIDUAL_BASIS.b],
    normal: [...RESIDUAL_BASIS.normal],
  };
  else frame = frameFromQuaternion(interpolateQuaternion(
    quaternionFromFrame(horizontal, vertical, normal),
    quaternionFromFrame(RESIDUAL_BASIS.a, RESIDUAL_BASIS.b, RESIDUAL_BASIS.normal),
    amount,
  ));
  return { ...frame, center: [360, 215 - 36 * amount], scale: 31 + 26 * amount };
}

export function projectWithCamera(point, camera) {
  return [
    camera.center[0] + camera.scale * dotND(point, camera.horizontal),
    camera.center[1] - camera.scale * dotND(point, camera.vertical),
  ];
}

export function projectResidualPoint(point, progress, cameraDegrees = 25) {
  return projectWithCamera(point, residualCamera(progress, cameraDegrees));
}

/** Subtract only PC2; the PC3 coordinate and observation identity stay fixed. */
export function residualPointAt(point, amount) {
  const score = dotND(point, pc2);
  return point.map((value, j) => value - clamp(amount) * score * pc2[j]);
}
