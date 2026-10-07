import {
  affineSurface, DEFAULT_PINCH, evaluateNetwork, INPUT_EXTENT,
  networkParameters, SURFACE_MODES, surfaceValue,
} from './model.mjs';

const EPSILON = 1e-10;
const close = (left, right) => Math.abs(left - right) <= EPSILON;
const planeValue = (plane, point) => plane.intercept
  + plane.gradient[0] * point[0] + plane.gradient[1] * point[1];
const samePoint = (left, right) => close(left[0], right[0]) && close(left[1], right[1]);

export function polygonArea(points) {
  return points.reduce((sum, point, index) => {
    const next = points[(index + 1) % points.length];
    return sum + point[0] * next[1] - next[0] * point[1];
  }, 0) / 2;
}

function cleanPolygon(points) {
  const unique = points.filter((point, index) => index === 0 || !samePoint(point, points[index - 1]));
  if (unique.length > 1 && samePoint(unique[0], unique.at(-1))) unique.pop();
  return unique;
}

function clipPolygon(points, plane, sign) {
  const clipped = [];
  points.forEach((point, index) => {
    const next = points[(index + 1) % points.length];
    const currentValue = planeValue(plane, point);
    const nextValue = planeValue(plane, next);
    if (sign * currentValue >= -EPSILON) clipped.push(point);
    if ((currentValue > EPSILON && nextValue < -EPSILON)
      || (currentValue < -EPSILON && nextValue > EPSILON)) {
      const fraction = currentValue / (currentValue - nextValue);
      clipped.push([
        point[0] + fraction * (next[0] - point[0]),
        point[1] + fraction * (next[1] - point[1]),
      ]);
    }
  });
  return cleanPolygon(clipped);
}

function splitPolygon(points, plane) {
  const values = points.map(point => planeValue(plane, point));
  if (!values.some(value => value > EPSILON) || !values.some(value => value < -EPSILON)) return [points];
  return [clipPolygon(points, plane, 1), clipPolygon(points, plane, -1)]
    .filter(polygon => polygon.length >= 3 && polygonArea(polygon) > EPSILON);
}

function interiorPoint(points) {
  return [0, 1].map(axis => points.reduce((sum, point) => sum + point[axis], 0) / points.length);
}

function edgeKey(left, right) {
  return left < right ? `${left}:${right}` : `${right}:${left}`;
}

function segmentFraction(point, start, end) {
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const squaredLength = dx * dx + dy * dy;
  const cross = dx * (point[1] - start[1]) - dy * (point[0] - start[0]);
  if (Math.abs(cross) > EPSILON * Math.max(1, Math.sqrt(squaredLength))) return null;
  const fraction = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / squaredLength;
  return fraction > EPSILON && fraction < 1 - EPSILON ? fraction : null;
}

export function surfaceMesh(mode, { pinch = DEFAULT_PINCH, extent = INPUT_EXTENT } = {}) {
  if (!SURFACE_MODES.includes(mode)) throw new RangeError(`Unknown surface mode: ${mode}`);
  if (!Number.isFinite(extent) || extent <= 0) throw new RangeError('extent must be a positive finite number');
  const parameters = networkParameters({ pinch });
  const square = [[-extent, -extent], [extent, -extent], [extent, extent], [-extent, extent]];
  let cones = [square];
  for (const gradient of parameters.omega0) {
    cones = cones.flatMap(polygon => splitPolygon(polygon, { gradient, intercept: 0 }));
  }

  const pieces = cones.flatMap(cone => {
    const sample = evaluateNetwork(...interiorPoint(cone), { pinch });
    const activeFirst = sample.firstPreactivation.map(value => value > 0);
    const firstGradients = parameters.omega0.map((gradient, index) => activeFirst[index] ? gradient : [0, 0]);
    const secondPlanes = parameters.omega1.map(row => ({
      gradient: [0, 1].map(axis => row.reduce((sum, weight, index) => sum + weight * firstGradients[index][axis], 0)),
      intercept: 1,
    }));
    let regions = [cone];
    for (const plane of secondPlanes) regions = regions.flatMap(polygon => splitPolygon(polygon, plane));
    return regions.map(points => {
      const evaluation = evaluateNetwork(...interiorPoint(points), { pinch });
      const activeSecond = evaluation.secondPreactivation.map(value => value > 0);
      return { points, activeFirst, activeSecond, ...affineSurface(mode, activeFirst, activeSecond, { pinch }) };
    });
  });

  const vertices = [];
  const vertexIndex = point => {
    const existing = vertices.findIndex(vertex => samePoint(vertex, point));
    if (existing >= 0) return existing;
    const x1 = close(point[0], 0) ? 0 : point[0];
    const x2 = close(point[1], 0) ? 0 : point[1];
    vertices.push([x1, x2, surfaceValue(mode, x1, x2, { pinch })]);
    return vertices.length - 1;
  };
  const faces = pieces.map(({ points, ...piece }) => ({
    ...piece, indices: points.map(vertexIndex),
  }));

  // A plane may coincide with an existing edge in a neighboring cone. Insert
  // every shared point on that edge so the final mesh has no T-junctions.
  for (const face of faces) {
    face.indices = face.indices.flatMap((start, index, indices) => {
      const end = indices[(index + 1) % indices.length];
      const intermediate = vertices.map((point, vertex) => ({
        vertex, fraction: segmentFraction(point, vertices[start], vertices[end]),
      })).filter(item => item.fraction !== null).sort((left, right) => left.fraction - right.fraction);
      return [start, ...intermediate.map(item => item.vertex)];
    });
  }

  const edges = new Map();
  faces.forEach((face, faceIndex) => {
    face.indices.forEach((start, index) => {
      const end = face.indices[(index + 1) % face.indices.length];
      const key = edgeKey(start, end);
      if (!edges.has(key)) edges.set(key, { indices: [start, end], faces: [] });
      edges.get(key).faces.push(faceIndex);
    });
  });
  const positive = faces.map(face => {
    const point = interiorPoint(face.indices.map(index => vertices[index]));
    return planeValue(face, point) > EPSILON;
  });
  const creases = [];
  const boundary = [];
  const zeroContours = [];
  const domainEdges = [];
  for (const edge of edges.values()) {
    const neighbors = edge.faces.map(index => faces[index]);
    if (neighbors.length === 1) domainEdges.push(edge.indices);
    if (neighbors.length === 2 && neighbors[0].gradient.some((value, axis) => !close(value, neighbors[1].gradient[axis]))) {
      creases.push(edge.indices);
    }
    const hasPositive = edge.faces.some(index => positive[index]);
    const boundsPositive = hasPositive && (neighbors.length === 1 || edge.faces.some(index => !positive[index]));
    if (boundsPositive) {
      boundary.push(edge.indices);
      if (edge.indices.every(index => Math.abs(vertices[index][2]) <= EPSILON)) zeroContours.push(edge.indices);
    }
  }
  const triangles = faces.flatMap(face => face.indices.slice(1, -1).map((vertex, index) =>
    [face.indices[0], vertex, face.indices[index + 2]]))
    .filter(indices => polygonArea(indices.map(index => vertices[index])) > EPSILON);

  return {
    mode, pinch, domain: [[-extent, extent], [-extent, extent]],
    vertices, faces, triangles, creases, boundary, zeroContours, domainEdges,
  };
}
