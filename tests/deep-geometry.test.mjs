import assert from 'node:assert/strict';
import test from 'node:test';
import { polygonArea, surfaceMesh } from '../docs/deep/geometry.mjs';
import { SURFACE_MODES, surfaceValue } from '../docs/deep/model.mjs';

const tolerance = 1e-8;
const close = (actual, expected, message = '') => assert.ok(Math.abs(actual - expected) < tolerance,
  `${message}: ${actual} should equal ${expected}`);
const edgeKey = ([left, right]) => left < right ? `${left}:${right}` : `${right}:${left}`;
const facePoints = (mesh, face) => face.indices.map(index => mesh.vertices[index]);
const centroid = points => [0, 1].map(axis => points.reduce((sum, point) => sum + point[axis], 0) / points.length);
const affineValue = (face, [x, y]) => face.intercept + face.gradient[0] * x + face.gradient[1] * y;

function adjacency(mesh) {
  const edges = new Map();
  mesh.faces.forEach((face, faceIndex) => face.indices.forEach((start, index) => {
    const edge = [start, face.indices[(index + 1) % face.indices.length]];
    const key = edgeKey(edge);
    if (!edges.has(key)) edges.set(key, { edge, faces: [] });
    edges.get(key).faces.push(faceIndex);
  }));
  return edges;
}

function contains(points, [x, y]) {
  return points.every((point, index) => {
    const next = points[(index + 1) % points.length];
    return (next[0] - point[0]) * (y - point[1]) - (next[1] - point[1]) * (x - point[0]) >= -tolerance;
  });
}

function lineLength(mesh, edges) {
  return edges.reduce((sum, [left, right]) => {
    const start = mesh.vertices[left], end = mesh.vertices[right];
    return sum + Math.hypot(end[0] - start[0], end[1] - start[1]);
  }, 0);
}

test('all surface meshes cover the domain once, with shared vertices and no T-junctions', () => {
  for (const pinch of [.5, 1, 1.37, 2, 3]) for (const mode of SURFACE_MODES) {
    const mesh = surfaceMesh(mode, { pinch });
    const edges = adjacency(mesh);
    const areas = mesh.faces.map(face => polygonArea(facePoints(mesh, face)));
    assert.ok(areas.every(area => area > 0), `${mode}: CCW nondegenerate faces`);
    close(areas.reduce((sum, area) => sum + area, 0), 3.2 ** 2, `${mode}: full domain area`);
    close(mesh.triangles.reduce((sum, indices) => sum + polygonArea(indices.map(index => mesh.vertices[index])), 0),
      3.2 ** 2, `${mode}: fill triangles cover the same domain`);
    for (const edge of edges.values()) assert.ok(edge.faces.length === 1 || edge.faces.length === 2,
      `${mode}: every edge has one or two adjacent faces`);
    assert.equal(mesh.vertices.length - edges.size + mesh.faces.length, 1, `${mode}: conforming disk topology`);
    assert.deepEqual(new Set(mesh.domainEdges.map(edgeKey)),
      new Set([...edges.values()].filter(edge => edge.faces.length === 1).map(edge => edgeKey(edge.edge))));
    close(lineLength(mesh, mesh.domainEdges), 12.8, 'domain perimeter');
    for (const [left, right] of mesh.domainEdges) {
      const a = mesh.vertices[left], b = mesh.vertices[right];
      assert.ok([0, 1].some(axis => Math.abs(Math.abs(a[axis]) - 1.6) < tolerance && Math.abs(a[axis] - b[axis]) < tolerance));
    }
  }
});

test('each face is an exact affine piece of its requested network surface', () => {
  for (const pinch of [.5, 1, 1.37, 2, 3]) for (const mode of SURFACE_MODES) {
    const mesh = surfaceMesh(mode, { pinch });
    for (const face of mesh.faces) {
      const points = facePoints(mesh, face);
      for (const [x, y, z] of points) {
        close(z, surfaceValue(mode, x, y, { pinch }), 'vertex height');
        close(affineValue(face, [x, y]), z, 'face plane at shared vertex');
      }
      const center = centroid(points);
      close(affineValue(face, center), surfaceValue(mode, ...center, { pinch }), 'interior interpolation');
      for (const vertex of points) {
        const interior = center.map((value, axis) => .7 * value + .3 * vertex[axis]);
        close(affineValue(face, interior), surfaceValue(mode, ...interior, { pinch }), 'interior convex combination');
      }
      for (const axis of [0, 1]) {
        const before = [...center], after = [...center];
        before[axis] -= 1e-6;
        after[axis] += 1e-6;
        close((surfaceValue(mode, ...after, { pinch }) - surfaceValue(mode, ...before, { pinch })) / 2e-6,
          face.gradient[axis], 'true interior gradient');
      }
    }
    // Independent domain probes include points away from all mesh vertices.
    for (let index = 0; index < 80; index += 1) {
      const x = -1.6 + 3.2 * ((index * .61803398875 + .13) % 1);
      const y = -1.6 + 3.2 * ((index * .41421356237 + .29) % 1);
      const covering = mesh.faces.filter(face => contains(facePoints(mesh, face), [x, y]));
      assert.ok(covering.length > 0, 'every probe is covered');
      for (const face of covering) close(affineValue(face, [x, y]), surfaceValue(mode, x, y, { pinch }), 'probe interpolation');
    }
  }
});

test('crease edges are exactly the shared boundaries across which the gradient changes', () => {
  for (const mode of SURFACE_MODES) {
    const mesh = surfaceMesh(mode);
    const expected = [...adjacency(mesh).values()].filter(edge => {
      if (edge.faces.length !== 2) return false;
      const [left, right] = edge.faces.map(index => mesh.faces[index]);
      return left.gradient.some((value, axis) => Math.abs(value - right.gradient[axis]) > tolerance);
    }).map(edge => edgeKey(edge.edge));
    assert.deepEqual(new Set(mesh.creases.map(edgeKey)), new Set(expected), mode);
    for (const [left, right] of mesh.creases) {
      const a = mesh.vertices[left], b = mesh.vertices[right];
      if (mode.startsWith('unit') || mode === 'combined') {
        // Internal splits in an entirely flat zero plane must not be drawn.
        const edge = adjacency(mesh).get(edgeKey([left, right]));
        assert.ok(edge.faces.some(index => mesh.faces[index].gradient.some(value => Math.abs(value) > tolerance)));
      }
      assert.ok(Math.hypot(a[0] - b[0], a[1] - b[1]) > 0);
    }
  }
});

test('the shallow zero contour and unit1 footprint follow the exact six-sided boundary', () => {
  for (const mode of ['shallow', 'preactivation', 'unit1', 'combined']) {
    const mesh = surfaceMesh(mode);
    close(lineLength(mesh, mesh.zeroContours), 4 + 2 * Math.SQRT2, `${mode}: hexagon perimeter`);
    for (const edge of mesh.zeroContours) for (const index of edge) {
      const [x, y, z] = mesh.vertices[index];
      close(z, 0, 'zero contour height');
      close(Math.max(Math.abs(x), Math.abs(y), Math.abs(x + y)), 1, 'hexagonal boundary');
    }
    assert.deepEqual(new Set(mesh.boundary.map(edgeKey)), new Set(mesh.zeroContours.map(edgeKey)), 'bounded positive footprint');
  }
});

test('feature zero contours are straight activation boundaries rather than outlines of the flat plane', () => {
  for (const [mode, equation, length] of [
    ['h11', ([x]) => x, 3.2],
    ['h12', ([, y]) => y, 3.2],
    ['h13', ([x, y]) => x + y, 3.2 * Math.SQRT2],
  ]) {
    const mesh = surfaceMesh(mode);
    close(lineLength(mesh, mesh.zeroContours), length, mode);
    mesh.zeroContours.flat().forEach(index => close(equation(mesh.vertices[index]), 0, 'straight activation line'));
    assert.ok(mesh.boundary.length > mesh.zeroContours.length, 'clipped positive footprint also meets the outer domain');
  }
});

test('the control changes exact support geometry and marks a footprint clipped by the view', () => {
  for (const pinch of [1, 1.37, 2, 3]) {
    const mesh = surfaceMesh('unit2', { pinch });
    assert.ok(mesh.vertices.some(([x, y]) => Math.abs(x - 1 / pinch) < tolerance && Math.abs(y) < tolerance),
      'moving support corner');
    for (const edge of mesh.zeroContours) for (const index of edge) {
      const [x, y, z] = mesh.vertices[index];
      close(z, 0, 'zero contour height');
      close(pinch * Math.max(0, x) + Math.max(0, y) + Math.max(0, -x - y), 1, 'controlled support boundary');
    }
  }
  const wide = surfaceMesh('unit2', { pinch: .5 });
  const domain = new Set(wide.domainEdges.map(edgeKey));
  assert.ok(wide.boundary.some(edge => domain.has(edgeKey(edge)) && edge.some(index => wide.vertices[index][2] > tolerance)),
    'wide support extends to the domain edge');
  assert.ok(wide.zeroContours.every(edge => edge.every(index => Math.abs(wide.vertices[index][2]) < tolerance)),
    'nonzero domain edges are excluded from the zero contour');
});

test('invalid domain and mode parameters fail explicitly', () => {
  for (const extent of [0, -1, Infinity, NaN]) assert.throws(() => surfaceMesh('unit1', { extent }), RangeError);
  assert.throws(() => surfaceMesh('unknown'), RangeError);
  assert.throws(() => surfaceMesh('unit1', { pinch: 4 }), RangeError);
});
