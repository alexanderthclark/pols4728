import assert from 'node:assert/strict';
import test from 'node:test';
import { curveGeometry } from '../docs/deep/geometry.mjs';
import { INPUT_DOMAIN, SURFACE_MODES, surfaceValue } from '../docs/deep/model.mjs';

const tolerance = 1e-8;
const close = (actual, expected, message = '') => assert.ok(Math.abs(actual - expected) < tolerance,
  message + ': ' + actual + ' should equal ' + expected);
const locations = geometry => geometry.bends.map(bend => bend.x);
const sameLocations = (actual, expected, message) => {
  assert.equal(actual.length, expected.length, message);
  actual.forEach((value, index) => close(value, expected[index], message));
};

test('exact samples include the domain ends, fixed first hinges, and relevant preactivation roots', () => {
  for (const mode of SURFACE_MODES) {
    const geometry = curveGeometry(mode);
    assert.deepEqual(geometry.domain, INPUT_DOMAIN);
    assert.equal(geometry.points[0][0], INPUT_DOMAIN[0]);
    assert.equal(geometry.points.at(-1)[0], INPUT_DOMAIN[1]);
    for (const x of [0, 1, 2]) assert.ok(geometry.points.some(point => point[0] === x), 'first-layer knot sample');
    assert.deepEqual(geometry.firstLayerKnots, [0, 1, 2]);
    assert.deepEqual(geometry.secondPreactivationRoots[0], [.5, 1.5, 2.5]);
    const roots = mode === 'combined' ? geometry.secondPreactivationRoots.flat()
      : mode.startsWith('unit') ? geometry.secondPreactivationRoots[Number(mode.at(-1)) - 1]
        : mode === 'shallow' || mode === 'preactivation' ? geometry.secondPreactivationRoots[0] : [];
    for (const x of roots) assert.ok(geometry.points.some(point => Math.abs(point[0] - x) < tolerance), 'relevant root sample');
  }
});

test('segments cover each requested domain without gaps and exactly interpolate network responses', () => {
  for (const threshold of [.2, .3, .5, .6, .8]) for (const mode of SURFACE_MODES) {
    const geometry = curveGeometry(mode, { threshold });
    assert.equal(geometry.segments.length, geometry.points.length - 1);
    geometry.segments.forEach((segment, index) => {
      assert.equal(segment.left, geometry.points[index][0]);
      assert.equal(segment.right, geometry.points[index + 1][0]);
      assert.ok(segment.left < segment.right);
      for (const fraction of [0, .1, .5, .9, 1]) {
        const x = segment.left + fraction * (segment.right - segment.left);
        close(segment.slope * x + segment.intercept, surfaceValue(mode, x, { threshold }), 'exact affine segment');
      }
      const center = (segment.left + segment.right) / 2;
      const epsilon = Math.min(1e-5, (segment.right - segment.left) / 10);
      close((surfaceValue(mode, center + epsilon, { threshold }) - surfaceValue(mode, center - epsilon, { threshold })) / (2 * epsilon),
        segment.slope, 'interior derivative');
    });
    for (let index = 0; index < 80; index += 1) {
      const x = INPUT_DOMAIN[0] + (INPUT_DOMAIN[1] - INPUT_DOMAIN[0]) * ((index * .61803398875 + .13) % 1);
      const segment = geometry.segments.find(item => item.left <= x && item.right >= x);
      assert.ok(segment, 'every independent domain probe is covered');
      close(segment.slope * x + segment.intercept, surfaceValue(mode, x, { threshold }), 'independent probe');
    }
  }
});

test('true bends exclude redundant samples, zero crossings without slope changes, and clipped hinges', () => {
  const expected = {
    h11: [0], h12: [1], h13: [2],
    shallow: [0, 1, 2], preactivation: [0, 1, 2],
    unit1: [.5, 1, 1.5, 2.5],
    unit2: [.3, 1, 1.35, 2.65],
    unit3: [.6, 1, 1.8, 2.2],
    combined: [.3, .5, .6, 1, 1.35, 1.5, 1.8, 2.2, 2.5, 2.65],
  };
  for (const [mode, bends] of Object.entries(expected)) sameLocations(locations(curveGeometry(mode)), bends, mode);
  for (const mode of ['unit1', 'unit2', 'unit3', 'combined']) {
    const geometry = curveGeometry(mode);
    assert.ok(!geometry.bends.some(bend => bend.x === 0 || bend.x === 2), 'inactive neighborhoods hide original hinges');
  }
});

test('the default output has ten non-cancelling bends with the stated one-sided slopes', () => {
  const geometry = curveGeometry('combined');
  const jumps = [.3, 1, .3, -3.35, .6, 1, .15, .15, 1, .6];
  assert.equal(geometry.bends.length, 10);
  geometry.bends.forEach((bend, index) => {
    close(bend.rightSlope - bend.leftSlope, jumps[index], 'slope jump');
    close(bend.y, surfaceValue('combined', bend.x), 'bend height');
    const epsilon = 1e-5;
    close((surfaceValue('combined', bend.x + epsilon) - surfaceValue('combined', bend.x)) / epsilon,
      bend.rightSlope, 'right derivative');
    close((surfaceValue('combined', bend.x) - surfaceValue('combined', bend.x - epsilon)) / epsilon,
      bend.leftSlope, 'left derivative');
  });
});

test('coincident roots add slope jumps rather than cancelling and yield nine bends', () => {
  for (const threshold of [.2, .3, .4, .5, .6, .7, .8]) {
    const geometry = curveGeometry('combined', { threshold });
    assert.equal(geometry.bends.length, threshold === .5 || threshold === .6 ? 9 : 10);
    for (const bend of geometry.bends) {
      assert.ok(Math.abs(bend.rightSlope - bend.leftSlope) > tolerance, 'no false bends');
      if (Math.abs(bend.x - 1) > tolerance) assert.ok(bend.rightSlope > bend.leftSlope, 'new-root jumps have the same sign');
    }
    close(geometry.bends.find(bend => bend.x === 1).rightSlope - geometry.bends.find(bend => bend.x === 1).leftSlope,
      -3.35, 'the surviving old hinge');
  }
  const mergedFirst = curveGeometry('combined', { threshold: .5 }).bends.find(bend => bend.x === .5);
  close(mergedFirst.rightSlope - mergedFirst.leftSlope, 1.3, 'merged unit1 and unit2 roots');
  const mergedThird = curveGeometry('combined', { threshold: .6 }).bends.find(bend => bend.x === .6);
  close(mergedThird.rightSlope - mergedThird.leftSlope, .6, 'merged unit2 and unit3 roots');
});

test('zero transitions exclude flat-zero interiors and depend on the displayed quantity', () => {
  for (const threshold of [.2, .3, .5, .6, .8]) {
    const expected = {
      h11: [0], h12: [1], h13: [2],
      shallow: [.5, 1.5, 2.5], preactivation: [.5, 1.5, 2.5],
      unit1: [.5, 1.5, 2.5],
      unit2: [threshold, (3 - threshold) / 2, (5 + threshold) / 2],
      unit3: [.6, 1.8, 2.2],
      combined: [Math.min(.5, threshold), 1.8, 2.2],
    };
    for (const [mode, zeros] of Object.entries(expected)) {
      const geometry = curveGeometry(mode, { threshold });
      sameLocations(geometry.zeroCrossings, zeros, mode);
      for (const zero of geometry.zeroCrossings) close(surfaceValue(mode, zero, { threshold }), 0, 'actual zero value');
    }
  }
});

test('moving the threshold changes unit2 roots within fixed first-layer intervals', () => {
  const low = curveGeometry('unit2', { threshold: .2 });
  const high = curveGeometry('unit2', { threshold: .8 });
  assert.deepEqual(low.firstLayerKnots, high.firstLayerKnots);
  sameLocations(low.secondPreactivationRoots[1], [.2, 1.4, 2.6], 'low threshold roots');
  sameLocations(high.secondPreactivationRoots[1], [.8, 1.1, 2.9], 'high threshold roots');
  for (const mode of ['h11', 'h12', 'h13', 'shallow', 'preactivation', 'unit1', 'unit3']) {
    const lowCurve = curveGeometry(mode, { threshold: .2 });
    const highCurve = curveGeometry(mode, { threshold: .8 });
    assert.deepEqual(lowCurve.points, highCurve.points, 'other responses remain fixed');
    assert.deepEqual(lowCurve.bends, highCurve.bends, 'other response bends remain fixed');
  }
});

test('a wider shallow hinge expansion exactly reproduces the constructed output', () => {
  for (const threshold of [.3, .5, .6]) {
    const geometry = curveGeometry('combined', { threshold });
    for (let index = 0; index <= 80; index += 1) {
      const x = -2 + index / 10;
      const shallowExpansion = geometry.bends.reduce((sum, bend) =>
        sum + (bend.rightSlope - bend.leftSlope) * Math.max(0, x - bend.x), 0);
      close(shallowExpansion, surfaceValue('combined', x, { threshold }), 'one shallow hinge per true bend');
    }
  }
});

test('custom domains clip samples and retain full first knots and preactivation roots for comparison', () => {
  const geometry = curveGeometry('combined', { domain: [1.1, 1.7] });
  assert.deepEqual(geometry.domain, [1.1, 1.7]);
  assert.ok(geometry.points.every(([x]) => x >= 1.1 && x <= 1.7));
  sameLocations(locations(geometry), [1.35, 1.5], 'visible bends');
  assert.deepEqual(geometry.firstLayerKnots, [0, 1, 2]);
  assert.equal(geometry.secondPreactivationRoots.flat().length, 9);
});

test('invalid domain, threshold, and mode parameters fail explicitly', () => {
  for (const domain of [[0, 0], [1, 0], [0], [0, Infinity], [NaN, 1], 'domain']) {
    assert.throws(() => curveGeometry('unit1', { domain }), RangeError);
  }
  assert.throws(() => curveGeometry('unknown'), RangeError);
  assert.throws(() => curveGeometry('unit1', { threshold: .9 }), RangeError);
});
