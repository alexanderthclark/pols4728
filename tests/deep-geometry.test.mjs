import assert from 'node:assert/strict';
import test from 'node:test';
import { curveGeometry, foldedCurveGeometry } from '../docs/deep/geometry.mjs';
import {
  foldedValue, INPUT_DOMAIN, PARAMETER_COUNTS, parameterCount, SURFACE_MODES, surfaceValue,
} from '../docs/deep/model.mjs';

const tolerance = 1e-8;
const close = (actual, expected, message = '') => assert.ok(Math.abs(actual - expected) < tolerance,
  message + ': ' + actual + ' should equal ' + expected);
const locations = geometry => geometry.bends.map(bend => bend.x);
const sameLocations = (actual, expected, message) => {
  assert.equal(actual.length, expected.length, message);
  actual.forEach((value, index) => close(value, expected[index], message));
};

function checkExactGeometry(geometry, evaluate) {
  assert.equal(geometry.segments.length, geometry.points.length - 1);
  geometry.segments.forEach((segment, index) => {
    assert.equal(segment.left, geometry.points[index][0]);
    assert.equal(segment.right, geometry.points[index + 1][0]);
    assert.ok(segment.left < segment.right);
    for (const fraction of [0, .1, .5, .9, 1]) {
      const x = segment.left + fraction * (segment.right - segment.left);
      close(segment.slope * x + segment.intercept, evaluate(x), 'exact affine piece');
    }
    const center = (segment.left + segment.right) / 2;
    const epsilon = Math.min(1e-5, (segment.right - segment.left) / 10);
    close((evaluate(center + epsilon) - evaluate(center - epsilon)) / (2 * epsilon),
      segment.slope, 'interior derivative');
  });
  for (let index = 0; index < 80; index += 1) {
    const x = geometry.domain[0] + (geometry.domain[1] - geometry.domain[0]) * ((index * .61803398875 + .13) % 1);
    const segment = geometry.segments.find(item => item.left <= x && item.right >= x);
    assert.ok(segment, 'independent domain probe is covered');
    close(segment.slope * x + segment.intercept, evaluate(x), 'independent domain probe');
  }
}

test('input geometry preserves endpoints, fixed first hinges, and relevant copied crossings', () => {
  for (const mode of SURFACE_MODES) {
    const geometry = curveGeometry(mode);
    assert.deepEqual(geometry.domain, INPUT_DOMAIN);
    assert.equal(geometry.points[0][0], INPUT_DOMAIN[0]);
    assert.equal(geometry.points.at(-1)[0], INPUT_DOMAIN[1]);
    for (const x of [0, 1, 2]) assert.ok(geometry.points.some(point => point[0] === x));
    assert.deepEqual(geometry.firstLayerKnots, [0, 1, 2]);
    assert.deepEqual(geometry.secondPreactivationRoots, [[.2, 1.8, 2.2], [.5, 1.5, 2.5], [.8, 1.2, 2.8]]);
    const roots = mode === 'combined' ? geometry.secondPreactivationRoots.flat()
      : mode.startsWith('unit') ? geometry.secondPreactivationRoots[Number(mode.at(-1)) - 1]
        : mode === 'preactivation' ? geometry.secondPreactivationRoots[1] : [];
    for (const x of roots) assert.ok(geometry.points.some(point => Math.abs(point[0] - x) < tolerance));
  }
});

test('every input and downstream curve is exact throughout the threshold range', () => {
  for (const threshold of [.35, .4, .45, .5, .55, .6, .65]) {
    for (const mode of SURFACE_MODES) checkExactGeometry(curveGeometry(mode, { threshold }),
      x => surfaceValue(mode, x, { threshold }));
    checkExactGeometry(foldedCurveGeometry({ threshold }), q => foldedValue(q, { threshold }));
  }
});

test('default true bends hide clipped hinges and exclude zero samples without slope changes', () => {
  const expected = {
    h11: [0], h12: [1], h13: [2],
    fold: [0, 1, 2], shallow: [0, 1, 2], preactivation: [0, 1, 2],
    unit1: [.2, 1, 1.8, 2.2],
    unit2: [.5, 1, 1.5, 2.5],
    unit3: [.8, 1, 1.2, 2.8],
    combined: [.2, .5, .8, 1, 1.2, 1.5, 1.8, 2.2, 2.5, 2.8],
  };
  for (const [mode, bends] of Object.entries(expected)) sameLocations(locations(curveGeometry(mode)), bends, mode);
  for (const mode of ['unit1', 'unit2', 'unit3', 'combined']) {
    assert.ok(!curveGeometry(mode).bends.some(bend => bend.x === 0 || bend.x === 2));
  }
});

test('three downstream hinges become nine copied hinges plus the surviving fold joint', () => {
  const jumps = [1, -2, 2, -2, 2, -2, 1, 1, -2, 2];
  for (const threshold of [.35, .4, .45, .5, .55, .6, .65]) {
    const downstream = foldedCurveGeometry({ threshold });
    sameLocations(locations(downstream), [.2, threshold, .8], 'downstream hinges');
    const input = curveGeometry('combined', { threshold });
    sameLocations(locations(input), [.2, threshold, .8, 1, 1.2, 2 - threshold, 1.8, 2.2, 2 + threshold, 2.8],
      'copied input hinges');
    assert.equal(input.bends.length, 10);
    input.bends.forEach((bend, index) => close(bend.rightSlope - bend.leftSlope, jumps[index], 'nonzero slope jump'));
    assert.equal(input.bends.filter(bend => bend.x === 1).length, 1, 'surviving original hinge');
  }
});

test('downstream affine-output zeros are extracted between activation hinges without creating new bends', () => {
  const cases = [
    [.35, [.2, .5], [.2, .5, 1.5, 1.8, 2.2, 2.5, 3.1]],
    [.4, [.2, .6, 1], [.2, .6, 1, 1.4, 1.8, 2.2, 2.6, 3]],
    [.45, [.2, .7, .9], [.2, .7, .9, 1.1, 1.3, 1.8, 2.2, 2.7, 2.9]],
    [.5, [.2, .8], [.2, .8, 1.2, 1.8, 2.2, 2.8]],
    [.65, [.2], [.2, 1.8, 2.2]],
  ];
  for (const [threshold, foldedZeros, inputZeros] of cases) {
    const downstream = foldedCurveGeometry({ threshold });
    const input = curveGeometry('combined', { threshold });
    sameLocations(downstream.zeroCrossings, foldedZeros, 'downstream zero contacts/crossings');
    sameLocations(input.zeroCrossings, inputZeros, 'input zero contacts/crossings');
    assert.equal(downstream.bends.length, 3);
    assert.equal(input.bends.length, 10);
    for (const zero of input.zeroCrossings) close(surfaceValue('combined', zero, { threshold }), 0);
  }
});

test('zero locations exclude interior points of flat-zero intervals', () => {
  const expected = {
    h11: [0], h12: [1], h13: [2],
    fold: [0, 2], shallow: [0, 2], preactivation: [.5, 1.5, 2.5],
    unit1: [.2, 1.8, 2.2], unit2: [.5, 1.5, 2.5], unit3: [.8, 1.2, 2.8],
  };
  for (const [mode, zeros] of Object.entries(expected)) sameLocations(curveGeometry(mode).zeroCrossings, zeros, mode);
});

test('the middle threshold moves three dependent joints while the fold and other activations remain fixed', () => {
  const low = curveGeometry('unit2', { threshold: .35 });
  const high = curveGeometry('unit2', { threshold: .65 });
  sameLocations(low.secondPreactivationRoots[1], [.35, 1.65, 2.35]);
  sameLocations(high.secondPreactivationRoots[1], [.65, 1.35, 2.65]);
  for (const mode of ['h11', 'h12', 'h13', 'fold', 'shallow', 'unit1', 'unit3']) {
    assert.deepEqual(curveGeometry(mode, { threshold: .35 }).points, curveGeometry(mode, { threshold: .65 }).points);
    assert.deepEqual(curveGeometry(mode, { threshold: .35 }).bends, curveGeometry(mode, { threshold: .65 }).bends);
  }
});

test('width ten gives an exact shallow matching curve, requiring 31 dense slots rather than 22', () => {
  for (const threshold of [.35, .4, .5, .6, .65]) {
    const geometry = curveGeometry('combined', { threshold });
    assert.equal(geometry.bends.length, 10, 'ten nonzero joints require at least ten shallow ReLU units');
    for (let index = 0; index <= 100; index += 1) {
      const x = -2 + index / 10;
      const shallowExpansion = geometry.bends.reduce((sum, bend) =>
        sum + (bend.rightSlope - bend.leftSlope) * Math.max(0, x - bend.x), 0);
      close(shallowExpansion, surfaceValue('combined', x, { threshold }), 'exact ten-hinge shallow reconstruction');
    }
    assert.equal(parameterCount([1, geometry.bends.length, 1]), PARAMETER_COUNTS.shallowMatch);
    assert.equal(PARAMETER_COUNTS.shallowMatch - PARAMETER_COUNTS.deep, 9);
    assert.equal(parameterCount([1, PARAMETER_COUNTS.shallowSameBudgetWidth, 1]), PARAMETER_COUNTS.deep);
    assert.ok(PARAMETER_COUNTS.shallowSameBudgetWidth < geometry.bends.length, 'same-budget width seven cannot supply ten joints');
  }
});

test('custom input and folded domains clip samples while preserving exact affine geometry', () => {
  const input = curveGeometry('combined', { threshold: .35, domain: [1.1, 1.7] });
  assert.ok(input.points.every(([x]) => x >= 1.1 && x <= 1.7));
  assert.deepEqual(input.firstLayerKnots, [0, 1, 2]);
  assert.equal(input.secondPreactivationRoots.flat().length, 9);
  checkExactGeometry(input, x => surfaceValue('combined', x, { threshold: .35 }));
  const downstream = foldedCurveGeometry({ threshold: .35, domain: [-.1, 1.25] });
  checkExactGeometry(downstream, q => foldedValue(q, { threshold: .35 }));
  sameLocations(downstream.zeroCrossings, [.2, .5, 1.1], 'complete downstream zeros in larger view');
});

test('invalid domains, thresholds, and curve modes fail explicitly', () => {
  for (const domain of [[0, 0], [1, 0], [0], [0, Infinity], [NaN, 1], 'domain']) {
    assert.throws(() => curveGeometry('unit1', { domain }), RangeError);
    assert.throws(() => foldedCurveGeometry({ domain }), RangeError);
  }
  assert.throws(() => curveGeometry('downstream'), RangeError);
  assert.throws(() => curveGeometry('unknown'), RangeError);
  assert.throws(() => curveGeometry('unit1', { threshold: .9 }), RangeError);
  assert.throws(() => foldedCurveGeometry({ threshold: .9 }), RangeError);
});
