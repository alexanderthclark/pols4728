import assert from 'node:assert/strict';
import test from 'node:test';
import {
  DEFAULT_THRESHOLD, evaluateNetwork, INPUT_DOMAIN, networkParameters,
  OUTPUT_WEIGHTS, preactivationRoots, surfaceValue, THRESHOLD_RANGE,
} from '../docs/deep/model.mjs';

const close = (actual, expected, message = '') => assert.ok(Math.abs(actual - expected) < 1e-10,
  message + ': ' + actual + ' should equal ' + expected);

// Independent scalar equations, including the first-layer interval formulas.
function expected(x, threshold) {
  const h1 = [x > 0 ? x : 0, x > 1 ? x - 1 : 0, x > 2 ? x - 2 : 0];
  let q;
  if (x <= 0) q = [-.5, -threshold, -.6];
  else if (x <= 1) q = [x - .5, x - threshold, x - .6];
  else if (x <= 2) q = [1.5 - x, 3 - threshold - 2 * x, .9 - .5 * x];
  else q = [x - 2.5, 2 * x - 5 - threshold, .5 * x - 1.1];
  const h2 = q.map(value => Math.max(0, value));
  return { h1, q, h2, y: h2[0] + .3 * h2[1] + .3 * h2[2] };
}

test('the illustrative network has the stated scalar input, two width-three hidden layers, and defaults', () => {
  assert.equal(DEFAULT_THRESHOLD, .3);
  assert.deepEqual(THRESHOLD_RANGE, [.2, .8]);
  assert.deepEqual(INPUT_DOMAIN, [-.25, 3.25]);
  assert.deepEqual(OUTPUT_WEIGHTS, [1, .3, .3]);
  assert.deepEqual(networkParameters(), {
    beta0: [0, -1, -2], omega0: [[1], [1], [1]],
    beta1: [-.5, -.3, -.6],
    omega1: [[1, -2, 2], [1, -3, 4], [1, -1.5, 1]],
    beta2: 0, omega2: [1, .3, .3],
  });
});

test('every intermediate matches independent equations across all first-layer intervals', () => {
  for (const threshold of [.2, .3, .5, .6, .8]) {
    for (const x of [-2, -.25, 0, .25, .5, .7, 1, 1.3, 1.7, 2, 2.3, 2.8, 3.25, 5]) {
      const actual = evaluateNetwork(x, { threshold });
      const independent = expected(x, threshold);
      assert.equal(actual.x, x);
      assert.deepEqual(actual.firstPreactivation, [x, x - 1, x - 2]);
      actual.h1.forEach((value, index) => close(value, independent.h1[index], 'first feature'));
      actual.secondPreactivation.forEach((value, index) => close(value, independent.q[index], 'second preactivation'));
      actual.h2.forEach((value, index) => close(value, independent.h2[index], 'second feature'));
      close(actual.y, independent.y, 'output');
      close(actual.combined, independent.y, 'output alias');
      close(actual.shallow, independent.q[0], 'shallow readout');
      close(actual.preactivation, independent.q[0], 'same sum before the added ReLU');
    }
  }
});

test('all three new units cross zero inside the fixed first-layer intervals', () => {
  for (const threshold of [.2, .3, .5, .6, .8]) {
    const roots = preactivationRoots({ threshold });
    roots.forEach((unitRoots, unit) => unitRoots.forEach((root, region) => {
      assert.ok(root > region && root < region + 1, 'root is inside its fixed interval');
      close(evaluateNetwork(root, { threshold }).secondPreactivation[unit], 0, 'preactivation root');
      const before = evaluateNetwork(root - 1e-5, { threshold }).secondPreactivation[unit];
      const after = evaluateNetwork(root + 1e-5, { threshold }).secondPreactivation[unit];
      assert.ok(before * after < 0, 'a genuine zero crossing');
    }));
  }
});

test('the live threshold changes only the bias and response of the second new unit', () => {
  for (const x of [-.25, .25, .7, 1.2, 1.7, 2.7, 3.25]) {
    const low = evaluateNetwork(x, { threshold: .2 });
    const high = evaluateNetwork(x, { threshold: .8 });
    assert.deepEqual(high.firstPreactivation, low.firstPreactivation);
    assert.deepEqual(high.h1, low.h1);
    close(high.secondPreactivation[1] - low.secondPreactivation[1], -.6, 'bias shift');
    close(high.secondPreactivation[0], low.secondPreactivation[0], 'unit1 preactivation fixed');
    close(high.secondPreactivation[2], low.secondPreactivation[2], 'unit3 preactivation fixed');
    close(high.unit1, low.unit1, 'unit1 activation fixed');
    close(high.unit3, low.unit3, 'unit3 activation fixed');
    close(high.y - low.y, .3 * (high.unit2 - low.unit2), 'output changes through only unit2');
  }
});

test('the three incoming patterns cannot all factor through one scalar affine readout', () => {
  const rows = networkParameters().omega1;
  // A nonzero two-by-two minor rules out a rank-one middle weight matrix.
  close(rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0], -1, 'nonzero minor');
  // These simple chosen rows have rank two, not three.
  rows[2].forEach((value, index) => close(value, 1.5 * rows[0][index] - .5 * rows[1][index], 'third-row relation'));
});

test('every named curve mode reports the intended quantity', () => {
  for (const x of [.2, .8, 1.4, 1.9, 2.6, 3.1]) {
    const independent = expected(x, .3);
    const quantities = {
      h11: independent.h1[0], h12: independent.h1[1], h13: independent.h1[2],
      shallow: independent.q[0], preactivation: independent.q[0],
      unit1: independent.h2[0], unit2: independent.h2[1], unit3: independent.h2[2],
      combined: independent.y,
    };
    for (const [mode, value] of Object.entries(quantities)) close(surfaceValue(mode, x), value, mode);
  }
});

test('the chosen scalar response includes a zero interval but is not globally compactly supported', () => {
  close(evaluateNetwork(1).y, .83, 'first peak');
  close(evaluateNetwork(2).y, 0, 'inactive interval');
  close(evaluateNetwork(3.25).y, 1.2675, 'right endpoint');
  assert.ok(evaluateNetwork(10).y > evaluateNetwork(3.25).y, 'the right-hand response continues rising');
});

test('invalid inputs, thresholds, and modes fail explicitly', () => {
  for (const x of [NaN, Infinity, -Infinity, '1']) assert.throws(() => evaluateNetwork(x), TypeError);
  for (const threshold of [0, .19, .81, NaN, Infinity]) {
    assert.throws(() => evaluateNetwork(0, { threshold }), RangeError);
    assert.throws(() => preactivationRoots({ threshold }), RangeError);
  }
  assert.throws(() => surfaceValue('unknown', 0), RangeError);
});
