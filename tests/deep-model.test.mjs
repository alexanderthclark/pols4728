import assert from 'node:assert/strict';
import test from 'node:test';
import {
  evaluateNetwork, networkParameters, PINCH_RANGE, surfaceValue,
} from '../docs/deep/model.mjs';

const close = (actual, expected, message = '') => assert.ok(Math.abs(actual - expected) < 1e-10,
  `${message}: ${actual} should equal ${expected}`);

// An independent scalar formula avoids checking a function against itself.
function expected(x, y, pinch) {
  const u = x > 0 ? x : 0;
  const v = y > 0 ? y : 0;
  const w = x + y < 0 ? -x - y : 0;
  const q = [1 - u - v - w, 1 - pinch * u - v - w, 1 - u - 2 * v - w];
  const h2 = q.map(value => value > 0 ? value : 0);
  return { h1: [u, v, w], q, h2, y: h2[0] + .3 * h2[1] + .3 * h2[2] };
}

test('the displayed network evaluates its stated matrices and every intermediate', () => {
  const parameters = networkParameters();
  assert.deepEqual(parameters.beta0, [0, 0, 0]);
  assert.deepEqual(parameters.omega0, [[1, 0], [0, 1], [-1, -1]]);
  assert.deepEqual(parameters.beta1, [1, 1, 1]);
  assert.deepEqual(parameters.omega1, [[-1, -1, -1], [-2, -1, -1], [-1, -2, -1]]);
  assert.deepEqual(parameters.omega2, [1, .3, .3]);
  for (const pinch of [.5, 1, 1.37, 2, 3]) {
    for (const [x, y] of [[0, 0], [.2, .3], [-.2, .3], [.3, -.2], [-.3, -.2], [1.6, -1.2]]) {
      const actual = evaluateNetwork(x, y, { pinch });
      const independent = expected(x, y, pinch);
      assert.deepEqual(actual.x, [x, y]);
      actual.h1.forEach((value, index) => close(value, independent.h1[index], 'first feature'));
      actual.secondPreactivation.forEach((value, index) => close(value, independent.q[index], 'second preactivation'));
      actual.h2.forEach((value, index) => close(value, independent.h2[index], 'second feature'));
      close(actual.shallow, independent.q[0], 'shallow readout');
      close(actual.preactivation, actual.shallow, 'same affine calculation before the added ReLU');
      close(actual.combined, independent.y, 'output');
      close(actual.y, actual.combined, 'output alias');
    }
  }
});

test('the first new neuron has the exact hexagonal support and local peak', () => {
  const corners = [[1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1]];
  close(evaluateNetwork(0, 0).unit1, 1, 'peak');
  for (const [x, y] of corners) {
    close(evaluateNetwork(x, y).unit1, 0, 'support vertex');
    close(evaluateNetwork(.5 * x, .5 * y).unit1, .5, 'inside along ray');
    close(evaluateNetwork(1.1 * x, 1.1 * y).unit1, 0, 'outside along ray');
  }
  for (let x = -2; x <= 2; x += .25) for (let y = -2; y <= 2; y += .25) {
    const evaluation = evaluateNetwork(x, y);
    const radius = Math.max(Math.abs(x), Math.abs(y), Math.abs(x + y));
    close(evaluation.h1.reduce((sum, value) => sum + value, 0), radius, 'hexagonal radius');
    close(evaluation.unit1, Math.max(0, 1 - radius), 'exact tent');
  }
});

test('the weight control affects only the second new unit, with a boundary at x1=1/pinch', () => {
  for (const pinch of [.5, .75, 1, 2, 3]) {
    const parameters = networkParameters({ pinch });
    assert.equal(parameters.omega1[1][0], -pinch);
    close(evaluateNetwork(1 / pinch, 0, { pinch }).unit2, 0, 'controlled boundary');
    close(evaluateNetwork(.5 / pinch, 0, { pinch }).unit2, .5, 'controlled interior');
    close(evaluateNetwork(1.1 / pinch, 0, { pinch }).unit2, 0, 'controlled exterior');
    const focal = evaluateNetwork(.2, .3, { pinch });
    const baseline = evaluateNetwork(.2, .3);
    assert.deepEqual(focal.h1, baseline.h1);
    close(focal.unit1, baseline.unit1, 'unit1 remains fixed');
    close(focal.unit3, baseline.unit3, 'unit3 remains fixed');
  }
});

test('every second feature and the combined output have bounded support throughout the control range', () => {
  for (const pinch of [.5, 1, 2, 3]) for (const [x, y] of [[3, 0], [0, 3], [-3, 3], [-3, 0], [0, -3], [3, -3]]) {
    const evaluation = evaluateNetwork(x, y, { pinch });
    assert.deepEqual(evaluation.h2, [0, 0, 0]);
    assert.equal(evaluation.combined, 0);
  }
});

test('the fixed-feature shallow readout is affine along rays whereas the new response clips to zero', () => {
  for (const [x, y] of [[.3, .1], [-.2, .4], [-.3, -.4], [.4, -.1]]) {
    const atOne = evaluateNetwork(x, y);
    for (const scale of [0, .5, 2, 10]) {
      const scaled = evaluateNetwork(scale * x, scale * y);
      close(scaled.shallow, 1 + scale * (atOne.shallow - 1), 'fixed-feature affine readout');
    }
    assert.equal(evaluateNetwork(10 * x, 10 * y).unit1, 0);
  }
});

test('all named surface modes expose the correct network quantity and invalid controls are rejected', () => {
  const evaluation = evaluateNetwork(.2, .3);
  for (const mode of ['h11', 'h12', 'h13', 'shallow', 'preactivation', 'unit1', 'unit2', 'unit3', 'combined']) {
    close(surfaceValue(mode, .2, .3), evaluation[mode], mode);
  }
  assert.deepEqual(PINCH_RANGE, [.5, 3]);
  assert.throws(() => evaluateNetwork(NaN, 0), TypeError);
  assert.throws(() => evaluateNetwork(0, Infinity), TypeError);
  for (const pinch of [0, .49, 3.01, NaN, Infinity]) assert.throws(() => evaluateNetwork(0, 0, { pinch }), RangeError);
  assert.throws(() => surfaceValue('unknown', 0, 0), RangeError);
});
