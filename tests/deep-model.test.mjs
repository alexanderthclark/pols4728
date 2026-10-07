import assert from 'node:assert/strict';
import test from 'node:test';
import {
  DEFAULT_THRESHOLD, evaluateNetwork, foldedPreimages, foldedValue, INPUT_DOMAIN,
  networkParameters, OUTPUT_WEIGHTS, parameterCount, PARAMETER_COUNTS,
  preactivationRoots, surfaceValue, THRESHOLD_RANGE,
} from '../docs/deep/model.mjs';

const close = (actual, expected, message = '') => assert.ok(Math.abs(actual - expected) < 1e-10,
  message + ': ' + actual + ' should equal ' + expected);

// Independent piecewise equations for the fold and downstream curve.
function fold(x) {
  return x <= 0 ? 0 : x <= 1 ? x : x <= 2 ? 2 - x : x - 2;
}
function downstream(q, threshold) {
  return q <= .2 ? 0 : q <= threshold ? q - .2
    : q <= .8 ? -q + 2 * threshold - .2 : q + 2 * threshold - 1.8;
}

test('the shared-fold construction has the stated matrices and defaults', () => {
  assert.equal(DEFAULT_THRESHOLD, .5);
  assert.deepEqual(THRESHOLD_RANGE, [.35, .65]);
  assert.deepEqual(INPUT_DOMAIN, [-.25, 3.25]);
  assert.deepEqual(OUTPUT_WEIGHTS, [1, -2, 2]);
  assert.deepEqual(networkParameters(), {
    beta0: [0, -1, -2], omega0: [[1], [1], [1]],
    beta1: [-.2, -.5, -.8],
    omega1: [[1, -2, 2], [1, -2, 2], [1, -2, 2]],
    beta2: 0, omega2: [1, -2, 2],
  });
});

test('every intermediate matches the common folded coordinate and independent downstream equations', () => {
  for (const threshold of [.35, .4, .45, .5, .55, .6, .65]) {
    for (const x of [-2, -.25, 0, .2, .6, 1, 1.3, 1.7, 2, 2.3, 2.8, 3.25, 5]) {
      const actual = evaluateNetwork(x, { threshold });
      assert.equal(actual.x, x);
      assert.deepEqual(actual.firstPreactivation, [x, x - 1, x - 2]);
      actual.h1.forEach((value, index) => close(value, Math.max(0, x - index), 'first feature'));
      close(actual.fold, fold(x), 'fold coordinate');
      close(actual.q, fold(x), 'q alias');
      close(actual.shallow, fold(x), 'shallow fold alias');
      [.2, threshold, .8].forEach((level, index) => {
        close(actual.secondPreactivation[index], fold(x) - level, 'shared input to new unit');
        close(actual.h2[index], Math.max(0, fold(x) - level), 'new activation');
      });
      close(actual.preactivation, fold(x) - threshold, 'middle unit before ReLU');
      close(actual.y, downstream(fold(x), threshold), 'output');
      close(actual.combined, actual.y, 'output alias');
      close(foldedValue(fold(x), { threshold }), actual.y, 'same downstream curve');
    }
  }
});

test('one interior folded coordinate represents three input pieces with the same prediction', () => {
  for (const threshold of [.35, .5, .65]) for (const q of [.1, .3, .5, .7, .9]) {
    const inputs = foldedPreimages(q);
    assert.equal(new Set(inputs).size, 3);
    inputs.forEach(x => {
      const actual = evaluateNetwork(x, { threshold });
      close(actual.q, q, 'folded preimage');
      close(actual.y, downstream(q, threshold), 'reused downstream prediction');
    });
  }
  assert.deepEqual(foldedPreimages(0), [0, 2, 2]);
  assert.deepEqual(foldedPreimages(1), [1, 1, 3]);
  close(evaluateNetwork(-.25).q, 0, 'extra constant branch at zero');
});

test('every downstream threshold creates one crossing in each nonconstant folded branch', () => {
  for (const threshold of [.35, .5, .65]) {
    const roots = preactivationRoots({ threshold });
    [.2, threshold, .8].forEach((level, unit) => {
      assert.deepEqual(roots[unit], [level, 2 - level, 2 + level]);
      roots[unit].forEach((root, branch) => {
        assert.ok(root > branch && root < branch + 1, 'inside a fixed first-layer interval');
        close(evaluateNetwork(root, { threshold }).secondPreactivation[unit], 0, 'copied crossing');
        const before = evaluateNetwork(root - 1e-5, { threshold }).secondPreactivation[unit];
        const after = evaluateNetwork(root + 1e-5, { threshold }).secondPreactivation[unit];
        assert.ok(before * after < 0, 'genuine preactivation crossing');
      });
    });
  }
});

test('the control changes only the middle bias and moves its three crossings as a group', () => {
  for (const x of [.1, .4, .7, 1.3, 1.7, 2.7, 3.25]) {
    const low = evaluateNetwork(x, { threshold: .35 });
    const high = evaluateNetwork(x, { threshold: .65 });
    assert.deepEqual(high.h1, low.h1);
    close(high.q, low.q, 'fold stays fixed');
    close(high.secondPreactivation[1] - low.secondPreactivation[1], -.3, 'middle bias shift');
    close(high.unit1, low.unit1, 'first new unit fixed');
    close(high.unit3, low.unit3, 'third new unit fixed');
    close(high.y - low.y, -2 * (high.unit2 - low.unit2), 'output changes through middle unit only');
  }
  assert.deepEqual(preactivationRoots({ threshold: .35 })[1], [.35, 1.65, 2.35]);
  assert.deepEqual(preactivationRoots({ threshold: .65 })[1], [.65, 1.35, 2.65]);
});

test('dense parameter counts include all weight and bias slots consistently', () => {
  assert.equal(parameterCount([1, 3, 3, 1]), (3 + 3) + (9 + 3) + (3 + 1));
  assert.equal(parameterCount([1, 10, 1]), 2 * 10 + 10 + 1);
  assert.equal(parameterCount([1, 7, 1]), 2 * 7 + 7 + 1);
  assert.deepEqual(PARAMETER_COUNTS, {
    deep: 22, shallowMatch: 31, shallowSameBudgetWidth: 7, shallowSameBudget: 22,
  });
  const parameters = networkParameters();
  const storedSlots = parameters.beta0.length + parameters.omega0.flat().length
    + parameters.beta1.length + parameters.omega1.flat().length
    + parameters.omega2.length + 1;
  assert.equal(storedSlots, PARAMETER_COUNTS.deep, 'identical rows and zero values still occupy dense slots');
});

test('all named response modes report the intended quantities', () => {
  for (const x of [.1, .6, 1.4, 1.9, 2.6, 3.1]) {
    const q = fold(x);
    const expected = {
      h11: Math.max(0, x), h12: Math.max(0, x - 1), h13: Math.max(0, x - 2),
      fold: q, shallow: q, preactivation: q - .5,
      unit1: Math.max(0, q - .2), unit2: Math.max(0, q - .5), unit3: Math.max(0, q - .8),
      combined: downstream(q, .5),
    };
    for (const [mode, value] of Object.entries(expected)) close(surfaceValue(mode, x), value, mode);
  }
});

test('the global context retains the constant left branch and rising right tail', () => {
  close(evaluateNetwork(-.25).y, 0, 'left constant response');
  close(evaluateNetwork(1).y, .2, 'surviving fold joint');
  close(evaluateNetwork(2).y, 0, 'flat response around fold minimum');
  close(evaluateNetwork(3.25).y, .45, 'right endpoint');
  assert.ok(evaluateNetwork(10).y > evaluateNetwork(3.25).y);
});

test('invalid inputs, thresholds, folded preimages, layer widths, and modes fail explicitly', () => {
  for (const value of [NaN, Infinity, -Infinity, '1']) {
    assert.throws(() => evaluateNetwork(value), TypeError);
    assert.throws(() => foldedValue(value), TypeError);
  }
  for (const threshold of [.34, .66, NaN, Infinity]) {
    assert.throws(() => evaluateNetwork(0, { threshold }), RangeError);
    assert.throws(() => foldedValue(.5, { threshold }), RangeError);
    assert.throws(() => preactivationRoots({ threshold }), RangeError);
  }
  for (const q of [-.01, 1.01, NaN, Infinity, '0.5']) assert.throws(() => foldedPreimages(q), RangeError);
  for (const widths of [[], [1], [1, 0, 1], [1, 2.5, 1], [1, Infinity], 'widths']) {
    assert.throws(() => parameterCount(widths), RangeError);
  }
  assert.throws(() => surfaceValue('downstream', .5), RangeError);
  assert.throws(() => surfaceValue('unknown', 0), RangeError);
});
