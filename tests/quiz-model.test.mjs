import assert from 'node:assert/strict';
import test from 'node:test';
import {
  LIMIT, YEAR, normalCDF, coarsePrediction, randomModelSelection,
  sampleCarnies, carnieMoments, integratedCarnies, tutoringRows, tutoringSummary,
} from '../docs/quiz/model.mjs';

const closeTo = (actual, expected, tolerance = 1e-10) => {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} should equal ${expected} within ${tolerance}`);
};

test('normal probabilities agree with independent standard normal reference values', () => {
  closeTo(normalCDF(0), 0.5);
  closeTo(normalCDF(1), 0.8413447460685429);
  closeTo(normalCDF(-1.96), 0.024997895148220435);
  closeTo(normalCDF(3), 0.9986501019683699);
  closeTo(normalCDF(-6), 9.865876450376946e-10, 2e-15);
  assert.equal(normalCDF(-Infinity), 0);
  assert.equal(normalCDF(Infinity), 1);
  assert.ok(Number.isNaN(normalCDF(NaN)));
  for (const x of [0.1, 0.75, 2, 5]) closeTo(normalCDF(x) + normalCDF(-x), 1);
});

test('flooring includes exact thresholds and never reports an infeasible age', () => {
  assert.equal(LIMIT, 730);
  assert.equal(YEAR, 365);
  const inputs = [-Infinity, -730, -1, 0, 364.999, 365, 365.001, 729.999, 730, 1095, Infinity];
  assert.deepEqual(inputs.map(coarsePrediction), [0, 0, 0, 0, 0, 365, 365, 365, 730, 730, 730]);
});

test('many random candidates can win validation without gaining test signal', () => {
  const result = randomModelSelection();
  closeTo(result.singlePerfectProbability, 1 / 1024);
  assert.equal(result.expectedPerfect, 32);
  assert.ok(result.perfectProbability > 0.9999999999999);
  assert.equal(result.testAccuracy, 0.5);
  closeTo(result.testSD, Math.sqrt(0.25 / 10));
  closeTo(randomModelSelection(2, 1).perfectProbability, 0.75);
  closeTo(randomModelSelection(1, 19).perfectProbability, 2 ** -19);
  assert.equal(randomModelSelection(1, 0).perfectProbability, 1);
  assert.throws(() => randomModelSelection(0, 10), /positive integer/);
  assert.throws(() => randomModelSelection(1, 20), /untouched test set/);
});

test('simulated slopes use separate noise draws and both correct sampling distances', () => {
  // Four uniforms per pair, chosen to make the two independent normal draws 1 and -2.
  const uniforms = [1 - Math.exp(-0.5), 0, 1 - Math.exp(-2), 0.5];
  let index = 0;
  const draws = sampleCarnies(() => uniforms[index++], 4, 1);
  closeTo(draws.firstSlopes[0], 1 + 2 / Math.sqrt(73000));
  closeTo(draws.secondSlopes[0], 0.6);
  assert.equal(index, 4);
  assert.deepEqual(sampleCarnies(() => 0, 1, 2), { firstSlopes: [1, 1], secondSlopes: [1, 1] });
  assert.deepEqual(sampleCarnies(() => 0.25, 0, 2), { firstSlopes: [1, 1], secondSlopes: [1, 1] });
  assert.deepEqual(sampleCarnies(Math.random, 1, 0), { firstSlopes: [], secondSlopes: [] });
  assert.throws(() => sampleCarnies(() => 1), /\[0, 1\)/);
});

test('the first two prediction means are unbiased and their variances differ by 730', () => {
  for (const x of [0, 1, 365, 730]) {
    const [first, second] = carnieMoments(x);
    assert.equal(first.mean, x);
    assert.equal(second.mean, x);
    assert.equal(first.biasSquared, 0);
    assert.equal(second.biasSquared, 0);
    closeTo(first.variance, x * x / 73000);
    closeTo(second.variance, x * x / 100);
    closeTo(second.variance, 730 * first.variance);
  }
  closeTo(carnieMoments(365, 4)[0].variance, 7.3);
});

test('threshold probabilities yield the coarsened predictor’s exact discrete moments', () => {
  const third = carnieMoments(365)[2];
  closeTo(third.pAtLeast365, 0.5);
  closeTo(third.pAtLeast730, 0);
  closeTo(third.mean, 182.5);
  closeTo(third.biasSquared, 33306.25);
  closeTo(third.variance, 33306.25);
  assert.equal(carnieMoments(0)[2].variance, 0);
  assert.equal(carnieMoments(0)[2].mean, 0);
  for (const x of [20, 365, 500, 730]) {
    const report = carnieMoments(x, 0)[2];
    assert.equal(report.mean, coarsePrediction(x));
    assert.equal(report.variance, 0);
  }
  assert.throws(() => carnieMoments(-1), /x must be/);
  assert.throws(() => carnieMoments(20, -1), /nonnegative/);
});

test('population integration reproduces normal-distribution reference calculations', () => {
  const [first, second, third] = integratedCarnies();
  closeTo(first.variance, 730 / 300);
  closeTo(second.variance, 730 ** 2 / 300);
  closeTo(first.noise, 3.65);
  closeTo(first.mse, 6.083333333333333);
  closeTo(second.mse, 1779.9833333333333);
  closeTo(third.biasSquared, 43938.70, 0.005);
  closeTo(third.variance, 277.46, 0.005);
  closeTo(third.meanMSE, 44216.16, 0.01);
  closeTo(third.mse, third.meanMSE + 3.65);
  assert.ok(first.mse < second.mse && second.mse < third.mse);
  const large = integratedCarnies(100000);
  closeTo(large[2].meanMSE, 89137.73, 0.005);
  closeTo(large[2].mse, 454137.73, 0.005);
  assert.ok(large[2].mse < large[0].mse);
  const noiseless = integratedCarnies(0);
  closeTo(noiseless[2].biasSquared, YEAR ** 2 / 3);
  assert.equal(noiseless[2].variance, 0);
  assert.equal(noiseless[0].mse, 0);
});

test('count-weighted tutoring comparisons distinguish the three estimates', () => {
  const rows = tutoringRows();
  assert.deepEqual(rows.map(({ group, count, D, Y, yhat, ehat, R, T }) =>
    [group, count, D, Y, yhat, ehat, R, T]), [
    ['Low', 12, 0, 50, 52.5, 0.25, -2.5, -0.25],
    ['Low', 4, 1, 60, 52.5, 0.25, 7.5, 0.75],
    ['High', 4, 0, 90, 97.5, 0.75, -7.5, -0.75],
    ['High', 12, 1, 100, 97.5, 0.75, 2.5, 0.25],
  ]);
  assert.equal(rows.reduce((sum, row) => sum + row.count, 0), 32);
  for (const row of rows) assert.equal(row.R, 10 * row.T);
  assert.deepEqual(tutoringSummary(), { rawDifference: 30, ate: 10, outcomeOnly: 7.5, numerator: 60, denominator: 6, both: 10 });
  rows[0].Y = 999;
  assert.equal(tutoringRows()[0].Y, 50);
});
