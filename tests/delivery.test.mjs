import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
const example = require('../docs/trees/delivery.js');
const { x, y, baseline, rounds, mse, predict, candidates } = example;
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-11, `${actual} should equal ${expected}`);

test('the delivery example preserves its first two stumps and fits all eight to current residual means', () => {
  assert.equal(baseline, 12);
  assert.equal(rounds.length, 8);
  assert.deepEqual(rounds.slice(0, 2).map(round => [round.threshold, round.left, round.right]), [[8, -3, 5], [3.5, -3, 1]]);
  assert.deepEqual(rounds[0].residual, [-6, -6, -1, -1, -1, 5, 5, 5]);
  assert.deepEqual(rounds[1].residual, [-3, -3, 2, 2, 2, 0, 0, 0]);
  for (const round of rounds) {
    assert.deepEqual(round.previous, x.map(input => predict(round.number - 1, input)));
    assert.deepEqual(round.residual, y.map((observed, i) => observed - round.previous[i]));
    for (const isLeft of [true, false]) {
      const group = round.residual.filter((_, i) => (x[i] <= round.threshold) === isLeft);
      const mean = group.reduce((sum, value) => sum + value, 0) / group.length;
      assert.equal(isLeft ? round.left : round.right, mean);
    }
    assert.deepEqual(round.prediction, round.previous.map((value, i) => value + round.correction[i]));
  }
});

test('each fitted partition is the unique SSE minimum among all seven candidates', () => {
  const thresholds = [2, 3.5, 5, 6.5, 8, 9.5, 11];
  const expectedSSE = [
    [762 / 7, 54, 898 / 15, 52, 30, 250 / 3, 850 / 7],
    [138 / 7, 6, 322 / 15, 28, 30, 30, 30]
  ];
  for (const [index, round] of rounds.entries()) {
    const rows = candidates(round.number);
    assert.deepEqual(rows.map(row => row.threshold), thresholds);
    // Independently evaluate the two within-group sums of squared deviations.
    const scores = thresholds.map(threshold => {
      const left = [], right = [];
      x.forEach((input, i) => (input <= threshold ? left : right).push(round.residual[i]));
      return [left, right].reduce((total, group) => {
        const mean = group.reduce((sum, value) => sum + value, 0) / group.length;
        return total + group.reduce((sum, value) => sum + (value - mean) ** 2, 0);
      }, 0);
    });
    if (index < 2) scores.forEach((score, candidate) => close(score, expectedSSE[index][candidate]));
    rows.forEach((row, candidate) => {
      close(row.sse, scores[candidate]);
      close(row.mse, scores[candidate] / x.length);
    });
    const winner = thresholds.indexOf(round.threshold);
    assert.ok(scores.every((score, candidate) => candidate === winner || score > scores[winner]));
    close(scores[winner] / y.length, round.mse);
    assert.deepEqual(rows[winner].values, round.correction);
  }
});

test('candidate rows expose the residual leaf means and preserve the fitted example', () => {
  const means = [
    [[-6, 6 / 7], [-6, 2], [-13 / 3, 13 / 5], [-3.5, 3.5], [-3, 5], [-5 / 3, 5], [-5 / 7, 5]],
    [[-3, 3 / 7], [-3, 1], [-4 / 3, 4 / 5], [-0.5, 0.5], [0, 0], [0, 0], [0, 0]]
  ];
  const before = JSON.stringify({ x, y, rounds });
  for (const [index, round] of rounds.slice(0, 2).entries()) {
    const rows = candidates(round.number);
    assert.equal(rows.length, 7);
    rows.forEach((row, cut) => {
      close(row.left, means[index][cut][0]);
      close(row.right, means[index][cut][1]);
      assert.deepEqual(row.values, [...Array(cut + 1).fill(row.left), ...Array(7 - cut).fill(row.right)]);
      assert.ok(Object.isFrozen(row));
      assert.ok(Object.isFrozen(row.values));
    });
    assert.ok(Object.isFrozen(rows));
    assert.throws(() => { rows[0].values[0] = 100; }, TypeError);
  }
  assert.equal(JSON.stringify({ x, y, rounds }), before);
  assert.throws(() => candidates(0), RangeError);
  assert.throws(() => candidates(9), RangeError);
});

test('the accumulated training predictions have MSE 18.75, 3.75, and 0.75', () => {
  assert.equal(mse(x.map(input => predict(0, input))), 18.75);
  assert.deepEqual(rounds[0].prediction, [9, 9, 9, 9, 9, 17, 17, 17]);
  assert.deepEqual(rounds[1].prediction, [6, 6, 10, 10, 10, 18, 18, 18]);
  assert.deepEqual(rounds.slice(0, 2).map(round => round.mse), [3.75, 0.75]);
  for (const round of rounds) {
    assert.deepEqual(x.map(input => predict(round.number, input)), round.prediction);
    assert.equal(mse(round.prediction), round.mse);
  }
});

test('the third fitted stump corrects the long deliveries and leaves a positive training error', () => {
  const third = rounds[2];
  assert.deepEqual(third.residual, [0, 0, 1, 1, 1, -1, -1, -1]);
  assert.equal(third.threshold, 8);
  close(third.left, 3 / 5);
  assert.equal(third.right, -1);
  const expected = [6.6, 6.6, 10.6, 10.6, 10.6, 17, 17, 17];
  third.prediction.forEach((value, index) => close(value, expected[index]));
  close(third.mse, 0.15);
  close(predict(3, 8), 10.6);
  close(predict(3, 8.000001), 17);
  close(predict(3, 2, 0), 6);
  close(predict(3, 2, 0.5), 6.3);
  assert.ok(third.mse > 0);
});

test('continuing alternates the same cuts and approaches zero with the exact residual recurrence', () => {
  assert.deepEqual(rounds.map(round => round.threshold), [8, 3.5, 8, 3.5, 8, 3.5, 8, 3.5]);
  // In exact arithmetic, odd residuals are scaled copies of r1, and even
  // residuals are scaled copies of r2. Neither nonzero pattern vanishes after
  // finitely many rounds; the displayed continuation must not claim exact fit.
  const oddPattern = [-3, -3, 2, 2, 2, 0, 0, 0];
  const evenPattern = [0, 0, 1, 1, 1, -1, -1, -1];
  for (const round of rounds) {
    const odd = round.number % 2 === 1;
    const factor = 5 ** (-(round.number - (odd ? 1 : 2)) / 2);
    const pattern = odd ? oddPattern : evenPattern;
    y.forEach((observed, i) => close(observed - round.prediction[i], pattern[i] * factor));
    close(round.mse, 18.75 / 5 ** round.number);
    assert.ok(round.mse > 0);
  }
  const final = [6, 6, 10.992, 10.992, 10.992, 17.008, 17.008, 17.008];
  rounds[7].prediction.forEach((value, index) => close(value, final[index]));
  close(rounds[7].mse, 0.000048);
});

test('prediction applies the left-boundary convention and adds only the current partial update', () => {
  const probes = [0, 1, 3.5, 3.500001, 8, 8.000001, 12, 15];
  const first = [9, 9, 9, 9, 9, 17, 17, 17];
  const second = [6, 6, 6, 10, 10, 18, 18, 18];
  assert.deepEqual(probes.map(input => predict(1, input)), first);
  assert.deepEqual(probes.map(input => predict(2, input)), second);
  for (const [i, input] of probes.entries()) {
    assert.equal(predict(0, input), 12);
    assert.equal(predict(1, input, 0), 12);
    assert.equal(predict(2, input, 0), first[i]);
    assert.equal(predict(1, input, 0.5), 12 + (first[i] - 12) / 2);
    assert.equal(predict(2, input, 0.5), first[i] + (second[i] - first[i]) / 2);
  }
});

test('calculating predictions and errors preserves observations and fitted rounds', () => {
  const before = JSON.stringify({ x, y, rounds });
  const predictions = [2, 4], observations = [1, 7];
  assert.equal(mse(predictions, observations), 5);
  assert.deepEqual(predictions, [2, 4]);
  assert.deepEqual(observations, [1, 7]);
  predict(2, 5, 0.3);
  assert.equal(JSON.stringify({ x, y, rounds }), before);
  assert.notEqual(rounds[0].prediction, rounds[1].previous);
  assert.notEqual(rounds[0].residual, rounds[1].residual);
  assert.throws(() => { rounds[0].correction[0] = 100; }, TypeError);
  assert.throws(() => predict(9, 5), RangeError);
  assert.throws(() => predict(1, NaN), TypeError);
  assert.throws(() => predict(2, 5, -0.1), RangeError);
});
