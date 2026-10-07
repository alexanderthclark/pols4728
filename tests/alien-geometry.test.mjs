import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {fitRule, summarize} from '../docs/alien/model.mjs';
import {predictionLayout, crowdLayout, decompositionSegments} from '../docs/alien/geometry.mjs';

const near = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-10, `${actual} ≠ ${expected}`);
const fixture = JSON.parse(readFileSync(new URL('../docs/alien/samples.json', import.meta.url)));
const stats = summarize(fixture.games.map(fitRule));

test('all 1,000 aliens retain identity as the crowd becomes 468 loss and 532 win predictions', () => {
  const crowd = crowdLayout(stats.predictions.length);
  const layout = predictionLayout(stats.predictions);
  assert.equal(crowd.length, 1000);
  assert.equal(layout.length, crowd.length);
  assert.equal(layout.filter(point => point.prediction === 0).length, 468);
  assert.equal(layout.filter(point => point.prediction === 1).length, 532);
  for (let index = 0; index < layout.length; index++) {
    assert.equal(crowd[index].index, index);
    assert.equal(layout[index].index, index);
    assert.equal(layout[index].prediction, stats.predictions[index]);
  }
  assert.equal(new Set(crowd.map(({x, y}) => `${x},${y}`)).size, crowd.length);
  assert.equal(new Set(layout.map(({x, y}) => `${x},${y}`)).size, layout.length);
});

test('both prediction groups stay centered on their outcome, including partial final rows', () => {
  const layout = predictionLayout(stats.predictions);
  for (const [prediction, anchor, count] of [[0, 180, 468], [1, 620, 532]]) {
    const group = layout.filter(point => point.prediction === prediction);
    near(group.reduce((sum, point) => sum + point.x, 0) / count, anchor);
    assert.equal(Math.max(...group.map(point => point.y)), 375);
    assert.equal(Math.min(...group.map(point => point.y)), 380 - Math.ceil(count / 20) * 5);
    const rows = new Map();
    for (const point of group) {
      if (!rows.has(point.y)) rows.set(point.y, []);
      rows.get(point.y).push(point.x);
    }
    for (const row of rows.values()) {
      near(row.reduce((sum, x) => sum + x, 0) / row.length, anchor);
      for (let index = 1; index < row.length; index++) near(row[index] - row[index - 1], 5);
    }
  }
});

test('small and single-outcome prediction groups preserve order on custom scales', () => {
  assert.deepEqual(predictionLayout([1, 0, 1, 0, 1], {
    left: 10, right: 100, baseline: 50, columns: 2, gap: 4,
  }), [
    {index: 0, prediction: 1, x: 98, y: 46},
    {index: 1, prediction: 0, x: 8, y: 46},
    {index: 2, prediction: 1, x: 102, y: 46},
    {index: 3, prediction: 0, x: 12, y: 46},
    {index: 4, prediction: 1, x: 100, y: 42},
  ]);
  assert.deepEqual(predictionLayout([0]), [{index: 0, prediction: 0, x: 180, y: 375}]);
  assert.deepEqual(predictionLayout([]), []);
});

test('crowd rows use one consistent gap and retain the original index order', () => {
  assert.deepEqual(crowdLayout(5, {left: 2, top: 3, columns: 3, gap: 7}), [
    {index: 0, x: 2, y: 3},
    {index: 1, x: 9, y: 3},
    {index: 2, x: 16, y: 3},
    {index: 3, x: 2, y: 10},
    {index: 4, x: 9, y: 10},
  ]);
  assert.deepEqual(crowdLayout(0), []);
});

test('error components have exact proportional widths and accumulate to the MSE width', () => {
  const {segments, totalWidth} = decompositionSegments(stats);
  assert.deepEqual(segments.map(segment => segment.key), ['biasSquared', 'variance', 'noise']);
  let expectedStart = 90;
  for (const segment of segments) {
    assert.equal(segment.value, stats[segment.key]);
    assert.equal(segment.width, stats[segment.key] * 640);
    assert.equal(segment.x, expectedStart);
    expectedStart += segment.width;
  }
  assert.equal(totalWidth, stats.mse * 640);
  near(expectedStart - 90, totalWidth);
  assert.equal(segments[2].width, 0);
  near(segments[2].x, 90 + totalWidth);
  const scaled = decompositionSegments(stats, {x: 10, width: 300, max: 0.5});
  assert.equal(scaled.totalWidth, stats.mse / 0.5 * 300);
  assert.equal(scaled.segments[0].width, stats.biasSquared / 0.5 * 300);
});

test('zero error keeps all three components at the origin with zero width', () => {
  const result = decompositionSegments({biasSquared: 0, variance: 0, noise: 0, mse: 0});
  assert.equal(result.totalWidth, 0);
  for (const segment of result.segments) {
    assert.equal(segment.x, 90);
    assert.equal(segment.width, 0);
  }
});

test('invalid geometry and inconsistent decompositions are rejected', () => {
  assert.throws(() => predictionLayout([0, 2]), /0 or 1/);
  assert.throws(() => predictionLayout([0], {columns: 0}), /columns/);
  assert.throws(() => predictionLayout([0], {gap: 0}), /gap/);
  assert.throws(() => predictionLayout([0], {baseline: Infinity}), /baseline/);
  assert.throws(() => predictionLayout([0], {left: 4, right: 4}), /right/);
  assert.throws(() => crowdLayout(-1), /count/);
  assert.throws(() => crowdLayout(1.5), /count/);
  assert.throws(() => crowdLayout(1, {top: NaN}), /top/);
  assert.throws(() => decompositionSegments(stats, {max: 0}), /max/);
  assert.throws(() => decompositionSegments(stats, {width: -1}), /width/);
  assert.throws(() => decompositionSegments(stats, {max: 0.1}), /mse/);
  assert.throws(() => decompositionSegments({...stats, variance: -1}), /variance/);
  assert.throws(() => decompositionSegments({...stats, mse: 0.1}), /sum to mse/);
});
