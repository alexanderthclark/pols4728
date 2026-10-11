import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
const growth = require('../docs/trees/growth.js');
const { observations, baseline, baseTree, stumps, treeStages, boostStages, evaluate, predict } = growth;
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-10, `${actual} should equal ${expected}`);

const sse = values => {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  return values.reduce((sum, value) => sum + (value - mean) ** 2, 0);
};

test('both methods fit the same 16 deliveries from their common mean with four split nodes', () => {
  assert.deepEqual(observations.map(row => row.id), [...'ABCDEFGHIJKLMNOP']);
  assert.equal(baseline, 14.8125);
  assert.deepEqual(treeStages[0].predictions, Array(16).fill(baseline));
  assert.deepEqual(boostStages[0].predictions, treeStages[0].predictions);
  assert.equal(sse(observations.map(row => row.y)), 286.4375);
  assert.deepEqual(treeStages.map(stage => stage.mse), [17.90234375, 8.5234375, 4.390625, 2.390625, 1.625]);
  boostStages.map(stage => stage.mse).forEach((value, index) => close(value,
    [17.90234375, 8.5234375, 2.58203125, 1.951822916666667, 1.618489583333334][index]));
  for (const stage of [...treeStages, ...boostStages]) {
    const errors = observations.map((row, index) => row.y - stage.predictions[index]);
    assert.equal(stage.mse, errors.reduce((sum, error) => sum + error * error, 0) / observations.length);
  }
});

test('recursive growth chooses the globally largest remaining SSE reduction inside the selected leaf', () => {
  assert.deepEqual(treeStages.slice(1).map(stage => [stage.split.feature, stage.split.threshold, stage.split.parentId]), [
    [0, 6, 'tree-root'], [1, 2, 'tree-root-R'], [1, 2, 'tree-root-L'], [0, 9, 'tree-root-R-R']
  ]);
  assert.deepEqual(treeStages.slice(1).map(stage => stage.split.reduction), [150.0625, 66.125, 32, 12.25]);
  assert.deepEqual(treeStages.slice(1).map(stage => [stage.split.left.value, stage.split.right.value]), [[11.75, 17.875], [15, 20.75], [9.75, 13.75], [19, 22.5]]);
  assert.deepEqual(treeStages.slice(1).map(stage => stage.split.bounds), [
    [[0, 12], [-0.5, 4.5]], [[6, 12], [-0.5, 4.5]], [[0, 6], [-0.5, 4.5]], [[6, 12], [2, 4.5]]
  ]);

  for (let index = 1; index < treeStages.length; index++) {
    const previous = treeStages[index - 1].tree;
    const groups = new Map();
    for (const row of observations) {
      let leaf = previous;
      while (!('value' in leaf)) leaf = row.x[leaf.feature] <= leaf.threshold ? leaf.left : leaf.right;
      if (!groups.has(leaf.id)) groups.set(leaf.id, []);
      groups.get(leaf.id).push(row);
    }
    let greatestReduction = 0;
    for (const rows of groups.values()) {
      for (const feature of [0, 1]) {
        const inputs = [...new Set(rows.map(row => row.x[feature]))].sort((a, b) => a - b);
        for (let cut = 1; cut < inputs.length; cut++) {
          const threshold = (inputs[cut - 1] + inputs[cut]) / 2;
          const left = rows.filter(row => row.x[feature] <= threshold).map(row => row.y);
          const right = rows.filter(row => row.x[feature] > threshold).map(row => row.y);
          greatestReduction = Math.max(greatestReduction, sse(rows.map(row => row.y)) - sse(left) - sse(right));
        }
      }
    }
    assert.equal(treeStages[index].split.reduction, greatestReduction);
  }
});

test('the four additive stumps fit current residual means and minimize residual SSE', () => {
  assert.deepEqual(stumps.map(stump => [stump.feature, stump.threshold]), [[0, 6], [1, 2], [0, 9], [1, 3.5]]);
  const expectedMeans = [[-3.0625, 3.0625], [-2.4375, 2.4375], [-11 / 24, 1.375], [-1 / 3, 1]];
  stumps.forEach((stump, index) => {
    close(stump.left.value, expectedMeans[index][0]);
    close(stump.right.value, expectedMeans[index][1]);
  });
  for (let index = 1; index < boostStages.length; index++) {
    const stage = boostStages[index];
    const residuals = observations.map((row, i) => row.y - boostStages[index - 1].predictions[i]);
    assert.deepEqual(stage.residuals, residuals);
    const scores = [];
    for (const feature of [0, 1]) {
      const inputs = [...new Set(observations.map(row => row.x[feature]))].sort((a, b) => a - b);
      for (let cut = 1; cut < inputs.length; cut++) {
        const threshold = (inputs[cut - 1] + inputs[cut]) / 2;
        const left = residuals.filter((_, i) => observations[i].x[feature] <= threshold);
        const right = residuals.filter((_, i) => observations[i].x[feature] > threshold);
        scores.push(sse(left) + sse(right));
      }
    }
    close(stage.mse * observations.length, Math.min(...scores));
    for (const branch of ['left', 'right']) {
      const group = residuals.filter((_, i) => (observations[i].x[stage.stump.feature] <= stage.stump.threshold) === (branch === 'left'));
      assert.equal(stage.stump[branch].value, group.reduce((sum, value) => sum + value, 0) / group.length);
    }
    assert.equal(stage.trees[0], baseTree);
  }
});

test('local tree splits and global stump splits give different maps with exact left-boundary membership', () => {
  const tree = treeStages.at(-1).tree;
  const trees = boostStages.at(-1).trees;
  assert.equal(evaluate(tree, [6, 2]), 9.75);
  assert.equal(evaluate(tree, [6.00001, 2]), 15);
  assert.equal(evaluate(tree, [6, 2.00001]), 13.75);
  assert.equal(evaluate(tree, [9, 4]), 19);
  assert.equal(evaluate(tree, [9.00001, 4]), 22.5);
  // The tree's final distance cut only partitions its upper-right leaf.
  assert.equal(evaluate(tree, [8, 1]), evaluate(tree, [10, 1]));
  assert.notEqual(predict(trees, [8, 1]), predict(trees, [10, 1]));
  // A stump applies its threshold throughout the other feature's domain.
  for (const stops of [-0.5, 0, 1, 2, 3, 3.5, 4, 4.5]) {
    close(predict(trees, [9.00001, stops]) - predict(trees, [9, stops]), 11 / 6);
  }
  for (const distance of [0, 2, 6, 8, 9, 10, 12]) {
    close(predict(trees, [distance, 3.50001]) - predict(trees, [distance, 3.5]), 4 / 3);
  }
  assert.notDeepEqual(treeStages.at(-1).predictions, boostStages.at(-1).predictions);
  assert.ok(treeStages.at(-1).mse > 0);
  assert.ok(boostStages.at(-1).mse > 0);
});

test('the same four-split budget makes five tree regions and nine stump regions without changing prior stages', () => {
  assert.deepEqual(treeStages.map(stage => [stage.splitCount, stage.regionCount]), [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]);
  assert.deepEqual(boostStages.map(stage => [stage.splitCount, stage.regionCount]), [[0, 1], [1, 2], [2, 4], [3, 6], [4, 9]]);
  assert.equal(treeStages[0].tree.value, 14.8125);
  assert.equal(treeStages[1].tree.left.value, 11.75);
  assert.equal(treeStages[1].tree.right.value, 17.875);
  assert.equal(treeStages[2].tree.left.value, 11.75);
  assert.throws(() => { treeStages[1].tree.left.value = 8; }, TypeError);
  assert.throws(() => { boostStages[0].trees.push(stumps[0]); }, TypeError);
});
