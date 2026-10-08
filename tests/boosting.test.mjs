import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
const math = require('../docs/boosting/model.js');
const close = (actual, expected, tolerance = 1e-11) => {
  assert.ok(Number.isFinite(actual), `Expected finite value; received ${actual}`);
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} should equal ${expected}`);
};

test('a stump fits group means and selects the best split, with stable ties', () => {
  assert.deepEqual(math.fitStump([1, 2, 3], [1, -1, 1]), {
    values: [1, 0, 0], threshold: 1.5, left: 1, right: 0, sse: 2
  });
  // Unsorted inputs and repeated feature values preserve observation order.
  const tree = math.fitStump([3, 1, 2, 2], [8, 1, 3, 5]);
  assert.equal(tree.threshold, 2.5);
  assert.equal(tree.left, 3);
  assert.equal(tree.right, 8);
  assert.deepEqual(tree.values, [8, 3, 3, 3]);
  assert.equal(tree.sse, 8);
  assert.deepEqual(math.fitStump([2, 2], [1, 3]), {
    values: [2, 2], threshold: null, left: 2, right: 2, sse: 2
  });
});

test('the example starts explicitly at zero and its first two updates are exact', () => {
  const path = math.buildPath({ rounds: 2 });
  assert.equal(path.length, 3);
  assert.deepEqual(path[0], {
    round: 0, pred: [0, 0, 0], residual: [1, -1, 1], loss: 1.5, tree: null
  });
  assert.deepEqual(path[1].correction, [1, 0, 0]);
  assert.deepEqual(path[1].pred, [0.5, 0, 0]);
  assert.deepEqual(path[1].residual, [0.5, -1, 1]);
  close(path[1].loss, 1.125);
  assert.equal(path[2].tree.threshold, 2.5);
  assert.deepEqual(path[2].correction, [-0.25, -0.25, 1]);
  assert.deepEqual(path[2].pred, [0.375, -0.125, 0.5]);
  assert.deepEqual(path[2].residual, [0.625, -0.875, 0.5]);
  close(path[2].loss, 0.703125);
});

test('each fitted tree is a descent direction and shrinkage gives the predicted loss decrease', () => {
  for (const rate of [0.05, 0.25, 0.5, 1]) {
    const path = math.buildPath({ rate, rounds: 24 });
    for (let k = 1; k < path.length; k++) {
      const previous = path[k - 1];
      const current = path[k];
      const h = current.correction;
      // Least-squares leaf means imply residual·h = h·h. This checks the
      // actual descent geometry, beyond merely checking monotone numbers.
      close(math.dot(previous.residual, h), math.dot(h, h));
      close(previous.loss - current.loss, (rate - rate * rate / 2) * math.dot(h, h));
      assert.ok(current.loss < previous.loss);
      for (let i = 0; i < 3; i++) {
        close(current.pred[i], previous.pred[i] + rate * h[i]);
        close(current.residual[i], [1, -1, 1][i] - current.pred[i]);
      }
    }
  }
});

test('angles are computed in prediction space, including the first fitted direction', () => {
  close(math.angleDegrees([1, 0], [0, 1]), 90);
  close(math.angleDegrees([1, 0], [-1, 0]), 180);
  close(math.angleDegrees([1, -1, 1], [1, 0, 0]), Math.acos(1 / Math.sqrt(3)) * 180 / Math.PI);
  assert.equal(math.angleDegrees([0, 0], [1, 0]), null);
});

test('an acute direction improves with a small step but overshooting can worsen loss', () => {
  const small = math.stepGeometry(60, 0.5);
  close(small.step[0], 0.25);
  close(small.step[1], Math.sqrt(3) / 4);
  close(small.beforeLoss, 0.5);
  close(small.afterLoss, 0.375);
  assert.ok(small.improvement > 0);
  assert.ok(math.stepGeometry(60, 1.2).improvement < 0);
  assert.ok(math.stepGeometry(90, 0.1).improvement < 0);
  assert.ok(math.stepGeometry(120, 0.1).improvement < 0);
  assert.equal(math.stepGeometry(90, 0.1).maxRelativeStep, 0);
});

test('the exact improvement bound is 0 < relativeStep < 2 cos(theta)', () => {
  for (const angle of [0, 15, 30, 60, 80, 89]) {
    const boundary = 2 * Math.cos(angle * Math.PI / 180);
    for (const length of [0.2, 1, 3]) {
      const inside = math.stepGeometry(angle, boundary / 2, length);
      const endpoint = math.stepGeometry(angle, boundary, length);
      const outside = math.stepGeometry(angle, boundary * 1.1, length);
      close(inside.maxRelativeStep, boundary);
      assert.ok(inside.improvement > 0);
      close(endpoint.improvement, 0);
      assert.ok(outside.improvement < 0);
      close(inside.improvement,
        length ** 2 * ((boundary / 2) * Math.cos(angle * Math.PI / 180) - (boundary / 2) ** 2 / 2));
    }
  }
});

test('path construction does not mutate input or alias residuals across rounds', () => {
  const target = [1, -1, 1];
  const x = [1, 2, 3];
  const path = math.buildPath({ target, x, rounds: 2 });
  path[0].residual[0] = 123;
  assert.deepEqual(target, [1, -1, 1]);
  assert.deepEqual(x, [1, 2, 3]);
  assert.deepEqual(path[1].residual, [0.5, -1, 1]);
  path[1].correction[0] = 123;
  assert.deepEqual(path[1].tree.values, [1, 0, 0]);
});
