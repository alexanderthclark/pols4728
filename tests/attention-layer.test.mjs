import assert from 'node:assert/strict';
import test from 'node:test';
import { X, WQ, WK, WV, WO, dk, handWeights, tokens, attention, matmul, transpose, softmax, mixRows } from '../docs/attention/model.mjs';

const near = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-12, `${actual} != ${expected}`);
const nearRow = (actual, expected) => {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, j) => near(value, expected[j]));
};

test('token rows, projections, and hand-chosen AX example match independent arithmetic', () => {
  assert.deepEqual(tokens, ['The', 'senator', 'thanked', 'staff']);
  assert.equal(X.length, 4);
  X.forEach(row => assert.equal(row.length, 3));
  assert.equal(dk, 2);
  const result = attention();
  assert.deepEqual(result.Q, [[0, 0], [0, 2], [1, 2], [2, 4]]);
  assert.deepEqual(result.K, [[0, 0], [2, 0], [0, 1], [0, 2]]);
  assert.deepEqual(result.V, [[.5, -.5], [2.5, -.5], [.5, .5], [.5, 1.5]]);
  nearRow(mixRows(handWeights), [1.2, .5, 1]);
  nearRow(mixRows([.25, .25, .25, .25]), [.5, .75, 1]);
});

test('query dot key, scaling, and softmax agree with the thanked-row hand calculation', () => {
  const { rawScores, S, A } = attention();
  assert.deepEqual(rawScores[2], [0, 2, 2, 4]);
  nearRow(S[2], [0, Math.SQRT2, Math.SQRT2, 2 * Math.SQRT2]);
  const e = Math.exp(Math.SQRT2);
  const denominator = 1 + 2 * e + e * e;
  nearRow(A[2], [1 / denominator, e / denominator, e / denominator, e * e / denominator]);
  assert.ok(A[2][3] > A[2][1]);
  assert.notEqual(A[2][3], A[3][2], 'attention is directed, not symmetric');
  A.forEach(row => {
    near(row.reduce((sum, value) => sum + value, 0), 1);
    row.forEach(value => assert.ok(value > 0));
  });
});

test('left multiplication mixes complete token rows and value projection commutes with mixing', () => {
  const { A, AX, V, Y, head } = attention();
  assert.equal(AX.length, 4);
  assert.equal(AX[0].length, 3);
  assert.equal(Y.length, 4);
  assert.equal(Y[0].length, 2);
  assert.equal(head[0].length, 3);
  for (let i = 0; i < 4; i++) {
    nearRow(AX[i], [2 * A[i][1], A[i][2] + 2 * A[i][3], 1]);
    nearRow(Y[i], [AX[i][0] + .5, AX[i][1] - .5]);
    nearRow(head[i], [Y[i][0], Y[i][1], 0]);
    nearRow(matmul(AX, WV)[i], Y[i]);
  }
  assert.deepEqual(V, matmul(X, WV));
  assert.deepEqual(matmul(X, WQ), attention().Q);
  assert.deepEqual(matmul(X, WK), attention().K);
  assert.deepEqual(matmul(Y, WO), head);
  assert.deepEqual(transpose([[1, 2, 3], [4, 5, 6]]), [[1, 4], [2, 5], [3, 6]]);
});

test('causal mask includes self, excludes future keys, and renormalizes allowed scores', () => {
  const { A, maskedScores, Y, V } = attention({ causal: true });
  assert.deepEqual(A[0], [1, 0, 0, 0]);
  nearRow(Y[0], V[0]);
  for (let i = 0; i < A.length; i++) {
    near(A[i].reduce((sum, value) => sum + value, 0), 1);
    for (let j = 0; j < A.length; j++) {
      if (j > i) {
        assert.equal(A[i][j], 0);
        assert.equal(maskedScores[i][j], -Infinity);
      } else assert.ok(A[i][j] > 0);
    }
  }
  const e = Math.exp(Math.SQRT2);
  nearRow(A[2], [1 / (1 + 2 * e), e / (1 + 2 * e), e / (1 + 2 * e), 0]);
});

test('changing a future representation cannot affect earlier causal outputs', () => {
  const altered = X.map(row => row.slice());
  altered[3] = [7, -3, 2];
  const original = attention({ causal: true });
  const changed = attention({ causal: true, input: altered });
  for (let i = 0; i < 3; i++) {
    nearRow(changed.A[i], original.A[i]);
    nearRow(changed.AX[i], original.AX[i]);
    nearRow(changed.Y[i], original.Y[i]);
    nearRow(changed.head[i], original.head[i]);
  }
  assert.notDeepEqual(attention({ input: altered }).Y[2], attention().Y[2],
    'unmasked attention can read the future representation');
});

test('softmax stays stable under large common offsets and permits masked sources', () => {
  nearRow(softmax([1000, 1001, -Infinity]), [1 / (1 + Math.E), Math.E / (1 + Math.E), 0]);
  assert.throws(() => softmax([-Infinity, -Infinity]), RangeError);
  assert.throws(() => matmul([[1, 2]], [[1, 2]]), RangeError);
});
