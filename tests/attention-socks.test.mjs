import assert from 'node:assert/strict';
import test from 'node:test';
import { socksAttention, SOCKS_WQ, SOCKS_WK, SOCKS_WV } from '../docs/attention/socks-model.mjs';

const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-12, `${actual} != ${expected}`);
function matrixClose(actual, expected) {
  assert.equal(actual.length, expected.length);
  actual.forEach((row, i) => {
    assert.equal(row.length, expected[i].length);
    row.forEach((value, j) => close(value, expected[i][j]));
  });
}
const logThree = 1.0986122886681098;

test('both token orders match an independent finite-softmax hand calculation', () => {
  const examples = [
    { reverse: false, tokens: ['socks', 'shoes'],
      X: [[1, 0, 1], [0, 1, -1]], Q: [[0], [1]], K: [[logThree], [0]],
      S: [[0, 0], [logThree, 0]], A: [[.5, .5], [.75, .25]],
      Y: [[0], [.5]], AX: [[.5, .5, 0], [.75, .25, .5]], pooled: .25, prediction: 0 },
    { reverse: true, tokens: ['shoes', 'socks'],
      X: [[0, 1, 1], [1, 0, -1]], Q: [[1], [0]], K: [[0], [logThree]],
      S: [[0, logThree], [0, 0]], A: [[.25, .75], [.5, .5]],
      Y: [[-.5], [0]], AX: [[.75, .25, -.5], [.5, .5, 0]], pooled: -.25, prediction: -1 },
  ];
  for (const expected of examples) {
    const result = socksAttention({ reverse: expected.reverse });
    assert.deepEqual(result.tokens, expected.tokens);
    for (const name of ['X', 'Q', 'K', 'S', 'A', 'Y', 'AX']) matrixClose(result[name], expected[name]);
    matrixClose(result.V, [[1], [-1]]);
    close(result.pooled, expected.pooled);
    close(result.prediction, expected.prediction);
    result.A.forEach(row => {
      close(row[0] + row[1], 1);
      row.forEach(weight => assert.ok(weight > 0 && weight < 1));
    });
  }
});

test('position removal or uniform attention erases the order contrast', () => {
  for (const reverse of [false, true]) {
    const original = socksAttention({ reverse });
    for (const options of [{ position: false }, { uniform: true }, { position: false, uniform: true }]) {
      const result = socksAttention({ reverse, ...options });
      matrixClose(result.Y, [[0], [0]]);
      close(result.pooled, 0);
      close(result.prediction, -.5);
      matrixClose(result.Q, original.Q);
      matrixClose(result.K, original.K);
      matrixClose(result.S, original.S);
      if (options.position === false) matrixClose(result.V, [[0], [0]]);
      if (options.uniform) matrixClose(result.A, [[.5, .5], [.5, .5]]);
      else matrixClose(result.A, original.A);
    }
  }
});

test('fixed identity-only queries and keys preserve source preference when positions swap', () => {
  assert.deepEqual(SOCKS_WQ, [[0], [1], [0]]);
  matrixClose(SOCKS_WK, [[logThree], [0], [0]]);
  assert.deepEqual(SOCKS_WV, [[0], [0], [1]]);
  const forward = socksAttention();
  const reverse = socksAttention({ reverse: true });
  for (const model of [forward, reverse]) {
    model.tokens.forEach((token, i) => {
      close(model.Q[i][0], token === 'shoes' ? 1 : 0);
      close(model.K[i][0], token === 'socks' ? logThree : 0);
      close(model.V[i][0], i === 0 ? 1 : -1);
    });
    close(model.A[model.tokens.indexOf('shoes')][model.tokens.indexOf('socks')], .75);
    close(model.A[model.tokens.indexOf('socks')][model.tokens.indexOf('shoes')], .5);
  }
  for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
    close(reverse.A[1 - i][1 - j], forward.A[i][j]);
  }
});

test('left mixing carries position into the pooled readout, with zero original position mean', () => {
  for (const reverse of [false, true]) {
    const { X, A, AX, V, Y, pooled, prediction } = socksAttention({ reverse });
    for (let i = 0; i < 2; i++) {
      close(Y[i][0], A[i][0] * V[0][0] + A[i][1] * V[1][0]);
      close(AX[i][2], Y[i][0]);
    }
    close((X[0][2] + X[1][2]) / 2, 0);
    close(pooled, (Y[0][0] + Y[1][0]) / 2);
    close(prediction, Y[0][0] + Y[1][0] - .5);
  }
});
