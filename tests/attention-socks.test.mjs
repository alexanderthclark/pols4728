import assert from 'node:assert/strict';
import test from 'node:test';
import { socksAttention, SOCKS_WQ, SOCKS_WK, SOCKS_WV, SOCKS_WO } from '../docs/attention/socks-model.mjs';

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
    matrixClose(result.maskedScores, result.S);
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

test('the output map writes the one-feature mixture into the position coordinate', () => {
  assert.deepEqual(SOCKS_WO, [[0, 0, 1]]);
  const expected = [
    { reverse: false, deltaX: [[0, 0, 0], [0, 0, .5]], updatedX: [[1, 0, 1], [0, 1, -.5]], prediction: 0 },
    { reverse: true, deltaX: [[0, 0, -.5], [0, 0, 0]], updatedX: [[0, 1, .5], [1, 0, -1]], prediction: -1 },
  ];
  for (const example of expected) {
    const result = socksAttention({ reverse: example.reverse });
    matrixClose(result.deltaX, example.deltaX);
    matrixClose(result.updatedX, example.updatedX);
    result.X.forEach((row, i) => {
      assert.equal(result.Y[i].length, 1);
      assert.equal(result.deltaX[i].length, 3);
      assert.equal(result.updatedX[i].length, 3);
      close(result.deltaX[i][2], result.Y[i][0]);
      close(result.updatedX[i][0], row[0]);
      close(result.updatedX[i][1], row[1]);
      close(result.updatedX[i][2] - row[2], result.Y[i][0]);
    });
    close(result.prediction, example.prediction);
    close(result.prediction, result.Y[0][0] + result.Y[1][0] - .5);
  }
});

test('causal attention masks future scores before softmax while preserving self and past sources', () => {
  const expected = [
    { reverse: false, S: [[0, 0], [logThree, 0]], A: [[1, 0], [.75, .25]],
      Y: [[1], [.5]], deltaX: [[0, 0, 1], [0, 0, .5]], updatedX: [[1, 0, 2], [0, 1, -.5]], prediction: 1 },
    { reverse: true, S: [[0, logThree], [0, 0]], A: [[1, 0], [.5, .5]],
      Y: [[1], [0]], deltaX: [[0, 0, 1], [0, 0, 0]], updatedX: [[0, 1, 2], [1, 0, -1]], prediction: .5 },
  ];
  for (const example of expected) {
    const result = socksAttention({ reverse: example.reverse, causal: true });
    matrixClose(result.S, example.S);
    assert.equal(result.maskedScores[0][1], -Infinity);
    close(result.maskedScores[0][0], example.S[0][0]);
    result.maskedScores[1].forEach((score, j) => close(score, example.S[1][j]));
    for (const name of ['A', 'Y', 'deltaX', 'updatedX']) matrixClose(result[name], example[name]);
    result.A.forEach((row, i) => {
      close(row[0] + row[1], 1);
      row.forEach((weight, j) => j <= i ? assert.ok(weight > 0) : assert.equal(weight, 0));
    });
    close(result.prediction, example.prediction);
  }
});

test('uniform causal intervention averages only allowed positions', () => {
  for (const reverse of [false, true]) {
    const result = socksAttention({ reverse, uniform: true, causal: true });
    matrixClose(result.A, [[1, 0], [.5, .5]]);
    matrixClose(result.Y, [[1], [0]]);
    close(result.prediction, .5);
    const noPosition = socksAttention({ reverse, uniform: true, causal: true, position: false });
    matrixClose(noPosition.Y, [[0], [0]]);
    matrixClose(noPosition.deltaX, [[0, 0, 0], [0, 0, 0]]);
    matrixClose(noPosition.updatedX, noPosition.X);
    close(noPosition.prediction, -.5);
  }
});
