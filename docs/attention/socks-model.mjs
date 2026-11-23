import { matmul, transpose, softmax } from './model.mjs';

// A constructed one-head model, not a fitted model. Each token is a row with
// [socks indicator, shoes indicator, slot position]. Identity and position
// occupy separate coordinates so Q and K can ignore position while V reads it.
export const SOCKS_WQ = [[0], [1], [0]];
export const SOCKS_WK = [[Math.log(3)], [0], [0]];
export const SOCKS_WV = [[0], [0], [1]];
export const SOCKS_WO = [[0, 0, 1]];

// Generalize the unmasked example: a and b are the weights on the socks source
// for the socks and shoes destinations. The rule follows token identity, so
// reversing the sequence permutes BOTH axes of A, while slot values stay fixed.
export function socksPattern(a, b, { reverse = false } = {}) {
  if (![a, b].every(weight => Number.isFinite(weight) && weight >= 0 && weight <= 1)) {
    throw new RangeError('Attention weights a and b must be finite numbers in [0, 1].');
  }
  const A = reverse ? [[1 - b, b], [1 - a, a]] : [[a, 1 - a], [b, 1 - b]];
  const U = matmul(A, [[1], [-1]]);
  // Use the analytic column-sum difference, avoiding multiplication roundoff
  // masquerading as an order signal for balanced examples such as (.8, .2).
  // Only an exactly zero difference collapses; no near-zero cutoff is imposed.
  const contrast = a + b - 1;
  const pooled = reverse ? -contrast : contrast;
  const coefficient = contrast === 0 ? null : 1 / (2 * contrast);
  return {
    A, U, pooled, coefficient,
    prediction: coefficient === null ? null : coefficient * pooled - .5,
    originalPrediction: 2 * pooled - .5,
    columnSums: [A[0][0] + A[1][0], A[0][1] + A[1][1]],
  };
}

export function socksAttention({ reverse = false, position = true, uniform = false, causal = false } = {}) {
  const tokens = reverse ? ['shoes', 'socks'] : ['socks', 'shoes'];
  const X = tokens.map((token, i) => [
    Number(token === 'socks'),
    Number(token === 'shoes'),
    position ? (i === 0 ? 1 : -1) : 0,
  ]);
  const Q = matmul(X, SOCKS_WQ);
  const K = matmul(X, SOCKS_WK);
  const V = matmul(X, SOCKS_WV);
  // d_k=1, so scaled dot products equal the dot products themselves.
  const S = matmul(Q, transpose(K));
  const maskedScores = S.map((row, i) => row.map((score, j) => causal && j > i ? -Infinity : score));
  // uniform is an explicit intervention, not the softmax of these scores.
  // With a causal mask it distributes weight uniformly over allowed sources.
  const A = uniform
    ? S.map((row, i) => row.map((_, j) => causal ? (j <= i ? 1 / (i + 1) : 0) : 1 / row.length))
    : maskedScores.map(softmax);
  const U = matmul(A, V);
  const AX = matmul(A, X);
  // A separate bridge to an attention contribution in the input feature space.
  // The worked task's readout below still uses U, not updatedX.
  const deltaX = matmul(U, SOCKS_WO);
  const updatedX = X.map((row, i) => row.map((value, j) => value + deltaX[i][j]));
  const pooled = (U[0][0] + U[1][0]) / 2;
  const prediction = 2 * pooled - .5;
  // In the default bidirectional construction, the readout gives 0 for
  // socks→shoes and -1 for shoes→socks. Causal masking changes those outputs.
  // No residual is needed for the readout: the original position mean is zero.
  return { tokens, X, Q, K, S, maskedScores, A, V, U, AX, deltaX, updatedX, pooled, prediction };
}
