import { matmul, transpose, softmax } from './model.mjs';

// A constructed one-head model, not a fitted model. Each token is a row with
// [socks indicator, shoes indicator, slot position]. Identity and position
// occupy separate coordinates so Q and K can ignore position while V reads it.
export const SOCKS_WQ = [[0], [1], [0]];
export const SOCKS_WK = [[Math.log(3)], [0], [0]];
export const SOCKS_WV = [[0], [0], [1]];

export function socksAttention({ reverse = false, position = true, uniform = false } = {}) {
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
  // uniform is an explicit intervention, not the softmax of these scores.
  const A = uniform ? [[.5, .5], [.5, .5]] : S.map(softmax);
  const Y = matmul(A, V);
  const AX = matmul(A, X);
  const pooled = (Y[0][0] + Y[1][0]) / 2;
  const prediction = 2 * pooled - .5;
  // The readout gives 0 for socks→shoes and -1 for shoes→socks. No residual
  // is needed for this construction: the original position mean is zero.
  return { tokens, X, Q, K, S, A, V, Y, AX, pooled, prediction };
}
