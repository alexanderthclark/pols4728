// Every token is a ROW. These small, invented coordinates illustrate the
// calculation; they are not embeddings fitted to this sentence.
export const tokens = ['The', 'senator', 'thanked', 'staff'];
export const X = [[0, 0, 1], [2, 0, 1], [0, 1, 1], [0, 2, 1]];
export const WQ = [[0, 1], [1, 2], [0, 0]];
export const WK = [[1, 0], [0, 1], [0, 0]];
export const WV = [[1, 0], [0, 1], [.5, -.5]];
export const WO = [[1, 0, 0], [0, 1, 0]];
export const dk = 2;

// The first mixing example introduces A before deriving it from Q and K.
// These hand-chosen weights are deliberately separate from attention().A.
export const handWeights = [.1, .6, .1, .2];
export const uniformWeights = [.25, .25, .25, .25];

export function transpose(matrix) {
  return matrix[0].map((_, column) => matrix.map(row => row[column]));
}

export function matmul(left, right) {
  if (left[0].length !== right.length) throw new RangeError('Matrix dimensions do not agree.');
  return left.map(row => right[0].map((_, column) =>
    row.reduce((sum, value, inner) => sum + value * right[inner][column], 0)));
}

export function softmax(scores) {
  const maximum = Math.max(...scores);
  if (!Number.isFinite(maximum)) throw new RangeError('A softmax row must have at least one finite score.');
  const exponentials = scores.map(score => Math.exp(score - maximum));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map(value => value / total);
}

export function mixRows(weights, input = X) {
  return matmul([weights], input)[0];
}

// scale multiplies the usual scaled-dot-product scores. The lesson's standard
// calculation uses scale=1; it is not an additional trained parameter.
export function attention({ causal = false, scale = 1, input = X } = {}) {
  const Q = matmul(input, WQ);
  const K = matmul(input, WK);
  const V = matmul(input, WV);
  const rawScores = matmul(Q, transpose(K));
  const S = rawScores.map(row => row.map(value => scale * value / Math.sqrt(dk)));
  // Destination i may read source j only when j <= i. Set forbidden scores
  // to -Infinity BEFORE softmax, so the remaining weights normalize to one.
  const maskedScores = S.map((row, i) => row.map((value, j) => causal && j > i ? -Infinity : value));
  const A = maskedScores.map(softmax);
  const AX = matmul(A, input);
  const Y = matmul(A, V);
  const head = matmul(Y, WO);
  return { X: input, Q, K, rawScores, S, maskedScores, A, V, Y, AX, head };
}
