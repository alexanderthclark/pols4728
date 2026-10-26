// Fixed, explicitly illustrative 2D data. These are not pretrained results.
export const vectors = Object.freeze(Object.fromEntries(Object.entries({
  water: [0.3, 1.2],
  dirt: [2.4, 0.6],
  rain: [0.2, 1.0],
  mud: [1.2, 0.9],
  soil: [1.0, 0.15],
  river: [0.3, 1.0],
  puddle: [0.7, 0.9],
  dust: [1.0, -0.25],
  clean: [-0.6, 0.8],
  dry: [0.7, -0.6],
}).map(([word, vector]) => [word, Object.freeze(vector)])));

export const candidates = Object.freeze([
  'mud', 'soil', 'river', 'puddle', 'dust', 'clean', 'dry', 'water', 'dirt', 'rain',
].map(word => Object.freeze({ word, vector: vectors[word] })));

function checkVector(vector) {
  if (!Array.isArray(vector) || vector.length === 0 || !vector.every(Number.isFinite)) {
    throw new TypeError('A vector must be a nonempty array of finite numbers.');
  }
}

function checkPair(a, b) {
  checkVector(a);
  checkVector(b);
  if (a.length !== b.length) throw new RangeError('Vector dimensions must agree.');
}

export function dot(a, b) {
  checkPair(a, b);
  return a.reduce((sum, value, i) => sum + value * b[i], 0);
}

export function norm(vector) {
  checkVector(vector);
  return Math.hypot(...vector);
}

export function scale(vector, factor) {
  checkVector(vector);
  if (!Number.isFinite(factor)) throw new TypeError('The scale factor must be finite.');
  return vector.map(value => value * factor);
}

export function unit(vector) {
  const length = norm(vector);
  if (length === 0) throw new RangeError('The zero vector has no direction.');
  return scale(vector, 1 / length);
}

export function add(a, b) {
  checkPair(a, b);
  return a.map((value, i) => value + b[i]);
}

export function cosine(a, b) {
  // Clamp only floating-point roundoff beyond the mathematical [-1, 1] range.
  return Math.max(-1, Math.min(1, dot(unit(a), unit(b))));
}

export function query(positive, negative = [], { normalizeSources = true } = {}) {
  const sources = [...positive, ...negative];
  if (sources.length === 0) throw new RangeError('A query needs at least one source vector.');
  checkVector(sources[0]);
  let result = sources[0].map(() => 0);
  for (const [list, sign] of [[positive, 1], [negative, -1]]) {
    for (const vector of list) {
      result = add(result, scale(normalizeSources ? unit(vector) : vector, sign));
    }
  }
  // Return the sum for drawing. Ranking normalizes the final query through
  // cosine; division by the source count would not affect its direction.
  return result;
}

function ranked(entries, excludeWords, score) {
  const excluded = new Set(excludeWords);
  return entries
    .filter(({ word }) => !excluded.has(word))
    .map(({ word, vector }) => ({ word, score: score(vector) }))
    .sort((a, b) => b.score - a.score);
}

export function rank(queryVector, entries = candidates, excludeWords = []) {
  return ranked(entries, excludeWords, vector => cosine(vector, queryVector));
}

export function cosmulScore(vector, positive, negative = []) {
  if (positive.length === 0) throw new RangeError('Multiplicative scoring needs a positive source.');
  const product = terms => terms.reduce((value, term) => value * (1 + cosine(vector, term)) / 2, 1);
  // Shift each cosine from [-1, 1] to [0, 1], then multiply candidate scores.
  // This is not coordinate-wise vector multiplication or a probability.
  return product(positive) / (product(negative) + 1e-6);
}

export function rankCosmul(positive, negative = [], entries = candidates, excludeWords = []) {
  return ranked(entries, excludeWords, vector => cosmulScore(vector, positive, negative));
}
