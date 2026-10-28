import assert from 'node:assert/strict';
import test from 'node:test';
import { documents, tokenize, vocabulary, counts, termStats, tfidf, cosine } from '../docs/text/model.mjs';

function close(actual, expected, tolerance = 1e-12) {
  assert.ok(Math.abs(actual - expected) < tolerance, `${actual} ≈ ${expected}`);
}

test('the tokenizer case-folds and removes punctuation while preserving repetitions', () => {
  assert.deepEqual(tokenize('The SENATE, the budget! 2026'), ['the', 'senate', 'the', 'budget']);
  assert.deepEqual(tokenize('... 123'), []);
  assert.deepEqual(tokenize(''), []);
});

test('four synthetic documents have fixed vocabulary coordinates and exact count vectors', () => {
  assert.equal(documents.length, 4);
  assert.deepEqual(vocabulary, ['budget', 'debated', 'health', 'parliament', 'passed', 'senate', 'the']);
  assert.deepEqual(counts, [
    [2, 1, 0, 1, 1, 0, 3],
    [1, 1, 0, 0, 0, 1, 2],
    [0, 1, 1, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 1],
  ]);
  assert.deepEqual(counts.map(row => row.reduce((sum, x) => sum + x, 0)), [8, 5, 4, 4]);
});

test('raw-count TF × ln(N/df) distinguishes document frequency from corpus count', () => {
  assert.deepEqual(termStats('budget', 0), { tf: 2, df: 2, N: 4, idf: Math.log(2), weight: 2 * Math.log(2) });
  assert.equal(termStats('budget', 2).tf, 0);
  assert.equal(termStats('budget', 2).weight, 0);
  close(termStats('passed', 0).idf, Math.log(4));
  close(termStats('passed', 0).weight, Math.log(4));
  for (const term of ['the', 'debated']) {
    assert.equal(termStats(term, 0).df, 4);
    assert.equal(termStats(term, 0).idf, 0);
    assert.equal(termStats(term, 0).weight, 0);
  }
  assert.throws(() => termStats('unknown', 0), RangeError);
  assert.throws(() => termStats('the', -1), RangeError);
  assert.throws(() => termStats('the', 4), RangeError);
});

test('the full TF-IDF matrix matches a hand-computed reference without smoothing', () => {
  const l2 = Math.log(2);
  const expected = [
    [2 * l2, 0, 0, l2, 2 * l2, 0, 0],
    [l2, 0, 0, 0, 0, l2, 0],
    [0, 0, l2, l2, 0, 0, 0],
    [0, 0, l2, 0, 0, l2, 0],
  ];
  tfidf.forEach((row, d) => row.forEach((value, t) => close(value, expected[d][t])));
});

test('cosine compares direction and explicitly handles zero vectors', () => {
  close(cosine([1, 2], [2, 4]), 1);
  close(cosine([1, 0], [0, 1]), 0);
  close(cosine([1, 0], [-1, 0]), -1);
  assert.equal(cosine([0, 0], [2, 1]), 0);
  assert.throws(() => cosine([1], [1, 2]), RangeError);
  // Parliament/health and Senate/health share only the health coordinate.
  close(cosine(tfidf[2], tfidf[3]), 0.5);
});
