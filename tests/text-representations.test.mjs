import assert from 'node:assert/strict';
import test from 'node:test';
import { documents, tokenize, vocabulary, counts, termStats, tfidf, cosine, displayTerms } from '../docs/text/model.mjs';

function close(actual, expected, tolerance = 1e-12) {
  assert.ok(Math.abs(actual - expected) < tolerance, `${actual} ≈ ${expected}`);
}

test('the tokenizer case-folds and removes punctuation while preserving repetitions', () => {
  assert.deepEqual(tokenize('The SENATE, the budget! 2026'), ['the', 'senate', 'the', 'budget']);
  assert.deepEqual(tokenize('... 123'), []);
  assert.deepEqual(tokenize(''), []);
});

test('the slide quotations retain exact wording and their preprocessed vocabulary coordinates', () => {
  assert.deepEqual(documents.map(d => [d.label, d.text]), [
    ['Schumpeter', 'Thus the typical citizen drops down to a lower level of mental performance as soon as he enters the political field.'],
    ['Whitman', 'Political Democracy, as it exists and practically works in America, with all its threatening evils, supplies a training school for making grand young men.'],
  ]);
  const expectedTokens = [
    'typical citizen drop down lower level mental performance enter political field'.split(' '),
    'political democracy exist practically work america threaten evil supply training school make grand young man'.split(' '),
  ];
  assert.deepEqual(documents.map(d => d.tokens), expectedTokens);
  assert.deepEqual(documents.map(d => tokenize(d.text).length), [21, 24]);
  assert.deepEqual(vocabulary, ['america', 'citizen', 'democracy', 'down', 'drop', 'enter', 'evil', 'exist', 'field', 'grand', 'level', 'lower', 'make', 'man', 'mental', 'performance', 'political', 'practically', 'school', 'supply', 'threaten', 'training', 'typical', 'work', 'young']);
  assert.deepEqual(displayTerms, ['political', 'citizen', 'mental', 'democracy', 'america', 'young']);
  assert.deepEqual(counts, [
    [0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1],
  ]);
  assert.deepEqual(counts.map(row => row.reduce((sum, x) => sum + x, 0)), [11, 15]);
});

test('raw-count TF × ln(N/df) zeros the shared term and preserves distinctive terms', () => {
  assert.deepEqual(termStats('citizen', 0), { tf: 1, df: 1, N: 2, idf: Math.log(2), weight: Math.log(2) });
  assert.deepEqual(termStats('citizen', 1), { tf: 0, df: 1, N: 2, idf: Math.log(2), weight: 0 });
  close(termStats('america', 1).idf, Math.log(2));
  close(termStats('america', 1).weight, Math.log(2));
  for (const doc of [0, 1]) {
    assert.deepEqual(termStats('political', doc), {tf: 1, df: 2, N: 2, idf: 0, weight: 0});
  }
  assert.throws(() => termStats('unknown', 0), RangeError);
  assert.throws(() => termStats('political', -1), RangeError);
  assert.throws(() => termStats('political', 2), RangeError);
});

test('the full TF-IDF matrix matches a hand-computed reference without smoothing', () => {
  const l2 = Math.log(2);
  const distinctiveTerms = [
    'typical citizen drop down lower level mental performance enter field'.split(' '),
    'democracy exist practically work america threaten evil supply training school make grand young man'.split(' '),
  ];
  tfidf.forEach((row, doc) => row.forEach((value, column) => close(value, distinctiveTerms[doc].includes(vocabulary[column]) ? l2 : 0)));
  assert.deepEqual(tfidf.map(row => row.filter(value => value > 0).length), [10, 14]);
});

test('cosine compares direction and explicitly handles zero vectors', () => {
  close(cosine([1, 2], [2, 4]), 1);
  close(cosine([1, 0], [0, 1]), 0);
  close(cosine([1, 0], [-1, 0]), -1);
  assert.equal(cosine([0, 0], [2, 1]), 0);
  assert.throws(() => cosine([1], [1, 2]), RangeError);
  // The only shared processed term, political, receives zero TF–IDF weight.
  close(cosine(tfidf[0], tfidf[1]), 0);
  close(cosine(counts[0], counts[1]), 1 / Math.sqrt(11 * 15));
});
