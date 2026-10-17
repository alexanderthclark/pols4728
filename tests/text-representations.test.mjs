import assert from 'node:assert/strict';
import test from 'node:test';
import { documents, tokenize, vocabulary, counts, termStats, tfidf, cosine, cbowExample, cbowStep } from '../docs/text/model.mjs';

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

test('CBOW context is the actual symmetric window around senate', () => {
  const { sentenceTokens, targetIndex, windowRadius, targetWord, contextWords } = cbowExample;
  assert.equal(sentenceTokens.join(' '), 'the senate debated the budget');
  assert.equal(sentenceTokens[targetIndex], targetWord);
  const actualWindow = sentenceTokens.slice(targetIndex - windowRadius, targetIndex + windowRadius + 1)
    .filter((_, offset) => offset !== windowRadius);
  assert.deepEqual(contextWords, actualWindow);
  assert.deepEqual(cbowExample.averageVector, [0.5, 0.5]);
  assert.equal(cbowExample.dimensions, 2);
});

test('CBOW scores, normalized probabilities and log loss match independent arithmetic', () => {
  const expectedScores = { the: -0.3, senate: 0.3, parliament: 0.2, debated: 0, budget: 0.1, health: 0 };
  const denominator = Math.exp(-0.3) + Math.exp(0.3) + Math.exp(0.2) + 1 + Math.exp(0.1) + 1;
  for (const word of cbowExample.words) {
    close(cbowExample.scores[word], expectedScores[word]);
    close(cbowExample.probabilities[word], Math.exp(expectedScores[word]) / denominator);
  }
  close(Object.values(cbowExample.probabilities).reduce((sum, value) => sum + value, 0), 1);
  close(cbowExample.loss, Math.log(denominator) - 0.3);
});

test('a simultaneous gradient update changes the context inputs and all outputs', () => {
  const original = JSON.stringify(cbowExample);
  const { before, after, gradients } = cbowStep(0.5);
  assert.ok(after.probabilities.senate > before.probabilities.senate);
  assert.ok(after.loss < before.loss);
  close(after.loss, -Math.log(after.probabilities.senate));
  close(gradients.outputVectors.senate[0], (before.probabilities.senate - 1) / 2);
  close(gradients.outputVectors.budget[1], before.probabilities.budget / 2);
  close(after.outputVectors.senate[0], 0.2 - 0.5 * (before.probabilities.senate - 1) / 2);
  for (const word of cbowExample.words) {
    if (cbowExample.contextWords.includes(word)) {
      assert.notDeepEqual(after.inputVectors[word], before.inputVectors[word]);
    } else {
      assert.deepEqual(after.inputVectors[word], before.inputVectors[word]);
    }
    assert.notDeepEqual(after.outputVectors[word], before.outputVectors[word]);
  }
  assert.deepEqual(cbowStep(0).after, cbowStep(0).before);
  assert.equal(JSON.stringify(cbowExample), original);
  assert.deepEqual(cbowStep(), cbowStep());
  assert.throws(() => cbowStep(-1), RangeError);
  assert.throws(() => cbowStep(Infinity), RangeError);
});

test('all CBOW parameter gradients agree with independent finite differences', () => {
  // Recompute the objective here, independently of the module's evaluation helper.
  const { before, gradients } = cbowStep(0);
  function objective(inputs, outputs) {
    const h = [0, 1].map(k => (inputs.the[k] + inputs.debated[k]) / 2);
    const scores = cbowExample.words.map(word => outputs[word][0] * h[0] + outputs[word][1] * h[1]);
    return Math.log(scores.reduce((sum, score) => sum + Math.exp(score), 0)) - scores[1];
  }
  const epsilon = 1e-6;
  for (const tableName of ['inputVectors', 'outputVectors']) {
    for (const word of cbowExample.words) {
      for (const coordinate of [0, 1]) {
        const plus = structuredClone(before);
        const minus = structuredClone(before);
        plus[tableName][word][coordinate] += epsilon;
        minus[tableName][word][coordinate] -= epsilon;
        const numerical = (objective(plus.inputVectors, plus.outputVectors) - objective(minus.inputVectors, minus.outputVectors)) / (2 * epsilon);
        close(gradients[tableName][word][coordinate], numerical, 3e-10);
      }
    }
  }
});
