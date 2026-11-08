import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import test from 'node:test';
import {cosine, tokenize} from '../docs/text/model.mjs';

const require = createRequire(import.meta.url);
globalThis.katex = require('../docs/word2vec/vendor/katex/katex.min.js');
const {shortDocuments, shortVocabulary, shortCounts, shortComparisons, shortSentencesVisual} = await import('../docs/text/short-sentences.mjs');
const {chapters} = await import('../docs/text/lesson.mjs');
const {bagVisual} = await import('../docs/text/visuals.mjs');

const texExpressions = markup => [...markup.matchAll(/<annotation encoding="application\/x-tex">([\s\S]*?)<\/annotation>/g)].map(match => match[1]);
const cells = markup => [...markup.matchAll(/<td\b([^>]*)>([\s\S]*?)<\/td>/g)].map(([, attributes, body]) => ({
  row: Number(attributes.match(/data-row="(\d+)"/)[1]),
  column: Number(attributes.match(/data-column="(\d+)"/)[1]),
  count: Number(attributes.match(/data-count="(\d+)"/)[1]),
  displayed: Number(body.replace(/<[^>]+>/g, '').trim()),
  body,
}));

test('the short-sentence corpus preserves the supplied wording and nine separate word forms', () => {
  assert.deepEqual(shortDocuments.map(document => document.text), [
    'Plebiscites are unwise.', 'Democracy is foolish.', 'Majority rule rules.',
  ]);
  assert.deepEqual(shortDocuments.map(document => document.tokens), shortDocuments.map(document => tokenize(document.text)));
  assert.deepEqual(shortVocabulary, ['plebiscites', 'are', 'unwise', 'democracy', 'is', 'foolish', 'majority', 'rule', 'rules']);
  assert.deepEqual(shortCounts, [
    [1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1],
  ]);
  for (const [document, row] of shortCounts.entries()) {
    assert.equal(row.length, shortVocabulary.length);
    assert.equal(row.reduce((sum, value) => sum + value, 0), 3);
    for (const [column, word] of shortVocabulary.entries()) {
      assert.equal(row[column], shortDocuments[document].tokens.filter(token => token === word).length);
    }
  }
});

test('disjoint count vectors give equal cosine distances from sentence 1 to sentences 2 and 3', () => {
  assert.deepEqual(shortComparisons.map(comparison => comparison.pair), [[0, 1], [0, 2]]);
  for (const {pair: [left, right], dot, similarity, distance} of shortComparisons) {
    const a = shortCounts[left], b = shortCounts[right];
    assert.equal(dot, a.reduce((sum, value, column) => sum + value * b[column], 0));
    assert.equal(dot, 0);
    assert.equal(similarity, cosine(a, b));
    assert.equal(similarity, 0);
    assert.equal(distance, 1 - cosine(a, b));
    assert.equal(distance, 1);
    assert.equal(a.reduce((sum, value) => sum + value * value, 0), 3);
    assert.equal(b.reduce((sum, value) => sum + value * value, 0), 3);
  }
  assert.equal(cosine(shortCounts[1], shortCounts[2]), 0);
  assert.equal(shortComparisons[0].distance, shortComparisons[1].distance);
});

test('the displayed count matrix retains every vocabulary coordinate and its actual value', () => {
  const markup = shortSentencesVisual('short-counts', 1);
  assert.match(markup, /short-matrix/);
  for (const term of shortVocabulary) assert.ok(markup.includes(`>${term}<`), `${term} must have a visible column label`);
  const displayed = cells(markup);
  assert.equal(displayed.length, 27);
  assert.equal(new Set(displayed.map(cell => `${cell.row}:${cell.column}`)).size, 27);
  for (const cell of displayed) {
    assert.equal(cell.count, shortCounts[cell.row][cell.column]);
    assert.equal(cell.displayed, shortCounts[cell.row][cell.column]);
  }
  const sentences = shortSentencesVisual('short-sentences', 1);
  for (const document of shortDocuments) assert.ok(sentences.includes(document.text));
});

test('the early comparison displays cosine similarity and the equal cosine distances as rendered mathematics', () => {
  const markup = shortSentencesVisual('short-cosine', 1);
  assert.doesNotMatch(markup, /katex-error|NaN|Infinity|undefined|\\\(|\\\)/);
  const expressions = texExpressions(markup);
  assert.ok(expressions.length > 0, 'The comparison must use the math renderer.');
  for (const expression of expressions) {
    assert.doesNotThrow(() => globalThis.katex.renderToString(expression.replaceAll('&amp;', '&').replaceAll('&lt;', '<').replaceAll('&gt;', '>'), {throwOnError: true}));
  }
  assert.ok(expressions.some(expression => /\\cos/.test(expression) && /\\frac/.test(expression)), 'The cosine similarity formula must be displayed.');
  const comparisons = [...markup.matchAll(/data-pair="([^"]+)"([\s\S]*?)(?=<div class="short-comparison"|<div class="short-distance")/g)];
  assert.deepEqual(comparisons.map(([, pair]) => pair), ['1-2', '1-3']);
  for (const [, pair, comparison] of comparisons) {
    assert.ok(texExpressions(comparison).includes(String.raw`\frac{0}{\sqrt{3}\sqrt{3}}=0`), `Sentences ${pair} must show zero dot product divided by both vector lengths, yielding similarity zero.`);
  }
  assert.ok(expressions.includes(String.raw`d_{\cos}(\mathbf{x}_1,\mathbf{x}_2)=d_{\cos}(\mathbf{x}_1,\mathbf{x}_3)=1`), 'The two cosine distances must be explicitly equal to one.');
});

test('count assembly reverses exactly and reduced motion shows the complete early comparison', () => {
  const positions = Array.from({length: 21}, (_, index) => index / 20);
  const forward = positions.map(progress => shortSentencesVisual('short-counts', progress));
  const backward = [...positions].reverse().map(progress => shortSentencesVisual('short-counts', progress)).reverse();
  assert.deepEqual(backward, forward);
  const start = cells(forward[0]), end = cells(forward.at(-1));
  assert.notDeepEqual(start.filter(cell => cell.count).map(cell => cell.body), end.filter(cell => cell.count).map(cell => cell.body), 'The occupied count cells must move into their vocabulary coordinates.');
  assert.deepEqual(start.map(({row, column, count}) => ({row, column, count})), end.map(({row, column, count}) => ({row, column, count})), 'Animation must preserve the underlying counts.');
  const bag = chapters.find(chapter => chapter.id === 'bag-of-words');
  for (const scene of ['short-sentences', 'short-counts', 'short-cosine']) {
    const stage = bag.stages.findIndex(item => item.visual === scene);
    assert.ok(stage >= 0);
    for (const delta of [-.4, 0, .4]) {
      assert.equal(bagVisual(stage, stage + delta, false), shortSentencesVisual(scene, 1));
    }
  }
});
