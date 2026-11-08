import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import test from 'node:test';
import {documents, tokenize, lemmaForms, vocabulary, counts, tfidf, displayTerms} from '../docs/text/model.mjs';

const require = createRequire(import.meta.url);
globalThis.katex = require('../docs/word2vec/vendor/katex/katex.min.js');
const {quoteFlowVisual} = await import('../docs/text/quote-flow.mjs');
const phases = ['corpus', 'tokens', 'lemma', 'stopwords', 'matrix', 'tf', 'df', 'idf', 'word-document', 'weighted', 'cosine'];
const attrs = tag => Object.fromEntries([...tag.matchAll(/([\w-]+)="([^"]*)"/g)].map(([, key, value]) => [key, value]));
const tokenTexts = markup => [...markup.matchAll(/<text\b([^>]*data-key="text-(\d)-(\d+)"[^>]*)>([^<]+)<\/text>/g)].map(([, attributes, row, index, word]) => ({...attrs(attributes), row: Number(row), index: Number(index), word}));
const cells = markup => [...markup.matchAll(/<g\b([^>]*data-key="cell-\d+-\d+"[^>]*)>/g)].map(([, attributes]) => attrs(attributes));

test('the quotation workspace preserves both exact quotations and every occurrence through preprocessing', () => {
  for (const [row, document] of documents.entries()) {
    const words = phase => tokenTexts(quoteFlowVisual(phase, 1)).filter(token => token.row === row).map(token => token.word);
    assert.equal(words('corpus').join(' '), document.text);
    assert.deepEqual(words('tokens'), tokenize(document.text));
    assert.deepEqual(words('lemma'), tokenize(document.text).map(word => lemmaForms[word] ?? word));
  }
  const stop = quoteFlowVisual('stopwords', 1);
  const kept = [...stop.matchAll(/<g data-key="word-(\d+)-(\d+)"[^>]*data-kept="true" opacity="1">([\s\S]*?)<\/g>/g)];
  for (const [row, document] of documents.entries()) {
    assert.deepEqual(kept.filter(([, r]) => Number(r) === row).map(([, , , body]) => body.match(/data-key="text-\d+-\d+"[^>]*>([^<]+)</)[1]), document.tokens);
  }
  for (const phase of phases) {
    const markup = quoteFlowVisual(phase, 1);
    assert.equal(tokenTexts(markup).length, 45, `${phase} must retain both documents' token identities`);
    assert.match(markup, /data-key="author-0"[^>]*>Schumpeter</);
    assert.match(markup, /data-key="author-1"[^>]*>Whitman</);
  }
});

test('the same six matrix columns survive every TF–IDF step and interpolate to the complete model values', () => {
  const expectedKeys = cells(quoteFlowVisual('matrix', 1)).map(cell => cell['data-key']);
  for (const phase of ['matrix', 'tf', 'df', 'idf', 'word-document', 'weighted', 'cosine']) {
    const markup = quoteFlowVisual(phase, 1), displayed = cells(markup);
    assert.deepEqual(displayed.map(cell => cell['data-key']), expectedKeys);
    assert.equal(displayed.length, 12);
    for (const cell of displayed) {
      const row = Number(cell['data-doc']), column = vocabulary.indexOf(cell['data-term']);
      const expected = ['weighted', 'cosine'].includes(phase) ? tfidf[row][column] : counts[row][column];
      assert.equal(Number(cell['data-value']), expected);
    }
    for (const word of displayTerms) assert.match(markup, new RegExp(`data-key="column-${word}"`));
  }
  for (const q of [0, .25, .5, .75, 1]) {
    for (const cell of cells(quoteFlowVisual('weighted', q))) {
      const row = Number(cell['data-doc']), column = vocabulary.indexOf(cell['data-term']);
      assert.equal(Number(cell['data-value']), counts[row][column] + (tfidf[row][column] - counts[row][column]) * q);
    }
  }
  const final = quoteFlowVisual('cosine', 1);
  assert.match(final, /Cosine similarity of the full 25-term vectors is 0/);
  assert.equal(cells(final).filter(cell => cell['data-term'] === 'political').every(cell => Number(cell['data-value']) === 0), true);
});

test('preprocessing boundaries preserve token positions and the matrix does not move between weighting steps', () => {
  const positions = markup => tokenTexts(markup).map(({row, index, x, y, word}) => ({row, index, x, y, word}));
  for (const [previous, next] of [['tokens', 'lemma'], ['lemma', 'stopwords'], ['stopwords', 'matrix'], ['matrix', 'tf']]) {
    assert.deepEqual(positions(quoteFlowVisual(previous, 1)), positions(quoteFlowVisual(next, 0)), `${previous} → ${next}`);
  }
  const matrixGeometry = markup => [...markup.matchAll(/<text\b([^>]*data-key="(?:author-|column-|cell-value-)[^"]+"[^>]*)>/g)].map(([, attributes]) => {
    const a = attrs(attributes);
    return {key: a['data-key'], x: a.x, y: a.y};
  });
  const expected = matrixGeometry(quoteFlowVisual('matrix', 1));
  for (const phase of phases.slice(5)) assert.deepEqual(matrixGeometry(quoteFlowVisual(phase, .5)), expected);
  assert.notDeepEqual(positions(quoteFlowVisual('matrix', 0)), positions(quoteFlowVisual('matrix', .5)), 'Tokens must actually move into the matrix.');
});

test('all quotation phases render finite geometry, valid mathematics, and deterministic reverse states', () => {
  for (const phase of phases) {
    const progress = [0, .2, .4, .6, .8, 1];
    const forward = progress.map(q => quoteFlowVisual(phase, q));
    assert.deepEqual([...progress].reverse().map(q => quoteFlowVisual(phase, q)).reverse(), forward);
    for (const markup of forward) {
      assert.doesNotMatch(markup, /NaN|Infinity|undefined|katex-error/);
      for (const [, tex] of markup.matchAll(/<annotation encoding="application\/x-tex">([\s\S]*?)<\/annotation>/g)) {
        assert.doesNotThrow(() => globalThis.katex.renderToString(tex.replaceAll('&amp;', '&').replaceAll('&lt;', '<').replaceAll('&gt;', '>'), {throwOnError: true}));
      }
    }
  }
});
