import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
globalThis.katex = require('../docs/word2vec/vendor/katex/katex.min.js');
const {wordAnalogy} = await import('../docs/text/analogy.mjs');

test('both analogy examples render valid LaTeX and MathML throughout the scroll sequence', () => {
  for (const example of ['royalty', 'capitals']) {
    for (const reducedMotion of [false, true]) {
      for (const stage of [0, 1]) {
        for (const progress of [0, 0.25, 0.5, 0.75, 1]) {
          const markup = wordAnalogy(progress, {example, reducedMotion, stage});
          assert.equal((markup.match(/class="katex"/g) ?? []).length, 2);
          assert.equal((markup.match(/<math\b/g) ?? []).length, 2);
          assert.equal((markup.match(/encoding="application\/x-tex"/g) ?? []).length, 2);
          assert.doesNotMatch(markup, /katex-error|NaN|Infinity/);
          assert.ok(markup.includes(example === 'royalty'
            ? String.raw`v_{\mathrm{king}} - v_{\mathrm{man}} + v_{\mathrm{woman}} \approx v_{\mathrm{queen}}`
            : String.raw`v_{\mathrm{Paris}} - v_{\mathrm{France}} + v_{\mathrm{Germany}} \approx v_{\mathrm{Berlin}}`));
        }
      }
    }
  }
});

test('scrolling translates the offset to the second word and retraces the same geometry', () => {
  const positions = Array.from({length: 21}, (_, index) => index / 20);
  for (const example of ['royalty', 'capitals']) {
    const forward = positions.map(progress => wordAnalogy(progress, {example}));
    const reverse = [...positions].reverse().map(progress => wordAnalogy(progress, {example})).reverse();
    assert.deepEqual(forward, reverse);
    const translations = forward.map(markup => Number(markup.match(/class="analogy-moving-arrow" transform="translate\(([\d.]+) 0\)"/)[1]));
    assert.equal(translations[0], 0);
    assert.equal(110 + translations.at(-1), 390);
    assert.ok(translations.every((value, index) => index === 0 || value > translations[index - 1]));
    assert.match(forward[0], /class="analogy-arithmetic" style="opacity:0\.000" aria-hidden="true"/);
    assert.match(forward.at(-1), /class="analogy-arithmetic" style="opacity:1\.000" aria-hidden="false"/);
  }
});

test('reduced motion holds complete, stable analogy states', () => {
  for (const example of ['royalty', 'capitals']) {
    for (const stage of [0, 1]) {
      const options = {example, reducedMotion: true, stage};
      assert.equal(wordAnalogy(0.1, options), wordAnalogy(0.9, options));
      const markup = wordAnalogy(0.5, options);
      assert.ok(markup.includes(`data-translation="${stage === 0 ? '0.000' : '280.000'}"`));
    }
  }
});
