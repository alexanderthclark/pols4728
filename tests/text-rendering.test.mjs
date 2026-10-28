import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
globalThis.katex = require('../docs/word2vec/vendor/katex/katex.min.js');
const {bagVisual, tfVisual, countAssembly, inspectTerm, tokenizationVisual, lemmatizationVisual, stopWordsVisual} = await import('../docs/text/visuals.mjs');
const {aiExample} = await import('../docs/text/ai-example.mjs');
const {prose} = await import('../docs/text/math.mjs');
const {chapters} = await import('../docs/text/lesson.mjs');
const {documents, vocabulary, counts, tfidf, tokenize} = await import('../docs/text/model.mjs');

const rounded = (value, places = 3) => Number(value.toFixed(places));
const attributes = tag => [...tag.matchAll(/([\w:-]+)="([^"]*)"/g)].map(([, name, value]) => [name, value]);
const texExpressions = markup => [...markup.matchAll(/<annotation encoding="application\/x-tex">([\s\S]*?)<\/annotation>/g)].map(match => match[1]);
const tableRows = markup => [...markup.match(/<tbody>([\s\S]*?)<\/tbody>/)[1].matchAll(/<tr\b([^>]*)>([\s\S]*?)<\/tr>/g)].map(([, attrs, body]) => ({
  active: /class="active"/.test(attrs),
  cells: [...body.matchAll(/<td class="([^"]*)">([\s\S]*?)<\/td>/g)].map(([, classes, value]) => ({classes, value: Number(value.replace(/<[^>]+>/g, ''))})),
}));

function checkMarkup(markup, label) {
  assert.doesNotMatch(markup, /katex-error|NaN|Infinity|undefined/, label);
  for (const [tag] of markup.matchAll(/<(?:svg|g|text|line|rect|circle|path|foreignObject)\b[^>]*>/g)) {
    const attrs = attributes(tag);
    assert.equal(new Set(attrs.map(([name]) => name)).size, attrs.length, `${label}: duplicate attributes in ${tag}`);
    for (const [name, value] of attrs) {
      if (/^(?:x|y|x1|x2|y1|y2|width|height|r|rx|ry|opacity|font-size|stroke-width)$/.test(name)) {
        assert.ok(Number.isFinite(Number.parseFloat(value)), `${label}: invalid ${name}=${value}`);
      }
    }
  }
  for (const tex of texExpressions(markup)) {
    assert.doesNotThrow(() => globalThis.katex.renderToString(tex.replaceAll('&amp;', '&').replaceAll('&lt;', '<').replaceAll('&gt;', '>'), {throwOnError: true}), `${label}: ${tex}`);
  }
}

test('every diagram and explanation renders valid mathematics and finite geometry', () => {
  const renderers = {'bag-of-words': bagVisual, 'tf-idf': tfVisual};
  for (const chapter of chapters) {
    for (const [stage, scene] of chapter.stages.entries()) {
      for (const motion of [true, false]) {
        for (const delta of [-0.4, 0, 0.4]) {
          const progress = Math.max(0, Math.min(chapter.stages.length - 1, stage + delta));
          const markup = chapter.id === 'ai-writing'
            ? aiExample(stage, progress, {reducedMotion: !motion})
            : renderers[chapter.id](stage, progress, motion);
          checkMarkup(markup, `${chapter.id} stage ${stage}, progress ${progress}, motion ${motion}`);
        }
      }
      for (const key of ['text', 'note', 'readingText', 'readingNote']) {
        const markup = prose(scene[key], scene.links);
        checkMarkup(markup, `${chapter.id} stage ${stage} ${key}`);
        assert.doesNotMatch(markup, /\\\(|\\\)/, 'Inline mathematics must be rendered rather than left as delimiters.');
      }
    }
  }
});

test('the TF–IDF narrative fixes the featured terms and document to match its explanation', () => {
  for (const [stage, term] of [[0, 'the'], [1, 'budget'], [3, 'budget']]) {
    const markup = tfVisual(stage, stage, true);
    const rows = tableRows(markup);
    assert.deepEqual(rows.map(row => row.active), [true, false, false, false]);
    for (const row of rows) {
      assert.deepEqual(row.cells.map((cell, column) => cell.classes.includes('focus') ? vocabulary[column] : null).filter(Boolean), [term]);
    }
    assert.doesNotMatch(markup, /<select\b|<input\b/);
  }
  const presence = tableRows(tfVisual(1, 1));
  assert.deepEqual(presence.map(row => row.cells[vocabulary.indexOf('budget')].classes.includes('present')), [true, true, false, false]);
  assert.ok(texExpressions(tfVisual(0, 0)).some(tex => tex.includes(String.raw`\operatorname{tf}(\mathrm{the},D_1)=3`)));
  assert.ok(texExpressions(tfVisual(1, 1)).some(tex => tex.includes(String.raw`\operatorname{df}(\mathrm{budget})=2`)));
  assert.doesNotMatch(tfVisual(4, 4), /<select\b|<input\b/);
});

test('count assembly preserves tokens, reverses exactly, and finishes at the full count matrix', () => {
  const positions = Array.from({length: 41}, (_, index) => 1 + index / 40);
  const forward = positions.map(countAssembly);
  const reverse = [...positions].reverse().map(countAssembly).reverse();
  assert.deepEqual(reverse, forward);
  const tokenPositions = markup => [...markup.matchAll(/<text\b([^>]*data-token="\d+"[^>]*)>([^<]+)<\/text>/g)].map(([, tag, word]) => ({...Object.fromEntries(attributes(tag)), word}));
  for (const markup of forward) assert.deepEqual(tokenPositions(markup).map(token => token.word), tokenize(documents[0].text));
  assert.match(forward[0], /data-matrix-reveal="0"/);
  assert.match(forward.at(-1), /data-matrix-reveal="1"/);
  assert.notDeepEqual(tokenPositions(forward[0]).map(token => [token.x, token.y]), tokenPositions(forward[20]).map(token => [token.x, token.y]));
  assert.ok(tokenPositions(forward.at(-1)).every(token => Number(token.opacity) === 0));
  const matrixGroup = forward.at(-1).match(/<g opacity="1" data-matrix-reveal="1">([\s\S]*?)<\/g>/)[1];
  const labels = [...matrixGroup.matchAll(/<text\b[^>]*>([^<]+)<\/text>/g)].map(match => match[1]);
  assert.deepEqual(labels.slice(0, vocabulary.length), vocabulary);
  for (const [row, expected] of counts.entries()) {
    const start = vocabulary.length + row * (vocabulary.length + 1);
    assert.equal(labels[start], `D${row + 1}`);
    assert.deepEqual(labels.slice(start + 1, start + 1 + vocabulary.length).map(Number), expected);
  }
  assert.equal(bagVisual(6, 5.8, false), countAssembly(2));
});

test('preprocessing examples preserve occurrences and distinguish optional choices from the worked matrix', () => {
  const tokenWords=markup=>[...markup.matchAll(/<text\b[^>]*data-token="\d+"[^>]*>([^<]+)<\/text>/g)].map(m=>m[1]);
  assert.deepEqual(tokenWords(tokenizationVisual(0)),documents[0].text.split(' '));
  assert.deepEqual(tokenWords(tokenizationVisual(1)),tokenize(documents[0].text));
  assert.deepEqual([...lemmatizationVisual(1).matchAll(/data-lemma="([^"]+)"/g)].map(m=>m[1]),['debate','debate','senator']);
  const final=stopWordsVisual(1);
  const retained=[...final.matchAll(/<g data-token="\d+" data-kept="true" opacity="1"><text\b[^>]*>([^<]+)<\/text>/g)].map(m=>m[1]);
  assert.deepEqual(retained,['parliament','debated','budget','budget','passed']);
  assert.equal((final.match(/data-kept="false" opacity="0"/g)??[]).length,3);
  // These illustrations must not silently change the following numerical example.
  assert.equal(counts[0][vocabulary.indexOf('the')],3);
  assert.ok(vocabulary.includes('debated'));
  assert.ok(!vocabulary.includes('debate'));
  for(const render of [tokenizationVisual,lemmatizationVisual,stopWordsVisual]){
    const positions=[0,.25,.5,.75,1],forward=positions.map(render);
    assert.deepEqual([...positions].reverse().map(render).reverse(),forward);
    assert.notEqual(forward[0],forward.at(-1));
    forward.forEach(markup=>checkMarkup(markup,'preprocessing'));
  }
});

test('TF–IDF reweighting retains exact starting counts and the model’s rounded endpoint values', () => {
  const values = markup => tableRows(markup).map(row => row.cells.map(cell => cell.value));
  assert.deepEqual(values(tfVisual(3, 2.5)), counts);
  assert.deepEqual(values(tfVisual(3, 3)), tfidf.map(row => row.map(value => rounded(value, 2))));
  assert.deepEqual(values(tfVisual(3, 2.6, false)), values(tfVisual(3, 3)));
  const mid = values(tfVisual(3, 2.75));
  assert.deepEqual(mid, counts.map((row, i) => row.map((value, j) => rounded((value + tfidf[i][j]) / 2, 2))));
});

test('every explorer term and document displays the matching counts and TF–IDF calculation', () => {
  for (const [column, term] of vocabulary.entries()) {
    for (let doc = 0; doc < documents.length; doc += 1) {
      const markup = inspectTerm(term, doc);
      const rows = tableRows(markup);
      assert.deepEqual(rows.map(row => row.cells.map(cell => cell.value)), counts);
      assert.deepEqual(rows.map(row => row.active), documents.map((_, index) => index === doc));
      assert.ok(rows.every(row => row.cells[column].classes.includes('focus')));
      const tf = counts[doc][column];
      const df = counts.filter(row => row[column] > 0).length;
      const expressions = texExpressions(markup);
      assert.ok(expressions.includes(String.raw`\operatorname{tf}=${tf},\qquad\operatorname{df}=${df},\qquad N=${documents.length}`));
      const weight = expressions[1].match(/=([\d.]+)$/)[1];
      assert.equal(Number(weight), rounded(tf * Math.log(documents.length / df)));
      checkMarkup(markup, `${term} in D${doc + 1}`);
    }
  }
});
