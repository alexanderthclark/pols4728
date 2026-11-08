import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
globalThis.katex = require('../docs/word2vec/vendor/katex/katex.min.js');
const {bagVisual, tfVisual, countAssembly, inspectTerm, tokenizationVisual, lemmatizationVisual, stopWordsVisual, wordDocumentVisual} = await import('../docs/text/visuals.mjs');
const {aiExample} = await import('../docs/text/ai-example.mjs');
const {prose} = await import('../docs/text/math.mjs');
const {chapters} = await import('../docs/text/lesson.mjs');
const {documents, vocabulary, counts, tfidf, tokenize, termStats, displayTerms} = await import('../docs/text/model.mjs');

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
  for (const [stage, term] of [[0, 'political'], [1, 'citizen'], [4, 'citizen']]) {
    const markup = tfVisual(stage, stage, true);
    const rows = tableRows(markup);
    assert.deepEqual(rows.map(row => row.active), [true, false]);
    for (const row of rows) {
      assert.deepEqual(row.cells.map((cell, column) => cell.classes.includes('focus') ? displayTerms[column] : null).filter(Boolean), [term]);
    }
    assert.doesNotMatch(markup, /<select\b|<input\b/);
  }
  const presence = tableRows(tfVisual(1, 1));
  assert.deepEqual(presence.map(row => row.cells[displayTerms.indexOf('citizen')].classes.includes('present')), [true, false]);
  assert.ok(texExpressions(tfVisual(0, 0)).some(tex => tex.includes(String.raw`c_{\mathrm{political},\mathrm{Schumpeter}}=1`)));
  assert.ok(texExpressions(tfVisual(1, 1)).some(tex => tex.includes(String.raw`\mathrm{df}_{\mathrm{citizen}}=1`)));
  const idfExpressions = texExpressions(tfVisual(2, 2)).slice(0, 3);
  for (const [index, term] of ['political', 'citizen', 'america'].entries()) {
    const {idf} = termStats(term, 0);
    const [, relation, value] = idfExpressions[index].match(/(=|\\approx)([\d.]+)$/);
    assert.equal(relation, idf ? String.raw`\approx` : '=');
    assert.equal(Number(value), rounded(idf));
  }
  assert.doesNotMatch(tfVisual(5, 5), /<select\b|<input\b/);
});

test('the word–document example gives each citizen count its own score with one shared IDF', () => {
  const markup = wordDocumentVisual();
  checkMarkup(markup, 'word–document TF–IDF example');
  const formula = texExpressions(markup)[0];
  assert.ok(formula.includes(String.raw`w_{t,d}`));
  assert.ok(formula.includes(String.raw`c_{t,d}`));
  assert.ok(formula.includes(String.raw`\frac{N}{\mathrm{df}_t}`));
  const rows = [...markup.matchAll(/<tr data-doc="(\d+)">([\s\S]*?)<\/tr>/g)];
  assert.equal(rows.length, documents.length);
  for (const [index, [, doc, row]] of rows.entries()) {
    const stats = termStats('citizen', index);
    assert.equal(Number(doc), index);
    assert.match(row, new RegExp(`<th scope="row">${documents[index].label}</th><td>${stats.tf}</td>`));
    assert.equal(Number(row.match(/data-weight="([^"]+)"/)[1]), stats.weight);
    const calculation = texExpressions(row)[0];
    assert.ok(calculation.startsWith(String.raw`${stats.tf}\ln(${stats.N}/${stats.df})`));
    assert.equal(Number(calculation.match(/([\d.]+)$/)[1]), rounded(stats.weight));
    if (stats.tf) assert.ok(calculation.includes(String.raw`\approx`), 'Rounded nonzero weights need the LaTeX approximation relation.');
  }
  assert.equal(tfVisual(3, 3), markup);
});

test('count assembly preserves the selected tokens, reverses exactly, and finishes at a labelled matrix excerpt', () => {
  const positions = Array.from({length: 41}, (_, index) => 1 + index / 40);
  const forward = positions.map(countAssembly);
  const reverse = [...positions].reverse().map(countAssembly).reverse();
  assert.deepEqual(reverse, forward);
  const tokenPositions = markup => [...markup.matchAll(/<text\b([^>]*data-token="\d+"[^>]*)>([^<]+)<\/text>/g)].map(([, tag, word]) => ({...Object.fromEntries(attributes(tag)), word}));
  for (const markup of forward) assert.deepEqual(tokenPositions(markup).map(token => token.word), displayTerms.filter(term => documents[0].tokens.includes(term)));
  assert.match(forward[0], /data-matrix-reveal="0"/);
  assert.match(forward.at(-1), /data-matrix-reveal="1"/);
  assert.notDeepEqual(tokenPositions(forward[0]).map(token => [token.x, token.y]), tokenPositions(forward[20]).map(token => [token.x, token.y]));
  assert.ok(tokenPositions(forward.at(-1)).every(token => Number(token.opacity) === 0));
  const matrixGroup = forward.at(-1).match(/<g opacity="1" data-matrix-reveal="1">([\s\S]*?)<\/g>/)[1];
  const labels = [...matrixGroup.matchAll(/<text\b[^>]*>([^<]+)<\/text>/g)].map(match => match[1]);
  assert.deepEqual(labels.slice(0, displayTerms.length), displayTerms);
  for (const [row, expected] of counts.entries()) {
    const start = displayTerms.length + row * (displayTerms.length + 1);
    assert.equal(labels[start], documents[row].label);
    assert.deepEqual(labels.slice(start + 1, start + 1 + displayTerms.length).map(Number), displayTerms.map(term => expected[vocabulary.indexOf(term)]));
  }
  assert.equal(bagVisual(6, 5.8, false), countAssembly(2));
});

test('preprocessing preserves occurrences and applies the slide choices to the worked matrix', () => {
  const tokenWords=markup=>[...markup.matchAll(/<text\b[^>]*data-token="\d+"[^>]*>([^<]+)<\/text>/g)].map(m=>m[1]);
  assert.deepEqual(tokenWords(tokenizationVisual(0)),documents[0].text.split(' '));
  assert.deepEqual(tokenWords(tokenizationVisual(1)),tokenize(documents[0].text));
  assert.deepEqual([...lemmatizationVisual(1).matchAll(/data-lemma="([^"]+)"/g)].map(m=>m[1]),['drop','enter','make']);
  const final=stopWordsVisual(1);
  const retained=[...final.matchAll(/<g data-token="\d+" data-kept="true" opacity="1"><text\b[^>]*>([^<]+)<\/text>/g)].map(m=>m[1]);
  assert.deepEqual(retained,documents[0].tokens);
  assert.equal((final.match(/data-kept="false" opacity="0"/g)??[]).length,10);
  // The retained, lemmatized words are exactly those used in the count matrix.
  assert.equal(counts[0][vocabulary.indexOf('political')],1);
  assert.ok(!vocabulary.includes('the'));
  assert.ok(vocabulary.includes('drop'));
  assert.ok(!vocabulary.includes('drops'));
  for(const render of [tokenizationVisual,lemmatizationVisual,stopWordsVisual]){
    const positions=[0,.25,.5,.75,1],forward=positions.map(render);
    assert.deepEqual([...positions].reverse().map(render).reverse(),forward);
    assert.notEqual(forward[0],forward.at(-1));
    forward.forEach(markup=>checkMarkup(markup,'preprocessing'));
  }
});

test('TF–IDF reweighting retains exact starting counts and the model’s rounded endpoint values', () => {
  const values = markup => tableRows(markup).map(row => row.cells.map(cell => cell.value));
  const excerpt = matrix => matrix.map(row => displayTerms.map(term => row[vocabulary.indexOf(term)]));
  assert.deepEqual(values(tfVisual(4, 3.5)), excerpt(counts));
  assert.deepEqual(values(tfVisual(4, 4)), excerpt(tfidf).map(row => row.map(value => rounded(value, 2))));
  assert.deepEqual(values(tfVisual(4, 3.6, false)), values(tfVisual(4, 4)));
  const mid = values(tfVisual(4, 3.75));
  assert.deepEqual(mid, excerpt(counts.map((row, i) => row.map((value, j) => rounded((value + tfidf[i][j]) / 2, 2)))));
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
      assert.ok(expressions.includes(String.raw`c_{t,d}=${tf},\qquad\mathrm{df}_t=${df},\qquad N=${documents.length}`));
      const expected = tf * Math.log(documents.length / df);
      const [, relation, weight] = expressions[1].match(/(=|\\approx)([\d.]+)$/);
      assert.equal(relation, expected ? String.raw`\approx` : '=');
      assert.equal(Number(weight), rounded(expected));
      checkMarkup(markup, `${term} in D${doc + 1}`);
    }
  }
});
