import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
globalThis.katex = require('../docs/word2vec/vendor/katex/katex.min.js');
const {bagVisual, tfVisual, topicVisual, authorshipVisual, inspectTerm} = await import('../docs/text/visuals.mjs');
const {quoteFlowVisual} = await import('../docs/text/quote-flow.mjs');
const {aiExample} = await import('../docs/text/ai-example.mjs');
const {prose} = await import('../docs/text/math.mjs');
const {chapters} = await import('../docs/text/lesson.mjs');
const {documents, vocabulary, counts, tfidf, termStats, displayTerms} = await import('../docs/text/model.mjs');

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
  const renderers = {'bag-of-words': bagVisual, 'tf-idf': tfVisual, authorship: authorshipVisual, bertopic: topicVisual};
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


test('the primers lead directly into one quotation walkthrough and applications follow the core', () => {
  const bagIndex = chapters.findIndex(chapter => chapter.id === 'bag-of-words');
  const bag = chapters[bagIndex];
  const visuals = bag.stages.map(scene => scene.visual);
  const quoteIndex = visuals.indexOf('corpus');
  assert.deepEqual(visuals.slice(0, quoteIndex), ['speech', 'speech-counts', 'order', 'short-sentences', 'short-counts', 'short-cosine']);
  assert.deepEqual(visuals.slice(quoteIndex), ['corpus', 'tokens', 'lemma', 'stopwords', 'matrix']);
  assert.equal(chapters[bagIndex + 1].id, 'tf-idf');
  assert.deepEqual(chapters[bagIndex + 1].stages.map(scene => scene.visual), ['tf', 'df', 'idf', 'word-document', 'weighted', 'cosine']);
  assert.deepEqual(chapters.filter(chapter => chapter.application).map(chapter => chapter.id), ['authorship', 'bertopic']);
  assert.ok(chapters.slice(0, bagIndex + 2).every(chapter => !chapter.application));
  assert.ok(chapters.slice(bagIndex + 2).every(chapter => chapter.application));
  assert.deepEqual(chapters.find(chapter => chapter.id === 'authorship').stages.map(scene => scene.visual), ['federalist']);
  assert.deepEqual(chapters.find(chapter => chapter.id === 'bertopic').stages.map(scene => scene.visual), ['topic-pooling', 'topic-words']);

  // Verify the rendered examples after moving their chapter and stage positions.
  const render = visual => {
    const stage = visuals.indexOf(visual);
    return bagVisual(stage, stage, false);
  };
  const corpus = render('corpus');
  for (const {text, label} of documents) {
    assert.ok(corpus.includes(text));
    assert.ok(corpus.includes(label));
  }
  assert.match(render('order'), /Senators support voters/);
  assert.match(render('order'), /Voters support senators/);
  assert.match(render('speech-counts'), /class="matrix speech-matrix"/);
  assert.match(authorshipVisual(0, 0, false), /Hamilton/);
  assert.match(authorshipVisual(0, 0, false), /Madison/);
  assert.match(topicVisual(0, 0, false), /application-pooling/);
  assert.match(topicVisual(1, 1, false), /application-ranking/);
});

test('bag-of-words and TF–IDF route every quotation phase through the shared workspace', () => {
  const bag = chapters.find(chapter => chapter.id === 'bag-of-words');
  for (const phase of ['corpus', 'tokens', 'lemma', 'stopwords', 'matrix']) {
    const stage = bag.stages.findIndex(scene => scene.visual === phase);
    for (const q of [0, .5, 1]) {
      assert.equal(bagVisual(stage, stage - .5 + q / 2, true), quoteFlowVisual(phase, q));
    }
    assert.equal(bagVisual(stage, stage - .2, false), quoteFlowVisual(phase, 1));
  }
  const tf = chapters.find(chapter => chapter.id === 'tf-idf');
  for (const [stage, scene] of tf.stages.entries()) {
    for (const q of [0, .5, 1]) {
      assert.equal(tfVisual(stage, stage - .5 + q / 2, true), quoteFlowVisual(scene.visual, q));
    }
    assert.equal(tfVisual(stage, stage - .2, false), quoteFlowVisual(scene.visual, 1));
  }
});

const quoteCells = markup => [...markup.matchAll(/<g\b([^>]*data-key="cell-\d+-\d+"[^>]*)>([\s\S]*?)<\/g>/g)].map(([, attrs, body]) => {
  const a = Object.fromEntries(attributes(attrs));
  return {
    row: Number(a['data-doc']), term: a['data-term'], value: Number(a['data-value']),
    highlighted: /fill="#edf2f5"/.test(body), present: /stroke="#234e70"/.test(body),
    displayed: Number(body.match(/<text\b[^>]*data-key="cell-value-[^"]+"[^>]*>([^<]+)<\/text>/)[1]),
  };
});

test('the TF–IDF narrative keeps the featured columns, document frequencies, and pair scores consistent', () => {
  for (const [stage, term] of [[0, 'political'], [1, 'citizen'], [4, 'citizen']]) {
    const markup = tfVisual(stage, stage, true), cells = quoteCells(markup);
    assert.equal(cells.length, documents.length * displayTerms.length);
    assert.deepEqual([...new Set(cells.filter(cell => cell.highlighted).map(cell => cell.term))], [term]);
    assert.doesNotMatch(markup, /<select\b|<input\b/);
  }
  assert.deepEqual(quoteCells(tfVisual(1, 1)).filter(cell => cell.present).map(cell => [cell.row, cell.term]), [[0, 'citizen']]);
  assert.ok(texExpressions(tfVisual(0, 0)).some(tex => tex.includes(String.raw`c_{\mathrm{political},S}=c_{\mathrm{political},W}=1`)));
  assert.ok(texExpressions(tfVisual(1, 1)).some(tex => tex.includes(String.raw`\mathrm{df}_{\mathrm{citizen}}=1`)));
  const idf = texExpressions(tfVisual(2, 2));
  for (const term of ['political', 'citizen']) {
    const {idf: weight} = termStats(term, 0);
    const expression = idf.find(tex => tex.includes(String.raw`\mathrm{${term}}:`));
    assert.ok(expression, `${term} must have an explicit IDF calculation.`);
    const [, relation, value] = expression.match(/(=|\\approx)([\d.]+)$/);
    assert.equal(relation, weight ? String.raw`\approx` : '=');
    assert.equal(Number(value), rounded(weight));
  }
  const pair = texExpressions(tfVisual(3, 3));
  assert.ok(pair.some(tex => tex.includes(String.raw`w_{t,d}=c_{t,d}\ln\!\left(\frac{N}{\mathrm{df}_t}\right)`)));
  for (const [row, {label}] of documents.entries()) {
    const stats = termStats('citizen', row);
    const expression = pair.find(tex => tex.includes(String.raw`w_{\mathrm{citizen},\mathrm{${label}}}`));
    assert.ok(expression?.includes(`=${stats.tf}\\ln2`));
    const [, relation, value] = expression.match(/(=|\\approx)([\d.]+)$/);
    assert.equal(relation, stats.weight ? String.raw`\approx` : '=');
    assert.equal(Number(value), rounded(stats.weight));
  }
  assert.match(tfVisual(5, 5), /Cosine similarity of the full 25-term vectors is 0/);
});

test('routed TF–IDF weighting keeps exact model values and correctly rounded cell labels', () => {
  for (const progress of [3.5, 3.75, 4]) {
    const q = (progress - 3.5) * 2;
    const displayed = quoteCells(tfVisual(4, progress));
    for (const cell of displayed) {
      const column = vocabulary.indexOf(cell.term);
      const expected = counts[cell.row][column] + (tfidf[cell.row][column] - counts[cell.row][column]) * q;
      assert.equal(cell.value, expected);
      assert.equal(cell.displayed, rounded(expected, 2));
    }
  }
  assert.deepEqual(quoteCells(tfVisual(4, 3.6, false)), quoteCells(tfVisual(4, 4)));
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
