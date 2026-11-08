import {documents, tokenize, lemmaForms, stopWords, displayTerms, vocabulary, counts, tfidf, cosine} from './model.mjs?v=20261108-flow-7';
import {escapeText as esc, mathLabel} from './math.mjs?v=20261108-flow-7';

const clamp = value => Math.max(0, Math.min(1, value));
const mix = (a, b, q) => a + (b - a) * q;
const blue = '#234e70', rust = '#a54f32', ink = '#242424', muted = '#606060', rule = '#deded9';
const phases = ['corpus', 'tokens', 'lemma', 'stopwords', 'matrix', 'tf', 'df', 'idf', 'word-document', 'weighted', 'cosine'];
const columns = displayTerms.map((word, index) => ({word, column: vocabulary.indexOf(word), x: 200 + index * 80}));
const rowY = [220, 290];
const quoteY = [62, 263];
const authorY = [28, 225];
const format = (value, weighted) => Math.abs(value) < 1e-10 ? '0' : weighted ? String(Number(value.toFixed(2))) : String(value);
const text = (key, x, y, value, attributes = '') => `<text data-key="${key}" x="${x}" y="${y}" ${attributes}>${esc(value)}</text>`;

function wordSlots(words, top) {
  let x = 14, y = top;
  return words.map(word => {
    // Deliberately loose advances leave punctuation and the widest glyphs room.
    const width = word.length * 16 + 16;
    if (x + width > 665) { x = 14; y += 38; }
    const slot = {x, y, width};
    x += width;
    return slot;
  });
}

const quoteTokens = documents.map((document, row) => {
  const raw = document.text.split(' '), lower = tokenize(document.text);
  const lemmas = lower.map(word => lemmaForms[word] ?? word);
  const start = wordSlots(raw, quoteY[row]);
  const retained = wordSlots(lemmas.filter(word => !stopWords.has(word)), quoteY[row]);
  let retainedIndex = 0;
  return raw.map((word, index) => ({
    raw: word, lower: lower[index], lemma: lemmas[index], index, row,
    removed: stopWords.has(lemmas[index]),
    start: start[index],
    end: stopWords.has(lemmas[index]) ? start[index] : retained[retainedIndex++],
  }));
});

function calculation(phase) {
  if (phase === 'tf') return {
    primary: String.raw`c_{\mathrm{political},S}=c_{\mathrm{political},W}=1`,
    secondary: '“Political” appears in both documents.',
  };
  if (phase === 'df') return {
    primary: String.raw`\mathrm{df}_{\mathrm{citizen}}=1,\qquad N=2`,
    secondary: 'One document contains “citizen”.',
  };
  if (phase === 'idf') return {
    primary: String.raw`\mathrm{idf}_t=\ln\!\left(\frac{N}{\mathrm{df}_t}\right)`,
    secondaryMath: [String.raw`\mathrm{political}:\;\ln(2/2)=0`, String.raw`\mathrm{citizen}:\;\ln(2/1)\approx0.693`],
  };
  if (phase === 'word-document') return {
    primary: String.raw`w_{t,d}=c_{t,d}\ln\!\left(\frac{N}{\mathrm{df}_t}\right)`,
    secondaryMath: [String.raw`w_{\mathrm{citizen},\mathrm{Schumpeter}}=1\ln2\approx0.693`, String.raw`w_{\mathrm{citizen},\mathrm{Whitman}}=0\ln2=0`],
  };
  if (phase === 'weighted') return {
    primary: String.raw`w_{\mathrm{citizen},\mathrm{Schumpeter}}=1\ln(2/1)\approx0.693`,
    secondary: 'Same vocabulary; counts become weights.',
  };
  if (phase === 'cosine') return {
    primary: String.raw`\cos(\mathbf{x}_S,\mathbf{x}_W)=\frac{\mathbf{x}_S\cdot\mathbf{x}_W}{\lVert\mathbf{x}_S\rVert\lVert\mathbf{x}_W\rVert}=0`,
    secondary: 'Full 25-term vectors · shared “political” has weight 0',
  };
  return {primary: '', secondary: ''};
}

/**
 * Both quotations occupy one workspace from prose through weighted vectors.
 * Node keys, author identities, and matrix coordinates stay fixed throughout.
 */
export function quoteFlowVisual(phase, progress = 1) {
  const phaseIndex = phases.indexOf(phase);
  if (phaseIndex < 0) throw new RangeError(`Unknown quotation phase: ${phase}`);
  const q = clamp(progress);
  const tokenAmount = phaseIndex < 1 ? 0 : phase === 'tokens' ? q : 1;
  const lemmaAmount = phaseIndex < 2 ? 0 : phase === 'lemma' ? q : 1;
  const removal = phaseIndex < 3 ? 0 : phase === 'stopwords' ? q : 1;
  const assembly = phaseIndex < 4 ? 0 : phase === 'matrix' ? q : 1;
  const tableOpacity = clamp((assembly - .15) / .65);
  const weightAmount = phaseIndex < 9 ? 0 : phase === 'weighted' ? q : 1;
  const values = counts.map((row, r) => row.map((count, c) => mix(count, tfidf[r][c], weightAmount)));
  const focus = phase === 'tf' || phase === 'cosine' ? 'political' : ['df', 'idf', 'word-document', 'weighted'].includes(phase) ? 'citizen' : null;
  const isWeighted = phaseIndex >= 9;
  const tableTitle = phase === 'weighted' && q < 1 ? 'Counts → TF–IDF weights · six of 25 terms' : isWeighted ? 'TF–IDF weights · six of 25 terms' : 'Word counts · six of 25 terms';

  const authors = documents.map((document, row) => text(`author-${row}`, 14, mix(authorY[row], rowY[row], assembly), document.label,
    `class="q-author" font-size="31" fill="${row ? rust : blue}"`)).join('');
  const sources = documents.map((document, row) => text(`source-${row}`, 14, row ? 453 : 177, document.source,
    `font-size="20" fill="${muted}" opacity="${(1 - tokenAmount) * (1 - assembly)}"`)).join('');
  const tokens = quoteTokens.flat().map(token => {
    const {row, index, raw, lower, lemma, removed, start, end} = token;
    const selected = columns.find(column => column.word === lemma);
    const compactX = mix(start.x, end.x, removal), compactY = mix(start.y, end.y, removal);
    const x = mix(compactX, selected ? selected.x : compactX, assembly);
    const y = mix(compactY, rowY[row], assembly);
    const opacity = (removed ? 1 - removal : 1) * (1 - clamp((assembly - .3) / .65));
    const word = lemmaAmount >= .5 ? lemma : tokenAmount >= .5 ? lower : raw;
    const formChanged = lemma !== lower && lemmaAmount > 0;
    const color = removed && removal > 0 ? muted : formChanged && assembly < 1 ? blue : ink;
    return `<g data-key="word-${row}-${index}" data-doc="${row}" data-token="${index}" data-kept="${!removed}" opacity="${opacity}">
      <rect data-key="box-${row}-${index}" x="${x - 3}" y="${y - 28}" width="${word.length * 16 + 8}" height="36" fill="none" stroke="${formChanged ? blue : rule}" stroke-width="1" opacity="${tokenAmount * (1 - removal) * (1 - assembly)}"/>
      ${text(`text-${row}-${index}`, x, y, word, `font-size="32" fill="${color}"`)}
      <path data-key="strike-${row}-${index}" d="M${x - 2} ${y - 10}h${word.length * 16}" stroke="${muted}" stroke-width="2" opacity="${removed ? removal * (1 - assembly) : 0}"/>
    </g>`;
  }).join('');
  const grid = `<g data-key="matrix" class="q-matrix" opacity="${tableOpacity}" aria-hidden="${tableOpacity === 0}">
    ${text('matrix-title', 340, 34, tableTitle, `text-anchor="middle" font-size="24" fill="${muted}"`)}
    ${columns.map(({word, x}) => text(`column-${word}`, x, 171, word, `transform="rotate(-55 ${x} 171)" font-size="28" fill="${word === focus ? blue : ink}"`)).join('')}
    ${[188, 255, 325].map((y, index) => `<path data-key="rule-${index}" d="M14 ${y}H665" stroke="${rule}" fill="none"/>`).join('')}
    ${counts.map((row, r) => columns.map(({word, column, x}) => {
      const value = values[r][column];
      const highlighted = word === focus || phase === 'idf' && word === 'political';
      const present = phase === 'df' && word === focus && row[column] > 0;
      return `<g data-key="cell-${r}-${column}" data-doc="${r}" data-term="${word}" data-value="${value}">
        <rect data-key="cell-fill-${r}-${column}" x="${x - 34}" y="${rowY[r] - 30}" width="68" height="51" fill="${highlighted ? '#edf2f5' : 'white'}" stroke="${present ? blue : 'none'}" stroke-width="2"/>
        ${text(`cell-value-${r}-${column}`, x, rowY[r], format(value, isWeighted), `text-anchor="middle" font-size="${mix(40, 32, weightAmount)}" fill="${Math.abs(value) < 1e-10 ? muted : highlighted ? blue : ink}"`)}
      </g>`;
    }).join('')).join('')}
  </g>`;
  const calc = calculation(phase);
  const equations = `<g data-key="calculation" class="q-calculation" opacity="${phaseIndex < 5 ? 0 : phase === 'tf' ? q : 1}">
    <g data-key="primary">${mathLabel(8, 337, 664, 60, calc.primary, 31)}</g>
    <g data-key="secondary">${calc.secondaryMath ? calc.secondaryMath.map((tex, index) => mathLabel(8, 398 + index * 38, 664, 38, tex, 29)).join('') : text('calculation-note', 340, 420, calc.secondary || '', `text-anchor="middle" font-size="25" fill="${muted}"`)}</g>
  </g>`;
  const description = phaseIndex < 4
    ? `${documents.map(document => `${document.label}: ${document.text}`).join(' ')} ${phase === 'stopwords' ? 'Selected stop words disappear; Schumpeter retains 11 tokens and Whitman retains 15.' : phase === 'lemma' ? 'Inflected forms change to their dictionary forms.' : phase === 'tokens' ? 'Each word becomes a lowercase alphabetic token.' : ''}`
    : `${isWeighted ? 'TF–IDF weights' : 'Word counts'} for Schumpeter and Whitman. Six displayed columns: ${columns.map(({word, column}) => `${word}: ${format(values[0][column], isWeighted)}, ${format(values[1][column], isWeighted)}`).join('; ')}. ${phase === 'cosine' ? `Cosine similarity of the full 25-term vectors is ${cosine(tfidf[0], tfidf[1])}.` : ''}`;
  return `<div class="quote-flow" data-visual-family="quotations" data-phase="${phase}"><svg class="quote-workspace" viewBox="0 0 680 480" role="group" aria-label="${esc(description)}">${grid}<g data-key="authors">${authors}</g><g data-key="sources">${sources}</g><g data-key="words">${tokens}</g>${equations}</svg></div>`;
}
