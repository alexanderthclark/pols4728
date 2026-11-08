import {tokenize, cosine} from './model.mjs?v=20261108-topics-3';
import {latex, escapeText as esc} from './math.mjs?v=20261108-topics-3';

// This small corpus keeps word forms and stop words, unlike the later quotations.
export const shortDocuments = [
  'Plebiscites are unwise.',
  'Democracy is foolish.',
  'Majority rule rules.',
].map(text => ({text, tokens: tokenize(text)}));
export const shortVocabulary = [...new Set(shortDocuments.flatMap(doc => doc.tokens))];
export const shortCounts = shortDocuments.map(doc => shortVocabulary.map(term => doc.tokens.filter(token => token === term).length));
export const shortComparisons = [1, 2].map(other => ({
  pair: [0, other],
  dot: shortCounts[0].reduce((sum, value, column) => sum + value * shortCounts[other][column], 0),
  similarity: cosine(shortCounts[0], shortCounts[other]),
  distance: 1 - cosine(shortCounts[0], shortCounts[other]),
}));

const clamp = value => Math.max(0, Math.min(1, value));
const math = (tex, key) => `<div class="math" data-key="${key}">${latex(tex, true)}</div>`;

export function shortSentencesVisual(scene, progress = 1) {
  const q = clamp(progress);
  if (!['short-sentences', 'short-counts', 'short-cosine'].includes(scene)) throw new RangeError(`Unknown short-sentence scene: ${scene}`);
  const countProgress = scene === 'short-sentences' ? 0 : scene === 'short-counts' ? q : 1;
  const comparisonProgress = scene === 'short-cosine' ? q : 0;
  const matrixReveal = clamp((countProgress - .2) / .5);
  const comparisonReveal = clamp(comparisonProgress * 5);
  const activeOther = comparisonProgress === 0 ? -1 : comparisonProgress < .55 ? 1 : 2;
  const active = row => activeOther >= 0 && (row === 0 || row === activeOther);
  // Every stage returns the same keyed structure. Sentence labels, table cells,
  // and formulas can be updated in place instead of replacing the diagram.
  return `<div class="short-flow" data-key="short-flow" data-visual-family="short-sentences" data-active-pair="${activeOther < 0 ? '' : `1-${activeOther + 1}`}" style="--count-progress:${countProgress}">
    <ol class="short-sentences" data-key="short-labels">${shortDocuments.map((doc, r) => `<li data-key="short-label-${r}" class="${active(r) ? 'compared' : ''}">${esc(doc.text)}</li>`).join('')}</ol>
    <div class="short-matrix-panel" data-key="short-matrix-panel" style="opacity:${matrixReveal}" aria-hidden="${matrixReveal === 0}">
      <table class="short-matrix" data-key="short-matrix"><caption>Word counts</caption><thead><tr><th scope="col">Sentence</th>${shortVocabulary.map((term, c) => `<th scope="col" data-key="short-term-${c}"><span>${term}</span></th>`).join('')}</tr></thead><tbody>${shortCounts.map((row, r) => `<tr data-key="short-row-${r}" data-sentence="${r + 1}" class="${active(r) ? 'compared' : ''}"><th scope="row">${r + 1}</th>${row.map((value, c) => {
        // Each occurrence travels down its own term column to the sentence row.
        const arrival = clamp((countProgress - .25 - c * .035) / .47);
        const lift = (1 - arrival) * (1.8 + r * 2.4);
        return `<td data-key="short-cell-${r}-${c}" data-row="${r}" data-column="${c}" data-count="${value}" class="${value ? 'occupied' : 'zero'}"><span${value ? ` style="transform:translateY(-${lift}em);opacity:${Math.min(1, arrival * 5)}"` : ''}>${value}</span></td>`;
      }).join('')}</tr>`).join('')}</tbody></table>
    </div>
    <div class="short-cosine" data-key="short-cosine" style="opacity:${comparisonReveal}" aria-hidden="${comparisonReveal === 0}">
      ${math(String.raw`\cos(\mathbf{x}_i,\mathbf{x}_j)=\frac{\mathbf{x}_i\cdot\mathbf{x}_j}{\lVert\mathbf{x}_i\rVert\lVert\mathbf{x}_j\rVert}`, 'short-cosine-definition')}
      <div class="short-comparisons" data-key="short-comparisons">${shortComparisons.map(({pair, dot, similarity}, index) => {
        const reveal = clamp((comparisonProgress - index * .5) * 4);
        return `<div class="short-comparison" data-key="short-pair-${index}" data-pair="${pair.map(i => i + 1).join('-')}" data-active="${activeOther === pair[1]}"><span>${pair.map(i => i + 1).join(' and ')}</span><div class="short-calculation" style="opacity:${reveal}">${latex(String.raw`\frac{${dot}}{\sqrt{3}\sqrt{3}}=${similarity}`)}</div></div>`;
      }).join('')}</div>
      <div class="short-distance" data-key="short-distance" style="opacity:${clamp((comparisonProgress - .7) / .3)}">${math(String.raw`d_{\cos}=1-\cos`, 'short-distance-definition')}${math(String.raw`d_{\cos}(\mathbf{x}_1,\mathbf{x}_2)=d_{\cos}(\mathbf{x}_1,\mathbf{x}_3)=1`, 'short-distance-result')}</div>
    </div>
  </div>`;
}
