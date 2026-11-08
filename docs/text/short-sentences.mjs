import {tokenize, cosine} from './model.mjs?v=20261108-cosine-5';
import {latex, escapeText as esc} from './math.mjs?v=20261108-cosine-5';

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
const math = tex => `<div class="math">${latex(tex, true)}</div>`;

export function shortSentencesVisual(scene, progress = 1) {
  const q = clamp(progress);
  if (scene === 'short-sentences') return `<ol class="short-sentences">${shortDocuments.map(doc => `<li>${esc(doc.text)}</li>`).join('')}</ol>`;
  if (scene === 'short-counts') return `<table class="short-matrix"><caption>Word counts</caption><thead><tr><th scope="col">Sentence</th>${shortVocabulary.map(term => `<th scope="col"><span>${term}</span></th>`).join('')}</tr></thead><tbody>${shortCounts.map((row, r) => `<tr><th scope="row">${r + 1}</th>${row.map((value, c) => {
    // A count travels down its vocabulary column into the corresponding row.
    const arrival = clamp((q - c * .045) / .64);
    const lift = (1 - arrival) * (1.8 + r * 2.4);
    return `<td data-row="${r}" data-column="${c}" data-count="${value}" class="${value ? 'occupied' : 'zero'}"><span${value ? ` style="transform:translateY(-${lift}em);opacity:${Math.min(1, arrival * 5)}"` : ''}>${value}</span></td>`;
  }).join('')}</tr>`).join('')}</tbody></table>`;
  if (scene === 'short-cosine') return `<div class="short-cosine">
    ${math(String.raw`\cos(\mathbf{x}_i,\mathbf{x}_j)=\frac{\mathbf{x}_i\cdot\mathbf{x}_j}{\lVert\mathbf{x}_i\rVert\lVert\mathbf{x}_j\rVert}`)}
    <div class="short-comparisons">${shortComparisons.map(({pair, dot, similarity}, index) => {
      const reveal = clamp(q * 2 - index * .65);
      return `<div class="short-comparison" data-pair="${pair.map(i => i + 1).join('-')}"><span>Sentences ${pair.map(i => i + 1).join(' and ')}</span><div class="short-calculation" style="clip-path:inset(0 ${(1 - reveal) * 100}% 0 0)">${latex(String.raw`\frac{${dot}}{\sqrt{3}\sqrt{3}}=${similarity}`)}</div></div>`;
    }).join('')}</div>
    <div class="short-distance">${math(String.raw`d_{\cos}=1-\cos`)}${math(String.raw`d_{\cos}(\mathbf{x}_1,\mathbf{x}_2)=d_{\cos}(\mathbf{x}_1,\mathbf{x}_3)=1`)}</div>
  </div>`;
  throw new RangeError(`Unknown short-sentence scene: ${scene}`);
}
