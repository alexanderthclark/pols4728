/**
 * Schumpeter and Whitman quotations used in the course slides.
 * Matrices use the slides' lowercase, stop-word-filtered, lemmatized tokens.
 * TF is a raw count; IDF is ln(N / df), without smoothing or normalization.
 */
export const documents = [
  { id: 'schumpeter', label: 'Schumpeter', source: 'Capitalism, Socialism, and Democracy',
    text: 'Thus the typical citizen drops down to a lower level of mental performance as soon as he enters the political field.',
    tokens: 'typical citizen drop down lower level mental performance enter political field'.split(' ') },
  { id: 'whitman', label: 'Whitman', source: 'Democratic Vistas',
    text: 'Political Democracy, as it exists and practically works in America, with all its threatening evils, supplies a training school for making grand young men.',
    tokens: 'political democracy exist practically work america threaten evil supply training school make grand young man'.split(' ') },
];

// Explicit teaching choices reproduce the slide example; not a general lemmatizer.
export const stopWords = new Set('thus the to a of as soon he it and in with all its for'.split(' '));
export const lemmaForms = {drops:'drop',enters:'enter',exists:'exist',works:'work',threatening:'threaten',evils:'evil',supplies:'supply',making:'make',men:'man'};
export const displayTerms = ['political','citizen','mental','democracy','america','young'];

/** Deliberately simple English tokenizer: case-fold, then retain a–z runs. */
export function tokenize(text) {
  return String(text).toLowerCase().match(/[a-z]+/g) ?? [];
}

const tokens = documents.map(document => document.tokens);
export const vocabulary = [...new Set(tokens.flat())].sort();
export const counts = tokens.map(words =>
  vocabulary.map(term => words.filter(word => word === term).length));

export function termStats(term, docIndex) {
  if (!Number.isInteger(docIndex) || docIndex < 0 || docIndex >= documents.length) {
    throw new RangeError('Choose an existing document index.');
  }
  const column = vocabulary.indexOf(term);
  if (column < 0) throw new RangeError('Choose a term in the displayed vocabulary.');
  const N = documents.length;
  const tf = counts[docIndex][column];
  const df = counts.filter(row => row[column] > 0).length;
  const idf = Math.log(N / df);
  return { tf, df, N, idf, weight: tf * idf };
}

export const tfidf = documents.map((_, docIndex) =>
  vocabulary.map(term => termStats(term, docIndex).weight));

/** Zero vectors receive similarity 0; unequal dimensions are a caller error. */
export function cosine(a, b) {
  if (a.length !== b.length) throw new RangeError('Vectors must have equal dimensions.');
  const dot = a.reduce((total, value, i) => total + value * b[i], 0);
  const normA = Math.hypot(...a);
  const normB = Math.hypot(...b);
  return normA && normB ? dot / (normA * normB) : 0;
}
