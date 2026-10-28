/**
 * Small, synthetic examples for the POLS 4728 text representation lesson.
 * Matrices are document × vocabulary; each column always names the same word.
 * TF is a raw token count. IDF is ln(N / df), without smoothing or normalization.
 */
export const documents = [
  { id: 'parliament-budget', label: 'Parliament / budget', text: 'The parliament debated the budget. The budget passed.' },
  { id: 'senate-budget', label: 'Senate / budget', text: 'The senate debated the budget.' },
  { id: 'parliament-health', label: 'Parliament / health', text: 'The parliament debated health.' },
  { id: 'senate-health', label: 'Senate / health', text: 'The senate debated health.' },
];

/** Deliberately simple English tokenizer: case-fold, then retain a–z runs. */
export function tokenize(text) {
  return String(text).toLowerCase().match(/[a-z]+/g) ?? [];
}

const tokens = documents.map(({ text }) => tokenize(text));
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
