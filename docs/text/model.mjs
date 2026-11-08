/**
 * Two full Peloton classes, represented in the same 127-class reference corpus.
 * Only brief transcript previews are published. Counts and document frequencies
 * come from the saved, audited document–term matrix; previews are not documents.
 * TF is a raw count; IDF is ln(N / df), without smoothing or normalization.
 */
import {pelotonVocabulary, pelotonDocumentFrequency, pelotonCounts, pelotonDocuments, pelotonCorpus, pelotonStopWords} from './peloton-data.mjs?v=20261108-peloton-1';

export const vocabulary = pelotonVocabulary;
export const counts = pelotonCounts;
export const documentFrequency = pelotonDocumentFrequency;
export const referenceCount = pelotonCorpus.documentCount;
export const corpus = pelotonCorpus;
export const stopWords = new Set(pelotonStopWords);
export const displayTerms = ['take','resistance','confidence','inhale','earth','tabletop'];
export const focusTerms = {common:'resistance',distinctive:'confidence'};
const vocabularySet = new Set(vocabulary);

/** Alphabetic preview tokens; the published previews have no contractions. */
export function tokenize(text) {
  return String(text).toLowerCase().match(/[a-z]+/g) ?? [];
}
export function isRetainedToken(token) {
  return token.length >= 3 && !stopWords.has(token) && vocabularySet.has(token);
}
export function preprocessPreview(text) {
  return tokenize(text).filter(isRetainedToken);
}
export const documents = pelotonDocuments.map(document => ({
  ...document,
  retainedVocabularySize:document.uniqueTermCount,
  tokens: preprocessPreview(document.text),
}));
export const explorerTerms = [
  ...displayTerms,'body','breath','cadence','exhale','feet','hands','heart','hips',
  'mat','move','pedal','pose','ride','saddle','shoulders','strength','work','love',
].filter(term => vocabularySet.has(term));

export function termStats(term, docIndex) {
  if (!Number.isInteger(docIndex) || docIndex < 0 || docIndex >= documents.length) {
    throw new RangeError('Choose an existing document index.');
  }
  const column = vocabulary.indexOf(term);
  if (column < 0) throw new RangeError('Choose a term in the corpus vocabulary.');
  const N = referenceCount;
  const tf = counts[docIndex][column];
  const df = documentFrequency[column];
  const idf = Math.log(N / df);
  return {tf, df, N, idf, weight:tf * idf};
}
export const tfidf = documents.map((_, docIndex) =>
  vocabulary.map(term => termStats(term, docIndex).weight));

/** Zero vectors receive similarity 0; unequal dimensions are a caller error. */
export function cosine(a, b) {
  if (a.length !== b.length) throw new RangeError('Vectors must have equal dimensions.');
  const dot = a.reduce((total, value, i) => total + value * b[i], 0);
  const normA = Math.hypot(...a), normB = Math.hypot(...b);
  return normA && normB ? dot / (normA * normB) : 0;
}
