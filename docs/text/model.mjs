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

/*
 * A 6-word × 2-coordinate CBOW example, using a radius-one symmetric window.
 * These input and output embeddings are fixed illustrative parameters, chosen
 * for transparent arithmetic. They are not empirical or corpus-trained vectors.
 * The input/output tables are distinct model parameters, as in word2vec.
 * We use full softmax here to expose every probability; large word2vec models
 * commonly use hierarchical softmax or negative sampling for efficiency.
 */
const words = ['the', 'senate', 'parliament', 'debated', 'budget', 'health'];
const dimensions = 2;
const sentenceTokens = ['the', 'senate', 'debated', 'the', 'budget'];
const targetIndex = 1;
const windowRadius = 1;
const contextWords = ['the', 'debated'];
const targetWord = 'senate';
const inputVectors = {
  the: [1, 0], senate: [0.8, 0.2], parliament: [0.7, 0.3],
  debated: [0, 1], budget: [0.2, 0.8], health: [0.3, 0.7],
};
const outputVectors = {
  the: [-0.4, -0.2], senate: [0.2, 0.4], parliament: [0.4, 0],
  debated: [-0.2, 0.2], budget: [0, 0.2], health: [0.2, -0.2],
};

const copyVectors = vectors => Object.fromEntries(words.map(word => [word, [...vectors[word]]]));
const dot = (a, b) => a.reduce((total, value, i) => total + value * b[i], 0);

function evaluate(inputs, outputs) {
  const averageVector = Array.from({ length: dimensions }, (_, coordinate) =>
    contextWords.reduce((total, word) => total + inputs[word][coordinate], 0) / contextWords.length);
  const scores = Object.fromEntries(words.map(word => [word, dot(outputs[word], averageVector)]));
  // Subtracting the maximum preserves softmax while preventing overflow.
  const maxScore = Math.max(...Object.values(scores));
  const exponentials = Object.fromEntries(words.map(word => [word, Math.exp(scores[word] - maxScore)]));
  const normalizer = Object.values(exponentials).reduce((total, value) => total + value, 0);
  const probabilities = Object.fromEntries(words.map(word => [word, exponentials[word] / normalizer]));
  // Log-sum-exp keeps the loss finite even if a tiny probability underflows.
  const loss = maxScore + Math.log(normalizer) - scores[targetWord];
  return {
    inputVectors: copyVectors(inputs), outputVectors: copyVectors(outputs),
    averageVector, scores, probabilities, loss,
  };
}

export const cbowExample = {
  words, dimensions, sentenceTokens, targetIndex, windowRadius, contextWords, targetWord,
  ...evaluate(inputVectors, outputVectors),
};

/**
 * One simultaneous gradient-descent update for L = -ln P(senate | the, debated).
 * dL/du_w = (p_w - 1[w=target]) h, where h is the average context embedding.
 * Each context input receives (dL/dh) / number of context positions.
 * Every call starts at the same illustrative parameters; it does not mutate them.
 */
export function cbowStep(rate = 0.5) {
  if (!Number.isFinite(rate) || rate < 0) throw new RangeError('The learning rate must be finite and nonnegative.');
  const before = evaluate(inputVectors, outputVectors);
  const outputGradients = Object.fromEntries(words.map(word => {
    const error = before.probabilities[word] - Number(word === targetWord);
    return [word, before.averageVector.map(value => error * value)];
  }));
  const averageGradient = Array.from({ length: dimensions }, (_, coordinate) =>
    words.reduce((total, word) => {
      const error = before.probabilities[word] - Number(word === targetWord);
      return total + error * outputVectors[word][coordinate];
    }, 0));
  const inputGradients = Object.fromEntries(words.map(word => {
    const occurrences = contextWords.filter(contextWord => contextWord === word).length;
    return [word, averageGradient.map(value => occurrences * value / contextWords.length)];
  }));
  const updatedInputs = Object.fromEntries(words.map(word => [word,
    inputVectors[word].map((value, coordinate) => value - rate * inputGradients[word][coordinate])
  ]));
  const updatedOutputs = Object.fromEntries(words.map(word => [word,
    outputVectors[word].map((value, coordinate) => value - rate * outputGradients[word][coordinate])
  ]));
  return {
    rate, before, after: evaluate(updatedInputs, updatedOutputs),
    gradients: { inputVectors: inputGradients, outputVectors: outputGradients, averageVector: averageGradient },
  };
}
