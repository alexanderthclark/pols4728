import {matmul, softmax, transpose} from './model.mjs';

// A constructed self-attention classifier, not measurements from a fitted
// language model. Each token receives an output row; mean pooling supplies
// the sentence summary that the linear classifier reads.
// There are no residual connections, normalization, MLPs, or later layers.
// The two sentiment words share a query and key and differ only in their value.
export const INTERPRETATION_ATTENTION_MATRIX = Object.freeze([
  Object.freeze([.80, .15, .05]),
  Object.freeze([.60, .30, .10]),
  Object.freeze([.40, .45, .15]),
]);
// These are the column means of A: effective source weights AFTER pooling.
export const INTERPRETATION_WEIGHTS = Object.freeze([.6, .3, .1]);
export const INTERPRETATION_READOUT = Object.freeze([1, 0]);
export const INTERPRETATION_QUERY_DIMENSION = 3;

export function attentionInterpretation({tone = 'awful', removeToken = null} = {}) {
  if (!['awful', 'great'].includes(tone)) throw new RangeError('Choose awful or great.');
  const sources = [
    {token: 'film', value: [.1, 4]},
    {token: 'was', value: [0, 1]},
    {token: tone, value: [tone === 'awful' ? -8 : 8, 2]},
  ].map((source, j) => ({
    ...source,
    query: [0, 1, 2].map(coordinate => Number(coordinate === j)),
    key: INTERPRETATION_ATTENTION_MATRIX.map(row =>
      Math.sqrt(INTERPRETATION_QUERY_DIMENSION) * Math.log(row[j])),
  }));
  if (removeToken !== null && !sources.some(source => source.token === removeToken)) {
    throw new RangeError('The deleted token must be present in the toy input.');
  }
  const retained = sources.filter(source => source.token !== removeToken);
  const tokens = retained.map(source => source.token);
  const queries = retained.map(source => source.query);
  const keys = retained.map(source => source.key);
  const values = retained.map(source => source.value);
  // Q=I_3 and K=sqrt(3)*transpose(log(A)) realize the displayed A.
  // Deletion removes a destination AND a source. Retained Q/K/V vectors
  // keep their original coordinates, so d_k stays 3, then every row is
  // renormalized over the remaining sources.
  const scores = matmul(queries, transpose(keys)).map(row => row.map(
    score => score / Math.sqrt(INTERPRETATION_QUERY_DIMENSION)));
  const attentionMatrix = scores.map(softmax);
  const outputs = matmul(attentionMatrix, values);
  const weights = tokens.map((_, source) => attentionMatrix.reduce(
    (sum, row) => sum + row[source], 0) / tokens.length);
  const output = [0, 1].map(coordinate => outputs.reduce(
    (sum, row) => sum + row[coordinate], 0) / tokens.length);
  const readout = [...INTERPRETATION_READOUT];
  const projections = values.map(value => value.reduce(
    (sum, coordinate, feature) => sum + coordinate * readout[feature], 0));
  const contributions = projections.map((projection, source) => weights[source] * projection);
  const bias = 0;
  const logit = bias + output.reduce((sum, coordinate, feature) => sum + coordinate * readout[feature], 0);
  const probability = 1 / (1 + Math.exp(-logit));
  return {tokens, queries, keys, scores, attentionMatrix, weights, values,
    outputs, output, readout, projections, contributions, bias, logit, probability};
}

export function deletionExperiment({tone = 'awful', token = 'film'} = {}) {
  const original = attentionInterpretation({tone});
  const deleted = attentionInterpretation({tone, removeToken: token});
  const originalTerm = original.contributions[original.tokens.indexOf(token)];
  // This is the change caused by this particular intervention in this toy,
  // not the negative of the original forward-pass contribution.
  return {original, deleted, originalTerm, logitChange: deleted.logit - original.logit};
}

