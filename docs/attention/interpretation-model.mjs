import {softmax} from './model.mjs';

// A constructed attention-pooling classifier, not measurements from a fitted
// language model. One fixed summary query reads the three source positions.
// There are no residual connections, normalization, MLPs, or later layers.
// The two sentiment words share a key and differ only in their value vector.
export const INTERPRETATION_WEIGHTS = Object.freeze([.6, .3, .1]);
export const INTERPRETATION_READOUT = Object.freeze([1, 0]);
export const INTERPRETATION_QUERY = 1;

export function attentionInterpretation({tone = 'awful', removeToken = null} = {}) {
  if (!['awful', 'great'].includes(tone)) throw new RangeError('Choose awful or great.');
  const sources = [
    {token: 'film', key: Math.log(.6), value: [.1, 4]},
    {token: 'was', key: Math.log(.3), value: [0, 1]},
    {token: tone, key: Math.log(.1), value: [tone === 'awful' ? -8 : 8, 2]},
  ];
  if (removeToken !== null && !sources.some(source => source.token === removeToken)) {
    throw new RangeError('The deleted token must be present in the toy input.');
  }
  const retained = sources.filter(source => source.token !== removeToken);
  const tokens = retained.map(source => source.token);
  const keys = retained.map(source => source.key);
  const values = retained.map(source => source.value);
  // d_k=1; softmax(q k_j / sqrt(d_k)) is exactly (.6, .3, .1)
  // without deletion. Deleting a source recomputes this normalization.
  const scores = keys.map(key => INTERPRETATION_QUERY * key);
  const weights = softmax(scores);
  const output = [0, 1].map(coordinate => values.reduce(
    (sum, value, source) => sum + weights[source] * value[coordinate], 0));
  const readout = [...INTERPRETATION_READOUT];
  const projections = values.map(value => value.reduce(
    (sum, coordinate, feature) => sum + coordinate * readout[feature], 0));
  const contributions = projections.map((projection, source) => weights[source] * projection);
  const bias = 0;
  const logit = bias + output.reduce((sum, coordinate, feature) => sum + coordinate * readout[feature], 0);
  const probability = 1 / (1 + Math.exp(-logit));
  return {tokens, query: INTERPRETATION_QUERY, keys, scores, weights, values,
    output, readout, projections, contributions, bias, logit, probability};
}

export function deletionExperiment({tone = 'awful', token = 'film'} = {}) {
  const original = attentionInterpretation({tone});
  const deleted = attentionInterpretation({tone, removeToken: token});
  const originalTerm = original.contributions[original.tokens.indexOf(token)];
  // This is the change caused by this particular intervention in this toy,
  // not the negative of the original forward-pass contribution.
  return {original, deleted, originalTerm, logitChange: deleted.logit - original.logit};
}
