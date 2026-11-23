/** Transparent arithmetic for the explainer. These parameters are fixed by hand,
 * not learned from LinkedIn data. The same parameters drive the scroll story and worked calculations. */
export const DETAIL_HEADS = [
  { queries: [[1, .2], [.8, .6], [.3, 1.1], [1.2, .8]], keys: [[.1, .3], [.7, .2], [.5, 1.1], [1.2, .6]] },
  { queries: [[.2, 1], [.6, .6], [1, .3], [.8, 1]], keys: [[.8, .1], [.3, 1.2], [1.1, .8], [.4, .6]] },
] as const;
export const DETAIL_VALUES = [.9, 1.8, 2.3, 2.7] as const;
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
const sum = (xs: readonly number[]) => xs.reduce((a, b) => a + b, 0);

export function traceAttention(head = 0, query = 0) {
  if (!Number.isInteger(head) || head < 0 || head >= DETAIL_HEADS.length || !Number.isInteger(query) || query < 0 || query > 3) throw new RangeError("Select one of the two heads and four queries.");
  const parameters = DETAIL_HEADS[head];
  const q = parameters.queries[query];
  const dots = parameters.keys.map(k => q[0] * k[0] + q[1] * k[1]);
  const scores = dots.map(d => d / Math.SQRT2);
  const exponents = scores.map(s => Math.exp(s));
  const denominator = sum(exponents);
  const weights = exponents.map(e => e / denominator);
  const contributions = weights.map((a, j) => a * DETAIL_VALUES[j]);
  return { q, keys: parameters.keys, dots, scores, exponents, denominator, weights, contributions, output: sum(contributions) };
}

export function traceWholeModel() {
  const traces = DETAIL_HEADS.map((_, h) => [0, 1, 2, 3].map(r => traceAttention(h, r)));
  const headOutputs = traces.map(rows => rows.map(t => t.output));
  const averagedOutputs = [0, 1, 2, 3].map(r => (headOutputs[0][r] + headOutputs[1][r]) / 2);
  const incoming = [0, 1, 2, 3].map(j => sum(traces.flatMap(rows => rows.map(t => t.weights[j]))));
  const credit = incoming.map(w => w / sum(incoming));
  const pooled = sum(averagedOutputs) / averagedOutputs.length;
  const bias = -2.4;
  return { traces, headOutputs, averagedOutputs, incoming, credit, pooled, bias, logit: bias + pooled, probability: sigmoid(bias + pooled) };
}
