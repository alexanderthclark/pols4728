/* Fixed constructions for the logistic-regression / sigmoid-network comparison.
 * These are illustrative functions, not fitted estimates from BKZ or the later
 * learned two-ReLU-unit example. Every curve and region is a model forward pass.
 */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.NNComparisonModel = api;
})(typeof window !== 'undefined' ? window : globalThis, function () {
  'use strict';

  const sigmoid = z => z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
  const freeze = value => {
    if (value && typeof value === 'object') {
      Object.values(value).forEach(freeze);
      Object.freeze(value);
    }
    return value;
  };

  // The ordinary model is sigmoid(b + w'x). The shallow model is
  // sigmoid(outputBias + sum_j outputWeights[j] * sigmoid(b_j + w_j'x)).
  const logistic1d = freeze({ bias: -3.6, weights: [12] });
  const logistic2d = freeze({ bias: -8.2, weights: [10, 6.5] });
  const network1d = freeze({
    hiddenBias: [-5.6, -14.4],
    hiddenWeights: [[20], [20]],
    outputBias: -6,
    outputWeights: [12, -12]
  });
  const network2d = freeze({
    hiddenBias: [-3, 9, -3, 9],
    hiddenWeights: [[12, 0], [-12, 0], [0, 12], [0, -12]],
    outputBias: -33,
    outputWeights: [10, 10, 10, 10]
  });
  const affine = (bias, weights, x) => weights.reduce((sum, w, i) => sum + w * x[i], bias);
  const hidden = (model, x) => model.hiddenWeights.map((weights, j) =>
    sigmoid(affine(model.hiddenBias[j], weights, x)));
  const networkScore = (model, x) => affine(model.outputBias, model.outputWeights, hidden(model, x));
  const logisticScore1d = x => affine(logistic1d.bias, logistic1d.weights, [x]);
  const logisticScore2d = (x1, x2) => affine(logistic2d.bias, logistic2d.weights, [x1, x2]);
  const networkScore1d = x => networkScore(network1d, [x]);
  const networkScore2d = (x1, x2) => networkScore(network2d, [x1, x2]);

  // A fixed binary response pattern: zero at either end, one in the middle.
  // There is no noise, fit, random seed, or changing observation during a reveal.
  const points1d = freeze([
    ...[0.04, 0.09, 0.14, 0.19, 0.24].map(x => ({ x, y: 0 })),
    ...[0.34, 0.38, 0.42, 0.46, 0.5, 0.54, 0.58, 0.62, 0.66].map(x => ({ x, y: 1 })),
    ...[0.76, 0.81, 0.86, 0.91, 0.96].map(x => ({ x, y: 0 }))
  ].map((point, i) => ({ ...point, id: 'comparison-1d-' + i })));

  // The 2D observations are fixed concentric rings centered at (1/2, 1/2):
  // class one at the center and radii .12/.23; class zero at radius .39.
  // Opposite outer observations have a class-one midpoint, proving no affine
  // boundary can separate the classes. The four hidden units instead produce
  // a smooth, closed rounded region; their level set is not an imposed circle.
  const ring = (radius, count, y) => Array.from({ length: count }, (_, i) => {
    const angle = 2 * Math.PI * i / count;
    return { x1: 0.5 + radius * Math.cos(angle), x2: 0.5 + radius * Math.sin(angle), y };
  });
  const points2d = freeze([
    { x1: 0.5, x2: 0.5, y: 1 },
    ...ring(0.12, 8, 1),
    ...ring(0.23, 12, 1),
    ...ring(0.39, 24, 0)
  ].map((point, i) => ({ ...point, id: 'comparison-2d-' + i })));

  return freeze({
    domain: [0, 1],
    sigmoid,
    logistic1d,
    logistic2d,
    network1d,
    network2d,
    logisticScore1d,
    logisticProbability1d: x => sigmoid(logisticScore1d(x)),
    networkScore1d,
    networkProbability1d: x => sigmoid(networkScore1d(x)),
    logisticScore2d,
    logisticProbability2d: (x1, x2) => sigmoid(logisticScore2d(x1, x2)),
    networkScore2d,
    networkProbability2d: (x1, x2) => sigmoid(networkScore2d(x1, x2)),
    points1d,
    points2d,
    provenance: {
      description: 'Fixed illustrative sigmoid constructions; not a fit to BKZ data.',
      observations: 'Fixed binary points: a middle interval in one dimension and concentric rings in two dimensions.',
      interpretation: 'The same observations remain fixed while the displayed model changes. These chosen weights demonstrate capacity, not estimated performance.',
      hiddenActivation: 'sigmoid',
      outputActivation: 'sigmoid'
    }
  });
});
