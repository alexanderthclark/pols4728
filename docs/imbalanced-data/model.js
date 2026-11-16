(function (root, factory) {
  "use strict";
  var model = factory();
  if (typeof module === "object" && module.exports) module.exports = model;
  if (root) root.ImbalanceModel = model;
})(typeof window !== "undefined" ? window : null, function () {
  "use strict";

  var POPULATION_PRIOR = 0.01;
  var SIGNALS = Object.freeze({
    A: Object.freeze({ name: "A", tpr: 0.5, fpr: 0 }),
    B: Object.freeze({ name: "B", tpr: 0.9, fpr: 0.1 })
  });

  function probability(value, label) {
    if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) {
      throw new RangeError((label || "Probability") + " must be between 0 and 1.");
    }
    return value;
  }

  function inclusionRate(value, label) {
    probability(value, label);
    if (value === 0) throw new RangeError(label + " must be positive to preserve both classes.");
    return value;
  }

  function sigmoid(value) {
    if (typeof value !== "number" || Number.isNaN(value)) throw new TypeError("Log odds must be a number.");
    if (value >= 0) return 1 / (1 + Math.exp(-value));
    var e = Math.exp(value);
    return e / (1 + e);
  }

  function logit(p) {
    probability(p);
    return Math.log(p) - Math.log1p(-p);
  }

  // Sampling is independent of features conditional on the observed class.
  // s1 and s0 are the rare/common inclusion probabilities, respectively.
  function sampledProbability(p, s1, s0) {
    probability(p);
    inclusionRate(s1, "Rare inclusion probability");
    inclusionRate(s0, "Common inclusion probability");
    return (p * s1) / (p * s1 + (1 - p) * s0);
  }

  function correctedProbability(pSample, s1, s0) {
    probability(pSample);
    inclusionRate(s1, "Rare inclusion probability");
    inclusionRate(s0, "Common inclusion probability");
    return (pSample * s0) / (pSample * s0 + (1 - pSample) * s1);
  }

  // Binary entropy in natural-log units (nats).
  function entropy(p) {
    probability(p);
    if (p === 0 || p === 1) return 0;
    return -p * Math.log(p) - (1 - p) * Math.log1p(-p);
  }

  function resolveSignal(signal) {
    var resolved = typeof signal === "string" ? SIGNALS[signal.toUpperCase()] : signal;
    if (!resolved || typeof resolved !== "object") throw new TypeError("Signal must be A, B, or an object with tpr and fpr.");
    probability(resolved.tpr, "True positive rate");
    probability(resolved.fpr, "False positive rate");
    return resolved;
  }

  function stump(signal, prior) {
    var s = resolveSignal(signal);
    var pi = probability(prior === undefined ? POPULATION_PRIOR : prior, "Class prior");
    var highMass = pi * s.tpr + (1 - pi) * s.fpr;
    var lowMass = 1 - highMass;
    // An empty leaf contributes zero loss; its probability is not identified.
    var highProb = highMass === 0 ? null : pi * s.tpr / highMass;
    var lowProb = lowMass === 0 ? null : pi * (1 - s.tpr) / lowMass;
    var logloss = (highMass === 0 ? 0 : highMass * entropy(highProb)) +
      (lowMass === 0 ? 0 : lowMass * entropy(lowProb));
    var populationHighMass = POPULATION_PRIOR * s.tpr + (1 - POPULATION_PRIOR) * s.fpr;
    var populationLowMass = 1 - populationHighMass;
    var precision = populationHighMass === 0 ? null : POPULATION_PRIOR * s.tpr / populationHighMass;
    var populationLowProb = populationLowMass === 0 ? null : POPULATION_PRIOR * (1 - s.tpr) / populationLowMass;
    var auc = s.tpr >= s.fpr ? (1 + s.tpr - s.fpr) / 2 : (1 + s.fpr - s.tpr) / 2;
    return {
      name: s.name || "Custom signal",
      prior: pi,
      tpr: s.tpr,
      fpr: s.fpr,
      highMass: highMass,
      lowMass: lowMass,
      highProb: highProb,
      lowProb: lowProb,
      // Aliases make the API convenient for text labels and plotting.
      highProbability: highProb,
      lowProbability: lowProb,
      logloss: logloss,
      loss: logloss,
      informationGain: entropy(pi) - logloss,
      auc: auc,
      precision: precision,
      populationPrecision: precision,
      populationHighProbability: precision,
      populationLowProbability: populationLowProb,
      populationLogloss: stumpPopulationLoss(s)
    };
  }

  function stumpPopulationLoss(s) {
    var hi = POPULATION_PRIOR * s.tpr + (1 - POPULATION_PRIOR) * s.fpr;
    var lo = 1 - hi;
    return (hi === 0 ? 0 : hi * entropy(POPULATION_PRIOR * s.tpr / hi)) +
      (lo === 0 ? 0 : lo * entropy(POPULATION_PRIOR * (1 - s.tpr) / lo));
  }

  function bestStump(prior) {
    var a = stump("A", prior);
    var b = stump("B", prior);
    return a.logloss <= b.logloss ? a : b;
  }

  function efficiency(pi, N, batch) {
    pi = probability(pi === undefined ? POPULATION_PRIOR : pi, "Class prior");
    N = N === undefined ? 1000000 : N;
    batch = batch === undefined ? 128 : batch;
    if (!Number.isInteger(N) || N <= 0) throw new RangeError("Dataset size must be a positive integer.");
    if (!Number.isInteger(batch) || batch <= 0) throw new RangeError("Batch size must be a positive integer.");
    if (pi === 0 || pi === 1) throw new RangeError("Both classes must be present for balancing.");
    var minorityCount = N * pi;
    var majorityCount = N * (1 - pi);
    // Random undersampling retains all of the rarer class, then the same number
    // from the other class. The explainer uses pi = .01, so Y=1 is the rare class.
    var balancedDatasetSize = 2 * Math.min(minorityCount, majorityCount);
    var zeroMinorityProbability = Math.pow(1 - pi, batch);
    var balancedZeroMinorityProbability = Math.pow(0.5, batch);
    var natural = {
      prior: pi,
      datasetSize: N,
      expectedMinorityInBatch: batch * pi,
      zeroMinorityProbability: zeroMinorityProbability,
      probabilityAtLeastOneMinority: 1 - zeroMinorityProbability,
      batchesPerPass: Math.ceil(N / batch)
    };
    var balanced = {
      prior: 0.5,
      datasetSize: balancedDatasetSize,
      expectedMinorityInBatch: batch / 2,
      zeroMinorityProbability: balancedZeroMinorityProbability,
      probabilityAtLeastOneMinority: 1 - balancedZeroMinorityProbability,
      // With an explicitly stratified batch, the count is controlled, not random.
      stratifiedMinorityInBatch: Math.floor(batch / 2),
      stratifiedZeroMinorityProbability: batch > 1 ? 0 : null,
      batchesPerPass: Math.ceil(balancedDatasetSize / batch)
    };
    return {
      pi: pi,
      N: N,
      batch: batch,
      minorityCount: minorityCount,
      majorityCount: majorityCount,
      balancedDatasetSize: balancedDatasetSize,
      retainedFraction: balancedDatasetSize / N,
      datasetReductionFactor: N / balancedDatasetSize,
      expectedMinorityInBatch: natural.expectedMinorityInBatch,
      zeroMinorityProbability: zeroMinorityProbability,
      probabilityAtLeastOneMinority: natural.probabilityAtLeastOneMinority,
      balancedExpectedMinorityInBatch: balanced.expectedMinorityInBatch,
      balancedZeroMinorityProbability: balancedZeroMinorityProbability,
      // These zero-count probabilities assume independent sampling. They are
      // not exact finite-dataset, without-replacement probabilities.
      samplingAssumption: "Independent draws with replacement",
      natural: natural,
      balanced: balanced
    };
  }

  return Object.freeze({
    POPULATION_PRIOR: POPULATION_PRIOR,
    SIGNALS: SIGNALS,
    sigmoid: sigmoid,
    logit: logit,
    correctedProbability: correctedProbability,
    sampledProbability: sampledProbability,
    entropy: entropy,
    stump: stump,
    bestStump: bestStump,
    efficiency: efficiency
  });
});
