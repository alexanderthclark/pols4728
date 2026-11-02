import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
const model = require('../docs/imbalanced-data/model.js');
const tree = require('../docs/imbalanced-data/tree-model.js');

function close(actual, expected, label, tolerance = 1e-12) {
  assert.ok(Math.abs(actual - expected) < tolerance, `${label}: ${actual} vs ${expected}`);
}

test('sampling correction, exact model selection, and batch composition', () => {
  // At a 1% population prior, keeping all positives and 1/99 of negatives
  // produces a balanced sample. The probability transform must be invertible.
  const s1 = 1;
  const s0 = 1 / 99;
  close(model.sampledProbability(0.01, s1, s0), 0.5, "Balanced sampled prior");
  close(model.correctedProbability(0.5, s1, s0), 0.01, "Recovered population prior");
  close(model.correctedProbability(0.9, s1, s0), 1 / 12, "90% becomes 8.33%");
  for (const p of [0, 1e-6, 0.01, 0.25, 0.5, 0.99, 1]) {
    close(model.correctedProbability(model.sampledProbability(p, s1, s0), s1, s0), p, "Round trip");
  }
  close(model.logit(0.5) + Math.log(s0 / s1), model.logit(0.01), "Intercept shift");
  assert.equal(model.sigmoid(Infinity), 1);
  assert.equal(model.sigmoid(-Infinity), 0);
  close(model.sigmoid(-1000), 0, "Numerically stable sigmoid");
  close(model.entropy(0.5), Math.log(2), "Entropy uses nats");
  assert.equal(model.entropy(0), 0);
  assert.equal(model.entropy(1), 0);

  for (const name of ["A", "B"]) {
    close(model.SIGNALS[name].tpr, tree.splits[name].positiveLeft, "Shared positive routing");
    close(model.SIGNALS[name].fpr, tree.splits[name].negativeLeft, "Shared negative routing");
  }

  const aNatural = model.stump("A", 0.01);
  const bNatural = model.stump("B", 0.01);
  const aBalanced = model.stump("A", 0.5);
  const bBalanced = model.stump("B", 0.5);
  assert.ok(aNatural.logloss < bNatural.logloss, "Population log loss chooses A");
  assert.ok(bBalanced.logloss < aBalanced.logloss, "Balanced log loss chooses B");
  assert.equal(model.bestStump(0.01).name, "A");
  assert.equal(model.bestStump(0.5).name, "B");
  close(aNatural.auc, 0.75, "A two-level AUC");
  close(bNatural.auc, 0.9, "B two-level AUC");
  close(aNatural.precision, 1, "A pure positive leaf");
  close(bBalanced.precision, 1 / 12, "B population precision");
  assert.ok(bBalanced.highProb > bBalanced.lowProb);
  const correctedHigh = model.correctedProbability(bBalanced.highProb, s1, s0);
  const correctedLow = model.correctedProbability(bBalanced.lowProb, s1, s0);
  assert.ok(correctedHigh > correctedLow, "Prior correction preserves ranking");
  close(correctedHigh, bNatural.highProb, "Corrected high leaf");
  close(correctedLow, bNatural.lowProb, "Corrected low leaf");
  close(bBalanced.populationLogloss, bNatural.logloss, "Population loss after correction");

  // These are exact conditional entropies, independently recomputed from joint
  // class/branch probabilities, not fitted or simulated estimates.
  function independentLoss(signal, prior) {
    const joint = [prior * signal.tpr, (1 - prior) * signal.fpr,
      prior * (1 - signal.tpr), (1 - prior) * (1 - signal.fpr)];
    let loss = 0;
    for (let branch = 0; branch < 2; branch++) {
      const positive = joint[2 * branch];
      const negative = joint[2 * branch + 1];
      const mass = positive + negative;
      for (const part of [positive, negative]) {
        if (part > 0) loss -= part * Math.log(part / mass);
      }
    }
    return loss;
  }
  for (const prior of [0.01, 0.1, 0.5]) {
    for (const name of ["A", "B"]) {
      close(model.stump(name, prior).logloss, independentLoss(model.SIGNALS[name], prior), "Joint loss calculation");
    }
  }

  const eff = model.efficiency();
  assert.equal(eff.minorityCount, 10000);
  assert.equal(eff.majorityCount, 990000);
  assert.equal(eff.balancedDatasetSize, 20000);
  assert.equal(eff.datasetReductionFactor, 50);
  close(eff.expectedMinorityInBatch, 1.28, "Natural expected rare cases per batch");
  close(eff.zeroMinorityProbability, 0.99 ** 128, "Natural empty rare batch probability");
  close(eff.balancedZeroMinorityProbability, 0.5 ** 128, "Balanced empty rare batch probability");
  assert.equal(eff.balanced.stratifiedMinorityInBatch, 64);
  assert.equal(eff.balanced.stratifiedZeroMinorityProbability, 0);
  close(model.efficiency(0.01, 1000000, 32).zeroMinorityProbability, 0.7249803359578534, "32-case empty rare batch probability");
  assert.throws(() => model.correctedProbability(0.5, 1, 0), RangeError);
  assert.throws(() => model.efficiency(0), RangeError);
  assert.throws(() => model.stump("C", 0.5), TypeError);
});
