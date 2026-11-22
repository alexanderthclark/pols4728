import test from 'node:test';
import assert from 'node:assert/strict';
import { MATCH_SURVIVAL, MISMATCH_SURVIVAL, survivalProbability, toyState, renderBayes, renderBayesMobile } from './bayes.mjs';

const near = (actual, expected, tolerance = 1e-12) => assert.ok(Math.abs(actual - expected) < tolerance, `${actual} != ${expected}`);

test('two hypothetical worlds map both species and strategies to the specified probabilities', () => {
  const p = 1 / (1 + Math.exp(-1));
  const q = 1 - p;
  near(MATCH_SURVIVAL, p);
  near(MISMATCH_SURVIVAL, q);
  for (const [brown, lieDown, expectedA] of [[0, 0, p], [0, 1, q], [1, 0, q], [1, 1, p]]) {
    near(survivalProbability('A', brown, lieDown), expectedA);
    near(survivalProbability('B', brown, lieDown), 1 - expectedA);
  }
  near(survivalProbability('A', true, true), p);
  near(survivalProbability('A', false, false), p);
});

test('posterior odds multiply by exp(1) for each Brown + Lie down survivor', () => {
  for (const n of [0, 1, 2, 4]) {
    const posterior = 1 / (1 + Math.exp(-n));
    const prediction = MISMATCH_SURVIVAL + (MATCH_SURVIVAL - MISMATCH_SURVIVAL) * posterior;
    const s = toyState(n);
    near(s.posterior[0], posterior);
    near(s.posterior[0] + s.posterior[1], 1);
    near(s.prediction, prediction);
    near(s.prediction, s.matchPrediction);
    near(s.mismatchPrediction + s.matchPrediction, 1);
    near(s.posterior[0] / s.posterior[1], Math.exp(n));
    near(s.logLikelihoodRatio, n);
  }
});

test('changing the prior changes predictions without changing likelihoods', () => {
  const equal = toyState(1);
  const skeptical = toyState(1, [MISMATCH_SURVIVAL, MATCH_SURVIVAL]);
  near(skeptical.posterior[0], .5);
  near(skeptical.prediction, .5);
  assert.deepEqual(equal.likelihood, skeptical.likelihood);
  near(toyState(1, [MISMATCH_SURVIVAL * 10, MATCH_SURVIVAL * 10]).prediction, skeptical.prediction);
});

test('zero-support hypotheses stay at zero; large samples do not underflow', () => {
  near(toyState(1000).prediction, MATCH_SURVIVAL);
  near(toyState(1000, [0, 1]).prediction, MISMATCH_SURVIVAL);
  near(toyState(1000, [1, 0]).prediction, MATCH_SURVIVAL);
  near(toyState(1000).posterior[0], 1);
  assert.ok(toyState(1000).prediction < 1, 'Certainty about the world does not remove outcome randomness.');
});

test('invalid mathematical inputs are rejected', () => {
  for (const n of [-1, .5, Infinity, NaN]) assert.throws(() => toyState(n), RangeError);
  for (const p of [[0, 0], [-1, 2], [NaN, 1], [1], [Infinity, 1]]) assert.throws(() => toyState(1, p), RangeError);
  assert.throws(() => survivalProbability('C', 0, 0), RangeError);
  assert.throws(() => survivalProbability('A', 2, 0), RangeError);
});

test('scroll progression changes bar geometry and prediction and is reversible', () => {
  const before = renderBayes(2, 0);
  const middle = renderBayes(2, .5);
  const after = renderBayes(2, 1);
  const p = s => Number(s.match(/data-prediction="([^"]+)"/)[1]);
  near(p(before), toyState(1).prediction);
  near(p(after), toyState(2).prediction);
  assert.ok(p(before) < p(middle) && p(middle) < p(after));
  assert.notEqual(before, middle);
  assert.notEqual(middle, after);
  assert.equal(renderBayes(2, 0), before);
  assert.match(renderBayes(4, 1), /98\.2% × 73\.1% \+ 1\.8% × 26\.9% ≈ 72\.3%/);
});

test('mobile and desktop render identical numerical states throughout the scroll', () => {
  const attr = (s, name) => Number(s.match(new RegExp(`${name}="([^"]+)"`))[1]);
  for (let stage = 0; stage < 5; stage++) {
    for (const progress of [0, .25, .5, .75, 1]) {
      const desktop = renderBayes(stage, progress);
      const mobile = renderBayesMobile(stage, progress);
      near(attr(desktop, 'data-posterior-a'), attr(mobile, 'data-posterior-a'));
      near(attr(desktop, 'data-prediction'), attr(mobile, 'data-prediction'));
      assert.match(mobile, /data-bayes-mobile="true"/);
      assert.match(mobile, /Brown \+ Lie down/);
      assert.doesNotMatch(mobile, /vot|turnout|high-interest/i);
      assert.doesNotMatch(desktop, /vot|turnout|high-interest/i);
      const sizes = [...mobile.matchAll(/font-size="(\d+)"/g)].map(m => Number(m[1]));
      assert.ok(sizes.every(size => size >= 15));
    }
  }
});
