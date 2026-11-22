import test from 'node:test';
import assert from 'node:assert/strict';
import { toyState, renderBayes, renderBayesMobile } from './bayes.mjs';

const near = (actual, expected, tolerance = 1e-12) => assert.ok(Math.abs(actual - expected) < tolerance, `${actual} != ${expected}`);

test('posterior odds multiply by four for each high-interest voter who voted', () => {
  const expected = [[0, .5, .5], [1, .8, .68], [2, 16/17, .2 + .6 * 16/17], [4, 256/257, .2 + .6 * 256/257]];
  for (const [n, posterior, prediction] of expected) {
    const s = toyState(n);
    near(s.posterior[0], posterior);
    near(s.posterior[0] + s.posterior[1], 1);
    near(s.prediction, prediction);
    near(s.lowPrediction + s.highPrediction, 1);
    near(s.posterior[0] / s.posterior[1], 4 ** n);
  }
});

test('changing the prior changes predictions without changing likelihoods', () => {
  const equal = toyState(1);
  const skeptical = toyState(1, [.2, .8]);
  near(skeptical.posterior[0], .5);
  near(skeptical.prediction, .5);
  assert.deepEqual(equal.likelihood, skeptical.likelihood);
  near(toyState(1, [2, 8]).prediction, skeptical.prediction);
});

test('zero-support hypotheses stay at zero; large samples do not underflow', () => {
  near(toyState(1000).prediction, .8);
  near(toyState(1000, [0, 1]).prediction, .2);
  near(toyState(1000, [1, 0]).prediction, .8);
});

test('invalid mathematical inputs are rejected', () => {
  for (const n of [-1, .5, Infinity, NaN]) assert.throws(() => toyState(n), RangeError);
  for (const p of [[0, 0], [-1, 2], [NaN, 1], [1], [Infinity, 1]]) assert.throws(() => toyState(1, p), RangeError);
});

test('scroll progression changes bar geometry and prediction and is reversible', () => {
  const before = renderBayes(2, 0);
  const middle = renderBayes(2, .5);
  const after = renderBayes(2, 1);
  const p = s => Number(s.match(/data-prediction="([^"]+)"/)[1]);
  near(p(before), .68);
  near(p(after), .2 + .6 * 16/17);
  assert.ok(p(before) < p(middle) && p(middle) < p(after));
  assert.notEqual(before, middle);
  assert.notEqual(middle, after);
  assert.equal(renderBayes(2, 0), before);
  assert.match(renderBayes(4, 1), /99\.6% × 80% \+ 0\.4% × 20% = 79\.8%/);
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
      const sizes = [...mobile.matchAll(/font-size="(\d+)"/g)].map(m => Number(m[1]));
      assert.ok(sizes.every(size => size >= 15));
    }
  }
});
