import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const E = require('../docs/boosting/loss-example.js');
const V = require('../docs/boosting/loss-view.js');

test('split comparisons use the same predictions, including the last selected round', () => {
  const options = { loss: 'huber', dataset: 'outlier', rate: .5, round: 24, width: 288, height: 220 };
  const model = E.build({ ...options, rounds: 25 });
  const rendered = V.render(1, options, true);
  const candidates = E.candidates(model.gradient[24].residual, model.x);
  assert.equal((rendered.match(/data-cut=/g) || []).length, 6);
  candidates.forEach(candidate => assert.ok(rendered.includes(`data-residual-sse="${candidate.sse}"`)));
  assert.match(V.presentation(1, options).caption, /From the same F₂₄/);
});

test('the default Huber example makes both split and leaf calibration differences explicit', () => {
  const cutReadout = V.readout(1);
  assert.equal(cutReadout[0][1], 'x = 3.5');
  assert.equal(cutReadout[1][1], 'x = 6.5');
  const corrections = V.readout(2);
  assert.equal(corrections[0][1], '−1; +0.75');
  assert.equal(corrections[1][1], '−1.667; +1.5');
  assert.match(V.render(2), /Mean negative gradient −1, original-loss total there 1/);
});

test('identical inputs remain a single fitted region in all four scenes', () => {
  const options = { loss: 'huber', dataset: 'shared', round: 1, width: 288, height: 220 };
  for (const scene of [0, 1, 2, 3]) {
    const rendered = V.render(scene, options, true);
    assert.doesNotMatch(rendered, /NaN|Infinity|undefined/);
    assert.doesNotThrow(() => V.presentation(scene, options));
  }
  assert.match(V.render(1, options, true), /no split is available/);
  assert.match(V.render(3, options, true), /Horizontal positions are observation indices/);
  assert.match(V.presentation(2, options).caption, /residual-mean correction/);
});

test('the shared-region raw mean marker stays inside the original-loss curve domain', () => {
  const rendered = V.render(2, { loss: 'huber', dataset: 'shared', round: 0, width: 266, height: 210 }, true);
  const match = rendered.match(/viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)"/);
  const width = Number(match[1]), height = Number(match[2]);
  for (const element of rendered.matchAll(/(?:x1|x2|cx|cy)="([^\"]+)"/g)) {
    const value = Number(element[1]);
    assert.ok(value >= 0 && value <= Math.max(width, height), `Invalid SVG coordinate ${value}`);
  }
  assert.match(rendered, /Raw residual mean 2\.83333333333/);
});

test('expectile descriptions use the derivative of the chosen loss normalization', () => {
  const caption = V.presentation(0, { loss: 'expectile', tau: .8 }).caption;
  assert.match(caption, /positive residuals by 1\.6/);
  assert.match(caption, /negative residuals by 0\.4/);
});

test('all controls render finite numerical descriptions at narrow and wide sizes', () => {
  for (const dataset of ['worked', 'outlier', 'shared']) {
    for (const loss of ['squared', 'huber', 'expectile']) {
      for (const round of [0, 1, 12, 24]) {
        for (const width of [266, 620]) {
          for (const scene of [0, 1, 2, 3]) {
            const options = { dataset, loss, round, width, height: 252 };
            const rendered = V.render(scene, options, width < 440);
            assert.doesNotMatch(rendered, /NaN|Infinity|undefined/);
            if (scene !== 1) assert.match(rendered, /<desc id="loss-plot-description">/);
            assert.ok(V.presentation(scene, options).caption.length > 0);
          }
        }
      }
    }
  }
});

test('round controls clamp to valid inspection states and scene identifiers validate', () => {
  assert.match(V.render(0, { round: 100 }), /data-round="24"/);
  assert.match(V.render(0, { round: -1 }), /data-round="0"/);
  for (const invalid of [-1, 4, .5]) assert.throws(() => V.render(invalid), RangeError);
});
