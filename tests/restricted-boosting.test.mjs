import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { runInNewContext } from 'node:vm';
import test from 'node:test';

const require = createRequire(import.meta.url);
const math = require('../docs/shared/boosting-math.js');
const view = require('../docs/boosting/restricted-view.js');
const close = (actual, expected, tolerance = 1e-9) => assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} should equal ${expected}`);
const vectorClose = (actual, expected, tolerance) => {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => close(value, expected[index], tolerance));
};
const attributes = fragment => Object.fromEntries([...fragment.matchAll(/([\w-]+)="([^"]*)"/g)].map(match => [match[1], match[2]]));
const dataValue = (html, name) => new RegExp(`data-${name}="([^"]+)"`).exec(html)?.[1];

// Independent definitions evaluate the displayed objective directly from y and
// prediction coordinates, rather than using the view's own loss evaluator.
function selectedLoss(prediction, state) {
  const residuals = [1 - prediction[0], -prediction[1]];
  if (state.loss === 'quartic') return residuals.reduce((sum, e) => sum + e ** 4 / 4, 0);
  if (state.loss === 'expectile') {
    return residuals.reduce((sum, e) => sum + (e >= 0 ? state.tau : 1 - state.tau) * e ** 2, 0);
  }
  if (state.loss === 'huber') {
    return residuals.reduce((sum, e) => {
      const magnitude = Math.abs(e);
      const quadraticPart = Math.min(magnitude, state.delta);
      return sum + quadraticPart ** 2 / 2 + state.delta * Math.max(0, magnitude - state.delta);
    }, 0);
  }
  return residuals.reduce((sum, e) => sum + e ** 2 / 2, 0);
}

const allLosses = [
  { loss: 'squared' },
  ...[.05, .2, .5, .8, .95].map(tau => ({ loss: 'expectile', tau })),
  ...[.05, .25, .49, .5, .75, 1, 2].map(delta => ({ loss: 'huber', delta })),
  { loss: 'quartic' }
];

test('the opening reveals the shared-prediction constraint before the optimum or new terminology', () => {
  const q = view.resolve();
  vectorClose(q.x, [1, 1]);
  vectorClose(q.target, [1, 0]);
  vectorClose(q.prediction, [0, 0]);
  assert.equal(q.lossSpec.key, 'squared');
  close(q.loss, .5);
  const html = view.render();
  assert.match(html, /class="restricted-available-line"/);
  assert.match(html, /observed y = \(1, 0\)/);
  assert.doesNotMatch(html, /gradient|∇|class="restricted-optimum"|class="restricted-residual-arrow"|\(0\.5, 0\.5\)/i);
  assert.doesNotMatch(view.presentation().caption, /gradient|∇/i);
});

test('the residual animation moves together from zero to the mean and decreases MSE', () => {
  let lastLoss = Infinity;
  for (let step = 0; step <= 20; step++) {
    const progress = step / 20, c = progress / 2;
    const state = { progress, prediction: .99, loss: 'quartic', width: 700, height: 440 };
    const q = view.resolve(1, state);
    vectorClose(q.prediction, [c, c]);
    vectorClose(q.residual, [1 - c, -c]);
    close(q.residualMean, .5 - c);
    close(q.loss, ((1 - c) ** 2 + c ** 2) / 2);
    assert.ok(q.loss <= lastLoss);
    lastLoss = q.loss;
    const html = view.render(1, state);
    const svg = attributes(/<svg\b[^>]*>/.exec(html)[0]);
    const residualArrow = attributes(/<line\b[^>]*class="restricted-residual-arrow"[^>]*>/.exec(html)[0]);
    const scale = Number(svg['data-scale']);
    vectorClose([(Number(residualArrow.x2) - Number(residualArrow.x1)) / scale, (Number(residualArrow.y1) - Number(residualArrow.y2)) / scale], q.residual);
    assert.match(html, /class="restricted-progress-path"/);
    assert.doesNotMatch(html, /gradient|∇/i);
  }
  close(lastLoss, .25);
  vectorClose(view.resolve(1, { progress: -10 }).prediction, [0, 0]);
  vectorClose(view.resolve(1, { progress: 10 }).prediction, [.5, .5]);
});

test('the residual progression explicitly labels its current values and the fitted destination', () => {
  for (const [width, height, compact] of [[273, 300, true], [450, 410, false]]) {
    for (const [progress, prediction, residual] of [
      [0, '(0, 0)', '(1, 0)'],
      [.5, '(0.25, 0.25)', '(0.75, −0.25)'],
      [1, '(0.5, 0.5)', '(0.5, −0.5)']
    ]) {
      const state = { progress, width, height }, html = view.render(1, state, compact);
      const labels = [...html.matchAll(/<text\b([^>]*)>([^<]*)<\/text>/g)].map(match => ({ ...attributes(match[1]), text: match[2] }));
      const predictions = labels.filter(label => label.class.includes('restricted-prediction-label'));
      const residuals = labels.filter(label => label.class.includes('restricted-residual-label'));
      assert.equal(predictions.length, 1, 'Only one label should identify the actual prediction');
      assert.equal(residuals.length, 1);
      assert.equal(predictions[0].text, `f = ${prediction}`);
      assert.equal(residuals[0].text, `r = ${residual}`);
      if (progress < 1) assert.ok(labels.some(label => label.class.includes('restricted-destination-label') && label.text === 'fit: (0.5, 0.5)'));
      else assert.ok(labels.every(label => !label.class.includes('restricted-destination-label')));
      for (const label of [...predictions, ...residuals]) {
        assert.ok(Number(label.x) > 0 && Number(label.x) < width);
        assert.ok(Number(label.y) > 0 && Number(label.y) < height);
      }
      const caption = view.presentation(1, state).caption;
      assert.ok(caption.includes(`Current predictions f = ${prediction}; residuals r = y − f = ${residual}.`));
      assert.match(caption, /starting residual \(1, 0\) has mean 0\.5, so its fitted correction is \(0\.5, 0\.5\)/);
      if (progress === 0) {
        const targetLabel = labels.find(label => label.class.includes('restricted-target-label'));
        assert.ok(Number(targetLabel.y) - Number(residuals[0].y) >= 22, 'Residual and observed-outcome labels need separate vertical space');
        const arrow = attributes(/<line\b[^>]*class="restricted-residual-arrow"[^>]*>/.exec(html)[0]);
        close(Number(arrow.y1), Number(arrow.y2));
        assert.equal(arrow['stroke-width'], '3');
        assert.equal(arrow['stroke-dasharray'], undefined, 'The initial residual stays visible on the x axis');
      }
    }
  }
});

test('orthogonality is visible and the MSE contour is tangent to the constraint', () => {
  const q = view.resolve(2, { progress: 0, prediction: 0, loss: 'expectile' });
  vectorClose(q.prediction, [.5, .5]);
  vectorClose(q.residual, [.5, -.5]);
  close(math.dot(q.residual, [1, 1]), 0);
  close(q.residualMean, 0);
  close(q.loss, .25);
  const html = view.render(2, { width: 900, height: 430 });
  assert.match(html, /class="restricted-right-angle" data-dot-product="0"/);
  assert.match(html, /90°/);
  assert.match(html, /Equal-MSE contour touches the diagonal/);
  assert.doesNotMatch(html, /gradient|∇/i);
  // Moving either way along the diagonal increases the actual average error.
  for (const shift of [-.1, -.001, .001, .1]) assert.ok(selectedLoss([.5 + shift, .5 + shift], { loss: 'squared' }) > q.loss);
});

test('changed loss establishes the counterexample before gradients are introduced', () => {
  for (const progress of [0, .25, .5, .75, 1]) {
    const q = view.resolve(3, { progress, tau: .2, loss: 'squared' });
    vectorClose(q.prediction, [.5 + .3 * progress, .5 + .3 * progress]);
    assert.equal(q.lossSpec.key, 'expectile');
    close(q.lossSpec.tau, .8);
    close(q.rawNextLoss, .25);
    close(q.minLoss, .16);
    assert.ok(q.loss <= .25 + 1e-12);
    assert.doesNotMatch(view.render(3, { progress }), /gradient|∇/i);
    assert.doesNotMatch(JSON.stringify(view.presentation(3, { progress })), /gradient|∇/i);
  }
  const firstGradient = view.resolve(4);
  vectorClose(firstGradient.prediction, [.5, .5]);
  vectorClose(firstGradient.residual, [.5, -.5]);
  close(firstGradient.residualMean, 0);
  vectorClose(firstGradient.negativeGradient, [.8, -.2]);
  close(firstGradient.negativeGradientMean, .3);
  vectorClose(firstGradient.nextTree, [.3, .3]);
  assert.match(view.render(4), /class="restricted-gradient-arrow"/);
  assert.match(view.presentation(4).caption, /negative-gradient mean is 0\.3/);
  const lastGradient = view.resolve(4, { progress: 1 });
  close(lastGradient.negativeGradientMean, 0);
  close(math.dot(lastGradient.negativeGradient, [1, 1]), 0);
  close(lastGradient.residualMean, -.3);
});

test('early equations display the exact two-point MSE and both axes retain half ticks', () => {
  for (const scene of [0, 1, 2]) {
    const html = view.equation({ loss: 'quartic' }, scene);
    assert.match(html, /MSE\(f₁,f₂\) = ½\[\(1 − f₁\)² \+ f₂²\]/);
    assert.match(html, /MSE\(c,c\) = ½\[\(1 − c\)² \+ c²\]/);
    assert.doesNotMatch(html, /ℓ|gradient|∇/i);
  }
  for (let scene = 0; scene <= 5; scene++) {
    const html = view.render(scene);
    assert.equal([...html.matchAll(/>½<\/text>/g)].length, 2);
    assert.match(html, new RegExp(`<title id="restricted-title">${view.topics[scene]}</title>`));
  }
  assert.throws(() => view.resolve(-1), RangeError);
  assert.throws(() => view.resolve(6), RangeError);
});

test('the loss explorer uses the requested observations and asymmetric squared loss', () => {
  const q = view.resolve(5);
  vectorClose(q.x, [1, 1]);
  vectorClose(q.target, [1, 0]);
  vectorClose(q.prediction, [0, 0]);
  vectorClose(q.residual, [1, 0]);
  assert.equal(q.lossSpec.key, 'expectile');
  close(q.lossSpec.tau, .8);
  vectorClose(q.negativeGradient, [1.6, 0]);
  vectorClose(q.gradient, [-1.6, 0]);
  vectorClose(q.nextTree, [.8, .8]);
  vectorClose(q.correction, [.8, .8]);
  vectorClose(q.nextPrediction, [.8, .8]);
  vectorClose(q.gradientNextPrediction, [.8, .8]);
  vectorClose(q.rawNextPrediction, [.5, .5]);
  close(q.loss, .8);
  close(q.nextLoss, .16);
  close(q.rawNextLoss, .25);
  assert.equal(q.tree.threshold, null, 'Identical inputs cannot be separated by a tree split');
});

test('two-point MSE and symmetric asymmetric squared loss have identical updates and losses', () => {
  const cases = [
    [0, [1, 0], [.5, .5], .5, 1],
    [.25, [.75, -.25], [.25, .25], .3125, Math.sqrt(.625)],
    [.5, [.5, -.5], [0, 0], .25, Math.sqrt(.5)],
    [1, [0, -1], [-.5, -.5], .5, 1]
  ];
  for (const [c, residual, correction, loss, norm] of cases) {
    for (const state of [{ loss: 'squared' }, { loss: 'expectile', tau: .5 }, { loss: 'huber', delta: 2 }]) {
      const q = view.resolve(5, { ...state, prediction: c });
      vectorClose(q.residual, residual);
      vectorClose(q.negativeGradient, residual);
      vectorClose(q.nextTree, correction);
      vectorClose(q.correction, correction);
      vectorClose(q.nextPrediction, [.5, .5]);
      vectorClose(q.optimalPrediction, [.5, .5]);
      close(q.loss, loss);
      close(q.norm, norm);
      close(q.nextLoss, .25);
      close(q.minLoss, .25);
    }
  }
});

test('asymmetry changes the diagonal optimum and the fitted gradient at every shared prediction', () => {
  for (const tau of [.05, .2, .5, .8, .95]) {
    for (let index = 0; index <= 100; index++) {
      const c = index / 100;
      const q = view.resolve(5, { loss: 'expectile', tau, prediction: c });
      // On [0,1], the two observations have opposite residual signs. Completing
      // the square gives L(c,c) = (c - tau)^2 + tau(1 - tau).
      close(q.loss, (c - tau) ** 2 + tau * (1 - tau));
      vectorClose(q.negativeGradient, [2 * tau * (1 - c), -2 * (1 - tau) * c]);
      vectorClose(q.nextTree, [tau - c, tau - c]);
      vectorClose(q.correction, [tau - c, tau - c]);
      vectorClose(q.nextPrediction, [tau, tau]);
      vectorClose(q.gradientNextPrediction, [tau, tau]);
      close(q.minLoss, tau * (1 - tau));
      close(q.nextLoss, q.minLoss);
      close(q.rawNextLoss, .25);
      close(q.loss - q.nextLoss, (c - tau) ** 2);
      assert.equal(q.tree.threshold, null);
    }
  }
});

test('zero mean residual can hide a needed gradient update; a zero fitted gradient can leave residuals', () => {
  const atMean = view.resolve(5, { loss: 'expectile', tau: .8, prediction: .5 });
  vectorClose(atMean.residual, [.5, -.5]);
  close(math.dot(atMean.residual, [1, 1]), 0);
  vectorClose(atMean.negativeGradient, [.8, -.2]);
  vectorClose(atMean.nextTree, [.3, .3]);
  close(atMean.loss, .25);
  close(atMean.nextLoss, .16);

  const atOptimum = view.resolve(5, { loss: 'expectile', tau: .8, prediction: .8 });
  vectorClose(atOptimum.negativeGradient, [.32, -.32]);
  close(math.dot(atOptimum.negativeGradient, [1, 1]), 0);
  vectorClose(atOptimum.nextTree, [0, 0]);
  vectorClose(atOptimum.correction, [0, 0]);
  close(math.dot(atOptimum.residual, [1, 1]), -.6);
  close(atOptimum.loss, .16);
  close(atOptimum.rawNextLoss, .25);
  assert.match(view.presentation(5, { loss: 'expectile', tau: .8, prediction: .8 }).caption, /gradient.*perpendicular.*fitted gradient is zero/i);
});

test('Huber flat minima choose the closest optimum and distinguish gradient targets from leaf corrections', () => {
  const cases = [
    [.25, 0, [.25, 0], .125, .25, .21875, .1875],
    [.25, .1, [.25, -.1], .075, .25, .19875, .1875],
    [.25, .3, [.25, -.25], 0, .3, .1875, .1875],
    [.25, .5, [.25, -.25], 0, .5, .1875, .1875],
    [.25, .9, [.1, -.25], -.075, .75, .19875, .1875],
    [.75, 0, [.75, 0], .375, .5, .46875, .25]
  ];
  for (const [delta, c, gradient, sharedTarget, nextC, loss, minLoss] of cases) {
    const q = view.resolve(5, { loss: 'huber', delta, prediction: c });
    vectorClose(q.negativeGradient, gradient);
    vectorClose(q.nextTree, [sharedTarget, sharedTarget]);
    vectorClose(q.correction, [nextC - c, nextC - c]);
    vectorClose(q.nextPrediction, [nextC, nextC]);
    close(q.loss, loss);
    close(q.minLoss, minLoss);
    close(q.nextLoss, minLoss);
  }
  for (const delta of [.05, .25, .49]) {
    const state = { loss: 'huber', delta };
    const q = view.resolve(5, state);
    vectorClose(q.optimumInterval, [delta, 1 - delta]);
    close(q.minLoss, delta - delta ** 2);
    for (const c of [delta, .5, 1 - delta]) {
      const inPlateau = view.resolve(5, { ...state, prediction: c });
      close(inPlateau.loss, q.minLoss);
      vectorClose(inPlateau.correction, [0, 0]);
      vectorClose(inPlateau.nextTree, [0, 0]);
    }
    for (const c of [0, delta / 2, 1 - delta / 2, 1]) {
      const outsidePlateau = view.resolve(5, { ...state, prediction: c });
      assert.ok(outsidePlateau.loss > q.minLoss);
    }
  }
});

test('fourth-power gradients differ from raw residuals and exact leaf corrections away from the origin', () => {
  const cases = [
    [0, [1, 0], .5, .5, .25],
    [.25, [27 / 64, -1 / 64], 13 / 64, .25, 41 / 512],
    [.5, [1 / 8, -1 / 8], 0, 0, 1 / 32],
    [.75, [1 / 64, -27 / 64], -13 / 64, -.25, 41 / 512],
    [1, [0, -1], -.5, -.5, .25]
  ];
  for (const [c, gradient, sharedTarget, correction, loss] of cases) {
    const q = view.resolve(5, { loss: 'quartic', prediction: c });
    vectorClose(q.negativeGradient, gradient);
    vectorClose(q.nextTree, [sharedTarget, sharedTarget]);
    vectorClose(q.correction, [correction, correction]);
    vectorClose(q.gradientNextPrediction, [c + sharedTarget, c + sharedTarget]);
    vectorClose(q.nextPrediction, [.5, .5]);
    close(q.loss, loss);
    close(q.minLoss, 1 / 32);
    close(q.nextLoss, 1 / 32);
  }
});

test('negative gradients match derivatives of the selected total loss in both coordinates', () => {
  const epsilon = 1e-6;
  for (const state of allLosses) {
    for (const c of [0, .13, .25, .5, .8, 1]) {
      const q = view.resolve(5, { ...state, prediction: c });
      const numerical = q.prediction.map((_, index) => {
        const plus = q.prediction.slice(), minus = q.prediction.slice();
        plus[index] += epsilon;
        minus[index] -= epsilon;
        return -(selectedLoss(plus, state) - selectedLoss(minus, state)) / (2 * epsilon);
      });
      vectorClose(q.negativeGradient, numerical, 1e-6);
      close(q.loss, selectedLoss(q.prediction, state));
      close(q.nextLoss, selectedLoss(q.nextPrediction, state));
      close(q.rawNextLoss, selectedLoss(q.rawNextPrediction, state));
      assert.ok(q.nextLoss <= q.loss + 1e-12);
    }
  }
});

test('every plotted contour sample has its labeled selected-loss value at responsive sizes', () => {
  for (const [width, height] of [[288, 260], [346, 300], [640, 450], [700, 260]]) {
    for (const state of allLosses) {
      for (const c of [0, .25, .5, .8, 1]) {
        const html = view.render(5, { ...state, prediction: c, width, height }, width < 400);
        assert.doesNotMatch(html, /NaN|Infinity|undefined/);
        const svg = attributes(/<svg\b[^>]*>/.exec(html)[0]);
        const scale = Number(svg['data-scale']);
        assert.ok(scale > 0);
        close(scale * Number(svg['data-units-per-pixel']), 1);
        assert.equal(svg['data-x-domain'], svg['data-y-domain']);
        const current = attributes(/<circle\b[^>]*class="restricted-current"[^>]*>/.exec(html)[0]);
        const currentPixel = [Number(current.cx), Number(current.cy)];
        const contours = [...html.matchAll(/<polyline\b[^>]*>/g)].map(match => attributes(match[0]));
        assert.ok(contours.length >= 1);
        for (const contour of contours) {
          const level = Number(contour['data-loss']);
          const points = contour.points.split(' ').map(point => point.split(',').map(Number));
          assert.equal(points.length, 193);
          for (const [px, py] of points) {
            const prediction = [c + (px - currentPixel[0]) / scale, c - (py - currentPixel[1]) / scale];
            close(selectedLoss(prediction, state), level, 1e-8);
          }
        }
        const diagonal = attributes(/<line\b[^>]*class="restricted-available-line"[^>]*>/.exec(html)[0]);
        close(Number(diagonal.x2) - Number(diagonal.x1), Number(diagonal.y1) - Number(diagonal.y2));
        const optimum = attributes(/<rect\b[^>]*class="restricted-optimum"[^>]*>/.exec(html)[0]);
        close(selectedLoss(optimum['data-values'].split(',').map(Number), state), Number(dataValue(html, 'min-loss')));
      }
    }
  }
});

test('residual and gradient arrows preserve direction with the disclosed half-length scale', () => {
  for (const [width, height] of [[288, 260], [640, 450]]) {
    for (const state of allLosses) {
      for (const c of [0, .25, .5, .8, 1]) {
        const q = view.resolve(5, { ...state, prediction: c });
        const html = view.render(5, { ...state, prediction: c, width, height }, width < 400);
        const svg = attributes(/<svg\b[^>]*>/.exec(html)[0]);
        const scale = Number(svg['data-scale']);
        close(Number(svg['data-arrow-scale']), .5);
        assert.match(html, /arrows are drawn at half their numerical lengths/);
        for (const [className, values, factor] of [
          ['restricted-residual-arrow', q.residual, .5],
          ['restricted-gradient-arrow', q.negativeGradient, .5],
          ['restricted-tree-arrow', q.correction, 1]
        ]) {
          const match = new RegExp(`<line\\b[^>]*class="${className}"[^>]*>`).exec(html);
          if (!match) {
            assert.ok(math.norm(values) * scale * factor < .25, 'Only imperceptibly short arrows may be omitted');
            continue;
          }
          const arrow = attributes(match[0]);
          vectorClose([
            (Number(arrow.x2) - Number(arrow.x1)) / scale,
            (Number(arrow.y1) - Number(arrow.y2)) / scale
          ], values.map(value => value * factor));
        }
      }
    }
  }
});

test('rendered numerical data describes the current selected-loss calculation', () => {
  for (const state of allLosses) {
    const q = view.resolve(5, { ...state, prediction: .37 });
    const html = view.render(5, { ...state, prediction: .37 });
    assert.equal(dataValue(html, 'loss-key'), state.loss);
    for (const [attribute, expected] of [
      ['target', q.target], ['prediction', q.prediction], ['residual', q.residual],
      ['gradient', q.gradient], ['negative-gradient', q.negativeGradient], ['next-tree', q.nextTree],
      ['correction', q.correction], ['next-prediction', q.nextPrediction], ['optimum-interval', q.optimumInterval]
    ]) vectorClose(JSON.parse(dataValue(html, attribute)), expected);
    close(Number(dataValue(html, 'loss')), q.loss);
    close(Number(dataValue(html, 'min-loss')), q.minLoss);
    close(Number(dataValue(html, 'next-loss')), q.nextLoss);
    assert.ok(view.readout(5, state).every(row => row.length >= 2 && typeof row[0] === 'string'));
  }
});

test('Huber captions describe selected loss and never claim all Huber gradients are raw residuals', () => {
  const state = { loss: 'huber', delta: .75, prediction: 0 };
  const caption = view.presentation(5, state).caption;
  assert.doesNotMatch(caption, /residual and negative gradient coincide|half-squared loss/i);
  assert.match(caption, /Huber|clipp|gradient/i);
  assert.match(view.presentation(5, { loss: 'huber', delta: .25 }).caption, /Every c in \[0\.25, 0\.75\]/);
});

test('the restricted-input visual runs without a module loader', () => {
  const context = { window: { BoostingMath: math } };
  runInNewContext(readFileSync(new URL('../docs/boosting/restricted-view.js', import.meta.url), 'utf8'), context);
  assert.equal(typeof context.window.RestrictedBoostingView.render, 'function');
  assert.equal(typeof context.window.RestrictedBoostingView.resolve, 'function');
  const q = context.window.RestrictedBoostingView.resolve(5, { prediction: .8 });
  vectorClose(q.target, [1, 0]);
  vectorClose(q.prediction, [.8, .8]);
  vectorClose(q.nextTree, [0, 0]);
  close(q.loss, .16);
});
