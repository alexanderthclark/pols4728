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

test('the opening reveals only outcomes and starting predictions', () => {
  const q = view.resolve();
  vectorClose(q.x, [1, 1]);
  vectorClose(q.target, [1, 0]);
  vectorClose(q.prediction, [0, 0]);
  assert.equal(q.lossSpec.key, 'squared');
  close(q.loss, .5);
  const html = view.render();
  assert.doesNotMatch(html, /class="restricted-available-line"/);
  assert.match(html, /observed y = \(1, 0\)/);
  assert.doesNotMatch(html, /gradient|∇|diagonal|correction|MSE|class="restricted-optimum"|class="restricted-residual-arrow"|\(0\.5, 0\.5\)/i);
  assert.doesNotMatch(view.presentation().caption, /gradient|∇/i);
});

test('the residual appears before any fitted update or restriction on the updates', () => {
  const state = { progress: 1, prediction: .9, loss: 'expectile', tau: .1 };
  const q = view.resolve(1, state);
  vectorClose(q.prediction, [0, 0]);
  vectorClose(q.residual, [1, 0]);
  for (const [width, height] of [[273, 300], [450, 410], [970, 430]]) {
    const html = view.render(1, { ...state, width, height }, width < 400);
    const svg = attributes(/<svg\b[^>]*>/.exec(html)[0]);
    const arrow = attributes(/<line\b[^>]*class="restricted-residual-arrow"[^>]*>/.exec(html)[0]);
    const current = attributes(/<circle\b[^>]*class="restricted-current"[^>]*>/.exec(html)[0]);
    const target = attributes(/<path\b[^>]*class="restricted-target"[^>]*>/.exec(html)[0]);
    close(Number(arrow.x1), Number(current.cx));
    close(Number(arrow.y1), Number(current.cy));
    close(Number(arrow.x2), Number(target['data-pixel-x']));
    close(Number(arrow.y2), Number(target['data-pixel-y']));
    close((Number(arrow.x2) - Number(arrow.x1)) / Number(svg['data-scale']), 1);
    close(Number(arrow.y2) - Number(arrow.y1), 0);
    assert.match(html, /restricted-residual-label[^>]*>r = \(1, 0\)<\/text>/);
    assert.match(html, /restricted-prediction-label[^>]*>f = \(0, 0\)<\/text>/);
    assert.doesNotMatch(html, /restricted-available-line|restricted-tree-arrow|restricted-optimum|restricted-.*contour|diagonal|45°|MSE|\(0\.5, 0\.5\)/);
  }
  assert.equal(view.equation(state, 0), '');
  assert.equal(view.equation(state, 1), '');
  assert.doesNotMatch(JSON.stringify(view.presentation(1, state)), /mean|diagonal|correction|45°|MSE|gradient|∇/i);
});

test('the identical-input restriction reveals a true 45-degree update line before fitting', () => {
  for (const [width, height] of [[273, 300], [450, 410], [970, 430]]) {
    const state = { progress: 1, prediction: .8, width, height };
    const q = view.resolve(2, state), html = view.render(2, state, width < 400);
    vectorClose(q.x, [1, 1]);
    vectorClose(q.prediction, [0, 0]);
    vectorClose(q.residual, [1, 0]);
    const svg = attributes(/<svg\b[^>]*>/.exec(html)[0]);
    assert.equal(svg['data-x-domain'], svg['data-y-domain']);
    const diagonal = attributes(/<line\b[^>]*class="restricted-available-line"[^>]*>/.exec(html)[0]);
    const horizontal = Number(diagonal.x2) - Number(diagonal.x1), vertical = Number(diagonal.y1) - Number(diagonal.y2);
    close(horizontal, vertical);
    close(Math.atan2(vertical, horizontal) * 180 / Math.PI, 45);
    const angle = attributes(/<path\b[^>]*class="restricted-update-angle"[^>]*>/.exec(html)[0]);
    assert.equal(angle['data-angle-degrees'], '45');
    assert.match(angle.d, /A[^ ]+ 0 0 0/);
    assert.match(html, /restricted-update-angle-label[^>]*>45°<\/text>/);
    assert.match(html, />h₁ = h₂<\/text>/);
    assert.match(view.presentation(2, state).caption, /x = 1/);
    assert.match(view.presentation(2, state).caption, /same correction/);
    assert.match(view.presentation(2, state).caption, /45° line/);
    assert.doesNotMatch(html, /restricted-tree-arrow|restricted-optimum|restricted-.*contour|MSE|\(0\.5, 0\.5\)/);
  }
  assert.equal(view.equation({}, 2), '');
  assert.match(view.annotation(2).status, /Same x → same correction h = \(a, a\)/);
  for (let scene = 0; scene <= 6; scene++) {
    const html = view.render(scene);
    assert.doesNotMatch(html, /gradient|∇/i);
    assert.doesNotMatch(JSON.stringify(view.presentation(scene)), /gradient|∇/i);
    if (scene < 3) assert.doesNotMatch(html, /data-(?:next|optimal|correction|min-loss|negative-gradient)/);
  }
  assert.match(view.render(7), /restricted-gradient-arrow/);
});

test('scroll reveal draws the residual and shared-update line without moving predictions', () => {
  for (const progress of [0, .25, .5, .75, 1]) {
    for (const scene of [1, 2]) {
      const state = { progress, width: 700, height: 520 }, q = view.resolve(scene, state), html = view.render(scene, state);
      vectorClose(q.prediction, [0, 0]);
      vectorClose(q.residual, [1, 0]);
      const svg = attributes(/<svg\b[^>]*>/.exec(html)[0]), scale = Number(svg['data-scale']);
      const arrow = attributes(/<line\b[^>]*class="restricted-residual-arrow"[^>]*>/.exec(html)[0]);
      close((Number(arrow.x2) - Number(arrow.x1)) / scale, scene === 1 ? progress : 1);
      close(Number(arrow.y1), Number(arrow.y2));
      if (scene === 1) close(Number(arrow.opacity), progress);
      else {
        const diagonal = attributes(/<line\b[^>]*class="restricted-available-line"[^>]*>/.exec(html)[0]);
        close((Number(diagonal.x2) - Number(diagonal.x1)) / scale, 1.6 * progress);
        close(Number(diagonal.opacity), progress);
        close(Number(diagonal.x2) - Number(diagonal.x1), Number(diagonal.y1) - Number(diagonal.y2));
      }
      assert.doesNotMatch(html, /restricted-optimum|restricted-tree-arrow|restricted-.*contour/);
    }
  }
});

test('MSE contours reveal before fitting and remain exact equal-loss circles', () => {
  for (const progress of [0, .25, .5, .75, 1]) {
    const state = { progress, width: 970, height: 520 }, q = view.resolve(3, state), html = view.render(3, state);
    vectorClose(q.prediction, [0, 0]);
    assert.equal(q.lossSpec.key, 'squared');
    const svg = attributes(/<svg\b[^>]*>/.exec(html)[0]), scale = Number(svg['data-scale']);
    const current = attributes(/<circle\b[^>]*class="restricted-current"[^>]*>/.exec(html)[0]);
    const group = attributes(/<g\b[^>]*class="restricted-contour-reveal"[^>]*>/.exec(html)[0]);
    close(Number(group.opacity), progress);
    const contours = [...html.matchAll(/<polyline\b[^>]*>/g)].map(match => attributes(match[0]));
    assert.equal(contours.length, 4);
    for (const contour of contours) {
      const level = Number(contour['data-loss']);
      for (const point of contour.points.split(' ')) {
        const [px, py] = point.split(',').map(Number), prediction = [(px - Number(current.cx)) / scale, (Number(current.cy) - py) / scale];
        close(selectedLoss(prediction, { loss: 'squared' }), level);
        close(Math.hypot(prediction[0] - 1, prediction[1]), Math.sqrt(2 * level));
      }
    }
    assert.doesNotMatch(html, /restricted-optimum|restricted-tree-arrow|restricted-destination-label|data-(?:optimal|next|correction)/);
  }
});

test('a separate trial moves away from the midpoint and increases MSE without moving the baseline', () => {
  for (const progress of [0, .125, .25, .375, .5, .625, .75, .875, 1]) {
    const state = { progress, width: 700, height: 520 }, q = view.resolve(5, state), html = view.render(5, state);
    vectorClose(q.prediction, [.5, .5]);
    vectorClose(q.residual, [.5, -.5]);
    vectorClose(q.trialPrediction, [q.trialC, q.trialC]);
    close(q.trialLoss, selectedLoss(q.trialPrediction, { loss: 'squared' }));
    close(q.trialLoss, .25 + (q.trialC - .5) ** 2);
    assert.ok(q.trialLoss >= q.loss);
    close(Number(dataValue(html, 'trial-loss')), q.trialLoss);
    vectorClose(JSON.parse(dataValue(html, 'trial-prediction')), q.trialPrediction);
    assert.match(html, /restricted-right-angle/);
    if (Math.abs(q.trialC - .5) > 1e-8) {
      assert.match(html, /class="restricted-trial"/);
      assert.match(html, /class="restricted-trial-contour"/);
      assert.ok(q.trialLoss > .25);
    } else assert.doesNotMatch(html, /class="restricted-trial"/);
    assert.doesNotMatch(html, /gradient|∇/i);
  }
  vectorClose(view.resolve(5, { progress: 0 }).trialPrediction, [.5, .5]);
  vectorClose(view.resolve(5, { progress: 1 }).trialPrediction, [.5, .5]);
  vectorClose(view.resolve(5, { progress: .25 }).trialPrediction, [.68, .68]);
  vectorClose(view.resolve(5, { progress: .75 }).trialPrediction, [.32, .32]);
});

test('loss contours morph while the midpoint and its actual residual stay fixed', () => {
  let baselinePoint, baselineArrow;
  for (const progress of [0, .25, .5, .75, 1]) {
    const state = { progress, width: 970, height: 520 }, q = view.resolve(6, state), html = view.render(6, state);
    const tau = .5 + .3 * progress;
    vectorClose(q.prediction, [.5, .5]);
    vectorClose(q.residual, [.5, -.5]);
    close(q.lossSpec.tau, tau);
    close(q.loss, .25);
    close(q.minLoss, tau * (1 - tau));
    vectorClose(q.optimalPrediction, [tau, tau]);
    const svg = attributes(/<svg\b[^>]*>/.exec(html)[0]), scale = Number(svg['data-scale']);
    const point = attributes(/<circle\b[^>]*class="restricted-current"[^>]*>/.exec(html)[0]);
    const arrow = attributes(/<line\b[^>]*class="restricted-residual-arrow"[^>]*>/.exec(html)[0]);
    const pointCoordinates = [Number(point.cx), Number(point.cy)], arrowCoordinates = [arrow.x1, arrow.y1, arrow.x2, arrow.y2].map(Number);
    baselinePoint ??= pointCoordinates;
    baselineArrow ??= arrowCoordinates;
    vectorClose(pointCoordinates, baselinePoint);
    vectorClose(arrowCoordinates, baselineArrow);
    for (const match of html.matchAll(/<polyline\b[^>]*data-loss="[^>]*>/g)) {
      const contour = attributes(match[0]), level = Number(contour['data-loss']);
      for (const pixel of contour.points.split(' ')) {
        const [px, py] = pixel.split(',').map(Number), prediction = [.5 + (px - Number(point.cx)) / scale, .5 - (py - Number(point.cy)) / scale];
        close(selectedLoss(prediction, { loss: 'expectile', tau }), level);
      }
    }
    assert.doesNotMatch(html, /gradient|∇/i);
    assert.match(view.annotation(6, state).equation, /L\(c,c\) = 0\.8\(1 − c\)² \+ 0\.2c²/);
    assert.doesNotMatch(view.annotation(6, state).equation, /ℓ|cases/);
  }
  close(view.resolve(6, { progress: 1 }).minLoss, .16);
});

test('compact midpoint residual labels clear the outcomes and stay fixed through the loss morph', () => {
  for (const width of [273, 288, 320]) {
    let residualPosition;
    for (const [scene, progress] of [[5, 0], [5, .25], [5, .75], [6, 0], [6, .25], [6, .5], [6, .75], [6, 1]]) {
      const html = view.render(scene, { progress, width, height: 520 }, true);
      const labels = [...html.matchAll(/<text\b([^>]*)>([^<]*)<\/text>/g)].map(match => ({ ...attributes(match[1]), text: match[2] }));
      const residual = labels.find(label => label.class.includes('restricted-residual-label'));
      const outcome = labels.find(label => label.class.includes('restricted-target-label'));
      const position = [Number(residual.x), Number(residual.y)];
      residualPosition ??= position;
      vectorClose(position, residualPosition);
      assert.ok(Number(outcome.y) - Number(residual.y) > 20, 'Compact residual and outcome labels need distinct vertical space');
      if (scene === 6 && progress > 0) {
        const best = labels.find(label => label.class.includes('restricted-best-label'));
        assert.equal(best.text, progress === 1 ? 'best: (0.8, 0.8)' : 'best shared fit');
      }
    }
  }
});

test('floating annotations stay sparse and responsive plot headroom stays stable', () => {
  assert.equal(view.annotation(11, { loss: 'expectile', tau: .8 }).caption, 'Asymmetric squared: underprediction costs 4 times as much.');
  assert.doesNotMatch(view.annotation(11, { loss: 'expectile', tau: .8 }).caption, /τ/);
  for (const [width, height, headroom] of [[273, 520, 86], [450, 520, 86], [970, 520, 86]]) {
    let lastScale;
    for (let scene = 0; scene <= 11; scene++) {
      const state = { progress: .5, width, height }, html = view.render(scene, state), note = view.annotation(scene, state);
      const svg = attributes(/<svg\b[^>]*>/.exec(html)[0]);
      close(Number(svg['data-annotation-headroom']), headroom);
      lastScale ??= Number(svg['data-scale']);
      close(Number(svg['data-scale']), lastScale);
      assert.equal(typeof note.equation, 'string');
      assert.equal(typeof note.caption, 'string');
      assert.equal(typeof note.status, 'string');
      if (![3, 6, 11].includes(scene)) assert.equal(note.equation, '');
      assert.ok(note.caption.split(/\s+/).length <= 12, 'Floating captions should stay brief');
      assert.doesNotMatch(note.equation, /loss-equation-cases/);
      if (scene < 7) assert.doesNotMatch(JSON.stringify(note), /gradient|∇/i);
    }
  }
});

test('the residual animation moves together from zero to the mean and decreases MSE', () => {
  let lastLoss = Infinity;
  for (let step = 0; step <= 20; step++) {
    const progress = step / 20, c = progress / 2;
    const state = { progress, prediction: .99, loss: 'quartic', width: 700, height: 440 };
    const q = view.resolve(4, state);
    vectorClose(q.prediction, [c, c]);
    vectorClose(q.residual, [1 - c, -c]);
    close(q.residualMean, .5 - c);
    close(q.loss, ((1 - c) ** 2 + c ** 2) / 2);
    assert.ok(q.loss <= lastLoss);
    lastLoss = q.loss;
    const html = view.render(4, state);
    const svg = attributes(/<svg\b[^>]*>/.exec(html)[0]);
    const residualArrow = attributes(/<line\b[^>]*class="restricted-residual-arrow"[^>]*>/.exec(html)[0]);
    const scale = Number(svg['data-scale']);
    vectorClose([(Number(residualArrow.x2) - Number(residualArrow.x1)) / scale, (Number(residualArrow.y1) - Number(residualArrow.y2)) / scale], q.residual);
    assert.match(html, /class="restricted-progress-path"/);
    assert.doesNotMatch(html, /gradient|∇/i);
  }
  close(lastLoss, .25);
  vectorClose(view.resolve(4, { progress: -10 }).prediction, [0, 0]);
  vectorClose(view.resolve(4, { progress: 10 }).prediction, [.5, .5]);
});

test('the residual progression explicitly labels its current values and the fitted destination', () => {
  for (const [width, height, compact] of [[273, 300, true], [450, 410, false]]) {
    for (const [progress, prediction, residual] of [
      [0, '(0, 0)', '(1, 0)'],
      [.5, '(0.25, 0.25)', '(0.75, −0.25)'],
      [1, '(0.5, 0.5)', '(0.5, −0.5)']
    ]) {
      const state = { progress, width, height }, html = view.render(4, state, compact);
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
      const caption = view.presentation(4, state).caption;
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
  const q = view.resolve(5, { progress: 0, prediction: 0, loss: 'expectile' });
  vectorClose(q.prediction, [.5, .5]);
  vectorClose(q.residual, [.5, -.5]);
  close(math.dot(q.residual, [1, 1]), 0);
  close(q.residualMean, 0);
  close(q.loss, .25);
  const html = view.render(5, { width: 900, height: 430 });
  assert.match(html, /class="restricted-right-angle"[^>]*data-dot-product="0"/);
  assert.match(html, /90°/);
  assert.match(view.presentation(5).caption, /Try moving either way: MSE increases/);
  assert.doesNotMatch(html, /gradient|∇/i);
  // Moving either way along the diagonal increases the actual average error.
  for (const shift of [-.1, -.001, .001, .1]) assert.ok(selectedLoss([.5 + shift, .5 + shift], { loss: 'squared' }) > q.loss);
});

test('changed loss establishes the counterexample before gradients are introduced', () => {
  for (const progress of [0, .25, .5, .75, 1]) {
    const q = view.resolve(6, { progress, tau: .2, loss: 'squared' });
    vectorClose(q.prediction, [.5, .5]);
    vectorClose(q.residual, [.5, -.5]);
    assert.equal(q.lossSpec.key, 'expectile');
    close(q.lossSpec.tau, .5 + .3 * progress);
    close(q.rawNextLoss, .25);
    close(q.minLoss, q.lossSpec.tau * (1 - q.lossSpec.tau));
    assert.ok(q.loss <= .25 + 1e-12);
    assert.doesNotMatch(view.render(6, { progress }), /gradient|∇/i);
    assert.doesNotMatch(JSON.stringify(view.presentation(6, { progress })), /gradient|∇/i);
  }
  const firstGradient = view.resolve(7);
  vectorClose(firstGradient.prediction, [.5, .5]);
  vectorClose(firstGradient.residual, [.5, -.5]);
  close(firstGradient.residualMean, 0);
  vectorClose(firstGradient.negativeGradient, [.8, -.2]);
  close(firstGradient.negativeGradientMean, .3);
  vectorClose(firstGradient.nextTree, [.3, .3]);
  assert.match(view.render(7), /class="restricted-gradient-arrow"/);
  assert.match(view.presentation(7).caption, /training targets.*mean is 0\.3/);
  const lastGradient = view.resolve(10, { progress: 1 });
  close(lastGradient.negativeGradientMean, 0);
  close(math.dot(lastGradient.negativeGradient, [1, 1]), 0);
  close(lastGradient.residualMean, -.3);
});

test('training targets use observed outcomes while a fitted tree predicts from inputs alone', () => {
  const prediction = [.5, .5], targets = view.computeTrainingTargets([1, 0], prediction);
  vectorClose(targets, [.8, -.2]);
  vectorClose(view.computeTrainingTargets([0, 0], prediction), [-.2, -.2]);
  const tree = math.fitStump([1, 1], targets);
  vectorClose(view.predictFittedTree(tree, [1, 1]), [.3, .3]);
  // This constant fitted tree can also be evaluated at a new x; no y or
  // training target is provided to its prediction function.
  vectorClose(view.predictFittedTree(tree, [4, -2]), [.3, .3]);
  vectorClose(view.predictFittedTree({ threshold: 2, left: .3, right: 1.1 }, [1, 3]), [.3, 1.1]);
  assert.match(view.presentation(8).caption, /fitted tree takes x alone/);
  assert.match(view.annotation(8).caption, /h\(x\) needs x alone/);
});

test('targets and the fitted-tree preview stay at the midpoint before any update is applied', () => {
  for (const progress of [0, .25, .5, .75, 1]) {
    for (const scene of [7, 8]) {
      const q = view.resolve(scene, { progress });
      vectorClose(q.prediction, [.5, .5]);
      vectorClose(q.targetPrediction, [.5, .5]);
      vectorClose(q.trainingTargets, [.8, -.2]);
      vectorClose(q.fittedCorrection, [.3, .3]);
      vectorClose(q.updatePrediction, [.65, .65]);
      close(q.learningRate, .5);
    }
  }
  const html = view.render(8, { progress: 1, width: 700, height: 520 });
  const svg = attributes(/<svg\b[^>]*>/.exec(html)[0]), scale = Number(svg['data-scale']);
  const fitted = attributes(/<line\b[^>]*class="restricted-tree-arrow"[^>]*>/.exec(html)[0]);
  const scaled = attributes(/<line\b[^>]*class="restricted-scaled-update"[^>]*>/.exec(html)[0]);
  vectorClose([(Number(fitted.x2) - Number(fitted.x1)) / scale, (Number(fitted.y1) - Number(fitted.y2)) / scale], [.3, .3]);
  vectorClose([(Number(scaled.x2) - Number(scaled.x1)) / scale, (Number(scaled.y1) - Number(scaled.y2)) / scale], [.15, .15]);
  assert.doesNotMatch(view.render(7, { progress: 1 }), /restricted-tree-arrow|restricted-update-endpoint/);
});

test('six actual fitted-tree updates halve the remaining error without reaching the limit exactly', () => {
  const trajectory = view.buildDescentPath();
  assert.equal(trajectory.rounds, 6);
  assert.equal(trajectory.steps.length, 6);
  assert.equal(trajectory.path.length, 7);
  close(trajectory.learningRate, .5);
  for (let round = 0; round <= 6; round++) {
    const c = .8 - .3 * 2 ** -round;
    vectorClose(trajectory.path[round], [c, c]);
    if (round < 6) {
      const step = trajectory.steps[round], next = c + .5 * (.8 - c);
      vectorClose(step.targetPrediction, [c, c]);
      vectorClose(step.trainingTargets, [1.6 * (1 - c), -.4 * c]);
      vectorClose(step.fittedCorrection, [.8 - c, .8 - c]);
      vectorClose(step.fittedCorrection, view.predictFittedTree(step.tree, [1, 1]));
      vectorClose(step.updatePrediction, [next, next]);
      assert.equal(step.tree.threshold, null);
      close(step.beforeLoss, selectedLoss([c, c], { loss: 'expectile', tau: .8 }));
      close(step.afterLoss, selectedLoss([next, next], { loss: 'expectile', tau: .8 }));
      assert.ok(step.afterLoss < step.beforeLoss);
    }
  }
  const final = view.resolve(9, { progress: 1 });
  vectorClose(final.prediction, [.7953125, .7953125]);
  close(final.fittedCorrection[0], .0046875);
  assert.equal(final.phase, 'complete');
  assert.ok(final.loss > final.minLoss);
  assert.ok(final.prediction[0] < .8);
  assert.match(view.presentation(9, { progress: 1 }).caption, /close to the optimum, not exactly there/);
  const html = view.render(9, { progress: 1 });
  assert.match(html, /restricted-prediction-label[^>]*>f ≈ \(0\.7953, 0\.7953\)/);
  assert.doesNotMatch(html, /restricted-prediction-label[^>]*>f = \(0\.8, 0\.8\)/);
});

test('each round reveals targets, fits once, and keeps that tree frozen during its scaled update', () => {
  for (let round = 0; round < 6; round++) {
    const baseC = .8 - .3 * 2 ** -round;
    const cases = [[.125, 'target', 0], [.375, 'fit', 0], [.625, 'update', .25], [.75, 'update', .5], [.875, 'update', .75]];
    for (const [withinRound, phase, updateFraction] of cases) {
      const progress = (round + withinRound) / 6, q = view.resolve(9, { progress });
      const current = baseC + .5 * (.8 - baseC) * updateFraction;
      assert.equal(q.round, round);
      assert.equal(q.phase, phase);
      vectorClose(q.targetPrediction, [baseC, baseC]);
      vectorClose(q.trainingTargets, [1.6 * (1 - baseC), -.4 * baseC]);
      vectorClose(q.fittedCorrection, [.8 - baseC, .8 - baseC]);
      vectorClose(q.nextTree, q.fittedCorrection);
      vectorClose(q.prediction, [current, current]);
      vectorClose(q.currentNegativeGradient, [1.6 * (1 - current), -.4 * current]);
      close(q.loss, selectedLoss(q.prediction, { loss: 'expectile', tau: .8 }));
      const html = view.render(9, { progress, width: 700, height: 520 });
      assert.equal(dataValue(html, 'phase'), phase);
      vectorClose(JSON.parse(dataValue(html, 'target-prediction')), q.targetPrediction);
      vectorClose(JSON.parse(dataValue(html, 'display-negative-gradient')), q.trainingTargets);
      vectorClose(JSON.parse(dataValue(html, 'current-negative-gradient')), q.currentNegativeGradient);
      const gradient = attributes(/<line\b[^>]*class="restricted-gradient-arrow"[^>]*>/.exec(html)[0]);
      vectorClose(gradient['data-at-values'].split(',').map(Number), q.targetPrediction);
      vectorClose(gradient['data-values'].split(',').map(Number), q.trainingTargets);
      if (phase === 'update') assert.match(html, /restricted-target-base/);
    }
  }
});

test('compact process labels separate the fitted endpoint from the fixed training base', () => {
  for (const width of [273, 288, 320]) {
    for (let round = 0; round < 6; round++) {
      for (const local of [.625, .75, .875]) {
        const html = view.render(9, { progress: (round + local) / 6, width, height: 520 }, true);
        const labels = [...html.matchAll(/<text\b([^>]*)>([^<]*)<\/text>/g)].map(match => ({ ...attributes(match[1]), text: match[2] }));
        const fitted = labels.find(label => label.class.includes('restricted-fitted-label'));
        const base = labels.find(label => label.class.includes('restricted-target-base-label'));
        assert.equal(fitted['text-anchor'], 'start');
        assert.equal(base['text-anchor'], 'end');
        // The training-base label ends to the left of the h(x) label's start,
        // leaving room for both glyph widths and their white outline strokes.
        assert.ok(Number(fitted.x) - Number(base.x) >= 18 - 1e-9);
        assert.ok(Number(fitted.x) + 30 < width, 'The short h(x) label fits inside the compact plot');
      }
    }
  }
});

test('the exact limit has a nonzero perpendicular gradient and a zero fitted tree', () => {
  const q = view.resolve(10);
  vectorClose(q.prediction, [.8, .8]);
  vectorClose(q.trainingTargets, [.32, -.32]);
  assert.ok(math.norm(q.trainingTargets) > 0);
  close(math.dot(q.trainingTargets, [1, 1]), 0);
  vectorClose(q.fittedCorrection, [0, 0]);
  vectorClose(q.updatePrediction, [.8, .8]);
  vectorClose(q.residual, [.2, -.8]);
  close(q.residualMean, -.3);
  close(q.loss, .16);
  const html = view.render(10);
  assert.match(html, /restricted-zero-fit-label[^>]*>h\(x\) = 0/);
  assert.match(html, /restricted-right-angle[^>]*data-vector="negative-gradient"[^>]*data-dot-product="0"/);
  assert.match(html, /restricted-prediction-label[^>]*>f = \(0\.8, 0\.8\)/);
  assert.doesNotMatch(html, /restricted-tree-arrow/);
});

test('early equations display the exact two-point MSE and both axes retain half ticks', () => {
  for (const scene of [3, 4, 5]) {
    const html = view.equation({ loss: 'quartic' }, scene);
    assert.match(html, /MSE\(f₁,f₂\) = ½\[\(1 − f₁\)² \+ f₂²\]/);
    assert.doesNotMatch(html, /MSE\(c,c\)/);
    assert.doesNotMatch(html, /ℓ|gradient|∇/i);
  }
  for (let scene = 0; scene <= 11; scene++) {
    const html = view.render(scene);
    assert.equal([...html.matchAll(/>½<\/text>/g)].length, 2);
    assert.match(html, new RegExp(`<title id="restricted-title">${view.topics[scene]}</title>`));
  }
  assert.throws(() => view.resolve(-1), RangeError);
  assert.throws(() => view.resolve(12), RangeError);
});

test('the loss explorer uses the requested observations and asymmetric squared loss', () => {
  const q = view.resolve(11);
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
      const q = view.resolve(11, { ...state, prediction: c });
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
      const q = view.resolve(11, { loss: 'expectile', tau, prediction: c });
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
  const atMean = view.resolve(11, { loss: 'expectile', tau: .8, prediction: .5 });
  vectorClose(atMean.residual, [.5, -.5]);
  close(math.dot(atMean.residual, [1, 1]), 0);
  vectorClose(atMean.negativeGradient, [.8, -.2]);
  vectorClose(atMean.nextTree, [.3, .3]);
  close(atMean.loss, .25);
  close(atMean.nextLoss, .16);

  const atOptimum = view.resolve(11, { loss: 'expectile', tau: .8, prediction: .8 });
  vectorClose(atOptimum.negativeGradient, [.32, -.32]);
  close(math.dot(atOptimum.negativeGradient, [1, 1]), 0);
  vectorClose(atOptimum.nextTree, [0, 0]);
  vectorClose(atOptimum.correction, [0, 0]);
  close(math.dot(atOptimum.residual, [1, 1]), -.6);
  close(atOptimum.loss, .16);
  close(atOptimum.rawNextLoss, .25);
  assert.match(view.presentation(11, { loss: 'expectile', tau: .8, prediction: .8 }).caption, /gradient.*perpendicular.*fitted gradient is zero/i);
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
    const q = view.resolve(11, { loss: 'huber', delta, prediction: c });
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
    const q = view.resolve(11, state);
    vectorClose(q.optimumInterval, [delta, 1 - delta]);
    close(q.minLoss, delta - delta ** 2);
    for (const c of [delta, .5, 1 - delta]) {
      const inPlateau = view.resolve(11, { ...state, prediction: c });
      close(inPlateau.loss, q.minLoss);
      vectorClose(inPlateau.correction, [0, 0]);
      vectorClose(inPlateau.nextTree, [0, 0]);
    }
    for (const c of [0, delta / 2, 1 - delta / 2, 1]) {
      const outsidePlateau = view.resolve(11, { ...state, prediction: c });
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
    const q = view.resolve(11, { loss: 'quartic', prediction: c });
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
      const q = view.resolve(11, { ...state, prediction: c });
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
        const html = view.render(11, { ...state, prediction: c, width, height }, width < 400);
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
        const q = view.resolve(11, { ...state, prediction: c });
        const html = view.render(11, { ...state, prediction: c, width, height }, width < 400);
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
    const q = view.resolve(11, { ...state, prediction: .37 });
    const html = view.render(11, { ...state, prediction: .37 });
    assert.equal(dataValue(html, 'loss-key'), state.loss);
    for (const [attribute, expected] of [
      ['target', q.target], ['prediction', q.prediction], ['residual', q.residual],
      ['gradient', q.gradient], ['negative-gradient', q.negativeGradient], ['next-tree', q.nextTree],
      ['correction', q.correction], ['next-prediction', q.nextPrediction], ['optimum-interval', q.optimumInterval]
    ]) vectorClose(JSON.parse(dataValue(html, attribute)), expected);
    close(Number(dataValue(html, 'loss')), q.loss);
    close(Number(dataValue(html, 'min-loss')), q.minLoss);
    close(Number(dataValue(html, 'next-loss')), q.nextLoss);
    assert.ok(view.readout(11, state).every(row => row.length >= 2 && typeof row[0] === 'string'));
  }
});

test('Huber captions describe selected loss and never claim all Huber gradients are raw residuals', () => {
  const state = { loss: 'huber', delta: .75, prediction: 0 };
  const caption = view.presentation(11, state).caption;
  assert.doesNotMatch(caption, /residual and negative gradient coincide|half-squared loss/i);
  assert.match(caption, /Huber|clipp|gradient/i);
  assert.match(view.presentation(11, { loss: 'huber', delta: .25 }).caption, /Every c in \[0\.25, 0\.75\]/);
});

test('the restricted-input visual runs without a module loader', () => {
  const context = { window: { BoostingMath: math } };
  runInNewContext(readFileSync(new URL('../docs/boosting/restricted-view.js', import.meta.url), 'utf8'), context);
  assert.equal(typeof context.window.RestrictedBoostingView.render, 'function');
  assert.equal(typeof context.window.RestrictedBoostingView.resolve, 'function');
  const q = context.window.RestrictedBoostingView.resolve(11, { prediction: .8 });
  vectorClose(q.target, [1, 0]);
  vectorClose(q.prediction, [.8, .8]);
  vectorClose(q.nextTree, [0, 0]);
  close(q.loss, .16);
});
