import { svgMathText } from './svg-math-text.mjs';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { runInNewContext } from 'node:vm';
import test from 'node:test';

const require = createRequire(import.meta.url);
const math = require('../docs/shared/boosting-math.js');
const typography = require('../docs/shared/svg-math.js');
const view = require('../docs/boosting/prediction-view.js');

test('prediction captions display native scripts and norms while preserving ordinary words', () => {
  const N = typography.notation;
  const ordinary = N.parts(view.presentation(0, { round: 2, rate: .5 }).captionParts);
  assert.match(ordinary, / stays fixed; each tree is refitted to /);
  assert.doesNotMatch(ordinary, /ever<math|sta<math/);
  const update = N.parts(view.presentation(1, { round: 1, rate: .5, progress: .5 }).captionParts);
  assert.match(update, /<msub><mi>h<\/mi><mrow><mn>2<\/mn>/);
  assert.match(update, /<msup><mi>F<\/mi><mrow><mo>′<\/mo>/);
  const contour = N.parts(view.presentation(2, { round: 1 }).captionParts);
  assert.match(contour, /<mo>‖<\/mo><mi>y<\/mi><mo>−<\/mo><mi>F<\/mi><mo>‖<\/mo>/);
  const readout = view.readoutNotation(1, { round: 1, rate: .5, progress: .5 });
  assert.match(N.parts(readout[0].nameParts), /<math/);
  assert.match(N.parts(readout[0].valueParts), /<mo stretchy="false">\(<\/mo>/);
});
const close = (actual, expected, tolerance = 1e-9) => assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} should equal ${expected}`);
const vectorClose = (actual, expected) => {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => close(value, expected[index]));
};
const squared = values => values.reduce((total, value) => total + value * value, 0);
const attributes = fragment => Object.fromEntries([...fragment.matchAll(/([\w-]+)="([^"]*)"/g)].map(match => [match[1], match[2]]));
const rootAttributes = html => attributes(/<[^>]+class="[^"]*\bprediction-view\b[^>]*>/.exec(html)[0]);

// The introductory three-vector geometry is independent of the worked four-
// observation example. These expected values are hand-calculated from its trees.
test('the observed vector stays fixed and its actual residual norm determines half-squared loss', () => {
  const initial = view.resolve(0);
  vectorClose(initial.prediction, [0, 0, 0]);
  vectorClose(initial.residual, [1, -1, 1]);
  vectorClose(initial.gradient, [-1, 1, -1]);
  close(initial.norm, Math.sqrt(3));
  close(initial.loss, 1.5);
  close(initial.radius, Math.sqrt(3));
  const first = view.resolve(0, { round: 1, rate: .5 });
  vectorClose(first.prediction, [.5, 0, 0]);
  vectorClose(first.residual, [.5, -1, 1]);
  vectorClose(first.gradient, [-.5, 1, -1]);
  close(first.norm, 1.5);
  close(first.loss, 1.125);
  const second = view.resolve(0, { round: 2, rate: .5 });
  vectorClose(second.prediction, [.375, -.125, .5]);
  close(second.loss, .703125);
  for (const state of [initial, first, second]) {
    close(state.loss, squared(state.residual) / 2);
    close(state.loss, state.norm * state.norm / 2);
  }
});

test('round and rate controls refit every tree to the current residual', () => {
  for (const rate of [0, .25, .5, 1]) {
    const expected = math.buildPath({ rate, rounds: 24 });
    for (const round of [0, 1, 2, 8, 24]) {
      const result = view.resolve(0, { round, rate });
      vectorClose(result.prediction, expected[round].pred);
      vectorClose(result.residual, expected[round].residual);
      close(result.norm, Math.sqrt(squared(expected[round].residual)));
      close(result.loss, expected[round].loss);
      vectorClose(result.gradient, expected[round].residual.map(value => -value));
    }
  }
});

test('adding a partial fitted correction updates F, both gradients and the true residual norm', () => {
  const before = view.resolve(1, { round: 0, rate: .5, progress: 0 });
  const halfway = view.resolve(1, { round: 0, rate: .5, progress: .5 });
  const after = view.resolve(1, { round: 0, rate: .5, progress: 1 });
  vectorClose(before.prediction, [0, 0, 0]);
  vectorClose(halfway.prediction, [.25, 0, 0]);
  close(halfway.loss, 1.28125);
  vectorClose(after.prediction, [.5, 0, 0]);
  vectorClose(after.gradient, [-.5, 1, -1]);
  close(after.norm, 1.5);
  close(after.loss, 1.125);
  const second = view.resolve(1, { round: 1, rate: .5, progress: 1 });
  vectorClose(second.prediction, [.375, -.125, .5]);
  close(second.loss, .703125);
  close(second.beforeLoss, 1.125);
});

test('successive correction updates start from the previous endpoint and refit a different tree', () => {
  // With α = ½, the best stump first isolates observation 1, then 3, then 1.
  // Keeping the first tree fixed would never produce the second or third F.
  const expected = [
    { base: [0, 0, 0], tree: [1, 0, 0], prediction: [.5, 0, 0], loss: 1.125 },
    { base: [.5, 0, 0], tree: [-.25, -.25, 1], prediction: [.375, -.125, .5], loss: .703125 },
    { base: [.375, -.125, .5], tree: [.625, -.1875, -.1875], prediction: [.6875, -.21875, .40625], loss: .5302734375 }
  ];
  let previousLoss = 1.5;
  expected.forEach((step, round) => {
    const q = view.resolve(1, { round, rate: .5, progress: 1 });
    vectorClose(q.base, step.base);
    vectorClose(q.tree.values, step.tree);
    vectorClose(q.prediction, step.prediction);
    vectorClose(q.residual, [1, -1, 1].map((value, index) => value - step.prediction[index]));
    close(q.beforeLoss, previousLoss);
    close(q.loss, step.loss);
    previousLoss = step.loss;
  });
});

test('numbered correction labels and loss readouts distinguish completed and partial updates', () => {
  for (const width of [320, 640]) {
    const complete = { round: 1, rate: .5, progress: 1, width };
    assert.match(view.presentation(1, complete).caption, /^Update 2: F₁ → F₂\./);
    assert.match(view.presentation(1, complete).caption, /Refit h₂ = \(−0\.25, −0\.25, 1\)/);
    assert.match(view.presentation(1, complete).caption, /Loss 1\.125 → 0\.7031\./);
    const rows = view.readout(1, complete);
    assert.equal(rows[1][0], 'Updated prediction F₂');
    assert.equal(rows[2][0], 'Gradient at F₂');
    assert.equal(rows[3][1], '1.125 → 0.7031; 1.186');
    const fullHtml = view.render(1, complete);
    assert.match(svgMathText(fullHtml), />h₂<\/text>/);
    assert.match(svgMathText(fullHtml), />F₁<\/text>/);
    assert.match(svgMathText(fullHtml), />F₂<\/text>/);
    const partial = { ...complete, progress: .5 };
    assert.match(view.presentation(1, partial).caption, /^Update 2: F₁ → F′\./);
    assert.match(view.presentation(1, partial).caption, /add 0\.5 × αh₂\./);
    assert.equal(view.readout(1, partial)[1][0], 'Updated prediction F′');
    assert.equal(view.readout(1, partial)[2][0], 'Gradient at F′');
    const partialHtml = view.render(1, partial);
    assert.match(svgMathText(partialHtml), />F′<\/text>/);
    assert.doesNotMatch(svgMathText(partialHtml), />F₂<\/text>/);
  }
  assert.match(view.presentation(1, { round: 24 }).caption, /^Update 25: F₂₄ → F₂₅\./);
});

test('the accumulated correction path ends at the next starting point without including the partial tree', () => {
  const point = (html, className) => {
    const circle = attributes(new RegExp(`<circle\\b[^>]*class="${className}"[^>]*>`).exec(html)[0]);
    return [Number(circle.cx), Number(circle.cy)];
  };
  const state = { rate: .5, progress: 1, width: 640, height: 450 };
  let previous = view.render(1, { ...state, round: 0 });
  for (const round of [1, 2, 3]) {
    const current = view.render(1, { ...state, round });
    const path = attributes(/<polyline\b[^>]*class="prediction-path"[^>]*>/.exec(current)[0]);
    const points = path.points.split(' ').map(pair => pair.split(',').map(Number));
    assert.equal(points.length, round + 1);
    vectorClose(points.at(-1), point(current, 'prediction-base'));
    vectorClose(points.at(-1), point(previous, 'prediction-current'));
    const partial = view.render(1, { ...state, round, progress: .5 });
    const partialPath = attributes(/<polyline\b[^>]*class="prediction-path"[^>]*>/.exec(partial)[0]);
    assert.equal(partialPath.points, path.points, 'Scrubbing only changes the current correction, not earlier trees');
    assert.notDeepEqual(point(partial, 'prediction-current'), points.at(-1));
    previous = current;
  }
});

test('the flat radius is an actual equal-loss distance, not a normalized or projected length', () => {
  for (const round of [0, 1, 2, 8]) {
    for (const angle of [0, 30, 90, 120, 180]) {
      for (const length of [0, .5, 1, 2.5]) {
        const result = view.resolve(2, { round, rate: .5, angle, length });
        const beforeResidual = result.base.map((value, index) => [1, -1, 1][index] - value);
        const radius = Math.sqrt(squared(beforeResidual));
        close(result.radius, radius);
        close(result.beforeLoss, radius * radius / 2);
        vectorClose(result.prediction, result.base.map((value, index) => value + result.step[index]));
        close(result.stepNorm, length * radius);
        close(result.stepNorm, Math.sqrt(squared(result.step)));
        close(result.norm, Math.sqrt(squared(result.residual)));
        close(result.loss, squared(result.residual) / 2);
        close(result.loss, squared([radius - result.planeStep[0], -result.planeStep[1]]) / 2);
      }
    }
  }
  close(view.resolve(2, { round: 0 }).radius, Math.sqrt(3));
  close(view.resolve(2, { round: 1 }).radius, 1.5);
});

test('the selected orientation handles obtuse, orthogonal and overshooting corrections correctly', () => {
  const aligned = view.resolve(2, { angle: 0, length: 1 });
  vectorClose(aligned.prediction, [1, -1, 1]);
  close(aligned.loss, 0);
  const obtuse = view.resolve(2, { angle: 120, length: .5 });
  assert.equal(obtuse.orientation, -1);
  assert.equal(obtuse.effectiveAngle, 60);
  assert.ok(obtuse.planeStep[0] > 0);
  assert.ok(obtuse.planeStep[1] < 0, 'Negation must reverse both plane coordinates');
  close(obtuse.loss, 1.125);
  const orthogonal = view.resolve(2, { angle: 90, length: .5 });
  close(orthogonal.loss, 1.875);
  assert.ok(orthogonal.loss > orthogonal.beforeLoss);
  const boundary = view.resolve(2, { angle: 0, length: 2 });
  close(boundary.loss, boundary.beforeLoss);
  const overshoot = view.resolve(2, { angle: 0, length: 2.5 });
  close(overshoot.loss, 3.375);
  assert.ok(overshoot.loss > overshoot.beforeLoss);
});

test('the exact plane basis preserves true three-dimensional distances and angles', () => {
  for (const round of [0, 1, 2, 8]) {
    const result = view.resolve(2, { round, angle: 35, length: .75 });
    const { e1: first, e2: second } = result.basis;
    close(squared(first), 1);
    close(squared(second), 1);
    close(first.reduce((sum, value, index) => sum + value * second[index], 0), 0);
    vectorClose(result.step, first.map((value, index) => value * result.planeStep[0] + second[index] * result.planeStep[1]));
    close(math.angleDegrees(result.step, result.base.map((value, index) => [1, -1, 1][index] - value)), 35);
  }
});

test('rendered geometry exposes computed vectors and uses a loss circle only in its exact plane', () => {
  for (const scene of [0, 1, 2]) {
    const state = { round: 1, rate: .5, angle: 120, length: .5, progress: .5, width: 640, height: 450 };
    const expected = view.resolve(scene, state);
    const html = view.render(scene, state);
    const attrs = rootAttributes(html);
    assert.equal(Number(attrs['data-geometry-scene']), scene);
    vectorClose(JSON.parse(attrs['data-prediction']), expected.prediction);
    vectorClose(JSON.parse(attrs['data-gradient']), expected.gradient);
    close(Number(attrs['data-norm']), expected.norm);
    close(Number(attrs['data-loss']), expected.loss);
    if (scene === 2) {
      assert.match(svgMathText(html), /data-geometry="equal-loss-plane"/);
      const circle = attributes(/<circle\b[^>]*class="equal-loss-circle"[^>]*>/.exec(html)[0]);
      close(Number(circle['data-radius']), expected.radius);
      const svg = attributes(/<svg\b[^>]*>/.exec(html)[0]);
      const base = attributes(/<circle\b[^>]*class="prediction-base"[^>]*>/.exec(html)[0]);
      const target = attributes(/<path\b[^>]*class="prediction-target"[^>]*>/.exec(html)[0]);
      close(Number(circle.r), Number(svg['data-scale']) * expected.radius);
      close(Math.hypot(Number(base.cx) - Number(target['data-x']), Number(base.cy) - Number(target['data-y'])), Number(circle.r));
      close(Number(svg['data-scale']) * Number(svg['data-units-per-pixel']), 1);
    } else {
      assert.match(svgMathText(html), /data-geometry="orthographic-3d"/);
      assert.doesNotMatch(svgMathText(html), /equal-loss-circle/);
    }
  }
});

test('the initial equal-loss circle fills a readable part of short and narrow figures', () => {
  for (const [width, height, compact] of [[288, 220, true], [346, 260, true], [734, 288, false], [640, 450, false]]) {
    for (const angle of [0, 89, 90, 91, 120, 180]) {
      const html = view.render(2, { width, height, angle, length: .5, round: 0 }, compact);
      const circle = attributes(/<circle\b[^>]*class="equal-loss-circle"[^>]*>/.exec(html)[0]);
      // A diameter of at least 60% of the figure's shorter side makes the
      // normal half-radius trial legible without losing its true distances.
      assert.ok(Number(circle.r) >= .3 * Math.min(width, height), `Circle should be readable at ${width}×${height}, θ=${angle}`);
    }
  }
});

test('the fitted camera keeps circles, long trials and direction guides inside the figure', () => {
  for (const [width, height, compact] of [[288, 220, true], [346, 260, true], [734, 288, false], [640, 450, false]]) {
    for (const angle of [0, 30, 89, 90, 91, 120, 180]) {
      for (const length of [0, .5, 1, 2.5]) {
        for (const round of [0, 2, 24]) {
          const state = { width, height, angle, length, round, rate: 1 };
          const html = view.render(2, state, compact);
          const context = JSON.stringify(state);
          const inside = (x, y, padding = 8) => {
            assert.ok(x >= padding && x <= width - padding, `Horizontal containment: (${x},${y}) ${context}`);
            assert.ok(y >= padding && y <= height - padding, `Vertical containment: (${x},${y}) ${context}`);
          };
          const circle = attributes(/<circle\b[^>]*class="equal-loss-circle"[^>]*>/.exec(html)[0]);
          const cx = Number(circle.cx), cy = Number(circle.cy), radius = Number(circle.r);
          inside(cx - radius, cy - radius);
          inside(cx + radius, cy + radius);
          for (const match of html.matchAll(/<line\b[^>]*class="prediction-(?:residual-arrow|gradient-arrow|trial-step|trial-residual|direction-guide)"[^>]*>/g)) {
            const line = attributes(match[0]);
            inside(Number(line.x1), Number(line.y1));
            inside(Number(line.x2), Number(line.y2));
          }
          for (const match of html.matchAll(/<text\b[^>]*class="pv-math pv-direction-label"[^>]*>/g)) {
            const label = attributes(match[0]);
            inside(Number(label.x), Number(label.y));
            assert.ok(Number(label.y) >= 30 && Number(label.y) <= height - 29, `Direction labels should avoid the legend and footer: ${context}`);
          }
        }
      }
    }
  }
});

test('changing the starting round shrinks the circle at the same physical camera scale', () => {
  for (const angle of [0, 89, 90, 91, 120, 180]) {
    for (const length of [.5, 2.5]) {
      for (const [width, height, compact] of [[288, 220, true], [734, 288, false]]) {
        let reference;
        let previousRadius = Infinity;
        for (const round of [0, 1, 2, 8, 24]) {
          const html = view.render(2, { width, height, angle, length, round, rate: .5 }, compact);
          const svg = attributes(/<svg\b[^>]*>/.exec(html)[0]);
          const circle = attributes(/<circle\b[^>]*class="equal-loss-circle"[^>]*>/.exec(html)[0]);
          const camera = [Number(svg['data-scale']), Number(circle.cx), Number(circle.cy)];
          if (!reference) reference = camera;
          else vectorClose(camera, reference);
          assert.ok(Number(circle.r) < previousRadius, `Circle should shrink at round ${round}`);
          previousRadius = Number(circle.r);
        }
      }
    }
  }
});

test('the geometry module runs without a module loader and keeps all three scenes responsive', () => {
  const context = { window: { BoostingMath: math, SvgMath: typography } };
  runInNewContext(readFileSync(new URL('../docs/boosting/prediction-view.js', import.meta.url), 'utf8'), context);
  assert.equal(typeof context.window.PredictionBoostingView.render, 'function');
  for (const scene of [0, 1, 2]) {
    for (const width of [288, 346, 640]) {
      const html = view.render(scene, { width, height: width < 400 ? 260 : 450 });
      assert.doesNotMatch(svgMathText(html), /NaN|Infinity|undefined/);
      assert.match(svgMathText(html), /<svg\b/);
      const readout = view.readout(scene);
      assert.ok(readout.length > 0);
      assert.ok(readout.every(row => row.length >= 2 && typeof row[0] === 'string'));
      assert.ok(view.presentation(scene).caption.length > 0);
    }
  }
});


test('a zero residual has a finite deterministic fallback plane basis', () => {
  const basis = view.basisForResidual([0, 0, 0]);
  close(basis.radius, 0);
  close(squared(basis.e1), 1);
  close(squared(basis.e2), 1);
  close(math.dot(basis.e1, basis.e2), 0);
  assert.ok([...basis.e1, ...basis.e2].every(Number.isFinite));
});

test('small nonzero residuals retain an aligned orthonormal basis', () => {
  const residual = [1e-15, 2e-15, -1e-15];
  const basis = view.basisForResidual(residual);
  assert.ok(basis.radius > 0);
  close(squared(basis.e1), 1);
  close(squared(basis.e2), 1);
  close(math.dot(basis.e1, basis.e2), 0);
  basis.e1.forEach((value, index) => close(value, residual[index] / Math.sqrt(6e-30)));
});

test('late fitted rounds display positive loss and nonzero gradients instead of exact zeros', () => {
  // For α = 1, every two rounds divide the residual by four. Thus round 24
  // has r = (2^-23, -2^-23, 0) and L = 2^-46, rather than a perfect fit.
  const state = { rate: 1, round: 24 };
  const result = view.resolve(0, state);
  assert.equal(result.loss, 2 ** -46);
  assert.equal(result.residual[0], 2 ** -23);
  assert.equal(result.residual[1], -(2 ** -23));
  const perfect = view.resolve(2, { ...state, angle: 0, length: 1 });
  assert.equal(perfect.loss, 0);
  assert.equal(perfect.norm, 0);
  for (const width of [320, 640]) {
    const rows = view.readout(0, { ...state, width });
    assert.match(rows[2][1], /(?:e[−-]7|10\^−7)/);
    assert.match(rows[3][1], /1\.42 × 10\^−14/);
    assert.match(view.presentation(2, { ...state, width, length: 0 }).caption, /1\.42 × 10\^−14/);
    assert.equal(view.readout(2, { ...state, width, angle: 0, length: 1 })[3][1], '0; 0');
  }
});

test('inside, boundary and outside trials stay correctly classified at tiny radii', () => {
  // These cases use exact hand-derived circle intersections: along r, the
  // second intersection is s = 2; at a 60° effective angle it is s = 1.
  const cases = [
    [0, .5, /reduces loss|Loss decreases/],
    [120, .5, /reduces loss|Loss decreases/],
    [90, .5, /increases loss|Loss increases/],
    [0, 2.5, /increases loss|Loss increases/],
    [0, 2, /leaves loss unchanged|Loss is unchanged/],
    [60, 1, /leaves loss unchanged|Loss is unchanged/],
    [120, 1, /leaves loss unchanged|Loss is unchanged/],
    [120, 0, /leaves loss unchanged|Loss is unchanged/]
  ];
  for (const rate of [.25, .5, 1]) {
    for (const round of [0, 1, 2, 18, 24]) {
      for (const width of [320, 640]) {
        for (const [angle, length, expected] of cases) {
          const state = { rate, round, angle, length, width };
          assert.match(view.presentation(2, state).caption, expected, JSON.stringify(state));
        }
      }
    }
  }
});
