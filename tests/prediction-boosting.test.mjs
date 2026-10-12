import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { runInNewContext } from 'node:vm';
import test from 'node:test';

const require = createRequire(import.meta.url);
const math = require('../docs/shared/boosting-math.js');
const view = require('../docs/boosting/prediction-view.js');
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
      assert.match(html, /data-geometry="equal-loss-plane"/);
      const circle = attributes(/<circle\b[^>]*class="equal-loss-circle"[^>]*>/.exec(html)[0]);
      close(Number(circle['data-radius']), expected.radius);
      const svg = attributes(/<svg\b[^>]*>/.exec(html)[0]);
      const base = attributes(/<circle\b[^>]*class="prediction-base"[^>]*>/.exec(html)[0]);
      const target = attributes(/<path\b[^>]*class="prediction-target"[^>]*>/.exec(html)[0]);
      close(Number(circle.r), Number(svg['data-scale']) * expected.radius);
      close(Math.hypot(Number(base.cx) - Number(target['data-x']), Number(base.cy) - Number(target['data-y'])), Number(circle.r));
      close(Number(svg['data-scale']) * Number(svg['data-units-per-pixel']), 1);
    } else {
      assert.match(html, /data-geometry="orthographic-3d"/);
      assert.doesNotMatch(html, /equal-loss-circle/);
    }
  }
});

test('the geometry module runs without a module loader and keeps all three scenes responsive', () => {
  const context = { window: { BoostingMath: math } };
  runInNewContext(readFileSync(new URL('../docs/boosting/prediction-view.js', import.meta.url), 'utf8'), context);
  assert.equal(typeof context.window.PredictionBoostingView.render, 'function');
  for (const scene of [0, 1, 2]) {
    for (const width of [288, 346, 640]) {
      const html = view.render(scene, { width, height: width < 400 ? 260 : 450 });
      assert.doesNotMatch(html, /NaN|Infinity|undefined/);
      assert.match(html, /<svg\b/);
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
