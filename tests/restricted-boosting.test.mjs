import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { runInNewContext } from 'node:vm';
import test from 'node:test';

const require = createRequire(import.meta.url);
const math = require('../docs/shared/boosting-math.js');
const view = require('../unpublished/boosting/restricted-view.js');
const close = (actual, expected, tolerance = 1e-9) => assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} should equal ${expected}`);
const vectorClose = (actual, expected) => {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => close(value, expected[index]));
};
const attributes = fragment => Object.fromEntries([...fragment.matchAll(/([\w-]+)="([^"]*)"/g)].map(match => [match[1], match[2]]));
const dataValue = (html, name) => new RegExp(`data-${name}="([^"]+)"`).exec(html)?.[1];

// These expectations are independent, hand-calculated cases. The duplicate x
// values leave one constant leaf, so its residual mean is the only fitted tree.
test('identical predictors restrict every tree correction to equal coordinates', () => {
  const cases = [
    [0, [1, 0], [.5, .5], .5, 1],
    [.25, [.75, -.25], [.25, .25], .3125, Math.sqrt(.625)],
    [.5, [.5, -.5], [0, 0], .25, Math.sqrt(.5)],
    [1, [0, -1], [-.5, -.5], .5, 1]
  ];
  for (const [c, residual, correction, loss, norm] of cases) {
    const result = view.resolve(0, { prediction: c });
    vectorClose(result.x, [1, 1]);
    vectorClose(result.target, [1, 0]);
    vectorClose(result.prediction, [c, c]);
    vectorClose(result.residual, residual);
    vectorClose(result.gradient, residual.map(value => -value));
    vectorClose(result.nextTree, correction);
    vectorClose(result.nextPrediction, [.5, .5]);
    close(result.loss, loss);
    close(result.norm, norm);
    close(result.nextLoss, .25);
    close(result.minLoss, .25);
    assert.equal(result.tree.threshold, null, 'Identical x values cannot create a split');
  }
});

test('the irreducible residual is perpendicular to every representable update', () => {
  const result = view.resolve(0, { prediction: .5 });
  vectorClose(result.optimalPrediction, [.5, .5]);
  vectorClose(result.optimalResidual, [.5, -.5]);
  for (const coefficient of [-2, -.5, 0, .25, 3]) {
    close(math.dot(result.residual, [coefficient, coefficient]), 0);
    const movedLoss = .25 + coefficient * coefficient;
    assert.ok(movedLoss >= result.loss);
    if (coefficient !== 0) assert.ok(movedLoss > result.loss);
  }
  vectorClose(result.nextTree, [0, 0]);
  vectorClose(result.nextPrediction, result.prediction);
  close(result.nextLoss, result.loss);
});

test('moving on the diagonal minimizes half-squared loss exactly at its projection', () => {
  for (let index = 0; index <= 100; index++) {
    const c = index / 100;
    const result = view.resolve(0, { prediction: c });
    // L(c,c) = 1/4 + (c - 1/2)^2. This separates the fixed perpendicular
    // component from the correctable component along the diagonal.
    close(result.loss, .25 + (c - .5) ** 2);
    close(result.norm ** 2, 2 * result.loss);
    close(result.loss - result.nextLoss, (c - .5) ** 2);
    close(math.dot(result.residual, result.nextTree), 2 * (c - .5) ** 2);
    vectorClose(result.nextPrediction, [.5, .5]);
  }
});

test('the rendered diagonal, residuals and loss circle describe the computed two-dimensional state', () => {
  for (const width of [288, 346, 640]) {
    const height = width < 400 ? 260 : 450;
    let referenceScale;
    for (const c of [0, .25, .5, 1]) {
      const result = view.resolve(0, { prediction: c });
      const html = view.render(0, { prediction: c, width, height }, width < 400);
      assert.doesNotMatch(html, /NaN|Infinity|undefined/);
      assert.equal(Number(dataValue(html, 'restricted-scene')), 0);
      for (const [attribute, expected] of [
        ['prediction', result.prediction], ['residual', result.residual],
        ['gradient', result.gradient], ['next-tree', result.nextTree],
        ['next-prediction', result.nextPrediction]
      ]) vectorClose(JSON.parse(dataValue(html, attribute)), expected);
      close(Number(dataValue(html, 'loss')), result.loss);
      close(Number(dataValue(html, 'norm')), result.norm);
      close(Number(dataValue(html, 'min-loss')), .25);
      const svg = attributes(/<svg\b[^>]*>/.exec(html)[0]);
      assert.ok(Number(svg['data-scale']) > 0);
      close(Number(svg['data-scale']) * Number(svg['data-units-per-pixel']), 1);
      assert.equal(svg['data-x-domain'], svg['data-y-domain']);
      if (referenceScale === undefined) referenceScale = Number(svg['data-scale']);
      close(Number(svg['data-scale']), referenceScale, 1e-9);
      const circle = attributes(/<circle\b[^>]*class="restricted-loss-circle"[^>]*>/.exec(html)[0]);
      const current = attributes(/<circle\b[^>]*class="restricted-current"[^>]*>/.exec(html)[0]);
      const target = attributes(/<path\b[^>]*class="restricted-target"[^>]*>/.exec(html)[0]);
      const projection = attributes(/<rect\b[^>]*class="[^"]*restricted-projection[^>]*>/.exec(html)[0]);
      const diagonal = attributes(/<line\b[^>]*class="restricted-available-line"[^>]*>/.exec(html)[0]);
      close(Number(circle.r), result.norm * Number(svg['data-scale']));
      close(Number(circle.cx), Number(target['data-pixel-x']));
      close(Number(circle.cy), Number(target['data-pixel-y']));
      close(Math.hypot(Number(current.cx) - Number(circle.cx), Number(current.cy) - Number(circle.cy)), Number(circle.r));
      vectorClose(projection['data-values'].split(',').map(Number), [.5, .5]);
      close(Number(diagonal.x2) - Number(diagonal.x1), Number(diagonal.y1) - Number(diagonal.y2));
      const dx = Number(target['data-pixel-x']) - Number(projection['data-pixel-x']);
      const dy = Number(target['data-pixel-y']) - Number(projection['data-pixel-y']);
      close(dx, dy, 1e-9);
      if (c === .5) close(Math.hypot(dx, dy), Number(circle.r));
      assert.match(html, /restricted-residual-arrow/);
      if (c === .5) assert.doesNotMatch(html, /class="restricted-tree-arrow"/);
      else assert.match(html, /restricted-tree-arrow/);
    }
  }
});

test('the display distinguishes a zero fitted correction from its nonzero residual and loss', () => {
  const state = { prediction: .5 };
  const html = view.render(0, state);
  const rows = view.readout(0, state);
  assert.ok(rows.length >= 3);
  assert.ok(rows.every(row => row.length >= 2 && typeof row[0] === 'string'));
  assert.match(rows.map(row => row[1]).join(' '), /0\.5|1\/2/);
  assert.match(rows.map(row => row[1]).join(' '), /[−-](?:0\.5|1\/2)/);
  assert.match(view.presentation(0, state).caption, /zero|no further|cannot reduce/i);
  assert.match(html, /zero|no further|cannot reduce/i);
  assert.ok(view.topics[0].length > 0);
});

test('the restricted-input visual runs without a module loader', () => {
  const context = { window: { BoostingMath: math } };
  runInNewContext(readFileSync(new URL('../unpublished/boosting/restricted-view.js', import.meta.url), 'utf8'), context);
  assert.equal(typeof context.window.RestrictedBoostingView.render, 'function');
  assert.equal(typeof context.window.RestrictedBoostingView.resolve, 'function');
  const result = context.window.RestrictedBoostingView.resolve(0, { prediction: .5 });
  vectorClose(result.prediction, [.5, .5]);
  vectorClose(result.nextTree, [0, 0]);
  close(result.loss, .25);
});
