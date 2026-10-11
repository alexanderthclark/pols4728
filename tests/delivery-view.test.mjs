import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
const model = require('../docs/trees/delivery.js');
const view = require('../docs/trees/delivery-view.js');
const story = require('../docs/trees/story.js');
const attributes = tag => Object.fromEntries(Array.from(tag.matchAll(/([\w-]+)="([^"]*)"/g), match => [match[1], match[2]]));
const groups = svg => Array.from(svg.matchAll(/<g\b([^>]*)>([\s\S]*?)<\/g>/g), match => ({ ...attributes(match[1]), contents: match[2] }));
const numbers = value => value === '' ? [] : value.split(',').map(Number);
const curve = (svg, name) => {
  const group = groups(svg).find(group => group['data-curve'] === name);
  assert.ok(group, `Missing ${name} curve`);
  return { values: numbers(group['data-values']), cuts: numbers(group['data-cuts']) };
};

test('all six delivery scenes keep fixed axes and aligned observation distances', () => {
  for (const compact of [false, true]) {
    for (let scene = 0; scene < 6; scene++) {
      const svg = view.render(scene, {}, compact);
      const panels = groups(svg);
      assert.deepEqual(panels.filter(group => group['data-axis']).map(group => ({
        axis: group['data-axis'], x: group['data-x-domain'], y: group['data-y-domain']
      })), [
        { axis: 'Residual (min)', x: '0,13', y: '-7,6' },
        { axis: 'Time (min)', x: '0,13', y: '0,20' }
      ]);
      const observations = panels.find(group => group['data-points'] === 'observations');
      assert.deepEqual(numbers(observations['data-values']), model.y);
      const residuals = panels.find(group => group['data-points'] === 'residuals');
      if (scene === 5) {
        assert.equal(residuals, undefined);
      } else {
        assert.deepEqual(numbers(residuals['data-values']), model.rounds[scene < 3 ? 0 : 1].residual);
        const xs = group => Array.from(group.contents.matchAll(/<circle\b[^>]*>/g), match => Number(attributes(match[0]).cx));
        assert.equal(xs(observations).length, model.x.length);
        assert.deepEqual(xs(residuals), xs(observations));
      }
    }
  }
});

test('each correction is fitted to the shown residuals and added to the previous function', () => {
  assert.deepEqual(curve(view.render(0), 'prediction'), { values: [12], cuts: [] });
  assert.deepEqual(curve(view.render(1), 'h1'), { values: [-3, 5], cuts: [8] });
  assert.deepEqual(curve(view.render(1), 'previous'), { values: [12], cuts: [] });
  assert.deepEqual(curve(view.render(3), 'h2'), { values: [-3, 1], cuts: [3.5] });
  assert.deepEqual(curve(view.render(3), 'previous'), { values: [9, 17], cuts: [8] });
  for (const compact of [false, true]) {
    for (const correction of [0, 1]) {
      const first = view.render(2, { correction }, compact);
      assert.deepEqual(curve(first, 'h1'), { values: [-3, 5], cuts: [8] });
      assert.deepEqual(curve(first, 'previous'), { values: [12], cuts: [] });
      assert.deepEqual(curve(first, 'prediction'), { values: correction ? [9, 17] : [12, 12], cuts: [8] });
      const second = view.render(4, { correction }, compact);
      assert.deepEqual(curve(second, 'h2'), { values: [-3, 1], cuts: [3.5] });
      assert.deepEqual(curve(second, 'previous'), { values: [9, 17], cuts: [8] });
      assert.deepEqual(curve(second, 'prediction'), { values: correction ? [6, 10, 18] : [9, 9, 17], cuts: [3.5, 8] });
    }
  }
  const summary = view.render(5);
  assert.deepEqual(curve(summary, 'h1'), { values: [-3, 5], cuts: [8] });
  assert.deepEqual(curve(summary, 'h2'), { values: [-3, 1], cuts: [3.5] });
  assert.deepEqual(curve(summary, 'prediction'), { values: [6, 10, 18], cuts: [3.5, 8] });
});

test('update readouts report MSE for the currently displayed partial prediction', () => {
  assert.equal(view.readout(0), 'F₀ = mean(y) = 12 min · MSE = 18.75');
  assert.equal(view.readout(1), 'h₁: −3 if x ≤ 8; +5 if x > 8');
  assert.equal(view.readout(3), 'h₂: −3 if x ≤ 3.5; +1 if x > 3.5');
  for (const [correction, firstMSE, secondMSE] of [[0, 18.75, 3.75], [0.5, 7.5, 1.5], [1, 3.75, 0.75]]) {
    assert.equal(view.readout(2, { correction }), `F₁ = F₀ + ${correction} × h₁ · MSE = ${firstMSE}`);
    assert.equal(view.readout(4, { correction }), `F₂ = F₁ + ${correction} × h₂ · MSE = ${secondMSE}`);
  }
});

test('all thirteen integrated scenes render and route to the correct numerical sequence', () => {
  const state = { x: 0.8, y: 0.8, addition: 1, interaction: false, splitProgress: 1, correction: 1 };
  for (const compact of [false, true]) {
    for (let scene = 0; scene < 13; scene++) {
      const svg = story.renderScene(scene, state, compact);
      const expected = scene < 6 ? view.render(scene, state, compact) : story.renderPartitionScene(scene - 5, state, compact);
      assert.equal(svg, expected);
      assert.equal(story.readout(scene, state), scene < 6 ? view.readout(scene, state) : story.partitionReadout(scene - 5, state));
      assert.doesNotMatch(svg, /NaN|Infinity|undefined/);
      assert.match(svg, /<title\b[^>]*>[^<]+<\/title>/);
      assert.match(svg, /<desc\b[^>]*>[^<]+<\/desc>/);
      for (const [, value] of svg.matchAll(/\b(?:x|y|x1|y1|x2|y2|cx|cy|width|height|r)="([^"]+)"/g)) {
        assert.ok(Number.isFinite(Number(value)), `Scene ${scene} contains invalid geometry: ${value}`);
      }
    }
  }
});
