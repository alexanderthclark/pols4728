import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
const model = require('../docs/trees/delivery.js');
const view = require('../docs/trees/delivery-view.js');
const story = require('../docs/trees/story.js');
const growth = require('../docs/trees/growth-view.js');
const attributes = tag => Object.fromEntries(Array.from(tag.matchAll(/([\w-]+)="([^"]*)"/g), match => [match[1], match[2]]));
const groups = svg => Array.from(svg.matchAll(/<g\b([^>]*)>([\s\S]*?)<\/g>/g), match => ({ ...attributes(match[1]), contents: match[2] }));
const numbers = value => value === '' ? [] : value.split(',').map(Number);
const curve = (svg, name) => {
  const group = groups(svg).find(group => group['data-curve'] === name);
  assert.ok(group, `Missing ${name} curve`);
  return { values: numbers(group['data-values']), cuts: numbers(group['data-cuts']) };
};

test('the residual fitting and update scenes keep fixed axes and aligned observation distances', () => {
  for (const compact of [false, true]) {
    for (let scene = 1; scene < 5; scene++) {
      const svg = view.render(scene, {}, compact);
      const panels = groups(svg);
      const expectedAxes = [
        { axis: 'Residual (min)', x: '0,13', y: '-7,6' },
        { axis: 'Time (min)', x: '0,13', y: '0,20' }
      ];
      assert.deepEqual(panels.filter(group => group['data-axis']).map(group => ({
        axis: group['data-axis'], x: group['data-x-domain'], y: group['data-y-domain']
      })), expectedAxes);
      const observations = panels.find(group => group['data-points'] === 'observations');
      assert.deepEqual(numbers(observations['data-values']), model.y);
      const residuals = panels.find(group => group['data-points'] === 'residuals');
      assert.deepEqual(numbers(residuals['data-values']), model.rounds[scene < 3 ? 0 : 1].residual);
      const xs = group => Array.from(group.contents.matchAll(/<circle\b[^>]*>/g), match => Number(attributes(match[0]).cx));
      assert.equal(xs(observations).length, model.x.length);
      assert.deepEqual(xs(residuals), xs(observations));
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
  const finalSecond = view.render(4, { process: 3 });
  assert.match(finalSecond, /Two cutpoints, three intervals/);
  assert.deepEqual(curve(finalSecond, 'prediction'), { values: [6, 10, 18], cuts: [3.5, 8] });
  const nextCorrection = view.render(5, { continuation: 0 });
  assert.deepEqual(curve(nextCorrection, 'prediction'), curve(finalSecond, 'prediction'));
  assert.deepEqual(curve(nextCorrection, 'h3'), { values: [.6, -1], cuts: [8] });
});

test('update readouts report MSE for the currently displayed partial prediction', () => {
  assert.equal(view.readout(0), 'F₀(x) = ȳ = 12');
  assert.equal(view.readout(0, { process: 0 }), 'Delivery time vs distance');
  assert.equal(view.readout(1), 'h₁: −3 if x ≤ 8; +5 if x > 8');
  assert.equal(view.readout(3), 'h₂: −3 if x ≤ 3.5; +1 if x > 3.5');
  for (const [correction, firstMSE, secondMSE] of [[0, 18.75, 3.75], [0.5, 7.5, 1.5], [1, 3.75, 0.75]]) {
    assert.equal(view.readout(2, { correction }), `F₁ = F₀ + ${correction} × h₁ · training MSE = ${firstMSE}`);
    assert.equal(view.readout(4, { correction }), `F₂ = F₁ + ${correction} × h₂ · training MSE = ${secondMSE} (was 3.75)`);
  }
});

test('fitting reveals residuals and groups before the fitted means without changing the current prediction', () => {
  for (const scene of [1, 3]) {
    const round = scene === 1 ? 1 : 2;
    const fixedPrediction = round === 1 ? { values: [12], cuts: [] } : { values: [9, 17], cuts: [8] };
    for (const process of [0, 1, 2, 3]) {
      const svg = view.render(scene, { process });
      assert.deepEqual(curve(svg, 'previous'), fixedPrediction);
      assert.equal(groups(svg).some(group => group['data-curve'] === `h${round}`), process === 3);
      if (process >= 1) {
        const residuals = groups(svg).find(group => group['data-points'] === 'residuals');
        assert.deepEqual(numbers(residuals['data-values']), model.rounds[round - 1].residual);
      }
    }
  }
  const middle = view.render(3, { process: 0.5 });
  const residuals = groups(middle).find(group => group['data-points'] === 'residuals');
  assert.deepEqual(numbers(residuals['data-values']), [-4.5, -4.5, 0.5, 0.5, 0.5, 2.5, 2.5, 2.5]);
});

test('adding uses a fixed correction and fixed targets while only accumulated predictions move', () => {
  for (const scene of [2, 4]) {
    const round = scene === 2 ? 1 : 2;
    for (const process of [0, 1, 1.5, 2, 3]) {
      const svg = view.render(scene, { process });
      const panels = groups(svg);
      assert.deepEqual(numbers(panels.find(group => group['data-points'] === 'observations')['data-values']), model.y);
      assert.deepEqual(numbers(panels.find(group => group['data-points'] === 'residuals')['data-values']), model.rounds[round - 1].residual);
      const prediction = curve(svg, 'prediction');
      const bounds = [0, ...prediction.cuts, 13];
      const amount = Math.max(0, Math.min(1, process - 1));
      assert.deepEqual(prediction.values, bounds.slice(0, -1).map((lo, i) => model.predict(round, (lo + bounds[i + 1]) / 2, amount)));
      assert.deepEqual(curve(svg, `h${round}`), {
        values: [model.rounds[round - 1].left, model.rounds[round - 1].right], cuts: [model.rounds[round - 1].threshold]
      });
    }
  }
  assert.match(view.render(4, { process: 3 }), /data-worsened-errors="9,10,12"/);
});

test('candidate exploration changes the trial stump while preserving its residual target and prior prediction', () => {
  for (const scene of [1, 3]) {
    const round = scene === 1 ? 1 : 2;
    for (const candidateIndex of [0, 4, 6]) {
      const svg = view.render(scene, { process: 0, candidateIndex });
      const candidate = model.candidates(round)[candidateIndex];
      assert.deepEqual(curve(svg, 'candidate'), { values: [candidate.left, candidate.right], cuts: [candidate.threshold] });
      const residuals = groups(svg).find(group => group['data-points'] === 'residuals');
      assert.deepEqual(numbers(residuals['data-values']), model.rounds[round - 1].residual);
      assert.deepEqual(curve(svg, 'previous'), round === 1 ? { values: [12], cuts: [] } : { values: [9, 17], cuts: [8] });
    }
  }
});

test('all ten integrated scenes render and route to the correct numerical sequence', () => {
  const state = { correction: 1, growth: 4 };
  for (const compact of [false, true]) {
    for (let scene = 0; scene < 10; scene++) {
      const svg = story.renderScene(scene, state, compact);
      const expected = scene < 6 ? view.render(scene, state, compact) : scene === 6 ? growth.renderTransition(compact) : growth.render(scene - 7, state, compact);
      assert.equal(svg, expected);
      assert.equal(story.readout(scene, state), scene < 6 ? view.readout(scene, state) : scene === 6 ? growth.transitionReadout() : growth.readout(scene - 7, state));
      assert.doesNotMatch(svg, /NaN|Infinity|undefined/);
      if (scene !== 6) {
        assert.match(svg, /<title\b[^>]*>[^<]+<\/title>/);
        assert.match(svg, /<desc\b[^>]*>[^<]+<\/desc>/);
      }
      for (const [, value] of svg.matchAll(/\s(?:x|y|x1|y1|x2|y2|cx|cy|width|height|r)="([^"]+)"/g)) {
        assert.ok(Number.isFinite(Number(value)), `Scene ${scene} contains invalid geometry: ${value}`);
      }
    }
  }
});


test('continuation spends time on the third correction before completing rounds four through eight', () => {
  assert.deepEqual(view.continuationState(0), { round: 2, amount: 1 });
  for (const progress of [.01, .1, .15]) assert.deepEqual(view.continuationState(progress), { round: 3, amount: 0 });
  assert.equal(view.continuationState(.275).round, 3);
  assert.ok(Math.abs(view.continuationState(.275).amount-.5)<1e-12);
  for (const progress of [.4, .45, .5]) assert.deepEqual(view.continuationState(progress), { round: 3, amount: 1 });
  for (let round = 2; round <= 8; round++) {
    const state = view.continuationState(view.continuationProgress(round));
    assert.deepEqual(state, { round, amount: 1 });
  }
  assert.deepEqual(view.continuationState(-1), { round: 2, amount: 1 });
  assert.deepEqual(view.continuationState(2), { round: 8, amount: 1 });
  assert.deepEqual(view.continuationState(.5999999999999999), { round: 4, amount: 1 });
  assert.deepEqual(view.continuationState(), { round: 8, amount: 1 });
  assert.throws(() => view.continuationProgress(1), RangeError);
  assert.throws(() => view.continuationProgress(9), RangeError);
});

test('continuation keeps observed outcomes above residuals on the original scales', () => {
  for (const compact of [false, true]) {
    for (const continuation of [0, .2, .45, .55, .7, 1]) {
      const svg = view.render(5, { continuation }, compact);
      const panels = groups(svg);
      const axes = panels.filter(group => group['data-axis']);
      assert.deepEqual(axes.map(group => ({ axis: group['data-axis'], x: group['data-x-domain'], y: group['data-y-domain'] })), [
        { axis: 'Time (min)', x: '0,13', y: '0,20' },
        { axis: 'Residual (min)', x: '0,13', y: '-7,6' }
      ]);
      assert.ok(numbers(axes[0]['data-panel'])[1] < numbers(axes[1]['data-panel'])[1]);
      const observed = panels.find(group => group['data-points'] === 'observations');
      const residuals = panels.find(group => group['data-points'] === 'residuals');
      assert.deepEqual(numbers(observed['data-values']), model.y);
      const { round, amount } = view.continuationState(continuation);
      const predictions = model.x.map(x => model.predict(round, x, amount));
      assert.deepEqual(numbers(residuals['data-values']), model.rounds[Math.max(3, round)-1].residual);
      assert.doesNotMatch(svg, /data-residual-reference|data-residual-stems/);
      assert.deepEqual(curve(svg, 'reference'), { values: [6, 10, 18], cuts: [3.5, 8] });
      const current = curve(svg, 'prediction');
      assert.deepEqual(current.cuts, [3.5, 8]);
      assert.deepEqual(current.values, [1.75, 5.75, 10.5].map(x => model.predict(round, x, amount)));
    }
  }
});

test('third update repairs the overshoot and later displayed MSE remains strictly positive', () => {
  const third = curve(view.render(5, { continuation: .45 }), 'prediction');
  third.values.forEach((value, i) => assert.ok(Math.abs(value - [6.6, 10.6, 17][i]) < 1e-12));
  assert.match(view.readout(5, { continuation: .45 }), /training MSE = 0\.15$/);
  assert.match(view.render(5, { continuation: .45 }), /h₃: \+0\.6 at x ≤ 8; −1 at x &gt; 8/);
  assert.match(view.readout(5), /training MSE = 0\.000048$/);
  let lastMSE = Infinity;
  for (let tick = 0; tick <= 100; tick++) {
    const progress = tick / 100;
    const svg = view.render(5, { continuation: progress });
    const stateTag = svg.match(/<g\b[^>]*data-continuation-round="[^"]*"[^>]*\/>/)[0];
    const mse = Number(attributes(stateTag)['data-training-mse']);
    assert.ok(mse > 0);
    assert.ok(mse <= lastMSE + 1e-12, `Training error increases at progress ${progress}`);
    assert.doesNotMatch(view.readout(5, { continuation: progress }), /training MSE = 0$/);
    lastMSE = mse;
  }
});


test('each continued update visibly fits a fixed dark orange stump and labels its selected split', () => {
  const checkpoints = [
    { round: 3, progress: [.1, .2, .275, .45] },
    { round: 4, progress: [.51, .55, .6] },
    { round: 5, progress: [.61, .65, .7] },
    { round: 6, progress: [.71, .75, .8] },
    { round: 7, progress: [.81, .85, .9] },
    { round: 8, progress: [.91, .95, 1] }
  ];
  for (const compact of [false, true]) {
    for (const { round, progress } of checkpoints) {
      const fitted = model.rounds[round - 1];
      for (const continuation of progress) {
        const svg = view.render(5, { continuation }, compact);
        assert.deepEqual(curve(svg, `h${round}`), { values: [fitted.left, fitted.right], cuts: [fitted.threshold] });
        const panels = groups(svg);
        const target = panels.find(group => group['data-points'] === 'residuals');
        assert.deepEqual(numbers(target['data-values']), fitted.residual);
        const fittedCurve = panels.find(group => group['data-curve'] === `h${round}`);
        assert.match(fittedCurve.contents, /stroke="#A54F32"/);
        const selectedCut = panels.find(group => group['data-selected-split']);
        assert.equal(Number(selectedCut['data-selected-split']), fitted.threshold);
        assert.match(selectedCut.contents, /stroke="#A54F32"/);
        assert.match(selectedCut.contents, new RegExp(`Split at ${fitted.threshold} km`));
      }
    }
  }
});


test('the opening reveals the mean on a single observed-outcome plot without residuals', () => {
  assert.equal(view.topics[0], 'Predict delivery time');
  assert.equal(view.processMax(0), 1);
  assert.equal(view.processMax(1), 3);
  assert.deepEqual(view.processLabels(0), ['Observed deliveries', 'F₀(x) = ȳ = 12']);
  for (const compact of [false, true]) {
    for (const process of [0, .5, 1]) {
      const svg = view.render(0, { process }, compact);
      const panels = groups(svg);
      assert.equal(panels.filter(group => group['data-axis']).length, 1);
      assert.equal(panels.find(group => group['data-axis'])['data-axis'], 'Time (min)');
      assert.doesNotMatch(svg, /data-points="residuals"|data-axis="Residual|MSE/);
      assert.deepEqual(numbers(panels.find(group => group['data-points'] === 'observations')['data-values']), model.y);
      assert.equal(panels.some(group => group['data-curve'] === 'prediction'), process > 0);
      if (process > 0) assert.deepEqual(curve(svg, 'prediction'), { values: [12], cuts: [] });
    }
  }
});

test('the completed opening and start of residual formation share exact observed and mean coordinates', () => {
  const geometry = (svg, kind, name) => groups(svg).find(group => group[kind] === name).contents;
  const coordinates = contents => [...contents.matchAll(/<(?:line|circle)\b[^>]*>/g)].map(match => {
    const fields = attributes(match[0]);
    return Object.fromEntries(['x1', 'y1', 'x2', 'y2', 'cx', 'cy', 'r'].filter(key => key in fields).map(key => [key, fields[key]]));
  });
  for (const compact of [false, true]) {
    const first = view.render(0, { process: 1 }, compact);
    const next = view.render(1, { process: 0 }, compact);
    assert.deepEqual(coordinates(geometry(first, 'data-points', 'observations')), coordinates(geometry(next, 'data-points', 'observations')));
    assert.deepEqual(coordinates(geometry(first, 'data-curve', 'prediction')), coordinates(geometry(next, 'data-curve', 'previous')));
    assert.doesNotMatch(next, /data-axis="Residual|data-residual-transfer/);
    const moving = view.render(1, { process: .15 }, compact);
    assert.doesNotMatch(moving, /data-axis="Residual|data-residual-transfer/);
    const formed = view.render(1, { process: .6 }, compact);
    assert.match(formed, /data-axis="Residual \(min\)"/);
    assert.match(formed, /data-residual-transfer="true"/);
    assert.doesNotMatch(formed, /data-curve="h1"/);
    const settled = view.render(1, { process: 1 }, compact);
    assert.deepEqual(coordinates(geometry(formed, 'data-points', 'observations')), coordinates(geometry(settled, 'data-points', 'observations')));
  }
});
