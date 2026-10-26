import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import test from 'node:test';
import { mse, predict, target } from '../docs/sgd/model.mjs';

globalThis.katex = createRequire(import.meta.url)('../docs/neural-networks/vendor/katex/katex.min.js');

const source = readFileSync(new URL('../docs/sgd/experiment.json', import.meta.url));
const experiment = JSON.parse(source);
const comparison = JSON.parse(readFileSync(new URL('../docs/sgd/initializations.json', import.meta.url)));
const { config, runs, x } = comparison;
const close = (actual, expected, tolerance = 1e-12) =>
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} should equal ${expected}`);
const attributes = tag => Object.fromEntries([...tag.matchAll(/([\w:-]+)="([^"]*)"/g)].map(match => match.slice(1)));
const tags = (markup, name, className) => [...markup.matchAll(new RegExp(`<${name}\\b[^>]*>`, 'g'))]
  .map(match => ({ markup: match[0], ...attributes(match[0]) }))
  .filter(tag => !className || tag.class?.split(/\s+/).includes(className));
const svgViews = markup => [...markup.matchAll(/<svg\b[^>]*>[\s\S]*?<\/svg>/g)]
  .map(match => ({ html: match[0], ...attributes(match[0].match(/^<svg\b[^>]*>/)[0]) }));
const figures = markup => [...markup.matchAll(/<figure\b[^>]*>[\s\S]*?<\/figure>/g)]
  .map(match => ({ html: match[0], ...attributes(match[0].match(/^<figure\b[^>]*>/)[0]) }))
  .filter(figure => figure.class?.split(/\s+/).includes('initialization-fit'));

test('the initialization comparison preserves every recorded start on the original training data', () => {
  assert.equal(config.estimator, 'sklearn.neural_network.MLPRegressor');
  assert.equal(config.source, 'experiment.json');
  assert.equal(config.sourceSHA256, createHash('sha256').update(source).digest('hex'));
  assert.equal(config.n, experiment.data.length);
  assert.equal(config.n, 2048);
  assert.deepEqual(config.widths, [2, 12]);
  assert.deepEqual(config.seeds, [1, 2, 3, 4, 5, 6, 7, 8]);
  assert.equal(config.runsPerWidth, 8);
  assert.equal(runs.length, 16);
  for (const width of config.widths) {
    const group = runs.filter(run => run.width === width);
    assert.deepEqual(group.map(run => run.seed), config.seeds);
    assert.deepEqual(group.map(run => run.run), [1, 2, 3, 4, 5, 6, 7, 8]);
    assert.ok(group.every(run => run.model.units.length === width));
    assert.ok(group.every(run => run.initialModel.units.length === width));
  }
  for (const version of Object.values(config.runtime)) assert.match(version, /^\d+\.\d+/);
  assert.match(config.batchOrderSHA256, /^[a-f0-9]{64}$/);
});

test('the comparison uses one optimizer and equal training budgets while isolating initialization', () => {
  const settings = config.estimatorSettings;
  assert.equal(config.solver, 'sgd');
  assert.equal(settings.solver, config.solver);
  assert.equal(settings.activation, 'relu');
  assert.equal(settings.alpha, 0);
  assert.equal(settings.shuffle, false, 'External common batch permutations must not be shuffled again by each estimator');
  assert.equal(settings.early_stopping, false);
  assert.equal(settings.batch_size, config.batchSize);
  assert.equal(settings.max_iter, 1, 'Each fit call makes one minibatch update');
  assert.equal(settings.warm_start, true, 'One-batch fit calls retain the learned parameters');
  assert.equal(settings.momentum, 0, 'A fresh optimizer on each fit call must not discard momentum');
  assert.equal(settings.nesterovs_momentum, false);
  assert.equal(settings.learning_rate_init, .05);
  assert.equal(config.learningRateSchedule, '0.05 / (1 + step / 6000)');
  assert.equal(config.epochs, 300);
  assert.equal(config.batchSize, 64);
  const updates = config.epochs * config.n / config.batchSize;
  assert.equal(config.updatesPerRun, updates);
  for (const run of runs) {
    assert.equal(run.epochs, config.epochs);
    assert.equal(run.updates, updates);
    assert.equal(run.observationVisits, config.epochs * config.n);
    assert.equal(run.converged, null, 'A fixed epoch budget does not establish convergence');
    assert.equal(run.stopReason, 'fixed epoch budget');
  }
});

test('exported initial and final sklearn parameters reproduce every plotted prediction and both reported errors', () => {
  assert.equal(x.length, 401);
  const metricRows = Array.from({ length: 4000 }, (_, i) => {
    const input = -1 + 2 * (i + .5) / 4000;
    return { x: input, y: target(input) };
  });
  x.forEach((input, i) => close(input, -1 + 2 * i / 400));
  for (const run of runs) {
    for (const [model, predictions] of [[run.model, run.predictions], [run.initialModel, run.initialPredictions]]) {
      assert.equal(predictions.length, x.length);
      assert.ok(Number.isFinite(model.c));
      for (const unit of model.units) {
        for (const parameter of ['w', 'b', 'a']) assert.ok(Number.isFinite(unit[parameter]));
      }
      x.forEach((input, i) => close(predictions[i], predict(model, input)));
    }
    assert.notDeepEqual(run.initialModel, run.model, 'The initial parameters must be saved before training');
    close(run.trainMSE, mse(run.model, experiment.data));
    close(run.gridMSE, mse(run.model, metricRows));
  }
});

test('curve colors identify network width without an error gradient', async () => {
  const { initializationColor } = await import('../docs/sgd/initializations.mjs');
  assert.equal(initializationColor(2), '#a54f32');
  assert.equal(initializationColor(12), '#234e70');
});

test('both gallery views use shared axes that contain all of their functions', async () => {
  const { initializationDomains } = await import('../docs/sgd/initializations.mjs');
  const domains = initializationDomains(comparison);
  for (const [name, domain] of Object.entries(domains)) {
    assert.ok(domain.every(Number.isFinite) && domain[0] < domain[1], name);
    assert.ok(domain[0] <= 0 && domain[1] >= 1, 'The target step must fit both views');
    for (const run of runs) {
      const values = name === 'expanded' ? [...run.predictions, ...run.initialPredictions] : run.predictions;
      assert.ok(values.every(value => value >= domain[0] && value <= domain[1]), `${name} must contain every applicable function`);
    }
  }
  assert.deepEqual(Object.keys(domains).sort(), ['expanded', 'final']);
  assert.ok(domains.expanded[0] <= domains.final[0]);
  assert.ok(domains.expanded[1] >= domains.final[1]);
  assert.ok(domains.expanded[1] - domains.expanded[0] > domains.final[1] - domains.final[0], 'The starting functions require a larger common scale');
});

function verifyCurve(path, predictions, targetPath) {
  // Recover the axes from the target's (-1,0), (0,0), (0,1), (1,1)
  // geometry, then verify the actual SVG curve rather than just its metadata.
  const geometry = targetPath.match(/^M([^,]+),([^H]+)H([^V]+)V([^H]+)H(.+)$/);
  assert.ok(geometry);
  const [left, zero, center, one, right] = geometry.slice(1).map(Number);
  close(center, (left + right) / 2);
  const points = [...path.matchAll(/[ML](-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)/g)]
    .map(match => match.slice(1).map(Number));
  assert.equal(points.length, x.length);
  points.forEach(([plotX, plotY], i) => {
    close(plotX, left + (x[i] + 1) * (right - left) / 2, .00051);
    close(plotY, zero + predictions[i] * (one - zero), .00051);
  });
}

test('the ending pairs exact final and initial functions in common coordinate systems', async () => {
  const { initializationColor, initializationDomains, renderInitializations } = await import('../docs/sgd/initializations.mjs');
  const html = renderInitializations(comparison);
  const domains = initializationDomains(comparison);
  assert.doesNotMatch(html, /NaN|Infinity|undefined|katex-error/);
  assert.doesNotMatch(html, /linear-gradient|coolwarm|initialization-scale/);
  const rendered = figures(html);
  assert.equal(rendered.length, runs.length);
  assert.equal(new Set(rendered.map(figure => `${figure['data-width']}:${figure['data-run']}`)).size, runs.length);
  const viewBoxes = new Set();
  const targetPaths = { final: new Set(), expanded: new Set() };
  for (const figure of rendered) {
    const run = runs.find(item => item.width === Number(figure['data-width']) && item.run === Number(figure['data-run']));
    assert.ok(run);
    close(Number(figure['data-mse']), run.trainMSE);
    assert.equal(figure['data-color'], initializationColor(run.width));
    const views = svgViews(figure.html);
    assert.equal(views.length, 2);
    assert.deepEqual(views.map(view => view['data-view']).sort(), ['expanded', 'final']);
    for (const view of views) {
      const name = view['data-view'];
      assert.equal(view.viewBox, '0 0 300 165');
      assert.ok(view['aria-label'] || view['aria-labelledby'], 'Every small plot needs an accessible description');
      assert.ok(view.class.split(/\s+/).includes(`initialization-view--${name}`));
      viewBoxes.add(view.viewBox);
      assert.deepEqual(view['data-x-domain'].split(',').map(Number), [-1, 1]);
      assert.deepEqual(view['data-y-domain'].split(',').map(Number), domains[name]);
      const curves = tags(view.html, 'path', 'initialization-curve');
      const starts = tags(view.html, 'path', 'initialization-start');
      const targets = tags(view.html, 'path', 'initialization-target');
      assert.equal(curves.length, 1);
      assert.equal(targets.length, 1);
      assert.equal(starts.length, name === 'expanded' ? 1 : 0);
      assert.equal(curves[0].stroke, figure['data-color']);
      targetPaths[name].add(targets[0].d);
      verifyCurve(curves[0].d, run.predictions, targets[0].d);
      if (name === 'expanded') {
        const [initial] = starts;
        assert.equal(initial.stroke, curves[0].stroke, 'A start and its final fit keep the same width color');
        assert.ok(Number(initial.opacity) > 0 && Number(initial.opacity) < 1, 'Initial predictions are faint');
        assert.ok(Number(initial['stroke-width']) < Number(curves[0]['stroke-width']), 'Initial predictions are thinner than final fits');
        assert.equal(initial['stroke-linecap'], 'round');
        assert.match(initial['stroke-dasharray'], /^1[ ,]+5$/, 'Initial predictions use a dotted stroke');
        verifyCurve(initial.d, run.initialPredictions, targets[0].d);
      }
      assert.match(view.html, /class="katex"/, 'Mathematical plot labels use the shared LaTeX renderer');
    }
  }
  assert.equal(viewBoxes.size, 1);
  for (const name of ['final', 'expanded']) assert.equal(targetPaths[name].size, 1, `The ${name} view must share one target and coordinate system across all runs`);
  for (const width of config.widths) {
    assert.deepEqual(rendered.filter(figure => Number(figure['data-width']) === width).map(figure => Number(figure['data-run'])), [1, 2, 3, 4, 5, 6, 7, 8]);
  }
});

test('show initialization is an unchecked native control with a visible label', async () => {
  const { renderInitializations } = await import('../docs/sgd/initializations.mjs');
  const html = renderInitializations(comparison);
  const controls = tags(html, 'input').filter(input => input.id === 'show-initialization');
  assert.equal(controls.length, 1);
  assert.equal(controls[0].type, 'checkbox');
  assert.doesNotMatch(controls[0].markup, /\b(?:checked|disabled|hidden)(?:[=\s>])/);
  assert.match(html, /<label\b[^>]*for="show-initialization"[^>]*>[\s\S]*?Show initialization[\s\S]*?<\/label>/);
  assert.doesNotMatch(html, /onclick=|onchange=/, 'The native toggle does not depend on scripted event handlers');
});

test('the static closing gallery follows the lesson and precedes references without changing the focused chapter', async () => {
  const { renderInitializationReading } = await import('../docs/sgd/initializations.mjs');
  const page = readFileSync(new URL('../docs/sgd/index.html', import.meta.url), 'utf8');
  const focusedPage = readFileSync(new URL('../docs/sgd/batches.html', import.meta.url), 'utf8');
  assert.equal((page.match(/id="initializations"/g) || []).length, 1);
  assert.ok(page.indexOf('id="initializations"') > page.indexOf('id="compare"'));
  assert.ok(page.indexOf('id="initializations"') < page.indexOf('id="references"'));
  assert.equal(figures(page).length, 16);
  assert.equal(figures(focusedPage).length, 0);
  assert.doesNotMatch(focusedPage, /id="initializations"/);
  const reading = renderInitializationReading(comparison);
  assert.ok(reading.length > 0);
  assert.ok(page.includes(reading), 'The full ending explanation is present in the reading version');
  assert.doesNotMatch(reading, /NaN|undefined|katex-error/);
});
