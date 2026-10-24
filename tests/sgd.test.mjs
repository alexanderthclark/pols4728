import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import test from 'node:test';
import vm from 'node:vm';
import { copy, data, gradient, initialize, loss, mse, predict, random, shuffle, target, update } from '../docs/sgd/model.mjs';
import { epochState, renderEpochCounter } from '../docs/sgd/epoch-counter.mjs';
import { scenes, foundationScenes, batchSteps } from '../docs/sgd/scenes.mjs';
import { render } from '../docs/sgd/views.mjs';
import { constructionKinds } from '../docs/sgd/construction.mjs';

globalThis.katex = createRequire(import.meta.url)('../docs/neural-networks/vendor/katex/katex.min.js');

const experiment = JSON.parse(readFileSync(new URL('../docs/sgd/experiment.json', import.meta.url)));
const close = (actual, expected, tolerance = 1e-12) =>
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} should equal ${expected}`);

// Inspect the real SVG output and numerical frame without making layout claims.
function renderScene(id, progress, width = 850, height = 430) {
  const svg = (name, h) => ({
    id: name, innerHTML: '', attrs: {},
    getBoundingClientRect() { return { width, height: h }; },
    setAttribute(key, value) { this.attrs[key] = value; }
  });
  const plot = svg('plot', height), detail = svg('detail', 110);
  const scene = scenes.find(item => item.id === id);
  assert.ok(scene, `Unknown scene ${id}`);
  const frame = render(plot, detail, scene, progress, experiment);
  assert.doesNotMatch(plot.innerHTML + detail.innerHTML, /NaN|Infinity|undefined/);
  return { frame, plot: plot.innerHTML, detail: detail.innerHTML, plotLabel: plot.attrs['aria-label'] };
}

test('the opening introduces the actual step target before constructing ReLUs and reverses across its jump', async () => {
  assert.equal(scenes[0].id, 'step-target');
  assert.equal(scenes[0].kind, 'step-target');
  assert.equal(scenes[1].id, 'pair');
  const snapshots = new Map();
  for (const progress of [0, .25, .5, .50001, .75, 1]) {
    const result = renderScene('step-target', progress);
    assert.equal(result.frame.probeY, target(result.frame.probeX));
    assert.equal(result.detail, '', 'The target comes before any network diagram');
    snapshots.set(progress, result);
  }
  close(snapshots.get(0).frame.probeX, -.8);
  close(snapshots.get(.5).frame.probeX, 0);
  close(snapshots.get(1).frame.probeX, .8);
  assert.equal(snapshots.get(.5).frame.probeY, 0, 'The target is zero at the threshold itself');
  assert.equal(snapshots.get(.50001).frame.probeY, 1, 'The upper branch starts strictly to the right of zero');
  for (const [progress, result] of [...snapshots].reverse()) {
    assert.deepEqual(renderScene('step-target', progress), result);
  }
  const opening = snapshots.get(0);
  const marker = kind => {
    const element = opening.plot.match(new RegExp(`<circle\\b[^>]*data-target-boundary="${kind}"[^>]*>`))?.[0];
    assert.ok(element, `${kind} must be visibly marked`);
    return Object.fromEntries([...element.matchAll(/([\w-]+)="([^"]*)"/g)].map(match => match.slice(1)));
  };
  const closed = marker('closed-zero'), open = marker('open-one');
  assert.equal(closed.cx, open.cx);
  assert.ok(Number(closed.cy) > Number(open.cy), 'Both threshold endpoints share x=0, with one above zero');
  assert.ok(closed.fill && !['white', '#fff', '#ffffff', 'none'].includes(closed.fill));
  assert.equal(open.fill, 'white');
  assert.match(opening.plotLabel, /step/i);
  assert.match(opening.plotLabel, /x\s*(?:≤|<=)\s*0/);
  assert.match(opening.plotLabel, /x\s*>\s*0/);

  const ui = await controller();
  const left = ui.at('step-target', .25), right = ui.at('step-target', .75);
  assert.equal(left.probeY, 0);
  assert.equal(right.probeY, 1);
  assert.deepEqual(ui.at('step-target', .25), left, 'Reverse scrolling restores the lower branch');
  assert.equal(ui.nodes.get('epoch-counter').hidden, true);
  ui.reduce(true);
  const reduced = ui.at('step-target', .1);
  assert.equal(reduced.probeY, target(reduced.probeX));
  ui.emit('document:keydown', { key: 'ArrowRight', target: { tagName: 'DIV' }, preventDefault() {} });
  assert.equal(ui.nodes.get('stage').dataset.scene, 'pair');
});

test('the target reference accompanies the combined output without obscuring the individual ReLUs', () => {
  const targetPath = result => result.plot.match(/<path data-target="step"[^>]*>/g) || [];
  for (const id of ['pair', 'input-first', 'input-second', 'flip']) {
    for (const progress of [0, .5, 1]) assert.equal(targetPath(renderScene(id, progress)).length, 0);
  }
  for (const progress of [0, .3]) assert.equal(targetPath(renderScene('add', progress)).length, 0);
  const revealed = renderScene('add', .5);
  assert.ok(revealed.frame.reveal > 0);
  assert.equal(targetPath(revealed).length, 1);
  assert.match(revealed.plotLabel, /target step/);
  const reference = targetPath(renderScene('add', 1))[0];
  assert.ok(reference);
  for (const progress of [0, .5, 1]) {
    const shifted = renderScene('output-bias', progress);
    assert.deepEqual(targetPath(shifted), [reference], 'Changing the output bias must not move or duplicate the target');
    assert.match(shifted.plotLabel, /target step/);
  }
  assert.equal(targetPath(renderScene('add', .3)).length, 0, 'Reverse scrolling restores the separate-contribution view');
});

test('each of the first two epochs partitions the same data and preserves the preceding update', () => {
  for (const width of [2, 12]) {
    const trace = experiment.traces[width];
    const rng = random(10001);
    for (const epoch of [1, 2]) {
      const snapshots = trace.filter(snapshot => snapshot.epoch === epoch);
      assert.equal(snapshots.length, 32);
      const indices = snapshots.flatMap(snapshot => snapshot.indices);
      assert.equal(indices.length, data.length);
      assert.equal(new Set(indices).size, data.length);
      assert.deepEqual(indices, shuffle(rng));
      snapshots.forEach((snapshot, index) => {
        assert.equal(snapshot.batch, index + 1);
        assert.equal(snapshot.step, (epoch - 1) * 32 + index);
        assert.equal(snapshot.indices.length, 64);
        if (snapshot.step > 0) {
          assert.deepEqual(snapshot.model, trace[snapshot.step - 1].after);
        }
      });
    }
    assert.notDeepEqual(trace[0].indices, trace[32].indices);
  }
});

const parameters = model => model.units.flatMap(unit => [unit.w, unit.b, unit.a]).concat(model.c);
const parameterNames = ['θ₁₁', 'θ₁₀', 'φ₁', 'θ₂₁', 'θ₂₀', 'φ₂', 'φ₀'];
function perturb(model, index, amount) {
  const changed = copy(model);
  if (index === model.units.length * 3) changed.c += amount;
  else changed.units[Math.floor(index / 3)][['w', 'b', 'a'][index % 3]] += amount;
  return changed;
}

test('the half-MSE objective averages half-squared point errors while reporting ordinary MSE', () => {
  const first = experiment.firstUpdate;
  const rows = first.indices.map(index => data[index]);
  const pointLosses = rows.map(row => .5 * (predict(first.before, row.x) - row.y) ** 2);
  close(pointLosses.reduce((sum, value) => sum + value, 0) / rows.length, first.batchBefore);
  close(first.batchBefore, loss(first.before, rows));
  close(mse(first.before, rows), 2 * first.batchBefore);
  assert.equal(first.batchBefore.toFixed(4), '0.2129');
  assert.equal(mse(first.before, rows).toFixed(4), '0.4257');
  const g = gradient(first.before, rows);
  const eps = 1e-6;
  parameters(g).forEach((value, index) => {
    const plus = perturb(first.before, index, eps), minus = perturb(first.before, index, -eps);
    close((loss(plus, rows) - loss(minus, rows)) / (2 * eps), value, 1e-9);
    close(rows.reduce((sum, row) => sum + parameters(gradient(first.before, [row]))[index], 0) / rows.length, value);
  });
  close(experiment.traces[2][0].eta, .05);
  close(first.before.c - experiment.traces[2][0].eta * g.c, first.after.c);
  assert.equal(first.after.c.toFixed(4), '0.5084');
  for (const width of [2, 12]) {
    for (const snapshot of experiment.traces[width]) {
      close(snapshot.loss, mse(snapshot.model));
      close(snapshot.afterLoss, mse(snapshot.after));
    }
  }
  for (const run of experiment.runs) close(run.loss, mse(run.model));
});

// Preserve the former squared-error derivative independently of the current
// objective. Replaying it at the former rate detects accidental changes to the
// learning-rate decay, update order, or gradients anywhere in training.
function legacyMSEGradient(model, rows) {
  const result = { c: 0, units: model.units.map(() => ({ w: 0, b: 0, a: 0 })) };
  for (const row of rows) {
    const errorDerivative = 2 * (predict(model, row.x) - row.y) / rows.length;
    result.c += errorDerivative;
    model.units.forEach((unit, index) => {
      const preactivation = unit.w * row.x + unit.b;
      result.units[index].a += errorDerivative * Math.max(0, preactivation);
      if (preactivation > 0) {
        result.units[index].w += errorDerivative * unit.a * row.x;
        result.units[index].b += errorDerivative * unit.a;
      }
    });
  }
  return result;
}

test('half-MSE with twice the learning rate preserves every recorded update across all 300 epochs', () => {
  for (const width of [2, 12]) {
    const model = initialize(width, 1), rng = random(10001);
    const recorded = new Map(experiment.traces[width].map(snapshot => [snapshot.step, snapshot]));
    for (let epoch = 0; epoch < 300; epoch++) {
      const order = shuffle(rng);
      for (let batch = 0; batch < 32; batch++) {
        const step = epoch * 32 + batch, snapshot = recorded.get(step);
        const rows = order.slice(batch * 64, (batch + 1) * 64).map(index => data[index]);
        const oldRate = .025 / (1 + step / 6000);
        if (snapshot) {
          assert.deepEqual(snapshot.model, model, `Width ${width}, step ${step}: preceding weights`);
          assert.equal(snapshot.eta, 2 * oldRate);
        }
        update(model, legacyMSEGradient(model, rows), oldRate);
        if (snapshot) assert.deepEqual(snapshot.after, model, `Width ${width}, step ${step}: updated weights`);
      }
    }
    assert.deepEqual(model, experiment.runs.find(run => run.width === width && run.seed === 1).model);
  }
});

test('epoch counters advance only after every update in the pass has completed', () => {
  for (const completed of [1, 2, 150, 300]) {
    const boundary = completed * 32;
    const before = epochState(boundary - 0.001);
    assert.equal(before.completedEpochs, completed - 1);
    assert.equal(before.currentEpoch, completed);
    assert.equal(before.batch, 32);
    assert.equal(before.finished, false);
    const after = epochState(boundary);
    assert.equal(after.completedEpochs, completed);
    assert.equal(after.currentEpoch, Math.min(300, completed + 1));
    assert.equal(after.batch, completed === 300 ? 32 : 1);
    assert.equal(after.finished, completed === 300);
  }
  assert.equal(epochState(1.9).usedUpdates, 1);
  assert.equal(epochState(-2).usedUpdates, 0);
  assert.equal(epochState(10000).usedUpdates, 9600);
  assert.throws(() => epochState(Number.NaN), TypeError);
});

test('counter text and accessible state rewind with the actual completed-update count', () => {
  const element = {
    classList: { add() {} }, dataset: {}, attrs: {}, innerHTML: '',
    style: { setProperty() {} },
    setAttribute(key, value) { this.attrs[key] = value; }
  };
  const forwards = new Map();
  for (const updates of [0, 1, 31, 32, 33, 63, 64, 9599, 9600]) {
    renderEpochCounter(element, epochState(updates));
    forwards.set(updates, element.innerHTML);
    assert.equal(Number(element.dataset.completedUpdates), updates);
    assert.equal(Number(element.dataset.completedEpochs), Math.floor(updates / 32));
    assert.equal(Number(element.dataset.completedBatches), updates === 9600 ? 32 : updates % 32);
    assert.match(element.attrs['aria-label'], updates === 9600 ? /Training complete: 300 epochs/ : /Epoch \d+ of 300: \d+ of 32 batches used/);
  }
  for (const updates of [...forwards.keys()].reverse()) {
    renderEpochCounter(element, epochState(updates));
    assert.equal(element.innerHTML, forwards.get(updates));
    assert.equal(Number(element.dataset.completedUpdates), updates);
  }
});

test('the partition diagram puts every actual observation inside exactly its batch', () => {
  const { plot, detail } = renderScene('partition', 1);
  const groups = new Map([...plot.matchAll(/data-batch="(\d+)" data-count="64"><rect x="([^"]+)" y="([^"]+)" width="([^"]+)" height="([^"]+)"/g)]
    .map(match => [Number(match[1]), match.slice(2).map(Number)]));
  assert.equal(groups.size, 32);
  const observations = [...plot.matchAll(/data-observation="(\d+)" x="([^"]+)" y="([^"]+)" width="([^"]+)" height="([^"]+)"/g)];
  assert.equal(observations.length, 2048);
  assert.equal(new Set(observations.map(match => match[1])).size, 2048);
  const membership = new Map(experiment.traces[2].slice(0, 32).flatMap((snapshot, index) => snapshot.indices.map(id => [id, index + 1])));
  observations.forEach(match => {
    const [id, x, y, width, height] = match.slice(1).map(Number);
    const [left, top, groupWidth, groupHeight] = groups.get(membership.get(id));
    assert.ok(x >= left && x + width <= left + groupWidth, `Observation ${id} escaped its batch horizontally`);
    assert.ok(y >= top && y + height <= top + groupHeight, `Observation ${id} escaped its batch vertically`);
  });
  assert.equal(detail, '');
});

test('the same point loss becomes the first term of the sum before averaging', () => {
  const first = experiment.firstUpdate;
  const point = renderScene('point-loss', 1).frame;
  assert.equal(point.pointIndex, first.indices[0]);
  const row = data[point.pointIndex];
  close(point.pointPrediction, predict(first.before, row.x));
  close(point.pointResidual, point.pointPrediction - row.y);
  close(point.pointLoss, .5 * point.pointResidual ** 2);
  const start = renderScene('batch-loss', 0).frame;
  assert.equal(start.lossesAdded, 1);
  assert.equal(start.normalized, false);
  close(start.batchSum, point.pointLoss);
  const end = renderScene('batch-loss', 1).frame;
  assert.equal(end.lossesAdded, 64);
  assert.equal(end.normalized, true);
  close(end.batchSum / 64, end.batchHalfMSE);
  close(end.batchHalfMSE, first.batchBefore);
  close(end.batchMSE, 2 * first.batchBefore);
  const reversed = renderScene('batch-loss', 0).frame;
  assert.deepEqual(reversed, start);
});

test('the supplied batch gradient has one component per parameter at the unchanged current weights', () => {
  const first = experiment.firstUpdate;
  const batch = first.indices.map(index => data[index]);
  const expected = parameters(gradient(first.before, batch));
  const residuals = batch.map(row => predict(first.before, row.x) - row.y);
  const meanResidual = residuals.reduce((sum, residual) => sum + residual, 0) / batch.length;
  for (const progress of [0, .25, .65, 1, .25, 0]) {
    const { frame, plot } = renderScene('batch-gradient', progress);
    assert.deepEqual(frame.model, first.before);
    assert.equal(frame.completedUpdates, 0);
    assert.equal(frame.residualCount, batch.length);
    assert.equal(frame.residualCount, 64);
    assert.deepEqual(frame.residuals, residuals, 'Residual coordinates must follow observations in the batch');
    assert.equal(frame.parameterCount, parameters(first.before).length);
    assert.equal(frame.parameterCount, 7);
    assert.equal(frame.gradient.length, parameters(first.before).length);
    frame.gradient.forEach((value, index) => close(value, expected[index]));
    close(frame.gradientC, expected[6]);
    close(frame.gradientC, meanResidual);
    close(frame.meanResidual, meanResidual);
    close(frame.outputBiasGradient, meanResidual);
    assert.match(plot, /data-space="observations"/);
    assert.match(plot, /data-space="parameters"/);
    const displayedGradient = [...plot.matchAll(/data-parameter="([^"]+)" data-gradient="([^"]+)"/g)];
    assert.equal(displayedGradient.length, 7);
    displayedGradient.forEach((match, index) => {
      assert.equal(match[1], parameterNames[index]);
      close(Number(match[2]), expected[index]);
    });
    const displayedResiduals = [...plot.matchAll(/data-observation="(\d+)" data-batch-position="(\d+)" data-residual="([^"]+)"/g)];
    assert.equal(displayedResiduals.length, 64);
    displayedResiduals.forEach((match, index) => {
      assert.equal(Number(match[1]), first.indices[index]);
      assert.equal(Number(match[2]), index + 1);
      close(Number(match[3]), residuals[index]);
    });
  }
  assert.deepEqual(expected.slice(0, 3), [0, 0, 0]);
  assert.equal(expected.filter(value => value !== 0).length, 4);
});

test('widening the network changes the parameter-gradient dimension without changing the batch residual dimension', () => {
  const first = experiment.traces[2][0];
  for (const width of [2, 12]) {
    const snapshot = experiment.traces[width][0];
    assert.deepEqual(snapshot.indices, first.indices);
    const rows = snapshot.indices.map(index => data[index]);
    const residuals = rows.map(row => predict(snapshot.model, row.x) - row.y);
    const g = gradient(snapshot.model, rows);
    assert.equal(residuals.length, 64);
    assert.equal(parameters(g).length, width === 2 ? 7 : 37);
    assert.equal(parameters(g).length, parameters(snapshot.model).length);
    const meanResidual = residuals.reduce((sum, residual) => sum + residual, 0) / rows.length;
    // The output bias moves every prediction by the same amount. Its one
    // gradient coordinate therefore averages all 64 residuals in either model.
    close(g.c, meanResidual);
    const eps = 1e-6;
    const plus = { ...snapshot.model, c: snapshot.model.c + eps };
    const minus = { ...snapshot.model, c: snapshot.model.c - eps };
    close((loss(plus, rows) - loss(minus, rows)) / (2 * eps), meanResidual, 1e-9);
  }
});

test('the single-parameter illustration leaves the actual model and update counter unchanged', () => {
  const first = experiment.firstUpdate;
  let previousPreview = first.before.c;
  for (const progress of [0, .3, .55, .8, 1]) {
    const frame = renderScene('parameter-step', progress).frame;
    assert.equal(frame.illustrationOnly, true);
    assert.equal(frame.completedUpdates, 0);
    assert.deepEqual(frame.model, first.before);
    close(frame.outputBias, first.before.c);
    assert.ok(frame.previewOutputBias >= previousPreview);
    assert.ok(frame.previewOutputBias <= first.after.c);
    previousPreview = frame.previewOutputBias;
  }
  close(previousPreview, first.after.c);
  assert.deepEqual(renderScene('update', 0).frame.model, first.before);
});

test('all seven parameters take the same simultaneous update, including the three zero-gradient coordinates', () => {
  const first = experiment.firstUpdate;
  const before = parameters(first.before), after = parameters(first.after);
  const g = parameters(first.gradient);
  const moving = g.map((value, index) => value !== 0 ? index : null).filter(index => index !== null);
  assert.equal(moving.length, 4);
  const states = new Map();
  for (const progress of [0, .3, .55, .8, 1]) {
    const { frame, detail } = renderScene('update', progress);
    const current = parameters(frame.model), fraction = frame.updateProgress;
    assert.ok(fraction >= 0 && fraction <= 1);
    assert.equal(frame.displayedParameters.length, 7);
    current.forEach((value, index) => {
      close(value, before[index] + fraction * (after[index] - before[index]));
      const displayed = frame.displayedParameters[index];
      assert.equal(displayed.label, parameterNames[index]);
      close(displayed.before, before[index]);
      close(displayed.current, value);
      close(displayed.gradient, g[index]);
      assert.equal(displayed.unchanged, g[index] === 0);
      if (g[index] === 0) assert.equal(value, before[index]);
      else close((value - before[index]) / (after[index] - before[index]), fraction, 1e-10);
    });
    const rendered = [...detail.matchAll(/data-parameter="[^"]+"/g)];
    assert.equal(rendered.length, 7, 'All parameter values must accompany the moving model curve');
    if (progress > .12 && progress < .82) {
      assert.ok(fraction > 0 && fraction < 1);
      moving.forEach(index => assert.notEqual(current[index], before[index]));
    }
    states.set(progress, frame);
  }
  assert.deepEqual(states.get(0).model, first.before);
  assert.deepEqual(states.get(1).model, first.after);
  for (const progress of [...states.keys()].reverse()) {
    assert.deepEqual(renderScene('update', progress).frame, states.get(progress));
  }
});

test('batch and epoch transitions continue from the exact preceding learned weights', () => {
  const trace = experiment.traces[2];
  const nextStart = renderScene('next-batch', 0).frame;
  const nextEnd = renderScene('next-batch', 1).frame;
  assert.equal(nextStart.completedUpdates, 1);
  assert.equal(nextEnd.completedUpdates, 2);
  assert.deepEqual(nextStart.model, trace[0].after);
  assert.deepEqual(nextEnd.model, trace[1].after);
  close(nextStart.outputBias, nextStart.model.c);
  close(nextEnd.outputBias, nextEnd.model.c);
  const partialNext = renderScene('next-batch', .65).frame;
  assert.ok(partialNext.updateProgress > 0 && partialNext.updateProgress < 1);
  assert.equal(partialNext.completedUpdates, 1);
  parameters(partialNext.model).forEach((value, index) => {
    const before = parameters(trace[1].model)[index], after = parameters(trace[1].after)[index];
    close(value, before + partialNext.updateProgress * (after - before));
  });
  assert.equal(nextEnd.batchOverlap, 0);
  for (const [id, progress, step, completed] of [
    ['batches-two', 0, 2, 2], ['batches-two', 1, 31, 32],
    ['epoch-two', 0, 32, 32], ['epoch-two', 1, 32, 33],
    ['train-two', 0, 33, 33], ['train-two', 1, 9599, 9600]
  ]) {
    const frame = renderScene(id, progress).frame;
    assert.equal(frame.step, step);
    assert.equal(frame.completedUpdates, completed);
    const snapshot = trace.find(item => item.step === step);
    const expected = completed === step ? snapshot.model : snapshot.after;
    close(frame.model.c, expected.c);
    frame.model.units.forEach((unit, index) => {
      for (const key of ['w', 'b', 'a']) close(unit[key], expected.units[index][key]);
    });
  }
  assert.equal(renderScene('train-wide', 0).frame.completedUpdates, 32);
  assert.equal(renderScene('train-wide', 1).frame.completedUpdates, 9600);
});

test('single-parameter probes compare independent nudges of the trained model with honest full-data losses', () => {
  const original = JSON.stringify(experiment);
  const baseline = experiment.traces[2].at(-1).after;
  const trained = experiment.runs.find(run => run.width === 2 && run.seed === 1);
  const baselineValues = parameters(baseline), baselineMSE = mse(baseline);
  assert.deepEqual(baseline, trained.model);
  assert.deepEqual(renderScene('parameter-probe', 0).frame.model, renderScene('train-two', 1).frame.model);
  const seen = new Map();
  for (let index = 0; index < 7; index++) {
    for (const [localProgress, expectedDelta] of [[0, 0], [.25, -.02], [.5, 0], [.75, .02]]) {
      const progress = (index + localProgress) / 7;
      const frame = renderScene('parameter-probe', progress).frame;
      assert.deepEqual(frame.baselineModel, baseline);
      assert.equal(frame.parameterIndex, index);
      assert.equal(frame.parameterName, parameterNames[index]);
      assert.equal(frame.completedUpdates, 9600, 'Trying a parameter is not an additional training update');
      close(frame.delta, expectedDelta);
      parameters(frame.model).forEach((value, coordinate) => {
        if (coordinate === index) close(value, baselineValues[coordinate] + expectedDelta);
        else assert.equal(value, baselineValues[coordinate], 'Each trial must hold every other parameter fixed');
      });
      close(frame.baselineMSE, baselineMSE);
      close(frame.currentMSE, mse(frame.model));
      close(frame.currentMSE, mse(perturb(baseline, index, expectedDelta)));
      assert.equal(frame.trialResults.length, 7);
      frame.trialResults.forEach((result, coordinate) => {
        assert.equal(result.parameterIndex, coordinate);
        assert.equal(result.parameterName, parameterNames[coordinate]);
        assert.equal(result.baselineValue, baselineValues[coordinate]);
        close(result.minus, mse(perturb(baseline, coordinate, -.02)));
        close(result.plus, mse(perturb(baseline, coordinate, .02)));
        // This observed result supports the scene's local claim; it does not
        // assert that coordinate changes can never improve other networks.
        assert.ok(result.minus >= baselineMSE - 1e-12);
        assert.ok(result.plus >= baselineMSE - 1e-12);
      });
      seen.set(progress, frame);
    }
  }
  const end = renderScene('parameter-probe', 1);
  assert.deepEqual(end.frame.model, baseline, 'Finish by restoring the actual trained model');
  assert.equal(end.frame.delta, 0);
  assert.equal(end.frame.completedUpdates, 9600);
  assert.equal([...end.detail.matchAll(/data-parameter="[^"]+"/g)].length, 7, 'The static end state must show every independent trial');
  assert.equal([...end.detail.matchAll(/>MSE after<\/text>/g)].length, 2, 'Both endpoint columns must identify their values as resulting MSE');
  for (const label of ['−0.02 nudge', '+0.02 nudge', 'Baseline MSE:', 'Current trial MSE:', 'Change in MSE:']) {
    assert.ok(end.detail.includes(label), `The trial summary must distinguish ${label}`);
  }
  const mathRows = [...end.detail.matchAll(/<g data-parameter="([^"]+)"[^>]*>([\s\S]*?)<\/g>/g)];
  const expectedTex = [String.raw`\theta_{11}`, String.raw`\theta_{10}`, String.raw`\phi_1`, String.raw`\theta_{21}`, String.raw`\theta_{20}`, String.raw`\phi_2`, String.raw`\phi_0`];
  mathRows.forEach((row, index) => {
    assert.equal(row[1], parameterNames[index]);
    assert.match(row[2], /<foreignObject\b/);
    assert.match(row[2], /class="katex"/);
    assert.ok(row[2].includes(`<annotation encoding="application/x-tex">${expectedTex[index]}</annotation>`), 'Each parameter must retain its typeset symbol and subscript');
  });
  assert.match(end.plot, /class="katex"/, 'The current curve trial must use the same mathematical notation');
  for (const [progress, frame] of [...seen].reverse()) {
    assert.deepEqual(renderScene('parameter-probe', progress).frame, frame);
  }
  assert.equal(JSON.stringify(experiment), original, 'The probe must not change the stored experiment or later training scenes');
});

// This harness runs the real controller and renderers. Browser geometry and
// actual scrolling are reviewed separately; here scroll events must alter and
// restore the scientific frames and counter, without relying on a timer.
async function controller(chapter = 'full') {
  const lesson = chapter === 'batches' ? foundationScenes : scenes;
  const html = readFileSync(new URL(`../docs/sgd/${chapter === 'batches' ? 'batches' : 'index'}.html`, import.meta.url), 'utf8');
  const source = readFileSync(new URL('../docs/sgd/story.mjs', import.meta.url), 'utf8').replace(/^import[^\n]+\n/gm, '');
  const listeners = new Map(), frames = [], nodes = new Map();
  let context;
  function on(type, callback) { listeners.set(type, [...(listeners.get(type) || []), callback]); }
  class Element {
    constructor(id = '') {
      this.id = id; this.dataset = {}; this.attrs = {}; this.children = [];
      this.textContent = ''; this.innerHTML = ''; this.hidden = false;
      this.classes = new Set();
      this.classList = {
        add: key => this.classes.add(key), remove: key => this.classes.delete(key),
        toggle: (key, value) => value ? this.classes.add(key) : this.classes.delete(key)
      };
      this.style = { setProperty(key, value) { this[key] = value; } };
    }
    setAttribute(key, value) { this.attrs[key] = String(value); }
    removeAttribute(key) { delete this.attrs[key]; }
    addEventListener(type, callback) { on(`${this.id}:${type}`, callback); }
    scrollIntoView() {}
    getBoundingClientRect() { return { top: 0, width: 288, height: this.id === 'detail' ? 90 : 240 }; }
  }
  for (const match of html.matchAll(/\bid="([^"]+)"/g)) nodes.set(match[1], new Element(match[1]));
  for (const id of ['stage', 'body', 'skip']) nodes.set(id, new Element(id));
  nodes.get('body').dataset.chapter = chapter;
  nodes.get('procedure').children = batchSteps.map((_, index) => new Element(`procedure-${index}`));
  let offset = 0;
  const steps = lesson.map(scene => {
    const step = new Element(scene.id);
    step.offset = offset; step.height = (scene.length || 1) * 710; offset += step.height;
    step.getBoundingClientRect = () => ({ top: step.offset - context.scrollY, height: step.height });
    return step;
  });
  const media = { matches: false, addEventListener: (type, callback) => on(`media:${type}`, callback) };
  context = {
    allScenes: scenes, foundationScenes, constructionKinds, render, epochState, renderEpochCounter,
    console, Math, Map, JSON, Number, String, innerWidth: 320, innerHeight: 568,
    scrollY: 0, location: { hash: '' }, matchMedia: () => media,
    requestAnimationFrame: callback => frames.push(callback), addEventListener: on,
    fetch: async () => {
      assert.notEqual(chapter, 'batches', 'Focused explanation must not require experiment data');
      return { ok: true, json: async () => experiment };
    }
  };
  context.document = {
    body: nodes.get('body'), fonts: { ready: Promise.resolve() },
    getElementById: id => nodes.get(id),
    querySelector: selector => nodes.get(selector === '.stage' ? 'stage' : 'skip'),
    querySelectorAll: () => steps,
    addEventListener: (type, callback) => on(`document:${type}`, callback)
  };
  context.window = context; context.katex = { renderToString: tex => tex };
  context.scrollTo = options => {
    context.scrollY = options.top; context.lastScroll = options;
    for (const callback of listeners.get('scroll') || []) callback();
  };
  vm.createContext(context);
  await vm.runInContext(`(async()=>{${source}\n})()`, context);
  const flush = () => { let count = 0; while (frames.length) { assert.ok(count++ < 20); frames.shift()(); } };
  const emit = (type, event) => { for (const callback of listeners.get(type) || []) callback(event); flush(); };
  flush();
  const at = (id, progress) => {
    const step = steps.find(item => item.id === id);
    context.scrollY = step.offset + progress * step.height; emit('scroll');
    assert.equal(nodes.get('stage').dataset.scene, id);
    return JSON.parse(nodes.get('stage').dataset.frame);
  };
  return {
    at, nodes, context, emit,
    reduce(value) { media.matches = value; emit('media:change'); }
  };
}

test('scroll events advance and restore the batch update and epoch counter together', async () => {
  const ui = await controller();
  const stops = ['update', 'next-batch', 'batches-two', 'epoch-two', 'train-two']
    .flatMap(id => [0, .35, .6, .9].map(progress => [id, progress]));
  const seen = [], counter = ui.nodes.get('epoch-counter');
  let previous = 0;
  for (const [id, progress] of stops) {
    const frame = ui.at(id, progress);
    assert.equal(counter.hidden, false);
    const used = Number(counter.dataset.completedUpdates);
    assert.equal(used, Math.floor(frame.completedUpdates));
    assert.ok(used >= previous, `${id} regressed to ${used} after ${previous}`);
    previous = used;
    seen.push({ id, progress, frame, counter: counter.innerHTML });
  }
  for (const stop of seen.reverse()) {
    assert.deepEqual(ui.at(stop.id, stop.progress), stop.frame);
    assert.equal(counter.innerHTML, stop.counter);
  }
});

test('parameter probes retain the completed epoch count and a complete reduced-motion summary', async () => {
  const ui = await controller(), counter = ui.nodes.get('epoch-counter');
  const stops = [0, .25 / 7, .75 / 7, 3.25 / 7, 3.75 / 7, 6.25 / 7, 6.75 / 7, .999];
  const seen = new Map();
  for (const progress of stops) {
    const frame = ui.at('parameter-probe', progress);
    assert.equal(counter.hidden, false);
    assert.equal(frame.completedUpdates, 9600);
    assert.equal(counter.dataset.completedUpdates, '9600');
    assert.equal(counter.dataset.completedEpochs, '300');
    seen.set(progress, { frame, counter: counter.innerHTML });
  }
  for (const [progress, state] of [...seen].reverse()) {
    assert.deepEqual(ui.at('parameter-probe', progress), state.frame);
    assert.equal(counter.innerHTML, state.counter);
  }
  ui.reduce(true);
  const reduced = ui.at('parameter-probe', .1);
  assert.deepEqual(reduced.model, experiment.traces[2].at(-1).after);
  assert.equal(reduced.delta, 0);
  assert.equal(reduced.trialResults.length, 7);
  assert.equal([...ui.nodes.get('detail').innerHTML.matchAll(/data-parameter="[^"]+"/g)].length, 7);
  assert.equal(counter.dataset.completedEpochs, '300');
  ui.emit('document:keydown', { key: 'ArrowRight', target: { tagName: 'DIV' }, preventDefault() {} });
  assert.equal(ui.nodes.get('stage').dataset.scene, 'initialize-wide');
  assert.equal(counter.dataset.completedUpdates, '0');
  assert.equal(counter.dataset.completedEpochs, '0');
  ui.emit('document:keydown', { key: 'ArrowLeft', target: { tagName: 'DIV' }, preventDefault() {} });
  assert.equal(ui.nodes.get('stage').dataset.scene, 'parameter-probe');
  assert.equal(counter.dataset.completedEpochs, '300');
});

test('reduced motion and keyboard navigation retain complete losses and epoch boundaries', async () => {
  const ui = await controller();
  ui.reduce(true);
  assert.equal(ui.at('batch-loss', .1).normalized, true);
  const gradientFrame = ui.at('batch-gradient', .1);
  assert.equal(gradientFrame.gradient.length, 7);
  assert.equal(gradientFrame.completedUpdates, 0);
  const illustrationFrame = ui.at('parameter-step', .1);
  assert.equal(illustrationFrame.illustrationOnly, true);
  assert.equal(illustrationFrame.completedUpdates, 0);
  assert.deepEqual(illustrationFrame.model, experiment.firstUpdate.before);
  assert.equal(ui.at('batches-two', .1).completedUpdates, 32);
  assert.equal(ui.nodes.get('epoch-counter').dataset.completedEpochs, '1');
  ui.emit('document:keydown', { key: 'ArrowRight', target: { tagName: 'DIV' }, preventDefault() {} });
  assert.equal(ui.context.lastScroll.behavior, 'instant');
  assert.equal(ui.nodes.get('stage').dataset.scene, 'epoch-two');
  assert.equal(ui.nodes.get('epoch-counter').dataset.completedUpdates, '33');
  ui.emit('document:keydown', { key: 'ArrowLeft', target: { tagName: 'DIV' }, preventDefault() {} });
  assert.equal(ui.nodes.get('stage').dataset.scene, 'batches-two');
  assert.equal(ui.nodes.get('epoch-counter').dataset.completedUpdates, '32');
  const focused = await controller('batches');
  focused.at('epoch-repeat', .5);
  assert.equal(focused.nodes.get('epoch-counter').hidden, true);
});

test('both static reading routes contain the same substantive captions and definitions', () => {
  for (const [file, lesson] of [['index.html', scenes], ['batches.html', foundationScenes]]) {
    const html = readFileSync(new URL(`../docs/sgd/${file}`, import.meta.url), 'utf8');
    const steps = html.slice(html.indexOf('<div id="steps">'), html.indexOf('<section class="methods"'));
    const reading = html.slice(html.indexOf('<details id="reading">'), html.indexOf('</details>'));
    assert.equal((steps.match(/class="step"/g) || []).length, lesson.length);
    assert.equal((reading.match(/<section>/g) || []).length, lesson.length);
    for (const scene of lesson) {
      assert.ok(steps.includes(scene.caption) && steps.includes(scene.note), `${file}: ${scene.id} fallback`);
      assert.ok(reading.includes(scene.caption) && reading.includes(scene.note), `${file}: ${scene.id} reading`);
    }
  }
});
