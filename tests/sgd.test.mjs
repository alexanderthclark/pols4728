import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import { copy, data, gradient, initialize, loss, mse, predict, random, shuffle, update } from '../docs/sgd/model.mjs';
import { epochState, renderEpochCounter } from '../docs/sgd/epoch-counter.mjs';
import { scenes, foundationScenes, batchSteps } from '../docs/sgd/scenes.mjs';
import { render } from '../docs/sgd/views.mjs';
import { constructionKinds } from '../docs/sgd/construction.mjs';

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
  return { frame, plot: plot.innerHTML, detail: detail.innerHTML };
}

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
  assert.deepEqual([...detail.matchAll(/data-observation="(\d+)"/g)].map(match => Number(match[1])), experiment.firstUpdate.indices);
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

test('the animation carries the signed prediction error through each parameter sensitivity', () => {
  const first = experiment.firstUpdate, eps = 1e-6;
  const point = renderScene('point-loss', 1).frame;
  for (const id of ['prediction-gradient', 'parameter-gradient']) {
    const frame = renderScene(id, 1).frame, row = data[frame.pointIndex];
    assert.equal(frame.pointIndex, point.pointIndex);
    close(frame.pointResidual, predict(first.before, row.x) - row.y);
    assert.ok(frame.pointResidual > 0, 'The example prediction is above its target');
    close(frame.predictionDerivative, frame.pointResidual);
    const pointLoss = prediction => .5 * (prediction - row.y) ** 2;
    close((pointLoss(frame.pointPrediction + eps) - pointLoss(frame.pointPrediction - eps)) / (2 * eps), frame.predictionDerivative, 1e-9);
    assert.equal(frame.parameterSensitivities.length, 7);
    assert.equal(frame.pointParameterGradient.length, 7);
    frame.parameterSensitivities.forEach((sensitivity, index) => {
      const plus = perturb(first.before, index, eps), minus = perturb(first.before, index, -eps);
      close((predict(plus, row.x) - predict(minus, row.x)) / (2 * eps), sensitivity, 1e-9);
      close(frame.pointParameterGradient[index], frame.pointResidual * sensitivity);
      close((loss(plus, [row]) - loss(minus, [row])) / (2 * eps), frame.pointParameterGradient[index], 1e-9);
      close(frame.pointParameterGradient[index], parameters(gradient(first.before, [row]))[index]);
    });
    assert.deepEqual(frame.parameterSensitivities.slice(0, 3), [0, 0, 0]);
    assert.equal(frame.parameterSensitivities[6], 1, 'The output bias changes the prediction one-for-one');
    assert.equal(frame.pointParameterGradient[6], frame.pointResidual);
    const start = renderScene(id, 0).frame;
    renderScene(id, 1);
    assert.deepEqual(renderScene(id, 0).frame, start);
  }
  // The batch prediction gradient includes the averaging factor; the parameter
  // gradient additionally combines the different sensitivities of all points.
  const batch = first.indices.map(index => data[index]);
  const predictionGradient = batch.map(row => (predict(first.before, row.x) - row.y) / batch.length);
  close(predictionGradient.reduce((sum, value) => sum + value, 0), gradient(first.before, batch).c);
  for (const progress of [0, .45, 1]) {
    const frame = renderScene('batch-gradient', progress).frame;
    const expected = parameters(gradient(first.before, batch.slice(0, frame.contributions)))
      .map(value => value * frame.contributions / batch.length);
    frame.gradient.forEach((value, index) => close(value, expected[index]));
    if (progress === 0) assert.equal(frame.contributions, 1);
    else if (progress === 1) assert.equal(frame.contributions, 64);
    else assert.ok(frame.contributions > 1 && frame.contributions < 64);
  }
  const negative = batch.find(row => predict(first.before, row.x) < row.y);
  assert.ok(negative, 'The batch also contains a prediction below its target');
  close(gradient(first.before, [negative]).c, predict(first.before, negative.x) - negative.y);
  assert.ok(gradient(first.before, [negative]).c < 0);
});

test('batch and epoch transitions continue from the exact preceding learned weights', () => {
  const trace = experiment.traces[2];
  const nextStart = renderScene('next-batch', 0).frame;
  const nextEnd = renderScene('next-batch', 1).frame;
  assert.equal(nextStart.completedUpdates, 1);
  assert.equal(nextEnd.completedUpdates, 2);
  close(nextStart.outputBias, trace[0].after.c);
  close(nextEnd.outputBias, trace[1].after.c);
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

test('reduced motion and keyboard navigation retain complete losses and epoch boundaries', async () => {
  const ui = await controller();
  ui.reduce(true);
  assert.equal(ui.at('batch-loss', .1).normalized, true);
  for (const id of ['prediction-gradient', 'parameter-gradient']) {
    const frame = ui.at(id, .1);
    assert.equal(frame.gradientReveal, 1);
    assert.equal(frame.completedUpdates, 0);
  }
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
