import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { runInNewContext } from 'node:vm';
import test from 'node:test';

const require = createRequire(import.meta.url);
const math = require('../docs/shared/boosting-math.js');
const typography = require('../docs/shared/svg-math.js');
const example = require('../docs/boosting/worked-example.js');
const view = require('../docs/boosting/worked-view.js');
const restricted = require('../docs/boosting/restricted-view.js');
const lossView = require('../docs/boosting/loss-view.js');
const html = readFileSync(new URL('../docs/boosting/index.html', import.meta.url), 'utf8');
const source = readFileSync(new URL('../docs/boosting/story.js', import.meta.url), 'utf8');
const transitionPhysical = 12;
const workedPhysical = local => local + 13;
const lossPhysical = local => local + 21;
const mseContourPhysical = 3;
const mseUpdatePhysical = 4;
const orthogonalPhysical = 5;
const changedLossPhysical = 6;
const gradientPhysical = 7;
const gradientFitPhysical = 8;
const gradientDescentPhysical = 9;
const gradientLimitPhysical = 10;
const sandboxPhysical = 11;
const stageTop = 400;

// This harness checks real controller events and numerical output. It does not
// emulate layout, parse generated HTML, or claim to replace browser visual QA.
function setup({ reducedMotion = false, compact = false, variableStageHeight = false, smoothScroll = false, hash = '', stageDocumentTop = stageTop } = {}) {
  let time = 0, nextId = 1;
  let pendingScroll = null;
  const visitedHashes = [];
  const scrollCalls = [];
  const frames = new Map();
  class Element {
    constructor(tagName = 'DIV') {
      this.tagName = tagName; this.listeners = new Map(); this.attrs = new Map();
      this.dataset = {}; this.style = {}; this.children = []; this.value = '';
      this._text = ''; this.hidden = false; this.disabled = false; this.innerHTML = '';
    }
    addEventListener(type, callback) { this.listeners.set(type, [...(this.listeners.get(type) || []), callback]); }
    emit(type, extra = {}) {
      const event = { type, target: this, preventDefault() { this.defaultPrevented = true; }, ...extra };
      for (const callback of this.listeners.get(type) || []) callback(event);
      return event;
    }
    setAttribute(name, value) { this.attrs.set(name, String(value)); if (name === 'hidden') this.hidden = true; }
    getAttribute(name) { return this.attrs.get(name); }
    removeAttribute(name) { this.attrs.delete(name); }
    toggleAttribute(name, enabled = !this.attrs.has(name)) {
      if (enabled) this.attrs.set(name, ''); else this.attrs.delete(name);
      if (name === 'hidden') this.hidden = enabled;
      return enabled;
    }
    appendChild(child) { this.children.push(child); return child; }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; this._text = ''; }
    set textContent(value) { this._text = String(value); this.children = []; this.innerHTML = ''; }
    get textContent() { return this.innerHTML && this.attrs.has('aria-label') ? this.attrs.get('aria-label') : this._text + this.children.map(child => child.textContent).join(' '); }
    getBoundingClientRect() { return { top: 64, bottom: 800, width: 640, height: 450 }; }
  }
  const nodes = new Map();
  for (const match of html.matchAll(/<(\w+)\b([^>]*\bid="([^"]+)"[^>]*)>/g)) {
    const element = new Element(match[1].toUpperCase());
    element.id = match[3];
    element.hidden = /\bhidden(?:\s|$)/.test(match[2]);
    element.disabled = /\bdisabled(?:\s|$)/.test(match[2]);
    element.value = /\bvalue="([^"]*)"/.exec(match[2])?.[1] || '';
    for (const attribute of match[2].matchAll(/\bdata-([\w-]+)="([^"]+)"/g)) {
      const name = attribute[1].replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
      element.dataset[name] = attribute[2];
      element.setAttribute('data-' + attribute[1], attribute[2]);
    }
    nodes.set(element.id, element);
  }
  const restrictedChoices = [...html.matchAll(/<button\b([^>]*\bdata-restricted-loss="([^"]+)"[^>]*)>/g)].map(match => {
    const id = /\bid="([^"]+)"/.exec(match[1])?.[1];
    const element = id ? nodes.get(id) : new Element('BUTTON');
    element.dataset.restrictedLoss = match[2];
    element.setAttribute('data-restricted-loss', match[2]);
    nodes.set('restricted-choice-' + match[2], element);
    return element;
  });
  for (const match of html.matchAll(/<select\b[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g)) {
    const options = [...match[2].matchAll(/<option\b([^>]*)>/g)];
    const selected = options.find(option => /\bselected\b/.test(option[1])) || options[0];
    nodes.get(match[1]).value = /\bvalue="([^"]*)"/.exec(selected[1])[1];
  }
  const get = id => nodes.get(id);
  const shell = new Element();
  const stageTools = get('stage-tools') || new Element();
  const stageHeight = () => !narrow.matches ? 500 : !variableStageHeight ? 620
    : shell.dataset.example === 'restricted' ? 660
    : shell.dataset.example === 'worked' && [3, 5, 6].includes(Number(shell.dataset.workedScene)) ? 528 : 392;
  const headingTop = physical => (narrow.matches ? stageDocumentTop + stageHeight() + 48 : stageDocumentTop + 600) + physical * 1000;
  // The figure remains pinned while the story cards move on every viewport.
  shell.getBoundingClientRect = () => {
    const top = Math.max(64, stageDocumentTop - win.scrollY);
    return { top, bottom: top + stageHeight(), height: stageHeight(), width: narrow.matches ? 288 : 1100 };
  };
  get('visual-area').getBoundingClientRect = () => ({ width: narrow.matches ? 288 : 790, height: narrow.matches ? 340 : 450 });
  const win = new Element();
  const reduced = new Element(); reduced.matches = reducedMotion;
  const narrow = new Element(); narrow.matches = compact;
  Object.assign(win, {
    BoostingMath: math, SvgMath: typography, WorkedBoostingExample: example, WorkedBoostingView: view, RestrictedBoostingView: restricted, LossBoostingView: lossView,
    innerHeight: 800, innerWidth: compact ? 320 : 1200, scrollY: 0,
    location: { hash },
    matchMedia: query => query.includes('reduced-motion') ? reduced : narrow,
    history: { pushState: (_, __, hash) => visitedHashes.push(hash) },
    scrollTo: ({ top, behavior }) => {
      scrollCalls.push({ top, behavior });
      if (smoothScroll && behavior === 'smooth') pendingScroll = { from: win.scrollY, to: top };
      else { win.scrollY = top; win.emit('scroll'); }
    }
  });
  const steps = [...html.matchAll(/<section\b[^>]*\bdata-step="(\d+)"[^>]*>/g)].map(match => {
    const element = new Element('SECTION');
    element.id = /\bid="([^"]+)"/.exec(match[0])[1];
    for (const attribute of match[0].matchAll(/\bdata-([\w-]+)="([^"]+)"/g)) {
      const name = attribute[1].replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
      element.dataset[name] = attribute[2];
      element.setAttribute('data-' + attribute[1], attribute[2]);
    }
    element.getBoundingClientRect = () => ({ top: headingTop(Number(match[1])) - win.scrollY, height: 300 });
    return element;
  });
  const layout = new Element();
  layout.getBoundingClientRect = () => ({ top: stageDocumentTop - win.scrollY, bottom: headingTop(steps.length) - win.scrollY });
  let referencesVisits = 0;
  get('references').scrollIntoView = () => { referencesVisits++; };
  const doc = new Element();
  doc.getElementById = get;
  doc.querySelector = selector => selector === '.stage-shell' ? shell : selector === '.story-layout' ? layout : selector === '.stage-tools' ? stageTools : null;
  doc.querySelectorAll = selector => selector === '.step' ? steps : selector === '[data-restricted-loss]' ? restrictedChoices : [];
  doc.createElement = tag => new Element(tag.toUpperCase());
  doc.createElementNS = (_, tag) => new Element(tag.toUpperCase());
  doc.documentElement = { scrollHeight: 1000 + steps.length * 1000 + 3000 };
  const requestAnimationFrame = callback => { const id = nextId++; frames.set(id, callback); return id; };
  const cancelAnimationFrame = id => frames.delete(id);
  const advance = elapsed => {
    time += elapsed;
    for (const id of [...frames.keys()]) {
      const callback = frames.get(id); frames.delete(id);
      if (callback) callback(time);
    }
  };
  const context = {
    window: win, document: doc, performance: { now: () => time },
    requestAnimationFrame, cancelAnimationFrame,
    ResizeObserver: class { constructor(callback) { this.callback = callback; } observe() {} }
  };
  Object.defineProperties(context, {
    innerHeight: { get: () => win.innerHeight }, scrollY: { get: () => win.scrollY }
  });
  runInNewContext(source, context, { filename: 'boosting/story.js' });
  const click = id => { assert.equal(get(id).disabled, false); get(id).emit('click'); advance(0); };
  const control = (id, value, event = 'input') => { get(id).value = String(value); get(id).emit(event); };
  const scrollAt = index => {
    const threshold = 64 + stageHeight() + 32;
    win.scrollY = headingTop(index) - threshold + 1;
    win.emit('scroll'); advance(0);
  };
  const key = (key, target = doc) => { const event = doc.emit('keydown', { key, target }); advance(0); return event; };
  const resize = () => { win.emit('resize'); advance(0); };
  const clickLink = href => {
    const link = new Element('A'); link.setAttribute('href', href);
    link.closest = () => link;
    const event = doc.emit('click', { target: link, button: 0 }); advance(0);
    return event;
  };
  const scrollFraction = fraction => {
    assert.ok(pendingScroll, 'No smooth scroll is pending');
    win.scrollY = pendingScroll.from + fraction * (pendingScroll.to - pendingScroll.from);
    win.emit('scroll'); advance(0);
  };
  return { get, shell, stageTools, click, clickLink, control, scrollAt, scrollFraction, advance, key, resize, reduced, frames, headingTop,
    setCompact: matches => { narrow.matches = matches; win.innerWidth = matches ? 320 : 1200; },
    historyBack: hash => { win.location.hash = hash; win.emit('popstate'); advance(0); },
    nativeFragment: index => { win.scrollY = headingTop(index) - 64; win.emit('scroll'); },
    hashChange: () => { win.emit('hashchange'); advance(0); advance(0); },
    load: () => { win.emit('load'); advance(0); advance(0); }, scrollCalls,
    scrollBy: distance => { win.scrollY += distance; win.emit('scroll'); advance(0); },
    wheel: () => win.emit('wheel'), scrollEnd: () => win.emit('scrollend'), visitedHashes, scrollY: () => win.scrollY, referencesVisits: () => referencesVisits };
}
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-9, `${actual} should equal ${expected}`);
const figureLoss = ui => Number(/data-training-loss="([^"]+)"/.exec(ui.get('worked-diagram').innerHTML)[1]);
const figurePredictions = ui => /data-predictions="([^"]+)"/.exec(ui.get('worked-diagram').innerHTML)[1].split(',').map(Number);
const numbers = ui => ui.get('vector-readout').textContent;
const restrictedAttribute = (ui, name) => new RegExp(`data-${name}="([^"]+)"`).exec(ui.get('restricted-diagram').innerHTML)?.[1];
const restrictedLoss = ui => Number(restrictedAttribute(ui, 'loss'));
const restrictedVector = (ui, name, expected) => {
  const actual = JSON.parse(restrictedAttribute(ui, name));
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => close(value, expected[index]));
};
const restrictedVisibleWords = ui => [
  ...[...ui.get('restricted-diagram').innerHTML.matchAll(/<(?:title|desc|text)\b[^>]*>([\s\S]*?)<\/(?:title|desc|text)>/g)].map(match => match[1]),
  ui.get('stage-caption').textContent, numbers(ui)
].join(' ');
const restrictedAnnotation = (ui, className) => new RegExp(`<text\\b[^>]*class="[^"]*\\b${className}\\b[^"]*"[^>]*>([\\s\\S]*?)<\\/text>`).exec(ui.get('restricted-diagram').innerHTML)?.[1]?.replace(/<[^>]+>/g, '');

const openingDurations = { 1: 1200, 2: 1400, 3: 1700, 4: 2600, 5: 3600, 6: 5000, 7: 1800, 8: 2600, 9: 18000 };
const finishOpening = ui => ui.advance(openingDurations[Number(ui.shell.dataset.restrictedScene)] || 0);
const chooseRestrictedLoss = (ui, loss) => ui.click('restricted-choice-' + loss);
const finishSharedMove = ui => ui.advance(1600);

test('the opening reveals residuals, the shared restriction, and MSE contours before fitting', () => {
  const ui = setup({ reducedMotion: true });
  let svg = ui.get('restricted-diagram').innerHTML;
  assert.doesNotMatch(svg, /class="restricted-(?:residual-arrow|available-line|tree-arrow|optimum|current-contour|optimal-contour)"/);
  ui.click('next');
  svg = ui.get('restricted-diagram').innerHTML;
  assert.match(svg, /class="restricted-residual-arrow"/);
  assert.equal(restrictedAnnotation(ui, 'restricted-residual-label'), 'r = (1, 0)');
  assert.doesNotMatch(svg, /class="restricted-(?:available-line|tree-arrow|optimum|current-contour|optimal-contour)"/);
  ui.click('next');
  svg = ui.get('restricted-diagram').innerHTML;
  assert.match(svg, /class="restricted-available-line"/);
  assert.match(restrictedVisibleWords(ui), /x\s*=\s*1/);
  assert.match(restrictedVisibleWords(ui), /same correction/i);
  assert.match(restrictedVisibleWords(ui), /45°/);
  assert.doesNotMatch(svg, /class="restricted-(?:tree-arrow|optimum|mean|current-contour|optimal-contour)"/);
  ui.click('next');
  assert.equal(Number(ui.shell.dataset.scene), mseContourPhysical);
  svg = ui.get('restricted-diagram').innerHTML;
  assert.match(svg, /restricted-(?:current|optimal)-contour/);
  restrictedVector(ui, 'prediction', [0, 0]);
  restrictedVector(ui, 'residual', [1, 0]);
});

test('all opening scenes hide the old pane, numbers, and sliders while revealing gradients last', () => {
  const ui = setup({ reducedMotion: true });
  for (let scene = 0; scene <= sandboxPhysical; scene++) {
    if (scene > 0) ui.click('next');
    assert.equal(ui.shell.dataset.opening, 'true');
    assert.equal(ui.stageTools.hidden, true);
    assert.equal(ui.get('numbers').hidden, true);
    assert.equal(ui.get('vector-readout').hidden, true);
    assert.equal(ui.get('restricted-controls').hidden, scene !== sandboxPhysical);
    assert.equal(ui.get('opening-replay').hidden, !openingDurations[scene]);
    restrictedVector(ui, 'target', [1, 0]);
    if (scene < gradientPhysical) {
      assert.doesNotMatch(restrictedVisibleWords(ui), /gradient|∇/i);
      assert.doesNotMatch(ui.get('restricted-diagram').innerHTML, /class="restricted-gradient-arrow"/);
      const narrative = new RegExp(`<section\\b[^>]*data-step="${scene}"[^>]*>([\\s\\S]*?)<\\/section>`).exec(html)[1];
      assert.doesNotMatch(narrative, /gradient|∇/i);
    }
  }
  assert.match(restrictedVisibleWords(ui), /gradient|∇/i);
  for (const id of ['restricted-progress', 'restricted-play', 'restricted-loss', 'restricted-tau', 'restricted-delta', 'restricted-prediction', 'restricted-loss-equation']) {
    assert.equal(ui.get(id), undefined, `${id} is removed from the opening`);
  }
  ui.click('next');
  assert.equal(ui.shell.dataset.example, 'transition');
  assert.equal(ui.stageTools.hidden, true);
  assert.equal(ui.get('numbers').hidden, true);
  ui.click('next');
  assert.equal(ui.shell.dataset.opening, 'false');
  assert.equal(ui.stageTools.hidden, false);
  assert.equal(ui.get('numbers').hidden, false);
});

test('scrolling into each reveal starts its animation while the predictions remain at zero', () => {
  const ui = setup();
  for (const scene of [1, 2, mseContourPhysical]) {
    ui.scrollAt(scene);
    assert.ok(ui.frames.size > 0);
    close(Number(ui.shell.dataset.updateProgress), 0);
    restrictedVector(ui, 'prediction', [0, 0]);
    ui.advance(openingDurations[scene] / 2);
    close(Number(ui.shell.dataset.updateProgress), .5);
    restrictedVector(ui, 'prediction', [0, 0]);
    ui.advance(openingDurations[scene] / 2);
    close(Number(ui.shell.dataset.updateProgress), 1);
    restrictedVector(ui, 'prediction', [0, 0]);
  }
});

test('the MSE update automatically moves from zero to the residual projection and replays', () => {
  const ui = setup();
  ui.clickLink('#residual-move');
  close(restrictedLoss(ui), .5);
  restrictedVector(ui, 'prediction', [0, 0]);
  restrictedVector(ui, 'residual', [1, 0]);
  assert.ok(ui.frames.size > 0, 'Entering the residual update starts its progression');
  assert.equal(restrictedAnnotation(ui, 'restricted-prediction-label'), 'f = (0, 0)');
  assert.equal(restrictedAnnotation(ui, 'restricted-residual-label'), 'r = (1, 0)');
  ui.advance(openingDurations[mseUpdatePhysical] / 2);
  restrictedVector(ui, 'prediction', [.25, .25]);
  restrictedVector(ui, 'residual', [.75, -.25]);
  close(restrictedLoss(ui), .3125);
  ui.advance(openingDurations[mseUpdatePhysical] / 2);
  restrictedVector(ui, 'prediction', [.5, .5]);
  restrictedVector(ui, 'residual', [.5, -.5]);
  assert.equal(restrictedAnnotation(ui, 'restricted-prediction-label'), 'f = (0.5, 0.5)');
  close(restrictedLoss(ui), .25);
  ui.click('opening-replay');
  restrictedVector(ui, 'prediction', [0, 0]);
  finishOpening(ui);
  restrictedVector(ui, 'prediction', [.5, .5]);
});

test('leaving an opening animation cancels stale frames and returning restarts it', () => {
  const ui = setup();
  ui.scrollAt(mseUpdatePhysical); ui.advance(500);
  assert.ok(Number(restrictedAttribute(ui, 'c')) > 0);
  ui.scrollAt(0);
  const baseline = ui.get('restricted-diagram').innerHTML;
  ui.advance(4000);
  assert.equal(ui.get('restricted-diagram').innerHTML, baseline);
  ui.scrollAt(mseUpdatePhysical);
  restrictedVector(ui, 'prediction', [0, 0]);
  close(Number(ui.shell.dataset.updateProgress), 0);
  finishOpening(ui);
  restrictedVector(ui, 'prediction', [.5, .5]);
});

test('the orthogonality trial keeps the best MSE prediction as its fixed baseline', () => {
  const ui = setup();
  ui.scrollAt(orthogonalPhysical);
  for (let quarter = 0; quarter <= 4; quarter++) {
    if (quarter > 0) ui.advance(openingDurations[orthogonalPhysical] / 4);
    restrictedVector(ui, 'prediction', [.5, .5]);
    restrictedVector(ui, 'residual', [.5, -.5]);
    close(restrictedLoss(ui), .25);
    if (quarter === 1 || quarter === 3) {
      const trial = JSON.parse(restrictedAttribute(ui, 'trial-prediction'));
      assert.equal(trial[0], trial[1], 'A trial still gives both observations the same prediction');
      assert.ok(quarter === 1 ? trial[0] > .5 : trial[0] < .5);
      assert.ok(Number(restrictedAttribute(ui, 'trial-loss')) > .25);
      assert.match(ui.get('restricted-diagram').innerHTML, /class="restricted-trial"/);
    }
  }
  const residual = JSON.parse(restrictedAttribute(ui, 'residual'));
  close(residual[0] + residual[1], 0);
  assert.match(ui.get('restricted-diagram').innerHTML, /class="restricted-right-angle"/);
  assert.match(restrictedVisibleWords(ui), /perpendicular|orthogonal/i);
});

test('changing the loss morphs contours while holding predictions and residuals fixed', () => {
  const ui = setup();
  ui.scrollAt(changedLossPhysical);
  const start = ui.get('restricted-diagram').innerHTML;
  for (const [elapsed, tau] of [[0, .5], [openingDurations[changedLossPhysical] / 2, .65], [openingDurations[changedLossPhysical] / 2, .8]]) {
    ui.advance(elapsed);
    restrictedVector(ui, 'prediction', [.5, .5]);
    restrictedVector(ui, 'residual', [.5, -.5]);
    close(restrictedLoss(ui), .25);
    close(Number(restrictedAttribute(ui, 'tau')), tau);
    assert.doesNotMatch(restrictedVisibleWords(ui), /gradient|∇/i);
  }
  assert.notEqual(ui.get('restricted-diagram').innerHTML, start);
  assert.equal(ui.get('opening-annotation').hidden, true, 'The loss annotation fades after the morph');
  restrictedVector(ui, 'optimal-prediction', [.8, .8]);
  ui.click('next');
  restrictedVector(ui, 'prediction', [.5, .5]);
  restrictedVector(ui, 'negative-gradient', [.8, -.2]);
  ui.advance(openingDurations[gradientPhysical] / 2);
  restrictedVector(ui, 'prediction', [.5, .5]);
  restrictedVector(ui, 'negative-gradient', [.8, -.2]);
  ui.advance(openingDurations[gradientPhysical] / 2);
  restrictedVector(ui, 'prediction', [.5, .5]);
  restrictedVector(ui, 'negative-gradient', [.8, -.2]);
  close(restrictedLoss(ui), .25);
});

test('fitting the first gradient tree reveals a half-rate step while keeping its starting prediction fixed', () => {
  const ui = setup();
  ui.scrollAt(gradientFitPhysical);
  for (const elapsed of [0, openingDurations[gradientFitPhysical] / 2, openingDurations[gradientFitPhysical] / 2]) {
    ui.advance(elapsed);
    restrictedVector(ui, 'prediction', [.5, .5]);
    restrictedVector(ui, 'training-targets', [.8, -.2]);
    restrictedVector(ui, 'target-prediction', [.5, .5]);
    restrictedVector(ui, 'fitted-correction', [.3, .3]);
    restrictedVector(ui, 'update-prediction', [.65, .65]);
    close(Number(restrictedAttribute(ui, 'learning-rate')), .5);
    close(Number(ui.shell.dataset.rate), .5);
  }
});

test('the descent timeline refits targets at round boundaries and keeps each fitted tree fixed during its update', () => {
  const ui = setup();
  ui.scrollAt(gradientDescentPhysical);
  let previousElapsed = 0;
  const basePredictions = [.5];
  for (let round = 0; round < 6; round++) {
    const c = basePredictions.at(-1);
    basePredictions.push(c + .5 * (.8 - c));
  }
  for (const elapsed of [0, 500, 750, 1500, 2250, 3000, 3750, 4500, 5250, 6000, 9000, 12000, 15000, 18000]) {
    ui.advance(elapsed - previousElapsed);
    previousElapsed = elapsed;
    const progress = elapsed / openingDurations[gradientDescentPhysical];
    const expected = restricted.resolve(gradientDescentPhysical, { progress });
    const round = Math.min(6, Math.floor(elapsed / 3000));
    const base = basePredictions[round];
    const trainingTargets = [1.6 * (1 - base), -.4 * base];
    const fittedMean = (trainingTargets[0] + trainingTargets[1]) / 2;
    close(Number(ui.shell.dataset.updateProgress), progress);
    assert.equal(Number(ui.shell.dataset.round), round);
    assert.equal(ui.shell.dataset.descentPhase, expected.phase);
    assert.equal(restrictedAttribute(ui, 'phase'), expected.phase);
    restrictedVector(ui, 'prediction', expected.prediction);
    restrictedVector(ui, 'target-prediction', [base, base]);
    restrictedVector(ui, 'training-targets', trainingTargets);
    restrictedVector(ui, 'display-negative-gradient', trainingTargets);
    restrictedVector(ui, 'fitted-correction', [fittedMean, fittedMean]);
    restrictedVector(ui, 'current-negative-gradient', expected.negativeGradient);
    close(restrictedLoss(ui), expected.loss);
    close(Number(ui.shell.dataset.rate), .5);
    if (elapsed === 750) assert.equal(expected.phase, 'fit');
    if (elapsed === 1500) assert.equal(expected.phase, 'update');
    if (elapsed === 2250) {
      restrictedVector(ui, 'prediction', [.575, .575]);
      restrictedVector(ui, 'fitted-correction', [.3, .3]);
      restrictedVector(ui, 'current-negative-gradient', [.68, -.23]);
    }
    if (elapsed === 3000) {
      assert.equal(expected.phase, 'target');
      restrictedVector(ui, 'prediction', [.65, .65]);
      restrictedVector(ui, 'fitted-correction', [.15, .15]);
    }
    if (elapsed === 5250) restrictedVector(ui, 'prediction', [.6875, .6875]);
  }
  restrictedVector(ui, 'prediction', [.7953125, .7953125]);
  assert.equal(restrictedAttribute(ui, 'phase'), 'complete');
  const finalGradient = JSON.parse(restrictedAttribute(ui, 'current-negative-gradient'));
  assert.ok(finalGradient[0] + finalGradient[1] > 0, 'Six finite updates still leave a small improving direction');
  ui.click('opening-replay');
  restrictedVector(ui, 'prediction', [.5, .5]);
  assert.equal(Number(ui.shell.dataset.round), 0);
  assert.equal(ui.shell.dataset.descentPhase, 'target');
});

test('the exact gradient limit follows the finite sequence as its own stationary scene', () => {
  const ui = setup({ reducedMotion: true });
  ui.scrollAt(gradientDescentPhysical);
  restrictedVector(ui, 'prediction', [.7953125, .7953125]);
  assert.equal(Number(ui.shell.dataset.round), 6);
  ui.click('next');
  assert.equal(Number(ui.shell.dataset.scene), gradientLimitPhysical);
  restrictedVector(ui, 'prediction', [.8, .8]);
  restrictedVector(ui, 'negative-gradient', [.32, -.32]);
  close(restrictedLoss(ui), .16);
  assert.equal(ui.get('opening-replay').hidden, true);
  assert.equal(ui.frames.size, 0);
  ui.advance(20000);
  restrictedVector(ui, 'prediction', [.8, .8]);
});

test('leaving the repeated descent cancels its frames and returning starts again at round zero', () => {
  const ui = setup();
  ui.scrollAt(gradientDescentPhysical); ui.advance(2250);
  restrictedVector(ui, 'prediction', [.575, .575]);
  ui.click('next');
  restrictedVector(ui, 'prediction', [.8, .8]);
  const limit = ui.get('restricted-diagram').innerHTML;
  ui.advance(openingDurations[gradientDescentPhysical]);
  assert.equal(ui.get('restricted-diagram').innerHTML, limit);
  ui.click('previous');
  restrictedVector(ui, 'prediction', [.5, .5]);
  assert.equal(Number(ui.shell.dataset.round), 0);
  assert.equal(ui.shell.dataset.descentPhase, 'target');
});

test('enabling reduced motion finishes the six-round sequence at its finite endpoint', () => {
  const ui = setup();
  ui.scrollAt(gradientDescentPhysical); ui.advance(5250);
  restrictedVector(ui, 'prediction', [.6875, .6875]);
  ui.reduced.matches = true; ui.reduced.emit('change');
  restrictedVector(ui, 'prediction', [.7953125, .7953125]);
  assert.equal(Number(ui.shell.dataset.round), 6);
  assert.equal(ui.shell.dataset.descentPhase, 'complete');
  ui.advance(20000);
  restrictedVector(ui, 'prediction', [.7953125, .7953125]);
});

test('reduced motion settles automatic opening transitions and replay immediately', () => {
  const ui = setup({ reducedMotion: true });
  ui.scrollAt(mseUpdatePhysical);
  restrictedVector(ui, 'prediction', [.5, .5]);
  close(Number(ui.shell.dataset.updateProgress), 1);
  ui.click('opening-replay');
  restrictedVector(ui, 'prediction', [.5, .5]);
  ui.scrollAt(changedLossPhysical);
  restrictedVector(ui, 'prediction', [.5, .5]);
  restrictedVector(ui, 'optimal-prediction', [.8, .8]);
  ui.scrollAt(gradientPhysical);
  restrictedVector(ui, 'prediction', [.5, .5]);
  ui.scrollAt(gradientDescentPhysical);
  restrictedVector(ui, 'prediction', [.7953125, .7953125]);
  close(Number(ui.shell.dataset.updateProgress), 1);
  ui.scrollAt(gradientLimitPhysical);
  restrictedVector(ui, 'prediction', [.8, .8]);
  assert.equal(ui.get('opening-replay').hidden, true);
});

test('enabling reduced motion finishes active opening playback without restarting it', () => {
  const ui = setup();
  ui.scrollAt(mseUpdatePhysical); ui.advance(500);
  assert.ok(Number(restrictedAttribute(ui, 'c')) > 0 && Number(restrictedAttribute(ui, 'c')) < .5);
  ui.reduced.matches = true; ui.reduced.emit('change');
  restrictedVector(ui, 'prediction', [.5, .5]);
  close(Number(ui.shell.dataset.updateProgress), 1);
  ui.advance(4000);
  restrictedVector(ui, 'prediction', [.5, .5]);
  ui.reduced.matches = false; ui.reduced.emit('change');
  restrictedVector(ui, 'prediction', [.5, .5]);
});

for (const compact of [false, true]) {
  test(`the two-point progression leads through a transition into the eight-scene boosting calculation (${compact ? 'narrow' : 'wide'} navigation)`, () => {
    const ui = setup({ reducedMotion: true, compact });
    assert.equal(Number(ui.shell.dataset.scene), 0);
    assert.equal(ui.shell.dataset.example, 'restricted');
    restrictedVector(ui, 'target', [1, 0]);
    restrictedVector(ui, 'prediction', [0, 0]);
    assert.equal(ui.get('previous').disabled, true);
    for (let scene = 1; scene <= 24; scene++) {
      ui.click('next');
      assert.equal(Number(ui.shell.dataset.scene), scene);
      assert.equal(ui.get('stage-count').textContent, `${String(scene + 1).padStart(2, '0')} / 25`);
      if (scene <= sandboxPhysical) {
        assert.equal(ui.shell.dataset.example, 'restricted');
        assert.match(ui.get('restricted-diagram').innerHTML, new RegExp(`data-restricted-scene="${scene}"`));
        assert.equal(ui.get('restricted-controls').hidden, scene !== sandboxPhysical);
        restrictedVector(ui, 'target', [1, 0]);
      } else if (scene === transitionPhysical) {
        assert.equal(ui.shell.dataset.example, 'transition');
        assert.equal(Number(ui.shell.dataset.transitionScene), 0);
        assert.equal(ui.get('worked-transition-diagram').hidden, false);
        assert.equal(ui.get('worked-diagram').hidden, true);
        assert.equal(ui.stageTools.hidden, true);
        assert.equal(ui.get('numbers').hidden, true);
      } else if (scene >= lossPhysical(0)) {
        assert.equal(ui.shell.dataset.example, 'loss');
        assert.equal(Number(ui.shell.dataset.lossScene), scene - lossPhysical(0));
        assert.equal(ui.get('loss-controls').hidden, false);
        assert.equal(ui.get('loss-choice').value, 'huber');
        assert.equal(ui.get('worked-diagram').hidden, true);
      } else {
        const local = scene - workedPhysical(0);
        assert.equal(ui.shell.dataset.example, 'worked');
        assert.match(ui.get('worked-diagram').innerHTML, new RegExp(`data-worked-scene="${local}"`));
        if (local === 0) {
          assert.match(numbers(ui), /\(2, 2, 2, 2\)/);
          close(figureLoss(ui), 7);
        }
        if (local === 1) {
          assert.match(numbers(ui), /\(2, 0, 1, [−-]3\)/);
          assert.match(numbers(ui), /\([−-]2, 0, [−-]1, 3\)/);
        }
      }
    }
    assert.equal(ui.get('next').textContent, 'Sources');
    ui.click('next');
    assert.equal(ui.referencesVisits(), 1);
    ui.scrollAt(0);
    assert.equal(ui.get('previous').disabled, true);
    assert.equal(ui.shell.dataset.example, 'restricted');
    ui.scrollAt(workedPhysical(0));
    assert.match(numbers(ui), /\(2, 2, 2, 2\)/);
    close(figureLoss(ui), 7);
  });
}

test('inspecting a candidate does not replace the selected stump, and half-rate refits the next targets', () => {
  const ui = setup({ reducedMotion: true });
  ui.scrollAt(workedPhysical(2));
  assert.equal(ui.get('worked-cut').value, '2');
  ui.control('worked-cut', 0, 'change');
  assert.match(numbers(ui), /Candidate cut x = 1\.5/);
  assert.match(numbers(ui), /26\/3/);
  ui.click('next');
  assert.match(numbers(ui), /\(1, 1, 1, 5\)/);
  close(figureLoss(ui), 1);
  ui.control('worked-rate', .5, 'change');
  assert.match(numbers(ui), /\(3\/2, 3\/2, 3\/2, 7\/2\)/);
  close(figureLoss(ui), 2.5);
  ui.click('next');
  assert.equal(ui.get('worked-cut').value, '0');
  assert.match(ui.get('worked-diagram').innerHTML, /data-values="-1\.5,0\.5,-0\.5,1\.5"/);
  ui.click('next');
  close(figureLoss(ui), 1.375);
  assert.match(numbers(ui), /\(3\/4, 7\/4, 7\/4, 15\/4\)/);
  ui.control('worked-rate', 1, 'change');
  close(figureLoss(ui), 1 / 3);
  assert.match(numbers(ui), /\(0, 4\/3, 4\/3, 16\/3\)/);
});

test('manual update progress represents a partial correction and cancels its running animation', () => {
  const ui = setup();
  ui.scrollAt(workedPhysical(3));
  close(figureLoss(ui), 7);
  assert.ok(ui.frames.size > 0);
  ui.control('worked-progress', .5);
  close(figureLoss(ui), 2.5);
  assert.equal(ui.get('worked-progress-value').textContent, '50%');
  ui.advance(2000);
  close(figureLoss(ui), 2.5);
  ui.control('worked-progress', 1);
  close(figureLoss(ui), 1);
  ui.click('next');
  assert.match(ui.get('worked-diagram').innerHTML, /data-values="-1,1,0,0"/);
});

test('leaving an animated update prevents a stale frame from overwriting the destination', () => {
  const ui = setup();
  ui.scrollAt(workedPhysical(3));
  assert.ok(ui.frames.size > 0);
  ui.scrollAt(workedPhysical(1));
  const gradient = ui.get('worked-diagram').innerHTML;
  ui.advance(2000);
  assert.equal(ui.get('worked-diagram').innerHTML, gradient);
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(1));
  assert.match(numbers(ui), /\([−-]2, 0, [−-]1, 3\)/);
  ui.scrollAt(workedPhysical(5));
  assert.ok(ui.frames.size > 0);
  ui.scrollAt(workedPhysical(6));
  const repeated = ui.get('worked-diagram').innerHTML;
  ui.advance(2000);
  assert.equal(ui.get('worked-diagram').innerHTML, repeated);
  close(figureLoss(ui), example.build(1, 8)[8].loss);
});

test('rate changes and reduced-motion changes settle an active update, while focused controls keep arrow keys', () => {
  const ui = setup();
  ui.scrollAt(workedPhysical(3)); ui.advance(250);
  assert.ok(figureLoss(ui) > 1);
  ui.control('worked-rate', .5, 'change');
  ui.advance(2000);
  close(figureLoss(ui), 2.5);
  ui.scrollAt(workedPhysical(5)); ui.advance(200);
  ui.reduced.matches = true; ui.reduced.emit('change');
  ui.advance(2000);
  close(figureLoss(ui), 1.375);
  assert.equal(ui.get('worked-progress-value').textContent, '100%');
  ui.scrollAt(workedPhysical(2));
  const nativeKey = ui.key('ArrowRight', ui.get('worked-cut'));
  assert.equal(nativeKey.defaultPrevented, undefined);
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(2));
  assert.equal(ui.key('ArrowRight').defaultPrevented, true);
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(3));
  ui.resize();
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(3));
});

test('the repeat control includes the baseline and rebuilds fresh gradients through twenty-four rounds', () => {
  const ui = setup({ reducedMotion: true });
  ui.scrollAt(workedPhysical(6));
  assert.equal(ui.get('worked-round').hidden, false);
  assert.equal(ui.get('worked-round-label').hidden, false);
  close(figureLoss(ui), example.build(1, 8)[8].loss);
  assert.equal(ui.get('worked-round').value, '8');
  ui.control('worked-round', 0);
  assert.match(numbers(ui), /\(2, 2, 2, 2\)/);
  close(figureLoss(ui), 7);
  ui.control('worked-round', 24);
  const fullRate = example.build(1, 24)[24];
  close(figureLoss(ui), fullRate.loss);
  assert.deepEqual(figurePredictions(ui), fullRate.predictions);
  ui.control('worked-rate', .5, 'change');
  const halfRate = example.build(.5, 24)[24];
  close(figureLoss(ui), halfRate.loss);
  assert.deepEqual(figurePredictions(ui), halfRate.predictions);
  assert.ok(halfRate.loss > fullRate.loss, 'The changed rate must affect the entire sequence');
  assert.equal(ui.get('worked-round').value, '24');
  const nativeKey = ui.key('ArrowLeft', ui.get('worked-round'));
  assert.equal(nativeKey.defaultPrevented, undefined);
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(6));
  ui.click('previous');
  close(figureLoss(ui), 1.375);
  ui.click('next');
  close(figureLoss(ui), halfRate.loss);
  ui.click('next');
  assert.equal(ui.get('worked-round').hidden, true);
  assert.equal(ui.get('worked-controls').hidden, true);
});

test('backward navigation retains its destination when the pinned stage changes height on a phone', () => {
  const ui = setup({ reducedMotion: true, compact: true, variableStageHeight: true });
  ui.scrollAt(workedPhysical(2));
  ui.click('next');
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(3));
  ui.resize();
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(3));
  ui.click('previous');
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(2));
  ui.resize();
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(2));
});

test('step links return to the phone stage and retain their target through smooth scrolling', () => {
  const ui = setup({ compact: true, variableStageHeight: true, smoothScroll: true });
  ui.scrollAt(workedPhysical(0));
  assert.equal(ui.clickLink('#worked-first-update').defaultPrevented, true);
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(3));
  ui.scrollFraction(.2);
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(3));
  ui.scrollFraction(1);
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(3));
  close(ui.shell.getBoundingClientRect().top, 64);
  assert.deepEqual(ui.visitedHashes, ['#worked-first-update']);
  ui.click('previous'); ui.scrollFraction(.5);
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(2));
  ui.scrollFraction(1); ui.resize();
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(2));
  ui.clickLink('#worked-data'); ui.scrollFraction(1);
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(0));
  ui.clickLink('#reading'); ui.scrollFraction(1);
  assert.equal(Number(ui.shell.dataset.scene), 0);
  ui.clickLink('#story'); ui.scrollFraction(1);
  assert.equal(Number(ui.shell.dataset.scene), 0);
});

test('manual wheel and page scrolling can interrupt an in-flight scene navigation', () => {
  const ui = setup({ smoothScroll: true });
  ui.scrollAt(workedPhysical(0)); ui.click('next');
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(1));
  ui.wheel(); ui.scrollAt(workedPhysical(0));
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(0));
  ui.click('next');
  ui.key('PageUp'); ui.scrollAt(workedPhysical(0));
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(0));
});

test('opening a worked-step hash returns to the phone stage, while references keeps native navigation', () => {
  const ui = setup({ reducedMotion: true, compact: true, variableStageHeight: true, hash: '#worked-first-update' });
  ui.advance(0); ui.advance(0);
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(3));
  close(ui.shell.getBoundingClientRect().top, 64);
  const references = setup({ reducedMotion: true, hash: '#references' });
  references.advance(0);
  assert.equal(references.scrollY(), 0);
  assert.equal(references.referencesVisits(), 0);
});

test('history navigation restores the linked step without adding another history entry', () => {
  const ui = setup({ reducedMotion: true, compact: true, variableStageHeight: true });
  ui.clickLink('#worked-first-fit');
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(2));
  ui.historyBack('#worked-first-update');
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(3));
  close(ui.shell.getBoundingClientRect().top, 64);
  ui.historyBack('');
  assert.equal(Number(ui.shell.dataset.scene), 0);
  assert.deepEqual(ui.visitedHashes, ['#worked-first-fit']);
});

test('the end of a cancelled old scroll does not release a newer scene navigation', () => {
  const ui = setup({ smoothScroll: true });
  ui.scrollAt(workedPhysical(0)); ui.click('next'); ui.scrollFraction(.2);
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(1));
  ui.click('next');
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(2));
  ui.scrollEnd(); ui.scrollFraction(.1);
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(2));
  ui.scrollFraction(1); ui.scrollEnd(); ui.resize();
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(2));
});

test('wide navigation aligns a short card below the sticky stage and scrolling still advances scenes', () => {
  const ui = setup({ reducedMotion: true });
  ui.clickLink('#loss-targets');
  assert.equal(Number(ui.shell.dataset.scene), lossPhysical(0));
  close(ui.headingTop(lossPhysical(0)) - ui.scrollY() - ui.shell.getBoundingClientRect().bottom, 10);
  ui.scrollAt(lossPhysical(1));
  assert.equal(Number(ui.shell.dataset.scene), lossPhysical(1));
  ui.scrollAt(workedPhysical(7));
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(7));
});

test('a phone hash aligns its card below the pinned figure and natural scrolling advances the story', () => {
  const ui = setup({ reducedMotion: true, compact: true, variableStageHeight: true, hash: '#loss-targets' });
  ui.advance(0); ui.advance(0);
  assert.equal(Number(ui.shell.dataset.scene), lossPhysical(0));
  close(ui.shell.getBoundingClientRect().top, 64);
  ui.scrollBy(100); ui.resize();
  assert.equal(Number(ui.shell.dataset.scene), lossPhysical(0));
  ui.scrollAt(2);
  assert.equal(Number(ui.shell.dataset.scene), 2);
  assert.ok(ui.shell.getBoundingClientRect().bottom > 64, 'The opening figure remains pinned while its chapters advance');
  ui.click('next');
  assert.equal(Number(ui.shell.dataset.scene), 3);
  close(ui.shell.getBoundingClientRect().top, 64);
  ui.resize(); ui.scrollBy(20);
  assert.equal(Number(ui.shell.dataset.scene), 3);
  ui.click('previous');
  assert.equal(Number(ui.shell.dataset.scene), 2);
  close(ui.shell.getBoundingClientRect().top, 64);
});

test('the number disclosure remeasures worked figures and is hidden for the opening, transition, and algorithm recap', () => {
  const ui = setup({ reducedMotion: true });
  assert.equal(ui.get('numbers').hidden, true);
  ui.clickLink('#worked-transition');
  assert.equal(ui.get('numbers').hidden, true);
  ui.clickLink('#worked-data');
  assert.equal(ui.get('numbers').hidden, false);
  const initialHeight = Number(/viewBox="0 0 790 ([^"]+)"/.exec(ui.get('worked-diagram').innerHTML)[1]);
  ui.get('visual-area').getBoundingClientRect = () => ({ width: 790, height: 540 });
  ui.get('numbers').emit('toggle');
  const expandedHeight = Number(/viewBox="0 0 790 ([^"]+)"/.exec(ui.get('worked-diagram').innerHTML)[1]);
  assert.equal(expandedHeight - initialHeight, 90, 'Opening the disclosure remeasures the available plot height');
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(0));
  ui.clickLink('#gradient-algorithm');
  assert.equal(ui.get('numbers').hidden, true);
  ui.clickLink('#loss-targets');
  assert.equal(ui.get('numbers').hidden, false);
});

for (const compact of [false, true]) {
  test(`late native fragment restoration keeps the initial plot-aware position (${compact ? 'phone' : 'wide'})`, () => {
    const ui = setup({ compact, smoothScroll: true, hash: '#identical-inputs' });
    ui.advance(0); ui.advance(0);
    assert.equal(Number(ui.shell.dataset.scene), 0);
    assert.equal(ui.scrollCalls.at(-1).behavior, 'instant');
    ui.nativeFragment(0); ui.load();
    assert.equal(Number(ui.shell.dataset.scene), 0);
    assert.equal(ui.scrollCalls.at(-1).behavior, 'instant');
    if (compact) close(ui.shell.getBoundingClientRect().top, 64);
    else close(ui.headingTop(0) - ui.scrollY() - ui.shell.getBoundingClientRect().bottom, 10);
    ui.click('next');
    assert.equal(ui.scrollCalls.at(-1).behavior, 'smooth');
    ui.scrollFraction(1);
    assert.equal(Number(ui.shell.dataset.scene), 1);
  });
}

test('late load does not restore the initial hash after the reader navigates elsewhere', () => {
  const ui = setup({ reducedMotion: true, hash: '#identical-inputs' });
  ui.advance(0); ui.advance(0);
  ui.click('next');
  const position = ui.scrollY();
  const requests = ui.scrollCalls.length;
  ui.load();
  assert.equal(Number(ui.shell.dataset.scene), 1);
  assert.equal(ui.scrollY(), position);
  assert.equal(ui.scrollCalls.length, requests);
});

for (const compact of [false, true]) {
  test(`same-document hash navigation restores plot placement after the native scroll (${compact ? 'phone' : 'wide'})`, () => {
    const ui = setup({ compact, smoothScroll: true, hash: '#identical-inputs' });
    ui.advance(0); ui.advance(0);
    ui.historyBack('#loss-targets');
    ui.nativeFragment(lossPhysical(0)); ui.hashChange();
    assert.equal(Number(ui.shell.dataset.scene), lossPhysical(0));
    assert.equal(ui.scrollCalls.at(-1).behavior, 'instant');
    if (compact) close(ui.shell.getBoundingClientRect().top, 64);
    else close(ui.headingTop(lossPhysical(0)) - ui.scrollY() - ui.shell.getBoundingClientRect().bottom, 10);
    const requests = ui.scrollCalls.length;
    ui.historyBack('#other-losses'); ui.hashChange();
    assert.equal(ui.scrollCalls.length, requests, 'Reading-only anchors keep native navigation');
  });
}

test('crossing the layout breakpoint preserves the selected chapter and restores its plot position', () => {
  const ui = setup({ reducedMotion: true, variableStageHeight: true });
  ui.clickLink('#loss-playground');
  assert.equal(Number(ui.shell.dataset.scene), sandboxPhysical);
  assert.ok(ui.scrollY() > stageTop + 660, 'The chosen chapter is deep into the story');
  ui.setCompact(true); ui.resize();
  assert.equal(Number(ui.shell.dataset.scene), sandboxPhysical);
  close(ui.shell.getBoundingClientRect().top, 64);
  assert.match(ui.get('restricted-diagram').innerHTML, /viewBox="0 0 288 340"/);
  ui.setCompact(false); ui.resize();
  assert.equal(Number(ui.shell.dataset.scene), sandboxPhysical);
  close(ui.headingTop(sandboxPhysical) - ui.scrollY() - ui.shell.getBoundingClientRect().bottom, 10);
  assert.match(ui.get('restricted-diagram').innerHTML, /viewBox="0 0 790 450"/);
});

test('resizing across the layout breakpoint keeps the opening title in place before the story is entered', () => {
  const ui = setup({ reducedMotion: true, stageDocumentTop: 1200 });
  ui.setCompact(true); ui.resize();
  assert.equal(ui.scrollY(), 0);
  assert.equal(ui.scrollCalls.length, 0);
  ui.setCompact(false); ui.resize();
  assert.equal(ui.scrollY(), 0);
  assert.equal(ui.scrollCalls.length, 0);
});

test('the transition introduces varied inputs and hides every plot, control, and readout', () => {
  const ui = setup();
  ui.scrollAt(sandboxPhysical);
  chooseRestrictedLoss(ui, 'expectile');
  ui.click('restricted-fit');
  assert.ok(ui.frames.size > 0, 'The preceding sandbox has an active fit animation');
  ui.click('next');
  assert.equal(Number(ui.shell.dataset.scene), transitionPhysical);
  assert.equal(ui.shell.dataset.example, 'transition');
  assert.equal(Number(ui.shell.dataset.transitionScene), 0);
  assert.equal(ui.shell.dataset.opening, 'true');
  assert.equal(ui.get('stage-topic').textContent, 'From two observations to four');
  assert.equal(ui.get('worked-transition-diagram').hidden, false);
  for (const id of ['restricted-diagram', 'worked-diagram', 'loss-diagram', 'restricted-controls', 'worked-controls', 'loss-controls', 'numbers', 'vector-readout', 'opening-annotation', 'opening-replay']) {
    assert.equal(ui.get(id).hidden, true, `${id} is hidden during the transition`);
  }
  assert.equal(ui.stageTools.hidden, true);
  for (const name of ['restrictedScene', 'workedScene', 'lossScene', 'descentPhase', 'round']) {
    assert.equal(ui.shell.dataset[name], undefined, `${name} is cleared on the transition`);
  }
  assert.equal(ui.frames.size, 0, 'The transition cancels the preceding fit without starting a worked animation');
  const before = ui.get('worked-diagram').innerHTML;
  ui.advance(4000);
  assert.equal(ui.get('worked-diagram').innerHTML, before);
  assert.equal(ui.frames.size, 0);
  assert.match(html, /id="worked-transition-diagram"[\s\S]*?Now let[\s\S]*?vary/);
  assert.match(html, /Four observations/);
  assert.equal(ui.get('prediction-diagram'), undefined, 'The old three-observation plot is removed from the main page');
  assert.equal(ui.get('geometry-controls'), undefined, 'Its controls are removed too');
});

for (const compact of [false, true]) {
  test(`next, previous, and keyboard navigation cross the transition in both directions (${compact ? 'narrow' : 'wide'})`, () => {
    const ui = setup({ reducedMotion: true, compact, variableStageHeight: true });
    ui.scrollAt(sandboxPhysical);
    for (const destination of [transitionPhysical, workedPhysical(0)]) {
      ui.click('next');
      assert.equal(Number(ui.shell.dataset.scene), destination);
      assert.equal(ui.get('stage-count').textContent, `${String(destination + 1).padStart(2, '0')} / 25`);
    }
    for (const destination of [transitionPhysical, sandboxPhysical]) {
      ui.click('previous');
      assert.equal(Number(ui.shell.dataset.scene), destination);
    }
    for (const destination of [transitionPhysical, workedPhysical(0)]) {
      assert.equal(ui.key('ArrowRight').defaultPrevented, true);
      assert.equal(Number(ui.shell.dataset.scene), destination);
    }
    for (const destination of [transitionPhysical, sandboxPhysical]) {
      assert.equal(ui.key('ArrowLeft').defaultPrevented, true);
      assert.equal(Number(ui.shell.dataset.scene), destination);
    }
    ui.clickLink('#loss-paths');
    assert.equal(Number(ui.shell.dataset.scene), 24);
    assert.equal(ui.get('stage-count').textContent, '25 / 25');
  });
}

test('legacy three-observation links go to the transition while the radius link returns to the opening', () => {
  const ui = setup({ reducedMotion: true });
  for (const hash of ['#vector-geometry', '#vector-correction']) {
    ui.clickLink('#worked-data');
    assert.equal(ui.clickLink(hash).defaultPrevented, true);
    assert.equal(Number(ui.shell.dataset.scene), transitionPhysical);
    assert.equal(ui.shell.dataset.example, 'transition');
  }
  assert.equal(ui.clickLink('#vector-radius').defaultPrevented, true);
  assert.equal(Number(ui.shell.dataset.scene), 0);
  assert.equal(ui.shell.dataset.example, 'restricted');
  restrictedVector(ui, 'target', [1, 0]);
  restrictedVector(ui, 'prediction', [0, 0]);
});

test('the opening sandbox and four-point calculation preserve independent choices across the transition', () => {
  const ui = setup({ reducedMotion: true });
  ui.clickLink('#loss-playground');
  chooseRestrictedLoss(ui, 'expectile');
  ui.click('restricted-fit');
  restrictedVector(ui, 'prediction', [.8, .8]);
  const before = ui.get('restricted-diagram').innerHTML;
  ui.click('next');
  assert.equal(ui.get('restricted-controls').hidden, true);
  assert.equal(ui.get('worked-controls').hidden, true);
  ui.click('next');
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(0));
  assert.equal(ui.get('worked-controls').hidden, false);
  ui.control('worked-rate', .5, 'change');
  ui.clickLink('#loss-playground');
  assert.equal(ui.get('restricted-diagram').innerHTML, before);
  assert.equal(ui.get('restricted-choice-expectile').getAttribute('aria-pressed'), 'true');
  ui.click('next'); ui.click('next');
  assert.equal(ui.get('worked-rate').value, '0.5');
});

test('native loss buttons select the sandbox loss without consuming focused navigation keys', () => {
  const ui = setup({ reducedMotion: true });
  ui.scrollAt(sandboxPhysical);
  assert.equal(ui.get('restricted-choice-squared').getAttribute('aria-pressed'), 'true');
  for (const loss of ['expectile', 'huber', 'quartic', 'squared']) {
    chooseRestrictedLoss(ui, loss);
    assert.equal(ui.shell.dataset.loss, loss);
    for (const option of ['squared', 'expectile', 'huber', 'quartic']) {
      assert.equal(ui.get('restricted-choice-' + option).getAttribute('aria-pressed'), String(option === loss));
    }
    for (const key of ['Home', 'End', 'ArrowRight', 'ArrowLeft']) {
      assert.equal(ui.key(key, ui.get('restricted-choice-' + loss)).defaultPrevented, undefined);
      assert.equal(Number(ui.shell.dataset.scene), sandboxPhysical);
    }
    restrictedVector(ui, 'prediction', [.5, .5]);
  }
  ui.click('restricted-reset');
  restrictedVector(ui, 'prediction', [0, 0]);
  restrictedVector(ui, 'next-tree', [.5, .5]);
  close(restrictedLoss(ui), .5);
  ui.click('restricted-fit');
  restrictedVector(ui, 'prediction', [.5, .5]);
  restrictedVector(ui, 'residual', [.5, -.5]);
  restrictedVector(ui, 'next-tree', [0, 0]);
  close(restrictedLoss(ui), .25);
});

for (const compact of [false, true]) {
  test(`the sandbox buttons expose asymmetric targets and improve on the residual mean (${compact ? 'narrow' : 'wide'})`, () => {
    const ui = setup({ reducedMotion: true, compact });
    ui.clickLink('#loss-playground');
    chooseRestrictedLoss(ui, 'expectile');
    restrictedVector(ui, 'target', [1, 0]);
    restrictedVector(ui, 'prediction', [.5, .5]);
    restrictedVector(ui, 'residual', [.5, -.5]);
    restrictedVector(ui, 'negative-gradient', [.8, -.2]);
    restrictedVector(ui, 'next-tree', [.3, .3]);
    close(restrictedLoss(ui), .25);
    ui.click('restricted-fit');
    restrictedVector(ui, 'prediction', [.8, .8]);
    restrictedVector(ui, 'negative-gradient', [.32, -.32]);
    restrictedVector(ui, 'next-tree', [0, 0]);
    close(restrictedLoss(ui), .16);
    ui.click('restricted-reset');
    restrictedVector(ui, 'prediction', [0, 0]);
    restrictedVector(ui, 'next-tree', [.8, .8]);
    close(restrictedLoss(ui), .8);
    ui.click('restricted-mean');
    restrictedVector(ui, 'prediction', [.5, .5]);
    close(restrictedLoss(ui), .25);
  });
}

test('sandbox moves animate the prediction and loss switches hold that prediction fixed', () => {
  const ui = setup();
  ui.scrollAt(sandboxPhysical);
  ui.click('restricted-reset');
  restrictedVector(ui, 'prediction', [.5, .5]);
  ui.advance(800);
  restrictedVector(ui, 'prediction', [.25, .25]);
  ui.advance(800);
  restrictedVector(ui, 'prediction', [0, 0]);
  ui.click('restricted-fit');
  ui.advance(800);
  restrictedVector(ui, 'prediction', [.25, .25]);
  finishSharedMove(ui);
  restrictedVector(ui, 'prediction', [.5, .5]);
  chooseRestrictedLoss(ui, 'expectile');
  restrictedVector(ui, 'prediction', [.5, .5]);
  ui.click('restricted-fit');
  ui.advance(800);
  restrictedVector(ui, 'prediction', [.65, .65]);
  ui.advance(800);
  restrictedVector(ui, 'prediction', [.8, .8]);
  close(restrictedLoss(ui), .16);
});

test('a loss-button click cancels a sandbox move and keeps its current prediction', () => {
  const ui = setup();
  ui.scrollAt(sandboxPhysical);
  ui.click('restricted-reset'); ui.advance(800);
  restrictedVector(ui, 'prediction', [.25, .25]);
  chooseRestrictedLoss(ui, 'quartic');
  restrictedVector(ui, 'prediction', [.25, .25]);
  ui.advance(4000);
  restrictedVector(ui, 'prediction', [.25, .25]);
  assert.equal(ui.shell.dataset.loss, 'quartic');
});

test('enabling reduced motion completes an active sandbox move at its chosen destination', () => {
  const ui = setup();
  ui.scrollAt(sandboxPhysical);
  chooseRestrictedLoss(ui, 'expectile');
  ui.click('restricted-fit'); ui.advance(400);
  assert.ok(Number(restrictedAttribute(ui, 'c')) > .5 && Number(restrictedAttribute(ui, 'c')) < .8);
  ui.reduced.matches = true; ui.reduced.emit('change');
  restrictedVector(ui, 'prediction', [.8, .8]);
  close(restrictedLoss(ui), .16);
  ui.advance(4000);
  restrictedVector(ui, 'prediction', [.8, .8]);
});

test('Huber and fourth-power buttons use their original-loss shared corrections', () => {
  const ui = setup({ reducedMotion: true });
  ui.scrollAt(sandboxPhysical);
  chooseRestrictedLoss(ui, 'huber');
  restrictedVector(ui, 'optimum-interval', [.25, .75]);
  restrictedVector(ui, 'next-tree', [0, 0]);
  restrictedVector(ui, 'correction', [0, 0]);
  const minimum = restrictedLoss(ui);
  ui.click('restricted-fit');
  restrictedVector(ui, 'prediction', [.5, .5]);
  close(restrictedLoss(ui), minimum);
  ui.click('restricted-reset');
  restrictedVector(ui, 'next-tree', [.125, .125]);
  restrictedVector(ui, 'correction', [.25, .25]);
  ui.click('restricted-fit');
  restrictedVector(ui, 'prediction', [.25, .25]);
  close(restrictedLoss(ui), minimum);
  chooseRestrictedLoss(ui, 'quartic');
  restrictedVector(ui, 'negative-gradient', [.421875, -.015625]);
  restrictedVector(ui, 'next-tree', [.203125, .203125]);
  restrictedVector(ui, 'correction', [.25, .25]);
  ui.click('restricted-fit');
  restrictedVector(ui, 'prediction', [.5, .5]);
  close(restrictedLoss(ui), 1 / 32);
  chooseRestrictedLoss(ui, 'squared');
  close(restrictedLoss(ui), .25);
});

test('the sandbox keeps its loss and prediction independently of the later loss comparison', () => {
  const ui = setup({ reducedMotion: true });
  ui.clickLink('#loss-playground');
  chooseRestrictedLoss(ui, 'expectile');
  ui.click('restricted-fit');
  const before = ui.get('restricted-diagram').innerHTML;
  ui.clickLink('#loss-targets');
  assert.equal(ui.get('restricted-controls').hidden, true);
  assert.equal(ui.get('loss-choice').value, 'huber');
  assert.equal(ui.get('loss-delta').value, '1');
  ui.control('loss-choice', 'expectile', 'change');
  ui.control('loss-tau', .2, 'change');
  ui.clickLink('#loss-playground');
  assert.equal(ui.get('restricted-diagram').innerHTML, before);
  assert.equal(ui.get('restricted-choice-expectile').getAttribute('aria-pressed'), 'true');
  restrictedVector(ui, 'prediction', [.8, .8]);
  close(Number(restrictedAttribute(ui, 'tau')), .8);
  close(Number(restrictedAttribute(ui, 'delta')), .25);
});

test('phone navigation goes from the opening sandbox through the transition into the four-point example', () => {
  const ui = setup({ compact: true, variableStageHeight: true, smoothScroll: true });
  ui.scrollAt(sandboxPhysical);
  for (const destination of [transitionPhysical, workedPhysical(0)]) {
    ui.click('next'); ui.scrollFraction(.2);
    assert.equal(Number(ui.shell.dataset.scene), destination);
    ui.scrollFraction(1); ui.resize();
    assert.equal(Number(ui.shell.dataset.scene), destination);
    close(ui.shell.getBoundingClientRect().top, 64);
  }
  for (const destination of [transitionPhysical, sandboxPhysical]) {
    ui.click('previous'); ui.scrollFraction(.5);
    assert.equal(Number(ui.shell.dataset.scene), destination);
    ui.scrollFraction(1); ui.resize();
    assert.equal(Number(ui.shell.dataset.scene), destination);
    close(ui.shell.getBoundingClientRect().top, 64);
  }
});

test('the identical-input hash opens the new scene before the worked-data hash', () => {
  const ui = setup({ reducedMotion: true, compact: true, variableStageHeight: true, hash: '#identical-inputs' });
  ui.advance(0); ui.advance(0);
  assert.equal(Number(ui.shell.dataset.scene), 0);
  assert.equal(ui.shell.dataset.example, 'restricted');
  close(ui.shell.getBoundingClientRect().top, 64);
  ui.clickLink('#worked-data');
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(0));
  assert.equal(ui.shell.dataset.example, 'worked');
  ui.historyBack('#identical-inputs');
  assert.equal(Number(ui.shell.dataset.scene), 0);
  assert.equal(ui.shell.dataset.example, 'restricted');
  assert.deepEqual(ui.visitedHashes, ['#worked-data']);
});

test('returning to the transition during an animated worked update cancels the stale correction', () => {
  const ui = setup();
  ui.scrollAt(workedPhysical(3));
  assert.ok(ui.frames.size > 0);
  ui.scrollAt(transitionPhysical);
  const restored = ui.get('worked-diagram').innerHTML;
  ui.advance(2000);
  assert.equal(ui.get('worked-diagram').innerHTML, restored);
  assert.equal(Number(ui.shell.dataset.scene), transitionPhysical);
  assert.equal(ui.frames.size, 0);
  assert.equal(ui.get('worked-diagram').hidden, true);
  assert.equal(ui.get('worked-controls').hidden, true);
  assert.equal(ui.get('worked-transition-diagram').hidden, false);
});


test('loss controls refit the compared paths and persist across narrative scenes', () => {
  const ui = setup({ reducedMotion: true });
  ui.clickLink('#loss-targets');
  assert.equal(Number(ui.shell.dataset.scene), lossPhysical(0));
  assert.equal(ui.shell.dataset.loss, 'huber');
  assert.equal(ui.get('loss-delta-label').hidden, false);
  assert.equal(ui.get('loss-tau-label').hidden, true);
  const initial = ui.get('loss-diagram').innerHTML;
  ui.control('loss-choice', 'expectile', 'change');
  assert.equal(ui.get('loss-delta-label').hidden, true);
  assert.equal(ui.get('loss-tau-label').hidden, false);
  assert.notEqual(ui.get('loss-diagram').innerHTML, initial);
  ui.control('loss-tau', .2, 'change');
  ui.control('loss-rate', .25, 'change');
  ui.control('loss-dataset', 'worked', 'change');
  ui.control('loss-round', 24);
  ui.click('next');
  assert.equal(ui.get('loss-tau').value, '0.2');
  assert.equal(ui.get('loss-rate').value, '0.25');
  assert.equal(ui.get('loss-round-value').textContent, '24');
  assert.equal(ui.get('loss-dataset').value, 'worked');
  assert.equal(ui.shell.dataset.loss, 'expectile');
  assert.equal(ui.shell.dataset.round, 24);
  ui.clickLink('#worked-data');
  assert.equal(ui.get('loss-controls').hidden, true);
  assert.equal(ui.get('loss-diagram').hidden, true);
  ui.clickLink('#loss-paths');
  assert.equal(ui.get('loss-round').value, '24');
  assert.equal(ui.get('loss-choice').value, 'expectile');
  assert.equal(ui.get('worked-controls').hidden, true);
});

test('native readouts retain precise values and update accessible captions on scene changes', () => {
  const ui = setup({ reducedMotion: true });
  ui.scrollAt(workedPhysical(5));
  assert.match(ui.get('stage-caption').innerHTML, /<math/);
  assert.match(ui.get('stage-caption').innerHTML, /<mfrac>/);
  assert.match(ui.get('vector-readout').children[0].children[1].innerHTML, /<math/);
  ui.scrollAt(lossPhysical(2));
  assert.match(ui.get('stage-caption').innerHTML, /<munder><mo>∑<\/mo>/);
  assert.match(ui.get('stage-caption').getAttribute('aria-label'), /Minimize Σℓ/);
  ui.scrollAt(transitionPhysical);
  assert.equal(ui.get('stage-caption').getAttribute('aria-label'), undefined);
  assert.equal(ui.get('stage-caption').textContent, '');
  ui.scrollAt(sandboxPhysical);
  assert.match(ui.get('stage-caption').getAttribute('aria-label'), /At \(0\.5, 0\.5\), the residual/);
  assert.match(ui.get('stage-caption').innerHTML, /<math/);
  assert.ok(ui.get('stage-caption').textContent.length > 0);
});

test('opening status and floating captions typeset corrections without changing announced values', () => {
  const ui = setup({ reducedMotion: true });
  ui.scrollAt(2);
  assert.match(ui.get('opening-status').innerHTML, /<mi>h<\/mi><mo>=<\/mo>.*<mi>a<\/mi>/);
  assert.equal(ui.get('opening-status').getAttribute('aria-label'), 'Same x → same correction h = (a, a)');
  ui.scrollAt(gradientFitPhysical);
  assert.match(ui.get('opening-annotation').innerHTML, /<mi>h<\/mi><mo stretchy="false">\(<\/mo><mi>x<\/mi>/);
  assert.match(ui.get('opening-status').innerHTML, /<mfrac>/);
  assert.equal(ui.get('opening-status').getAttribute('aria-label'), 'h(1) = 0.3 · α = ½ · Next predictions = (0.65, 0.65)');
  ui.scrollAt(transitionPhysical);
  assert.equal(ui.get('opening-status').getAttribute('aria-label'), undefined);
  assert.equal(ui.get('opening-status').textContent, 'Scroll to continue');
});
