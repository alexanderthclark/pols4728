import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { runInNewContext } from 'node:vm';
import test from 'node:test';

const require = createRequire(import.meta.url);
const math = require('../docs/shared/boosting-math.js');
const example = require('../docs/boosting/worked-example.js');
const view = require('../docs/boosting/worked-view.js');
const geometry = require('../docs/boosting/prediction-view.js');
const restricted = require('../docs/boosting/restricted-view.js');
const lossView = require('../docs/boosting/loss-view.js');
const html = readFileSync(new URL('../docs/boosting/index.html', import.meta.url), 'utf8');
const source = readFileSync(new URL('../docs/boosting/story.js', import.meta.url), 'utf8');
const workedPhysical = local => local + 10;
const geometryPhysical = local => local + 8;
const lossPhysical = local => local + 18;
const mseUpdatePhysical = 3;
const orthogonalPhysical = 4;
const changedLossPhysical = 5;
const gradientPhysical = 6;
const sandboxPhysical = 7;
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
    toggleAttribute(name, enabled = !this.attrs.has(name)) {
      if (enabled) this.attrs.set(name, ''); else this.attrs.delete(name);
      if (name === 'hidden') this.hidden = enabled;
      return enabled;
    }
    appendChild(child) { this.children.push(child); return child; }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; this._text = ''; }
    set textContent(value) { this._text = String(value); this.children = []; }
    get textContent() { return this._text + this.children.map(child => child.textContent).join(' '); }
    getBoundingClientRect() { return { top: 64, bottom: 800, width: 640, height: 450 }; }
  }
  const nodes = new Map();
  for (const match of html.matchAll(/<(\w+)\b([^>]*\bid="([^"]+)"[^>]*)>/g)) {
    const element = new Element(match[1].toUpperCase());
    element.id = match[3];
    element.hidden = /\bhidden(?:\s|$)/.test(match[2]);
    element.disabled = /\bdisabled(?:\s|$)/.test(match[2]);
    element.value = /\bvalue="([^"]*)"/.exec(match[2])?.[1] || '';
    nodes.set(element.id, element);
  }
  for (const match of html.matchAll(/<select\b[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g)) {
    const options = [...match[2].matchAll(/<option\b([^>]*)>/g)];
    const selected = options.find(option => /\bselected\b/.test(option[1])) || options[0];
    nodes.get(match[1]).value = /\bvalue="([^"]*)"/.exec(selected[1])[1];
  }
  const get = id => nodes.get(id);
  const shell = new Element();
  const stageHeight = () => !narrow.matches ? 500 : !variableStageHeight ? 620
    : shell.dataset.example === 'restricted' ? 660
    : shell.dataset.example === 'worked' && [3, 5, 6].includes(Number(shell.dataset.workedScene)) ? 528 : 392;
  const headingTop = physical => (narrow.matches ? stageDocumentTop + stageHeight() + 48 : stageDocumentTop + 600) + physical * 1000;
  // The phone stage moves with the document; only the wide stage is sticky.
  shell.getBoundingClientRect = () => {
    const top = narrow.matches ? stageDocumentTop - win.scrollY : Math.max(64, stageDocumentTop - win.scrollY);
    return { top, bottom: top + stageHeight(), height: stageHeight(), width: narrow.matches ? 288 : 1100 };
  };
  get('visual-area').getBoundingClientRect = () => ({ width: narrow.matches ? 288 : 790, height: narrow.matches ? 340 : 450 });
  const win = new Element();
  const reduced = new Element(); reduced.matches = reducedMotion;
  const narrow = new Element(); narrow.matches = compact;
  Object.assign(win, {
    BoostingMath: math, WorkedBoostingExample: example, WorkedBoostingView: view, PredictionBoostingView: geometry, RestrictedBoostingView: restricted, LossBoostingView: lossView,
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
  let afterwordVisits = 0;
  get('other-losses').scrollIntoView = () => { afterwordVisits++; };
  const doc = new Element();
  doc.getElementById = get;
  doc.querySelector = selector => selector === '.stage-shell' ? shell : selector === '.story-layout' ? layout : null;
  doc.querySelectorAll = selector => selector === '.step' ? steps : [];
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
    const threshold = narrow.matches ? win.innerHeight * .35 : 64 + stageHeight() + 32;
    win.scrollY = headingTop(index) - threshold + 1;
    if (narrow.matches) win.scrollY = Math.max(win.scrollY, stageDocumentTop + stageHeight() - 64 + 1);
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
  return { get, shell, click, clickLink, control, scrollAt, scrollFraction, advance, key, resize, reduced, frames, headingTop,
    setCompact: matches => { narrow.matches = matches; win.innerWidth = matches ? 320 : 1200; },
    historyBack: hash => { win.location.hash = hash; win.emit('popstate'); advance(0); },
    nativeFragment: index => { win.scrollY = headingTop(index) - 64; win.emit('scroll'); },
    hashChange: () => { win.emit('hashchange'); advance(0); advance(0); },
    load: () => { win.emit('load'); advance(0); advance(0); }, scrollCalls,
    scrollBy: distance => { win.scrollY += distance; win.emit('scroll'); advance(0); },
    wheel: () => win.emit('wheel'), scrollEnd: () => win.emit('scrollend'), visitedHashes, scrollY: () => win.scrollY, afterwordVisits: () => afterwordVisits };
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
const restrictedAnnotation = (ui, className) => new RegExp(`<text\\b[^>]*class="[^"]*\\b${className}\\b[^"]*"[^>]*>([\\s\\S]*?)<\\/text>`).exec(ui.get('restricted-diagram').innerHTML)?.[1];

test('the opening reveals observations, residuals, and then the shared-input restriction separately', () => {
  const ui = setup({ reducedMotion: true });
  let svg = ui.get('restricted-diagram').innerHTML;
  assert.doesNotMatch(svg, /class="restricted-(?:residual-arrow|available-line|tree-arrow|optimum|current-contour|optimal-contour)"/);
  assert.equal(ui.get('restricted-loss-equation').hidden, true);
  ui.click('next');
  svg = ui.get('restricted-diagram').innerHTML;
  assert.equal(Number(ui.shell.dataset.scene), 1);
  assert.match(svg, /class="restricted-residual-arrow"/);
  assert.equal(restrictedAnnotation(ui, 'restricted-residual-label'), 'r = (1, 0)');
  assert.doesNotMatch(svg, /class="restricted-(?:available-line|tree-arrow|optimum|current-contour|optimal-contour)"/);
  assert.equal(ui.get('restricted-loss-equation').hidden, true);
  ui.click('next');
  svg = ui.get('restricted-diagram').innerHTML;
  assert.equal(Number(ui.shell.dataset.scene), 2);
  assert.match(svg, /class="restricted-available-line"/);
  assert.match(restrictedVisibleWords(ui), /x\s*=\s*1/);
  assert.match(restrictedVisibleWords(ui), /same correction/i);
  assert.match(restrictedVisibleWords(ui), /45°/);
  assert.doesNotMatch(svg, /class="restricted-(?:tree-arrow|optimum|mean|current-contour|optimal-contour)"/);
  assert.equal(ui.get('restricted-loss-equation').hidden, false);
});

test('the opening introduces residual fitting and orthogonality before gradient terminology or loss controls', () => {
  const ui = setup({ reducedMotion: true });
  for (let scene = 0; scene <= changedLossPhysical; scene++) {
    if (scene > 0) ui.click('next');
    assert.equal(ui.shell.dataset.example, 'restricted');
    assert.equal(ui.get('restricted-sandbox-controls').hidden, true);
    assert.equal(ui.get('restricted-progression-controls').hidden, ![mseUpdatePhysical, changedLossPhysical].includes(scene));
    assert.doesNotMatch(restrictedVisibleWords(ui), /gradient|∇/i);
    assert.doesNotMatch(ui.get('restricted-diagram').innerHTML, /class="restricted-gradient-arrow"/);
    restrictedVector(ui, 'target', [1, 0]);
    const narrative = new RegExp(`<section\\b[^>]*data-step="${scene}"[^>]*>([\\s\\S]*?)<\\/section>`).exec(html)[1];
    assert.doesNotMatch(narrative, /gradient|∇/i);
    if (scene === orthogonalPhysical) {
      restrictedVector(ui, 'prediction', [.5, .5]);
      restrictedVector(ui, 'residual', [.5, -.5]);
      const residual = JSON.parse(restrictedAttribute(ui, 'residual'));
      close(residual[0] + residual[1], 0);
      close(restrictedLoss(ui), .25);
      assert.match(ui.get('restricted-diagram').innerHTML, /class="restricted-right-angle"/);
      assert.match(restrictedVisibleWords(ui), /perpendicular|orthogonal/i);
    }
  }
  restrictedVector(ui, 'prediction', [.5, .5]);
  restrictedVector(ui, 'optimal-prediction', [.8, .8]);
  close(restrictedLoss(ui), .25);
  ui.click('next');
  assert.equal(Number(ui.shell.dataset.scene), gradientPhysical);
  assert.match(restrictedVisibleWords(ui), /negative gradient/i);
  assert.match(ui.get('restricted-diagram').innerHTML, /class="restricted-gradient-arrow"/);
  restrictedVector(ui, 'negative-gradient', [.8, -.2]);
  restrictedVector(ui, 'next-tree', [.3, .3]);
  ui.click('next');
  assert.equal(ui.get('restricted-sandbox-controls').hidden, false);
  assert.equal(ui.get('restricted-progression-controls').hidden, true);
  assert.equal(ui.get('restricted-loss').value, 'squared');
  restrictedVector(ui, 'prediction', [.5, .5]);
});

test('the MSE update starts paused at zero and plays toward the residual projection on request', () => {
  const ui = setup();
  ui.clickLink('#residual-move');
  close(restrictedLoss(ui), .5);
  restrictedVector(ui, 'prediction', [0, 0]);
  restrictedVector(ui, 'residual', [1, 0]);
  assert.equal(ui.frames.size, 0, 'Entering the residual update leaves it paused');
  ui.advance(3000);
  restrictedVector(ui, 'prediction', [0, 0]);
  restrictedVector(ui, 'residual', [1, 0]);
  assert.equal(restrictedAnnotation(ui, 'restricted-prediction-label'), 'f = (0, 0)');
  assert.equal(restrictedAnnotation(ui, 'restricted-residual-label'), 'r = (1, 0)');
  assert.equal(ui.get('restricted-progress-value').textContent, '0%');
  ui.click('restricted-play');
  assert.ok(ui.frames.size > 0, 'Playing the residual update starts its progression');
  ui.advance(900);
  restrictedVector(ui, 'prediction', [.25, .25]);
  restrictedVector(ui, 'residual', [.75, -.25]);
  assert.equal(restrictedAnnotation(ui, 'restricted-prediction-label'), 'f = (0.25, 0.25)');
  assert.equal(restrictedAnnotation(ui, 'restricted-residual-label'), 'r = (0.75, −0.25)');
  close(restrictedLoss(ui), .3125);
  assert.equal(ui.get('restricted-progress-value').textContent, '50%');
  ui.advance(900);
  restrictedVector(ui, 'prediction', [.5, .5]);
  restrictedVector(ui, 'residual', [.5, -.5]);
  assert.equal(restrictedAnnotation(ui, 'restricted-prediction-label'), 'f = (0.5, 0.5)');
  assert.equal(restrictedAnnotation(ui, 'restricted-residual-label'), 'r = (0.5, −0.5)');
  close(restrictedLoss(ui), .25);
  assert.equal(ui.get('restricted-play').textContent, 'Replay update');
  ui.click('restricted-play');
  restrictedVector(ui, 'prediction', [0, 0]);
  ui.advance(1800);
  restrictedVector(ui, 'prediction', [.5, .5]);
});

test('scrubbing an opening update cancels playback and changing chapters cancels stale frames', () => {
  const ui = setup();
  ui.scrollAt(mseUpdatePhysical); ui.click('restricted-play'); ui.advance(200);
  ui.control('restricted-progress', .6);
  restrictedVector(ui, 'prediction', [.3, .3]);
  ui.advance(2000);
  restrictedVector(ui, 'prediction', [.3, .3]);
  assert.equal(ui.get('restricted-progress-value').textContent, '60%');
  ui.click('restricted-play'); ui.advance(200);
  ui.scrollAt(orthogonalPhysical);
  const orthogonal = ui.get('restricted-diagram').innerHTML;
  ui.advance(2000);
  assert.equal(ui.get('restricted-diagram').innerHTML, orthogonal);
  restrictedVector(ui, 'prediction', [.5, .5]);
  ui.scrollAt(mseUpdatePhysical);
  restrictedVector(ui, 'prediction', [0, 0]);
  for (const key of ['Home', 'End', 'ArrowRight', 'ArrowLeft']) {
    assert.equal(ui.key(key, ui.get('restricted-progress')).defaultPrevented, undefined);
    assert.equal(Number(ui.shell.dataset.scene), mseUpdatePhysical);
  }
});

test('the changed-loss progression improves on the residual mean before revealing its gradient target', () => {
  const ui = setup();
  ui.scrollAt(changedLossPhysical);
  restrictedVector(ui, 'prediction', [.5, .5]);
  close(restrictedLoss(ui), .25);
  assert.doesNotMatch(restrictedVisibleWords(ui), /gradient|∇/i);
  ui.click('restricted-play'); ui.advance(900);
  restrictedVector(ui, 'prediction', [.65, .65]);
  close(restrictedLoss(ui), .1825);
  ui.advance(900);
  restrictedVector(ui, 'prediction', [.8, .8]);
  close(restrictedLoss(ui), .16);
  ui.click('next');
  restrictedVector(ui, 'prediction', [.5, .5]);
  restrictedVector(ui, 'residual', [.5, -.5]);
  restrictedVector(ui, 'negative-gradient', [.8, -.2]);
  ui.click('restricted-play'); ui.advance(1800);
  restrictedVector(ui, 'prediction', [.8, .8]);
  restrictedVector(ui, 'negative-gradient', [.32, -.32]);
  const gradient = JSON.parse(restrictedAttribute(ui, 'negative-gradient'));
  close(gradient[0] + gradient[1], 0);
  close(restrictedLoss(ui), .16);
});

test('reduced motion keeps opening examples paused and settles updates only when played', () => {
  const ui = setup({ reducedMotion: true });
  ui.scrollAt(mseUpdatePhysical);
  restrictedVector(ui, 'prediction', [0, 0]);
  assert.equal(ui.get('restricted-progress-value').textContent, '0%');
  ui.advance(3000);
  restrictedVector(ui, 'prediction', [0, 0]);
  ui.click('restricted-play');
  restrictedVector(ui, 'prediction', [.5, .5]);
  assert.equal(ui.get('restricted-progress-value').textContent, '100%');
  ui.scrollAt(changedLossPhysical);
  restrictedVector(ui, 'prediction', [.5, .5]);
  assert.equal(ui.get('restricted-progress-value').textContent, '0%');
  ui.click('restricted-play');
  restrictedVector(ui, 'prediction', [.8, .8]);
  ui.advance(2000);
  restrictedVector(ui, 'prediction', [.8, .8]);
});

test('motion-preference changes preserve a paused opening update and settle active playback', () => {
  const ui = setup();
  ui.scrollAt(mseUpdatePhysical);
  ui.reduced.matches = true; ui.reduced.emit('change');
  restrictedVector(ui, 'prediction', [0, 0]);
  assert.equal(ui.get('restricted-progress-value').textContent, '0%');
  ui.control('restricted-progress', .6);
  ui.reduced.matches = false; ui.reduced.emit('change');
  restrictedVector(ui, 'prediction', [.3, .3]);
  assert.equal(ui.get('restricted-progress-value').textContent, '60%');
  ui.advance(3000);
  restrictedVector(ui, 'prediction', [.3, .3]);
  ui.click('restricted-play'); ui.advance(500);
  assert.ok(Number(restrictedAttribute(ui, 'c')) > 0 && Number(restrictedAttribute(ui, 'c')) < .5);
  ui.reduced.matches = true; ui.reduced.emit('change');
  restrictedVector(ui, 'prediction', [.5, .5]);
  assert.equal(ui.get('restricted-progress-value').textContent, '100%');
  ui.advance(3000);
  restrictedVector(ui, 'prediction', [.5, .5]);
});

for (const compact of [false, true]) {
  test(`the two-point progression precedes vectors and the eight-scene boosting calculation (${compact ? 'narrow' : 'wide'} navigation)`, () => {
    const ui = setup({ reducedMotion: true, compact });
    assert.equal(Number(ui.shell.dataset.scene), 0);
    assert.equal(ui.shell.dataset.example, 'restricted');
    restrictedVector(ui, 'target', [1, 0]);
    restrictedVector(ui, 'prediction', [0, 0]);
    assert.equal(ui.get('previous').disabled, true);
    for (let scene = 1; scene <= 21; scene++) {
      ui.click('next');
      assert.equal(Number(ui.shell.dataset.scene), scene);
      assert.equal(ui.get('stage-count').textContent, `${String(scene + 1).padStart(2, '0')} / 22`);
      if (scene <= sandboxPhysical) {
        assert.equal(ui.shell.dataset.example, 'restricted');
        assert.match(ui.get('restricted-diagram').innerHTML, new RegExp(`data-restricted-scene="${scene}"`));
        assert.equal(ui.get('restricted-controls').hidden, false);
        restrictedVector(ui, 'target', [1, 0]);
      } else if (scene < workedPhysical(0)) {
        assert.equal(ui.shell.dataset.example, 'geometry');
        assert.match(ui.get('prediction-diagram').innerHTML, new RegExp(`data-geometry-scene="${scene - geometryPhysical(0)}"`));
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
    ui.click('next');
    assert.equal(ui.afterwordVisits(), 1);
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

test('backward navigation retains its destination when the static stage changes height on a phone', () => {
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

test('opening a worked-step hash returns to the phone stage, while other-losses keeps native navigation', () => {
  const ui = setup({ reducedMotion: true, compact: true, variableStageHeight: true, hash: '#worked-first-update' });
  ui.advance(0); ui.advance(0);
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(3));
  close(ui.shell.getBoundingClientRect().top, 64);
  const afterword = setup({ reducedMotion: true, hash: '#other-losses' });
  afterword.advance(0);
  assert.equal(afterword.scrollY(), 0);
  assert.equal(afterword.afterwordVisits(), 0);
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

test('a phone hash stays selected while the figure is visible, and Next returns from manual reading', () => {
  const ui = setup({ reducedMotion: true, compact: true, variableStageHeight: true, hash: '#loss-targets' });
  ui.advance(0); ui.advance(0);
  assert.equal(Number(ui.shell.dataset.scene), lossPhysical(0));
  close(ui.shell.getBoundingClientRect().top, 64);
  ui.scrollBy(100); ui.resize();
  assert.equal(Number(ui.shell.dataset.scene), lossPhysical(0));
  ui.scrollAt(2);
  assert.equal(Number(ui.shell.dataset.scene), 2);
  assert.ok(ui.shell.getBoundingClientRect().bottom <= 64);
  ui.click('next');
  assert.equal(Number(ui.shell.dataset.scene), 3);
  close(ui.shell.getBoundingClientRect().top, 64);
  ui.resize(); ui.scrollBy(20);
  assert.equal(Number(ui.shell.dataset.scene), 3);
  ui.click('previous');
  assert.equal(Number(ui.shell.dataset.scene), 2);
  close(ui.shell.getBoundingClientRect().top, 64);
});

test('the number disclosure remeasures the figure and is hidden for the algorithm recap', () => {
  const ui = setup({ reducedMotion: true });
  ui.clickLink('#identical-inputs');
  assert.equal(ui.get('numbers').hidden, false);
  assert.match(ui.get('restricted-diagram').innerHTML, /viewBox="0 0 790 450"/);
  ui.get('visual-area').getBoundingClientRect = () => ({ width: 790, height: 540 });
  ui.get('numbers').emit('toggle');
  assert.match(ui.get('restricted-diagram').innerHTML, /viewBox="0 0 790 540"/);
  assert.equal(Number(ui.shell.dataset.scene), 0);
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
  assert.ok(ui.scrollY() > stageTop + 660, 'The desktop chapter is deep enough to move the static stage off screen');
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

const geometryAttribute = (ui, name) => new RegExp(`data-${name}="([^"]+)"`).exec(ui.get('prediction-diagram').innerHTML)?.[1];
const geometryLoss = ui => Number(geometryAttribute(ui, 'loss'));

test('geometry controls move the original three-vector path and expose the residual norm', () => {
  const ui = setup({ reducedMotion: true });
  ui.scrollAt(geometryPhysical(0));
  assert.equal(ui.get('geometry-controls').hidden, false);
  assert.equal(ui.get('geometry-path-controls').hidden, false);
  assert.equal(ui.get('geometry-round-label').textContent, 'Trees added');
  assert.equal(ui.get('geometry-tree-controls').hidden, true);
  assert.equal(ui.get('geometry-update-controls').hidden, true);
  assert.equal(ui.get('worked-controls').hidden, true);
  close(Number(geometryAttribute(ui, 'norm')), Math.sqrt(3));
  close(geometryLoss(ui), 1.5);
  ui.control('geometry-round', 1);
  close(Number(geometryAttribute(ui, 'norm')), 1.5);
  close(geometryLoss(ui), 1.125);
  assert.equal(ui.get('geometry-round-value').textContent, '1');
  ui.control('geometry-rate', 1, 'change');
  close(geometryLoss(ui), 1);
  ui.control('geometry-rate', .5, 'change');
  ui.control('geometry-round', 0);
  ui.click('next');
  assert.equal(Number(ui.shell.dataset.scene), geometryPhysical(1));
  assert.equal(ui.get('geometry-path-controls').hidden, false);
  assert.equal(ui.get('geometry-round-label').textContent, 'Starting round');
  assert.equal(ui.get('geometry-tree-controls').hidden, false);
  assert.equal(ui.get('geometry-update-controls').hidden, false);
  close(geometryLoss(ui), 1.125);
  ui.control('geometry-progress', .5);
  close(geometryLoss(ui), 1.28125);
  assert.equal(ui.get('geometry-progress-value').textContent, '50%');
  ui.click('next');
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(0));
  assert.equal(ui.shell.dataset.example, 'worked');
  assert.equal(ui.get('geometry-controls').hidden, true);
  assert.equal(ui.get('worked-controls').hidden, false);
});

test('the vector tree controls disappear when leaving vector correction for other examples', () => {
  const ui = setup({ reducedMotion: true });
  assert.equal(ui.get('geometry-tree-controls').hidden, true);
  ui.clickLink('#vector-correction');
  assert.equal(Number(ui.shell.dataset.scene), geometryPhysical(1));
  assert.equal(ui.get('geometry-tree-controls').hidden, false);
  ui.click('next');
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(0));
  assert.equal(ui.shell.dataset.example, 'worked');
  assert.equal(ui.get('geometry-tree-controls').hidden, true);
  ui.click('previous');
  assert.equal(Number(ui.shell.dataset.scene), geometryPhysical(1));
  assert.equal(ui.get('geometry-tree-controls').hidden, false);
  ui.clickLink('#identical-inputs');
  assert.equal(Number(ui.shell.dataset.scene), 0);
  assert.equal(ui.shell.dataset.example, 'restricted');
  assert.equal(ui.get('geometry-tree-controls').hidden, true);
  ui.clickLink('#vector-geometry');
  assert.equal(Number(ui.shell.dataset.scene), geometryPhysical(0));
  assert.equal(ui.get('geometry-tree-controls').hidden, true);
});

test('successive vector corrections refit the tree and gradient, while earlier updates remain selectable', () => {
  const ui = setup({ reducedMotion: true });
  ui.scrollAt(geometryPhysical(1));
  const vectorAttribute = name => JSON.parse(geometryAttribute(ui, name));
  assert.equal(ui.get('geometry-previous-tree').disabled, true);
  assert.equal(ui.get('geometry-next-tree').disabled, false);
  assert.deepEqual(vectorAttribute('tree'), [1, 0, 0]);
  assert.deepEqual(vectorAttribute('prediction'), [.5, 0, 0]);
  close(geometryLoss(ui), 1.125);

  // Refitting after F1 changes both the split and leaf values; repeating h1
  // would instead put the next point at (1, 0, 0).
  ui.click('geometry-next-tree');
  assert.equal(ui.get('geometry-round').value, '1');
  assert.equal(ui.get('geometry-previous-tree').disabled, false);
  assert.deepEqual(vectorAttribute('tree'), [-.25, -.25, 1]);
  assert.deepEqual(vectorAttribute('prediction'), [.375, -.125, .5]);
  assert.deepEqual(vectorAttribute('gradient'), [-.625, .875, -.5]);
  close(geometryLoss(ui), .703125);
  close(Number(geometryAttribute(ui, 'before-loss')), 1.125);

  ui.control('geometry-progress', .5);
  assert.deepEqual(vectorAttribute('prediction'), [.4375, -.0625, .25]);
  ui.click('geometry-previous-tree');
  assert.equal(ui.get('geometry-progress').value, '1');
  assert.deepEqual(vectorAttribute('prediction'), [.5, 0, 0]);
  close(geometryLoss(ui), 1.125);
  assert.equal(ui.get('geometry-previous-tree').disabled, true);

  // Jumping to another start commits a complete displayed correction. The
  // scrubber still provides a partial update within that selected round.
  ui.control('geometry-progress', .25);
  ui.control('geometry-round', 2);
  assert.equal(ui.get('geometry-progress').value, '1');
  assert.deepEqual(vectorAttribute('tree'), [.625, -.1875, -.1875]);
  assert.deepEqual(vectorAttribute('prediction'), [.6875, -.21875, .40625]);
  close(geometryLoss(ui), .5302734375);
  ui.control('geometry-progress', 0);
  assert.deepEqual(vectorAttribute('prediction'), [.375, -.125, .5]);
  close(geometryLoss(ui), .703125);
  ui.control('geometry-progress', 1);

  for (const button of ['geometry-previous-tree', 'geometry-next-tree']) {
    assert.equal(ui.get(button).tagName, 'BUTTON');
    const focusedKey = ui.key('ArrowRight', ui.get(button));
    assert.notEqual(focusedKey.defaultPrevented, true);
    assert.equal(Number(ui.shell.dataset.scene), geometryPhysical(1));
    assert.equal(ui.get('geometry-round').value, '2');
  }
  ui.control('geometry-round', 24);
  assert.equal(ui.get('geometry-next-tree').disabled, true);
  assert.equal(ui.get('geometry-previous-tree').disabled, false);
  assert.ok(geometryLoss(ui) < .001);
  ui.click('geometry-previous-tree');
  assert.equal(ui.get('geometry-round').value, '23');
  assert.equal(ui.get('geometry-next-tree').disabled, false);
  ui.click('geometry-next-tree');
  assert.equal(ui.get('geometry-round').value, '24');
  assert.equal(ui.get('geometry-next-tree').disabled, true);
  ui.control('geometry-round', 0);
  assert.equal(ui.get('geometry-previous-tree').disabled, true);
  assert.deepEqual(vectorAttribute('prediction'), [.5, 0, 0]);
});

test('legacy radius links skip directly to the identical-input example', () => {
  const ui = setup({ reducedMotion: true });
  assert.equal(ui.clickLink('#vector-radius').defaultPrevented, true);
  assert.equal(Number(ui.shell.dataset.scene), 0);
  assert.equal(ui.shell.dataset.example, 'restricted');
  assert.equal(ui.get('geometry-controls').hidden, true);
  assert.equal(ui.get('restricted-controls').hidden, false);
  restrictedVector(ui, 'target', [1, 0]);
  restrictedVector(ui, 'prediction', [0, 0]);
  ui.clickLink('#vector-geometry');
  assert.equal(Number(ui.shell.dataset.scene), geometryPhysical(0));
  assert.equal(ui.shell.dataset.example, 'geometry');
});

test('crossing geometry, identical inputs and the four-point calculation preserves independent controls', () => {
  const ui = setup({ reducedMotion: true });
  ui.scrollAt(geometryPhysical(1));
  ui.control('geometry-rate', 1, 'change');
  ui.control('geometry-progress', .75);
  const before = ui.get('prediction-diagram').innerHTML;
  ui.clickLink('#loss-playground');
  assert.equal(Number(ui.shell.dataset.scene), sandboxPhysical);
  assert.equal(ui.shell.dataset.example, 'restricted');
  assert.equal(ui.get('prediction-diagram').hidden, true);
  assert.equal(ui.get('restricted-diagram').hidden, false);
  assert.equal(ui.get('worked-diagram').hidden, true);
  assert.equal(ui.get('geometry-controls').hidden, true);
  assert.equal(ui.get('restricted-controls').hidden, false);
  assert.equal(ui.get('worked-controls').hidden, true);
  ui.control('restricted-prediction', .25);
  const restrictedBefore = ui.get('restricted-diagram').innerHTML;
  close(restrictedLoss(ui), .3125);
  ui.clickLink('#worked-data');
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(0));
  assert.equal(ui.shell.dataset.example, 'worked');
  assert.equal(ui.get('restricted-diagram').hidden, true);
  assert.equal(ui.get('worked-diagram').hidden, false);
  assert.equal(ui.get('restricted-controls').hidden, true);
  assert.equal(ui.get('worked-controls').hidden, false);
  assert.equal(ui.get('worked-rate').value, '1');
  close(figureLoss(ui), 7);
  ui.control('worked-rate', .5, 'change');
  ui.clickLink('#loss-playground');
  assert.equal(Number(ui.shell.dataset.scene), sandboxPhysical);
  assert.equal(ui.get('restricted-diagram').innerHTML, restrictedBefore);
  assert.equal(ui.get('restricted-prediction').value, '0.25');
  ui.clickLink('#vector-correction');
  assert.equal(Number(ui.shell.dataset.scene), geometryPhysical(1));
  assert.equal(ui.shell.dataset.example, 'geometry');
  assert.equal(ui.get('prediction-diagram').hidden, false);
  assert.equal(ui.get('worked-diagram').hidden, true);
  assert.equal(ui.get('restricted-diagram').hidden, true);
  assert.equal(ui.get('prediction-diagram').innerHTML, before);
  assert.equal(ui.get('geometry-rate').value, '1');
  assert.equal(ui.get('geometry-progress').value, '0.75');
  ui.click('next');
  assert.equal(ui.get('worked-rate').value, '0.5');
});

test('the identical-input control uses native focused keys and fits its only tree to the projection', () => {
  const ui = setup({ reducedMotion: true });
  ui.scrollAt(sandboxPhysical);
  ui.control('restricted-loss', 'squared', 'change');
  ui.click('restricted-reset');
  assert.equal(ui.shell.dataset.example, 'restricted');
  assert.equal(Number(ui.shell.dataset.restrictedScene), sandboxPhysical);
  assert.equal(ui.get('restricted-prediction').value, '0');
  assert.deepEqual(JSON.parse(restrictedAttribute(ui, 'prediction')), [0, 0]);
  assert.deepEqual(JSON.parse(restrictedAttribute(ui, 'next-tree')), [.5, .5]);
  close(restrictedLoss(ui), .5);
  // The controller must leave Home, End and arrow handling to this native slider.
  for (const key of ['Home', 'End', 'ArrowRight', 'ArrowLeft']) {
    const event = ui.key(key, ui.get('restricted-prediction'));
    assert.equal(event.defaultPrevented, undefined);
    assert.equal(Number(ui.shell.dataset.scene), sandboxPhysical);
  }
  ui.control('restricted-prediction', 1);
  assert.deepEqual(JSON.parse(restrictedAttribute(ui, 'prediction')), [1, 1]);
  assert.deepEqual(JSON.parse(restrictedAttribute(ui, 'next-tree')), [-.5, -.5]);
  close(restrictedLoss(ui), .5);
  ui.click('restricted-fit');
  assert.equal(ui.get('restricted-prediction').value, '0.5');
  assert.deepEqual(JSON.parse(restrictedAttribute(ui, 'prediction')), [.5, .5]);
  assert.deepEqual(JSON.parse(restrictedAttribute(ui, 'residual')), [.5, -.5]);
  assert.deepEqual(JSON.parse(restrictedAttribute(ui, 'next-tree')), [0, 0]);
  close(restrictedLoss(ui), .25);
  assert.match(ui.get('stage-caption').textContent, /zero|no further|cannot reduce/i);
  ui.click('restricted-fit');
  close(restrictedLoss(ui), .25);
  ui.control('restricted-prediction', 0);
  close(restrictedLoss(ui), .5);
});

for (const compact of [false, true]) {
  test(`the two-observation controls expose asymmetric gradients at the residual mean (${compact ? 'narrow' : 'wide'})`, () => {
    const ui = setup({ reducedMotion: true, compact });
    ui.clickLink('#loss-playground');
    assert.equal(ui.get('restricted-loss').value, 'squared');
    ui.control('restricted-loss', 'expectile', 'change');
    ui.click('restricted-reset');
    assert.equal(ui.get('restricted-loss').value, 'expectile');
    assert.equal(ui.get('restricted-tau').value, '0.8');
    assert.equal(ui.get('restricted-tau-controls').hidden, false);
    assert.equal(ui.get('restricted-delta-label').hidden, true);
    restrictedVector(ui, 'target', [1, 0]);
    restrictedVector(ui, 'prediction', [0, 0]);
    restrictedVector(ui, 'next-tree', [.8, .8]);
    assert.match(ui.get('restricted-weights').textContent, /Underpredict ×0\.8.*Overpredict ×0\.2/);

    ui.click('restricted-mean');
    restrictedVector(ui, 'prediction', [.5, .5]);
    restrictedVector(ui, 'residual', [.5, -.5]);
    restrictedVector(ui, 'negative-gradient', [.8, -.2]);
    restrictedVector(ui, 'next-tree', [.3, .3]);
    close(restrictedLoss(ui), .25);
    assert.match(numbers(ui), /Negative gradient.*0\.8, −0\.2/);
    ui.click('restricted-fit');
    assert.equal(ui.get('restricted-prediction').value, '0.8');
    restrictedVector(ui, 'negative-gradient', [.32, -.32]);
    restrictedVector(ui, 'next-tree', [0, 0]);
    close(restrictedLoss(ui), .16);

    // The native range retains its own keyboard behavior and updates on input.
    for (const key of ['Home', 'End', 'ArrowRight', 'ArrowLeft']) {
      assert.equal(ui.key(key, ui.get('restricted-tau')).defaultPrevented, undefined);
      assert.equal(Number(ui.shell.dataset.scene), sandboxPhysical);
    }
    ui.control('restricted-tau', .2);
    assert.equal(ui.get('restricted-tau-value').textContent, '0.2');
    assert.match(ui.get('restricted-weights').textContent, /Underpredict ×0\.2.*Overpredict ×0\.8/);
    restrictedVector(ui, 'next-tree', [-.6, -.6]);
    ui.click('restricted-fit');
    assert.equal(ui.get('restricted-prediction').value, '0.2');
    close(restrictedLoss(ui), .16);
    ui.click('restricted-reset');
    restrictedVector(ui, 'prediction', [0, 0]);
    assert.equal(ui.get('restricted-tau').value, '0.2');
    close(restrictedLoss(ui), .2);
  });
}

test('switching to Huber and fourth-power loss calibrates the shared leaf correction', () => {
  const ui = setup({ reducedMotion: true });
  ui.scrollAt(sandboxPhysical);
  ui.control('restricted-loss', 'huber', 'change');
  assert.equal(ui.get('restricted-tau-controls').hidden, true);
  assert.equal(ui.get('restricted-delta-label').hidden, false);
  assert.equal(ui.get('restricted-delta').value, '0.25');
  ui.click('restricted-mean');
  restrictedVector(ui, 'optimum-interval', [.25, .75]);
  restrictedVector(ui, 'next-tree', [0, 0]);
  restrictedVector(ui, 'correction', [0, 0]);
  const minimum = restrictedLoss(ui);
  ui.click('restricted-fit');
  assert.equal(ui.get('restricted-prediction').value, '0.5');
  close(restrictedLoss(ui), minimum);
  ui.click('restricted-reset');
  restrictedVector(ui, 'next-tree', [.125, .125]);
  restrictedVector(ui, 'correction', [.25, .25]);
  ui.click('restricted-fit');
  assert.equal(ui.get('restricted-prediction').value, '0.25');
  close(restrictedLoss(ui), minimum);

  ui.control('restricted-delta', .1, 'change');
  restrictedVector(ui, 'optimum-interval', [.1, .9]);
  ui.click('restricted-reset');
  ui.click('restricted-fit');
  assert.equal(ui.get('restricted-prediction').value, '0.1');

  ui.control('restricted-loss', 'quartic', 'change');
  assert.equal(ui.get('restricted-tau-controls').hidden, true);
  assert.equal(ui.get('restricted-delta-label').hidden, true);
  ui.control('restricted-prediction', .25);
  restrictedVector(ui, 'negative-gradient', [.421875, -.015625]);
  restrictedVector(ui, 'next-tree', [.203125, .203125]);
  restrictedVector(ui, 'correction', [.25, .25]);
  ui.click('restricted-fit');
  assert.equal(ui.get('restricted-prediction').value, '0.5');
  close(restrictedLoss(ui), 1 / 32);
  ui.control('restricted-loss', 'squared', 'change');
  close(restrictedLoss(ui), .25);
});

test('two-observation loss parameters persist independently of the later loss comparison', () => {
  const ui = setup({ reducedMotion: true });
  ui.clickLink('#loss-playground');
  ui.control('restricted-loss', 'expectile', 'change');
  ui.control('restricted-tau', .65);
  ui.control('restricted-delta', .1, 'change');
  ui.control('restricted-prediction', .3);
  const before = ui.get('restricted-diagram').innerHTML;
  ui.clickLink('#loss-targets');
  assert.equal(ui.get('restricted-controls').hidden, true);
  assert.equal(ui.get('loss-choice').value, 'huber');
  assert.equal(ui.get('loss-delta').value, '1');
  ui.control('loss-choice', 'expectile', 'change');
  ui.control('loss-tau', .2, 'change');
  ui.clickLink('#loss-playground');
  assert.equal(ui.get('restricted-diagram').innerHTML, before);
  assert.equal(ui.get('restricted-loss').value, 'expectile');
  assert.equal(ui.get('restricted-tau').value, '0.65');
  assert.equal(ui.get('restricted-delta').value, '0.1');
  assert.equal(ui.get('restricted-prediction').value, '0.3');
});

test('phone navigation goes from the opening sandbox through vectors into the four-point example', () => {
  const ui = setup({ compact: true, variableStageHeight: true, smoothScroll: true });
  ui.scrollAt(sandboxPhysical);
  for (const destination of [geometryPhysical(0), geometryPhysical(1), workedPhysical(0)]) {
    ui.click('next'); ui.scrollFraction(.2);
    assert.equal(Number(ui.shell.dataset.scene), destination);
    ui.scrollFraction(1); ui.resize();
    assert.equal(Number(ui.shell.dataset.scene), destination);
    close(ui.shell.getBoundingClientRect().top, 64);
  }
  for (const destination of [geometryPhysical(1), geometryPhysical(0), sandboxPhysical]) {
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

test('returning to geometry during an animated worked update cancels the stale correction', () => {
  const ui = setup();
  ui.scrollAt(workedPhysical(3));
  assert.ok(ui.frames.size > 0);
  ui.scrollAt(geometryPhysical(1));
  const restored = ui.get('prediction-diagram').innerHTML;
  ui.advance(2000);
  assert.equal(ui.get('prediction-diagram').innerHTML, restored);
  assert.equal(Number(ui.shell.dataset.scene), geometryPhysical(1));
  close(geometryLoss(ui), 1.125);
  assert.equal(ui.get('worked-diagram').hidden, true);
  assert.equal(ui.get('geometry-controls').hidden, false);
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
