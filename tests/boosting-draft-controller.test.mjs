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
const workedPhysical = local => local + 4;
const headingTop = physical => 1000 + physical * 1000;

// This harness checks real controller events and numerical output. It does not
// emulate layout, parse generated HTML, or claim to replace browser visual QA.
function setup({ reducedMotion = false, compact = false, variableStageHeight = false, smoothScroll = false, hash = '' } = {}) {
  let time = 0, nextId = 1;
  let pendingScroll = null;
  const visitedHashes = [];
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
  const stageHeight = () => !compact ? 736 : !variableStageHeight ? 450
    : shell.dataset.example === 'restricted' ? 660 : Number(shell.dataset.geometryScene) === 2
      || (shell.dataset.example === 'worked' && [3, 5, 6].includes(Number(shell.dataset.workedScene))) ? 528 : 392;
  shell.getBoundingClientRect = () => ({ top: compact ? 56 : 64, bottom: (compact ? 56 : 64) + stageHeight(), height: stageHeight(), width: compact ? 288 : 640 });
  get('visual-area').getBoundingClientRect = () => ({ width: compact ? 288 : 640, height: compact ? 260 : 450 });
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
    element.getBoundingClientRect = () => ({ top: 1000 + Number(match[1]) * 1000 - win.scrollY, height: 800 });
    return element;
  });
  const layout = new Element();
  layout.getBoundingClientRect = () => ({ top: 1000 - win.scrollY, bottom: (1000 + steps.length * 1000) - win.scrollY });
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
    const threshold = compact ? shell.getBoundingClientRect().bottom + 65 : win.innerHeight * .52;
    win.scrollY = 1000 + index * 1000 - threshold + 1;
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
  return { get, shell, click, clickLink, control, scrollAt, scrollFraction, advance, key, resize, reduced, frames,
    historyBack: hash => { win.location.hash = hash; win.emit('popstate'); advance(0); },
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

for (const compact of [false, true]) {
  test(`prediction vectors and the identical-input restriction lead into the eight-scene gradient-boosting calculation (${compact ? 'narrow' : 'wide'} navigation)`, () => {
    const ui = setup({ reducedMotion: true, compact });
    assert.equal(Number(ui.shell.dataset.scene), 0);
    assert.equal(ui.shell.dataset.example, 'geometry');
    assert.match(numbers(ui), /\(1, [−-]1, 1\)/);
    assert.equal(ui.get('previous').disabled, true);
    for (let scene = 1; scene <= 15; scene++) {
      ui.click('next');
      assert.equal(Number(ui.shell.dataset.scene), scene);
      assert.equal(ui.get('stage-count').textContent, `${String(scene + 1).padStart(2, '0')} / 16`);
      if (scene < 3) {
        assert.equal(ui.shell.dataset.example, 'geometry');
        assert.match(ui.get('prediction-diagram').innerHTML, new RegExp(`data-geometry-scene="${scene}"`));
      } else if (scene === 3) {
        assert.equal(ui.shell.dataset.example, 'restricted');
        assert.match(ui.get('restricted-diagram').innerHTML, /data-restricted-scene="0"/);
        assert.equal(ui.get('restricted-controls').hidden, false);
        assert.equal(ui.get('restricted-loss').value, 'expectile');
        close(restrictedLoss(ui), .8);
      } else if (scene >= 12) {
        assert.equal(ui.shell.dataset.example, 'loss');
        assert.equal(Number(ui.shell.dataset.lossScene), scene - 12);
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
    assert.equal(ui.shell.dataset.example, 'geometry');
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

test('step links show their heading below the phone stage and retain their target through smooth scrolling', () => {
  const ui = setup({ compact: true, variableStageHeight: true, smoothScroll: true });
  ui.scrollAt(workedPhysical(0));
  assert.equal(ui.clickLink('#worked-first-update').defaultPrevented, true);
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(3));
  ui.scrollFraction(.2);
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(3));
  ui.scrollFraction(1);
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(3));
  assert.ok(headingTop(workedPhysical(3)) - ui.scrollY() > ui.shell.getBoundingClientRect().bottom, 'The heading is hidden by the pinned figure');
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

test('opening a worked-step hash uses the phone offset, while other-losses keeps native navigation', () => {
  const ui = setup({ reducedMotion: true, compact: true, variableStageHeight: true, hash: '#worked-first-update' });
  ui.advance(0); ui.advance(0);
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(3));
  assert.ok(headingTop(workedPhysical(3)) - ui.scrollY() > ui.shell.getBoundingClientRect().bottom);
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
  assert.ok(headingTop(workedPhysical(3)) - ui.scrollY() > ui.shell.getBoundingClientRect().bottom);
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

const geometryAttribute = (ui, name) => new RegExp(`data-${name}="([^"]+)"`).exec(ui.get('prediction-diagram').innerHTML)?.[1];
const geometryLoss = ui => Number(geometryAttribute(ui, 'loss'));

test('geometry controls move the original three-vector path and expose the residual norm', () => {
  const ui = setup({ reducedMotion: true });
  assert.equal(ui.get('geometry-controls').hidden, false);
  assert.equal(ui.get('geometry-path-controls').hidden, false);
  assert.equal(ui.get('geometry-update-controls').hidden, true);
  assert.equal(ui.get('geometry-angle-controls').hidden, true);
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
  assert.equal(Number(ui.shell.dataset.scene), 1);
  assert.equal(ui.get('geometry-path-controls').hidden, true);
  assert.equal(ui.get('geometry-update-controls').hidden, false);
  close(geometryLoss(ui), 1.125);
  ui.control('geometry-progress', .5);
  close(geometryLoss(ui), 1.28125);
  assert.equal(ui.get('geometry-progress-value').textContent, '50%');
  ui.click('next');
  assert.equal(Number(ui.shell.dataset.scene), 2);
  assert.equal(ui.get('geometry-update-controls').hidden, true);
  assert.equal(ui.get('geometry-angle-controls').hidden, false);
  assert.equal(ui.get('geometry-rate-label').hidden, true);
  close(Number(geometryAttribute(ui, 'radius')), Math.sqrt(3));
});

test('the exact-radius experiment reports flipped directions, no orthogonal descent and overshoot', () => {
  const ui = setup({ reducedMotion: true });
  ui.scrollAt(2);
  ui.control('geometry-angle', 120);
  ui.control('geometry-length', .5);
  assert.equal(geometryAttribute(ui, 'orientation'), '-1');
  close(geometryLoss(ui), 1.125);
  ui.control('geometry-angle', 90);
  close(geometryLoss(ui), 1.875);
  ui.control('geometry-angle', 0);
  ui.control('geometry-length', 2.5);
  close(geometryLoss(ui), 3.375);
  ui.control('geometry-start', 1, 'change');
  close(Number(geometryAttribute(ui, 'radius')), 1.5);
  close(Number(geometryAttribute(ui, 'before-loss')), 1.125);
  close(geometryLoss(ui), 2.53125);
  const focused = ui.key('ArrowRight', ui.get('geometry-angle'));
  assert.equal(focused.defaultPrevented, undefined);
  assert.equal(Number(ui.shell.dataset.scene), 2);
});

test('crossing geometry, identical inputs and the four-point calculation preserves independent controls', () => {
  const ui = setup({ reducedMotion: true });
  ui.scrollAt(2);
  ui.control('geometry-angle', 30);
  ui.control('geometry-length', .75);
  ui.control('geometry-start', 1, 'change');
  const before = ui.get('prediction-diagram').innerHTML;
  ui.click('next');
  assert.equal(Number(ui.shell.dataset.scene), 3);
  assert.equal(ui.shell.dataset.example, 'restricted');
  assert.equal(ui.get('prediction-diagram').hidden, true);
  assert.equal(ui.get('restricted-diagram').hidden, false);
  assert.equal(ui.get('worked-diagram').hidden, true);
  assert.equal(ui.get('geometry-controls').hidden, true);
  assert.equal(ui.get('restricted-controls').hidden, false);
  assert.equal(ui.get('worked-controls').hidden, true);
  ui.control('restricted-prediction', .25);
  const restrictedBefore = ui.get('restricted-diagram').innerHTML;
  close(restrictedLoss(ui), .4625);
  ui.click('next');
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(0));
  assert.equal(ui.shell.dataset.example, 'worked');
  assert.equal(ui.get('restricted-diagram').hidden, true);
  assert.equal(ui.get('worked-diagram').hidden, false);
  assert.equal(ui.get('restricted-controls').hidden, true);
  assert.equal(ui.get('worked-controls').hidden, false);
  assert.equal(ui.get('worked-rate').value, '1');
  close(figureLoss(ui), 7);
  ui.control('worked-rate', .5, 'change');
  ui.click('previous');
  assert.equal(Number(ui.shell.dataset.scene), 3);
  assert.equal(ui.get('restricted-diagram').innerHTML, restrictedBefore);
  assert.equal(ui.get('restricted-prediction').value, '0.25');
  ui.click('previous');
  assert.equal(Number(ui.shell.dataset.scene), 2);
  assert.equal(ui.shell.dataset.example, 'geometry');
  assert.equal(ui.get('prediction-diagram').hidden, false);
  assert.equal(ui.get('worked-diagram').hidden, true);
  assert.equal(ui.get('restricted-diagram').hidden, true);
  assert.equal(ui.get('prediction-diagram').innerHTML, before);
  assert.equal(ui.get('geometry-angle').value, '30');
  assert.equal(ui.get('geometry-length').value, '0.75');
  assert.equal(ui.get('geometry-start').value, '1');
  ui.click('next'); ui.click('next');
  assert.equal(ui.get('worked-rate').value, '0.5');
});

test('the identical-input control uses native focused keys and fits its only tree to the projection', () => {
  const ui = setup({ reducedMotion: true });
  ui.scrollAt(3);
  ui.control('restricted-loss', 'squared', 'change');
  assert.equal(ui.shell.dataset.example, 'restricted');
  assert.equal(Number(ui.shell.dataset.restrictedScene), 0);
  assert.equal(ui.get('restricted-prediction').value, '0');
  assert.deepEqual(JSON.parse(restrictedAttribute(ui, 'prediction')), [0, 0]);
  assert.deepEqual(JSON.parse(restrictedAttribute(ui, 'next-tree')), [.5, .5]);
  close(restrictedLoss(ui), .5);
  // The controller must leave Home, End and arrow handling to this native slider.
  for (const key of ['Home', 'End', 'ArrowRight', 'ArrowLeft']) {
    const event = ui.key(key, ui.get('restricted-prediction'));
    assert.equal(event.defaultPrevented, undefined);
    assert.equal(Number(ui.shell.dataset.scene), 3);
  }
  ui.control('restricted-prediction', 1);
  assert.deepEqual(JSON.parse(restrictedAttribute(ui, 'prediction')), [1, 1]);
  assert.deepEqual(JSON.parse(restrictedAttribute(ui, 'next-tree')), [-.5, -.5]);
  close(restrictedLoss(ui), .5);
  ui.click('restricted-fit');
  assert.equal(ui.get('restricted-prediction').value, '0.5');
  assert.deepEqual(JSON.parse(restrictedAttribute(ui, 'prediction')), [.5, .5]);
  assert.deepEqual(JSON.parse(restrictedAttribute(ui, 'residual')), [-.5, .5]);
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
    ui.clickLink('#identical-inputs');
    assert.equal(ui.get('restricted-loss').value, 'expectile');
    assert.equal(ui.get('restricted-tau').value, '0.8');
    assert.equal(ui.get('restricted-tau-controls').hidden, false);
    assert.equal(ui.get('restricted-delta-label').hidden, true);
    restrictedVector(ui, 'target', [0, 1]);
    restrictedVector(ui, 'prediction', [0, 0]);
    restrictedVector(ui, 'next-tree', [.8, .8]);
    assert.match(ui.get('restricted-weights').textContent, /Underpredict ×0\.8.*Overpredict ×0\.2/);

    ui.click('restricted-mean');
    restrictedVector(ui, 'prediction', [.5, .5]);
    restrictedVector(ui, 'residual', [-.5, .5]);
    restrictedVector(ui, 'negative-gradient', [-.2, .8]);
    restrictedVector(ui, 'next-tree', [.3, .3]);
    close(restrictedLoss(ui), .25);
    assert.match(numbers(ui), /Negative gradient.*−0\.2, 0\.8/);
    ui.click('restricted-fit');
    assert.equal(ui.get('restricted-prediction').value, '0.8');
    restrictedVector(ui, 'negative-gradient', [-.32, .32]);
    restrictedVector(ui, 'next-tree', [0, 0]);
    close(restrictedLoss(ui), .16);

    // The native range retains its own keyboard behavior and updates on input.
    for (const key of ['Home', 'End', 'ArrowRight', 'ArrowLeft']) {
      assert.equal(ui.key(key, ui.get('restricted-tau')).defaultPrevented, undefined);
      assert.equal(Number(ui.shell.dataset.scene), 3);
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
  ui.scrollAt(3);
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
  restrictedVector(ui, 'negative-gradient', [-.015625, .421875]);
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
  ui.clickLink('#identical-inputs');
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
  ui.clickLink('#identical-inputs');
  assert.equal(ui.get('restricted-diagram').innerHTML, before);
  assert.equal(ui.get('restricted-loss').value, 'expectile');
  assert.equal(ui.get('restricted-tau').value, '0.65');
  assert.equal(ui.get('restricted-delta').value, '0.1');
  assert.equal(ui.get('restricted-prediction').value, '0.3');
});

test('phone navigation keeps both inserted-scene boundaries below the changed stage height', () => {
  const ui = setup({ compact: true, variableStageHeight: true, smoothScroll: true });
  ui.scrollAt(1);
  for (const destination of [2, 3, workedPhysical(0)]) {
    ui.click('next'); ui.scrollFraction(.2);
    assert.equal(Number(ui.shell.dataset.scene), destination);
    ui.scrollFraction(1); ui.resize();
    assert.equal(Number(ui.shell.dataset.scene), destination);
    assert.ok(headingTop(destination) - ui.scrollY() > ui.shell.getBoundingClientRect().bottom);
  }
  for (const destination of [3, 2]) {
    ui.click('previous'); ui.scrollFraction(.5);
    assert.equal(Number(ui.shell.dataset.scene), destination);
    ui.scrollFraction(1); ui.resize();
    assert.equal(Number(ui.shell.dataset.scene), destination);
    assert.ok(headingTop(destination) - ui.scrollY() > ui.shell.getBoundingClientRect().bottom);
  }
});

test('the identical-input hash opens the new scene before the worked-data hash', () => {
  const ui = setup({ reducedMotion: true, compact: true, variableStageHeight: true, hash: '#identical-inputs' });
  ui.advance(0); ui.advance(0);
  assert.equal(Number(ui.shell.dataset.scene), 3);
  assert.equal(ui.shell.dataset.example, 'restricted');
  assert.ok(headingTop(3) - ui.scrollY() > ui.shell.getBoundingClientRect().bottom);
  ui.clickLink('#worked-data');
  assert.equal(Number(ui.shell.dataset.scene), workedPhysical(0));
  assert.equal(ui.shell.dataset.example, 'worked');
  ui.historyBack('#identical-inputs');
  assert.equal(Number(ui.shell.dataset.scene), 3);
  assert.equal(ui.shell.dataset.example, 'restricted');
  assert.deepEqual(ui.visitedHashes, ['#worked-data']);
});

test('returning to geometry during an animated worked update cancels the stale correction', () => {
  const ui = setup();
  ui.scrollAt(workedPhysical(3));
  assert.ok(ui.frames.size > 0);
  ui.scrollAt(1);
  const restored = ui.get('prediction-diagram').innerHTML;
  ui.advance(2000);
  assert.equal(ui.get('prediction-diagram').innerHTML, restored);
  assert.equal(Number(ui.shell.dataset.scene), 1);
  close(geometryLoss(ui), 1.125);
  assert.equal(ui.get('worked-diagram').hidden, true);
  assert.equal(ui.get('geometry-controls').hidden, false);
});


test('loss controls refit the compared paths and persist across narrative scenes', () => {
  const ui = setup({ reducedMotion: true });
  ui.clickLink('#loss-targets');
  assert.equal(Number(ui.shell.dataset.scene), 12);
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
