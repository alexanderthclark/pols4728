import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { runInNewContext } from 'node:vm';

const controller = readFileSync(new URL('../docs/assets/coronation.js', import.meta.url), 'utf8');

// Exercise the real controller's clock and lifecycle. Browser checks cover
// proportions, face alignment, and the rendered crown separately.
function setup({ reducedMotion = false, imageLoaded = true } = {}) {
  let time = 0;
  let nextId = 1;
  let intersection;
  const timers = new Map();
  const frames = new Map();

  class Element {
    constructor() {
      this.listeners = new Map();
      this.attrs = new Map();
      this.dataset = {};
      this.children = [];
      this.isConnected = true;
    }
    addEventListener(type, callback) {
      this.listeners.set(type, [...(this.listeners.get(type) || []), callback]);
    }
    removeEventListener(type, callback) {
      this.listeners.set(type, (this.listeners.get(type) || []).filter(item => item !== callback));
    }
    emit(type) {
      for (const callback of [...(this.listeners.get(type) || [])]) callback({ type, target: this });
    }
    setAttribute(key, value) { this.attrs.set(key, String(value)); }
    append(...children) { this.children.push(...children); }
  }

  const root = new Element();
  const image = new Element();
  image.complete = imageLoaded;
  image.naturalWidth = imageLoaded ? 1692 : 0;
  const svg = new Element();
  const parts = new Map([
    '.animated-eyes', '.eye-clips', '.descending-crown',
    '.crown-contact-shadow', '.crown-rays-mask', '.pleased-smile'
  ].map(selector => [selector, new Element()]));
  root.querySelector = selector => selector === 'img.course-logo' ? image : selector === '.character-layer' ? svg : null;
  svg.querySelector = selector => parts.get(selector);

  const document = new Element();
  document.hidden = false;
  document.getElementById = id => id === 'course-coronation' ? root : null;
  document.createElementNS = () => new Element();
  const media = new Element();
  media.matches = reducedMotion;
  const window = new Element();
  const requestAnimationFrame = callback => {
    const id = nextId++;
    frames.set(id, callback);
    return id;
  };
  const cancelAnimationFrame = id => frames.delete(id);
  const setTimeout = (callback, delay) => {
    const id = nextId++;
    timers.set(id, { at: time + delay, callback });
    return id;
  };
  const clearTimeout = id => timers.delete(id);
  const IntersectionObserver = class {
    constructor(callback) { intersection = callback; }
    observe() {}
    disconnect() { intersection = null; }
  };
  Object.assign(window, { IntersectionObserver, innerWidth: 1200, innerHeight: 800 });
  runInNewContext(controller, {
    window, document, IntersectionObserver,
    performance: { now: () => time },
    matchMedia: () => media,
    requestAnimationFrame, cancelAnimationFrame, setTimeout, clearTimeout
  });

  const snapshot = element => ({
    attrs: Object.fromEntries(element.attrs),
    children: element.children.map(snapshot)
  });
  return {
    state: () => svg.dataset.coronationState,
    visual: () => [...parts].map(([selector, element]) => [selector, snapshot(element)]),
    pendingTimers: () => timers.size,
    pendingFrames: () => frames.size,
    activity: type => document.emit(type),
    intersect(visible) {
      intersection?.([{ isIntersecting: visible, intersectionRatio: visible ? 1 : 0 }]);
    },
    hide(hidden) { document.hidden = hidden; document.emit('visibilitychange'); },
    reduce(matches) { media.matches = matches; media.emit('change'); },
    load() { image.complete = true; image.naturalWidth = 1692; image.emit('load'); },
    advance(milliseconds) {
      const until = time + milliseconds;
      while (true) {
        const due = [...timers].filter(([, timer]) => timer.at <= until).sort((a, b) => a[1].at - b[1].at)[0];
        if (!due) break;
        const [id, timer] = due;
        time = timer.at;
        timers.delete(id);
        timer.callback();
      }
      time = until;
      for (const [id, callback] of [...frames]) {
        if (!frames.has(id)) continue;
        frames.delete(id);
        callback(time);
      }
    }
  };
}

test('coronation starts after five visible idle seconds and ignores mouse movement', () => {
  const ui = setup();
  ui.intersect(true);
  ui.advance(2000);
  ui.activity('mousemove');
  ui.activity('pointermove');
  ui.advance(2999);
  assert.equal(ui.state(), 'waiting');
  ui.advance(1);
  assert.equal(ui.state(), 'animating');
  assert.equal(ui.pendingTimers(), 0);
  assert.equal(ui.pendingFrames(), 1);
});

test('scroll, pointer press, keyboard input, and touch each restart the idle delay', () => {
  for (const event of ['scroll', 'pointerdown', 'keydown', 'touchstart']) {
    const ui = setup();
    ui.intersect(true);
    ui.advance(4000);
    ui.activity(event);
    ui.advance(4999);
    assert.equal(ui.state(), 'waiting', event);
    ui.advance(1);
    assert.equal(ui.state(), 'animating', event);
  }
});

test('the idle delay waits for the logo image to load', () => {
  const ui = setup({ imageLoaded: false });
  ui.intersect(true);
  ui.advance(20000);
  assert.equal(ui.state(), 'waiting');
  assert.equal(ui.pendingTimers(), 0);
  ui.load();
  ui.advance(4999);
  assert.equal(ui.state(), 'waiting');
  ui.advance(1);
  assert.equal(ui.state(), 'animating');
});

test('hidden documents and offscreen logos freeze the remaining idle time', () => {
  for (const visibility of ['document', 'logo']) {
    const ui = setup();
    const pause = paused => visibility === 'document' ? ui.hide(paused) : ui.intersect(!paused);
    ui.intersect(true);
    ui.advance(2000);
    pause(true);
    assert.equal(ui.pendingTimers(), 0);
    ui.advance(20000);
    assert.equal(ui.state(), 'waiting');
    pause(false);
    ui.advance(2999);
    assert.equal(ui.state(), 'waiting', visibility);
    ui.advance(1);
    assert.equal(ui.state(), 'animating', visibility);
  }
});

test('hidden documents and offscreen logos freeze animation without a resume jump', () => {
  for (const visibility of ['document', 'logo']) {
    const ui = setup();
    const pause = paused => visibility === 'document' ? ui.hide(paused) : ui.intersect(!paused);
    ui.intersect(true);
    ui.advance(5000);
    ui.advance(1000);
    const before = ui.visual();
    pause(true);
    assert.equal(ui.pendingFrames(), 0, visibility);
    ui.advance(20000);
    assert.equal(ui.state(), 'animating', visibility);
    assert.deepEqual(ui.visual(), before, visibility);
    pause(false);
    ui.advance(0);
    assert.deepEqual(ui.visual(), before, `${visibility} resumed without advancing`);
    ui.advance(100);
    assert.notDeepEqual(ui.visual(), before, `${visibility} continues after resuming`);
  }
});

test('the coronation completes once and does not restart after activity or visibility changes', () => {
  const ui = setup();
  ui.intersect(true);
  ui.advance(5000);
  ui.advance(3900);
  assert.equal(ui.state(), 'crowned');
  assert.equal(ui.pendingTimers(), 0);
  assert.equal(ui.pendingFrames(), 0);
  const final = ui.visual();
  ui.activity('scroll');
  ui.activity('keydown');
  ui.hide(true);
  ui.hide(false);
  ui.intersect(false);
  ui.intersect(true);
  ui.advance(20000);
  assert.equal(ui.state(), 'crowned');
  assert.deepEqual(ui.visual(), final);
  assert.equal(ui.pendingTimers(), 0);
  assert.equal(ui.pendingFrames(), 0);
});

test('initial reduced motion reveals the static final pose after the idle delay', () => {
  const ui = setup({ reducedMotion: true });
  ui.intersect(true);
  ui.advance(4999);
  assert.equal(ui.state(), 'waiting');
  ui.advance(1);
  assert.equal(ui.state(), 'crowned');
  assert.equal(ui.pendingFrames(), 0);
  const final = ui.visual();
  ui.advance(20000);
  assert.deepEqual(ui.visual(), final);
});

test('enabling reduced motion during animation finishes immediately and cancels future frames', () => {
  const ui = setup();
  ui.intersect(true);
  ui.advance(5000);
  ui.advance(1000);
  assert.equal(ui.state(), 'animating');
  assert.equal(ui.pendingFrames(), 1);
  ui.reduce(true);
  assert.equal(ui.state(), 'crowned');
  assert.equal(ui.pendingFrames(), 0);
  const final = ui.visual();
  ui.reduce(false);
  ui.advance(20000);
  assert.equal(ui.state(), 'crowned');
  assert.deepEqual(ui.visual(), final);
});
