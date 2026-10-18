import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { runInNewContext } from 'node:vm';

const source = readFileSync(new URL('../docs/shap/title-motion.js', import.meta.url), 'utf8');

// Deterministic controller tests, without substituting for browser layout or art QA.
function setup({ reducedMotion = false, intersectionObserver = true,
  coverTop = 64, coverHeight = 1400, viewportHeight = 800, mastheadHeight = 64,
  activatedCoverHeight, contentHeight,
  missingArt = false, missingCover = false, missingRenderer = false, renderFails = false } = {}) {
  let time = 0, nextFrame = 1, intersection, disconnected = false;
  const frames = new Map(), renders = [];
  class Target {
    constructor() { this.listeners = new Map(); this.hidden = false; }
    addEventListener(type, callback) {
      const callbacks = this.listeners.get(type) || new Set();
      callbacks.add(callback); this.listeners.set(type, callbacks);
    }
    removeEventListener(type, callback) { this.listeners.get(type)?.delete(callback); }
    emit(type) { for (const callback of this.listeners.get(type) || []) callback(); }
    countListeners() { return [...this.listeners.values()].reduce((sum, set) => sum + set.size, 0); }
  }
  const art = new Target(), cover = new Target(), fallback = new Target();
  const classes = new Set();
  art.hidden = true;
  art.getBoundingClientRect = () => ({ top: art.top ?? 220, bottom: art.bottom ?? 650 });
  cover.offsetHeight = coverHeight;
  cover.getBoundingClientRect = () => ({ top: coverTop });
  cover.querySelector = selector => selector === '.title-page-content' && contentHeight !== undefined
    ? { getBoundingClientRect: () => ({ height: contentHeight }) } : null;
  cover.classList = {
    add(name) {
      classes.add(name);
      if (name === 'has-title-animation' && activatedCoverHeight !== undefined) cover.offsetHeight = activatedCoverHeight;
    },
    remove: name => classes.delete(name), contains: name => classes.has(name),
  };
  const document = new Target();
  document.documentElement = { clientHeight: viewportHeight };
  const nodes = new Map([
    ['shap-title-picture', missingArt ? null : art],
    ['beginning', missingCover ? null : cover],
    ['shap-title-fallback', fallback],
  ]);
  document.getElementById = id => nodes.get(id);
  document.querySelector = selector => selector === '.masthead'
    ? { getBoundingClientRect: () => ({ height: mastheadHeight }) } : null;
  const media = new Target(); media.matches = reducedMotion;
  const window = new Target();
  Object.assign(window, {
    innerHeight: viewportHeight,
    SHAPTitlePicture: missingRenderer ? undefined : {
      render(elapsedMs, openProgress, reduced) {
        if (renderFails) throw new Error('Illustration unavailable');
        renders.push({ elapsedMs, openProgress, reduced });
        return `<svg data-elapsed="${elapsedMs}" data-open="${openProgress}"></svg>`;
      },
    },
    matchMedia: () => media,
    requestAnimationFrame(callback) { const id = nextFrame++; frames.set(id, callback); return id; },
    cancelAnimationFrame(id) { frames.delete(id); },
  });
  if (intersectionObserver) window.IntersectionObserver = class {
    constructor(callback) { intersection = callback; }
    observe(node) { assert.equal(node, art); }
    disconnect() { disconnected = true; }
  };
  runInNewContext(source, { window, document });
  return {
    art, cover, fallback, renders, window, document, media,
    motion: window.SHAPTitleMotion,
    latest: () => renders.at(-1),
    pending: () => frames.size,
    listenerCount: () => window.countListeners() + document.countListeners() + media.countListeners(),
    disconnected: () => disconnected,
    intersect(visible) { intersection([{ isIntersecting: visible }]); },
    hide(hidden) { document.hidden = hidden; document.emit('visibilitychange'); },
    reduce(matches) { media.matches = matches; media.emit('change'); },
    scroll(top) { coverTop = top; window.emit('scroll'); },
    resize(height = window.innerHeight, trackHeight = cover.offsetHeight) {
      window.innerHeight = height; cover.offsetHeight = trackHeight; window.emit('resize');
    },
    advance(milliseconds) {
      time += milliseconds;
      for (const [id, callback] of [...frames]) { frames.delete(id); callback(time); }
    },
  };
}

test('the cover begins closed and activates only after the renderer succeeds', () => {
  const ui = setup();
  assert.deepEqual(ui.latest(), { elapsedMs: 0, openProgress: 0, reduced: false });
  assert.equal(ui.art.hidden, false);
  assert.equal(ui.fallback.hidden, true);
  assert.equal(ui.cover.classList.contains('has-title-animation'), true);
  assert.equal(ui.pending(), 0);
  ui.advance(10000);
  assert.equal(ui.renders.length, 1);
  ui.intersect(true);
  assert.equal(ui.pending(), 1);
  ui.advance(0);
  assert.equal(ui.renders.length, 1, 'an unchanged first frame does not rewrite the SVG');
  ui.advance(250);
  assert.equal(ui.latest().elapsedMs, 250);
});

test('only native cover travel opens the hatch, and reversing scroll closes it again', () => {
  const ui = setup();
  const travel = 1400 - (800 - 64);
  ui.intersect(true); ui.advance(0); ui.advance(2000);
  assert.equal(ui.latest().openProgress, 0, 'looping people do not open the hatch on a timer');
  const retainedElapsed = ui.latest().elapsedMs;
  for (const progress of [.25, .5, 1, .5, 0]) {
    ui.scroll(64 - travel * progress); ui.advance(0);
    assert.equal(ui.latest().openProgress, progress);
    assert.equal(ui.latest().elapsedMs, retainedElapsed);
  }
  ui.scroll(64 - travel * 2); ui.advance(0);
  assert.equal(ui.latest().openProgress, 1);
  ui.scroll(100); ui.advance(0);
  assert.equal(ui.latest().openProgress, 0);
});

test('the first activated pose uses the added track rather than the fallback cover height', () => {
  const ui = setup({ coverTop: -268, coverHeight: 736, activatedCoverHeight: 1400 });
  assert.equal(ui.latest().openProgress, .5);
  assert.equal(ui.motion.state().openProgress, .5);
  assert.equal(ui.pending(), 0);
});

test('hatch travel follows the measured sticky content when small viewport and innerHeight differ', () => {
  const ui = setup({ coverHeight: 1400, viewportHeight: 900, contentHeight: 736 });
  ui.intersect(true); ui.advance(0);
  ui.scroll(64 - 332); ui.advance(0);
  assert.equal(ui.latest().openProgress, .5);
  ui.resize(1000); ui.advance(0);
  assert.equal(ui.latest().openProgress, .5, 'browser chrome height does not change the sticky track');
  ui.scroll(64 - 664); ui.advance(0);
  assert.equal(ui.latest().openProgress, 1);
});

test('offscreen and hidden loops freeze and resume without adding paused wall time', () => {
  const ui = setup();
  ui.intersect(true); ui.advance(0); ui.advance(200);
  for (const pause of [paused => ui.intersect(!paused), paused => ui.hide(paused)]) {
    const before = ui.latest().elapsedMs, count = ui.renders.length;
    pause(true); assert.equal(ui.pending(), 0);
    ui.advance(10000); ui.scroll(-100);
    assert.equal(ui.renders.length, count);
    pause(false); assert.equal(ui.pending(), 1);
    ui.advance(0);
    assert.equal(ui.latest().elapsedMs, before);
    ui.advance(100);
    assert.equal(ui.latest().elapsedMs, before + 100);
  }
});

test('resize and hash updates recompute reversible progress without restarting the loop', () => {
  const ui = setup();
  ui.intersect(true); ui.advance(0); ui.advance(100);
  ui.scroll(64 - 332); ui.advance(0);
  assert.equal(ui.latest().openProgress, .5);
  ui.resize(800, 2064); ui.advance(0);
  assert.equal(ui.latest().openProgress, .25);
  assert.equal(ui.latest().elapsedMs, 100);
  ui.scroll(64); ui.window.emit('hashchange'); ui.advance(0);
  assert.equal(ui.latest().openProgress, 0);
  assert.equal(ui.pending(), 1, 'events share one scheduled frame');
});

test('initial reduced motion presents one complete open pose with no loop or repeated SVG writes', () => {
  const ui = setup({ reducedMotion: true });
  assert.deepEqual(ui.latest(), { elapsedMs: 0, openProgress: 1, reduced: true });
  ui.intersect(true); ui.scroll(-500); ui.resize();
  ui.advance(10000);
  assert.equal(ui.pending(), 0);
  assert.equal(ui.renders.length, 1);
});

test('changing reduced motion cancels the loop and returning to motion retains the elapsed phase', () => {
  const ui = setup();
  ui.intersect(true); ui.advance(0); ui.advance(300);
  ui.reduce(true);
  assert.equal(ui.pending(), 0);
  assert.deepEqual(ui.latest(), { elapsedMs: 0, openProgress: 1, reduced: true });
  ui.advance(10000);
  ui.reduce(false); ui.advance(0);
  assert.equal(ui.latest().elapsedMs, 300);
  assert.equal(ui.latest().openProgress, 0);
  ui.advance(100);
  assert.equal(ui.latest().elapsedMs, 400);
});

test('pagehide stops scheduled work while pageshow resumes the retained visible phase', () => {
  const ui = setup();
  ui.intersect(true); ui.advance(0); ui.advance(250);
  ui.window.emit('pagehide'); assert.equal(ui.pending(), 0);
  ui.advance(10000);
  ui.window.emit('pageshow'); assert.equal(ui.pending(), 1);
  ui.advance(0); assert.equal(ui.latest().elapsedMs, 250);
});

test('destroy cancels the frame, disconnects observation, and removes all event listeners', () => {
  const ui = setup();
  ui.intersect(true); ui.advance(0); ui.advance(200);
  assert.ok(ui.listenerCount() > 0);
  ui.motion.destroy();
  assert.equal(ui.pending(), 0);
  assert.equal(ui.disconnected(), true);
  assert.equal(ui.listenerCount(), 0);
  assert.equal(ui.motion.state().destroyed, true);
  const count = ui.renders.length;
  ui.advance(10000); ui.scroll(-500); ui.reduce(true); ui.intersect(true);
  ui.motion.refresh(); ui.motion.destroy();
  assert.equal(ui.renders.length, count);
  assert.equal(ui.pending(), 0);
});

test('without IntersectionObserver, viewport geometry still pauses offscreen animation', () => {
  const ui = setup({ intersectionObserver: false });
  assert.equal(ui.pending(), 1);
  ui.advance(0); ui.advance(100);
  ui.art.top = -500; ui.art.bottom = 30; ui.scroll(-100);
  assert.equal(ui.pending(), 0);
  ui.advance(10000);
  ui.art.top = 200; ui.art.bottom = 600; ui.scroll(64);
  ui.advance(0);
  assert.equal(ui.latest().elapsedMs, 100);
});

test('zero travel is finite, and unavailable markup preserves the static fallback', () => {
  const ui = setup({ coverHeight: 736 });
  assert.equal(ui.latest().openProgress, 0);
  ui.intersect(true); ui.advance(0); ui.scroll(63); ui.advance(0);
  assert.equal(ui.latest().openProgress, 1);
  const failed = setup({ renderFails: true });
  assert.equal(failed.pending(), 0);
  assert.equal(failed.art.hidden, true);
  assert.equal(failed.fallback.hidden, false);
  assert.equal(failed.cover.classList.contains('has-title-animation'), false);
  assert.equal(failed.motion.state().failed, true);
  for (const option of ['missingArt', 'missingCover', 'missingRenderer']) {
    const absent = setup({ [option]: true });
    assert.equal(absent.pending(), 0);
    assert.equal(absent.listenerCount(), 0);
    assert.equal(absent.renders.length, 0);
  }
});
