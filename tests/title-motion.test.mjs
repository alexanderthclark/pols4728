import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { runInNewContext } from 'node:vm';

const controller = readFileSync(new URL('../docs/trees/title-motion.js', import.meta.url), 'utf8');
const duration = 1000;

// This clock checks controller state and scheduling, not browser layout or art.
function setup({ reducedMotion = false } = {}) {
  let time = 0, nextFrame = 1, intersection;
  const frames = new Map(), renders = [];
  class Target {
    constructor() { this.listeners = new Map(); this.hidden = false; }
    addEventListener(type, callback) {
      const callbacks = this.listeners.get(type) || [];
      callbacks.push(callback);
      this.listeners.set(type, callbacks);
    }
    emit(type) { for (const callback of this.listeners.get(type) || []) callback(); }
  }
  const art = new Target();
  const nodes = new Map([['title-picture', art]]);
  const document = new Target();
  document.getElementById = id => nodes.get(id);
  const media = new Target();
  media.matches = reducedMotion;
  const window = {
    TitlePicture: {
      duration,
      render(elapsed) { renders.push(elapsed); return `<svg data-elapsed="${elapsed}"></svg>`; }
    },
    matchMedia: () => media,
    requestAnimationFrame(callback) { const id = nextFrame++; frames.set(id, callback); return id; },
    cancelAnimationFrame(id) { frames.delete(id); },
    IntersectionObserver: class {
      constructor(callback) { intersection = callback; }
      observe() {}
    }
  };
  runInNewContext(controller, { window, document });
  return {
    renders,
    elapsed: () => renders.at(-1),
    pending: () => frames.size,
    intersect(visible) { intersection([{ isIntersecting: visible }]); },
    hide(hidden) { document.hidden = hidden; document.emit('visibilitychange'); },
    reduce(matches) { media.matches = matches; media.emit('change'); },
    advance(milliseconds) {
      time += milliseconds;
      for (const [id, callback] of [...frames]) {
        frames.delete(id);
        callback(time);
      }
    }
  };
}

test('title autoplay waits until the illustration intersects the viewport', () => {
  const ui = setup();
  assert.equal(ui.elapsed(), 0);
  assert.equal(ui.pending(), 0);
  ui.advance(10000);
  ui.intersect(false);
  assert.equal(ui.pending(), 0);
  ui.intersect(true);
  assert.equal(ui.pending(), 1);
  ui.advance(0);
  ui.advance(200);
  assert.equal(ui.elapsed(), 200);
});

test('hidden documents and offscreen titles freeze elapsed time without a resume jump', () => {
  const ui = setup();
  ui.intersect(true);
  ui.advance(0);
  ui.advance(200);
  for (const setPaused of [paused => ui.hide(paused), paused => ui.intersect(!paused)]) {
    const before = ui.elapsed();
    setPaused(true);
    assert.equal(ui.pending(), 0);
    ui.advance(10000);
    assert.equal(ui.elapsed(), before);
    setPaused(false);
    assert.equal(ui.pending(), 1);
    ui.advance(0);
    assert.equal(ui.elapsed(), before);
    ui.advance(100);
    assert.equal(ui.elapsed(), before + 100);
  }
});

test('title finishes once and does not restart when visibility changes', () => {
  const ui = setup();
  ui.intersect(true);
  ui.advance(0);
  ui.advance(duration + 500);
  assert.equal(ui.elapsed(), duration);
  assert.equal(ui.pending(), 0);
  const count = ui.renders.length;
  ui.advance(10000);
  ui.intersect(false);
  ui.intersect(true);
  ui.hide(true);
  ui.hide(false);
  ui.advance(10000);
  assert.equal(ui.elapsed(), duration);
  assert.equal(ui.renders.length, count);
  assert.equal(ui.pending(), 0);

});

test('initial reduced motion shows the final pose without scheduling animation', () => {
  const ui = setup({ reducedMotion: true });
  assert.equal(ui.elapsed(), duration);
  ui.intersect(true);
  ui.advance(10000);
  assert.equal(ui.pending(), 0);
  assert.deepEqual(ui.renders, [duration]);
});

test('enabling reduced motion finishes immediately and cancels the pending frame', () => {
  const ui = setup();
  ui.intersect(true);
  ui.advance(0);
  ui.advance(200);
  assert.equal(ui.pending(), 1);
  ui.reduce(true);
  assert.equal(ui.elapsed(), duration);
  assert.equal(ui.pending(), 0);
  const count = ui.renders.length;
  ui.advance(10000);
  ui.reduce(false);
  assert.equal(ui.renders.length, count);
  assert.equal(ui.pending(), 0);
});
