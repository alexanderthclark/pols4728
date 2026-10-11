import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
const story = require('../docs/trees/story.js');
const html = readFileSync(new URL('../docs/trees/index.html', import.meta.url), 'utf8');

// A small event/clock harness exercises the real controller without pretending
// to check browser layout. Native details toggles are queued, not synchronous.
function setup(reducedMotion = false) {
  let time = 0, nextId = 1;
  const frames = new Map(), timers = new Map(), domTasks = [];
  class Element {
    constructor(id = '') {
      this.id = id; this.listeners = new Map(); this.attrs = new Map();
      this.dataset = {}; this.style = {}; this.value = ''; this.textContent = '';
      this.hidden = false; this.disabled = false; this.innerHTML = ''; this._open = false;
      this.classList = { toggle() {} };
    }
    addEventListener(type, listener) { this.listeners.set(type, [...(this.listeners.get(type) || []), listener]); }
    emit(type) { for (const listener of this.listeners.get(type) || []) listener({ type, target: this, preventDefault() {} }); }
    setAttribute(name, value) { this.attrs.set(name, String(value)); }
    getAttribute(name) { return this.attrs.get(name); }
    get open() { return this._open; }
    set open(value) {
      if (this._open !== Boolean(value)) { this._open = Boolean(value); domTasks.push(() => this.emit('toggle')); }
    }
    closest(selector) { return selector === '.stage-shell' ? shell : null; }
    getBoundingClientRect() { return { top: 64, bottom: 800, height: 736 }; }
  }
  const nodes = new Map(Array.from(html.matchAll(/\bid="([^"]+)"/g), match => [match[1], new Element(match[1])]));
  const shell = new Element('stage-shell');
  const steps = Array.from(html.matchAll(/<section\b[^>]*data-step="(\d+)"[^>]*>/g), match => {
    const step = new Element();
    step.getBoundingClientRect = () => ({ top: 1000 + Number(match[1]) * 1000 - win.scrollY, height: 800 });
    return step;
  });
  const reduced = new Element(); reduced.matches = reducedMotion;
  const compact = new Element(); compact.matches = false;
  const win = new Element();
  Object.assign(win, {
    innerWidth: 1200, innerHeight: 800, scrollY: 0,
    performance: { now: () => time },
    matchMedia: query => query.includes('reduced-motion') ? reduced : compact,
    requestAnimationFrame: callback => { const id = nextId++; frames.set(id, callback); return id; },
    cancelAnimationFrame: id => frames.delete(id),
    setTimeout: (callback, delay) => { const id = nextId++; timers.set(id, { at: time + delay, callback }); return id; },
    clearTimeout: id => timers.delete(id),
    scrollTo: ({ top }) => { win.scrollY = top; }
  });
  const doc = new Element();
  doc.getElementById = id => nodes.get(id);
  doc.querySelectorAll = selector => selector === '.step' ? steps : [];
  doc.documentElement = { scrollHeight: 15000 };
  const flushDom = () => { while (domTasks.length) domTasks.shift()(); };
  const advance = elapsed => {
    time += elapsed;
    for (const [id, timer] of Array.from(timers)) if (timer.at <= time) { timers.delete(id); timer.callback(); }
    const batch = Array.from(frames.values()); frames.clear();
    for (const callback of batch) callback(time);
    flushDom();
  };
  const get = id => nodes.get(id);
  const click = id => { assert.equal(get(id).disabled, false, `${id} is disabled`); get(id).emit('click'); flushDom(); };
  const input = (id, value) => { get(id).value = value; get(id).emit('input'); flushDom(); };
  const toggle = open => { get('candidate-details').open = open; flushDom(); };
  story.init(doc, win);
  advance(0);
  return { get, click, input, toggle, advance, shell, reduced };
}

test('process actions step forward and backward, and reduced motion applies completed states immediately', () => {
  const ui = setup(true);
  ui.click('next');
  assert.equal(ui.shell.dataset.scene, '1');
  assert.equal(Number(ui.get('process-progress').value), 0);
  for (const expected of [1, 2, 3]) {
    ui.click('process-next');
    assert.equal(Number(ui.get('process-progress').value), expected);
  }
  assert.equal(ui.get('process-next').disabled, true);
  ui.click('process-back');
  assert.equal(Number(ui.get('process-progress').value), 2);
  ui.click('process-replay');
  assert.equal(Number(ui.get('process-progress').value), 3);
  ui.click('next');
  ui.input('process-progress', 1.5);
  assert.match(ui.get('calculation').textContent, /MSE = 7\.5$/);
});

test('candidate closure does not cancel a new animation, and paused progress resumes before scene changes cancel it', () => {
  const ui = setup();
  ui.click('next'); ui.toggle(true);
  ui.input('candidate-cut', 0);
  assert.match(ui.get('candidate-description').textContent, /Residual SSE: 108\.86/);
  ui.click('process-next');
  assert.equal(ui.get('candidate-details').open, false);
  assert.equal(ui.get('process-replay').textContent, 'Pause');
  ui.advance(725);
  assert.equal(Number(ui.get('process-progress').value), 0.5);
  ui.click('process-replay');
  assert.equal(ui.get('process-replay').textContent, 'Resume sequence');
  ui.advance(1000);
  assert.equal(Number(ui.get('process-progress').value), 0.5);
  ui.click('process-replay'); ui.advance(725);
  assert.equal(Number(ui.get('process-progress').value), 1);
  ui.click('process-replay'); ui.advance(1450);
  assert.equal(Number(ui.get('process-progress').value), 1);
  ui.click('next'); ui.advance(10000);
  assert.equal(ui.shell.dataset.scene, '2');
  assert.equal(Number(ui.get('process-progress').value), 0);
  assert.equal(ui.get('process-replay').textContent, 'Replay sequence');
});

test('candidate selection resets to the winning split for each new residual target', () => {
  const ui = setup(true);
  ui.click('next'); ui.toggle(true);
  assert.equal(Number(ui.get('candidate-cut').value), 4);
  assert.equal(ui.get('candidate-value').textContent, '8 km');
  ui.input('candidate-cut', 0); ui.toggle(false);
  assert.equal(ui.get('candidate-value').textContent, '8 km');
  ui.click('next'); ui.click('next'); ui.toggle(true);
  assert.equal(Number(ui.get('candidate-cut').value), 1);
  assert.equal(ui.get('candidate-value').textContent, '3.5 km');
  assert.match(ui.get('candidate-description').textContent, /Residual SSE: 6 min²/);
});

test('the contribution scanner follows both cutpoints and pauses without changing the inspected input', () => {
  const ui = setup();
  for (let i = 0; i < 5; i++) ui.click('next');
  assert.equal(ui.get('contribution-controls').hidden, false);
  for (const [input, expected] of [[3.5, 6], [3.51, 10], [8, 10], [8.01, 18]]) {
    ui.input('contribution-x', input);
    assert.match(ui.get('calculation').textContent, new RegExp(`= ${expected} min$`));
  }
  ui.click('replay-contributions'); ui.advance(2500);
  assert.equal(ui.get('contribution-x-value').textContent, '6.5 km');
  ui.click('replay-contributions'); ui.advance(3000);
  assert.equal(ui.get('contribution-x-value').textContent, '6.5 km');
  ui.reduced.matches = true;
  ui.click('replay-contributions');
  assert.equal(ui.get('contribution-x-value').textContent, '12 km');
  assert.match(ui.get('calculation').textContent, /= 18 min$/);
});
