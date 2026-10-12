import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { runInNewContext } from 'node:vm';
import test from 'node:test';

const math = createRequire(import.meta.url)('../docs/shared/boosting-math.js');
const html = readFileSync(new URL('../docs/boosting/geometry.html', import.meta.url), 'utf8');
const source = readFileSync(new URL('../docs/boosting/geometry.js', import.meta.url), 'utf8');

// Exercise the companion's actual controller and readouts. This small DOM
// harness supplies events and numerical state without claiming layout coverage.
function setup() {
  class Element {
    constructor() { this.listeners = new Map(); this.attrs = {}; this.dataset = {}; this.style = {}; this.children = []; this.value = ''; this._text = ''; }
    addEventListener(type, callback) { this.listeners.set(type, callback); }
    emit(type) { this.listeners.get(type)?.({ target: this }); }
    setAttribute(name, value) { this.attrs[name] = String(value); }
    toggleAttribute() {}
    appendChild(child) { this.children.push(child); return child; }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; this._text = ''; }
    set textContent(value) { this._text = String(value); this.children = []; }
    get textContent() { return this._text + this.children.map(child => child.textContent).join(' '); }
    getBoundingClientRect() { return { width: 640, height: 450, bottom: 800 }; }
  }
  const nodes = new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(match => [match[1], new Element()]));
  nodes.get('rate').value = '.5';
  nodes.get('round').value = '16';
  const shell = new Element(), window = new Element(), document = new Element();
  Object.assign(window, { BoostingMath: math, innerHeight: 800, scrollY: 0,
    matchMedia: query => ({ matches: query.includes('reduced-motion') }) });
  const steps = Array.from({ length: 12 }, (_, index) => {
    const step = new Element();
    step.getBoundingClientRect = () => ({ top: 1000 + 1000 * index - window.scrollY });
    return step;
  });
  Object.assign(document, { getElementById: id => nodes.get(id),
    querySelector: () => shell, querySelectorAll: () => steps,
    createElement: () => new Element(), createElementNS: () => new Element(),
    documentElement: { scrollHeight: 15000 } });
  const frames = [];
  const context = { window, document, performance: { now: () => 0 },
    requestAnimationFrame: callback => { frames.push(callback); return frames.length; },
    cancelAnimationFrame() {}, ResizeObserver: class { observe() {} } };
  Object.defineProperties(context, { innerHeight: { get: () => window.innerHeight }, scrollY: { get: () => window.scrollY } });
  runInNewContext(source, context);
  return { get: id => nodes.get(id), scene(index) {
    window.scrollY = 1000 + index * 1000 - window.innerHeight * .52 + 1;
    window.emit('scroll');
    while (frames.length) frames.shift()(0);
  }, control(id, value, event = 'input') {
    nodes.get(id).value = String(value); nodes.get(id).emit(event);
  } };
}

test('companion late-round readouts preserve a small residual and positive training loss', () => {
  const ui = setup();
  ui.scene(9);
  ui.control('rate', 1, 'change');
  ui.control('round', 24);
  const svg = ui.get('space');
  assert.equal(Number(svg.dataset.loss), 2 ** -46);
  const prediction = JSON.parse(svg.dataset.prediction);
  assert.equal(1 - prediction[0], 2 ** -23);
  assert.equal(-1 - prediction[1], -(2 ** -23));
  const rows = ui.get('vector-readout').children.map(child => child.textContent);
  assert.match(rows[1], /1\.19 × 10\^−7, −1\.19 × 10\^−7, 0/);
  assert.match(rows[2], /1\.42 × 10\^−14/);
  assert.match(html, /Vector readouts are rounded for display/);
});

test('companion exact zero entries remain zero', () => {
  const ui = setup();
  ui.scene(9);
  ui.control('round', 0);
  const rows = ui.get('vector-readout').children.map(child => child.textContent);
  assert.match(rows[0], /\(0, 0, 0\)/);
  assert.match(rows[1], /\(1, −1, 1\)/);
  assert.equal(Number(ui.get('space').dataset.loss), 1.5);
});
