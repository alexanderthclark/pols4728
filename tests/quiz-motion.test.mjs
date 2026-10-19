import assert from 'node:assert/strict';
import test from 'node:test';
import {clamp01, easeInOut, phase, lerp, sceneProgress, activeScene} from '../docs/quiz/motion.mjs';
import {createMotionView} from '../docs/quiz/animate.js';

const closeTo = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-10, `${actual} should equal ${expected}`);

test('math helpers clamp phases and ease symmetrically without changing interpolation units', () => {
  assert.deepEqual([-Infinity, -1, 0, 0.4, 1, 2, Infinity].map(clamp01), [0, 0, 0, 0.4, 1, 1, 1]);
  assert.deepEqual([-1, 0, 0.5, 1, 2].map(easeInOut), [0, 0, 0.5, 1, 1]);
  closeTo(easeInOut(0.25), 0.15625);
  for (const p of [0.1, 0.2, 0.6, 0.95]) closeTo(easeInOut(p) + easeInOut(1 - p), 1);
  for (const [p, expected] of [[0, 0], [0.2, 0], [0.4, 0.5], [0.6, 1], [1, 1]]) closeTo(phase(p, 0.2, 0.6), expected);
  assert.equal(phase(0.5), 0.5);
  assert.equal(phase(0.49, 0.5, 0.5), 0);
  assert.equal(phase(0.5, 0.5, 0.5), 1);
  assert.throws(() => phase(0.5, 0.7, 0.3), /at least/);
  assert.equal(lerp(100, 300, 0.25), 150);
  assert.equal(lerp(10, -10, 0.5), 0);
});

test('scroll construction reverses exactly and holds the completed calculation through the last 28% of a step', () => {
  const tops = [600, 300, 120, -60, -240, -420, -600, -700];
  const expected = [0, 0, 0.25, 0.5, 0.75, 1, 1, 1];
  tops.forEach((top, i) => closeTo(sceneProgress(top, 1000, 300), expected[i]));
  [...tops].reverse().forEach((top, i) => closeTo(sceneProgress(top, 1000, 300), [...expected].reverse()[i]));
  assert.equal(sceneProgress(Infinity, 1000, 300), 0);
  assert.equal(sceneProgress(-Infinity, 1000, 300), 1);
  assert.equal(sceneProgress(300, 0, 300), 0);
  assert.equal(sceneProgress(299, 0, 300), 1);
  assert.equal(sceneProgress(299, -100, 300), 1);
});

test('reduced motion uses five reversible logical steps with stable boundaries', () => {
  const at = p => sceneProgress(300 - p * 720, 1000, 300, true);
  assert.equal(at(-1), 0);
  assert.equal(at(0), 0);
  for (let i = 1; i <= 5; i++) {
    const boundary = i / 5;
    closeTo(at(boundary - 1e-6), (i - 1) / 5);
    closeTo(at(boundary), boundary);
    closeTo(at(boundary + 1e-6), boundary);
  }
  assert.deepEqual([1.2, 0.9, 0.7, 0.5, 0.3, 0.1, -1].map(at), [1, 0.8, 0.6, 0.4, 0.2, 0, 0]);
});

test('active scene is determined by current geometry in either scroll direction, including exact boundaries', () => {
  const tops = [100, 1100, 2100];
  const scroll = [0, 100, 799, 800, 1799, 1800, 4000];
  const expected = [0, 0, 0, 1, 1, 2, 2];
  const at = offset => activeScene(tops.map(top => top - offset), 300);
  assert.deepEqual(scroll.map(at), expected);
  assert.deepEqual([...scroll].reverse().map(at), [...expected].reverse());
  assert.equal(activeScene([], 300), 0);
});

// A small DOM surface exercises the actual renderer without a browser or clocks.
class Element {
  constructor(dataset = {}) {
    this.dataset = dataset;
    this.style = {};
    this.attributes = new Map();
    this.selectors = new Map();
    this.textContent = '';
  }
  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  getAttribute(name) { return this.attributes.get(name) ?? null; }
  hasAttribute(name) { return this.attributes.has(name); }
  querySelectorAll(selector) { return this.selectors.get(selector) ?? []; }
  querySelector(selector) { return this.querySelectorAll(selector)[0] ?? null; }
  bind(selector, ...elements) { this.selectors.set(selector, elements); return this; }
}

function fixture() {
  const root = new Element();
  const progress = new Element();
  const readout = new Element();
  root.bind('progress', progress).bind('.motion-readout', readout);
  const view = createMotionView({getDraws: () => ({firstSlopes: [1], secondSlopes: [1]}), getRisk: () => []});
  return {root, progress, readout, view};
}

test('candidate search bars and their expected perfect count return to the same intermediate state on reverse scrolling', () => {
  const {root, progress, readout, view} = fixture();
  const bar = new Element({searchBar: '220,120'});
  const perfect = new Element({searchPerfect: '32'});
  root.bind('[data-search-bar]', bar).bind('[data-search-perfect]', perfect);
  view.mount(root, 1);
  assert.equal(bar.getAttribute('height'), '0');
  assert.equal(bar.getAttribute('y'), '220');
  view.update(root, 0.425);
  assert.equal(bar.getAttribute('height'), '60');
  assert.equal(bar.getAttribute('y'), '160');
  assert.equal(perfect.textContent, 'Perfect: 16 expected');
  const snapshot = () => [bar.getAttribute('height'), bar.getAttribute('y'), perfect.textContent, readout.textContent, progress.value];
  const middle = snapshot();
  view.update(root, 1);
  assert.equal(bar.getAttribute('height'), '120');
  assert.equal(perfect.textContent, 'Perfect: 32 expected');
  view.update(root, 0.425);
  assert.deepEqual(snapshot(), middle);
  view.update(root, -5);
  assert.equal(bar.getAttribute('height'), '0');
  assert.equal(progress.value, 0);
  view.update(root, 5);
  assert.equal(bar.getAttribute('height'), '120');
  assert.equal(progress.value, 1);
  view.remove(root);
  view.update(root, 0);
  assert.equal(bar.getAttribute('height'), '120');
});

test('every partial weighted calculation displays row counts and exact products that sum to its totals, forward and backward', () => {
  const {root, view} = fixture();
  const table = [];
  for (let i = 0; i < 4; i++) {
    const row = new Element();
    const rule = new Element(), count = new Element(), tr = new Element(), t2 = new Element();
    row.bind('[data-weight-rule]', rule).bind('[data-weight-count]', count).bind('[data-weight-tr]', tr).bind('[data-weight-t2]', t2);
    root.bind(`[data-weight-row="${i}"]`, row);
    table.push({count, tr, t2});
  }
  const totalCount = new Element(), totalTR = new Element(), totalT2 = new Element(), ratio = new Element();
  root.bind('[data-total-count]', totalCount).bind('[data-total-tr]', totalTR).bind('[data-total-t2]', totalT2).bind('[data-total-ratio]', ratio);
  view.mount(root, 15);
  // Explicit individual students provide an independent count-weighted reference.
  const groups = [[12, -0.25, -2.5], [4, 0.75, 7.5], [4, -0.75, -7.5], [12, 0.25, 2.5]];
  const students = groups.flatMap(([count, T, R], group) => Array.from({length: count}, () => ({group, T, R})));
  const positions = Array.from({length: 33}, (_, n) => n === 0 ? 0 : n === 32 ? 1 : 0.02 + 0.93 * (0.5 - Math.sin(Math.asin(1 - 2 * n / 32) / 3)));
  const snapshots = [];
  function check(n) {
    view.update(root, positions[n]);
    const included = students.slice(0, n);
    const displayed = table.map(row => [Number(row.count.textContent), Number(row.tr.textContent), Number(row.t2.textContent)]);
    displayed.forEach(([count, tr, t2], group) => {
      const members = included.filter(student => student.group === group);
      assert.equal(count, members.length);
      assert.equal(tr, members.reduce((sum, student) => sum + student.T * student.R, 0));
      assert.equal(t2, members.reduce((sum, student) => sum + student.T ** 2, 0));
    });
    assert.equal(Number(totalCount.textContent), n);
    assert.equal(Number(totalTR.textContent), displayed.reduce((sum, row) => sum + row[1], 0));
    assert.equal(Number(totalT2.textContent), displayed.reduce((sum, row) => sum + row[2], 0));
    if (n) {
      assert.equal(Number(totalTR.textContent) / Number(totalT2.textContent), 10);
      assert.match(ratio.textContent, / = 10$/);
    } else assert.equal(ratio.textContent, 'τ̂ = —');
    return [displayed, totalCount.textContent, totalTR.textContent, totalT2.textContent, ratio.textContent];
  }
  for (let n = 0; n <= 32; n++) snapshots.push(check(n));
  assert.equal(Number(totalTR.textContent), 60);
  assert.equal(Number(totalT2.textContent), 6);
  for (let n = 32; n >= 0; n--) assert.deepEqual(check(n), snapshots[n]);
});

test('treatment adjustment restores partial positions and changes the axis description only when adjustment is complete', () => {
  const {root, view} = fixture();
  const point = new Element({adjustX: '100'}), axis = new Element();
  root.bind('[data-adjust-x]', point).bind('[data-treatment-axis]', axis);
  view.mount(root, 14);
  assert.equal(point.getAttribute('transform'), 'translate(100 0)');
  assert.equal(axis.textContent, 'Subtracting ê(X) from D');
  view.update(root, 0.43);
  const middle = point.getAttribute('transform');
  closeTo(Number(middle.match(/translate\(([^ ]+)/)[1]), 50);
  assert.equal(axis.textContent, 'Subtracting ê(X) from D');
  view.update(root, 1);
  assert.equal(point.getAttribute('transform'), 'translate(0 0)');
  assert.equal(axis.textContent, 'Adjusted treatment T = D − ê(X)');
  view.update(root, 0.43);
  assert.equal(point.getAttribute('transform'), middle);
  assert.equal(axis.textContent, 'Subtracting ê(X) from D');
});

test('outcome adjustment uses the responsive plot geometry and keeps ticks aligned while zooming to residuals', () => {
  const {root, view} = fixture();
  const bottom = 321, span = 280;
  const scale = new Element({outcomeScale: `${bottom},${span}`});
  const point = new Element({adjustY: `0,${bottom - 7.5 / 20 * span}`});
  point.bind('text', new Element());
  const ticks = Array.from({length: 5}, () => new Element());
  const zero = new Element(), axis = new Element();
  root.bind('[data-outcome-scale]', scale).bind('[data-adjust-y]', point).bind('[data-y-tick]', ...ticks).bind('[data-zero-line]', zero).bind('[data-outcome-axis]', axis);
  view.mount(root, 13);
  assert.deepEqual(ticks.map(tick => Number(tick.textContent)), [-10, 20, 50, 80, 110]);
  closeTo(Number(point.getAttribute('transform').match(/translate\(0 ([^)]+)/)[1]), (bottom - 60 / 120 * span) - (bottom - 7.5 / 20 * span));
  const original = point.getAttribute('transform');
  view.update(root, 1);
  closeTo(Number(point.getAttribute('transform').match(/translate\(0 ([^)]+)/)[1]), 0);
  closeTo(Number(zero.getAttribute('y1')), bottom - span / 2);
  assert.equal(zero.getAttribute('y1'), zero.getAttribute('y2'));
  assert.deepEqual(ticks.map(tick => Number(tick.textContent)), [-10, -5, 0, 5, 10]);
  assert.equal(axis.textContent, 'Adjusted score R (points)');
  view.update(root, 0);
  assert.equal(point.getAttribute('transform'), original);
  assert.equal(axis.textContent, 'Subtracting ŷ(X) from Y (points)');
});
