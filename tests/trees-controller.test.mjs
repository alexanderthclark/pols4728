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
  let time = 0, nextId = 1, stageHeight = 736;
  const frames = new Map(), timers = new Map(), domTasks = [];
  class Element {
    constructor(id = '') {
      this.id = id; this.listeners = new Map(); this.attrs = new Map();
      this.dataset = {}; this.style = {}; this.value = ''; this.textContent = '';
      this.hidden = false; this.disabled = false; this.innerHTML = ''; this._open = false;
      this.classList = { toggle() {} };
    }
    set textContent(value) { this._text = String(value); this.innerHTML = ''; }
    get textContent() { return this.innerHTML.includes('<math') && this.attrs.has('aria-label') ? this.attrs.get('aria-label') : this._text; }
    addEventListener(type, listener) { this.listeners.set(type, [...(this.listeners.get(type) || []), listener]); }
    emit(type) { for (const listener of this.listeners.get(type) || []) listener({ type, target: this, preventDefault() {} }); }
    setAttribute(name, value) { this.attrs.set(name, String(value)); }
    getAttribute(name) { return this.attrs.get(name); }
    removeAttribute(name) { this.attrs.delete(name); }
    get open() { return this._open; }
    set open(value) {
      if (this._open !== Boolean(value)) { this._open = Boolean(value); domTasks.push(() => this.emit('toggle')); }
    }
    closest(selector) { return selector === '.stage-shell' ? shell : null; }
    getBoundingClientRect() { return { top: 64, bottom: 64 + stageHeight, height: stageHeight }; }
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
  doc.documentElement = { scrollHeight: 15000, dataset: {} };
  const flushDom = () => { while (domTasks.length) domTasks.shift()(); };
  const advance = elapsed => {
    time += elapsed;
    for (const [id, timer] of Array.from(timers)) if (timer.at <= time) { timers.delete(id); timer.callback(); }
    for (const id of Array.from(frames.keys())) {
      const callback = frames.get(id); frames.delete(id);
      if (callback) callback(time);
    }
    flushDom();
  };
  const get = id => nodes.get(id);
  const click = id => { assert.equal(get(id).disabled, false, `${id} is disabled`); get(id).emit('click'); flushDom(); };
  const input = (id, value) => { get(id).value = value; get(id).emit('input'); flushDom(); };
  const toggle = open => { get('candidate-details').open = open; flushDom(); };
  const scroll = top => { win.scrollY = top; win.emit('scroll'); advance(0); };
  // Position the reading line at a chosen fraction through a narrative step.
  const scrollAt = (scene, fraction) => {
    const readingLine = win.innerHeight * 0.46;
    scroll(1000 + scene * 1000 - readingLine + fraction * 800);
  };
  const resize = height => { win.innerHeight = height; win.emit('resize'); advance(0); };
  story.init(doc, win);
  advance(0);
  return {
    get, click, input, toggle, advance, shell, reduced, scrollAt, resize, sameScroll: () => scroll(win.scrollY),
    setCompact: (height = 900) => { compact.matches = true; win.innerWidth = 600; win.innerHeight = height; },
    setStageHeight: height => { stageHeight = height; }
  };
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



const processValue = ui => Number(ui.get('process-progress').value);
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) <= 0.0011, `${actual} should equal ${expected}`);

test('opening starts with observations and reveals the mean before entering residual fitting', () => {
  const ui = setup();
  assert.equal(ui.shell.dataset.layout, 'visual');
  assert.equal(ui.get('stage-topic').hidden, true);
  assert.equal(ui.get('visual-heading').hidden, false);
  assert.equal(Number(ui.get('process-progress').max), 1);
  assert.doesNotMatch(ui.get('diagram').innerHTML, /data-axis="Residual/);
  assert.doesNotMatch(ui.get('diagram').innerHTML, /data-curve="prediction"/);
  ui.scrollAt(0, 0.35);
  assert.equal(processValue(ui), 0);
  ui.scrollAt(0, 0.65);
  close(processValue(ui), 0.5);
  ui.scrollAt(0, 0.95);
  assert.equal(processValue(ui), 1);
  assert.equal(ui.get('process-next').disabled, true);
  ui.scrollAt(1, 0.04);
  assert.equal(Number(ui.get('process-progress').max), 3);
  assert.equal(processValue(ui), 0);
  assert.doesNotMatch(ui.get('diagram').innerHTML, /data-axis="Residual/);
  ui.scrollAt(2, 0.04);
  assert.equal(ui.shell.dataset.layout, 'visual');
  assert.equal(ui.get('visual-heading').hidden, false);
  assert.equal(ui.get('stage-topic').hidden, true);
});

test('all scenes keep the full-width presentation without a link to the unpublished chapter', () => {
  const ui = setup(true);
  assert.equal(ui.get('full-explanation'), undefined);
  assert.ok(ui.get('further-reading'));
  assert.equal(ui.get('stage-equation').textContent, '');
  ui.click('process-next');
  assert.equal(ui.get('stage-equation').textContent, 'F₀(x) = ȳ = 12');
  for(let scene=0;scene<10;scene++){
    if(scene)ui.click('next');
    assert.equal(ui.get('stage-count').textContent, `${String(scene+1).padStart(2,'0')} / 10`);
    assert.equal(ui.shell.dataset.layout, 'visual');
    assert.equal(ui.get('visual-heading').hidden, false);
    assert.ok(ui.get('visual-heading').textContent.length>0);
    assert.equal(ui.get('stage-topic').hidden, true);
    assert.equal(ui.get('continue-boosting'), undefined);
    assert.equal(ui.get('stage-equation').hidden, scene===6);
    assert.equal(ui.get('stage-caption').hidden, scene===6);
    assert.equal(ui.get('calculation').hidden, scene===6);
  }
  ui.click('previous');
  assert.equal(ui.get('continue-boosting'), undefined);
});

test('later-round headings follow the fitted tree and keep the final training-fit qualification', () => {
  const ui = setup(true);
  for(let scene=0;scene<5;scene++)ui.click('next');
  assert.equal(ui.get('stage-count').textContent, '06 / 10');
  assert.equal(ui.get('stage-equation').textContent, 'h₃ ← fit(y − F₂)');
  ui.click('continuation-next');
  assert.equal(ui.get('stage-equation').textContent, 'F₃ = F₂ + h₃');
  for(let round=4;round<=8;round++)ui.click('continuation-next');
  assert.equal(ui.get('stage-equation').textContent, 'F₈ = F₇ + h₈');
  assert.match(ui.get('stage-caption').textContent, /never reaches exact zero/);
  assert.match(ui.get('stage-caption').textContent, /accuracy on new data/);
});

test('opening manual controls and reduced motion finish and reverse the mean reveal', () => {
  const ui = setup(true);
  ui.click('process-next');
  assert.equal(processValue(ui), 1);
  assert.match(ui.get('calculation').textContent, /F₀.*12/);
  ui.click('process-back');
  assert.equal(processValue(ui), 0);
  ui.click('process-replay');
  assert.equal(processValue(ui), 1);
  ui.scrollAt(0, 0.65);
  assert.equal(processValue(ui), 0);
  ui.scrollAt(0, 0.95);
  assert.equal(processValue(ui), 1);
});

test('scroll position advances and rewinds the delivery process, including direct scene jumps', () => {
  const ui = setup();
  ui.scrollAt(1, 0.04);
  assert.equal(ui.shell.dataset.scene, '1');
  assert.equal(processValue(ui), 0);
  ui.scrollAt(1, 0.465); close(processValue(ui), 1.5);
  ui.scrollAt(1, 0.9); assert.equal(processValue(ui), 3);
  ui.scrollAt(1, 0.2725); close(processValue(ui), 0.75);
  ui.scrollAt(4, 0.85);
  assert.equal(ui.shell.dataset.scene, '4');
  assert.equal(processValue(ui), 3);
  assert.match(ui.get('calculation').textContent, /MSE = 0\.75/);
  ui.scrollAt(2, 0.465);
  assert.equal(ui.shell.dataset.scene, '2');
  close(processValue(ui), 1.5);
  assert.match(ui.get('calculation').textContent, /MSE = 7\.5/);
});

test('manual controls hold until the scroll position changes, which closes candidates and cancels replay', () => {
  const ui = setup();
  ui.scrollAt(1, 0.465);
  ui.input('process-progress', 2.25);
  ui.sameScroll(); ui.advance(1000);
  assert.equal(processValue(ui), 2.25);
  ui.toggle(true); ui.input('candidate-cut', 0);
  ui.sameScroll();
  assert.equal(ui.get('candidate-details').open, true);
  ui.scrollAt(1, 0.388);
  assert.equal(ui.get('candidate-details').open, false);
  close(processValue(ui), 1.2);
  ui.click('process-replay'); ui.advance(725);
  close(processValue(ui), 0.5);
  ui.scrollAt(1, 0.6575);
  close(processValue(ui), 2.25);
  assert.equal(ui.get('process-replay').textContent, 'Replay sequence');
  ui.advance(5000);
  close(processValue(ui), 2.25);
});

test('settling a navigation scroll preserves a manual action and resize resynchronizes geometry', () => {
  const ui = setup();
  ui.click('next'); ui.input('process-progress', 1.25);
  ui.advance(1000); ui.sameScroll();
  assert.equal(ui.shell.dataset.scene, '1');
  assert.equal(processValue(ui), 1.25);
  ui.scrollAt(1, 0.465); close(processValue(ui), 1.5);
  ui.input('process-progress', 2.7);
  ui.resize(1000);
  // Raising the reading line by 92 px advances it 92/800 through the step.
  close(processValue(ui), (0.465 + 92 / 800 - 0.08) / 0.77 * 3);
});

test('reduced-motion scrolling advances in completed process phases and rewinds discretely', () => {
  const ui = setup(true);
  for (const [normalized, expected] of [[0.2, 0], [0.4, 1], [0.8, 2], [1, 3], [0.4, 1]]) {
    ui.scrollAt(3, 0.08 + 0.77 * normalized);
    assert.equal(processValue(ui), expected);
  }
});

test('compact scroll progress stays stable when the stage controls change height', () => {
  const ui = setup();
  ui.setCompact(900); ui.setStageHeight(400);
  ui.scrollAt(2, 0.465);
  close(processValue(ui), 1.5);
  // Force a geometry refresh at the same scroll position after controls grow.
  // Their changing height must not move the viewport's reading line.
  ui.setStageHeight(550); ui.resize(900);
  assert.equal(ui.shell.dataset.scene, '2');
  close(processValue(ui), 1.5);
  ui.setStageHeight(350); ui.resize(900);
  close(processValue(ui), 1.5);
  ui.scrollAt(2, 0.2725);
  close(processValue(ui), 0.75);
});



test('later rounds follow scroll, preserve a manual choice, and rewind to F2', () => {
  const ui = setup();
  const at = progress => ui.scrollAt(5, 0.08 + 0.77 * progress);
  at(0.45);
  assert.equal(ui.get('continuation-controls').hidden, false);
  assert.equal(ui.get('continuation-label').textContent, 'After 3 trees');
  assert.match(ui.get('calculation').textContent, /F₃.*MSE = 0\.15$/);
  ui.click('continuation-next');
  assert.equal(ui.get('continuation-label').textContent, 'After 4 trees');
  ui.sameScroll();
  assert.equal(ui.get('continuation-label').textContent, 'After 4 trees');
  at(1);
  assert.match(ui.get('calculation').textContent, /F₈.*MSE = 0\.000048$/);
  assert.equal(ui.get('continuation-next').disabled, true);
  at(0);
  assert.match(ui.get('calculation').textContent, /^F₂.*MSE = 0\.75$/);
  assert.equal(ui.get('continuation-back').disabled, true);
});

test('reduced-motion continuation uses completed fits before the new growth frames', () => {
  const ui = setup(true);
  for (const [progress, round] of [[0.2, 2], [0.45, 3], [0.65, 4], [1, 8], [0.45, 3]]) {
    ui.scrollAt(5, 0.08 + 0.77 * progress);
    assert.equal(ui.get('continuation-label').textContent, `After ${round} trees`);
  }
  ui.scrollAt(6, 0);
  assert.equal(ui.get('continuation-controls').hidden, true);
  assert.equal(ui.get('growth-controls').hidden, true);
  assert.equal(ui.get('process-controls').hidden, true);
  assert.equal(ui.get('visual-heading').textContent, 'Two predictors, one outcome');
  assert.equal(ui.get('stage-count').textContent, '07 / 10');
  ui.scrollAt(7, 0);
  assert.equal(ui.get('continuation-controls').hidden, true);
  assert.equal(ui.get('growth-controls').hidden, false);
  assert.equal(ui.get('growth-next').textContent, 'Next split');
  ui.scrollAt(8, 0);
  assert.equal(ui.get('growth-next').textContent, 'Next stump');
  ui.scrollAt(9, 0);
  assert.equal(ui.get('growth-controls').hidden, true);
  assert.equal(ui.get('continue-boosting'), undefined);
});

test('both growth frames follow and reverse scrolling, preserving manual choices until movement', () => {
  const ui = setup(true);
  for (const scene of [7, 8]) {
    const unit=scene===7?'split':'stump';
    ui.scrollAt(scene, 0);
    assert.equal(ui.get('growth-label').textContent, `0 / 4 ${unit}s`);
    ui.scrollAt(scene, 0.08+0.77*0.625);
    assert.equal(ui.get('growth-label').textContent, `2 / 4 ${unit}s`);
    ui.click('growth-next');
    assert.equal(ui.get('growth-label').textContent, `3 / 4 ${unit}s`);
    ui.sameScroll();
    assert.equal(ui.get('growth-label').textContent, `3 / 4 ${unit}s`);
    ui.scrollAt(scene, 0.9);
    assert.equal(ui.get('growth-label').textContent, `4 / 4 ${unit}s`);
    assert.equal(ui.get('growth-next').disabled, true);
    ui.scrollAt(scene, 0.08+0.77*0.25);
    assert.equal(ui.get('growth-label').textContent, `1 / 4 ${unit}s`);
    ui.click('growth-back');
    assert.equal(ui.get('growth-back').disabled, true);
  }
});

test('growth replay pauses, resumes and yields to scrolling or a different frame', () => {
  const ui=setup();
  ui.scrollAt(7,0);
  ui.click('growth-next'); ui.advance(1100);
  assert.equal(ui.get('growth-label').textContent,'Drawing split 1 of 4');
  ui.click('growth-replay');
  assert.equal(ui.get('growth-replay').textContent,'Resume');
  const paused=ui.get('diagram').innerHTML;
  ui.advance(5000); assert.equal(ui.get('diagram').innerHTML,paused);
  ui.click('growth-replay'); ui.advance(1100);
  assert.equal(ui.get('growth-label').textContent,'1 / 4 splits');
  ui.click('growth-replay'); ui.advance(1100);
  assert.equal(ui.get('growth-replay').textContent,'Pause');
  ui.scrollAt(7,0.9); ui.advance(10000);
  assert.equal(ui.get('growth-label').textContent,'4 / 4 splits');
  assert.equal(ui.get('growth-replay').textContent,'Replay');
  ui.click('growth-replay'); ui.advance(1100);
  ui.scrollAt(8,0); ui.advance(10000);
  assert.equal(ui.get('growth-label').textContent,'0 / 4 stumps');
  assert.equal(ui.get('growth-replay').textContent,'Replay');
});
