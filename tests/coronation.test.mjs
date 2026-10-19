import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { runInNewContext } from 'node:vm';

const controller = readFileSync(new URL('../docs/assets/coronation.js', import.meta.url), 'utf8');
const homepage = readFileSync(new URL('../docs/index.html', import.meta.url), 'utf8');

// Exercise the real controller's clock and lifecycle. Browser checks cover
// proportions, face alignment, and the rendered crown separately.
function setup({ reducedMotion = false, imageLoaded = true, imageBounds = {}, scrollY = 0 } = {}) {
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
  let bounds = { left: 300, top: 180, width: 576, ...imageBounds };
  image.getBoundingClientRect = () => ({
    ...bounds,
    height: bounds.width * 930 / 1692,
    right: bounds.left + bounds.width,
    bottom: bounds.top + bounds.width * 930 / 1692
  });
  const svg = new Element();
  const parts = new Map([
    '.animated-eyes', '.eye-clips', '.descending-crown',
    '.crown-contact-shadow', '.crown-rays-mask',
    '.animated-mouth', '.o-mouth', '.pleased-smile'
  ].map(selector => {
    const part = new Element();
    const markup = homepage.match(new RegExp('<[^>]*class="' + selector.slice(1) + '"[^>]*>'))?.[0];
    const opacity = markup?.match(/\bopacity="([^"]+)"/)?.[1];
    if (opacity !== undefined) part.setAttribute('opacity', opacity);
    return [selector, part];
  }));
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
  Object.assign(window, { IntersectionObserver, innerWidth: 1200, innerHeight: 800, scrollY });
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
    mouth: () => ({
      cover: Number(parts.get('.animated-mouth').attrs.get('opacity')),
      o: Number(parts.get('.o-mouth').attrs.get('opacity')),
      smile: Number(parts.get('.pleased-smile').attrs.get('opacity'))
    }),
    crownPosition() {
      const crown = parts.get('.descending-crown');
      const [, x, y] = crown.attrs.get('transform').match(/^translate\(([-\d.]+) ([-\d.]+)\)$/);
      const scale = bounds.width / 1692;
      return {
        x: Number(x), y: Number(y), scale,
        viewportY: bounds.top + Number(y) * scale,
        pageY: bounds.top + window.scrollY + Number(y) * scale,
        opacity: Number(crown.attrs.get('opacity'))
      };
    },
    setImageBounds(next) { bounds = { ...bounds, ...next }; window.emit('resize'); },
    scrollTo(next) {
      bounds.top += window.scrollY - next;
      window.scrollY = next;
      document.emit('scroll');
    },
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
  assert.equal(ui.mouth().cover, 0, 'the original mouth stays visible while waiting');
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
    ui.advance(4000);
    const mouth = ui.mouth();
    assert.equal(mouth.cover, 1);
    assert.ok(mouth.o > 0 && mouth.o < 1);
    assert.ok(mouth.smile > 0 && mouth.smile < 1);
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
  ui.advance(4900);
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

test('the visible crown enters fully above the page top on desktop, mobile, and scrolled pages', () => {
  for (const geometry of [
    { imageBounds: { top: 180, width: 576 } },
    { imageBounds: { top: 72, width: 320 } },
    { imageBounds: { top: 120, width: 576 }, scrollY: 260 }
  ]) {
    const ui = setup(geometry);
    ui.intersect(true);
    ui.advance(5000);
    ui.advance(420);
    const crown = ui.crownPosition();
    assert.equal(crown.opacity, 1);
    // 176 logo units enclose the crown's tilted artwork, depth, and shadow.
    assert.ok(crown.pageY + 176 * crown.scale < 0, JSON.stringify(geometry));
  }
});

test('the crown travels through the space above the logo and settles without bouncing', () => {
  const ui = setup();
  ui.intersect(true);
  ui.advance(5000);
  ui.advance(420);
  let previous = ui.crownPosition().y;
  ui.advance(1000);
  const entering = ui.crownPosition();
  assert.ok(entering.pageY + 176 * entering.scale > 0, 'the crown has entered the page');
  assert.ok(entering.y < 0, 'the crown travels outside the logo before reaching I');
  assert.ok(entering.y > previous);
  previous = entering.y;
  for (const delta of [1000, 1000, 200, 1280]) {
    ui.advance(delta);
    const crown = ui.crownPosition();
    assert.ok(crown.y >= previous, 'descent remains monotonic');
    assert.ok(crown.y <= 121.5, 'the crown never overshoots its seat');
    previous = crown.y;
  }
  assert.equal(ui.state(), 'crowned');
  assert.equal(ui.crownPosition().y, 121.5);
});

test('scrolling during descent preserves the page-relative path', () => {
  const ui = setup();
  ui.intersect(true);
  ui.advance(5000);
  ui.advance(1420);
  const before = ui.crownPosition();
  ui.scrollTo(260);
  ui.advance(0);
  const after = ui.crownPosition();
  assert.equal(after.y, before.y);
  assert.equal(after.pageY, before.pageY);
  assert.equal(after.viewportY, before.viewportY - 260);
  ui.advance(3480);
  assert.equal(ui.state(), 'crowned');
  assert.equal(ui.crownPosition().y, 121.5);
});

test('a layout change during descent recomputes the page path and retains the same landing', () => {
  const ui = setup();
  const mobile = setup({ imageBounds: { top: 72, width: 320 } });
  for (const scene of [ui, mobile]) {
    scene.intersect(true);
    scene.advance(5000);
    scene.advance(1420);
  }
  const desktopPosition = ui.crownPosition();
  ui.setImageBounds({ top: 72, width: 320 });
  ui.advance(0);
  assert.notEqual(ui.crownPosition().y, desktopPosition.y);
  assert.deepEqual(ui.crownPosition(), mobile.crownPosition());
  ui.advance(3480);
  assert.equal(ui.state(), 'crowned');
  assert.equal(ui.crownPosition().y, 121.5);
});

test('I keeps a quiet O mouth during descent and through the crown landing', () => {
  const ui = setup();
  ui.intersect(true);
  ui.advance(5000);
  ui.advance(1000);
  assert.deepEqual(ui.mouth(), { cover: 1, o: 1, smile: 0 });
  ui.advance(2600);
  assert.deepEqual(ui.mouth(), { cover: 1, o: 1, smile: 0 });
  ui.advance(20);
  assert.equal(ui.crownPosition().y, 121.5);
  assert.deepEqual(ui.mouth(), { cover: 1, o: 1, smile: 0 });
  ui.advance(100);
  assert.deepEqual(ui.mouth(), { cover: 1, o: 1, smile: 0 });
});

test('the O becomes a full smile while the original mouth remains covered', () => {
  const ui = setup();
  ui.intersect(true);
  ui.advance(5000);
  ui.advance(3830);
  let previous = ui.mouth();
  assert.deepEqual(previous, { cover: 1, o: 1, smile: 0 });
  for (const delta of [170, 150, 200, 130, 420]) {
    ui.advance(delta);
    const mouth = ui.mouth();
    assert.equal(mouth.cover, 1, 'the shared patch stays opaque throughout the transition');
    assert.ok(mouth.o <= previous.o && mouth.smile >= previous.smile);
    assert.ok(Math.abs(mouth.o + mouth.smile - 1) < 1e-12, 'one expression replaces the other');
    previous = mouth;
  }
  assert.equal(ui.state(), 'crowned');
  assert.deepEqual(ui.mouth(), { cover: 1, o: 0, smile: 1 });
});

test('initial reduced motion reveals the static final pose after the idle delay', () => {
  const ui = setup({ reducedMotion: true });
  ui.intersect(true);
  ui.advance(4999);
  assert.equal(ui.state(), 'waiting');
  assert.equal(ui.mouth().cover, 0);
  ui.advance(1);
  assert.equal(ui.state(), 'crowned');
  assert.equal(ui.pendingFrames(), 0);
  assert.deepEqual(ui.mouth(), { cover: 1, o: 0, smile: 1 });
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
  assert.deepEqual(ui.mouth(), { cover: 1, o: 1, smile: 0 });
  ui.reduce(true);
  assert.equal(ui.state(), 'crowned');
  assert.equal(ui.pendingFrames(), 0);
  assert.deepEqual(ui.mouth(), { cover: 1, o: 0, smile: 1 });
  const final = ui.visual();
  ui.reduce(false);
  ui.advance(20000);
  assert.equal(ui.state(), 'crowned');
  assert.deepEqual(ui.visual(), final);
});
