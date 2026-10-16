'use strict';

// DOM/scroll integration checks. This does not launch a browser or test layout.
const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const directory = path.resolve(__dirname, '../docs/neural-networks');
const html = fs.readFileSync(path.join(directory, 'index.html'), 'utf8');
const scripts = ['math.js', 'state.js', 'views.js', 'story.js'];
const sceneMarkup = [...html.matchAll(/<section\b([^>]*\bdata-scene="([^"]+)"[^>]*)>([\s\S]*?)<\/section>/g)];
const sceneNames = sceneMarkup.map(match => match[2]);
const svgSnapshots = [];

function attrs(markup) {
  return Object.fromEntries([...markup.matchAll(/([\w-]+)="([^"]*)"/g)].map(match => [match[1], match[2]]));
}

function mount({ mobile = false, reduced = false } = {}) {
  const elements = new Map(), windowEvents = new Map(), documentEvents = new Map(), raf = new Map();
  let nextRaf = 1, renders = 0;
  const runtime = { mobile, reduced, innerHeight: 800, scrollY: 0, scrollCalls: [] };
  const listen = (map, name, callback) => map.set(name, [...(map.get(name) || []), callback]);
  const emit = (map, name, event = {}) => (map.get(name) || []).forEach(callback => callback(event));
  function makeNode(attributes) {
    const node = {
      attrs: { ...attributes }, dataset: {}, style: {}, disabled: false, innerHTML: '', textContent: '',
      events: new Map(), classes: new Set((attributes.class || '').split(/\s+/)),
      setAttribute(name, value) { this.attrs[name] = String(value); },
      removeAttribute(name) { delete this.attrs[name]; },
      getAttribute(name) { return this.attrs[name]; },
      addEventListener(name, callback) { listen(this.events, name, callback); },
      querySelector() { return { textContent: this.heading || '' }; },
      getBoundingClientRect() {
        if (attributes.id === 'stage-shell') {
          const height = runtime.mobile ? 496 : 768, top = runtime.mobile ? 0 : 16;
          return { top, height, bottom: top + height };
        }
        return { top: 0, bottom: 0, height: 0 };
      }
    };
    node.classList = { toggle(name, enabled) { if (enabled) node.classes.add(name); else node.classes.delete(name); } };
    return node;
  }
  for (const match of html.matchAll(/<[^>]+\bid="[^"]+"[^>]*>/g)) {
    const attributes = attrs(match[0]);
    assert(!elements.has(attributes.id), `Duplicate HTML id ${attributes.id}`);
    elements.set(attributes.id, makeNode(attributes));
  }
  const steps = sceneMarkup.map((match, index) => {
    const attributes = attrs(match[1]), node = elements.get(attributes.id);
    node.dataset.scene = match[2];
    node.heading = match[3].match(/<h[23][^>]*>([\s\S]*?)<\/h[23]>/)?.[1].replace(/<[^>]*>/g, '') || match[2];
    node.sceneIndex = index;
    node.getBoundingClientRect = () => {
      let top = 1200;
      for (let i = 0; i < index; i++) top += heightOf(i);
      top -= runtime.scrollY;
      const height = heightOf(index);
      return { top, bottom: top + height, height };
    };
    return node;
  });
  function heightOf(index) { return sceneNames[index] === 'sine-wide' ? 2080 : runtime.mobile ? 624 : 736; }
  const document = {
    documentElement: { scrollHeight: 17000 },
    getElementById(id) { assert(elements.has(id), `Missing referenced HTML id ${id}`); return elements.get(id); },
    querySelectorAll(selector) { assert.equal(selector, '.step[data-scene]'); return steps; },
    addEventListener(name, callback) { listen(documentEvents, name, callback); }
  };
  const window = {
    document,
    get innerHeight() { return runtime.innerHeight; },
    get scrollY() { return runtime.scrollY; },
    matchMedia(query) {
      if (query === '(max-width: 760px)') return { matches: runtime.mobile };
      if (query === '(prefers-reduced-motion: reduce)') return { matches: runtime.reduced };
      throw new Error(`Unexpected media query ${query}`);
    },
    requestAnimationFrame(callback) { const id = nextRaf++; raf.set(id, callback); return id; },
    addEventListener(name, callback) { listen(windowEvents, name, callback); },
    scrollTo(options) {
      runtime.scrollCalls.push(options);
      runtime.scrollY = options.top;
      emit(windowEvents, 'scroll');
    }
  };
  const context = vm.createContext({ window, document, console,
    setTimeout() { throw new Error('Story must not use timers'); },
    setInterval() { throw new Error('Story must not use timers'); },
    clearTimeout() { throw new Error('Story must not use timers'); },
    clearInterval() { throw new Error('Story must not use timers'); }
  });
  for (const script of scripts) {
    vm.runInContext(fs.readFileSync(path.join(directory, script), 'utf8'), context, { filename: script });
    if (script === 'views.js') {
      const actual = window.NNViews.render;
      window.NNViews = { render(...args) { renders++; return actual(...args); } };
    }
  }
  function flush() {
    for (let pass = 0; raf.size && pass < 20; pass++) {
      const callbacks = [...raf.values()]; raf.clear(); callbacks.forEach(callback => callback());
    }
    assert.equal(raf.size, 0, 'Rendering settles without an animation loop');
  }
  function position(scene, progress) {
    const index = sceneNames.indexOf(scene);
    assert(index >= 0);
    const stageBottom = runtime.mobile ? 496 : 784;
    const focus = runtime.mobile ? Math.min(runtime.innerHeight * .85, stageBottom + runtime.innerHeight * .13) : runtime.innerHeight * .5;
    let top = 1200;
    for (let i = 0; i < index; i++) top += heightOf(i);
    runtime.scrollY = top + heightOf(index) * progress - focus;
  }
  return {
    window, runtime, steps, elements,
    get renderCount() { return renders; },
    frame() { return window.NNScrollStory.getFrame(); },
    element(id) { return document.getElementById(id); },
    position,
    scroll(scene, progress = .5) { position(scene, progress); emit(windowEvents, 'scroll'); flush(); return this.frame(); },
    resize() { emit(windowEvents, 'resize'); flush(); },
    click(id) { const node = document.getElementById(id); if (!node.disabled) emit(node.events, 'click'); flush(); },
    key(key, extras = {}) {
      const event = { key, target: { tagName: 'BODY' }, defaultPrevented: false,
        preventDefault() { this.defaultPrevented = true; }, ...extras };
      emit(documentEvents, 'keydown', event); flush(); return event;
    },
    checkSvg(label) {
      for (const id of ['function-plot', 'network-plot']) {
        const node = document.getElementById(id), markup = node.innerHTML;
        assert(markup.includes('<path') || markup.includes('<circle'), `${label}: ${id} draws geometry`);
        assert(!/NaN|undefined|Infinity/.test(markup), `${label}: ${id} contains finite geometry`);
        assert(node.attrs['aria-label'].length > 10, `${label}: ${id} has an accessible description`);
        const viewBox = node.attrs.viewBox.split(/\s+/).map(Number);
        assert.equal(viewBox.length, 4); assert(viewBox.every(Number.isFinite));
        assert(viewBox[2] > 0 && viewBox[3] > 0);
        svgSnapshots.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="${node.attrs.viewBox}">${markup}</svg>`);
      }
    }
  };
}

test('the HTML is scroll driven and its local assets resolve', () => {
  assert.equal(sceneNames.length, 16);
  assert.equal(new Set(sceneNames).size, 16);
  assert(!/<input\b|<select\b|<form\b/i.test(html), 'No form controls substitute for scrolling');
  assert(!/aria-pressed|type="range"/i.test(html), 'No toggles or range sliders');
  for (const match of html.matchAll(/<(?:script|img|link)\b[^>]*(?:src|href)="([^"]+)"[^>]*>/g)) {
    const reference = match[1];
    if (/^(?:[a-z]+:|\/\/|#)/i.test(reference)) continue;
    assert(fs.existsSync(path.resolve(directory, reference.split(/[?#]/)[0])), `Asset exists: ${reference}`);
  }
});

test('scroll alone traverses all 16 scenes and reverses the sine construction', () => {
  const app = mount();
  assert.equal(app.frame().scene, 'affine');
  assert(app.element('previous').disabled);
  const seen = [];
  for (const scene of sceneNames) {
    const frame = app.scroll(scene, scene === 'sine-wide' ? .99 : .5);
    seen.push(frame.scene);
    assert.equal(frame.scene, scene);
    assert.equal(app.element('story').dataset.scene, scene);
    assert.equal(app.steps.filter(step => step.classes.has('is-active')).length, 1);
    assert.equal(app.steps.find(step => step.classes.has('is-active')).attrs['aria-current'], 'step');
    assert.equal(app.element('stage-count').textContent, String(seen.length).padStart(2, '0') + ' / 16');
    if (scene === 'sine-coarse') {
      assert.equal(frame.width, 6);
      assert(!app.element('function-plot').innerHTML.includes('A closer look'));
    }
    if (scene === 'sine-zoom') {
      assert.equal(frame.width, 6);
      assert(app.element('function-plot').innerHTML.includes('A closer look'));
    }
    if (scene === 'sine-joint') assert.equal(frame.width, 7);
    if (scene === 'sine-wide') assert.equal(frame.width, 26);
    app.checkSvg(scene);
  }
  assert.deepEqual(seen, sceneNames);
  assert(app.element('next').disabled);
  const coarse = app.scroll('sine-coarse');
  assert.equal(coarse.width, 6);
  assert.equal(coarse.model.width, 6);
  assert(!app.element('function-plot').innerHTML.includes('A closer look'));
  assert(!app.element('next').disabled);
});

test('mobile scroll keeps updating and renders the compact views', () => {
  const app = mount({ mobile: true });
  for (const scene of sceneNames) {
    const frame = app.scroll(scene, scene === 'sine-wide' ? .99 : .5);
    assert.equal(frame.scene, scene);
    assert.equal(app.element('function-plot').attrs.viewBox, '0 0 620 350');
    assert.equal(app.element('network-plot').attrs.viewBox, '0 0 620 220');
    if (scene === 'sine-wide') assert.equal(frame.width, 26);
    app.checkSvg('compact ' + scene);
  }
  assert.equal(app.scroll('sine-coarse').width, 6);
});

test('resize at unchanged scene/progress forces the compact render', () => {
  const app = mount();
  app.scroll('sine-zoom', .5);
  const previous = app.frame(), before = app.renderCount;
  assert.equal(app.element('function-plot').attrs.viewBox, '0 0 620 390');
  app.runtime.mobile = true;
  app.position('sine-zoom', .5);
  app.resize();
  assert.equal(app.frame().scene, previous.scene);
  assert.equal(app.frame().progress, previous.progress);
  assert.equal(app.renderCount, before + 1);
  assert.equal(app.element('network-plot').attrs.viewBox, '0 0 620 220');
  app.runtime.mobile = false;
  app.position('sine-zoom', .5);
  app.resize();
  assert.equal(app.element('network-plot').attrs.viewBox, '0 0 400 390');
});

test('reduced motion uses finite completed scenes and discrete widening', () => {
  const app = mount({ reduced: true });
  for (const scene of sceneNames) {
    const frame = app.scroll(scene, .53);
    assert.equal(frame.scene, scene);
    if (!['depth-five', 'sine-wide'].includes(scene)) assert.equal(frame.progress, 1);
    else assert(Math.abs(frame.progress * 10 - Math.round(frame.progress * 10)) < 1e-9);
    if (scene === 'sine-joint') {
      assert.equal(frame.width, 7);
      assert(Math.abs(app.window.NNMath.evaluate(frame.model, frame.x) - Math.sin(frame.x)) < 1e-10);
    }
    app.checkSvg('reduced ' + scene);
  }
  assert.equal(app.scroll('depth-five', .99).depth, 5);
  assert.equal(app.scroll('sine-wide', .99).width, 26);
});

test('navigation handlers scroll to adjacent scenes and honor keyboard context', () => {
  const app = mount();
  app.scroll('shallow', .5);
  app.click('next');
  assert.equal(app.frame().scene, 'depth-one');
  assert.equal(app.runtime.scrollCalls.at(-1).behavior, 'smooth');
  app.click('previous');
  assert.equal(app.frame().scene, 'shallow');
  assert(app.key('ArrowRight').defaultPrevented);
  assert.equal(app.frame().scene, 'depth-one');
  assert(app.key('ArrowLeft').defaultPrevented);
  assert.equal(app.frame().scene, 'shallow');
  const before = app.runtime.scrollCalls.length;
  assert(!app.key('ArrowRight', { target: { tagName: 'INPUT' } }).defaultPrevented);
  assert(!app.key('ArrowRight', { ctrlKey: true }).defaultPrevented);
  assert.equal(app.runtime.scrollCalls.length, before);
  const mobile = mount({ mobile: true, reduced: true });
  mobile.scroll('shallow'); mobile.click('next');
  assert.equal(mobile.frame().scene, 'depth-one');
  assert.equal(mobile.runtime.scrollCalls.at(-1).behavior, 'auto');
});

test('all generated SVG samples are balanced and contain valid XML entities', () => {
  // A deliberately small XML well-formedness check keeps the test Node-only.
  // Values/geometry are checked above; these views use only the SVG tags below.
  assert(svgSnapshots.length >= 96);
  for (const markup of svgSnapshots) {
    const stack = [];
    const tokens = markup.match(/<[^>]+>|[^<]+/g) || [];
    for (const token of tokens) {
      if (!token.startsWith('<')) {
        assert(!/&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[\da-f]+);)/i.test(token), 'Text entities are escaped');
        continue;
      }
      if (token.startsWith('</')) {
        assert.equal(stack.pop(), token.match(/^<\/([\w:-]+)/)[1], 'SVG tags are properly nested');
      } else {
        const tag = token.match(/^<([\w:-]+)/)?.[1];
        assert(tag, 'Every SVG token has a valid tag name');
        assert(!/=(?!"[^"]*"|'[^']*')/.test(token), 'SVG attributes are quoted');
        if (!token.endsWith('/>')) stack.push(tag);
      }
    }
    assert.equal(stack.length, 0, 'SVG tags are closed');
  }
});
