'use strict';

// DOM/scroll integration checks. This does not launch a browser or test layout.
const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const directory = path.resolve(__dirname, '../docs/neural-networks');
const html = fs.readFileSync(path.join(directory, 'index.html'), 'utf8');
const scripts = [...html.matchAll(/<script\b[^>]*src="([^"]+)"[^>]*>/g)]
  .map(match => match[1].split(/[?#]/)[0]);
const sceneMarkup = [...html.matchAll(/<section\b([^>]*\bdata-scene="([^"]+)"[^>]*)>([\s\S]*?)<\/section>/g)];
const sceneNames = sceneMarkup.map(match => match[2]);
const svgSnapshots = [];

function attrs(markup) {
  return Object.fromEntries([...markup.matchAll(/([\w-]+)="([^"]*)"/g)].map(match => [match[1], match[2]]));
}

function assertTypesetNode(app, id, source, label) {
  const equation = app.element(id);
  const markup = equation.innerHTML;
  assert.equal(equation.attrs['data-tex'], source, `${label}: the current formula source is exposed`);
  assert(/class="[^"]*\bkatex-html\b/.test(markup), `${label}: visible mathematical typesetting is present`);
  assert(/class="[^"]*\bkatex-mathml\b/.test(markup), `${label}: semantic MathML is present`);
  assert(!/class="[^"]*\bkatex-error\b/.test(markup), `${label}: the formula has no parse error`);
  const annotation = markup.match(/<annotation\b([^>]*)>([\s\S]*?)<\/annotation>/);
  assert(annotation, `${label}: MathML preserves its exact formula source`);
  assert.equal(attrs(annotation[1]).encoding, 'application/x-tex');
  const entities = { amp: '&', lt: '<', gt: '>', quot: '"', '#x27': "'" };
  const annotatedSource = annotation[2].replace(/&(amp|lt|gt|quot|#x27);/g, (_, name) => entities[name]);
  assert.equal(annotatedSource, source, `${label}: the rendered MathML matches the current source`);
  return markup;
}

function assertTypesetEquation(app, frame = app.frame()) {
  if (frame.equationComparison) {
    return ['logistic', 'neural'].map(model => {
      const source = frame.compact && frame[model + 'EquationCompact']
        ? frame[model + 'EquationCompact'] : frame[model + 'Equation'];
      return assertTypesetNode(app, model + '-equation', source, frame.scene + ': ' + model);
    }).join('');
  }
  const source = frame.compact && frame.equationCompact ? frame.equationCompact : frame.equation;
  return assertTypesetNode(app, 'stage-equation', source, frame.scene);
}

function assertComparisonPlots(app, label) {
  const frame = app.frame();
  assert.equal(app.element('comparison-plots').hidden, !frame.comparisonPlots,
    `${label}: the probability and boundary plots follow the scroll reveal`);
  if (!frame.comparisonPlots) return;
  for (const id of ['comparison-probability-plot', 'comparison-boundary-plot']) {
    const node = app.element(id), markup = node.innerHTML;
    assert(markup.includes('<path'), `${label}: ${id} draws model geometry`);
    assert(!/NaN|undefined|Infinity/.test(markup), `${label}: ${id} contains finite geometry`);
    assert(!/class="[^"]*\bkatex-error\b/.test(markup), `${label}: ${id} has no math parse errors`);
    assert(/class="[^"]*\bkatex-html\b/.test(markup), `${label}: ${id} typesets mathematical labels`);
    assert(/class="[^"]*\bkatex-mathml\b/.test(markup), `${label}: ${id} retains semantic MathML`);
    assert(node.attrs['aria-label'].length > 10, `${label}: ${id} has an accessible description`);
    const box = node.attrs.viewBox.split(/\s+/).map(Number);
    assert.equal(box.length, 4);
    assert(box.every(Number.isFinite) && box[2] > 0 && box[3] > 0);
  }
}

function mount({ mobile = false, reduced = false } = {}) {
  const elements = new Map(), windowEvents = new Map(), documentEvents = new Map(), raf = new Map();
  let nextRaf = 1, renders = 0;
  const runtime = { mobile, reduced, innerHeight: 800, scrollY: 0, scrollCalls: [] };
  const listen = (map, name, callback) => map.set(name, [...(map.get(name) || []), callback]);
  const emit = (map, name, event = {}) => (map.get(name) || []).forEach(callback => callback(event));
  function makeNode(attributes) {
    const node = {
      attrs: { ...attributes }, dataset: {}, style: {}, disabled: false, hidden: false,
      value: attributes.value || '', min: attributes.min || '', max: attributes.max || '', step: attributes.step || '',
      tagName: attributes.type ? 'INPUT' : '', innerHTML: '', textContent: '',
      events: new Map(), classes: new Set((attributes.class || '').split(/\s+/)),
      setAttribute(name, value) { this.attrs[name] = String(value); },
      removeAttribute(name) { delete this.attrs[name]; },
      getAttribute(name) { return this.attrs[name]; },
      addEventListener(name, callback) { listen(this.events, name, callback); },
      querySelector(selector) {
        if (selector === 'h2,h3') return { textContent: this.heading || '' };
        return this.querySelectorAll(selector)[0] || null;
      },
      querySelectorAll(selector) {
        const attribute = selector.match(/^\[([\w-]+)\]$/)?.[1];
        assert(attribute, `Unexpected node selector ${selector}`);
        // Retained SVG layers mutate their attributes without replacing their
        // parent. Reflect those mutations in the serialized markup as a browser
        // does, so opacity and re-entry assertions inspect the actual result.
        return [...this.innerHTML.matchAll(/<[^/!][^>]*>/g)]
          .filter(match => Object.hasOwn(attrs(match[0]), attribute))
          .map(match => {
            let tag = match[0];
            return {
              getAttribute(name) { return attrs(tag)[name] ?? null; },
              setAttribute: (name, value) => {
                const pattern = new RegExp('\\b' + name + '="[^"]*"');
                const replacement = name + '="' + String(value).replace(/&/g, '&amp;').replace(/"/g, '&quot;') + '"';
                const updated = pattern.test(tag) ? tag.replace(pattern, replacement) : tag.replace(/>$/, ' ' + replacement + '>');
                this.innerHTML = this.innerHTML.replace(tag, updated);
                tag = updated;
              }
            };
          });
      },
      getBoundingClientRect() {
        if (attributes.id === 'stage-shell' || attributes.id === 'story-stage') {
          const height = runtime.mobile ? 496 : 768, top = runtime.mobile ? 0 : 16;
          return { top, height, bottom: top + height };
        }
        return { top: 0, bottom: 0, height: 0 };
      }
    };
    node.style.setProperty = function(name, value) { this[name] = String(value); };
    node.style.removeProperty = function(name) { const old = this[name] || ''; delete this[name]; return old; };
    // Native textContent replacement removes existing element children (MathML
    // included). Keep that behavior here rather than retaining stale markup.
    let markup = '', text = '';
    Object.defineProperties(node, {
      innerHTML: {
        get() { return markup; },
        set(value) { markup = String(value); text = markup.replace(/<[^>]*>/g, ''); }
      },
      textContent: {
        get() { return text; },
        set(value) {
          text = String(value);
          markup = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        }
      }
    });
    node.classList = {
      toggle(name, enabled) { if (enabled) node.classes.add(name); else node.classes.delete(name); },
      add(...names) { names.forEach(name => node.classes.add(name)); },
      remove(...names) { names.forEach(name => node.classes.delete(name)); },
      contains(name) { return node.classes.has(name); }
    };
    return node;
  }
  for (const match of html.matchAll(/<[^>]+\bid="[^"]+"[^>]*>/g)) {
    const attributes = attrs(match[0]);
    assert(!elements.has(attributes.id), `Duplicate HTML id ${attributes.id}`);
    elements.set(attributes.id, makeNode(attributes));
  }
  for (const match of html.matchAll(/<template\b([^>]*)>([\s\S]*?)<\/template>/g)) {
    const node = elements.get(attrs(match[1]).id);
    if (node) node.innerHTML = match[2];
  }
  const steps = sceneMarkup.map((match, index) => {
    const attributes = attrs(match[1]), node = elements.get(attributes.id);
    node.innerHTML = match[3];
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
  function heightOf(index) {
    return runtime.innerHeight * (sceneNames[index] === 'sigmoid-one' ? 4.2 : sceneNames[index] === 'sine-wide' ? 2.6 : sceneNames[index] === 'hinge' ? 1.8 : 1);
  }
  const document = {
    compatMode: 'CSS1Compat',
    documentElement: { scrollHeight: 17000 },
    getElementById(id) { assert(elements.has(id), `Missing referenced HTML id ${id}`); return elements.get(id); },
    querySelector(selector) {
      if (selector.startsWith('#')) return this.getElementById(selector.slice(1));
      throw new Error(`Unexpected DOM selector ${selector}`);
    },
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
  // In a browser the global object is window; this also gives UMD libraries
  // (including local KaTeX) the same global location as ordinary script code.
  const context = vm.createContext(Object.assign(window, { window, console,
    setTimeout() { throw new Error('Story must not use timers'); },
    setInterval() { throw new Error('Story must not use timers'); },
    clearTimeout() { throw new Error('Story must not use timers'); },
    clearInterval() { throw new Error('Story must not use timers'); }
  }));
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
    const focus = runtime.mobile ? 0 : 16;
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
    input(id, value) {
      const node = document.getElementById(id);
      node.value = String(value);
      assert((node.events.get('input') || []).length > 0, `Missing input handler for ${id}`);
      emit(node.events, 'input', { target: node, currentTarget: node });
      flush();
      return this.frame();
    },
    key(key, extras = {}) {
      const event = { key, target: { tagName: 'BODY' }, defaultPrevented: false,
        preventDefault() { this.defaultPrevented = true; }, ...extras };
      emit(documentEvents, 'keydown', event); flush(); return event;
    },
    checkSvg(label) {
      if (this.frame().equationOnly || this.frame().equationComparison) {
        assert(['conflict', 'sigmoid-one'].includes(this.frame().scene), `${label}: model equations replace both plots`);
        assert.equal(document.getElementById('figure-grid').hidden, true, `${label}: the figure area is hidden`);
        for (const id of ['function-plot', 'network-plot']) {
          assert.equal(document.getElementById(id).innerHTML, '', `${label}: ${id} has no stale diagram`);
        }
        assertTypesetEquation(this);
        if (this.frame().equationComparison) assertComparisonPlots(this, label);
        return;
      }
      assert.equal(document.getElementById('figure-grid').hidden, false, `${label}: plotted scenes retain their figure area`);
      assert.equal(document.getElementById('model-comparison').hidden, true, `${label}: plotted scenes hide the formula comparison`);
      assert.equal(document.getElementById('stage-equation').hidden, false, `${label}: plotted scenes retain their current formula`);
      for (const id of ['function-plot', 'network-plot']) {
        const node = document.getElementById(id), markup = node.innerHTML;
        assert(markup.includes('<path') || markup.includes('<circle'), `${label}: ${id} draws geometry`);
        assert(!/NaN|undefined|Infinity/.test(markup), `${label}: ${id} contains finite geometry`);
        assert(!/class="[^"]*\bkatex-error\b/.test(markup), `${label}: diagram labels have no math parse errors`);
        if (window.NNScrollStory.getFrame().hideInspector) {
          assert(/class="[^"]*\bkatex-html\b/.test(markup), `${label}: mathematical diagram labels use LaTeX`);
          assert(/class="[^"]*\bkatex-mathml\b/.test(markup), `${label}: mathematical diagram labels retain semantic MathML`);
        }
        assert(node.attrs['aria-label'].length > 10, `${label}: ${id} has an accessible description`);
        const viewBox = node.attrs.viewBox.split(/\s+/).map(Number);
        assert.equal(viewBox.length, 4); assert(viewBox.every(Number.isFinite));
        assert(viewBox[2] > 0 && viewBox[3] > 0);
        svgSnapshots.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="${node.attrs.viewBox}">${markup}</svg>`);
      }
    }
  };
}

test('the HTML is scroll driven, has one labelled input inspector, and its local assets resolve', () => {
  assert.equal(sceneNames.length, 26);
  assert.equal(new Set(sceneNames).size, 26);
  assert(!/<select\b|<form\b/i.test(html), 'No form controls substitute for scrolling');
  const inputs = [...html.matchAll(/<input\b[^>]*>/gi)].map(match => attrs(match[0]));
  assert.equal(inputs.length, 1, 'The input inspector is the only input');
  assert.equal(inputs[0].id, 'input-x');
  assert.equal(inputs[0].type, 'range');
  assert(/<label\b[^>]*for="input-x"[^>]*>[\s\S]*?<\/label>/.test(html), 'The x slider has an explicit label');
  assert(inputs[0]['aria-describedby'], 'The slider points to its explanatory hint');
  assert(!/aria-pressed/i.test(html), 'No toggles replace the scroll story');
  const opening = sceneMarkup.find(match => match[2] === 'affine')[3].replace(/<[^>]*>/g, ' ');
  assert(/\bconstant\b/i.test(opening) && /\b(?:not|isn.t)\b/i.test(opening) && /\b(?:input|feature)\b/i.test(opening),
    'The opening distinguishes the constant bias source from an input feature');
  assert(/\b(?:bias|intercept)\b/i.test(opening) && /omit|hidden|hide|simplif|disappear|left out/i.test(opening) &&
    /remain|still|keep|kept|retain|stay|unchanged|included/i.test(opening),
  'The opening explains that hiding the bias drawing retains the intercept');
  for (const match of html.matchAll(/<(?:script|img|link)\b[^>]*(?:src|href)="([^"]+)"[^>]*>/g)) {
    const reference = match[1];
    if (/^(?:[a-z]+:|\/\/|#)/i.test(reference)) continue;
    assert(fs.existsSync(path.resolve(directory, reference.split(/[?#]/)[0])), `Asset exists: ${reference}`);
  }
  const stylesheets = [...html.matchAll(/<link\b[^>]*>/g)]
    .map(match => attrs(match[0]))
    .filter(link => link.rel === 'stylesheet' && !/^(?:[a-z]+:|\/\/)/i.test(link.href));
  for (const stylesheet of stylesheets) {
    const cssPath = path.resolve(directory, stylesheet.href.split(/[?#]/)[0]);
    const css = fs.readFileSync(cssPath, 'utf8');
    for (const match of css.matchAll(/url\(\s*['"]?([^'"\s)]+)['"]?\s*\)/g)) {
      const reference = match[1];
      if (/^(?:[a-z]+:|\/\/|#)/i.test(reference)) continue;
      assert(fs.existsSync(path.resolve(path.dirname(cssPath), reference.split(/[?#]/)[0])),
        `Stylesheet asset exists: ${stylesheet.href} → ${reference}`);
    }
  }
});

test('scroll alone traverses all 26 scenes and reverses the sine construction', () => {
  const app = mount();
  assert.equal(app.frame().scene, 'affine');
  assert(app.element('previous').disabled);
  const seen = [];
  for (const scene of sceneNames) {
    const frame = app.scroll(scene, scene === 'sine-wide' ? .99 : .5);
    seen.push(frame.scene);
    assert.equal(frame.scene, scene);
    assert.notEqual(frame.inspectInput, true, 'Automatic scrolling needs no slider interaction');
    assert.equal(Number(app.element('input-x').value), frame.x);
    assert.equal(app.element('story').dataset.scene, scene);
    assert.equal(app.steps.filter(step => step.classes.has('is-active')).length, 1);
    assert.equal(app.steps.find(step => step.classes.has('is-active')).attrs['aria-current'], 'step');
    assert.equal(app.element('stage-count').textContent, String(seen.length).padStart(2, '0') + ' / 26');
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
    if (scene === 'affine') assert([0, 1].includes(frame.progress), 'Reduced-motion affine scenes use complete visible/hidden bias states');
    else if (scene === 'sigmoid-one') {
      assert.equal(frame.progress, .45, 'Reduced motion keeps the middle comparison beat available');
      assert.equal(frame.comparisonPhase, 'logistic');
    }
    else if (!['width-many', 'sine-wide'].includes(scene)) assert.equal(frame.progress, 1);
    else assert(Math.abs(frame.progress * 10 - Math.round(frame.progress * 10)) < 1e-9);
    if (scene === 'sine-joint') {
      assert.equal(frame.width, 7);
      assert(Math.abs(app.window.NNMath.evaluate(frame.model, frame.x) - Math.sin(frame.x)) < 1e-10);
    }
    app.checkSvg('reduced ' + scene);
  }
  assert.equal(app.scroll('width-many', .99).model.width, 6);
  assert.equal(app.scroll('sine-wide', .99).width, 26);
});

test('navigation handlers scroll to adjacent scenes and honor keyboard context', () => {
  const app = mount();
  app.scroll('shallow', .5);
  app.click('next');
  assert.equal(app.frame().scene, 'width-one');
  assert.equal(app.runtime.scrollCalls.at(-1).behavior, 'smooth');
  app.click('previous');
  assert.equal(app.frame().scene, 'shallow');
  assert(app.key('ArrowRight').defaultPrevented);
  assert.equal(app.frame().scene, 'width-one');
  assert(app.key('ArrowLeft').defaultPrevented);
  assert.equal(app.frame().scene, 'shallow');
  const before = app.runtime.scrollCalls.length;
  assert(!app.key('ArrowRight', { target: { tagName: 'INPUT' } }).defaultPrevented);
  assert(!app.key('ArrowRight', { ctrlKey: true }).defaultPrevented);
  assert.equal(app.runtime.scrollCalls.length, before);
  const mobile = mount({ mobile: true, reduced: true });
  mobile.scroll('shallow'); mobile.click('next');
  assert.equal(mobile.frame().scene, 'width-one');
  assert.equal(mobile.runtime.scrollCalls.at(-1).behavior, 'auto');
});

test('the input inspector changes activations without changing the sine approximation', () => {
  for (const mobile of [false, true]) {
    const app = mount({ mobile });
    const automatic = app.scroll('sine-wide', .99);
    const expectedModel = JSON.stringify(automatic.model);
    const expectedHighlight = automatic.highlightX;
    const baselineProgress = automatic.progress;
    assert.equal(automatic.width, 26);
    assert.notEqual(automatic.inspectInput, true, 'Scrolling alone keeps the automatic input');
    assert.equal(Number(app.element('input-x').min), 0);
    assert(Math.abs(Number(app.element('input-x').max) - 2 * Math.PI) < .001);

    const atZero = app.input('input-x', 0);
    const zeroNetwork = app.element('network-plot').innerHTML;
    const zeroPlot = app.element('function-plot').innerHTML;
    const activeAtZero = app.window.NNMath.contributions(atZero.model, atZero.x).filter(unit => unit.active).length;
    assert.equal(atZero.x, 0);
    assert.equal(atZero.inspectInput, true);
    assert.equal(atZero.scene, 'sine-wide');
    assert.equal(atZero.progress, baselineProgress);
    assert.equal(JSON.stringify(atZero.model), expectedModel);

    const atSix = app.input('input-x', 6);
    const activeAtSix = app.window.NNMath.contributions(atSix.model, atSix.x).filter(unit => unit.active).length;
    assert.equal(atSix.x, 6);
    assert.equal(atSix.inspectInput, true);
    assert.equal(atSix.width, 26);
    assert.equal(atSix.scene, 'sine-wide');
    assert.equal(atSix.progress, baselineProgress);
    assert.equal(atSix.highlightX, expectedHighlight, 'The UAT insertion location stays fixed');
    assert.equal(JSON.stringify(atSix.model), expectedModel, 'Inspection never retrains or changes network weights');
    assert(activeAtSix > activeAtZero, 'The selected x activates more hidden neurons');
    assert.notEqual(app.element('network-plot').innerHTML, zeroNetwork, 'The network drawing responds to x');
    assert.notEqual(app.element('function-plot').innerHTML, zeroPlot, 'The function marker responds to x');
    assert.equal(Number(app.element('input-x').value), 6);
    assert.equal(app.element('input-x').attrs['aria-valuetext'], 'x = 6.00');
    assert.equal(app.element('input-x-value').textContent, '6.00');
    const crosshair = [...app.element('function-plot').innerHTML.matchAll(/<line\b[^>]*>/g)]
      .map(match => attrs(match[0]))
      .find(line => line.stroke === '#007F73' && line['stroke-dasharray'] === '4 4');
    assert(crosshair, 'The selected input has its own dashed crosshair');
    assert(Math.abs(Number(crosshair.x1) - (53 + 6 / (2 * Math.PI) * (589 - 53))) < 1e-8,
      'The crosshair is at the selected x on the plot scale');
    assert.equal(crosshair.x1, crosshair.x2, 'The x crosshair is vertical');
    app.checkSvg(mobile ? 'mobile inspected sine' : 'desktop inspected sine');

    const clamped = app.scroll('shallow', .5);
    assert.equal(Number(app.element('input-x').min), -2);
    assert.equal(Number(app.element('input-x').max), 2);
    assert.equal(clamped.x, 2, 'The chosen input is clamped to the current plot domain');
    assert.equal(clamped.inspectInput, true);
    const returned = app.scroll('sine-wide', .99);
    assert.equal(returned.x, 6, 'Returning to sine restores the desired input, not the clamped value');
    assert.equal(JSON.stringify(returned.model), expectedModel);

    const beforeCalls = app.runtime.scrollCalls.length;
    const event = app.key('ArrowRight', { target: app.element('input-x') });
    assert(!event.defaultPrevented, 'The input keeps its native arrow-key interaction');
    assert.equal(app.runtime.scrollCalls.length, beforeCalls, 'Input arrow keys never change the scene');
    assert.equal(app.frame().scene, 'sine-wide');

    const joint = app.scroll('sine-joint', .5);
    const expectedJoint = app.window.NNStoryState.frame('sine-joint', .5);
    assert.equal(joint.x, 6);
    assert.equal(joint.highlightX, expectedJoint.highlightX);
    assert.equal(JSON.stringify(joint.model), JSON.stringify(expectedJoint.model),
      'Inspection leaves the partially updated UAT weights unchanged');
    app.input('input-x', 0);
    assert.equal(JSON.stringify(app.frame().model), JSON.stringify(expectedJoint.model));
    assert.equal(app.frame().highlightX, expectedJoint.highlightX);

    if (app.elements.has('follow-input')) {
      app.click('follow-input');
      const reset = app.frame();
      assert.notEqual(reset.inspectInput, true);
      assert.equal(reset.x, expectedJoint.x);
      assert.equal(reset.highlightX, expectedJoint.highlightX);
      assert.equal(JSON.stringify(reset.model), JSON.stringify(expectedJoint.model));
      const nextAutomatic = app.scroll('shallow', .5);
      assert.equal(nextAutomatic.x, app.window.NNStoryState.frame('shallow', .5).x);
    }
  }
});

test('the affine bias stays in predictions after its temporary constant node disappears', () => {
  const texts = markup => [...markup.matchAll(/<text\b([^>]*)>([\s\S]*?)<\/text>/g)]
    .map(match => ({ ...attrs(match[1]), text: match[2].replace(/<[^>]*>/g, '') }));
  const functionPath = markup => [...markup.matchAll(/<path\b[^>]*>/g)]
    .map(match => attrs(match[0]))
    .find(item => item.stroke === '#234E70' && item['stroke-width'] === '3.2').d;
  const outputText = (markup, mobile) => texts(markup)
    .find(item => Number(item.x) === (mobile ? 342 : 365) && Number(item.y) === (mobile ? 206 : 244)).text;

  for (const reduced of [false, true]) for (const mobile of [false, true]) {
    const app = mount({ mobile, reduced });
    app.scroll('affine', .2);
    const shown = app.input('input-x', 0);
    assert.equal(shown.kind, 'affine');
    assert.equal(shown.x, 0);
    assert.equal(shown.weight, 1);
    assert.equal(shown.bias, .5);
    assert.equal(shown.showBias, true);
    assert.equal(shown.weight * shown.x + shown.bias, .5);
    const networkShown = app.element('network-plot').innerHTML;
    const plotShown = app.element('function-plot').innerHTML;
    const constant = networkShown.match(/<g\b[^>]*data-bias-input="true"[^>]*>([\s\S]*?)<\/g>/);
    assert(constant, 'The opening network draws a separate constant-bias source');
    assert(texts(constant[1]).some(item => item.text === '1'), 'The source has constant value one');
    assert(texts(constant[1]).some(item => item.text === 'b = 0.5'), 'Its connection displays the intercept weight');
    const faintEdge = attrs(constant[1].match(/<line\b[^>]*>/)[0]);
    assert(Number(faintEdge.opacity) > 0 && Number(faintEdge.opacity) < 1, 'The bias connection is faint');
    assert.equal(outputText(networkShown, mobile), mobile ? '= 0.5' : '0.5');
    assert(texts(plotShown).some(item => /x = 0\s+→\s+ŷ = 0\.5/.test(item.text)),
      'The function plot shows the nonzero prediction at x = 0');
    const shownDescription = app.element('network-plot').attrs['aria-label'];
    assert(shownDescription.includes('At x = 0, the output is 0.5'));
    assert(shownDescription.includes('not an additional input feature'));
    if (!mobile) assert(texts(networkShown).some(item => item.text === '1 input · 1 output'));
    app.checkSvg(mobile ? 'compact affine visible bias' : 'desktop affine visible bias');

    const hidden = app.scroll('affine', .9);
    assert.equal(hidden.x, 0, 'The slider selection persists while the bias drawing is simplified');
    assert.equal(hidden.showBias, false);
    assert.equal(hidden.weight, 1);
    assert.equal(hidden.bias, .5, 'The intercept remains in the actual render frame');
    const networkHidden = app.element('network-plot').innerHTML;
    assert(!networkHidden.includes('data-bias-input'), 'The temporary constant node is omitted');
    assert.equal(functionPath(app.element('function-plot').innerHTML), functionPath(plotShown),
      'Hiding the bias drawing does not change the plotted function');
    assert.equal(outputText(networkHidden, mobile), mobile ? '= 0.5' : '0.5');
    assert(app.element('network-plot').attrs['aria-label'].includes('At x = 0, the output is 0.5'));
    assert(app.element('network-plot').attrs['aria-label'].includes('bias is included in the calculation'));
    assert(texts(app.element('function-plot').innerHTML).some(item => /x = 0\s+→\s+ŷ = 0\.5/.test(item.text)));

    const one = app.input('input-x', 1);
    assert.equal(one.weight * one.x + one.bias, 1.5);
    assert.equal(one.showBias, false);
    assert.equal(outputText(app.element('network-plot').innerHTML, mobile), mobile ? '= 1.5' : '1.5');
    assert(app.element('network-plot').attrs['aria-label'].includes('At x = 1, the output is 1.5'));
    assert(texts(app.element('function-plot').innerHTML).some(item => /x = 1\s+→\s+ŷ = 1\.5/.test(item.text)));
    assert.equal(functionPath(app.element('function-plot').innerHTML), functionPath(plotShown),
      'The slider traces the same affine curve rather than changing its weights');
    app.checkSvg(mobile ? 'compact affine hidden bias' : 'desktop affine hidden bias');
  }
});

test('equations use local KaTeX assets and the affine formula retains its accessible precompiled math', () => {
  const expectedTex = String.raw`\hat{y}=wx+b=1\times x+0.5`;
  const references = [...html.matchAll(/<script\b[^>]*src="([^"]+)"[^>]*>/g)].map(match => match[1].split(/[?#]/)[0]);
  assert(references.includes('formula.js'));
  const runtime = references.find(reference => /katex(?:\.min)?\.js$/.test(reference));
  assert(runtime, 'The KaTeX runtime loads for changing formulas');
  assert(!/^(?:[a-z]+:|\/\/)/i.test(runtime), 'Math typesetting is available without a remote script');
  assert(references.indexOf(runtime) < references.indexOf('formula.js'), 'KaTeX loads before the formula renderer');
  assert(references.indexOf('formula.js') < references.indexOf('story.js'), 'The formula renderer loads before the controller');
  const katexStylesheet = [...html.matchAll(/<link\b[^>]*>/g)].map(match => attrs(match[0]))
    .find(link => link.rel === 'stylesheet' && /katex/i.test(link.href));
  assert(katexStylesheet, 'The visible KaTeX HTML has its local stylesheet');
  assert(!/^(?:[a-z]+:|\/\/)/i.test(katexStylesheet.href), 'Math typesetting is available without a remote stylesheet');
  const katexCssPath = path.resolve(directory, katexStylesheet.href.split(/[?#]/)[0]);
  const katexCss = fs.readFileSync(katexCssPath, 'utf8');
  assert(/@font-face/.test(katexCss), 'The KaTeX stylesheet provides the required mathematical fonts');
  assert(/url\([^)]*\.woff2(?:[?)]|["'])/.test(katexCss), 'KaTeX uses locally served web fonts');
  const mathmlRule = katexCss.match(/\.katex\s+\.katex-mathml\s*\{([^}]*)\}/)?.[1] || '';
  assert(/clip\s*:/.test(mathmlRule) && /position\s*:\s*absolute/.test(mathmlRule),
    'The semantic MathML is visually clipped while the HTML branch remains visible');
  assert(!/display\s*:\s*none|visibility\s*:\s*hidden/.test(mathmlRule),
    'The semantic expression is retained for assistive technology');
  const initialEquation = html.match(/<p\b[^>]*id="stage-equation"[^>]*>([\s\S]*?)<\/p>/)?.[1] || '';
  assert(/class="[^"]*\bkatex-html\b/.test(initialEquation), 'The initial HTML includes visible precompiled math');
  assert(/class="[^"]*\bkatex-mathml\b/.test(initialEquation), 'The initial HTML includes semantic MathML');
  for (const mobile of [false, true]) {
    const app = mount({ mobile });
    app.scroll('affine', .2);
    const equation = app.element('stage-equation');
    const markup = equation.innerHTML;
    assert.equal(app.window.NNFormula.affineTex, expectedTex);
    assert.equal(markup, app.window.NNFormula.affineMarkup);
    assert.equal(equation.attrs['data-tex'], expectedTex);
    const visible = [...markup.matchAll(/<span\b([^>]*)>/g)].map(match => attrs(match[1]))
      .find(span => /(?:^|\s)katex-html(?:\s|$)/.test(span.class || ''));
    assert(visible, 'KaTeX provides a visible HTML typesetting branch');
    assert.equal(visible['aria-hidden'], 'true', 'The visual branch avoids duplicate screen-reader output');
    assert(/class="[^"]*\bkatex-mathml\b/.test(markup), 'The same expression has an accessible MathML branch');
    assert(/class="[^"]*\baccent\b/.test(markup), 'The visible HTML constructs the prediction hat accent');
    const math = markup.match(/<math\b([^>]*)>/);
    assert(math, 'The equation retains a semantic MathML math element');
    const mathAttrs = attrs(math[1]);
    assert.equal(mathAttrs.xmlns, 'http://www.w3.org/1998/Math/MathML');
    assert(mathAttrs['aria-label'].length > 20, 'The equation has an accessible spoken label');
    assert(/<mover\b[^>]*>\s*<mi\b[^>]*>y<\/mi>\s*<mo\b[^>]*>\^<\/mo>\s*<\/mover>/.test(markup),
      'The prediction y has a typeset hat accent');
    for (const variable of ['w', 'x', 'b']) {
      assert(new RegExp('<mi\\b[^>]*>' + variable + '<\\/mi>').test(markup), `${variable} is a mathematical identifier`);
    }
    assert(/<mo\b[^>]*>×<\/mo>/.test(markup), 'The fixed input weight uses a multiplication operator');
    const annotation = markup.match(/<annotation\b([^>]*)>([\s\S]*?)<\/annotation>/);
    assert(annotation);
    assert.equal(attrs(annotation[1]).encoding, 'application/x-tex');
    assert.equal(annotation[2], expectedTex);

    app.input('input-x', 0);
    assert.equal(equation.innerHTML, markup, 'Tracing the input preserves the symbolic equation');
    app.scroll('affine', .9);
    assert.equal(app.frame().showBias, false);
    assert.equal(equation.innerHTML, markup, 'Omitting the bias node preserves the full affine equation');
    assert.equal(equation.attrs['data-tex'], expectedTex);
    app.runtime.mobile = !mobile;
    app.position('affine', .9);
    app.resize();
    assert.equal(equation.innerHTML, markup, 'The same equation survives responsive rerendering');

    const relu = app.scroll('relu', .5);
    assertTypesetEquation(app, relu);
    assert.notEqual(equation.innerHTML, markup, 'The next scene replaces both branches with its own formula');
    app.scroll('affine', .2);
    assert.equal(equation.innerHTML, markup, 'Reverse scrolling restores the typeset affine formula');
    assert.equal(equation.attrs['data-tex'], expectedTex);
  }
});

test('every scene equation keeps exact TeX, visible math, and semantic MathML while scrolling in both directions', () => {
  for (const mobile of [false, true]) {
    const app = mount({ mobile });
    for (const scene of [...sceneNames, ...sceneNames.slice().reverse()]) {
      const frame = app.scroll(scene, .5);
      assertTypesetEquation(app, frame);
    }
    app.scroll('shallow', .5);
    const shallow = app.element('stage-equation').innerHTML;
    assert(/<mover\b/.test(shallow), 'The prediction has a mathematical hat');
    assert(/<msub\b/.test(shallow), 'Neuron indices are mathematical subscripts');
    assert(/<mo\b[^>]*>∑<\/mo>/.test(shallow), 'The network sum has a mathematical summation operator');
    const universal = app.scroll('universal', .5);
    assert(universal.equation.includes(String.raw`\max_{x\in[a,b]}`), 'The approximation bound is restricted to the stated interval');
  }
});

test('typeset coefficients follow the model and compact formulas update correctly after resizing', () => {
  const rounded = value => Number(value.toFixed(2)).toString();
  for (const mobile of [false, true]) {
    const app = mount({ mobile });
    let previous = '';
    for (const progress of [.05, .55, .95]) {
      const frame = app.scroll('hinge', progress);
      const expected = String.raw`\hat{y}=${rounded(frame.hingeA)}\,\operatorname{ReLU}(x${frame.hingeT < 0 ? '+' : '-'}${rounded(Math.abs(frame.hingeT))})`;
      assert.equal(frame.equation, expected, 'The hinge formula uses its current slope and joint');
      const markup = assertTypesetEquation(app, frame);
      assert.notEqual(markup, previous, 'Scrolling updates the visible coefficient');
      previous = markup;
      app.input('input-x', 1);
      assert.equal(app.element('stage-equation').innerHTML, markup, 'Tracing an input retains the current model formula');
    }
    for (const progress of [.15, .85]) {
      const frame = app.scroll('hill-fall', progress);
      const coefficient = rounded(-frame.model.neurons[1].a);
      assert.equal(frame.equation, String.raw`\hat{y}=\operatorname{ReLU}(x+1)-${coefficient}\,\operatorname{ReLU}(x)`);
      assertTypesetEquation(app, frame);
    }
    for (const scene of ['hill-flat', 'width-two', 'universal']) {
      const before = app.scroll(scene, .5);
      assert(before.equationCompact && before.equationCompact !== before.equation, `${scene}: a compact formula is available`);
      const previousMarkup = assertTypesetEquation(app, before);
      app.runtime.mobile = !app.runtime.mobile;
      app.position(scene, .5);
      app.resize();
      const after = app.frame();
      assert.equal(after.compact, app.runtime.mobile);
      const resizedMarkup = assertTypesetEquation(app, after);
      assert.notEqual(resizedMarkup, previousMarkup, `${scene}: resizing switches the typeset formula layout`);
    }
  }
});

test('the active-neuron counter follows positive hidden activations and excludes the affine bias constant', () => {
  for (const mobile of [false, true]) {
    const app = mount({ mobile });
    const counter = app.element('active-neurons');
    function expectCount(active, total) {
      const frame = app.frame();
      assert.equal(frame.activeCount, active);
      assert.equal(frame.hiddenCount, total);
      assert.equal(counter.hidden, total === 0);
      assert.equal(counter.textContent, total === 0 ? '' :
        `${active} of ${total} neurons active · x = ${frame.x.toFixed(2)}`);
    }

    app.scroll('affine', .2);
    assert(app.element('network-plot').innerHTML.includes('data-bias-input'));
    expectCount(0, 0);
    app.input('input-x', 1);
    expectCount(0, 0);
    app.scroll('affine', .9);
    expectCount(0, 0);

    app.scroll('relu', .5);
    expectCount(1, 1);
    app.input('input-x', -1);
    expectCount(0, 1);
    app.input('input-x', 0);
    expectCount(0, 1);
    // Synthetic precise values exercise the activation tolerance, independently
    // of the slider's coarser visual step size.
    app.input('input-x', 1e-9);
    expectCount(0, 1);
    app.input('input-x', 1e-7);
    expectCount(1, 1);

    app.scroll('shallow', .5);
    app.input('input-x', 0);
    expectCount(1, 3);
    app.input('input-x', 1);
    expectCount(2, 3);
    app.input('input-x', 2);
    expectCount(3, 3);
    assert.equal(app.window.NNMath.evaluate(app.frame().model, 2), 0,
      'All three activations count even when their weighted contributions cancel');

    app.scroll('width-many', 0);
    expectCount(3, 3);
    app.scroll('width-many', .99);
    expectCount(6, 6);

    app.scroll('sine-coarse', .5);
    app.input('input-x', 6);
    expectCount(6, 6);
    app.scroll('sine-joint', .5);
    expectCount(7, 7);
    app.scroll('sine-wide', 0);
    expectCount(7, 7);
    app.scroll('sine-wide', .99);
    expectCount(26, 26);
    app.input('input-x', 0);
    expectCount(1, 26); // The always-active starting-slope ReLU is a real neuron.
    app.input('input-x', 6);
    expectCount(26, 26);

    app.scroll('affine', .2);
    expectCount(0, 0);
    assert(app.element('network-plot').innerHTML.includes('data-bias-input'),
      'The visible bookkeeping constant is still excluded after leaving the wider network');
  }
});

test('the single stage retains the scene caption and accessible scene status without an explanation column', () => {
  assert(!/\bid="(?:narrative-display|narrative-caption)"/.test(html), 'The right-hand explanation column is removed');
  assert(!/<script\b[^>]*src="[^"\n]*narrative\.js/.test(html), 'No removed explanation controller is loaded');
  for (const mobile of [false, true]) {
    const app = mount({ mobile });
    const caption = app.element('stage-caption');
    const status = app.element('story-status');
    const rails = app.element('scroll-steps');
    assert(app.element('story').classes.has('is-enhanced'));
    assert.equal(rails.attrs['aria-hidden'], 'true', 'The scroll rails do not duplicate the active scene for assistive technology');
    assert('inert' in rails.attrs || rails.inert === true, 'Hidden scroll-rail links cannot receive focus');
    assert.equal(status.attrs.role, 'status');
    assert.equal(status.attrs['aria-live'], 'polite');
    for (const scene of sceneNames) {
      const frame = app.scroll(scene, .5);
      assert.equal(caption.textContent, frame.caption, `The ${scene} stage retains its current short caption`);
      const intro = app.element('stage-intro');
      assert.equal(intro.textContent, frame.intro || '');
      assert.equal(intro.hidden, !frame.intro, 'The introduction appears only in its scene');
      assert.equal(caption.hidden, !frame.caption, 'Only nonempty captions appear below the visual');
      assert((intro.textContent || caption.textContent).trim().length > 0);
      assert.notEqual(caption.attrs['aria-hidden'], 'true');
      assert(!('inert' in caption.attrs), 'The stage caption remains accessible');
      assert.equal(status.dataset.scene, scene);
      assert.equal(status.textContent, app.steps.find(step => step.dataset.scene === scene).heading);
      assert.equal(app.element('stage-name').textContent, status.textContent, 'The visible title identifies the current step');
      assert.equal(app.element('return-link').hidden, scene !== 'class-compare', 'The return link appears only in the final scene');
    }
    app.scroll('sine-zoom', .5);
    const before = caption.textContent;
    app.input('input-x', 0);
    assert.equal(caption.textContent, before, 'Inspecting an input preserves the scene caption');
    app.runtime.mobile = !mobile;
    app.position('sine-zoom', .5);
    app.resize();
    assert.equal(caption.textContent, before, 'Responsive rendering preserves the active caption');
    const earlier = app.scroll('shallow', .5);
    assert.equal(caption.textContent, earlier.caption, 'Reverse scrolling restores the earlier caption');
  }
});

test('the affine stage caption follows the bias drawing in both directions', () => {
  for (const mobile of [false, true]) {
    for (const reduced of [false, true]) {
      const app = mount({ mobile, reduced });
      const caption = app.element('stage-caption');
      const visible = app.scroll('affine', .2);
      assert.equal(visible.showBias, true);
      assert.equal(caption.textContent, visible.caption);
      assert(caption.textContent.includes('constant-1'));
      const shownText = caption.textContent;
      const omitted = app.scroll('affine', .9);
      assert.equal(omitted.showBias, false);
      assert.equal(caption.textContent, omitted.caption);
      assert(/intercept.*(?:still|remain).*0\.5/i.test(caption.textContent), 'The hidden drawing retains the same numerical intercept');
      assert(/connection.*hidden|hidden.*connection/i.test(caption.textContent), 'The caption distinguishes the hidden connection from the retained intercept');
      assert.notEqual(caption.textContent, shownText);
      const omittedText = caption.textContent;
      app.input('input-x', 1);
      assert.equal(caption.textContent, omittedText, 'Inspecting an input retains the omitted-bias explanation');
      app.scroll('affine', .2);
      assert.equal(caption.textContent, shownText, 'Reverse scrolling restores the visible-bias caption');
    }
  }
});

test('the universal-approximation assumptions remain present before the classification extension', () => {
  for (const mobile of [false, true]) {
    const app = mount({ mobile });
    const frame = app.scroll('universal', .9);
    assert(frame.caption.includes('continuous function on a closed, bounded interval'));
    assert(/does not guarantee (?:successful training|training success)/.test(frame.caption));
    assert.equal(app.element('return-link').hidden, true, 'The approximation is no longer the last scene');
    assert.equal(app.element('stage-instruction').hidden, false, 'The reader can continue into classification');
    app.click('next');
    assert.equal(app.frame().scene, 'conflict');
  }
});

test('the historical sigmoid equation leads to a model comparison before plotted flexibility, in either direction', () => {
  for (const mobile of [false, true]) for (const reduced of [false, true]) {
    const app = mount({ mobile, reduced });
    app.scroll('universal', .5);
    app.checkSvg('before historical equation');
    app.click('next');
    const historical = app.frame();
    assert.equal(historical.scene, 'conflict');
    assert.equal(historical.equationOnly, true);
    assert.equal(app.element('figure-grid').hidden, true);
    assert.equal(app.element('model-comparison').hidden, true);
    assert.equal(app.element('stage-equation').hidden, false);
    assert(app.element('stage-shell').classes.has('is-equation-only'));
    assert.equal(app.element('input-inspector').hidden, true);
    const equationMarkup = assertTypesetEquation(app, historical);
    const equationSource = app.element('stage-equation').attrs['data-tex'];
    assert(equationSource.includes(String.raw`Y_i`) && equationSource.includes(String.raw`\operatorname{Bernoulli}(\pi_i)`),
      'Equation 3 retains the Bernoulli outcome model');
    assert(equationSource.includes(String.raw`\operatorname{sigmoid}`), 'The activation uses its requested function name');
    assert(equationSource.includes(String.raw`\operatorname{sigmoid}(z)=\frac{1}{1+e^{-z}}`),
      'The named activation retains the inverse-logit definition');
    assert(!equationSource.includes(String.raw`\operatorname{expit}`), 'Sigmoid replaces the expit name in the equation');
    assert(!equationSource.includes(String.raw`\sigma`), 'The historical equation uses a named function rather than sigma');
    assert(equationSource.includes(String.raw`\gamma_0`), 'The output intercept remains explicit');
    if (mobile) {
      assert(equationSource.includes(String.raw`\sum_{j=1}^{M}\gamma_j\operatorname{sigmoid}(X_i\beta_{(j)})`),
        'The compact equation includes every indexed hidden neuron');
    } else {
      for (const index of ['1', '2', 'M']) {
        assert(equationSource.includes(String.raw`\gamma_` + index + String.raw`\operatorname{sigmoid}(X_i\beta_{(` + index + ')})'),
          `The expanded equation includes the paper's hidden term ${index}`);
      }
    }
    app.checkSvg('historical equation');

    app.click('next');
    const comparison = app.frame();
    assert.equal(comparison.scene, 'sigmoid-one');
    assert.equal(comparison.equationComparison, true);
    assert.equal(comparison.comparisonPhase, 'equations');
    assert.equal(app.element('figure-grid').hidden, true, 'The comparison uses its own plot area');
    assert.equal(app.element('comparison-plots').hidden, true, 'Both equations appear before any plot is revealed');
    assert.equal(app.element('stage-equation').hidden, true, 'The comparison uses its own two formulas');
    assert.equal(app.element('model-comparison').hidden, false);
    assert.equal(app.element('input-inspector').hidden, true);
    const comparisonMarkup = assertTypesetEquation(app, comparison);
    const logistic = app.element('logistic-equation').attrs['data-tex'];
    const neural = app.element('neural-equation').attrs['data-tex'];
    for (const source of [logistic, neural]) {
      assert(source.includes(String.raw`\operatorname{Bernoulli}(\pi_i)`), 'Both models predict the same Bernoulli outcome');
      assert(source.includes(String.raw`\operatorname{sigmoid}`), 'Both models use the same named sigmoid function');
    }
    assert(logistic.includes(String.raw`X_i\beta`), 'Logistic regression acts directly on the linear predictor');
    assert.equal(logistic.match(/\\operatorname\{sigmoid\}/g).length, 1, 'Logistic regression has no hidden sigmoid layer');
    assert(!logistic.includes(String.raw`\gamma`), 'Logistic regression does not acquire hidden-layer output weights');
    assert(neural.includes(String.raw`\gamma_0`) && neural.includes(String.raw`\gamma_j`) && neural.includes(String.raw`X_i\beta_{(j)}`),
      'The neural-network comparison preserves its hidden features and output intercept');
    assert(neural.includes(String.raw`\sum_{j=1}^{M}`), 'The comparison includes M hidden neurons');
    assert.equal(neural.match(/\\operatorname\{sigmoid\}/g).length, 2, 'The network applies sigmoid at the hidden layer and the output');
    app.checkSvg('model comparison');

    app.click('next');
    assert.equal(app.frame().scene, 'sigmoid-one');
    assert.equal(app.frame().comparisonPhase, 'logistic');
    assert.equal(app.element('comparison-plots').hidden, false);
    app.checkSvg('logistic comparison plots');
    app.click('next');
    assert.equal(app.frame().scene, 'sigmoid-one');
    assert.equal(app.frame().comparisonPhase, 'neural');
    app.checkSvg('neural comparison plots');
    app.click('next');
    assert.equal(app.frame().scene, 'sigmoid-two');
    assert.equal(app.element('figure-grid').hidden, false);
    assert.equal(app.element('model-comparison').hidden, true);
    assert.equal(app.element('stage-equation').hidden, false);
    assert(!app.element('stage-shell').classes.has('is-equation-only'));
    app.checkSvg('flexibility after comparison');
    assert.notEqual(assertTypesetEquation(app), equationMarkup, 'The flexibility scene displays its own formula');

    app.click('previous');
    assert.equal(app.frame().scene, 'sigmoid-one');
    assert.equal(app.frame().comparisonPhase, 'neural', 'Reverse navigation enters the last comparison beat');
    assert.equal(app.element('model-comparison').hidden, false);
    assert.equal(app.element('stage-equation').hidden, true);
    assert.equal(assertTypesetEquation(app), comparisonMarkup, 'Reverse navigation restores both comparison equations exactly');
    app.checkSvg('reversed comparison');
    app.click('previous');
    assert.equal(app.frame().comparisonPhase, 'logistic');
    app.click('previous');
    assert.equal(app.frame().comparisonPhase, 'equations');
    assert.equal(app.element('comparison-plots').hidden, true, 'Reverse navigation hides the plots before leaving the comparison');
    app.click('previous');
    assert.equal(app.frame().scene, 'conflict');
    assert.equal(app.element('model-comparison').hidden, true);
    assert.equal(app.element('stage-equation').hidden, false);
    assert.equal(assertTypesetEquation(app), equationMarkup, 'Reverse navigation restores Equation 3 exactly');
    assert(app.element('stage-shell').classes.has('is-equation-only'));
    app.checkSvg('reversed historical equation');
    app.click('previous');
    assert.equal(app.frame().scene, 'universal');
    assert(!app.element('stage-shell').classes.has('is-equation-only'));
    assert.equal(app.element('input-inspector').hidden, false);
    app.checkSvg('reversed before historical equation');
  }
});

test('the equation comparison unfolds into logistic and neural plots, then reverses exactly', () => {
  for (const mobile of [false, true]) for (const reduced of [false, true]) {
    const app = mount({ mobile, reduced });
    const snapshots = new Map();
    const stops = [[.12, 'equations', 0, 0], [.45, 'logistic', 1, 0], [.9, 'neural', 1, 1]];
    for (const [progress, phase, open, mix] of stops) {
      const frame = app.scroll('sigmoid-one', progress);
      assert.equal(frame.comparisonPhase, phase);
      assert.equal(frame.comparisonOpen, open);
      assert.equal(frame.comparisonMix, mix);
      assert.equal(frame.comparisonPlotOpacity, open);
      assert.equal(frame.comparisonPlots, Boolean(open));
      assert.equal(Number(app.element('model-comparison').style['--comparison-open']), open,
        'The formulas move to their plotted layout only after the opening comparison');
      assert.equal(Number(app.element('model-comparison').style['--logistic-focus']), 1 - open * mix,
        'The logistic equation recedes when the network takes focus');
      assert.equal(Number(app.element('model-comparison').style['--neural-focus']), 1 - open * (1 - mix),
        'The network equation comes forward with its flexible plots');
      assert.equal(app.element('stage-count').textContent, '18 / 26', 'All comparison beats remain on slide 18');
      assert.equal(app.element('input-inspector').hidden, true, 'Scrolling controls the entire comparison');
      app.checkSvg(`${phase} ${mobile ? 'mobile' : 'desktop'} ${reduced ? 'reduced' : 'animated'}`);
      snapshots.set(phase, {
        formulas: assertTypesetEquation(app),
        probability: app.element('comparison-probability-plot').innerHTML,
        boundary: app.element('comparison-boundary-plot').innerHTML,
        caption: app.element('stage-caption').textContent
      });
    }
    for (const [progress, phase] of stops.slice().reverse()) {
      app.scroll('sigmoid-one', progress);
      const saved = snapshots.get(phase);
      assert.equal(assertTypesetEquation(app), saved.formulas, 'Reverse scrolling restores the comparison formulas');
      assert.equal(app.element('comparison-probability-plot').innerHTML, saved.probability,
        'Reverse scrolling restores the exact one-input model display');
      assert.equal(app.element('comparison-boundary-plot').innerHTML, saved.boundary,
        'Reverse scrolling restores the exact two-input model display');
      assert.equal(app.element('stage-caption').textContent, saved.caption);
    }
    app.scroll('sigmoid-one', .9);
    app.scroll('sigmoid-two', .5);
    assert.equal(app.element('model-comparison').hidden, true);
    for (const id of ['comparison-probability-plot', 'comparison-boundary-plot']) {
      assert.equal(app.element(id).innerHTML, '', 'Leaving slide 18 clears its diagrams');
    }
  }
});

test('comparison animation has intermediate scroll frames while reduced motion retains all three beats', () => {
  for (const mobile of [false, true]) {
    const app = mount({ mobile });
    const opening = app.scroll('sigmoid-one', .275);
    assert(opening.comparisonOpen > 0 && opening.comparisonOpen < 1, 'The plot introduction has intermediate positions');
    assert.equal(opening.comparisonMix, 0, 'The first introduction focuses on logistic regression');
    const transition = app.scroll('sigmoid-one', .69);
    assert.equal(transition.comparisonOpen, 1);
    assert(transition.comparisonMix > 0 && transition.comparisonMix < 1,
      'Scrolling crossfades the two model displays through intermediate states');
    app.checkSvg('mid-transition');
    const before = app.element('comparison-boundary-plot').innerHTML;
    app.scroll('sigmoid-one', .91);
    app.scroll('sigmoid-one', .69);
    assert.equal(app.element('comparison-boundary-plot').innerHTML, before,
      'Intermediate graph frames are reversible without elapsed-time animations');
    const reduced = mount({ mobile, reduced: true });
    for (const [progress, expected, phase] of [[.05, .12, 'equations'], [.27, .45, 'logistic'], [.69, .9, 'neural']]) {
      const frame = reduced.scroll('sigmoid-one', progress);
      assert.equal(frame.progress, expected);
      assert.equal(frame.comparisonPhase, phase);
      assert([0, 1].includes(frame.comparisonOpen));
      assert([0, 1].includes(frame.comparisonMix));
      reduced.checkSvg('reduced ' + phase);
    }
  }
});

test('the comparison changes its model curves and boundaries while the two-class observations stay fixed', () => {
  const points = markup => [...markup.matchAll(/<g\b[^>]*data-comparison-point-id="[^"]+"[^>]*>/g)]
    .map(match => attrs(match[0]));
  const layers = markup => [...markup.matchAll(/<g\b[^>]*data-comparison-model="[^"]+"[^>]*>/g)]
    .map(match => attrs(match[0]));
  for (const mobile of [false, true]) {
    const app = mount({ mobile });
    app.scroll('sigmoid-one', .45);
    const initial = new Map();
    for (const [id, pathAttribute] of [
      ['comparison-probability-plot', 'data-comparison-curve'],
      ['comparison-boundary-plot', 'data-comparison-boundary']
    ]) {
      const markup = app.element(id).innerHTML;
      const observations = points(markup);
      assert(observations.length > 10, 'Each view shows a genuine set of fixed binary observations');
      assert.equal(new Set(observations.map(p => p['data-comparison-point-id'])).size, observations.length);
      assert.deepEqual([...new Set(observations.map(p => p['data-label']))].sort(), ['0', '1']);
      assert(observations.every(p => /^translate\([\d\s.,-]+\)$/.test(p.transform)),
        'Each point has an explicit finite plot position');
      const curves = [...markup.matchAll(/<path\b[^>]*>/g)].map(match => attrs(match[0]))
        .filter(p => p[pathAttribute]);
      assert.equal(curves.length, 2, 'Both model geometries are retained for the crossfade');
      assert.deepEqual(curves.map(p => p[pathAttribute]).sort(), ['logistic', 'network']);
      assert.notEqual(curves[0].d, curves[1].d, 'The network supplies a different curve or decision boundary');
      assert.deepEqual(layers(markup).map(layer => [layer['data-comparison-model'], Number(layer.opacity)]),
        [['logistic', 1], ['network', 0]], 'Logistic regression begins in the foreground');
      initial.set(id, observations);
    }
    for (const progress of [.69, .9]) {
      const frame = app.scroll('sigmoid-one', progress);
      for (const id of initial.keys()) {
        const markup = app.element(id).innerHTML;
        assert.deepEqual(points(markup), initial.get(id), 'Model comparison preserves point identities, coordinates, shapes, and class labels');
        const modelLayers = layers(markup);
        assert.equal(Number(modelLayers[0].opacity), 1 - frame.comparisonMix);
        assert.equal(Number(modelLayers[1].opacity), frame.comparisonMix);
      }
    }
  }
});

test('arrow keys visit every comparison beat before advancing to the next scene', () => {
  for (const mobile of [false, true]) for (const reduced of [false, true]) {
    const app = mount({ mobile, reduced });
    app.scroll('conflict', .5);
    for (const phase of ['equations', 'logistic', 'neural']) {
      assert(app.key('ArrowRight').defaultPrevented);
      assert.equal(app.frame().scene, 'sigmoid-one');
      assert.equal(app.frame().comparisonPhase, phase);
      assert.equal(app.runtime.scrollCalls.at(-1).behavior, reduced ? 'auto' : 'smooth');
    }
    app.key('ArrowRight');
    assert.equal(app.frame().scene, 'sigmoid-two');
    for (const phase of ['neural', 'logistic', 'equations']) {
      assert(app.key('ArrowLeft').defaultPrevented);
      assert.equal(app.frame().scene, 'sigmoid-one');
      assert.equal(app.frame().comparisonPhase, phase);
    }
    app.key('ArrowLeft');
    assert.equal(app.frame().scene, 'conflict');
  }
});

test('the final stage caption and return link remain available at and beyond the end of the story', () => {
  for (const mobile of [false, true]) {
    for (const reduced of [false, true]) {
      const app = mount({ mobile, reduced });
      const caption = app.element('stage-caption');
      const link = app.element('return-link');
      assert.equal(link.attrs.href, '#page-title', 'The final link returns to the page heading');
      assert.equal(app.element('page-title').attrs.id, 'page-title', 'The return destination exists');
      for (const progress of [.12, .5, .96, 1, 1.2]) {
        const frame = app.scroll('class-compare', progress);
        assert.equal(frame.scene, 'class-compare');
        assert.equal(caption.textContent, frame.caption, `The final caption stays current at progress ${progress}`);
        assert(caption.textContent.length > 20);
        assert(!caption.hidden, `The final caption stays visible at progress ${progress}`);
        assert.notEqual(caption.style.opacity, '0');
        assert.notEqual(caption.attrs['aria-hidden'], 'true');
        assert(!('inert' in caption.attrs));
        assert.equal(link.hidden, false, `The return link stays visible at progress ${progress}`);
        assert.notEqual(link.style.opacity, '0');
        assert.notEqual(link.attrs['aria-hidden'], 'true');
        assert(!('inert' in link.attrs), 'The return link remains keyboard accessible');
      }
      const earlier = app.scroll('class-head', .5);
      assert.equal(caption.textContent, earlier.caption, 'Reverse scrolling restores the earlier caption');
      assert.equal(link.hidden, true, 'The final return link hides when leaving the conclusion');
      const final = app.scroll('class-compare', 1);
      assert.equal(caption.textContent, final.caption, 'Returning to the end restores the final caption');
      assert.equal(link.hidden, false, 'Returning to the end restores the final link');
    }
  }
});

test('hinge reference curves remain examples while scrolling controls one actual hidden neuron', () => {
  const referencePaths = markup => [...markup.matchAll(/<path\b[^>]*data-hinge-reference="true"[^>]*>/g)]
    .map(match => attrs(match[0]));
  for (const mobile of [false, true]) {
    const app = mount({ mobile });
    app.scroll('relu', .5);
    assert.equal(referencePaths(app.element('function-plot').innerHTML).length, 0);
    const early = app.scroll('hinge', .2);
    const originalModel = JSON.stringify(early.model);
    const earlyPlot = app.element('function-plot').innerHTML;
    const references = referencePaths(earlyPlot);
    assert.equal(references.length, 2);
    assert.deepEqual(references.map(item => [Number(item['data-a']), Number(item['data-t'])]), [[1, -.8], [1, .6]]);
    assert(references.every(item => item['stroke-dasharray']), 'The fixed examples are visibly distinguished by dashed strokes');
    assert(earlyPlot.includes('data-hinge-current-parameters="true"'), 'The current a and t are labelled separately');
    assert.equal(early.hiddenCount, 1, 'Display-only references do not add hidden neurons');
    assert.equal(early.activeCount, 1);
    app.input('input-x', -2);
    assert.equal(app.frame().activeCount, 0);
    assert.equal(app.frame().hiddenCount, 1);
    assert.equal(JSON.stringify(app.frame().model), originalModel, 'The slider traces the chosen hinge without changing its parameters');
    app.input('input-x', 2);
    assert.equal(app.frame().activeCount, 1);
    assert.equal(JSON.stringify(app.frame().model), originalModel);
    for (const [progress, phase] of [[.55, 'steepen'], [.99, 'flip']]) {
      const frame = app.scroll('hinge', progress);
      assert.equal(frame.hingePhase, phase);
      assert.equal(frame.hiddenCount, 1);
      assert.equal(frame.activeCount, 1);
      assert.equal(frame.model.width, 1);
      assert.equal(frame.model.neurons.length, 1);
      assert.equal(referencePaths(app.element('function-plot').innerHTML).length, 2);
      const expected = app.window.NNStoryState.frame('hinge', progress).model;
      assert.equal(JSON.stringify(frame.model), JSON.stringify(expected));
      app.checkSvg((mobile ? 'mobile ' : 'desktop ') + phase + ' hinge');
    }
    assert(app.frame().hingeA < 0, 'The final active contribution slopes downward');
    assert(app.window.NNMath.evaluate(app.frame().model, 2) < 0);
    const restored = app.scroll('hinge', .2);
    assert.equal(restored.hingePhase, 'threshold');
    assert.equal(JSON.stringify(restored.model), originalModel, 'Reverse scrolling restores the exact earlier hinge');
    assert.equal(referencePaths(app.element('function-plot').innerHTML).length, 2);
    app.scroll('hill-rise', .5);
    assert.equal(referencePaths(app.element('function-plot').innerHTML).length, 0, 'Reference examples appear only in the hinge explanation');
    assert.equal(app.frame().hiddenCount, 1);
  }
});

test('the scalar input inspector hides only for the extension and restores the chosen scalar on return', () => {
  for (const mobile of [false, true]) {
    const app = mount({ mobile });
    const inspector = app.element('input-inspector');
    app.scroll('shallow', .4);
    app.input('input-x', 1.25);
    assert.equal(inspector.hidden, false);
    for (const scene of sceneNames.slice(16)) {
      const frame = app.scroll(scene, .5);
      assert.equal(inspector.hidden, true, `${scene} does not expose an irrelevant scalar input`);
      assert.equal(frame.hideInspector, true);
      assert.notEqual(frame.inspectInput, true, 'A previous slider selection does not alter the extension');
      assert.equal(app.element('active-neurons').hidden, true, 'The old scalar activation counter is not reused');
    }
    const restored = app.scroll('shallow', .4);
    assert.equal(inspector.hidden, false);
    assert.equal(restored.inspectInput, true);
    assert.equal(restored.x, 1.25);
    assert.equal(Number(app.element('input-x').value), 1.25);
    for (const scene of sceneNames.slice(0, 16)) {
      app.scroll(scene, .5);
      assert.equal(inspector.hidden, false, `The original ${scene} inspector stays available`);
    }
  }
});

function observationGeometry(markup) {
  return [...markup.matchAll(/<(?:circle|rect)\b[^>]*data-point-id="[^"]+"[^>]*>/g)].map(match => {
    const a = attrs(match[0]);
    const circle = match[0].startsWith('<circle');
    return {
      id: a['data-point-id'], space: a['data-space'],
      x: circle ? Number(a.cx) : Number(a.x) + Number(a.width) / 2,
      y: circle ? Number(a.cy) : Number(a.y) + Number(a.height) / 2,
      shape: circle ? 'circle' : 'square', fill: a.fill
    };
  });
}

function representationBoundaries(markup) {
  return [...markup.matchAll(/<g\b[^>]*data-classification-boundary="representation"[^>]*>/g)]
    .map(match => attrs(match[0]));
}

test('observations move before any straight classification boundary appears, then reverse exactly', () => {
  for (const mobile of [false, true]) {
    const app = mount({ mobile });
    const plot = app.element('function-plot');
    const observations = app.window.NNClassificationData.inputs;
    const positions = [];
    for (const progress of [0, .25, .5, .75, .99]) {
      const frame = app.scroll('class-move', progress);
      assert.equal(frame.scene, 'class-move');
      const points = observationGeometry(plot.innerHTML);
      assert.equal(points.length, observations.length, 'Every observation has its own plotted marker');
      assert.equal(new Set(points.map(point => point.id)).size, observations.length, 'Observation identities are unique');
      assert(points.every(point => Number.isFinite(point.x) && Number.isFinite(point.y)));
      assert.equal(representationBoundaries(plot.innerHTML).length, 0,
        'The head boundary is not present during movement, even near its completed state');
      if (positions.length) {
        assert.deepEqual(points.map(({ id, shape, fill }) => ({ id, shape, fill })),
          positions[0].map(({ id, shape, fill }) => ({ id, shape, fill })),
          'Observation identities, shapes, and colors survive the change of coordinates');
      }
      positions.push(points);
      app.checkSvg(`movement ${progress} ${mobile ? 'compact' : 'desktop'}`);
    }
    const moved = positions.at(-1).filter((point, i) =>
      Math.hypot(point.x - positions[0][i].x, point.y - positions[0][i].y) > 1);
    assert(moved.length > observations.length / 2, 'Scrolling changes actual marker coordinates');
    assert.notDeepEqual(positions[1], positions[2], 'The movement has intermediate geometry');
    assert.notDeepEqual(positions[2], positions[3], 'The movement is not only an endpoint jump');

    app.scroll('class-move', .35);
    const earlierMarkup = plot.innerHTML;
    app.scroll('class-head', .75);
    const boundaries = representationBoundaries(plot.innerHTML);
    assert.equal(boundaries.length, 1, 'The next scene reveals the single straight binary head boundary');
    assert(Number(boundaries[0].opacity) > 0, 'The revealed boundary is visible');
    assert(Number(boundaries[0]['data-boundary-opacity']) > 0);
    app.scroll('class-move', .35);
    assert.equal(plot.innerHTML, earlierMarkup, 'Reverse scrolling restores the exact intermediate point configuration');
    assert.equal(representationBoundaries(plot.innerHTML).length, 0, 'Reverse scrolling removes the straight boundary again');
  }
});

test('the rendered straight head boundary actually separates the transformed observations', () => {
  for (const mobile of [false, true]) {
    const app = mount({ mobile });
    app.scroll('class-compare', .5);
    const plot = app.element('function-plot').innerHTML;
    const points = observationGeometry(plot);
    const model = app.window.NNClassificationData;
    const lines = [...plot.matchAll(/<line\b[^>]*data-zero-logit="true"[^>]*>/g)].map(match => attrs(match[0]));
    assert.equal(lines.length, 1, 'The binary head has one straight decision boundary');
    const line = lines[0], ax = Number(line.x1), ay = Number(line.y1), bx = Number(line.x2), by = Number(line.y2);
    const signs = [new Set(), new Set()];
    points.forEach(point => {
      const side = (bx - ax) * (point.y - ay) - (by - ay) * (point.x - ax);
      assert(Math.abs(side) > 1e-5, 'Every marker lies strictly to one side of the visible boundary');
      signs[model.labels[Number(point.id)]].add(Math.sign(side));
    });
    assert.equal(signs[0].size, 1, 'All class-zero markers occupy one side');
    assert.equal(signs[1].size, 1, 'All class-one markers occupy one side');
    assert.notEqual([...signs[0]][0], [...signs[1]][0], 'The two classes occupy opposite sides');
    const pointTags = [...plot.matchAll(/<(?:circle|rect)\b[^>]*data-point-id="[^"]+"[^>]*>/g)].map(match => attrs(match[0]));
    pointTags.forEach(point => {
      const i = Number(point['data-point-id']), input = model.inputs[i];
      for (let j = 0; j < 2; j++) {
        const expected = Math.max(0, model.hiddenBias[j] + model.hiddenWeights[j][0] * input[0] + model.hiddenWeights[j][1] * input[1]);
        assert(Math.abs(Number(point['data-h' + (j + 1)]) - expected) < 1e-10,
          'Each displayed hidden coordinate is the fitted neuron activation');
      }
    });
    const reference = app.element('network-plot').innerHTML;
    const referenceIds = [...reference.matchAll(/data-reference-point-id="([^"]+)"/g)].map(match => match[1]);
    assert.deepEqual(referenceIds, points.map(point => point.id), 'The comparison preserves the identical observations in both spaces');
    assert(reference.includes('data-classification-boundary="input"'), 'The original-space view shows the corresponding fitted boundary');
  }
});

test('reduced motion preserves the representation reveal as a separate scene from the head', () => {
  for (const mobile of [false, true]) {
    const app = mount({ mobile, reduced: true });
    const plot = app.element('function-plot');
    app.scroll('class-trace', .5);
    const original = observationGeometry(plot.innerHTML);
    app.click('next');
    assert.equal(app.frame().scene, 'class-move');
    assert.equal(app.frame().progress, 1, 'The motion completes immediately for reduced motion');
    const transformed = observationGeometry(plot.innerHTML);
    assert.equal(transformed.length, original.length);
    assert(transformed.every(point => point.space === 'representation'), 'All points have reached their hidden coordinates');
    assert.notDeepEqual(transformed, original);
    assert.equal(representationBoundaries(plot.innerHTML).length, 0, 'Completed movement still does not reveal the line');
    app.click('next');
    assert.equal(app.frame().scene, 'class-head');
    assert(representationBoundaries(plot.innerHTML).some(boundary => Number(boundary.opacity) > 0));
    app.click('previous');
    assert.equal(app.frame().scene, 'class-move');
    assert.equal(representationBoundaries(plot.innerHTML).length, 0);
    assert.deepEqual(observationGeometry(plot.innerHTML), transformed, 'Returning leaves the fully transformed points but hides the line');
  }
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
        assert(/^<[\w:-]+(?:\s+[\w:.-]+\s*=\s*(?:"[^"]*"|'[^']*'))*\s*\/?>$/.test(token),
          'SVG attributes are quoted, including mathematical source strings containing equals signs');
        if (!token.endsWith('/>')) stack.push(tag);
      }
    }
    assert.equal(stack.length, 0, 'SVG tags are closed');
  }
});
