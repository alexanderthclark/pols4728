'use strict';

// DOM/scroll integration checks. This does not launch a browser or test layout.
const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const directory = path.resolve(__dirname, '../docs/neural-networks');
const html = fs.readFileSync(path.join(directory, 'index.html'), 'utf8');
const scripts = ['math.js', 'state.js', 'views.js', 'formula.js', 'story.js'];
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
      attrs: { ...attributes }, dataset: {}, style: {}, disabled: false, hidden: false,
      value: attributes.value || '', min: attributes.min || '', max: attributes.max || '', step: attributes.step || '',
      tagName: attributes.type ? 'INPUT' : '', innerHTML: '', textContent: '',
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

test('the HTML is scroll driven, has one labelled input inspector, and its local assets resolve', () => {
  assert.equal(sceneNames.length, 16);
  assert.equal(new Set(sceneNames).size, 16);
  assert(!/<select\b|<form\b/i.test(html), 'No form controls substitute for scrolling');
  const inputs = [...html.matchAll(/<input\b[^>]*>/gi)].map(match => attrs(match[0]));
  assert.equal(inputs.length, 1, 'The input inspector is the only input');
  assert.equal(inputs[0].id, 'input-x');
  assert.equal(inputs[0].type, 'range');
  assert(/<label\b[^>]*for="input-x"[^>]*>[\s\S]*?<\/label>/.test(html), 'The x slider has an explicit label');
  assert(inputs[0]['aria-describedby'], 'The slider points to its explanatory hint');
  assert(!/aria-pressed/i.test(html), 'No toggles replace the scroll story');
  const opening = sceneMarkup.find(match => match[2] === 'affine')[3].replace(/<[^>]*>/g, ' ');
  assert(/not a second observed feature/.test(opening), 'The constant bias source is distinguished from an observed feature');
  assert(/intercept remains part of the equation and the model/.test(opening), 'Hiding the bias drawing retains the model intercept');
  for (const match of html.matchAll(/<(?:script|img|link)\b[^>]*(?:src|href)="([^"]+)"[^>]*>/g)) {
    const reference = match[1];
    if (/^(?:[a-z]+:|\/\/|#)/i.test(reference)) continue;
    assert(fs.existsSync(path.resolve(directory, reference.split(/[?#]/)[0])), `Asset exists: ${reference}`);
  }
  const css = fs.readFileSync(path.join(directory, 'style.css'), 'utf8');
  for (const match of css.matchAll(/url\(\s*['"]?([^'"\s)]+)['"]?\s*\)/g)) {
    const reference = match[1];
    if (/^(?:[a-z]+:|\/\/|#)/i.test(reference)) continue;
    assert(fs.existsSync(path.resolve(directory, reference.split(/[?#]/)[0])), `Stylesheet asset exists: ${reference}`);
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
    assert.notEqual(frame.inspectInput, true, 'Automatic scrolling needs no slider interaction');
    assert.equal(Number(app.element('input-x').value), frame.x);
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
    if (scene === 'affine') assert([0, 1].includes(frame.progress), 'Reduced-motion affine scenes use complete visible/hidden bias states');
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

test('only the affine stage equation uses accessible native MathML generated from its exact TeX', () => {
  const expectedTex = String.raw`\hat{y}=wx+b=1\times x+0.5`;
  const references = [...html.matchAll(/<script\b[^>]*src="([^"]+)"[^>]*>/g)].map(match => match[1]);
  assert(references.includes('formula.js'));
  assert(references.indexOf('formula.js') < references.indexOf('story.js'), 'The formula renderer loads before the controller');
  for (const mobile of [false, true]) {
    const app = mount({ mobile });
    app.scroll('affine', .2);
    const equation = app.element('stage-equation');
    const markup = equation.innerHTML;
    assert.equal(app.window.NNFormula.affineTex, expectedTex);
    assert.equal(markup, app.window.NNFormula.affineMarkup);
    assert.equal(equation.attrs['data-tex'], expectedTex);
    const math = markup.match(/<math\b([^>]*)>/);
    assert(math, 'The equation contains a native MathML math element');
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
    assert(!/<math\b|<mover\b|<annotation\b/.test(equation.innerHTML), 'The next scene clears the affine MathML');
    assert.equal(equation.attrs['data-tex'], undefined);
    assert.equal(equation.textContent, relu.equation, 'Other scene equations retain their existing text');
    app.scroll('affine', .2);
    assert.equal(equation.innerHTML, markup, 'Reverse scrolling restores the typeset affine formula');
    assert.equal(equation.attrs['data-tex'], expectedTex);
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
