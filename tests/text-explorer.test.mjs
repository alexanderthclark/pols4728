import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
globalThis.katex = require('../docs/word2vec/vendor/katex/katex.min.js');
const {initializeExplorer} = await import('../docs/text/explorer.mjs');
const {counts, vocabulary} = await import('../docs/text/model.mjs');

const attributes = source => Object.fromEntries([...source.matchAll(/([\w:-]+)="([^"]*)"/g)].map(([, name, value]) => [name, value]));
const texExpressions = markup => [...markup.matchAll(/<annotation encoding="application\/x-tex">([\s\S]*?)<\/annotation>/g)].map(match => match[1]);

// A small DOM double parses the real rendered table and implements only the
// event, focus, and class APIs used by the controller. No browser package needed.
function fixture() {
  const listeners = new Map();
  const state = {writes: 0, resultWrites: 0, focusCalls: [], activeElement: null};
  const node = (attrs = {}, parent = null) => {
    const classes = new Set((attrs.class ?? '').split(/\s+/).filter(Boolean));
    const element = {
      attrs, parent,
      classList: {
        contains: value => classes.has(value),
        toggle(value, force) { if (force) classes.add(value); else classes.delete(value); },
      },
      getAttribute: name => attrs[name] ?? null,
      setAttribute(name, value) { attrs[name] = String(value); },
      closest(selector) {
        if (selector === '.matrix-cell' && classes.has('matrix-cell')) return element;
        if (selector === 'td' && element.tag === 'td') return element;
        return parent?.closest(selector) ?? null;
      },
    };
    return element;
  };
  const root = {
    addEventListener(type, handler) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type).add(handler);
    },
    removeEventListener(type, handler) { listeners.get(type)?.delete(handler); },
    querySelectorAll(selector) {
      if (selector === '.matrix-cell') return state.buttons;
      if (selector === 'tbody tr') return state.rows;
      if (selector === 'thead th') return state.headings;
      throw new Error(`Unsupported fixture selector: ${selector}`);
    },
    querySelector(selector) {
      assert.equal(selector, '.explorer-result');
      return state.result;
    },
    set innerHTML(markup) {
      state.writes++;
      state.markup = markup;
      state.buttons = [];
      state.headings = [...markup.match(/<thead>([\s\S]*?)<\/thead>/)[1].matchAll(/<th\b([^>]*)>/g)].map(([, attrs]) => node(attributes(attrs)));
      state.rows = [...markup.match(/<tbody>([\s\S]*?)<\/tbody>/)[1].matchAll(/<tr\b([^>]*)>([\s\S]*?)<\/tr>/g)].map(([, attrs, body]) => {
        const row = node(attributes(attrs));
        for (const [, cellAttrs, buttonAttrs, value] of body.matchAll(/<td\b([^>]*)><button\b([^>]*)>([^<]*)<\/button><\/td>/g)) {
          const cell = node(attributes(cellAttrs), row);
          cell.tag = 'td';
          const button = node(attributes(buttonAttrs), cell);
          button.dataset = Object.fromEntries(Object.entries(button.attrs).filter(([key]) => key.startsWith('data-')).map(([key, value]) => [key.slice(5), value]));
          button.value = Number(value);
          Object.defineProperty(button, 'tabIndex', {get: () => Number(button.attrs.tabindex), set: value => { button.attrs.tabindex = String(value); }});
          button.focus = options => {
            state.activeElement = button;
            state.focusCalls.push({button, options});
            emit('focusin', button);
          };
          state.buttons.push(button);
        }
        return row;
      });
      state.result = {
        markup: markup.match(/<div class="explorer-result[^\"]*"[^>]*>([\s\S]*)<\/div>$/)[1],
        get innerHTML() { return this.markup; },
        set innerHTML(value) { state.resultWrites++; this.markup = value; },
      };
    },
  };
  function emit(type, target, options = {}) {
    const event = {target, defaultPrevented: false, preventDefault() { this.defaultPrevented = true; }, ...options};
    for (const handler of listeners.get(type) ?? []) handler(event);
    return event;
  }
  const cell = (doc, column) => state.buttons[doc * vocabulary.length + column];
  return {root, state, listeners, emit, cell, node};
}

function assertSelection(page, doc, column) {
  const selected = page.cell(doc, column);
  assert.deepEqual(page.state.buttons.filter(button => button.getAttribute('aria-pressed') === 'true'), [selected]);
  assert.deepEqual(page.state.buttons.filter(button => button.tabIndex === 0), [selected]);
  assert.deepEqual(page.state.rows.map(row => row.classList.contains('active')), counts.map((_, index) => index === doc));
  assert.deepEqual(page.state.headings.slice(1).map(heading => heading.classList.contains('term-focus')), vocabulary.map((_, index) => index === column));
  for (const button of page.state.buttons) assert.equal(button.closest('td').classList.contains('focus'), Number(button.dataset.col) === column);
  assert.match(page.state.result.innerHTML, new RegExp(`${vocabulary[column]} in D${doc + 1}`));
  const tf = counts[doc][column], df = counts.filter(row => row[column] > 0).length;
  const expressions = texExpressions(page.state.result.innerHTML);
  assert.equal(expressions[0], String.raw`\operatorname{tf}=${tf},\qquad\operatorname{df}=${df},\qquad N=${counts.length}`);
  assert.equal(Number(expressions[1].match(/=([\d.]+)$/)[1]), Number((tf * Math.log(counts.length / df)).toFixed(3)));
}

test('the explorer starts with one accessible button per count and one tab stop', () => {
  const page = fixture();
  initializeExplorer(page.root);
  assert.equal(page.state.writes, 1);
  assert.doesNotMatch(page.state.markup, /<select\b/);
  assert.match(page.state.markup, /class="explorer-result[^\"]*" aria-live="polite" aria-atomic="true"/);
  assert.equal(page.state.buttons.length, 28);
  for (const [doc, row] of counts.entries()) {
    for (const [column, value] of row.entries()) {
      const button = page.cell(doc, column);
      assert.equal(button.value, value);
      assert.equal(button.getAttribute('type'), 'button');
      assert.equal(button.dataset.term, vocabulary[column]);
      assert.equal(button.getAttribute('aria-label'), `D${doc + 1}, ${vocabulary[column]}, count ${value}`);
    }
  }
  assertSelection(page, 0, 0);
});

test('hovering all 28 cells updates the calculation without replacing or focusing the table', () => {
  const page = fixture();
  initializeExplorer(page.root);
  const buttons = [...page.state.buttons], rows = [...page.state.rows];
  for (let doc = 0; doc < counts.length; doc++) {
    for (let column = 0; column < vocabulary.length; column++) {
      page.emit('pointerover', page.cell(doc, column));
      assertSelection(page, doc, column);
    }
  }
  assert.equal(page.state.writes, 1);
  assert.deepEqual(page.state.buttons, buttons);
  assert.deepEqual(page.state.rows, rows);
  assert.equal(page.state.focusCalls.length, 0);
  const updates = page.state.resultWrites;
  page.emit('pointerover', page.cell(3, 6));
  page.emit('pointerleave', page.cell(3, 6));
  assert.equal(page.state.resultWrites, updates);
  assertSelection(page, 3, 6);
});

test('tap and keyboard focus select cells, including events from nested contents', () => {
  const page = fixture();
  initializeExplorer(page.root);
  page.emit('click', page.node({}, page.cell(2, 2)));
  assertSelection(page, 2, 2);
  page.cell(1, 4).focus();
  assertSelection(page, 1, 4);
  page.emit('pointerover', page.cell(0, 6));
  assertSelection(page, 0, 6);
  assert.equal(page.state.activeElement, page.cell(1, 4), 'Hover must not steal keyboard focus.');
  page.emit('click', page.node());
  assertSelection(page, 0, 6);
});

test('keyboard navigation follows the focused cell, clamps at edges, and supports row and table ends', () => {
  const page = fixture();
  initializeExplorer(page.root);
  const move = (key, doc, column, options = {}) => {
    const event = page.emit('keydown', page.state.activeElement, {key, ...options});
    assert.equal(event.defaultPrevented, true);
    assertSelection(page, doc, column);
    assert.equal(page.state.activeElement, page.cell(doc, column));
    assert.deepEqual(page.state.focusCalls.at(-1).options, {preventScroll: true});
  };
  page.cell(1, 2).focus();
  page.emit('pointerover', page.cell(3, 6));
  move('ArrowRight', 1, 3);
  move('ArrowDown', 2, 3);
  move('ArrowLeft', 2, 2);
  move('ArrowUp', 1, 2);
  move('Home', 1, 0);
  move('ArrowLeft', 1, 0);
  move('End', 1, 6);
  move('ArrowRight', 1, 6);
  move('Home', 0, 0, {ctrlKey: true});
  move('ArrowUp', 0, 0);
  move('End', 3, 6, {ctrlKey: true});
  move('ArrowDown', 3, 6);
  for (const options of [{key: 'Tab'}, {key: 'Escape'}, {key: 'ArrowLeft', altKey: true}, {key: 'ArrowLeft', metaKey: true}]) {
    const before = page.state.focusCalls.length;
    assert.equal(page.emit('keydown', page.state.activeElement, options).defaultPrevented, false);
    assert.equal(page.state.focusCalls.length, before);
    assertSelection(page, 3, 6);
  }
});

test('disposing the explorer removes every handler and leaves the last selection intact', () => {
  const page = fixture();
  const dispose = initializeExplorer(page.root);
  page.emit('pointerover', page.cell(2, 4));
  const updates = page.state.resultWrites;
  dispose();
  dispose();
  assert.equal([...page.listeners.values()].reduce((sum, handlers) => sum + handlers.size, 0), 0);
  for (const type of ['pointerover', 'focusin', 'click', 'keydown']) page.emit(type, page.cell(0, 0), {key: 'ArrowRight'});
  assert.equal(page.state.resultWrites, updates);
  assert.equal(page.state.focusCalls.length, 0);
  assertSelection(page, 2, 4);
});
