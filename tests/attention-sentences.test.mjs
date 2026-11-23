import test from 'node:test';
import assert from 'node:assert/strict';
import katex from '../docs/attention/vendor/katex/katex.min.js';
import {createSentenceScenes, SENTENCE_EXAMPLES, WORD_TOKENS} from '../docs/attention/sentence-scenes.mjs';

test('the motivating sentences and complete word-level source sequence are preserved', () => {
  assert.deepEqual(SENTENCE_EXAMPLES.map(example => example.text), [
    "Good and Infinite Jest don't belong in the same sentence",
    "Infinite Jest wasn't just good. It was spectacular.",
  ]);
  assert.deepEqual(SENTENCE_EXAMPLES.map(example => example.reading), ['Negative', 'Positive']);
  assert.equal(WORD_TOKENS.join(' '), SENTENCE_EXAMPLES[0].text);
  assert.equal(WORD_TOKENS.length, 10);
  assert.equal(WORD_TOKENS[0], 'Good');
  assert.equal(WORD_TOKENS[4], "don't");
});

test('the schematic retains the residual update and reads out the final stream', () => {
  const equations = [];
  const drawing = {C: {blue:'#234e70',teal:'#007f73',rust:'#a54f32',ink:'#242424',muted:'#606060',rule:'#dededb'},
    m: tex => {equations.push(tex); return `<math>${tex}</math>`;},
    line: () => '<line/>', rect: () => '<rect/>'};
  const scenes = createSentenceScenes(drawing);
  assert.equal(scenes.length, 8);
  const overview = scenes[0].draw(0);
  assert.match(overview, /Attention/);
  assert.match(overview, /MLP/);
  assert.match(overview, /Residual stream/);
  const inset = overview.match(/<g class="attention-matrix-inset"[^>]*>([\s\S]*?)<\/g>/)?.[1];
  assert.ok(inset, 'the attention matrix is visible before any scrolling');
  assert.equal((inset.match(/<rect\//g) || []).length, 100, 'the inset has all ten source and destination positions');
  assert.ok(equations.includes('A'));
  assert.match(scenes[0].detail, /symbolic.*no measured weights/);
  assert.ok(equations.includes('X_0') && equations.includes('X_1') && equations.includes('X_2'));
  equations.length = 0;
  const final = scenes.at(-1).draw(1);
  assert.match(final, /MLP/);
  assert.ok(equations.includes('AXW_1W_2'));
  assert.ok(equations.includes('r=\\operatorname{mean}_i(X_2)_i'));
  assert.ok(equations.includes('z=b+r\\beta'));
  assert.match(scenes.find(s => s.id === 'sentence-entry').detail, /no claim.*large weight/);
});

test('all sentence diagrams render strict mathematics and restore earlier scroll states', () => {
  const drawing = {C: {blue:'#234e70',teal:'#007f73',rust:'#a54f32',ink:'#242424',muted:'#606060',rule:'#dededb'},
    m: tex => katex.renderToString(tex,{throwOnError:true,strict:'error',output:'htmlAndMathml'}),
    line: (...args) => `<line data-args="${args.join(',')}"/>`,
    rect: (...args) => `<rect data-args="${args.join(',')}"/>`};
  for (const scene of createSentenceScenes(drawing)) {
    drawing.m(scene.equation);
    const initial = scene.draw(0);
    for (const phase of [.1,.33,.5,.75,1]) assert.doesNotMatch(scene.draw(phase), /NaN|undefined/);
    assert.notEqual(scene.draw(1), initial, scene.id);
    assert.equal(scene.draw(0), initial, scene.id);
  }
});
