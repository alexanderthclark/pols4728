import test from 'node:test';
import assert from 'node:assert/strict';
import katex from '../docs/attention/vendor/katex/katex.min.js';
import {createSocksScenes} from '../docs/attention/socks-scenes.mjs';
import {createExtensionScenes, createDecoderScenes} from '../docs/attention/extension-scenes.mjs';

const clamp = value => Math.max(0, Math.min(1, value));
const escape = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
const shape = tag => (...args) => `<${tag} data-args="${args.join(',')}"/>`;
const drawing = {
  C: {blue:'#234e70', teal:'#007f73', rust:'#a54f32', purple:'#7b4f9d', ink:'#242424', muted:'#606060', rule:'#dededb'},
  m: tex => katex.renderToString(tex, {throwOnError:true, strict:'error', output:'htmlAndMathml'}),
  text: (value, ...args) => `<text data-args="${args.join(',')}">${escape(value)}</text>`,
  line: shape('line'),
  rect: shape('rect'),
  circle: shape('circle'),
  clamp,
  lerp: (a, b, t) => a + (b - a) * t,
  ease: value => { const t = clamp(value); return t * t * (3 - 2 * t); },
  num: value => Math.abs(value) < .0005 ? '0' : Number.isInteger(value) ? String(value) : value.toFixed(2),
};

const mainScenes = () => [...createSocksScenes(drawing), ...createExtensionScenes(drawing)];

test('the classification route compares pooling before attention and after the residual update', () => {
  const scenes = mainScenes();
  assert.equal(scenes.length, 20);
  const positionIndex = scenes.findIndex(scene => scene.equation === 'X=E+P');
  assert.ok(positionIndex >= 0, 'the full positional input is introduced');
  assert.equal(scenes[positionIndex + 1].id, 'socks-position-pooling');
  const residualIndex = scenes.findIndex(scene => scene.id === 'socks-residual-update');
  assert.ok(residualIndex > positionIndex + 1, 'attention precedes the residual readout');
  assert.equal(scenes[residualIndex + 1].id, 'socks-residual-readout');
  assert.equal(scenes.at(-1).id, 'socks-attention-summary');
  assert.ok(scenes.every(scene => !scene.id?.startsWith('decoder-')));
});

test('decoder masking remains a separate three-scene sequence', () => {
  const decoder = createDecoderScenes(drawing);
  assert.deepEqual(decoder.map(scene => scene.id), [
    'decoder-next-token', 'decoder-mask-scores', 'decoder-renormalize',
  ]);
  const mainIds = new Set(mainScenes().map(scene => scene.id).filter(Boolean));
  assert.ok(decoder.every(scene => !mainIds.has(scene.id)));
});

test('main and decoder diagrams render strict mathematics and restore normal and reduced-motion states', () => {
  const scenes = [...mainScenes(), ...createDecoderScenes(drawing)];
  const phases = [0, .1, .25, .33, .5, .66, .75, .9, 1];
  for (const [index, scene] of scenes.entries()) {
    const name = scene.id || `main-scene-${index}`;
    drawing.m(scene.equation);
    for (const field of ['caption', 'detail']) {
      for (const match of scene[field].matchAll(/data-math="([^"]+)"/g)) drawing.m(match[1]);
    }
    for (const reducedMotion of [false, true]) {
      const states = phases.map(phase => scene.draw(phase, {reducedMotion}));
      for (const state of states) {
        assert.equal(typeof state, 'string', name);
        assert.ok(state.length > 0, name);
        assert.doesNotMatch(state, /NaN|undefined|katex-error/, name);
      }
      if (!reducedMotion) assert.ok(new Set(states).size > 1, `${name} responds to scroll progress`);
      for (let phaseIndex = phases.length - 1; phaseIndex >= 0; phaseIndex--) {
        assert.equal(scene.draw(phases[phaseIndex], {reducedMotion}), states[phaseIndex],
          `${name} restores phase ${phases[phaseIndex]} with reducedMotion=${reducedMotion}`);
      }
    }
  }
});
