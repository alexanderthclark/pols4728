import test from 'node:test';
import assert from 'node:assert/strict';
import katex from '../docs/attention/vendor/katex/katex.min.js';
import {createIntuitionScenes, DESERT_EXAMPLES} from '../docs/attention/intuition-scenes.mjs';

const drawing={
  C:{blue:'#234e70',teal:'#007f73',rust:'#a54f32',ink:'#242424',muted:'#606060',rule:'#dededb'},
  m:tex=>katex.renderToString(tex,{throwOnError:true,strict:'error',output:'htmlAndMathml'}),
  line:(...args)=>`<line data-args="${args.join(',')}"/>`,
  rect:(...args)=>`<rect data-args="${args.join(',')}"/>`,
};

test('the two complete desert sentences and distinct intended readings are preserved',()=>{
  assert.deepEqual(DESERT_EXAMPLES.map(example=>example.text),[
    'The jury was concerned with moral desert and not the laws.',
    'The upper west side is a taco bell desert.',
  ]);
  const first=createIntuitionScenes(drawing)[0];
  for(const example of DESERT_EXAMPLES){
    assert.ok(first.draw(0).includes(example.text));
    assert.ok(first.draw(1).includes(example.reading));
  }
});

test('lookup identity is distinguished from position and schematic contextual updates',()=>{
  const scenes=createIntuitionScenes(drawing);
  assert.equal(scenes.length,5);
  assert.match(scenes[1].detail,/full input rows can differ because position/);
  assert.match(scenes[2].detail,/not measured attention weights/);
  assert.match(scenes[2].equation,/x_i\^\{\(1\)\}=x_i\^\{\(0\)\}\+\\Delta x_i/);
});

test('intuition scenes render strict mathematics and reverse their scroll-driven states',()=>{
  for(const scene of createIntuitionScenes(drawing)){
    drawing.m(scene.equation);
    const initial=scene.draw(0);
    for(const t of [.1,.35,.6,.85,1]) assert.doesNotMatch(scene.draw(t),/NaN|undefined/);
    assert.notEqual(scene.draw(1),initial,scene.id);
    assert.equal(scene.draw(0),initial,scene.id);
  }
});
