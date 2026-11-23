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
  const first=createIntuitionScenes(drawing).find(scene=>scene.id==='desert-senses');
  for(const example of DESERT_EXAMPLES){
    assert.ok(first.draw(0).includes(example.text));
    assert.ok(first.draw(1).includes(example.reading));
  }
});

test('lookup identity is distinguished from position and schematic contextual updates',()=>{
  const scenes=createIntuitionScenes(drawing);
  assert.equal(scenes.length,6);
  const lookup=scenes.find(scene=>scene.id==='desert-lookup');
  const context=scenes.find(scene=>scene.id==='desert-context');
  assert.match(lookup.detail,/full input rows can differ because position/);
  assert.match(context.equation,/x_i\^\{\(1\)\}=x_i\^\{\(0\)\}\+\\Delta x_i/);
});

test('intuition scenes render strict mathematics and reverse their scroll-driven states',()=>{
  for(const scene of createIntuitionScenes(drawing)){
    drawing.m(scene.equation);
    const initial=scene.draw(0);
    for(const t of [.1,.35,.6,.85,1]) assert.doesNotMatch(scene.draw(t),/NaN|undefined/);
    assert.ok([.1,.35,.6,.85,1].some(t=>scene.draw(t)!==initial),scene.id);
    assert.equal(scene.draw(0),initial,scene.id);
  }
});
