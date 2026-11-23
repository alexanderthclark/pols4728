import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import katex from '../docs/attention/vendor/katex/katex.min.js';
import {createShapScenes,displayShapWords,shapWaterfall} from '../docs/attention/shap-scenes.mjs';

const data=JSON.parse(readFileSync(new URL('../docs/attention/data/infinite-jest-shap.json',import.meta.url),'utf8'));
const original=JSON.parse(readFileSync(new URL('../docs/attention/data/infinite-jest-attention.json',import.meta.url),'utf8'));
const near=(actual,expected,tolerance=1e-8)=>assert.ok(Math.abs(actual-expected)<=tolerance,`${actual} ≠ ${expected}`);
const drawing={
  C:{blue:'#234e70',teal:'#007f73',rust:'#a54f32',ink:'#242424',muted:'#606060',rule:'#dededb'},
  m:(tex,x,y,w,h,size,color)=>`<foreignObject data-x="${x}" data-y="${y}">${katex.renderToString(tex,{throwOnError:true,strict:'error',output:'htmlAndMathml'})}</foreignObject>`,
  text:(value,x,y,size,color,anchor,extra='')=>`<text x="${x}" y="${y}" style="fill:${color}" ${extra}>${String(value).replaceAll('&','&amp;').replaceAll('<','&lt;')}</text>`,
  line:(...args)=>`<line data-args="${args.join(',')}"/>`,
  rect:(...args)=>`<rect data-args="${args.join(',')}"/>`,
  circle:(...args)=>`<circle data-args="${args.join(',')}"/>`,
  clamp:(value)=>Math.max(0,Math.min(1,value)),
  ease:(value)=>{const p=Math.max(0,Math.min(1,value));return p*p*(3-2*p);},
};

test('published token attributions reconstruct both original classifier scores',()=>{
  assert.equal(data.method.name,'exact-token-shapley');
  assert.equal(data.model.revision,original.model.revision);
  for(const example of data.examples){
    const source=original.examples.find(item=>item.text===example.text);
    assert.ok(source);
    assert.deepEqual(example.tokens,source.tokens);
    assert.equal(example.feature_positions.length,13);
    assert.equal(example.coalition_count,8192);
    near(example.baseline_margin+example.shap_values.reduce((a,b)=>a+b,0),example.full_margin);
    near(example.full_margin,source.logits[1]-source.logits[0],2e-4);
    near(example.positive_probability,1/(1+Math.exp(-example.full_margin)));
    for(let position=0;position<example.tokens.length;position++){
      if(example.special_tokens_mask[position])assert.equal(example.shap_values[position],0);
    }
  }
});

test('display grouping covers each token once and keeps punctuation contributions separate',()=>{
  for(const example of data.examples){
    const words=displayShapWords(example);
    assert.deepEqual(words.flatMap(word=>word.token_positions),example.feature_positions);
    for(const word of words){
      assert.equal(word.text,example.text.slice(word.start,word.end));
      near(word.shap_value,word.token_positions.reduce((sum,p)=>sum+example.shap_values[p],0));
    }
    near(words.reduce((sum,word)=>sum+word.shap_value,0),example.full_margin-example.baseline_margin);
    const jest=words.find(word=>word.text==='Jest');
    assert.equal(jest.token_positions.length,2);
  }
  const positive=data.examples.find(example=>example.id==='positive');
  const words=displayShapWords(positive);
  assert.equal(words.filter(word=>word.text==='.').length,2);
  assert.equal(words.find(word=>word.text==='good').token_positions.length,1);
});

test('waterfall endpoints carry each signed allocation into the next row',()=>{
  for(const example of data.examples){
    const rows=shapWaterfall(example);
    assert.equal(rows[0].kind,'baseline');
    near(rows[0].end,example.baseline_margin);
    for(let i=1;i<rows.length-1;i++){
      assert.equal(rows[i].kind,'contribution');
      near(rows[i].start,rows[i-1].end);
      near(rows[i].end-rows[i].start,rows[i].value);
    }
    near(rows.at(-2).end,example.full_margin);
    assert.equal(rows.at(-1).kind,'prediction');
    near(rows.at(-1).end,example.full_margin);
  }
});

test('one-token example is one measured marginal gain, distinct from its SHAP average',()=>{
  const example=data.examples.find(item=>item.id==='positive'),comparison=example.illustrative_comparison;
  assert.equal(example.tokens[comparison.position],'good');
  near(comparison.with_margin,example.full_margin);
  near(comparison.marginal_gain,comparison.with_margin-comparison.without_margin);
  assert.ok(Math.abs(comparison.marginal_gain-example.shap_values[comparison.position])>.1);
});

test('SHAP scenes preserve exact text, render strict mathematics, and reverse scroll states',()=>{
  const scenes=createShapScenes(drawing,data,original);
  assert.deepEqual(scenes.map(scene=>scene.id),['trained-shap-mask','trained-shap-negative','trained-shap-positive']);
  for(const scene of scenes){
    assert.equal(scene.equation,'');assert.equal(scene.detail,'');
    for(const match of scene.reading.matchAll(/data-math="([^"]+)"/g))drawing.m(match[1]);
    const start=scene.draw(0),end=scene.draw(1);
    assert.notEqual(start,end);
    for(const progress of [0,.1,.35,.65,.85,1]){
      assert.doesNotMatch(scene.draw(progress),/NaN|undefined/);
      assert.doesNotMatch(scene.draw(progress,{reducedMotion:true}),/NaN|undefined/);
    }
    assert.equal(scene.draw(0),start);
  }
  for(const example of data.examples){
    const scene=scenes.find(item=>item.id===`trained-shap-${example.id}`),svg=scene.draw(1);
    assert.ok(svg.includes(`aria-label="${example.text}"`));
    assert.equal((svg.match(/data-kind="contribution"/g)||[]).length,displayShapWords(example).length);
    assert.match(scene.reading,/intermediate totals are not predictions/);
  }
  assert.throws(()=>createShapScenes(drawing,data,{...original,model:{...original.model,revision:'different'}}),/same pinned checkpoint/);
});
