import assert from 'node:assert/strict';
import test from 'node:test';
import {createRequire} from 'node:module';
import {theta,beta,contentBeta,conditionalTopics,wordContributions,wordProbability,baselineSoftmax,interpolatedTopics,pelotonExample} from '../docs/text/lda-model.mjs';
const require=createRequire(import.meta.url);
globalThis.katex=require('../docs/word2vec/vendor/katex/katex.min.js');
const {mechanismState,ldaMechanismVisual}=await import('../docs/text/lda-mechanism.mjs');
const near=(actual,expected)=>assert.ok(Math.abs(actual-expected)<1e-10,`${actual} ≠ ${expected}`);

test('the topic model uses normalized probability vectors and a correctly marginalized word probability',()=>{
  near(theta.reduce((a,b)=>a+b,0),1);
  for(const row of [...beta,...contentBeta]){near(row.reduce((a,b)=>a+b,0),1);assert.ok(row.every(p=>p>=0&&p<=1));}
  assert.deepEqual(wordContributions(2),[.01,.002,.06]);
  near(wordProbability(2),.072);
  const conditional=conditionalTopics(2);
  conditional.forEach((v,k)=>near(v,[5/36,1/36,5/6][k]));
  near(conditional.reduce((a,b)=>a+b,0),1);
  for(const q of [0,.1,.5,.9,1])for(const row of interpolatedTopics(q))near(row.reduce((a,b)=>a+b,0),1);
});

test('STM baseline softmax maps two log-ratios to three valid topic shares',()=>{
  baselineSoftmax([Math.log(.5/.3),Math.log(.2/.3)]).forEach((v,k)=>near(v,theta[k]));
  near(baselineSoftmax([1000,-1000]).reduce((a,b)=>a+b,0),1);
  assert.ok(baselineSoftmax([1000,-1000]).every(Number.isFinite));
  baselineSoftmax([0,0]).forEach(v=>near(v,1/3));
});

test('the same workspace and exact endpoints persist through all seven mechanism stages',()=>{
  const keys=markup=>[...markup.matchAll(/data-key="([^"]+)"/g)].map(match=>match[1]);
  const expected=keys(ldaMechanismVisual(0,0));
  assert.equal(new Set(expected).size,expected.length);
  for(let stage=1;stage<7;stage++){
    assert.equal(ldaMechanismVisual(stage,stage-.5),ldaMechanismVisual(stage-1,stage-1));
    assert.deepEqual(keys(ldaMechanismVisual(stage,stage)),expected);
    assert.equal(ldaMechanismVisual(stage,stage-.3,false),ldaMechanismVisual(stage,stage));
  }
  assert.equal(pelotonExample.excerptLines.join(' ').slice(1,-1),pelotonExample.excerpts[0]);
});

test('scrolling updates mixture inference and optional content while conserving their meaning',()=>{
  near(mechanismState(4,3.75).posterior,.5);
  near(mechanismState(6,5.75).content,.5);
  const before=ldaMechanismVisual(4,3.6),middle=ldaMechanismVisual(4,3.75),after=ldaMechanismVisual(4,4);
  assert.notEqual(before,middle);assert.notEqual(middle,after);
  assert.equal(middle,ldaMechanismVisual(4,3.75));
  assert.match(after,/data-share="0.8333"/);
  const content=ldaMechanismVisual(6,6);
  assert.match(content,/data-probability="0.35"/);
  assert.match(content,/data-share="0.5"/);
  assert.match(content,/not fitted to these classes/);
});

test('every mechanism state renders valid mathematics and finite geometry',()=>{
  for(let stage=0;stage<7;stage++)for(const delta of [-.5,-.25,0]){
    const markup=ldaMechanismVisual(stage,Math.max(0,stage+delta));
    assert.doesNotMatch(markup,/NaN|Infinity|undefined|katex-error/);
    for(const [_,tex] of markup.matchAll(/<annotation encoding="application\/x-tex">([\s\S]*?)<\/annotation>/g))assert.doesNotThrow(()=>globalThis.katex.renderToString(tex.replaceAll('&amp;','&').replaceAll('&lt;','<').replaceAll('&gt;','>'),{throwOnError:true}));
    for(const [tag] of markup.matchAll(/<(?:text|rect|path|g|foreignObject)\b[^>]*>/g)){
      const attributes=[...tag.matchAll(/([\w:-]+)="([^"]*)"/g)].map(([,name])=>name);
      assert.equal(attributes.length,new Set(attributes).size,tag);
    }
  }
});
