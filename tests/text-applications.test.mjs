import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import test from 'node:test';

const require=createRequire(import.meta.url);
globalThis.katex=require('../docs/word2vec/vendor/katex/katex.min.js');
const {applicationModel,applicationVisual,topicDocuments}=await import('../docs/text/applications.mjs');

test('the constructed topics preserve their tokens and use class-based rather than document-frequency weights',()=>{
  const {terms,counts,lengths,averageLength,termTotals,weights}=applicationModel;
  assert.deepEqual(topicDocuments,[['budget tax policy','budget spending policy'],['health clinic policy','health care policy']]);
  assert.deepEqual(lengths,[6,6]);
  assert.equal(averageLength,6);
  const expected=[{budget:2,policy:2,spending:1,tax:1},{care:1,clinic:1,health:2,policy:2}];
  for(let row=0;row<2;row++){
    assert.deepEqual(counts[row],terms.map(term=>expected[row][term]??0));
    const top=terms[weights[row].indexOf(Math.max(...weights[row]))];
    assert.equal(top,row===0?'budget':'health');
    const policy=terms.indexOf('policy');
    assert.equal(termTotals[policy],4);
    assert.ok(Math.abs(weights[row][policy]-Math.log(2.5)/3)<1e-12,
      'The shared word retains a positive c-TF–IDF weight; its denominator is four occurrences.');
  }
  assert.ok(Math.abs(weights[0][terms.indexOf('budget')]-Math.log(4)/3)<1e-12);
  assert.ok(Math.abs(weights[1][terms.indexOf('clinic')]-Math.log(7)/6)<1e-12);
  assert.equal(weights[0][terms.indexOf('health')],0);
});

test('application diagrams change with scrolling, reverse exactly, and hold completed reduced-motion states',()=>{
  for(const stage of [6,7,8]){
    const positions=Array.from({length:21},(_,i)=>stage-.5+i/40);
    const forward=positions.map(p=>applicationVisual(stage,p,true));
    const backward=[...positions].reverse().map(p=>applicationVisual(stage,p,true)).reverse();
    assert.deepEqual(backward,forward);
    assert.notEqual(forward[0],forward.at(-1),`Stage ${stage} must construct its diagram.`);
    for(const markup of forward)assert.doesNotMatch(markup,/NaN|Infinity|undefined|katex-error/);
    assert.equal(applicationVisual(stage,stage-.5,false),applicationVisual(stage,stage,false));
    assert.equal(applicationVisual(stage,stage,false),forward.at(-1));
  }
});
