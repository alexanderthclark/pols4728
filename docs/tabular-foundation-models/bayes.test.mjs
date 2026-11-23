import test from 'node:test';
import assert from 'node:assert/strict';
import {miniPfnResults as run} from './mini-pfn-results.mjs';
import {bayesScenes} from './bayes-scenes.mjs';
import {
  MATCH_SURVIVAL, MISMATCH_SURVIVAL, survivalProbability, contextState, toyState,
  observedContext, reversedContext, expectedLogLoss, renderBayes, renderBayesMobile,
} from './bayes.mjs';

const near=(actual,expected,tolerance=1e-12)=>assert.ok(Math.abs(actual-expected)<tolerance,`${actual} != ${expected}`);
const attr=(svg,name)=>Number(svg.match(new RegExp(`${name}="([^"]+)"`))[1]);

test('both worlds preserve all four conditional survival probabilities',()=>{
  const p=1/(1+Math.exp(-1)),q=1-p;
  near(MATCH_SURVIVAL,p);near(MISMATCH_SURVIVAL,q);
  for(const [b,l,pa] of [[0,0,p],[0,1,q],[1,0,q],[1,1,p]]) {
    near(survivalProbability('A',b,l),pa);
    near(survivalProbability('B',b,l),1-pa);
  }
});

test('the evidence is exactly the real Mini-PFN context, including two deaths',()=>{
  assert.deepEqual(observedContext,run.examples.A.context);
  assert.deepEqual(reversedContext,run.examples.B.context);
  assert.deepEqual(observedContext.map(row=>row[2]),[1,0,0,1]);
  const state=contextState(observedContext);
  for(const likelihood of state.rowLikelihoods) {
    near(likelihood[0],MATCH_SURVIVAL);
    near(likelihood[1],MISMATCH_SURVIVAL);
    near(likelihood[0]/likelihood[1],Math.E);
  }
  near(state.likelihood[0],MATCH_SURVIVAL**4);
  near(state.likelihood[1],MISMATCH_SURVIVAL**4);
  near(state.likelihoodRatio,Math.exp(4));
  near(state.posterior[0],1/(1+Math.exp(-4)));
  near(state.prediction,run.examples.A.exact_bayes_probability,1e-7);
});

test('reversing context labels reverses posterior evidence and query prediction',()=>{
  const a=contextState(observedContext),b=contextState(reversedContext);
  near(a.posterior[0]+b.posterior[0],1);
  near(a.prediction+b.prediction,1);
  near(b.prediction,run.examples.B.exact_bayes_probability,1e-7);
  near(contextState([[0,1,0]]).posterior[0],MATCH_SURVIVAL);
  near(contextState([[0,1,1]]).posterior[0],MISMATCH_SURVIVAL);
});

test('changing only the query preserves world weights but changes predictive survival',()=>{
  const match=contextState(observedContext,[1,1]);
  const mismatch=contextState(observedContext,[1,0]);
  assert.deepEqual(match.posterior,mismatch.posterior);
  near(match.prediction+mismatch.prediction,1);
  near(match.prediction,0.7227468424124416);
  near(mismatch.prediction,0.2772531575875584);
});

test('compatibility helper and alternative priors retain exact stable inference',()=>{
  for(const n of [0,1,2,4]) {
    const s=toyState(n),actual=contextState(observedContext.slice(0,n));
    near(s.prediction,actual.prediction);
    near(s.posterior[0],actual.posterior[0]);
    near(s.matchPrediction+s.mismatchPrediction,1);
  }
  const skeptical=contextState(observedContext.slice(0,1),[1,1],[MISMATCH_SURVIVAL,MATCH_SURVIVAL]);
  near(skeptical.posterior[0],.5);near(skeptical.prediction,.5);
  near(toyState(1000).prediction,MATCH_SURVIVAL);
  near(toyState(1000,[0,1]).prediction,MISMATCH_SURVIVAL);
  near(toyState(1000,[1,0]).prediction,MATCH_SURVIVAL);
});

test('expected log loss is entropy plus nonnegative Bernoulli KL, minimized at the predictive target',()=>{
  const r=contextState(observedContext).prediction,H=expectedLogLoss(r,r);
  near(H,0.5903402794647451);
  for(const q of [.01,.12,.5,.7,r,.74,.9,.99]) {
    const kl=r*Math.log(r/q)+(1-r)*Math.log((1-r)/(1-q));
    near(expectedLogLoss(q,r),H+kl);
    assert.ok(expectedLogLoss(q,r)>=H-1e-12);
  }
  assert.ok(expectedLogLoss(.7,r)>expectedLogLoss(.71,r));
  assert.ok(expectedLogLoss(.8,r)<expectedLogLoss(.9,r));
  assert.equal(expectedLogLoss(0,r),Infinity);
  assert.equal(expectedLogLoss(1,r),Infinity);
});

test('six scenes preserve exact versus learned output provenance',()=>{
  assert.equal(bayesScenes.length,6);
  assert.deepEqual(bayesScenes.map(s=>s.bayesStage),[0,1,2,3,4,5]);
  assert.ok(bayesScenes.every(s=>s.kind==='bayes'));
  const comparison=renderBayes(4,1);
  for(const value of ['72.3%','27.7%','74.0%','23.6%'])assert.ok(comparison.includes(value));
  assert.ok(run.examples.A.probability>MATCH_SURVIVAL);
  assert.ok(run.examples.B.probability<MISMATCH_SURVIVAL);
  assert.match(bayesScenes[4].reading,/approximation errors/);
});

test('scrolling incorporates the actual rows and moves predictions reversibly',()=>{
  const before=renderBayes(2,0),middle=renderBayes(2,.5),after=renderBayes(2,1);
  near(attr(before,'data-posterior-a'),contextState(observedContext.slice(0,1)).posterior[0]);
  near(attr(after,'data-posterior-a'),contextState(observedContext).posterior[0]);
  assert.ok(attr(before,'data-posterior-a')<attr(middle,'data-posterior-a'));
  assert.ok(attr(middle,'data-posterior-a')<attr(after,'data-posterior-a'));
  assert.equal(renderBayes(2,0),before);
  assert.notEqual(before,middle);
  assert.notEqual(middle,after);
  const lossStart=renderBayes(5,0),lossEnd=renderBayes(5,1);
  near(attr(lossStart,'data-candidate-q'),.12);
  near(attr(lossEnd,'data-candidate-q'),contextState(observedContext).prediction);
  assert.ok(attr(lossStart,'data-expected-loss')>attr(lossEnd,'data-expected-loss'));
});

test('desktop and mobile have identical numeric states and no invalid SVG coordinates',()=>{
  for(let stage=0;stage<6;stage++)for(const progress of [0,.25,.5,.75,1]) {
    const desktop=renderBayes(stage,progress),mobile=renderBayesMobile(stage,progress);
    for(const name of ['data-posterior-a','data-prediction'])near(attr(desktop,name),attr(mobile,name));
    assert.match(mobile,/data-bayes-mobile="true"/);
    for(const svg of [desktop,mobile]) {
      assert.doesNotMatch(svg,/NaN|Infinity|vot|turnout|high-interest/i);
      const sizes=[...svg.matchAll(/font-size="(\d+)"/g)].map(m=>Number(m[1]));
      assert.ok(sizes.every(size=>size>=(svg===mobile?15:19)));
    }
  }
});

test('in-progress displays never move measured probability markers or claim four completed rows early',()=>{
  for(const render of [renderBayes,renderBayesMobile]) {
    const positions=svg=>[...svg.matchAll(/<rect x="([^"]+)"[^>]+data-learned-probability="([^"]+)"/g)].map(m=>[Number(m[1]),Number(m[2])]);
    assert.deepEqual(positions(render(4,0)),positions(render(4,1)));
    assert.deepEqual(positions(render(4,.5)),positions(render(4,1)));
    assert.equal(positions(render(4,.5)).length,2);
    assert.doesNotMatch(render(2,.5),/e⁴|Four rows/);
    assert.match(render(2,1),/e⁴/);
  }
});

test('invalid mathematical inputs are rejected',()=>{
  for(const n of [-1,.5,Infinity,NaN])assert.throws(()=>toyState(n),RangeError);
  for(const p of [[0,0],[-1,2],[NaN,1],[1],[Infinity,1]])assert.throws(()=>toyState(1,p),RangeError);
  assert.throws(()=>contextState([[0,0,2]]),RangeError);
  assert.throws(()=>contextState([[0,0,1]],[1]),RangeError);
  assert.throws(()=>survivalProbability('C',0,0),RangeError);
  assert.throws(()=>survivalProbability('A',2,0),RangeError);
  assert.throws(()=>expectedLogLoss(-.1,.5),RangeError);
});
