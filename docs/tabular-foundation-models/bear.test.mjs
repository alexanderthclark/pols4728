import test from 'node:test';
import assert from 'node:assert/strict';
import {bearResults} from './bear-results.mjs';
import {renderBear,renderBearMobile,bearAucCases} from './bear.mjs';
import {scenes} from './scenes.mjs';
import {renderVisual,renderMobile} from './visuals.mjs';
import {renderBayesMobile} from './bayes.mjs';
import {renderDgp} from './dgp.mjs';
import {renderMechanism} from './mechanism.mjs';
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-12,`${a} != ${b}`);

test('published means and sample SDs are supported by all twelve seed results',()=>{
  assert.equal(bearResults.records.length,120);
  assert.equal(bearResults.metadata.tabpfn_evaluated,false);
  for(const result of bearResults.cases){
    const records=bearResults.records.filter(r=>r.id===result.id&&r.depth===result.depth);
    assert.deepEqual(records.map(r=>r.seed),Array.from({length:12},(_,i)=>i));
    const mean=records.reduce((sum,r)=>sum+r.auc,0)/records.length;
    const sd=Math.sqrt(records.reduce((sum,r)=>sum+(r.auc-mean)**2,0)/(records.length-1));
    near(mean,result.mean_auc);near(sd,result.sd_auc);
  }
});

test('every plotted AUC uses the measured depth-two result',()=>{
  for(const item of bearAucCases){
    const measured=bearResults.cases.find(r=>r.id===item.id&&r.depth===2);
    near(item.auc,measured.mean_auc);
    for(const render of [renderBear,renderBearMobile]){
      const svg=render(4,1);
      assert.ok(svg.includes(`data-case="${item.id}" data-auc="${measured.mean_auc}"`));
    }
  }
});

test('worked episodes preserve retained scene routing and every scroll state renders',()=>{
  assert.equal(scenes.length,28);
  assert.equal(scenes[0].kind,'dgp');
  const old=scenes.filter(s=>s.visualIndex!==undefined);
  assert.deepEqual(old.map(s=>s.visualIndex),[0,1,10,11,12,13,14,15,16,17]);
  assert.deepEqual(scenes.filter(s=>s.kind==='bear').map(s=>s.bearStage),[0,1,2,3,4,5,6]);
  for(const scene of scenes)for(const mobile of [false,true])for(const p of [0,.25,.5,.75,1]){
    const index=scene.visualIndex;
    const svg=scene.kind==='mechanism'?renderMechanism(scene.mechanismStage,p,mobile):scene.kind==='dgp'?renderDgp(p,mobile):scene.kind==='bear'?(mobile?renderBearMobile(scene.bearStage,p):renderBear(scene.bearStage,p)):
      mobile?(index>=10&&index<=14?renderBayesMobile(index-10,p):renderMobile(index,p)):renderVisual(index,p);
    assert.ok(svg.length>100);
    assert.doesNotMatch(svg,/NaN|undefined|Infinity/);
  }
});

test('bear visual progress is reversible and completed states preserve both probabilities',()=>{
  for(const render of [renderBear,renderBearMobile]){
    for(let stage=0;stage<7;stage++){
      const first=render(stage,0),last=render(stage,1);
      assert.notEqual(first,last);
      assert.equal(first,render(stage,0));
    }
    assert.match(render(1,1),/73\.1%/);
    assert.match(render(1,1),/26\.9%/);
  }
  near(bearResults.metadata.p_survival_match+bearResults.metadata.p_survival_mismatch,1);
});
