import test from 'node:test';
import assert from 'node:assert/strict';
import {bearResults} from './bear-results.mjs';
import {renderBear,renderBearMobile,bearAucCases,bearOracleAuc} from './bear.mjs';
import {scenes} from './scenes.mjs';
import {renderVisual,renderMobile} from './visuals.mjs';
import {renderBayes,renderBayesMobile,MATCH_SURVIVAL} from './bayes.mjs';
import {bayesScenes} from './bayes-scenes.mjs';
import {mechanismScenes} from './mechanism-scenes.mjs';
import {renderDgp} from './dgp.mjs';
import {renderMechanism} from './mechanism.mjs';
import {natureScenes} from './nature-scenes.mjs';
import {renderNature,renderNatureMobile} from './nature.mjs';
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

test('all sixty seed dots preserve exact test AUCs on a common vertical scale',()=>{
  for(const [mobile,render] of [[false,renderBear],[true,renderBearMobile]]){
    const svg=render(4,1);
    const dots=[...svg.matchAll(/<circle([^>]+)data-mark="seed" data-case="([^"]+)" data-seed="(\d+)" data-auc="([^"]+)"/g)];
    assert.equal(dots.length,60);
    for(const [,attrs,id,seed,auc] of dots){
      const record=bearResults.records.find(r=>r.depth===2&&r.id===id&&r.seed===Number(seed));
      near(Number(auc),record.auc);
      const cy=Number(attrs.match(/cy="([^"]+)"/)[1]);
      const top=mobile?75:82,bottom=mobile?242:332;
      near(cy,bottom-(record.auc-.46)/(.78-.46)*(bottom-top));
    }
  }
});

test('seed results appear before means and reverse scrolling restores the reveal',()=>{
  for(const render of [renderBear,renderBearMobile]){
    const beginning=render(4,0),dots=render(4,.65),means=render(4,1);
    assert.match(beginning,/data-seeds-visible="0"/);
    assert.match(dots,/data-seeds-visible="12" data-means-visible="false"/);
    assert.match(means,/data-seeds-visible="12" data-means-visible="true"/);
    assert.equal(render(4,.65),dots);
    assert.equal(render(4,0),beginning);
    assert.equal((dots.match(/data-mark="seed"/g)||[]).length,60);
  }
});

test('oracle AUC is the population ranking reference, which sample AUC may exceed',()=>{
  const p=MATCH_SURVIVAL;
  near(bearOracleAuc,p*p+.5*(2*p*(1-p)));
  assert.ok(bearResults.records.some(r=>r.depth===2&&r.auc>bearOracleAuc));
  for(const render of [renderBear,renderBearMobile]){
    assert.ok(render(4,1).includes(`data-oracle-auc="${bearOracleAuc}"`));
    assert.match(render(4,1),/not a test-sample ceiling/);
  }
});

test('worked episodes preserve retained scene routing and every scroll state renders',()=>{
  assert.equal(scenes.length,12+mechanismScenes.length+bayesScenes.length+natureScenes.length);
  assert.equal(scenes[0].kind,'dgp');
  const old=scenes.filter(s=>s.visualIndex!==undefined);
  assert.deepEqual(old.map(s=>s.visualIndex),[0,1,16,17]);
  assert.deepEqual(scenes.filter(s=>s.kind==='bear').map(s=>s.bearStage),[0,1,2,3,4,5,6]);
  for(const scene of scenes)for(const mobile of [false,true])for(const p of [0,.25,.5,.75,1]){
    const index=scene.visualIndex;
    const svg=scene.kind==='nature'?(mobile?renderNatureMobile(scene.natureStage,p):renderNature(scene.natureStage,p)):scene.kind==='bayes'?(mobile?renderBayesMobile(scene.bayesStage,p):renderBayes(scene.bayesStage,p)):scene.kind==='mechanism'?renderMechanism(scene.mechanismStage,p,mobile):scene.kind==='dgp'?renderDgp(p,mobile):scene.kind==='bear'?(mobile?renderBearMobile(scene.bearStage,p):renderBear(scene.bearStage,p)):
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
