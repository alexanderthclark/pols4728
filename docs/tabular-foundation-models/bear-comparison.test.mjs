import test from 'node:test';
import assert from 'node:assert/strict';
import {bearComparisonResults as results} from './bear-comparison-results.mjs';
import {renderBearComparison} from './bear-comparison.mjs';

const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-12,`${a} != ${b}`);
test('three-model comparisons use every paired simulation seed and exact summaries',()=>{
  const seeds=results.metadata.seeds;
  assert.deepEqual(seeds,Array.from({length:12},(_,i)=>i));
  assert.notEqual(results.metadata.incomplete_preview_only,true);
  assert.equal(results.cases.length,6);
  assert.equal(results.records.length,6*seeds.length);
  for(const summary of results.cases){
    const rows=results.records.filter(r=>r.model===summary.model&&r.id===summary.id);
    assert.deepEqual(rows.map(r=>r.seed).sort((a,b)=>a-b),seeds);
    for(const row of rows){
      assert.equal(row.n_train,results.metadata.n_train);
      assert.equal(row.n_test,results.metadata.n_test);
      assert.ok(row.auc>=.46&&row.auc<=.78,'The published AUC must fit inside the displayed scale.');
    }
    const mean=rows.reduce((s,r)=>s+r.auc,0)/rows.length;
    near(mean,summary.mean_auc);
    if(rows.length>1)near(Math.sqrt(rows.reduce((s,r)=>s+(r.auc-mean)**2,0)/(rows.length-1)),summary.sd_auc);
  }
});

test('comparison endpoints preserve measured geometry in both responsive layouts',()=>{
  const order=['tree_depth2','random_forest','tabicl_v2'];
  for(const mobile of [false,true])for(let stage=0;stage<3;stage++){
    const end=renderBearComparison(stage,1,mobile),start=renderBearComparison(stage,0,mobile);
    assert.notEqual(start,end);
    assert.equal(start,renderBearComparison(stage,0,mobile));
    assert.equal(end,renderBearComparison(stage,1,mobile));
    assert.doesNotMatch(end,/NaN|undefined|Infinity/);
    const left=mobile?24:260,right=mobile?376:810;
    const dots=[...end.matchAll(/<circle cx="([^"]+)"[^>]+data-seed="(\d+)" data-case="([^"]+)" data-auc="([^"]+)"/g)];
    assert.equal(dots.length,2*results.metadata.seeds.length*(stage+1));
    for(const [,x,,,auc] of dots)assert.ok(Math.abs(Number(x)-(left+(Number(auc)-.46)/.32*(right-left)))<.000051);
    assert.equal((end.match(/data-endpoint="true"/g)||[]).length,stage+1);
    for(const model of order.slice(0,stage+1)){
      const clean=results.cases.find(r=>r.model===model&&r.id==='signal_only');
      const noise=results.cases.find(r=>r.model===model&&r.id==='continuous_noise');
      assert.ok(end.includes(`${clean.mean_auc.toFixed(3)} / ${noise.mean_auc.toFixed(3)}`));
    }
    const middle=renderBearComparison(stage,.5,mobile);
    assert.match(middle,/data-mark="mean-label"[^>]*><\/text>/);
  }
});
