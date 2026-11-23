import test from 'node:test';
import assert from 'node:assert/strict';
import {scmFamilies,makeScmTask,evaluateScm,scmTasks} from './pretraining-data.mjs';
import {renderPretraining} from './pretraining.mjs';
import {bearComparisonResults} from './bear-comparison-results.mjs';
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-12,`${a} != ${b}`);
const textContents=s=>[...s.matchAll(/<text\b[^>]*>([^<]*)<\/text>/g)].map(m=>m[1]);

test('four original SCMs generate the declared schemas and independently checked structural equations',()=>{
  assert.deepEqual(scmFamilies.map(f=>f.id),['chain','latent','nonlinear','interaction']);
  const u=[.5,-.2,.1,.4,-.7];
  const chain=evaluateScm('chain',u);assert.deepEqual(chain.features,[.5,.34]);near(chain.score,.448);assert.equal(chain.target,1);
  const latent=evaluateScm('latent',u);near(latent.latent,.5);[.44,-.31,.41].forEach((v,i)=>near(latent.features[i],v));near(latent.score,.22);assert.equal(latent.target,1);
  const nonlinear=evaluateScm('nonlinear',u);assert.deepEqual(nonlinear.features,[.5,-.2,.4]);near(nonlinear.score,Math.sin(1.5)+.04+.035-.3);assert.equal(nonlinear.target,1);
  const interaction=evaluateScm('interaction',u);assert.deepEqual(interaction.features,[1,0]);near(interaction.score,-.85);assert.equal(interaction.target,0);
  assert.equal(evaluateScm('interaction',[.5,-.2,.9,.4,-.7]).target,1,'Exogenous outcome noise can reverse the binary interaction.');
  assert.equal(evaluateScm('nonlinear',[.5,-.2,.1,-.9,-.7]).target,nonlinear.target,'Changing the disconnected distractor does not change Y.');
});

test('each generated task has matching context/query rows but keeps query truth outside its model input',()=>{
  for(const family of scmFamilies){
    const task=makeScmTask(family.id,{seed:9128});assert.equal(task.context.length,4);assert.equal(task.featureNames.length,family.features.length);
    assert.deepEqual(Object.keys(task.query),['features']);assert.deepEqual(task.query.features,task.queryRecord.features);assert.equal(task.scoringTarget,task.queryRecord.target);
    for(const row of [...task.context,task.queryRecord]){const result=evaluateScm(family.id,row.noise);assert.deepEqual(row.features,result.features);assert.equal(row.target,result.target);near(row.score,result.score);}
    assert.deepEqual(task,makeScmTask(family.id,{seed:9128}));assert.notDeepEqual(task.context,makeScmTask(family.id,{seed:9129}).context);
  }
  for(const mobile of [false,true])for(const [index,p] of [0,.26,.51,.76].entries()){
    const svg=renderPretraining(1,p,mobile);assert.match(svg,new RegExp(`data-selected-family="${scmFamilies[index].id}"`));
    const input=svg.match(/<g data-model-input="true"[^>]*>(.*?)<\/g>/s)?.[1];assert.ok(input);assert.match(input,/data-query-target="hidden">\?<\/text>/);assert.doesNotMatch(input,/data-scoring-target/);
    const scoring=svg.match(/<text\b[^>]*data-scoring-target="true"[^>]*>([^<]*)<\/text>/)?.[1];assert.ok(scoring?.includes(String(scmTasks[index].scoringTarget)));
  }
});

test('schematic pretraining and measured bear inference retain separate output provenance',()=>{
  const measured=bearComparisonResults.cases.find(r=>r.model==='tabicl_v2'&&r.id==='continuous_noise');
  for(const mobile of [false,true]){
    const training=renderPretraining(2,1,mobile);assert.match(training,/data-weight-status="learning"/);assert.doesNotMatch(training,/data-benchmark-auc/);
    const comparison=training.match(/<text\b[^>]*data-training-comparison="true"[^>]*>([^<]*)<\/text>/)?.[1];assert.ok(comparison);
    const inference=renderPretraining(3,1,mobile);assert.match(inference,/data-weight-status="fixed"/);assert.match(inference,/data-query-target="hidden">\?<\/text>/);
    near(Number(inference.match(/data-benchmark-auc="([^\"]+)"/)[1]),measured.mean_auc);
    assert.ok(inference.includes(measured.mean_auc.toFixed(3)));assert.ok(inference.includes('12'));
    const prediction=textContents(inference.match(/<text\b[^>]*data-schematic-prediction="true"[^>]*>[^<]*<\/text>/)?.[0]??'').join('');assert.doesNotMatch(prediction,/\d|%/);
  }
});

test('all four scenes restore finite responsive geometry when scroll direction reverses',()=>{
  for(const mobile of [false,true])for(let stage=0;stage<4;stage++){
    const a=renderPretraining(stage,0,mobile),b=renderPretraining(stage,.5,mobile),c=renderPretraining(stage,1,mobile);
    assert.notEqual(a,c);assert.equal(a,renderPretraining(stage,0,mobile));assert.equal(b,renderPretraining(stage,.5,mobile));assert.equal(c,renderPretraining(stage,1,mobile));
    for(const p of [0,.25,.5,.75,1,.75,.5,.25,0]){const svg=renderPretraining(stage,p,mobile);assert.doesNotMatch(svg,/NaN|undefined|Infinity/);assert.match(svg,/<title>[^<]+<\/title><desc>[^<]+<\/desc>/);for(const [,key,value] of svg.matchAll(/\b(x|y|x1|x2|y1|y2|cx|cy|width|height)="([^\"]+)"/g)){const limit=['x','x1','x2','cx','width'].includes(key)?(mobile?400:900):(mobile?350:460);assert.ok(Number(value)>=0&&Number(value)<=limit,`${stage}: ${key}=${value}`);}}
  }
});
