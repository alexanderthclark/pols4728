import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import test from 'node:test';
const require=createRequire(import.meta.url);
const model=require('../docs/trees/growth.js');
const view=require('../docs/trees/growth-view.js');
const attrs=tag=>Object.fromEntries([...tag.matchAll(/([\w-]+)="([^"]*)"/g)].map(m=>[m[1],m[2]]));
const panels=svg=>[...svg.matchAll(/<g data-map="[^"]+"[^>]*>/g)].map(m=>attrs(m[0]));

test('growth figures keep displayed predictions, scores and counts at the committed stage while cuts draw',()=>{
  for(const compact of [false,true])for(const scene of [0,1])for(const growth of [0,.25,.9,1,1.5,2,2.9,3,3.5,4]){
    const svg=view.render(scene,{growth},compact),stage=Math.floor(growth);
    const expected=(scene===0?model.treeStages:model.boostStages)[stage];
    const panel=panels(svg)[0],predictions=panel['data-predictions'].split(',').map(Number);
    assert.deepEqual(predictions,expected.predictions);
    const independentMSE=predictions.reduce((sum,p,i)=>sum+(p-model.observations[i].y)**2,0)/model.observations.length;
    assert.ok(Math.abs(Number(panel['data-mse'])-independentMSE)<1e-10);
    assert.equal(Number(panel['data-split-count']),stage);
    assert.equal(Number(panel['data-region-count']),expected.regionCount);
    assert.doesNotMatch(svg,/NaN|Infinity|undefined|data-probe|tabindex|role="button"/);
    assert.equal((svg.match(/data-observation="/g)||[]).length,16);
    for(const [,value] of svg.matchAll(/\s(?:x|y|x1|y1|x2|y2|cx|cy|width|height|r)="([^"]+)"/g))assert.ok(Number.isFinite(Number(value)));
    const pending=[...svg.matchAll(/<line\b[^>]*data-pending="true"[^>]*>/g)].map(m=>attrs(m[0]));
    assert.equal(pending.length,Number(!Number.isInteger(growth)));
    if(pending.length){
      const next=(scene===0?model.treeStages:model.boostStages)[stage+1];
      assert.deepEqual(pending[0]['data-cut-bounds'].split(',').map(Number),(scene===0?next.split.bounds:model.bounds).flat());
      assert.equal(Number(pending[0]['data-cut-threshold']),scene===0?next.split.threshold:next.stump.threshold);
    }
  }
});

test('final comparison uses equal split budgets and the two distinct computed fits',()=>{
  for(const compact of [false,true]){
    const svg=view.render(2,{},compact),map=panels(svg);
    assert.deepEqual(map.map(p=>Number(p['data-split-count'])),[4,4]);
    assert.deepEqual(map.map(p=>Number(p['data-region-count'])),[5,9]);
    assert.deepEqual(map[0]['data-predictions'].split(',').map(Number),model.treeStages[4].predictions);
    assert.deepEqual(map[1]['data-predictions'].split(',').map(Number),model.boostStages[4].predictions);
    assert.notEqual(map[0]['data-predictions'],map[1]['data-predictions']);
    assert.match(svg,/Training MSE 1\.625/);assert.match(svg,/Training MSE 1\.618/);
    assert.equal((svg.match(/data-observation="/g)||[]).length,32);
  }
});
