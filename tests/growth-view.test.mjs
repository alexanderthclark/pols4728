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

const luminance=rgb=>rgb.map(v=>v/255).map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4).reduce((sum,v,i)=>sum+v*[.2126,.7152,.0722][i],0);
const rgb=fill=>fill==='white'?[255,255,255]:fill==='#000000'?[0,0,0]:fill.match(/\d+/g).map(Number);

test('observed and fitted colors use one fixed quantitative scale with readable labels',()=>{
  assert.deepEqual(view.timeDomain,[8,24]);
  assert.equal(view.timeColor(8),'rgb(237,244,250)');
  assert.equal(view.timeColor(24),'rgb(35,78,112)');
  for(let value=8;value<24;value+=.25) assert.ok(luminance(rgb(view.timeColor(value)))>luminance(rgb(view.timeColor(value+.25))));
  const seenColors=new Map();
  for(const compact of [false,true])for(const scene of [0,1,2])for(const growth of [0,1,2,3,4]){
    const svg=view.render(scene,{growth},compact);
    assert.match(svg,/data-color-scale="8,24" data-color-quantity="predicted-F"/);
    assert.match(svg,/Predicted F \(minutes\)/);
    const cells=[...svg.matchAll(/<rect\b[^>]*data-region="[^>]*>/g)].map(m=>attrs(m[0]));
    const labels=[...svg.matchAll(/<text\b[^>]*class="prediction"[^>]*>/g)].map(m=>attrs(m[0]));
    assert.equal(cells.length,labels.length);
    cells.forEach((cell,i)=>{
      const value=Number(cell['data-value']);
      assert.equal(cell.fill,view.timeColor(value));
      if(seenColors.has(value))assert.equal(cell.fill,seenColors.get(value));
      seenColors.set(value,cell.fill);
      const a=luminance(rgb(cell.fill)),b=luminance(rgb(labels[i].fill));
      assert.ok((Math.max(a,b)+.05)/(Math.min(a,b)+.05)>=4.5,`unreadable label at ${value}`);
    });
  }
});

test('transition shows the same 16 observed deliveries in outcome and predictor coordinates',()=>{
  for(const compact of [false,true]){
    const svg=view.renderTransition(compact);
    assert.match(svg,/New|new example/);
    assert.match(svg,/data-color-scale="8,24" data-color-quantity="observed-y"/);
    assert.match(svg,/Observed y \(minutes\)/);
    assert.match(svg,/16 ×2/);
    assert.match(svg,/data-overlap-count="2"/);
    assert.doesNotMatch(svg,/data-region=|NaN|Infinity|undefined|tabindex/);
    for(const kind of ['outcome','input']){
      const panel=svg.match(new RegExp(`<g data-observation-panel="${kind}">([\\s\\S]*?)<\\/g>`))[1];
      const marks=[...panel.matchAll(/<(?:circle|rect)\b[^>]*data-observation="[^>]*>/g)].map(m=>attrs(m[0]));
      assert.equal(marks.length,16);
      assert.deepEqual(marks.map(m=>m['data-observation']),model.observations.map(o=>o.id));
      marks.forEach((mark,i)=>{
        const observation=model.observations[i];
        assert.deepEqual(mark['data-x'].split(',').map(Number),observation.x);
        assert.equal(Number(mark['data-y']),observation.y);
        assert.equal(mark.fill,view.timeColor(observation.y));
      });
    }
  }
});
