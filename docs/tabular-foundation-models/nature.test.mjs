import test from 'node:test';
import assert from 'node:assert/strict';
import {renderNature,renderNatureMobile} from './nature.mjs';

test('Nature scenes restore earlier states when scrolling backward',()=>{
  for(const render of [renderNature,renderNatureMobile])for(let stage=0;stage<3;stage++){
    const start=render(stage,0),middle=render(stage,.5),end=render(stage,1);
    assert.notEqual(start,end);
    assert.equal(render(stage,.5),middle);
    assert.equal(render(stage,0),start);
    for(const svg of [start,middle,end])assert.doesNotMatch(svg,/NaN|undefined|Infinity/);
  }
});

test('sample-efficiency bars keep the reported 2:1 row-count ratio at every scroll state',()=>{
  for(const render of [renderNature,renderNatureMobile])for(const progress of [0,.25,.5,.75,1]){
    const svg=render(2,progress);
    const bars=[...svg.matchAll(/<rect ([^>]*data-model="(catboost|tabpfn)"[^>]*)\/>/g)];
    assert.equal(bars.length,2);
    const values=Object.fromEntries(bars.map(([,attrs,model])=>[model,{
      x:Number(attrs.match(/\bx="([^"]+)"/)[1]),
      width:Number(attrs.match(/\bwidth="([^"]+)"/)[1]),
      fraction:Number(attrs.match(/data-fraction="([^"]+)"/)[1])
    }]));
    assert.equal(values.catboost.fraction,1);
    assert.equal(values.tabpfn.fraction,.5);
    assert.equal(values.catboost.x,values.tabpfn.x);
    assert.equal(values.catboost.width,2*values.tabpfn.width);
  }
});
