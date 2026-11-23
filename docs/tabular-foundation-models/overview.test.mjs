import test from 'node:test';
import assert from 'node:assert/strict';
import {overviewScenes} from './overview-scenes.mjs';
import {renderOverview} from './overview.mjs';

const texts=svg=>[...svg.matchAll(/<text\b[^>]*>([^<]*)<\/text>/g)].map(match=>match[1]);
const query=svg=>svg.match(/<g data-overview-query="true">(.*?)<\/g>/s)?.[1];
const targets=svg=>[...svg.matchAll(/<text\b[^>]*data-context-target="true"[^>]*>([^<]*)<\/text>/g)].map(match=>match[1]);
const geometry=svg=>[...svg.matchAll(/<(?:rect|line|circle|path)\b[^>]*>/g)].map(match=>match[0]).join('');

test('three abstract scenes introduce pretraining, context, and internal computation in order',()=>{
  assert.equal(overviewScenes.length,3);
  assert.deepEqual(overviewScenes.map(scene=>[scene.kind,scene.overviewStage]),[['overview',0],['overview',1],['overview',2]]);
  assert.match(overviewScenes[0].title,/Pretraining/);
  assert.match(overviewScenes[1].title,/In-context learning/);
  assert.match(overviewScenes[2].equation,/encode.*exchange.*decode/);
  for(const scene of overviewScenes){
    assert.ok(scene.reading.split(/\s+/).length<=80);
    assert.ok(scene.references.some(reference=>new URL(reference.url).hostname==='tabularfoundationmodels.com'));
  }
});

test('in-context scenes keep weights fixed and withhold the same query target',()=>{
  for(const mobile of [false,true]){
    const originalQuery=query(renderOverview(1,0,mobile));
    assert.ok(originalQuery);
    assert.match(originalQuery,/data-query-target="hidden">\?<\/text>/);
    assert.deepEqual(texts(originalQuery),['New row','.6','.2','?','Target hidden']);
    for(const stage of [1,2])for(const progress of [0,.25,.5,.75,1]){
      const svg=renderOverview(stage,progress,mobile);
      assert.match(svg,/data-weight-status="fixed">Weights fixed<\/text>/);
      assert.doesNotMatch(svg,/data-weight-status="learning"/);
      assert.equal(query(svg),originalQuery,'Changing context or scroll must not reveal or change the query.');
    }
  }
});

test('changing context changes labeled examples while outputs stay explicitly unnumbered',()=>{
  for(const mobile of [false,true]){
    const before=renderOverview(2,0,mobile),after=renderOverview(2,1,mobile);
    assert.match(before,/data-context-label="true">Labeled context A<\/text>/);
    assert.match(after,/data-context-label="true">Labeled context B<\/text>/);
    assert.notDeepEqual(targets(before),targets(after));
    assert.equal(targets(before).length,4);
    assert.equal(targets(after).length,4);
    for(const stage of [1,2])for(const progress of [0,.5,1]){
      const output=renderOverview(stage,progress,mobile).match(/<g data-overview-output="schematic">(.*?)<\/g>/s)?.[1];
      assert.ok(output);
      const outputText=texts(output).join(' ');
      assert.match(outputText,/Prediction/);
      assert.doesNotMatch(outputText,/\d|%/,'Schematic outputs must not fabricate numerical predictions.');
    }
  }
});

test('both responsive layouts use finite, bounded, reversible scroll geometry',()=>{
  for(const mobile of [false,true])for(let stage=0;stage<3;stage++){
    const width=mobile?400:900,height=mobile?350:460;
    const initial=renderOverview(stage,0,mobile),middle=renderOverview(stage,.5,mobile),end=renderOverview(stage,1,mobile);
    assert.notEqual(geometry(initial),geometry(end),'Scrolling must move geometry, not only change copy or opacity.');
    for(const progress of [0,.25,.5,.75,1,.75,.5,.25,0]){
      const svg=renderOverview(stage,progress,mobile);
      assert.doesNotMatch(svg,/NaN|undefined|Infinity/);
      assert.match(svg,/<title>[^<]+<\/title><desc>[^<]+<\/desc>/);
      for(const [,name,value] of svg.matchAll(/\b(x|x1|x2|cx|y|y1|y2|cy|width|height)="([^"]+)"/g)){
        const limit=name.startsWith('x')||name==='cx'||name==='width'?width:height;
        assert.ok(Number.isFinite(Number(value))&&Number(value)>=0&&Number(value)<=limit,`${name}=${value} exceeds its viewBox.`);
      }
    }
    assert.equal(renderOverview(stage,.5,mobile),middle);
    assert.equal(renderOverview(stage,1,mobile),end);
    assert.equal(renderOverview(stage,0,mobile),initial);
  }
});
