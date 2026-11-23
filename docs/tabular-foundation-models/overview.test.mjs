import test from 'node:test';
import assert from 'node:assert/strict';
import {overviewScenes} from './overview-scenes.mjs';
import {renderOverview} from './overview.mjs';

const texts=svg=>[...svg.matchAll(/<text\b[^>]*>([^<]*)<\/text>/g)].map(match=>match[1]);
const query=svg=>svg.match(/<g data-overview-query="true">(.*?)<\/g>/s)?.[1];
const targets=svg=>[...svg.matchAll(/<text\b[^>]*data-context-target="true"[^>]*>([^<]*)<\/text>/g)].map(match=>match[1]);
const geometry=svg=>[...svg.matchAll(/<(?:rect|line|circle|path)\b[^>]*>/g)].map(match=>match[0]).join('');
const attrs=tag=>Object.fromEntries([...tag.matchAll(/([\w-]+)="([^"]*)"/g)].map(m=>[m[1],m[2]]));

test('five early scenes explain pretraining, the prompt, cell computation, roles, and table sizes',()=>{
  assert.deepEqual(overviewScenes.map(s=>[s.kind,s.overviewStage]),Array.from({length:5},(_,i)=>['overview',i]));
  assert.match(overviewScenes[0].body,/not supplied again/);
  assert.match(overviewScenes[1].title,/table is the prompt/);
  assert.match(overviewScenes[1].body,/including their labels/);
  assert.match(overviewScenes[2].body,/Embed the cells/);
  assert.match(overviewScenes[2].reading,/not separate Transformer encoder and decoder stacks/);
  assert.match(overviewScenes[3].body,/not generated left to right/);
  assert.match(overviewScenes[4].reading,/adding a predictor requires values in both/);
  for(const s of overviewScenes){assert.ok(s.reading.split(/\s+/).length<=80);assert.ok(s.references.length>0);}
  assert.ok(overviewScenes[2].references.some(r=>r.title.includes('4.6')));
  assert.match(overviewScenes[1].detailEquation,/begin\{bmatrix\}/);
  assert.match(overviewScenes[2].detailEquation,/d_\{\\rm model\}/);
});

test('known labels and query features are inputs while the query target remains hidden',()=>{
  for(const mobile of [false,true])for(const stage of [1,2])for(const p of [0,.25,.5,.75,1]){
    const svg=renderOverview(stage,p,mobile),q=query(svg);
    assert.match(svg,/data-overview-input="whole-table"/);
    assert.deepEqual(targets(svg),['0','1','1','0']);
    assert.ok(q);assert.deepEqual(texts(q),['.6','.2','?']);
    assert.match(q,/data-query-target="hidden">\?<\/text>/);
    assert.match(svg,/data-weight-status="fixed"/);
    assert.doesNotMatch(svg,/data-weight-status="learning"/);
  }
});

test('cell-based computation embeds labels and the placeholder, then reads the final query-target vector',()=>{
  for(const mobile of [false,true])for(const p of [0,.55,1]){
    const svg=renderOverview(2,p,mobile);
    assert.match(svg,/data-table-tensor="true"/);
    assert.equal([...svg.matchAll(/data-cell-role="known-label"/g)].length,4);
    assert.equal([...svg.matchAll(/data-cell-role="context-feature"/g)].length,8);
    assert.equal([...svg.matchAll(/data-cell-role="query-feature"/g)].length,2);
    assert.equal([...svg.matchAll(/data-cell-role="query-target-placeholder"/g)].length,1);
    assert.match(svg,/data-final-query-target="true"/);
    assert.match(svg,/data-readout-source="query-target"/);
    const output=texts(svg).filter(t=>t.startsWith('Pr('));
    assert.deepEqual(output,['Pr(Y = 0)','Pr(Y = 1)']);
    assert.doesNotMatch(svg,/>\s*0\.\d+\s*<|>\s*\d+%\s*</,'No fabricated probability values.');
    assert.match(svg,/Decode/);assert.match(svg,/prediction head/);
  }
});

test('the attention schematic permits context exchange and within-query attention without cross-query leakage',()=>{
  for(const mobile of [false,true]){
    const svg=renderOverview(3,1,mobile);
    const edges=[...svg.matchAll(/<g\b[^>]*data-attention-source="[^"]+"[^>]*>/g)].map(m=>attrs(m[0]));
    assert.equal(edges.filter(e=>e['data-source-role']==='context'&&e['data-reader-role']==='context').length,6);
    assert.equal(edges.filter(e=>e['data-source-role']==='context'&&e['data-reader-role']==='query').length,2);
    const within=edges.filter(e=>e['data-source-role']==='query');assert.equal(within.length,4);
    for(const edge of within){
      const id=edge['data-query-id'];assert.ok(id);
      assert.equal(edge['data-reader-role'],'query');
      assert.ok(edge['data-attention-source'].startsWith(id+'-'));
      assert.ok(edge['data-attention-reader'].startsWith(id+'-'));
    }
    assert.equal([...svg.matchAll(/data-query-target="hidden">\?<\/text>/g)].length,2);
  }
});

test('different task shapes retain matching context/query feature counts and use fixed weights',()=>{
  for(const mobile of [false,true]){
    const svg=renderOverview(4,1,mobile);
    const tasks=[...svg.matchAll(/<g\b[^>]*data-size-task="[^"]+"[^>]*>/g)].map(m=>attrs(m[0]));
    assert.equal(tasks.length,2);
    assert.deepEqual(tasks.map(t=>Number(t['data-feature-count'])),[2,4]);
    assert.notEqual(tasks[0]['data-context-count'],tasks[1]['data-context-count']);
    for(const t of tasks)assert.equal(t['data-query-feature-count'],t['data-feature-count']);
    assert.match(svg,/data-weight-status="fixed"/);assert.match(svg,/Separate/);
    assert.equal([...svg.matchAll(/data-query-target="hidden">\?<\/text>/g)].length,2);
  }
});

test('both responsive layouts use finite, bounded, reversible scroll geometry',()=>{
  for(const mobile of [false,true])for(let stage=0;stage<5;stage++){
    const width=mobile?400:900,height=mobile?350:460;
    const initial=renderOverview(stage,0,mobile),middle=renderOverview(stage,.5,mobile),end=renderOverview(stage,1,mobile);
    assert.notEqual(geometry(initial),geometry(end));
    for(const p of [0,.25,.5,.75,1,.75,.5,.25,0]){
      const svg=renderOverview(stage,p,mobile);
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
