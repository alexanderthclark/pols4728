import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {estimateFromAudit} from '../docs/llm-corrections/model.js';
import {regressionAt,renderVisual} from '../docs/llm-corrections/visuals.js';
import {scenes} from '../docs/llm-corrections/scenes.js';
import {sceneAt} from '../docs/llm-corrections/state.js';
const source=JSON.parse(readFileSync(new URL('../docs/llm-corrections/example-results.json',import.meta.url)));
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,`${a} differs from ${b}`);
const katex=createRequire(import.meta.url)('../docs/transformers/vendor/katex/katex.min.js');

test('the same original audit reproduces all corrected moments and estimates',()=>{
  const result=estimateFromAudit(source.rows);
  [3,11.0625,82.25].forEach((m,i)=>close(result.moments[i].corrected,m));
  close(result.fits.dsl.slope,68/33);
  close(result.fits.expert_only.slope,1.705989110707804);
  close(result.population_se.dsl,source.population_se.dsl);
  close(result.population_se.expert_only,source.population_se.expert_only);
});

test('unobserved expert labels are never consulted by the feasible estimator',()=>{
  const hidden=source.rows.map(r=>r.audited?{...r}:{q:r.q,y:r.y,audited:false,get x(){throw new Error('Unobserved expert score accessed');}});
  assert.deepEqual(estimateFromAudit(hidden),estimateFromAudit(source.rows));
});

test('expert-coding the entire corpus recovers the complete-data regression',()=>{
  const full=estimateFromAudit(source.rows.map(r=>({...r,audited:true})));
  [3,11,82].forEach((m,i)=>close(full.moments[i].corrected,m));
  close(full.fits.dsl.slope,2);
  close(full.audit_only_se.dsl,0);
});

test('the moving regression fits actual intermediate positions, and shifts leave its slope unchanged',()=>{
  for(const p of [0,.13,.4,.5,.81,1]){
    const a=1-p,fit=regressionAt(source.rows,p);
    close(fit.slope,(4+3*a)/(2+.25*a*a));
    close(fit.meanX,3);
    for(const shift of [-1,-.5,.5,1])close(regressionAt(source.rows,p,shift).slope,fit.slope);
  }
});

test('scroll progress is reversible; reduced motion supplies finished scene states',()=>{
  const forward=[0,200,450,850,1200,1800].map(x=>sceneAt(x,1900,19));
  const backward=[1800,1200,850,450,200,0].map(x=>sceneAt(x,1900,19)).reverse();
  assert.deepEqual(forward,backward);
  assert.equal(sceneAt(1900,1900,19).index,18);
  assert.equal(sceneAt(450,1900,19,true).progress,1);
});

test('every desktop and mobile equation typesets, and visual states have finite geometry',()=>{
  for(const scene of scenes){
    for(const tex of [scene.equation,scene.equationMobile].filter(Boolean))assert.doesNotThrow(()=>katex.renderToString(tex,{throwOnError:true}));
    for(const [width,height] of [[640,430],[360,310],[360,200]])for(const progress of [0,.5,1]){
      const el={clientWidth:width,clientHeight:height,dataset:{},innerHTML:'',querySelectorAll(){return[];}};
      renderVisual(el,scene.visual,progress,source,{mobile:width<760,workflow:scene.workflow});
      assert.ok(el.innerHTML.length>0);
      assert.doesNotMatch(el.innerHTML,/NaN|undefined|Infinity/);
      for(const match of el.innerHTML.matchAll(/data-visual-math="([^"]+)"/g)){
        const tex=match[1].replace(/&amp;/g,'&').replace(/&quot;/g,'"').replace(/&#39;/g,"'").replace(/&lt;/g,'<').replace(/&gt;/g,'>');
        assert.doesNotThrow(()=>katex.renderToString(tex,{throwOnError:true}));
      }
    }
  }
});
