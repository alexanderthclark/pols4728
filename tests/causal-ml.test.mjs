import test from 'node:test';
import assert from 'node:assert/strict';
import {people,mu,aipw,estimates,mean,plr,theta,m,ell,residualSlope,partialResiduals,position} from '../docs/causal-ml/model.mjs';
import {ivWorld,ivEstimate,ivOLS,ivMoment,q,compliance,wald,treeSample,leafStats,partition,treeCriterion,outcomeGain} from '../docs/causal-ml/advanced-model.mjs';
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);
test('constructed ATE and selection difference have their stated values',()=>{near(mean(people.map(o=>mu(1,o.x)-mu(0,o.x))),4);near(estimates().raw,13);near(estimates().ipw,4);});
test('AIPW exact four-case cancellation, including both misspecified',()=>{for(const [a,b,t] of [[true,true,4],[true,false,4],[false,true,4],[false,false,13]])near(mean(aipw(a,b).map(o=>o.phi)),t);near(estimates(false,true).outcome,2);});
test('each full residual follows the structural equation and recovers theta',()=>{for(const o of plr){near(o.d-m(o.x),o.v);near(o.y-ell(o.x),theta*o.v+o.u);}near(residualSlope(),2);});
test('partialling-out perturbation has the analytical second-order behavior',()=>{for(const [a,b] of [[.2,0],[0,.2],[.1,.1],[-.4,.3]])near(residualSlope(a,b),(1+2*a*b)/(.5+2*b*b));const h=1e-5;near((residualSlope(h,h)-residualSlope(-h,-h))/(2*h),0);});
test('scroll geometry returns to identical earlier coordinates',()=>{const before=partialResiduals(.4,.7);partialResiduals(.8,1);assert.deepEqual(before,partialResiduals(.4,.7));assert.equal(position(1,15).index,14);assert.equal(position(0,15).index,0);});
test('an endogenous treatment biases the residual slope while valid IV recovers the effect',()=>{
 const rows=ivWorld();assert.equal(rows.length,24);
 for(const o of rows){near(o.z-q(o.x),o.v);near(o.dt,o.v+o.u);near(o.yt,2*o.dt+o.u+o.e);}
 near(ivOLS(),2.5);near(ivEstimate().numerator,2);near(ivEstimate().denominator,1);near(ivEstimate().value,2);
 for(const pi of [.1,.3,2]){near(ivOLS(pi),2+1/(pi*pi+1));near(ivEstimate(ivWorld(pi)).value,2);}
 assert.equal(ivEstimate(ivWorld(0)).value,null);
});
test('IV nuisance perturbations are second order but exclusion violations bias the ratio',()=>{
 for(const t of [-.8,-.2,0,.2,.8])near(ivEstimate(ivWorld(),t).value,(2+2*t*t/3)/(1+2*t*t/3));
 for(const pi of [.2,1,2]){near(ivEstimate(ivWorld(pi,.6)).value,2+.6/pi);near(ivMoment(2+.08/pi,pi,.08),0);}
 const h=1e-5;near((ivEstimate(ivWorld(),h).value-ivEstimate(ivWorld(),-h).value)/(2*h),0);
});
test('binary IV isolates compliers and differs from the ATE',()=>{
 assert.ok(compliance.every(o=>o.d1>=o.d0));
 near(wald.first,.5);near(wald.reduced,2);near(wald.reduced/wald.first,4);near(wald.ate,3.8);
});
test('outcome and effect objectives prefer different initial covariate splits',()=>{
 const rows=treeSample();assert.equal(rows.length,64);
 near(leafStats(rows).tau,4);near(outcomeGain(rows,'x1'),1);near(outcomeGain(rows,'x2'),9);
 assert.deepEqual(partition(rows,'x1').map(l=>leafStats(l).tau),[6,2]);
 assert.deepEqual(partition(rows,'x2').map(l=>leafStats(l).tau),[4,4]);
 near(treeCriterion(rows,'root').fit,16);near(treeCriterion(rows,'x1').fit,20);near(treeCriterion(rows,'x2').fit,16);
});
test('honest tree criterion accounts for future estimation variance and precision splits',()=>{
 const rows=treeSample(),root=treeCriterion(rows),effect=treeCriterion(rows,'x1'),outcome=treeCriterion(rows,'x2'),four=treeCriterion(rows,'four');
 near(root.value,447/31);near(effect.penalty,41/15);near(outcome.penalty,13/15);near(four.penalty,5/7);
 assert.ok(effect.value>outcome.value&&effect.value>root.value);
 assert.ok(four.value>effect.value);assert.equal(four.fit,effect.fit);
 assert.ok(four.leaves.every(l=>l.n0===8&&l.n1===8));
 const small=partition(partition(rows,'four')[0],'x1',.25).map(leafStats);assert.ok(small.every(l=>l.n1===4&&l.n0===4));
});
test('honest estimates use a different realization and have the stated leaf uncertainty',()=>{
 const structure=treeSample(),estimation=treeSample('estimation'),ids=new Set(structure.map(o=>o.id));
 assert.ok(estimation.every(o=>!ids.has(o.id)));
 for(const o of [...structure,...estimation])near(o.y1-o.y0,o.x1<.5?6:2);
 const leaves=partition(estimation,'four').map(leafStats);
 leaves.forEach((l,i)=>{near(l.tau,i<2?6.4:1.8);near(l.s1,10/7);near(l.s0,10/7);near(l.se,Math.sqrt(5/14));});
});
test('selected differences in the separate constant-effect example are realization noise',()=>{
 const search=treeSample('search',true),confirm=treeSample('confirm',true);
 for(const o of [...search,...confirm])near(o.y1-o.y0,4);
 assert.deepEqual(partition(search,'x1').map(l=>leafStats(l).tau),[6,2]);
 assert.deepEqual(partition(confirm,'x1').map(l=>leafStats(l).tau),[4,4]);
 near(treeCriterion(search,'x1',.25).fit,52/3);near(treeCriterion(search,'x1',.5).fit,20);near(treeCriterion(search,'x1',.75).fit,52/3);
});
