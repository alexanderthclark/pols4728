import test from 'node:test';
import assert from 'node:assert/strict';
import {mean} from '../docs/causal-ml/model.mjs';
import {metaRows,metaX,metaMu,metaTau,metaE,metaEll,fitS,tFit,xPseudo,xArm,xBlend,drRows,drFit,rRows,rFit,rLoss,rLocal,rInformation} from '../docs/causal-ml/meta-model.mjs';
const near=(a,b,tolerance=1e-10)=>assert.ok(Math.abs(a-b)<tolerance,a+' != '+b);

test('shared meta-learner world has exact overlap, conditional means, and potential-outcome effects',()=>{
 assert.equal(metaRows.length,80);
 for(const x of metaX){
  const group=metaRows.filter(o=>o.x===x);assert.equal(group.length,16);near(mean(group.map(o=>o.d)),metaE(x));
  near(mean(group.map(o=>o.y)),metaEll(x));
  for(const d of [0,1])near(mean(group.filter(o=>o.d===d).map(o=>o.y)),metaMu(d,x));
 }
 for(const o of metaRows)near(o.y1-o.y0,metaTau(o.x));
 near(mean(metaRows.map(o=>o.y1-o.y0)),4);
});
test('S-learner fits distinguish full interactions, an additive constraint, and treatment-term regularization',()=>{
 const full=fitS(),additive=fitS(0,false),penalized=fitS(.01);
 full.beta.forEach((v,i)=>near(v,[20,8,4,6,-4][i]));
 for(const x of metaX){near(full.effect(x),metaTau(x));near(additive.effect(x),4);}
 near(full.mse,1);near(penalized.effect(0),4.850061729107111);near(penalized.effect(1),2.706504055660119);
 near(penalized.mse,1.0955666279288991);assert.ok(penalized.mse<1.1);
});
test('two separately fitted quadratic outcomes recover the T contrast',()=>{
 for(const x of [0,.1,.25,.5,.9,1])for(const d of [0,1])near(tFit.predict(x,d),metaMu(d,x));
 near(tFit.predict(.25,0),22.25);near(tFit.predict(.25,1),27.25);near(tFit.effect(.25),5);
});
test('X pseudo-outcome signs, arm regressions, and propensity blend follow their stated convention',()=>{
 for(const o of xPseudo())near(o.pseudo,metaTau(o.x)+(o.d?o.noise:-o.noise));
 for(const x of metaX)for(const d of [0,1])near(xArm(x,d),metaTau(x));
 near(xBlend(6,5.2,.5),5.6);near(xBlend(6,5.2,.2),5.36);
 // A small g puts most weight on tau_1, learned in the treated arm.
 assert.ok(Math.abs(xBlend(6,5.2,.2)-5.2)<Math.abs(xBlend(6,5.2,.5)-5.2));
});
test('DR conditional means are correct in three nuisance cases and can be wrong despite the correct ATE',()=>{
 for(const [a,b] of [[true,true],[false,true],[true,false],[false,false]]){
  const rows=drRows(a,b);
  for(const x of metaX)near(mean(rows.filter(o=>o.x===x).map(o=>o.phi)),metaTau(x)+(!a&&!b?.5-x:0));
  near(drFit(a,b).average,4);
 }
 const one=drRows(false,true).find(o=>o.x===0&&o.d===1&&o.noise===1);
 near(one.base,3);near(one.correction,8);near(one.phi,11);
 drFit(false,true).beta.forEach((v,i)=>near(v,[6,-4][i]));
 drFit(false,false).beta.forEach((v,i)=>near(v,[6.5,-5][i]));
});
test('R residual identity and actual varying-slope fit recover the constructed effect',()=>{
 for(const o of rRows())near(o.yt,o.dt*metaTau(o.x)+o.noise);
 rFit().beta.forEach((v,i)=>near(v,[6,-4][i]));
 near(metaEll(.25),metaMu(0,.25)+metaE(.25)*metaTau(.25));
});
test('R-loss equals the squared-weight quotient loss, not unweighted quotient regression',()=>{
 for(const c of [-2,0,1.5]){
  const candidate=x=>metaTau(x)+c,weighted=mean(rRows().map(o=>o.dt**2*(o.yt/o.dt-candidate(o.x))**2));
  near(rLoss(candidate),weighted);near(rLoss(candidate),1+.21875*c*c);
 }
 const unweighted=mean(rRows().map(o=>(o.yt/o.dt-metaTau(o.x))**2));
 assert.ok(Math.abs(unweighted-rLoss(metaTau))>1);
 near(6-1/.02,-44);near(.02**2,.0004);
});
test('local R perturbations are second order but a correct marginal outcome nuisance alone does not remove bias',()=>{
 const h=1e-5;near((rLocal(0,h,h)-rLocal(0,-h,-h))/(2*h),0);
 near(rLocal(0,.1,0),6);near(rLocal(0,0,.1),5.69620253164557);near(rLocal(0,.1,.1),5.746835443037974);
 for(const [a,b] of [[0,.1],[.1,0],[.1,.1],[-.1,.2]]){
  const rows=rRows(a,b).filter(o=>o.x===0);
  near(mean(rows.map(o=>o.dt*o.yt))/mean(rows.map(o=>o.dt*o.dt)),rLocal(0,a,b));
 }
});
test('effect restrictions and overlap weighting produce the stated oracle residual risks',()=>{
 near(rLoss(metaTau),1);near(rLoss(()=>4),1.39375);near(rLoss(()=>0),4.89375);
 const approximationError=mean(metaX.map(x=>rInformation(metaE(x))*(4-metaTau(x))**2));
 near(rLoss(()=>4)-rLoss(metaTau),approximationError);
 near(rInformation(.5),.25);near(rInformation(.03),.0291);
});
