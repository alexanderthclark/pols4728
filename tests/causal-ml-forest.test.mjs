import test from 'node:test';
import assert from 'node:assert/strict';
import {forestRows,forestCells,forestTau,forestE,forestEll,forestQuery,forestTrees,treeWeights,forestWeights,cellWeights,forestESS,weightedFit,forestFit,centeredExample,splitProxy,windowFit,weightMonteCarlo,forestAverage,forestAipw} from '../docs/causal-ml/forest-model.mjs';
const avg=a=>a.reduce((s,v)=>s+v,0)/a.length;
const near=(a,b,tol=1e-10)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);

test('forest construction satisfies its cell propensities, outcome means, and residual identity',()=>{
 assert.equal(forestRows.length,240);
 assert.ok(forestCells.some(c=>c.x===forestQuery.x&&c.z===forestQuery.z));
 near(forestTau(forestQuery.x),4.8);
 for(const c of forestCells){
  const rows=forestRows.filter(o=>o.cell===c.id);
  near(avg(rows.map(o=>o.d)),forestE(c.x));
  near(avg(rows.map(o=>o.y)),forestEll(c.x,c.z));
  near(avg(rows.map(o=>o.dt)),0);near(avg(rows.map(o=>o.yt)),0);
  for(const o of rows)near(o.yt,o.dt*forestTau(o.x)+o.noise);
 }
 near(avg(forestRows.map(o=>forestTau(o.x))),4);
});
test('gradient labels target effect differences rather than prognostic differences',()=>{
 const engagement=splitProxy('x',.4),knowledge=splitProxy('z',.35);
 near(engagement.parent.slope,4);
 near(engagement.left,39/35);near(engagement.right,-26/35);
 near(engagement.score,.8277551020408163);near(knowledge.score,0);
});
test('each prediction leaf is normalized before the forest averages weights',()=>{
 const n=[144,64,128];
 for(let b=0;b<3;b++){
  const w=treeWeights(forestQuery,b);near(w.reduce((s,v)=>s+v,0),1);
  assert.equal(w.filter(a=>a>0).length,n[b]);
  w.filter(a=>a>0).forEach(a=>near(a,1/n[b]));
 }
 const w=forestWeights(),i=forestRows.findIndex(o=>o.x===.3&&o.z===.5);
 near(w[i],(1/144+1/64+1/128)/3);
 near(w.reduce((s,a)=>s+a,0),1);assert.equal(cellWeights(w).filter(a=>a>0).length,13);
 near(forestESS(w),163.2755905511811);
 // Pooling unnormalized co-occurrences would incorrectly give each of this
 // person's memberships the same mass, regardless of leaf size.
 assert.ok(Math.abs(w[i]-3/(144+64+128))>.001);
});
test('all specified leaves are nonempty and weights remain valid across query boundaries',()=>{
 for(const x of [.1,.2,.200001,.4,.400001,.5,.500001,.9])for(const z of [.2,.35,.350001,.65,.650001,.8]){
  const w=forestWeights({x,z});near(w.reduce((s,a)=>s+a,0),1);assert.ok(w.every(a=>a>=0));
  assert.ok(Number.isFinite(forestFit({x,z}).slope));
  for(const tree of forestTrees)assert.ok(tree.regions[tree.leaf({x,z})]);
 }
});
test('the local intercept absorbs constant shifts but a through-origin fit does not',()=>{
 const original=forestFit(),shift=centeredExample();
 near(shift.fit.md,-.15);near(shift.fit.my,-.8);
 near(shift.fit.slope,original.slope);near(shift.fit.intercept,-.8+.15*original.slope);
 near(shift.origin,4.57787939411261);
 assert.ok(Math.abs(shift.origin-original.slope)>.07);
});
test('pooled regression solves the local moments and differs from averaging tree slopes',()=>{
 const f=forestFit(),w=forestWeights(),trees=[0,1,2].map(b=>forestFit(forestQuery,[b]));
 near(f.covariance,.9923611111111111);near(f.variance,.2204861111111111);
 near(f.slope,4.500787401574803);
 near(avg(trees.map(t=>t.slope)),4.516700822715261);
 near(forestRows.reduce((s,o,i)=>s+w[i]*(o.yt-f.intercept-f.slope*o.dt),0),0);
 near(forestRows.reduce((s,o,i)=>s+w[i]*o.dt*(o.yt-f.intercept-f.slope*o.dt),0),0);
 // General data need pooled first moments too, not just mean within-leaf covariances.
 const data=[{dt:0,yt:0},{dt:1,yt:1},{dt:2,yt:6},{dt:3,yt:7}];
 const pooled=weightedFit(data,[.25,.25,.25,.25]);
 near(pooled.slope,2.6);near((weightedFit(data,[.5,.5,0,0]).slope+weightedFit(data,[0,0,.5,.5]).slope)/2,1);
});
test('finite neighborhoods target treatment-variance-weighted effect averages',()=>{
 for(const r of [.6,.4,.2,.05]){
  const f=windowFit(r),num=forestRows.reduce((s,o,i)=>s+f.weights[i]*forestE(o.x)*(1-forestE(o.x))*forestTau(o.x),0);
  const den=forestRows.reduce((s,o,i)=>s+f.weights[i]*forestE(o.x)*(1-forestE(o.x)),0);
  near(f.slope,num/den);
 }
 near(windowFit(.6).slope,4);near(windowFit(.05).slope,4.8);
 assert.equal(windowFit(.05).n,48);
});
test('conditional tree-weight Monte Carlo variation scales with the number of independent draws',()=>{
 const one=weightMonteCarlo(1),many=weightMonteCarlo(64);
 near(one.mean,.010127314814814815);near(one.sd,.0039035698967685305);
 near(many.sd,one.sd/8);near(many.mean,one.mean);
});
test('duplicating trees preserves all forest weights, moments, and predictions',()=>{
 const repeated=Array.from({length:30},(_,j)=>j%3),once=forestWeights(),copies=forestWeights(forestQuery,repeated);
 copies.forEach((a,i)=>near(a,once[i]));
 near(forestFit(forestQuery,repeated).slope,forestFit().slope);
});
test('correct propensity repairs the imperfect forest contrast conditionally and in the ATE',()=>{
 const scores=forestAipw();
 near(forestAverage(),4.010822343502857);
 for(const c of forestCells)near(avg(scores.filter(o=>o.cell===c.id).map(o=>o.phi)),forestTau(c.x));
 near(avg(scores.map(o=>o.phi)),4);
});
