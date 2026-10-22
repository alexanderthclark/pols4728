import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {ramp,initialize,gradient,loss,copy,data,train,update,target} from './model.mjs';
// Independently check every parameter's backprop derivative at both widths.
for(const width of [2,12]){
 const m=initialize(width,1),g=gradient(m,data),eps=1e-6;
 for(const [j,k] of [[-1,'c'],...m.units.flatMap((_,i)=>['w','b','a'].map(k=>[i,k]))]){
  const plus=copy(m),minus=copy(m);if(j<0){plus.c+=eps;minus.c-=eps;}else{plus.units[j][k]+=eps;minus.units[j][k]-=eps;}
  const numerical=(loss(plus)-loss(minus))/(2*eps),analytic=j<0?g.c:g.units[j][k];assert.ok(Math.abs(numerical-analytic)<1e-7,`${width}: ${j},${k}: ${numerical} vs ${analytic}`);
 }
}
assert.equal(data.length,2048);assert.ok(data.every(d=>d.y===target(d.x)));
for(const a of [1,4,32,64]){let integral=0,n=10000;for(let i=0;i<n;i++)integral+=(1-ramp((i+.5)/(n*a),a))/(n*a);assert.ok(Math.abs(integral-1/(2*a))<1e-10);}
const recorded=JSON.parse(readFileSync(new URL('./experiment.json',import.meta.url)));
for(const width of [2,12]){
 const trace=recorded.traces[width],firstEpoch=trace.slice(0,32);
 assert.equal(new Set(firstEpoch.flatMap(s=>s.indices)).size,2048);
 for(const s of trace){
  assert.equal(s.indices.length,64);assert.equal(new Set(s.indices).size,64);
  const next=copy(s.model);update(next,gradient(next,s.indices.map(i=>data[i])),s.eta);
  assert.deepEqual(next,s.after);assert.equal(loss(next),s.afterLoss);
 }
 const final=recorded.runs.find(r=>r.width===width&&r.seed===1);
 assert.deepEqual(trace.at(-1).after,final.model);
 assert.equal(trace.at(-1).step+1,9600);
}
assert.deepEqual(recorded.traces[2].map(s=>s.indices),recorded.traces[12].map(s=>s.indices));
for(const r of recorded.runs){const fresh=train({width:r.width,seed:r.seed});assert.equal(fresh.loss,r.loss);assert.equal(fresh.gridMSE,r.gridMSE);assert.ok(Number.isFinite(r.loss));}
console.log('Verified: gradients at both widths; noiseless data; first-epoch coverage; matched batches; every recorded update; all 16 final runs; integrated ramp error.');
