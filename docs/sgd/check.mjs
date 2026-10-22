import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {ramp,initialize,gradient,loss,mse,copy,data,train,update,target,predict,relu,random,shuffle} from './model.mjs';
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
assert.equal(recorded.config.objective,'Half mean squared error');
assert.equal(recorded.config.metric,'Mean squared error');
assert.equal(recorded.config.learningRate,'0.05 / (1 + step / 6000)');
const firstBatch=recorded.firstUpdate.indices.map(i=>data[i]);
assert.equal(recorded.firstUpdate.batchBefore,loss(recorded.firstUpdate.before,firstBatch));
assert.equal(recorded.firstUpdate.batchAfter,loss(recorded.firstUpdate.after,firstBatch));
assert.deepEqual(recorded.firstUpdate.gradient,gradient(recorded.firstUpdate.before,firstBatch));
for(const width of [2,12]){
 const trace=recorded.traces[width],firstEpoch=trace.slice(0,32);
 assert.equal(new Set(firstEpoch.flatMap(s=>s.indices)).size,2048);
 for(const s of trace){
  assert.equal(s.indices.length,64);assert.equal(new Set(s.indices).size,64);
  const next=copy(s.model);update(next,gradient(next,s.indices.map(i=>data[i])),s.eta);
  assert.deepEqual(next,s.after);assert.equal(mse(s.model),s.loss);assert.equal(mse(next),s.afterLoss);
  assert.equal(loss(next),s.afterLoss/2);
 }
 const final=recorded.runs.find(r=>r.width===width&&r.seed===1);
 assert.deepEqual(trace.at(-1).after,final.model);
 assert.equal(trace.at(-1).step+1,9600);
}
assert.deepEqual(recorded.traces[2].map(s=>s.indices),recorded.traces[12].map(s=>s.indices));
for(const r of recorded.runs){const fresh=train({width:r.width,seed:r.seed});assert.equal(fresh.loss,r.loss);assert.equal(fresh.gridMSE,r.gridMSE);assert.ok(Number.isFinite(r.loss));}
// Independently reproduce the previous ordinary-MSE gradient and learning-rate
// schedule. Halving the gradient and doubling eta must preserve every update,
// not only the saved snapshots or the final fitted curve.
function previousMSEGradient(model,rows){
 const g={c:0,units:model.units.map(()=>({w:0,b:0,a:0}))};
 for(const {x,y} of rows){
  const derivative=2*(predict(model,x)-y)/rows.length;
  g.c+=derivative;
  model.units.forEach((u,j)=>{
   const z=u.w*x+u.b;
   g.units[j].a+=derivative*relu(z);
   if(z>0){g.units[j].w+=derivative*u.a*x;g.units[j].b+=derivative*u.a;}
  });
 }
 return g;
}
for(const {width,seed,model:expectedFinal} of recorded.runs){
 const previous=initialize(width,seed),current=copy(previous),rng=random(seed+10000);
 for(let epoch=0;epoch<recorded.config.epochs;epoch++){
  const order=shuffle(rng);
  for(let batch=0;batch<recorded.config.batchesPerEpoch;batch++){
   const step=epoch*recorded.config.batchesPerEpoch+batch;
   const rows=order.slice(batch*64,(batch+1)*64).map(i=>data[i]);
   update(previous,previousMSEGradient(previous,rows),.025/(1+step/6000));
   update(current,gradient(current,rows),.05/(1+step/6000));
   assert.deepEqual(current,previous,`Preserved trajectory: width ${width}, seed ${seed}, step ${step}`);
  }
 }
 assert.deepEqual(current,expectedFinal);
}
console.log('Verified: half-MSE gradients at both widths; ordinary-MSE metrics; noiseless data; first-epoch coverage; matched batches; every recorded update; all 16 final runs; exact legacy trajectory at every update; integrated ramp error.');
