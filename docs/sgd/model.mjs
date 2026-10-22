// One scalar input, one hidden ReLU layer, one linear output. No framework.
export const relu = x => Math.max(0, x);
export const target = x => x > 0 ? 1 : 0;
export const ramp = (x, a = 1, t = 0) => relu(a * (x - t)) - relu(a * (x - t) - 1);
export function random(seed) {
  let s = seed >>> 0;
  return () => { s += 0x6D2B79F5; let t = s; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
export const data = Array.from({length:2048}, (_,i) => {
  const x = -1 + 2 * (i + .5) / 2048;
  return {x,y:target(x)};
});
export function initialize(width, seed) {
  const rng = random(seed);
  return {c:.5, units:Array.from({length:width}, () => ({w:2*rng()-1,b:rng()-.5,a:(2*rng()-1)/Math.sqrt(width)}))};
}
export function predict(model,x) { return model.c + model.units.reduce((s,u)=>s+u.a*relu(u.w*x+u.b),0); }
export function loss(model,rows=data) { return rows.reduce((s,d)=>s+(predict(model,d.x)-d.y)**2,0)/rows.length; }
export function gradient(model, rows) {
  const g = {c:0,units:model.units.map(()=>({w:0,b:0,a:0}))};
  for (const {x,y} of rows) {
    const e = 2 * (predict(model,x)-y) / rows.length;
    g.c += e;
    model.units.forEach((u,j)=>{const z=u.w*x+u.b; g.units[j].a += e*relu(z); if(z>0){g.units[j].w+=e*u.a*x;g.units[j].b+=e*u.a;}});
  }
  return g;
}
export function update(model, g, eta) {
  model.c -= eta*g.c;
  model.units.forEach((u,j)=>{for(const key of ['w','b','a'])u[key]-=eta*g.units[j][key];});
}
export const copy = model => ({c:model.c,units:model.units.map(u=>({...u}))});
export function shuffle(rng) {
  const indices=data.map((_,i)=>i);
  for(let i=indices.length-1;i>0;i--){const j=Math.floor(rng()*(i+1));[indices[i],indices[j]]=[indices[j],indices[i]];}
  return indices;
}
export function train({width=2,seed=1,epochs=300,batchSize=64,record=false}={}) {
  const model=initialize(width,seed),rng=random(seed+10000),trace=[],batches=data.length/batchSize,steps=epochs*batches;
  if(!Number.isInteger(batches))throw new Error('Batch size must divide the dataset.');
  for(let epoch=1;epoch<=epochs;epoch++) {
    const order=shuffle(rng);
    for(let b=0;b<batches;b++){
      const step=(epoch-1)*batches+b,eta=.025/(1+step/6000),indices=order.slice(b*batchSize,(b+1)*batchSize);
      const capture=record&&(epoch<=2||step%31===0||step===steps-1);
      const snapshot=capture?{step,epoch,batch:b+1,eta,indices,model:copy(model),loss:loss(model)}:null;
      update(model,gradient(model,indices.map(i=>data[i])),eta);
      if(snapshot){snapshot.after=copy(model);snapshot.afterLoss=loss(model);trace.push(snapshot);}
    }
  }
  const grid=Array.from({length:4000},(_,i)=>({x:-1+(i+.5)/2000,y:target(-1+(i+.5)/2000)}));
  return {width,seed,epochs,steps,batchSize,model:copy(model),loss:loss(model),gridMSE:loss(model,grid),trace};
}
