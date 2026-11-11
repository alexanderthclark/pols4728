import {mean} from './model.mjs';

export const forestX=[.1,.3,.5,.7,.9],forestZ=[.2,.5,.8];
export const forestTau=x=>6-4*x;
export const forestE=x=>.1875+.625*x;
export const forestMu=(d,x,z)=>20+18*z+4*x*x+d*forestTau(x);
export const forestEll=(x,z)=>forestMu(0,x,z)+forestE(x)*forestTau(x);
export const forestCells=forestX.flatMap((x,j)=>forestZ.map((z,k)=>({id:3*j+k,x,z,j,k})));
// A balanced 240-point arithmetic construction, not a random research sample.
export const forestRows=forestCells.flatMap(c=>Array.from({length:16},(_,r)=>{
 const d=Number(r<4+2*c.j),noise=r%2?1:-1,y=forestMu(d,c.x,c.z)+noise;
 return {...c,cell:c.id,id:16*c.id+r,d,noise,y,dt:d-forestE(c.x),yt:y-forestEll(c.x,c.z)};
}));
export const forestQuery={x:.3,z:.5};

// Three specified partitions isolate neighborhood arithmetic. They are not
// presented as the result of a fitted forest or used to claim coverage.
export const forestTrees=[
 {leaf:q=>q.x<=.5?0:1,regions:[[0,.5,0,1],[.5,1,0,1]],cuts:[[.5,0,.5,1]]},
 {leaf:q=>q.z>.65?2:q.x<=.4?0:1,regions:[[0,.4,0,.65],[.4,1,0,.65],[0,1,.65,1]],cuts:[[0,.65,1,.65],[.4,0,.4,.65]]},
 {leaf:q=>q.x<=.2?0:q.z<=.35?1:2,regions:[[0,.2,0,1],[.2,1,0,.35],[.2,1,.35,1]],cuts:[[.2,0,.2,1],[.2,.35,1,.35]]}
];
export function treeWeights(q=forestQuery,b=0,rows=forestRows){
 const tree=forestTrees[b],leaf=tree.leaf(q),n=rows.filter(o=>tree.leaf(o)===leaf).length;
 if(!n)throw new Error('Empty illustrative prediction leaf');
 return rows.map(o=>tree.leaf(o)===leaf?1/n:0);
}
export function forestWeights(q=forestQuery,trees=[0,1,2],rows=forestRows){
 const all=trees.map(b=>treeWeights(q,b,rows));
 return rows.map((_,i)=>mean(all.map(a=>a[i])));
}
export function weightedFit(rows,weights,{d=o=>o.dt,y=o=>o.yt}={}){
 const total=weights.reduce((a,b)=>a+b,0),w=weights.map(a=>a/total);
 const md=rows.reduce((s,o,i)=>s+w[i]*d(o),0),my=rows.reduce((s,o,i)=>s+w[i]*y(o),0);
 const variance=rows.reduce((s,o,i)=>s+w[i]*(d(o)-md)**2,0);
 const covariance=rows.reduce((s,o,i)=>s+w[i]*(d(o)-md)*(y(o)-my),0);
 const slope=covariance/variance;
 return {md,my,variance,covariance,slope,intercept:my-slope*md};
}
export const forestFit=(q=forestQuery,trees=[0,1,2])=>weightedFit(forestRows,forestWeights(q,trees));
export const forestESS=w=>1/w.reduce((s,a)=>s+a*a,0);
export const cellWeights=w=>forestCells.map(c=>forestRows.reduce((s,o,i)=>s+(o.cell===c.id?w[i]:0),0));
export function centeredExample(){
 const w=forestWeights(),rows=forestRows.map(o=>({...o,dt:o.dt-.15,yt:o.yt-.8}));
 const fit=weightedFit(rows,w),origin=rows.reduce((s,o,i)=>s+w[i]*o.dt*o.yt,0)/rows.reduce((s,o,i)=>s+w[i]*o.dt**2,0);
 return {rows,w,fit,origin};
}
export function splitProxy(axis,cut){
 const n=forestRows.length,parent=weightedFit(forestRows,forestRows.map(()=>1/n));
 const relabeled=forestRows.map(o=>({...o,rho:(o.dt-parent.md)*(o.yt-parent.my-parent.slope*(o.dt-parent.md))/parent.variance}));
 const left=relabeled.filter(o=>o[axis]<=cut),right=relabeled.filter(o=>o[axis]>cut);
 return {left:mean(left.map(o=>o.rho)),right:mean(right.map(o=>o.rho)),score:left.length*right.length/n**2*(mean(left.map(o=>o.rho))-mean(right.map(o=>o.rho)))**2,parent};
}
export function windowFit(radius,x=.3){
 const included=forestRows.map(o=>Math.abs(o.x-x)<=radius+1e-9?1:0),n=included.reduce((s,v)=>s+v,0),weights=included.map(v=>v/n);
 return {...weightedFit(forestRows,weights),weights,n,ess:forestESS(weights)};
}
// Conditional Monte Carlo SD for one individual's weight, if these three
// specified partitions are sampled independently with equal probability.
export function weightMonteCarlo(b){
 const i=forestRows.findIndex(o=>o.x===.3&&o.z===.5),values=[0,1,2].map(k=>treeWeights(forestQuery,k)[i]);
 const avg=mean(values),variance=mean(values.map(v=>(v-avg)**2));
 return {mean:avg,sd:Math.sqrt(variance/b),values};
}
export const forestAverage=()=>mean(forestCells.map(c=>forestFit(c).slope));
export function forestAipw(){
 return forestRows.map(o=>{
  const tau=forestFit(o).slope,mu0=forestMu(0,o.x,o.z),mu1=mu0+tau,e=forestE(o.x);
  return {...o,tau,phi:tau+o.d*(o.y-mu1)/e-(1-o.d)*(o.y-mu0)/(1-e)};
 });
}
