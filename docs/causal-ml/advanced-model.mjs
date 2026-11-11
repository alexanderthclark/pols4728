import {mean, m, g, ell, slope} from './model.mjs';

// Full factorial arithmetic worlds. These are not fitted models or random samples.
export const q=x=>1+.5*x+.2*x*x;
export function ivWorld(pi=1,delta=0){
 return [-1,0,1].flatMap(x=>[-1,1].flatMap(v=>[-1,1].flatMap(u=>[-.5,.5].map(e=>{
  const z=q(x)+v,d=m(x)+pi*v+u,y=2*d+g(x)+u+e+delta*v;
  return {x,v,u,e,z,d,y,zt:v,dt:d-m(x),yt:y-ell(x)};
 }))));
}
export function ivEstimate(rows=ivWorld(),t=0){
 const numerator=mean(rows.map(o=>(o.zt-t*o.x)*(o.yt-t*o.x)));
 const denominator=mean(rows.map(o=>(o.zt-t*o.x)*(o.dt-t*o.x)));
 return {numerator,denominator,value:denominator===0?null:numerator/denominator};
}
export const ivOLS=(pi=1)=>slope(ivWorld(pi).map(o=>[o.dt,o.yt]),false);
export const ivMoment=(theta,pi=1,shock=0)=>pi*(2-theta)+shock;
export const compliance=[...Array(3).fill('always'),...Array(5).fill('complier'),...Array(2).fill('never')].map((type,id)=>({id,type,d0:type==='always'?1:0,d1:type==='never'?0:1,tau:type==='always'?2:type==='complier'?4:6}));
export const wald={first:mean(compliance.map(o=>o.d1-o.d0)),reduced:mean(compliance.map(o=>o.tau*(o.d1-o.d0))),ate:mean(compliance.map(o=>o.tau))};

// Fixed realizations of an equal-probability randomized experiment. A unit's
// realized baseline shock is shared by both potential outcomes. Assignment
// labels index a deliberately balanced realization, not an assignment mechanism.
export function treeSample(sample='structure',constant=false){
 return [.15,.35,.65,.85].flatMap(x1=>[.25,.75].flatMap(x2=>[0,1].flatMap(d=>[-1.5,-.5,.5,1.5].map((e,j)=>{
  const tau=constant?4:(x1<.5?6:2);
  const offset=sample==='estimation'?(x1<.5?.4:-.2):sample==='search'?(x1<.5?2:-2):0;
  const y0=20+6*(x2>.5)+e+d*offset,y1=y0+tau;
  return {id:`${sample}-${x1}-${x2}-${d}-${j}`,x1,x2,d,e,y0,y1,tau,y:d?y1:y0};
 }))));
}
export const variance=xs=>xs.length<2?null:xs.reduce((v,x)=>v+(x-mean(xs))**2,0)/(xs.length-1);
export function leafStats(rows){
 const treated=rows.filter(o=>o.d),control=rows.filter(o=>!o.d);
 if(!treated.length||!control.length)return {n:rows.length,n1:treated.length,n0:control.length,tau:null,se:null};
 const y1=mean(treated.map(o=>o.y)),y0=mean(control.map(o=>o.y));
 const s1=variance(treated.map(o=>o.y)),s0=variance(control.map(o=>o.y));
 return {n:rows.length,n1:treated.length,n0:control.length,y1,y0,tau:y1-y0,s1,s0,se:s1===null||s0===null?null:Math.sqrt(s1/treated.length+s0/control.length)};
}
export function partition(rows,kind='root',cut=.5){
 if(kind==='root')return [rows];
 if(kind==='four')return [rows.filter(o=>o.x1<.5&&o.x2<.5),rows.filter(o=>o.x1<.5&&o.x2>=.5),rows.filter(o=>o.x1>=.5&&o.x2<.5),rows.filter(o=>o.x1>=.5&&o.x2>=.5)];
 return [rows.filter(o=>o[kind]<cut),rows.filter(o=>o[kind]>=cut)];
}
export function treeCriterion(rows,kind='root',cut=.5,nEst=rows.length){
 const leaves=partition(rows,kind,cut).map(leafStats),n=rows.length,p=mean(rows.map(o=>o.d));
 if(leaves.some(l=>l.s1==null||l.s0==null))return null;
 const fit=leaves.reduce((v,l)=>v+l.n/n*l.tau*l.tau,0);
 const penalty=(1/n+1/nEst)*leaves.reduce((v,l)=>v+l.s1/p+l.s0/(1-p),0);
 return {fit,penalty,value:fit-penalty,leaves};
}
export function outcomeGain(rows,kind,cut=.5){
 const grand=mean(rows.map(o=>o.y));
 return partition(rows,kind,cut).reduce((v,l)=>v+l.length/rows.length*(mean(l.map(o=>o.y))-grand)**2,0);
}
