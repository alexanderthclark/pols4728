import {mean} from './model.mjs';

export const metaTau=x=>6-4*x;
export const metaMu=(d,x)=>20+8*x+4*x*x+d*metaTau(x);
export const metaE=x=>.25+.5*x;
export const metaEll=x=>metaMu(0,x)+metaE(x)*metaTau(x);
export const metaX=[0,.25,.5,.75,1];
// The 80 observations exhaust a constructed distribution: conditional treatment
// counts match e(x), and each arm has balanced +/-1 outcome noise at every x.
export const metaRows=metaX.flatMap((x,j)=>Array.from({length:16},(_,k)=>{
 const d=Number(k<4+2*j),noise=k%2?1:-1,y0=metaMu(0,x)+noise;
 return {id:j*16+k,x,d,noise,y0,y1:y0+metaTau(x),y:metaMu(d,x)+noise};
}));
export const byX=rows=>metaX.map(x=>({x,rows:rows.filter(o=>o.x===x)}));

// Small, deterministic least-squares learners make the toy fits actual fits.
// Penalties correspond to mean squared error plus sum_j lambda_j beta_j^2.
export function leastSquares(rows,features,target,penalties=[]){
 const k=features(rows[0]).length;
 const a=Array.from({length:k},()=>Array(k+1).fill(0));
 rows.forEach(o=>{const f=features(o),y=target(o);for(let i=0;i<k;i++){for(let j=0;j<k;j++)a[i][j]+=f[i]*f[j]/rows.length;a[i][k]+=f[i]*y/rows.length;}});
 for(let i=0;i<k;i++)a[i][i]+=penalties[i]||0;
 for(let i=0;i<k;i++){
  let pivot=i;for(let j=i+1;j<k;j++)if(Math.abs(a[j][i])>Math.abs(a[pivot][i]))pivot=j;
  if(Math.abs(a[pivot][i])<1e-12)throw new Error('Singular teaching-example fit');
  [a[i],a[pivot]]=[a[pivot],a[i]];
  const scale=a[i][i];for(let j=i;j<=k;j++)a[i][j]/=scale;
  for(let r=0;r<k;r++)if(r!==i){const f=a[r][i];for(let j=i;j<=k;j++)a[r][j]-=f*a[i][j];}
 }
 return a.map(row=>row[k]);
}
const dot=(a,b)=>a.reduce((v,x,i)=>v+x*b[i],0);
export function fitS(lambda=0,interaction=true){
 const features=o=>interaction?[1,o.x,o.x*o.x,o.d,o.d*o.x]:[1,o.x,o.x*o.x,o.d];
 const beta=leastSquares(metaRows,features,o=>o.y,[0,0,0,lambda,lambda]);
 const predict=(x,d)=>dot(beta,features({x,d}));
 return {beta,predict,effect:x=>predict(x,1)-predict(x,0),mse:mean(metaRows.map(o=>(o.y-predict(o.x,o.d))**2))};
}
export function fitT(){
 const beta=[0,1].map(d=>leastSquares(metaRows.filter(o=>o.d===d),o=>[1,o.x,o.x*o.x],o=>o.y));
 const predict=(x,d)=>dot(beta[d],[1,x,x*x]);
 return {beta,predict,effect:x=>predict(x,1)-predict(x,0)};
}
export const tFit=fitT();
export function xPseudo(){
 return metaRows.map(o=>({...o,pseudo:o.d?o.y-tFit.predict(o.x,0):tFit.predict(o.x,1)-o.y}));
}
export const xFits=[0,1].map(d=>leastSquares(xPseudo().filter(o=>o.d===d),o=>[1,o.x],o=>o.pseudo));
export const xArm=(x,d)=>dot(xFits[d],[1,x]);
export const xBlend=(tau0,tau1,g)=>g*tau0+(1-g)*tau1;
export const tError=(d,x)=>(d?-1:1)*.6*Math.sin(2*Math.PI*x);
export function drRows(outcomeCorrect=true,propensityCorrect=true){
 return metaRows.map(o=>{
  const mu0=metaMu(0,o.x)+(outcomeCorrect?0:2),mu1=metaMu(1,o.x)-(outcomeCorrect?0:1),e=propensityCorrect?metaE(o.x):.5;
  const base=mu1-mu0,correction=o.d?(o.y-mu1)/e:-(o.y-mu0)/(1-e);
  return {...o,mu0,mu1,e,base,correction,phi:base+correction};
 });
}
export function drFit(outcomeCorrect=true,propensityCorrect=true){
 const rows=drRows(outcomeCorrect,propensityCorrect),beta=leastSquares(rows,o=>[1,o.x],o=>o.phi);
 return {beta,predict:x=>dot(beta,[1,x]),average:mean(rows.map(o=>o.phi))};
}
export function rRows(a=0,b=0){return metaRows.map(o=>({...o,dt:o.d-metaE(o.x)-b,yt:o.y-metaEll(o.x)-a}));}
export function rFit(){
 const beta=leastSquares(rRows(),o=>[o.dt,o.dt*o.x],o=>o.yt);
 return {beta,predict:x=>dot(beta,[1,x])};
}
export const rLoss=(candidate,a=0,b=0)=>mean(rRows(a,b).map(o=>(o.yt-o.dt*candidate(o.x))**2));
export function rLocal(x,a=0,b=0){const v=metaE(x)*(1-metaE(x));return (v*metaTau(x)+a*b)/(v+b*b);}
export const rInformation=e=>e*(1-e);
