export const clamp=(x,a=0,b=1)=>Math.max(a,Math.min(b,x));
export const lerp=(a,b,p)=>a+(b-a)*clamp(p);
export const mean=a=>a.reduce((s,x)=>s+x,0)/a.length;
export const mu=(d,x)=>x?40+2*d:20+6*d;
export const propensity=x=>x?.75:.25;
export const ate=4;
export const people=Array.from({length:16},(_,i)=>{const x=i<8?0:1,d=(i%8)<(x?6:2)?1:0;return {id:i,x,d,y:mu(d,x),y0:mu(0,x),y1:mu(1,x)};});
export function aipw(outcomeCorrect=true,propensityCorrect=true){return people.map(o=>{const m0=outcomeCorrect?mu(0,o.x):28,m1=outcomeCorrect?mu(1,o.x):30,e=propensityCorrect?propensity(o.x):.5;const base=m1-m0,correction=o.d?(o.y-m1)/e:-(o.y-m0)/(1-e);return {...o,m0,m1,e,base,correction,phi:base+correction};});}
export function estimates(outcomeCorrect=true,propensityCorrect=true){const rows=aipw(outcomeCorrect,propensityCorrect);return {raw:mean(people.filter(o=>o.d).map(o=>o.y))-mean(people.filter(o=>!o.d).map(o=>o.y)),outcome:mean(rows.map(o=>o.base)),ipw:mean(rows.map(o=>o.d*o.y/o.e-(1-o.d)*o.y/(1-o.e))),aipw:mean(rows.map(o=>o.phi))};}
export const theta=2;
export const m=x=>3+.7*x+.25*x*x;
export const g=x=>20+.6*x*x+.2*x*x*x;
export const ell=x=>theta*m(x)+g(x);
export const plr=Array.from({length:50},(_,i)=>{const x=Math.floor(i/10)-2,v=[-1,-.5,0,.5,1][Math.floor((i%10)/2)],u=i%2?.6:-.6,d=m(x)+v;return{id:i,x,v,u,d,y:theta*d+g(x)+u};});
export function slope(rows,intercept=true){const mx=intercept?mean(rows.map(r=>r[0])):0,my=intercept?mean(rows.map(r=>r[1])):0;return rows.reduce((s,[x,y])=>s+(x-mx)*(y-my),0)/rows.reduce((s,[x])=>s+(x-mx)**2,0);}
export const residualSlope=(a=0,b=0)=>slope(plr.map(o=>[o.d-m(o.x)-b*o.x,o.y-ell(o.x)-a*o.x]),false);
export function partialResiduals(yProgress,dProgress){return plr.map(o=>[o.d-dProgress*m(o.x),o.y-yProgress*ell(o.x)]);}
export const position=(value,n)=>{const s=clamp(value,0,1)*n;return {index:Math.min(n-1,Math.floor(s)),progress:clamp((s-Math.floor(s)-.08)/.65)};};
