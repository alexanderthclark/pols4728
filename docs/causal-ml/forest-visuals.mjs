import {lerp,mean} from './model.mjs';
import {forestX,forestCells,forestRows,forestQuery,forestTrees,forestTau,forestE,forestMu,forestEll,treeWeights,forestWeights,cellWeights,forestFit,centeredExample,splitProxy,windowFit,weightMonteCarlo,forestAverage,forestAipw} from './forest-model.mjs';
const R=String.raw;
const C={ink:'#242424',blue:'#234E70',rust:'#A54F32',teal:'#007F73',muted:'#606060',line:'#d4d8da',pale:'#f3f6f7',white:'#fff'};
const fmt=(n,k=3)=>Number(n).toFixed(k).replace(/\.?0+$/,'');
const path=(d,points,col=C.blue,width=2.5,dash='')=>d.path(points.map(([x,y],j)=>`${j?'L':'M'}${x} ${y}`).join(' '),col,width,dash);
const curve=(d,a,f,col=C.blue,p=1,dash='')=>path(d,Array.from({length:81},(_,i)=>{const x=i/80*p;return [a.x(x),a.y(f(x))];}),col,2.5,dash);
const axes=(d,options)=>d.axes({bottom:d.mobile?70:65,...options});
const baseWeights=forestWeights(),baseFit=forestFit(),firstTreeWeights=treeWeights(),offset=centeredExample();
const treeFits=[0,1,2].map(b=>forestFit(forestQuery,[b]));
const aipw=forestAipw(),average=forestAverage();
const cellEffects=forestCells.map(c=>({cell:c,estimate:forestFit(c).slope,phi:mean(aipw.filter(o=>o.cell===c.id).map(o=>o.phi))}));
function plane(d,{left=d.mobile?53:65,right=d.W-25,top=55,bottom=d.H-55,labels=true,ticks=true}={}){
 const x=v=>lerp(left,right,v),y=v=>lerp(bottom,top,v);
 d.line(left,top,left,bottom,C.ink,1);d.line(left,bottom,right,bottom,C.ink,1);
 if(ticks){[0,.5,1].forEach(v=>{d.text(v,x(v),bottom+18,d.small,C.muted);d.text(v,left-8,y(v),d.small,C.muted,'end');});}
 if(labels){d.math('X_1',(left+right)/2,bottom+40,80,30,d.fs);d.math('X_2',left,top-24,80,30,d.fs);}
 return {x,y,left,right,top,bottom};
}
function cellDots(d,a,weights=null,p=1){
 forestCells.forEach((c,j)=>{
  const w=weights?weights[j]:1/15,r=weights?Math.sqrt(w)*(d.mobile?52:76):d.mobile?4:6;
  if(w>0)d.circle(a.x(c.x),a.y(c.z),lerp(d.mobile?4:6,r,p),weights?C.blue:C.white,C.blue,weights?.55:1);
  else d.circle(a.x(c.x),a.y(c.z),d.mobile?3:4,C.white,C.line);
 });
}
function queryMark(d,a,q=forestQuery){
 const x=a.x(q.x),y=a.y(q.z);d.line(x-7,y,x+7,y,C.rust,2.5);d.line(x,y-7,x,y+7,C.rust,2.5);
}
function partition(d,a,b,q=forestQuery,p=1){
 const tree=forestTrees[b],r=tree.regions[tree.leaf(q)];
 d.rect(a.x(r[0]),a.y(r[3]),a.x(r[1])-a.x(r[0]),a.y(r[2])-a.y(r[3]),C.blue,'none',0,.09*p);
 tree.cuts.forEach(([x0,y0,x1,y1])=>d.line(a.x(x0),a.y(y0),a.x(lerp(x0,x1,p)),a.y(lerp(y0,y1,p)),C.muted,1.5,'4 3'));
}
function target(d,p){
 const a=axes(d,{xmin:0,xmax:1,ymin:28,ymax:39,xlabel:'X_1',ylabel:'Y',xticks:[0,.5,1],yticks:[28,33,38]});
 curve(d,a,x=>forestMu(0,x,.5),C.rust,1,'5 4');curve(d,a,x=>forestMu(1,x,.5),C.blue);
 const x=lerp(.1,.3,p),y0=forestMu(0,x,.5),y1=forestMu(1,x,.5);
 d.line(a.x(x),a.y(y0),a.x(x),a.y(y1),C.teal,3);d.circle(a.x(x),a.y(y0),5,C.white,C.rust);d.circle(a.x(x),a.y(y1),5,C.blue,C.blue);
 d.math(R`\tau(x)=`+fmt(forestTau(x),2),a.x(x)+95,a.y((y0+y1)/2),180,35,d.fs,C.teal);
 d.math(R`\mu_0`,a.x(.83),a.y(forestMu(0,.83,.5))+22,85,30,d.small,C.rust);d.math(R`\mu_1`,a.x(.83),a.y(forestMu(1,.83,.5))-25,85,30,d.small,C.blue);
}
function identify(d,p){
 const {W,H,mobile:z}=d,left=z?85:135,gap=(W-left-24)/16;
 forestX.forEach((x,j)=>{
  const y=45+j*(H-90)/4;d.math('X_1='+x,left*.43,y,left-10,30,d.small);
  forestRows.filter(o=>o.x===x&&o.z===.5).forEach((o,k)=>d.mark(o,left+(k+.5)*gap,lerp(H*.5,y,p),z?3.6:5));
 });
 d.text('Treated counts: 4, 6, 8, 10, 12 of 16',W/2,H-13,d.small,C.muted);
}
function residuals(d,p){
 const a=axes(d,{xmin:lerp(-.1,-.9,p),xmax:lerp(1.1,.9,p),ymin:lerp(24,-5,p),ymax:lerp(40,5.5,p),xlabel:p===0?'D':R`\widetilde D`,ylabel:p===0?'Y':R`\widetilde Y`,xticks:[lerp(0,-.75,p),lerp(.5,0,p),lerp(1,.75,p)],yticks:[lerp(25,-4,p),lerp(32,0,p),lerp(39,4,p)]});
 forestRows.filter(o=>o.z===.5).forEach(o=>d.mark(o,a.x(o.d-p*forestE(o.x)),a.y(o.y-p*forestEll(o.x,o.z)),d.mobile?3:4,.6));
 d.text('Subtract the covariate predictions',d.W*.57,18,d.small,C.muted);
}
function split(d,p){
 const {W,H,mobile:z}=d,left=z?65:110,right=W-85,score=splitProxy('x',.4).score;
 [0,1].forEach(j=>{
  const y=H*(j?.65:.27),v=j?0:score;
  d.math(j?R`X_2\le0.35`:R`X_1\le0.4`,W/2,y-35,240,35,d.fs,j?C.muted:C.blue);
  d.line(left,y,right,y,C.line,z?14:20);d.line(left,y,lerp(left,right,v*p),y,C.blue,z?14:20);
  d.text(fmt(v),right+12,y,d.fs,j?C.muted:C.blue,'start');
 });
 d.math(R`\Delta`,W/2,H-21,110,30,d.fs);d.text('Normalized gradient split score',W/2,20,d.small,C.muted);
}
function honesty(d,p){
 const {W,H,mobile:z}=d,centers=[W*.25,W*.75];
 d.text('One tree’s subsample',W/2,20,d.fs);
 for(let i=0;i<16;i++){
  const role=i<8?0:1,j=i%8,startX=W/2+(i-7.5)*(z?18:28),x=centers[role]+(j%4-1.5)*(z?19:30),y=H*.4+Math.floor(j/4)*(z?22:30);
  d.circle(lerp(startX,x,p),lerp(65,y,p),z?5:7,role?C.white:C.blue,role?C.rust:C.blue);
 }
 [R`S_b`,R`I_b`].forEach((t,j)=>d.math(t,centers[j],H*.28,90,30,d.fs,j?C.rust:C.blue));
 d.text('Choose splits',centers[0],H*.66,d.fs,C.blue);d.text('Estimate leaves',centers[1],H*.66,d.fs,C.rust);
 [0,1].forEach(j=>{const x=centers[j],y=H*.79;d.line(x,y,x-30,y+31,C.ink,1.5);d.line(x,y,x+30,y+31,C.ink,1.5);d.circle(x,y,4,C.white,C.ink);[-30,30].forEach(dx=>d.rect(x+dx-10,y+27,20,18,j?C.pale:C.white,j?C.rust:C.blue));});
}
function partitions(d,p){
 const {W,H,mobile:z}=d,gap=z?10:25,width=(W-2*gap)/3;
 for(let b=0;b<3;b++){
  const start=b*(width+gap),a=plane(d,{left:start+20,right:start+width-5,top:66,bottom:H-61,labels:false,ticks:false});
  partition(d,a,b,forestQuery,p);cellDots(d,a);queryMark(d,a);
  d.text('Tree '+(b+1),start+width/2,23,d.fs);
  d.math('n_'+(b+1)+'='+[144,64,128][b],start+width/2,H-26,width,34,d.fs);
  d.math('X_1',start+width-12,H-46,45,23,d.small);d.math('X_2',start+20,48,45,25,d.small);
 }
}
function leafWeights(d,p){
 const {W,H,mobile:z}=d,left=z?83:140,right=W-55,max=1/64;
 d.math(R`X_i=(0.3,0.5)`,W/2,23,300,35,d.fs);
 [144,64,128].forEach((n,b)=>{
  const y=H*.30+b*H*.24;d.text('Tree '+(b+1),left-15,y,d.small,C.ink,'end');
  d.line(left,y,right,y,C.line,z?15:22);d.line(left,y,lerp(left,right,p/(n*max)),y,C.blue,z?15:22);
  d.math('1/'+n,(left+right)/2,y-24,150,30,d.fs,C.blue);
 });
 d.text('Individual weight in the selected leaf',W/2,H-13,d.small,C.muted);
}
function weights(d,p,moving=false){
 const q=moving?{x:lerp(.1,.9,p),z:.5}:forestQuery,w=moving?forestWeights(q):baseWeights.map((v,i)=>lerp(firstTreeWeights[i],v,p)),cw=cellWeights(w),a=plane(d);
 if(moving)for(let b=0;b<3;b++)partition(d,a,b,q,1);
 cellDots(d,a,cw,1);queryMark(d,a,q);
 if(!moving)d.text('Query cell: '+(100*cw[4]).toFixed(2)+'% of total weight',d.W*.60,47,d.small,C.muted);
 d.math(moving?R`\widehat\tau(x)=`+fmt(forestFit(q).slope):R`\sum_i\alpha_i(x)=1`,d.W*.62,20,d.W*.65,35,d.fs,C.blue);
}
function residualAxes(d){return axes(d,{xmin:-1,xmax:1,ymin:-5.5,ymax:5.5,xlabel:R`D^\circ`,ylabel:R`Y^\circ`,xticks:[-1,0,1],yticks:[-4,0,4]});}
function center(d,p){
 const a=residualAxes(d);d.line(a.x(-1),a.y(0),a.x(1),a.y(0),C.line,1,'4 4');d.line(a.x(0),a.y(-5.5),a.x(0),a.y(5.5),C.line,1,'4 4');
 offset.rows.forEach((o,i)=>{if(baseWeights[i]>0)d.mark(o,a.x(o.dt-p*offset.fit.md),a.y(o.yt-p*offset.fit.my),d.mobile?2.4:3.6,.28);});
 const md=(1-p)*offset.fit.md,my=(1-p)*offset.fit.my;d.circle(a.x(md),a.y(my),8,C.white,C.teal);d.arrow(a.x(offset.fit.md),a.y(offset.fit.my),a.x(md),a.y(my),C.teal);
 d.math(R`(\overline D_\alpha,\overline Y_\alpha)\to(0,0)`,d.W*.59,19,d.W*.72,35,d.mobile?15:20,C.teal);
}
function slope(d,p){
 const a=residualAxes(d);d.line(a.x(-1),a.y(0),a.x(1),a.y(0),C.line,1,'4 4');
 forestRows.forEach((o,i)=>{if(baseWeights[i]>0)d.mark(o,a.x(o.dt),a.y(o.yt),(d.mobile?2.5:3.5)*Math.sqrt(baseWeights[i]/.006),.22);});
 const end=lerp(-.8,.8,p);d.line(a.x(-.8),a.y(-.8*baseFit.slope),a.x(end),a.y(end*baseFit.slope),C.teal,3);
 d.math(R`\widehat\tau(x)=`+fmt(baseFit.slope),d.W*.6,20,d.W*.65,35,d.fs+2,C.teal);
}
function pooling(d,p){
 const {W,H,mobile:z}=d,xs=[W*.2,W*.5,W*.8];
 treeFits.forEach((f,b)=>{
  const x=xs[b];d.text('Tree '+(b+1),x,23,d.small);
  d.math(fmt(f.covariance,3),x,H*.22,W*.29,35,d.fs,C.blue);d.line(x-W*.11,H*.30,x+W*.11,H*.30,C.ink,1.3);d.math(fmt(f.variance,3),x,H*.38,W*.29,35,d.fs,C.rust);
  d.arrow(x,H*.48,lerp(x,W/2,p),H*.66,C.muted);
 });
 d.math(R`\frac{0.992361}{0.220486}\approx4.501`,W/2,H*.76,W-15,z?55:75,z?23:31,C.teal);
 d.text('Mean of the three slopes: 4.517',W/2,H-16,d.small,C.muted);
}
function locality(d,p){
 const {W,H}=d,r=lerp(.6,.05,p),fit=windowFit(r),a=axes(d,{xmin:0,xmax:1,ymin:0,ymax:7,xlabel:'X_1',ylabel:R`\tau`,xticks:[0,.3,.6,1],yticks:[0,2,4,6]});
 const lo=Math.max(0,.3-r),hi=Math.min(1,.3+r);d.rect(a.x(lo),a.y(7),a.x(hi)-a.x(lo),a.y(0)-a.y(7),C.blue,'none',0,.07);
 curve(d,a,forestTau,C.muted,1,'5 4');forestX.forEach(x=>d.circle(a.x(x),a.y(forestTau(x)),6,Math.abs(x-.3)<=r+1e-9?C.blue:C.white,C.blue));
 d.line(a.x(lo),a.y(fit.slope),a.x(hi),a.y(fit.slope),C.teal,3);d.circle(a.x(.3),a.y(4.8),7,C.white,C.rust);
 d.math(R`n_A=`+fit.n+R`,\quad\widehat\tau_A=`+fmt(fit.slope),W*.61,20,W*.75,36,d.fs,C.teal);
}
function overlap(d,p){
 const e=lerp(.5,.02,p),v=e*(1-e),a=axes(d,{xmin:1,xmax:8,ymin:0,ymax:4,xlabel:'t',ylabel:'',xticks:[1,4.8,8],yticks:[0,2,4],left:d.mobile?60:76,top:78});
 const points=Array.from({length:81},(_,i)=>{const t=1+7*i/80;return [a.x(t),a.y(v*(t-4.8)**2)];});path(d,points,C.blue);
 d.math(R`L(t)-L(\tau)`,d.W*.58,18,240,34,d.fs);
 d.math('e='+fmt(e,2)+R`,\quad e(1-e)=`+fmt(v,4),d.W*.62,49,d.W*.72,30,d.small,C.blue);
 d.circle(a.x(4.8),a.y(0),6,C.teal,C.teal);
}
function monteCarlo(d,p){
 const B=Math.max(1,Math.round(lerp(1,64,p))),a=axes(d,{xmin:1,xmax:64,ymin:0,ymax:.0045,xlabel:'B',ylabel:'',xticks:[1,16,32,64],yticks:[],left:d.mobile?66:85,top:55});
 [0,.002,.004].forEach(v=>d.text(v===0?'0':v.toFixed(3),a.left-8,a.y(v),d.small,C.muted,'end'));
 path(d,Array.from({length:64},(_,j)=>[a.x(j+1),a.y(weightMonteCarlo(j+1).sd)]),C.blue);
 d.circle(a.x(B),a.y(weightMonteCarlo(B).sd),7,C.blue,C.blue);
 d.math(R`\operatorname{SD}_\Theta(\alpha_i\mid\mathcal D)=`+weightMonteCarlo(B).sd.toFixed(6),d.W*.59,20,d.W*.8,35,d.mobile?15:20,C.blue);
}
function separation(d,p){
 const {W,H,mobile:z}=d,xs=[W*.19,W*.5,W*.81],names=['Split','Estimate','Omitted'];
 d.text('The same focal person',W/2,19,d.fs);
 names.forEach((t,b)=>{
  const x=xs[b];d.text('Tree '+(b+1),x,H*.21,d.small,C.muted);
  d.rect(x-W*.14,H*.29,W*.28,H*.22,b===2?C.pale:C.white,b===2?C.teal:C.line,3);
  d.text(t,x,H*.40,d.fs,b===2?C.teal:C.muted);
  if(b===2)d.arrow(x,H*.53,lerp(x,W/2,p),H*.73,C.teal);
  else{d.line(x-7,H*.59-7,x+7,H*.59+7,C.rust,2);d.line(x-7,H*.59+7,x+7,H*.59-7,C.rust,2);}
 });
 d.text('Eligible for this person’s OOB prediction',W/2,H*.80,z?14:19,C.teal);
 d.text('Nuisance fitting and evaluation are separate tasks.',W/2,H-15,z?12:16,C.muted);
}
function inference(d,p){
 const {W,H,mobile:z}=d,copies=1+Math.floor(9*p),B=3*copies,sd=Math.sqrt(mean(treeFits.map(f=>(f.slope-mean(treeFits.map(g=>g.slope)))**2))),invalid=sd/Math.sqrt(B);
 d.math('B='+B,W*.22,20,140,32,d.fs,C.muted);d.text('Exact copies',W*.7,20,d.small,C.muted);
 for(let i=0;i<B;i++){const x=W*.13+(i%10)*W*.082,y=H*.20+Math.floor(i/10)*H*.105;d.line(x,y,x-6,y+10,C.muted,1.2);d.line(x,y,x+6,y+10,C.muted,1.2);}
 const x=v=>W*.1+(v-3.8)/1.4*W*.8,y=H*.65;
 d.line(x(3.8),y,x(5.2),y,C.line,1);[4,4.5,5].forEach(v=>d.text(v,x(v),y+22,d.small,C.muted));
 d.line(x(baseFit.slope-invalid),y,x(baseFit.slope+invalid),y,C.rust,z?10:14);d.circle(x(baseFit.slope),y,6,C.white,C.ink);
 d.text('Invalid uncertainty rule',W/2,H*.84,d.fs,C.rust);
 d.math(R`s_{\mathrm{trees}}/\sqrt B=`+fmt(invalid),W/2,H-15,W-12,35,d.fs,C.rust);
}
function evaluate(d,p){
 const {W,H}=d,a=axes(d,{xmin:0,xmax:1,ymin:0,ymax:7,xlabel:'',ylabel:R`\tau`,xticks:[],yticks:[0,2,4,6]});
 const groups=[.2,.5,.8],values=[5.2,4,2.8];
 forestCells.forEach(c=>{const group=c.x<.5?0:c.x>.5?2:1,x=lerp(c.x,groups[group]+(c.k-1)*.025,p);d.circle(a.x(x),a.y(forestTau(c.x)),d.mobile?4:6,C.white,C.blue);});
 values.forEach((v,j)=>{d.line(a.x(groups[j]-.09),a.y(v),a.x(groups[j]-.09+.18*p),a.y(v),C.teal,3);d.text(v,a.x(groups[j]),a.y(v)-25,d.fs,C.teal);d.text(['Lower','Middle','Higher'][j],a.x(groups[j]),H-35,d.small);});
 d.text('Engagement groups',W*.55,H-13,d.small,C.muted);d.text('Known group effects',W*.6,20,d.small,C.muted);
}
function ate(d,p){
 const a=axes(d,{xmin:0,xmax:1,ymin:0,ymax:7,xlabel:'X_1',ylabel:R`\tau`,xticks:[0,.5,1],yticks:[0,2,4,6]});
 curve(d,a,forestTau,C.muted,1,'5 4');
 cellEffects.forEach(({cell:c,estimate,phi})=>{const x=a.x(c.x)+(c.k-1)*(d.mobile?5:8),y=lerp(estimate,phi,p);d.circle(x,a.y(y),d.mobile?4:6,C.white,C.blue);});
 d.math(R`\text{average}=`+lerp(average,4,p).toFixed(6),d.W*.6,20,d.W*.75,36,d.fs,C.teal);
}
export function drawForest(d,kind,p){
 const renderers={'forest-target':target,'forest-identify':identify,'forest-residuals':residuals,'forest-split':split,'forest-honesty':honesty,'forest-partitions':partitions,'forest-leaf-weights':leafWeights,'forest-weights':weights,'forest-query':(d,p)=>weights(d,p,true),'forest-center':center,'forest-slope':slope,'forest-pool':pooling,'forest-locality':locality,'forest-overlap':overlap,'forest-monte-carlo':monteCarlo,'forest-separation':separation,'forest-inference':inference,'forest-evaluate':evaluate,'forest-ate':ate};
 const render=renderers[kind];if(!render)throw new Error('Unknown forest scene: '+kind);render(d,p);
}
