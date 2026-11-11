import {lerp,m,ell} from './model.mjs';
import {q,ivWorld,ivEstimate,ivOLS,ivMoment,compliance,wald,treeSample,leafStats,partition,treeCriterion} from './advanced-model.mjs';
const R=String.raw;
const C={ink:'#242424',blue:'#234E70',rust:'#A54F32',teal:'#007F73',muted:'#606060',line:'#d4d8da',pale:'#f3f6f7',white:'#fff'};
const fmt=(v,n=2)=>Number(v).toFixed(n).replace(/\.?0+$/,'');

function ivDiagram(d,p,instrument){
 const {W,H,mobile:z}=d,r=z?24:34,D=[W*.44,H*.48],Y=[W*.84,H*.48],A=[W*.64,H*.15],X=[W*.40,H*.84],Z=[W*.13,H*.48];
 const edge=(a,b,col=C.muted,progress=1)=>{const angle=Math.atan2(b[1]-a[1],b[0]-a[0]),start=[a[0]+(r+4)*Math.cos(angle),a[1]+(r+4)*Math.sin(angle)],end=[b[0]-(r+5)*Math.cos(angle),b[1]-(r+5)*Math.sin(angle)];d.arrow(...start,lerp(start[0],end[0],progress),lerp(start[1],end[1],progress),col);};
 edge(D,Y,C.blue);edge(A,D,C.rust,instrument?1:p);edge(A,Y,C.rust,instrument?1:p);edge(X,D);edge(X,Y);
 if(instrument){edge(X,Z);edge(Z,D,C.teal,p);}
 [[D,'D',C.blue],[Y,'Y',C.blue],[A,'A',C.rust],[X,'X',C.muted],...(instrument?[[Z,'Z',C.teal]]:[])].forEach(([pos,label,color])=>{d.circle(...pos,r,C.white,color);d.math(label,...pos,80,50,z?25:32,color);});
 d.text('Unobserved cause',W*.64,12,d.small,C.rust);
 d.math(R`\theta_0=2`,W*.66,H*.59,150,36,d.fs,C.blue);
 if(instrument)d.text('Encouragement',Z[0]+6,H*.70,z?12:16,C.teal);
}
function ivScatter(d,p,kind){
 const ols=kind==='iv-ols',first=kind==='iv-first',rows=ivWorld();
 const a=d.axes({xmin:ols?-2.5:-1.4,xmax:ols?2.5:1.4,ymin:first?-2.7:-6.3,ymax:first?2.7:6.3,xlabel:ols?R`\widetilde D`:R`\widetilde Z`,ylabel:first?R`\widetilde D`:R`\widetilde Y`,xticks:ols?[-2,0,2]:[-1,0,1],yticks:first?[-2,0,2]:[-5,0,5],bottom:ols?(d.mobile?90:100):(d.mobile?70:65)});
 d.line(a.x(ols?-2.5:-1.4),a.y(0),a.x(ols?2.5:1.4),a.y(0),C.line,1,'4 4');
 rows.forEach(o=>{const x=a.x(ols?o.dt:o.zt),y=a.y(first?o.dt:o.yt);if(ols)d.circle(x,y,d.mobile?4:5,C.white,C.blue,.65);else d.mark({d:o.v>0},x,y,d.mobile?4:5,.65);});
 const b=ols?ivOLS():first?1:2,x0=ols?-2.1:-1,x1=lerp(x0,ols?2.1:1,p);
 d.line(a.x(x0),a.y(b*x0),a.x(x1),a.y(b*x1),ols?C.rust:C.teal,3,ols?'':'6 4');
 if(ols){d.line(a.x(-2.1),a.y(-4.2),a.x(2.1),a.y(4.2),C.teal,2,'5 5');d.math(R`\text{slope}=2.5`,d.W*.65,24,200,35,d.fs,C.rust);d.text('Dashed: structural slope 2',d.W*.56,d.H-50,d.small,C.teal);}
 else{[-1,1].forEach(v=>d.circle(a.x(v),a.y(b*v),d.mobile?7:9,C.white,C.teal));d.math(R`\text{slope}=`+b,d.W*.56,22,180,35,d.fs,C.teal);}
}
function ivNuisances(d,p,residual=false){
 const {W,H,mobile:z}=d,rows=ivWorld(),width=(W-12)/3,keys=['y','d','z'],fns=[ell,m,q],rawRanges=[[17,36],[-.3,6.5],[-.7,3]],resRanges=[[-6,6],[-2.5,2.5],[-1.3,1.3]];
 keys.forEach((key,k)=>{
  const progress=residual?p:0,left=8+k*width+(z?20:30),right=8+(k+1)*width-12,top=55,bottom=H-58;
  const min=lerp(rawRanges[k][0],resRanges[k][0],progress),max=lerp(rawRanges[k][1],resRanges[k][1],progress),x=v=>lerp(left,right,(v+1.25)/2.5),y=v=>lerp(bottom,top,(v-min)/(max-min));
  d.line(left,top,left,bottom,C.line,1);d.line(left,bottom,right,bottom,C.ink,1);
  [residual?0:Math.ceil(min),Math.floor(max)].forEach(v=>{d.line(left-3,y(v),left,y(v),C.muted,1);d.text(fmt(v),left-6,y(v),z?11:14,C.muted,'end');});
  [-1,1].forEach(v=>d.text(v,x(v),bottom+17,d.small,C.muted));
  d.math(residual?['\\widetilde Y','\\widetilde D','\\widetilde Z'][k]:key.toUpperCase(),(left+right)/2,20,120,30,d.fs+3);
  d.math('X',(left+right)/2,H-15,60,30,d.small);
  rows.forEach(o=>d.mark({d:o.v>0},x(o.x),y(o[key]-progress*fns[k](o.x)),z?2.6:3.8,.6));
  if(residual)d.line(left,y(0),right,y(0),C.teal,1.5,'4 4',p);
  const pts=Array.from({length:Math.max(2,Math.floor((residual?1:p)*50))},(_,j)=>{const u=-1+2*j/49;return [x(u),y((1-progress)*fns[k](u))];});
  d.path(pts.map(([x,y],i)=>`${i?'L':'M'}${x} ${y}`).join(' '),C.rust,2.5);
 });
}
function momentBars(d,p,kind){
 const {W,H,mobile:z}=d,late=kind==='iv-late',invalid=kind==='iv-invalid',delta=invalid?.6*p:0;
 const vals=late?[wald.reduced,wald.first,wald.reduced/wald.first]:[2+delta,1,2+delta];
 const names=late?[R`\Delta_ZY`,R`\Delta_ZD`,R`\theta_{\mathrm{LATE}}`]:[R`\mathbb E[\widetilde Z\widetilde Y]`,R`\mathbb E[\widetilde Z\widetilde D]`,R`\widehat\theta`];
 const left=z?168:270,right=W-42,scale=v=>left+(right-left)*v/(late?4.6:3.2),ys=[H*.23,H*.48,H*.76],cols=[C.blue,C.rust,C.teal];
 vals.forEach((v,i)=>{d.math(names[i],left/2,ys[i],left-12,45,d.fs,cols[i]);d.line(left,ys[i],scale(invalid?v:v*p),ys[i],cols[i],z?12:18);d.text(fmt(v),scale(v)+12,ys[i],d.fs,cols[i],'start');});
 d.text(late?'Separate binary-IV population':invalid?`Direct effect: ${fmt(delta)}`:'Pooled cross-products / sample size',W/2,H-18,d.small,C.muted);
}
function ivScore(d,p,weak=false){
 const pi=weak?lerp(1,.08,p):1,proposal=lerp(.4,2,p),a=d.axes({xmin:0,xmax:4,ymin:-2.2,ymax:2.2,xlabel:R`\theta`,ylabel:R`\overline\psi(\theta)`,xticks:[0,1,2,3,4],yticks:[-2,0,2],bottom:d.mobile?70:65});
 d.line(a.x(0),a.y(0),a.x(4),a.y(0),C.line,1.5);
 d.line(a.x(0),a.y(ivMoment(0,pi)),a.x(4),a.y(ivMoment(4,pi)),C.blue,3);
 if(weak){
  d.line(a.x(0),a.y(ivMoment(0,pi,.08)),a.x(4),a.y(ivMoment(4,pi,.08)),C.rust,2,'5 4');
  d.circle(a.x(2),a.y(0),5,C.white,C.blue);d.circle(a.x(2+.08/pi),a.y(0),6,C.rust,C.rust);
  d.math(R`\pi=`+fmt(pi),d.W*.68,24,160,35,d.fs,C.blue);
  d.math(R`2+0.08/\pi=`+fmt(2+.08/pi),d.W*.65,d.mobile?60:70,260,35,d.small,C.rust);
 }else{
  d.line(a.x(proposal),a.y(0),a.x(proposal),a.y(ivMoment(proposal)),C.rust,2,'4 4');
  d.circle(a.x(proposal),a.y(ivMoment(proposal)),7,C.teal,C.teal);
  d.math(R`\theta=`+fmt(proposal),d.W*.65,24,160,35,d.fs,C.teal);
 }
}
function ivOrthogonal(d,p){
 const a=d.axes({xmin:-.8,xmax:.8,ymin:1.65,ymax:2.08,xlabel:'t',ylabel:R`\widehat\theta(t)`,xticks:[-.8,0,.8],yticks:[1.7,1.85,2],bottom:d.mobile?70:65});
 const pts=Array.from({length:101},(_,i)=>{const t=-.8+1.6*i/100;return [a.x(t),a.y(ivEstimate(ivWorld(),t).value)];});
 d.path(pts.map(([x,y],i)=>`${i?'L':'M'}${x} ${y}`).join(' '),C.blue,3);
 d.line(a.x(-.8),a.y(2),a.x(.8),a.y(2),C.teal,1.5,'5 4');
 const t=.8*(1-p);d.circle(a.x(t),a.y(ivEstimate(ivWorld(),t).value),7,C.blue,C.blue);
 d.math(R`\theta_0=2`,d.W*.65,25,160,35,d.fs,C.teal);
}
function ivInference(d,p){
 const {W,H,mobile:z}=d,J=lerp(1,.25,p),left=z?170:270,right=W-45,x=v=>left+(right-left)*v/4.4;
 [J,1/J].forEach((v,i)=>{const y=H*(i?.65:.25),col=i?C.rust:C.blue;d.math(i?R`\mathrm{se}/\mathrm{se}_{J=1}`:R`J`,left/2,y,left-15,50,d.fs,col);d.line(left,y,x(v),y,col,z?15:22);d.text(fmt(v),x(v)+10,y,d.fs,col,'start');});
 d.text('Hold score variation and sample size fixed',W/2,H-25,d.small,C.muted);
}
function ivCompliers(d,p){
 const {W,H,mobile:z}=d,x0=W*.58,x1=W*.84,gap=(H-80)/10;
 d.math('Z=0',x0,23,110,34,d.fs);d.math('Z=1',x1,23,110,34,d.fs);
 compliance.forEach((o,i)=>{const y=55+i*gap;d.text(o.type==='always'?'Always-taker':o.type==='complier'?'Complier':'Never-taker',z?18:55,y,d.small,o.type==='complier'?C.blue:C.muted,'start');d.mark({d:o.d0},x0,y,z?4:6);d.arrow(x0+12,y,x1-12,y,C.line,p);d.mark({d:p<.5?o.d0:o.d1},lerp(x0,x1,p),y,z?4:6);});
 d.text('Circle: treated · square: control',W/2,H-10,d.small,C.muted);
}
function ivLateScore(d,p){
 const {W,H,mobile:z}=d,xs=[W*.15,W*.5,W*.84],ys=[H*.25,H*.55];
 ['Y','D'].forEach((a,i)=>{const y=ys[i],col=i?C.rust:C.blue;d.rect(xs[0]-30,y-23,60,46,C.white,col,2);d.math(a,xs[0],y,60,45,d.fs+4,col);d.arrow(xs[0]+34,y,lerp(xs[0]+34,xs[1]-48,p),y,col);d.math(R`\widehat\phi^{`+a+'}',xs[1],y,100,55,d.fs+7,col);d.arrow(xs[1]+50,y,lerp(xs[1]+50,xs[2]-32,p),y,col);d.math(i?'0.5':'2',xs[2],y,80,50,d.fs+8,col);});
 d.math(R`\widehat\theta_{\mathrm{LATE}}=2/0.5=4`,W/2,H*.83,W-10,55,z?24:32,C.teal);
 d.text('Cross-fit → average each contrast → take the ratio',W/2,H-11,z?12:16,C.muted);
}
function mapPoints(d,rows,x,y,p=1,from=null,small=false){
 rows.forEach(o=>{
  const j=[-1.5,-.5,.5,1.5].indexOf(o.e),spread=small?2.4:d.mobile?4:6;
  const px=x(o.x1)+(j-1.5)*spread,py=y(o.x2)+(o.d?-1:1)*(small?3.5:d.mobile?7:10);
  d.mark(o,from?lerp(from[0],px,p):px,from?lerp(from[1],py,p):py,small?1.8:d.mobile?2.8:4,.8);
 });
}
function treeMap(d,p,kind){
 const {W,H,mobile:z}=d,rows=treeSample(kind==='tree-freeze'?'estimation':'structure'),a=d.axes({xmin:0,xmax:1,ymin:0,ymax:1,xlabel:'X_1',ylabel:'X_2',xticks:[0,.5,1],yticks:[0,.5,1],bottom:z?112:120});
 if(kind==='tree-outcome')d.rect(a.x(0),a.y(1),a.x(1)-a.x(0),a.y(.5)-a.y(1),C.pale,'none');
 if(['tree-effect','tree-freeze','tree-support'].includes(kind))d.rect(a.x(0),a.y(1),a.x(.5)-a.x(0),a.y(0)-a.y(1),C.pale,'none');
 mapPoints(d,rows,a.x,a.y,p,kind==='tree-setup'?[W/2,H*.43]:kind==='tree-freeze'?[a.x(0),a.y(.5)]:null);
 let summary=R`n_1=n_0=32`;
 if(kind==='tree-root')summary=R`\widehat\tau=27-23=4`;
 if(kind==='tree-outcome'){
  d.line(a.x(0),a.y(.5),a.x(p),a.y(.5),C.rust,2.5);summary=R`\overline Y:22,\ 28\qquad\widehat\tau:4,\ 4`;
 }
 if(kind==='tree-effect'){
  d.line(a.x(0),a.y(.5),a.x(1),a.y(.5),C.rust,1,'5 4',1-p);
  d.line(a.x(.5),a.y(0),a.x(.5),a.y(p),C.teal,2.5);summary=R`\widehat\tau_L=6\qquad\widehat\tau_R=2`;
 }
 if(['tree-freeze','tree-support'].includes(kind)){
  d.line(a.x(.5),a.y(0),a.x(.5),a.y(1),C.teal,2.5);d.line(a.x(0),a.y(.5),a.x(1),a.y(.5),C.teal,2);
  if(kind==='tree-freeze')[[.25,.08],[.25,.58],[.75,.08],[.75,.58]].forEach(([x,y],i)=>d.math('L_'+(i+1),a.x(x),a.y(y),55,28,d.small,C.teal));
  summary=R`n_{1,L}=n_{0,L}=8`;
  if(kind==='tree-support'){
   d.line(a.x(.25),a.y(0),a.x(.25),a.y(.5*p),C.rust,2,'5 4');summary=R`\text{new leaves: }n_1=n_0=4<8`;
  }
 }
 d.math(summary,W/2,H-63,W-8,36,z?18:24,kind==='tree-support'?C.rust:C.teal);
}
function treeCriterionBars(d,p,fitOnly=false){
 const {W,H,mobile:z}=d,rows=treeSample(),kinds=['root','x2','x1'],names=['Root','Baseline knowledge','Engagement'],left=z?162:260,right=W-38,x=v=>left+(right-left)*v/22,ys=[H*.22,H*.46,H*.70];
 kinds.forEach((kind,i)=>{const c=treeCriterion(rows,kind),penalty=fitOnly?0:c.penalty*p,y=ys[i];d.text(names[i],left-14,y,d.small,C.ink,'end');d.line(left,y,x(c.fit*(fitOnly?p:1)),y,C.blue,z?16:22);if(!fitOnly)d.line(x(c.fit-penalty),y,x(c.fit),y,C.rust,z?16:22);d.line(x(c.fit-penalty),y-13,x(c.fit-penalty),y+13,C.ink,2);d.text(fmt(c.fit-penalty),x(c.fit-penalty),y-25,d.small,C.ink);});
 d.line(left,H-45,right,H-45,C.ink,1);[0,10,20].forEach(v=>d.text(v,x(v),H-25,d.small,C.muted));
 d.text(fitOnly?'Weighted squared leaf effects':'Blue: remaining score · rust: subtracted penalty',W/2,15,z?13:17,C.muted);
}
function treeGrow(d,p){
 const {W,H,mobile:z}=d,left=30,right=W*.44,top=55,bottom=H*.7,x=v=>lerp(left,right,v),y=v=>lerp(bottom,top,v),root=[W*.73,55],childY=H*.36,leafY=H*.65;
 d.rect(left,top,right-left,bottom-top,C.white,C.line);mapPoints(d,treeSample(),x,y,1,null,true);
 d.line(x(.5),top,x(.5),bottom,C.teal,2);d.line(x(.5),y(.5),x(.5-.5*p),y(.5),C.teal,2);d.line(x(.5),y(.5),x(.5+.5*p),y(.5),C.teal,2);
 [[.25,.08],[.25,.58],[.75,.08],[.75,.58]].forEach(([u,v],i)=>d.math('L_'+(i+1),x(u),y(v),45,24,z?11:15,C.teal,p));
 d.math('X_1',(left+right)/2,bottom+27,80,30,d.small);d.math('X_2',left,top-25,80,30,d.small);
 const children=[W*.60,W*.87],leaves=[W*.53,W*.67,W*.80,W*.94];
 children.forEach((cx,i)=>{d.line(root[0],root[1]+17,cx,childY-17,C.ink,1.5);d.math(R`X_2<.5`,cx,childY,z?105:140,38,d.small,C.ink);for(let j=0;j<2;j++){const lx=leaves[2*i+j];d.line(cx,childY+18,lerp(cx,lx,p),lerp(childY+18,leafY-17,p),C.teal,1.5);d.math(i?'2':'6',lx,leafY,60,40,d.fs+5,C.teal,p);d.math('L_'+(2*i+j+1),lx,leafY+33,60,28,d.small,C.muted,p);}});
 d.math(R`X_1<.5`,...root,z?120:170,38,d.small,C.ink);
 d.math(R`H=20\qquad P=`+fmt(lerp(treeCriterion(treeSample(),'x1').penalty,treeCriterion(treeSample(),'four').penalty,p)),W/2,H-25,W-10,42,z?21:28,C.teal);
}
function treeSearch(d,p){
 const {W,H,mobile:z}=d,rows=treeSample('search',true),cuts=[.25,.5,.75],left=z?148:240,right=W-35,x=v=>left+(right-left)*(v-15)/6;
 cuts.forEach((cut,i)=>{const c=treeCriterion(rows,'x1',cut),y=H*(.25+i*.23),active=i===Math.min(2,Math.floor(p*3));d.math('X_1<'+cut,left/2,y,left-10,40,d.fs);d.line(x(15),y,x(lerp(15,c.fit,p)),y,i===1?C.blue:C.muted,z?14:20);d.text(fmt(c.fit),x(c.fit)+10,y,d.small,C.ink,'start');if(active)d.line(x(15),y+17,x(c.fit),y+17,C.teal,2);});
 d.line(x(16),H*.13,x(16),H*.82,C.rust,1.5,'4 4');d.math(R`4^2=16`,x(16),H-25,140,35,d.small,C.rust);
 d.text('Effect-fit term after searching three cuts',W/2,17,d.small,C.muted);
}
function treeReplicate(d,p){
 const {W,H,mobile:z}=d,initial=partition(treeSample('search',true),'x1').map(leafStats),fresh=partition(treeSample('confirm',true),'x1').map(leafStats),left=z?95:150,right=W-40,x=v=>left+(right-left)*v/7;
 initial.forEach((o,i)=>{const y=H*(i?.65:.3),value=lerp(o.tau,fresh[i].tau,p);d.math(i?'R':'L',left/2,y,75,40,d.fs+4);d.line(left,y,x(value),y,C.blue,z?16:23);d.text(fmt(value),x(value)+14,y,d.fs,C.blue,'start');});
 d.line(x(4),35,x(4),H-57,C.teal,2,'5 4');d.math(R`\tau=4`,x(4),H-25,150,38,d.fs,C.teal);
 d.text('Fixed split · new outcomes',W/2,15,d.small,C.muted);
}
function treeHonesty(d,p,distinction=false){
 const {W,H,mobile:z}=d;
 if(distinction){
  const cols=[W*.17,W*.5,W*.84],bw=W*.27,ys=[H*.30,H*.72],labels=[['Structure','Partition','Leaf effects'],['Training folds','Nuisances','Held-out score']];
  ys.forEach((y,i)=>{d.text(i?'Cross-fitting':'Honesty',W/2,y-H*.17,d.fs,i?C.blue:C.teal);labels[i].forEach((label,j)=>{d.rect(cols[j]-bw/2,y-23,bw,46,C.white,i?C.blue:C.teal,3);d.text(label,cols[j],y,z?13:18);if(j<2)d.arrow(cols[j]+bw/2+2,y,lerp(cols[j]+bw/2+2,cols[j+1]-bw/2-3,p),y,C.muted);});});
  d.text('New outcomes estimate the final object',W/2,H-15,d.small,C.muted);return;
 }
 const centers=[W*.25,W*.75],boxY=H*.55,boxW=W*.40,boxH=H*.54;
 centers.forEach((cx,i)=>{d.rect(cx-boxW/2,boxY-boxH/2,boxW,boxH,C.white,i?C.blue:C.teal,3);d.text(i?'Estimate effects':'Choose the tree',cx,32,d.fs,i?C.blue:C.teal);d.math(i?R`S_{\mathrm{estimation}}`:R`S_{\mathrm{structure}}`,cx,H*.22,boxW,35,d.small);});
 for(let i=0;i<32;i++){const group=i<16?0:1,j=i%16,cx=centers[group]+((j%4)-1.5)*(z?25:42),cy=boxY+(Math.floor(j/4)-1.5)*(z?24:37);d.circle(lerp(W/2,cx,p),lerp(H*.5,cy,p),z?4:6,C.white,group?C.blue:C.teal);}
 d.text('Select and prune',centers[0],H-22,d.small,C.teal);d.text('Keep outcomes reserved',centers[1],H-22,z?12:17,C.blue);
}
function leafEstimates(d,p,intervals=false){
 const {W,H,mobile:z}=d,leaves=partition(treeSample('estimation'),'four').map(leafStats),left=z?57:85,right=W-32,min=intervals?0:18,max=intervals?8:35,x=v=>lerp(left,right,(v-min)/(max-min)),ys=[.16,.35,.54,.73].map(v=>H*v);
 leaves.forEach((l,i)=>{const y=ys[i];d.math('L_'+(i+1),left-30,y,55,32,d.small);
  if(intervals){const hw=1.96*l.se*p;d.line(x(l.tau-hw),y,x(l.tau+hw),y,C.blue,2.5);[l.tau-hw,l.tau+hw].forEach(v=>d.line(x(v),y-5,x(v),y+5,C.blue,1.5));d.circle(x(l.tau),y,z?4:6,C.blue,C.blue);const truth=i<2?6:2;d.line(x(truth),y-14,x(truth),y+14,C.teal,1.5,'3 3');d.text(fmt(l.tau),x(l.tau),y-21,d.small,C.blue);}
  else{d.line(x(l.y0),y,x(lerp(l.y0,l.y1,p)),y,C.teal,2);d.mark({d:0},x(l.y0),y,z?4:6);d.mark({d:1},x(l.y1),y,z?4:6);d.math(R`\widehat\tau=`+fmt(l.tau),(x(l.y0)+x(l.y1))/2,y-23,140,30,d.small,C.teal);}
 });
 const base=H-49;d.line(left,base,right,base,C.ink,1);
 (intervals?[0,2,4,6,8]:[20,25,30,35]).forEach(v=>d.text(v,x(v),base+18,d.small,C.muted));
 d.text(intervals?'Effect · dashed marks show known truth':'Knowledge-score means · circle treated / square control',W/2,H-8,z?12:16,C.muted);
}
function treeTarget(d,p){
 const a=d.axes({xmin:0,xmax:1,ymin:1.5,ymax:6.5,xlabel:'X',ylabel:R`\tau(X)`,xticks:[0,.5,1],yticks:[2,3,4,5,6],bottom:d.mobile?70:65});
 d.line(a.x(0),a.y(6),a.x(1),a.y(2),C.muted,2,'5 4');d.line(a.x(.5),a.y(1.5),a.x(.5),a.y(6.5),C.line,1,'4 4');
 [[0,.5,5],[.5,1,3]].forEach(([lo,hi,avg])=>{d.line(a.x(lo),a.y(lerp(6-4*lo,avg,p)),a.x(hi),a.y(lerp(6-4*hi,avg,p)),C.blue,3);d.math(R`\tau_L=`+avg,a.x((lo+hi)/2),a.y(avg)+(avg===5?-25:25),140,35,d.small,C.blue,p);});
 d.text('Dashed: smooth effect · solid: tree approximation',d.W*.53,12,d.mobile?11:16,C.muted);
}
export function drawAdvanced(d,kind,p){
 switch(kind){
  case 'iv-endogeneity':ivDiagram(d,p,false);break;
  case 'iv-instrument':ivDiagram(d,p,true);break;
  case 'iv-ols':case 'iv-first':case 'iv-reduced':ivScatter(d,p,kind);break;
  case 'iv-nuisance':ivNuisances(d,p);break;
  case 'iv-residuals':ivNuisances(d,p,true);break;
  case 'iv-ratio':case 'iv-invalid':case 'iv-late':momentBars(d,p,kind);break;
  case 'iv-score':ivScore(d,p);break;
  case 'iv-weak':ivScore(d,p,true);break;
  case 'iv-orthogonal':ivOrthogonal(d,p);break;
  case 'iv-inference':ivInference(d,p);break;
  case 'iv-compliers':ivCompliers(d,p);break;
  case 'iv-late-score':ivLateScore(d,p);break;
  case 'tree-setup':case 'tree-root':case 'tree-outcome':case 'tree-effect':case 'tree-support':case 'tree-freeze':treeMap(d,p,kind);break;
  case 'tree-fit':treeCriterionBars(d,p,true);break;
  case 'tree-criterion':treeCriterionBars(d,p);break;
  case 'tree-grow':treeGrow(d,p);break;
  case 'tree-search':treeSearch(d,p);break;
  case 'tree-replicate':treeReplicate(d,p);break;
  case 'tree-honesty':treeHonesty(d,p);break;
  case 'tree-distinction':treeHonesty(d,p,true);break;
  case 'tree-estimate':leafEstimates(d,p);break;
  case 'tree-intervals':leafEstimates(d,p,true);break;
  case 'tree-target':treeTarget(d,p);break;
  default:throw new Error('Unknown advanced scene '+kind);
 }
}
