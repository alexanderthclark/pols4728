import {lerp,mean} from './model.mjs';
import {metaRows,metaX,metaMu,metaTau,metaE,metaEll,fitS,tFit,xPseudo,xArm,xBlend,tError,drRows,drFit,rLoss,rLocal,rInformation} from './meta-model.mjs';
const R=String.raw;
const C={ink:'#242424',blue:'#234E70',rust:'#A54F32',teal:'#007F73',muted:'#606060',line:'#d4d8da',pale:'#f3f6f7',white:'#fff'};
const fmt=(v,n=2)=>Number(v).toFixed(n).replace(/\.?0+$/,'');
const curve=(d,a,fn,col=C.blue,p=1,dash='',lo=0,hi=1)=>{const end=lerp(lo,hi,p),points=Array.from({length:81},(_,j)=>{const x=lerp(lo,end,j/80);return [a.x(x),a.y(fn(x))];});d.path(points.map(([x,y],i)=>`${i?'L':'M'}${x} ${y}`).join(' '),col,2.5,dash);};
const effectAxes=(d,options={})=>d.axes({xmin:-.05,xmax:1.05,ymin:0,ymax:7,xlabel:'X',ylabel:R`\tau(X)`,xticks:[0,.5,1],yticks:[0,2,4,6],bottom:d.mobile?70:65,...options});
const outcomeAxes=d=>effectAxes(d,{ymin:18,ymax:36,ylabel:'Y',yticks:[20,25,30,35]});
const dots=(d,a,rows,yfn=o=>o.y,opacity=.6)=>rows.forEach(o=>d.mark(o,a.x(o.x),a.y(yfn(o)),d.mobile?3:4.2,opacity));
function outcomeCurves(d,a,p=1,labels=true){
 [0,1].forEach(t=>curve(d,a,x=>metaMu(t,x),t?C.blue:C.rust,p,t?'':'5 4'));
 if(labels){d.math(R`\mu_0`,d.W*.37,20,90,34,d.fs,C.rust);d.math(R`\mu_1`,d.W*.70,20,90,34,d.fs,C.blue);}
}
function metaTarget(d,p,contrast=false){
 const a=outcomeAxes(d),x=contrast?.25:lerp(.1,.9,p),low=metaMu(0,x),high=metaMu(1,x);outcomeCurves(d,a);
 if(!contrast)dots(d,a,metaRows,o=>o.y,.22);
 d.line(a.x(x),a.y(low),a.x(x),a.y(contrast?lerp(low,high,p):high),C.teal,3);
 d.circle(a.x(x),a.y(low),5,C.white,C.rust);d.circle(a.x(x),a.y(contrast?lerp(low,high,p):high),6,C.blue,C.blue);
 d.math(R`\tau(x)=`+fmt(high-low),a.x(x)+(x>.65?-75:75),a.y((low+high)/2),150,40,d.fs,C.teal);
 if(contrast){d.text('22.25',a.x(x)-14,a.y(low)+17,d.small,C.rust,'end');d.text('27.25',a.x(x)-14,a.y(high)-17,d.small,C.blue,'end');}
}
function assignment(d,p){
 const {W,H,mobile:z}=d,start=z?92:160,gap=(W-start-25)/16;
 metaX.forEach((x,j)=>{const y=lerp(H*.5,48+j*(H-88)/4,p);d.math('X='+x,start*.48,48+j*(H-88)/4,start-15,34,d.small);metaRows.filter(o=>o.x===x).forEach((o,i)=>d.mark(o,start+(i+.5)*gap,y,z?3.4:5));});
 d.text('Treated counts: 4, 6, 8, 10, 12 out of 16',W/2,H-14,d.small,C.muted);
}
function sFit(d,p){
 const a=outcomeAxes(d),fit=fitS();dots(d,a,metaRows);
 [0,1].forEach(t=>{curve(d,a,x=>fit.predict(x,t),t?C.blue:C.rust,p,t?'':'5 4');d.math(R`\widehat f(x,`+t+')',d.W*(t?.72:.36),20,160,36,d.fs,t?C.blue:C.rust);});
}
function sEffect(d,p,shrink=false){
 const a=effectAxes(d),fit=fitS(shrink?.01*p:0,shrink),lambda=.01*p;
 curve(d,a,metaTau,C.muted,1,'5 4');
 curve(d,a,x=>shrink?fit.effect(x):lerp(metaTau(x),fit.effect(x),p),C.blue);
 metaX.forEach(x=>d.circle(a.x(x),a.y(shrink?fit.effect(x):lerp(metaTau(x),fit.effect(x),p)),4,C.blue,C.blue));
 d.math(shrink?R`\lambda=`+fmt(lambda,3)+R`,\ \operatorname{MSE}_Y=`+fmt(fit.mse):R`\widehat\tau_S(x)\longrightarrow4`,d.W*.61,20,d.W*.65,40,d.mobile?16:21,C.blue);
}
function twoPanels(d,mode,p){
 const {W,H,mobile:z}=d,width=W/2;
 [0,1].forEach(t=>{
  const left=t*width+(z?33:47),right=(t+1)*width-18,top=65,bottom=H-62,ymin=mode==='outcome'?18:0,ymax=mode==='outcome'?36:8;
  const a={x:x=>lerp(left,right,(x+.05)/1.1),y:y=>bottom-(bottom-top)*(y-ymin)/(ymax-ymin)};
  d.line(left,top,left,bottom,C.ink,1);d.line(left,bottom,right,bottom,C.ink,1);
  (mode==='outcome'?[20,28,36]:[0,4,8]).forEach(y=>d.text(y,left-7,a.y(y),d.small,C.muted,'end'));
  [0,1].forEach(x=>d.text(x,a.x(x),bottom+18,d.small,C.muted));d.math('X',(left+right)/2,H-15,70,30,d.small);
  d.math(mode==='outcome'?R`\widehat\mu_`+t:R`\widehat\tau_`+t,(left+right)/2,25,130,40,d.fs+3,t?C.blue:C.rust);
  const rows=(mode==='outcome'?metaRows:xPseudo()).filter(o=>o.d===t);dots(d,a,rows,mode==='outcome'?o=>o.y:o=>o.pseudo);
  curve(d,a,x=>mode==='outcome'?tFit.predict(x,t):xArm(x,t),t?C.blue:C.rust,p,t?'':'5 4');
 });
}
function tDifference(d,p){
 const a=effectAxes(d,{ymin:lerp(18,-1,p),ymax:lerp(36,8,p),ylabel:p===1?R`\widehat\tau_T`:'Y',yticks:[lerp(20,0,p),lerp(28,4,p),lerp(36,8,p)]});
 curve(d,a,x=>tFit.predict(x,0)*(1-p),C.rust,1,'5 4');curve(d,a,x=>tFit.predict(x,1)-p*tFit.predict(x,0),C.blue);
 metaX.forEach(x=>{d.circle(a.x(x),a.y(tFit.predict(x,1)-p*tFit.predict(x,0)),4,C.blue,C.blue);});
 d.math(p===1?R`\widehat\tau_T(x)=6-4x`:R`\widehat\mu_1-\widehat\mu_0`,d.W*.60,20,280,38,d.fs,C.blue);
}
function tErrors(d,p){
 const a=effectAxes(d),fn=x=>metaTau(x)+p*(tError(1,x)-tError(0,x));
 curve(d,a,metaTau,C.muted,1,'5 4');curve(d,a,fn,C.blue);
 d.line(a.x(.25),a.y(metaTau(.25)),a.x(.25),a.y(fn(.25)),C.rust,3);
 d.math(R`\delta_1-\delta_0=`+fmt(-1.2*p),a.x(.35),a.y(2.5),240,38,d.small,C.rust);
 d.text('Dashed: true effect',d.W*.65,18,d.small,C.muted);
}
function xImpute(d,p,t){
 const a=effectAxes(d,{ymin:lerp(18,0,p),ymax:lerp(36,8,p),ylabel:p===1?(t?R`\widehat\Delta^{\,1}`:R`\widehat\Delta^{\,0}`):'Y',yticks:[lerp(20,0,p),lerp(28,4,p),lerp(36,8,p)]});
 const rows=xPseudo().filter(o=>o.d===t);dots(d,a,rows,o=>lerp(o.y,o.pseudo,p));
 curve(d,a,x=>lerp(metaMu(t,x),metaTau(x),p),t?C.blue:C.rust,1,t?'':'5 4');
 d.math(t?R`Y-\widehat\mu_0(X)`:R`\widehat\mu_1(X)-Y`,d.W*.62,20,250,36,d.fs,t?C.blue:C.rust);
}
function xWeights(d,p){
 const {W,H,mobile:z}=d,g=lerp(.5,.2,p),left=W*.48,right=W-25;
 [g,1-g].forEach((w,i)=>{const y=H*(i?.40:.17),col=i?C.blue:C.rust;d.math(i?R`\widehat\tau_1=5.2`:R`\widehat\tau_0=6`,W*.22,y,W*.42,45,d.fs,col);d.line(left,y,right,y,C.line,14);d.line(left,y,lerp(left,right,w),y,col,14);d.math((i?'1-g=':'g=')+fmt(w),W*.72,y+27,W*.5,32,d.small,col);});
 const v=xBlend(6,5.2,g),x=u=>35+(W-70)*(u-5)/1.2,y=H*.82;
 d.math(R`\widehat\tau_X=`+fmt(v),W/2,H*.63,W-15,45,z?26:34,C.teal);
 d.line(x(5),y,x(6.2),y,C.ink,1);[5,5.5,6].forEach(v=>d.text(v,x(v),y+20,d.small,C.muted));d.circle(x(v),y,6,C.teal,C.teal);d.line(x(5),y-10,x(5),y+10,C.muted,1.5,'3 3');
 d.text('True effect: 5',W/2,H-7,d.small,C.muted);
}
function xImbalance(d,p){
 const {W,H,mobile:z}=d,cx=[W*.25,W*.75],top=H*.20,gap=z?8:12;
 d.text('80 controls',cx[0],20,d.fs,C.rust);d.text('20 treated',cx[1],20,d.fs,C.blue);
 for(let i=0;i<80;i++)d.mark({d:0},cx[0]+((i%10)-4.5)*gap,top+Math.floor(i/10)*gap,z?2:3);
 for(let i=0;i<20;i++)d.mark({d:1},cx[1]+((i%5)-2)*gap*1.4,top+Math.floor(i/5)*gap*1.4,z?2.6:3.6);
 const y=H*.61;d.arrow(cx[0],top+8*gap,cx[0],y-25,C.rust);d.math(R`\widehat\mu_0`,cx[0],y,100,50,d.fs+7,C.rust);
 d.arrow(cx[0]+55,y,lerp(cx[0]+55,cx[1]-75,p),y,C.rust);d.arrow(cx[1],top+6*gap,cx[1],y-25,C.blue);
 d.math(R`Y-\widehat\mu_0`,cx[1],y,z?160:240,50,d.fs+2,C.blue);
 d.arrow(cx[1],y+24,cx[1],H*.83-18,C.blue);d.math(R`\widehat\tau_1`,cx[1],H*.85,120,45,d.fs+8,C.blue);
 d.math(R`1-e=0.8`,cx[0],H*.85,160,38,d.fs,C.teal);
 d.text('Control predictions support treated-arm effect learning',W/2,H-8,z?11:16,C.muted);
}
function structure(d,p){
 const {W,H,mobile:z}=d,split=W*.48,top=55,bottom=H-70;
 const a={x:x=>30+(split-50)*x,y:y=>bottom-(bottom-top)*(y-18)/18};
 const b={x:x=>split+45+(W-split-65)*x,y:y=>bottom-(bottom-top)*y/8};
 [a,b].forEach(v=>{d.line(v.x(0),top,v.x(0),bottom,C.line,1);d.line(v.x(0),bottom,v.x(1),bottom,C.ink,1);[0,1].forEach(x=>d.text(x,v.x(x),bottom+17,d.small,C.muted));d.math('X',(v.x(0)+v.x(1))/2,H-15,80,30,d.small);});
 [20,28,36].forEach(y=>d.text(y,a.x(0)-6,a.y(y),z?11:15,C.muted,'end'));[0,4,8].forEach(y=>d.text(y,b.x(0)-6,b.y(y),z?11:15,C.muted,'end'));
 [0,1].forEach(t=>curve(d,a,x=>metaMu(t,x),t?C.blue:C.rust,1,t?'':'5 4'));curve(d,b,metaTau,C.teal,p);
 d.math(R`\mu_0,\ \mu_1`,split/2,25,170,40,d.fs);d.math(R`\tau`,(split+W)/2,25,150,40,d.fs+4,C.teal);
}
function evaluation(d,p){
 const {W,H,mobile:z}=d,xs=[W*.18,W*.50,W*.83],bw=W*.27,y=H*.48;
 d.rect(10,H*.19,W*.64,H*.58,C.pale,C.line,3);d.rect(W*.69,H*.19,W*.29,H*.58,C.white,C.teal,3);
 d.text('Training observations',W*.34,H*.12,d.fs);d.text('Held out',W*.835,H*.12,d.fs,C.teal);
 ['Outcome fits','Effect estimate','Evaluation'].forEach((name,i)=>{d.rect(xs[i]-bw/2,y-25,bw,50,C.white,i===2?C.teal:C.blue,2);d.text(name,xs[i],y,z?12:18);if(i<2)d.arrow(xs[i]+bw/2,y,lerp(xs[i]+bw/2,xs[i+1]-bw/2,p),y,C.muted);});
 d.text('Keep evaluation outcomes outside model selection',W/2,H*.89,z?12:17,C.muted);
}
function drPlot(d,p,kind){
 const a=effectAxes(d,{ymin:-3,ymax:12,ylabel:R`\widehat\phi`,yticks:[0,4,8,12]}),rows=drRows(false,true),bad=drRows(false,false),truth=drFit(false,true),wrong=drFit(false,false);
 if(kind==='dr-average'){
  dots(d,a,drRows(),o=>o.phi,.45);curve(d,a,x=>lerp(4,metaTau(x),p),C.teal);d.math(R`\overline\phi=4`,d.W*.68,20,180,36,d.fs,C.teal);
 }else{
  dots(d,a,rows,(o)=>kind==='dr-pseudo'?lerp(o.base,o.phi,p):kind==='dr-both-wrong'?lerp(o.phi,bad[o.id].phi,p):o.phi,.5);
  curve(d,a,metaTau,C.muted,1,'5 4');
  const f=kind==='dr-pseudo'?x=>lerp(metaTau(x)-3,metaTau(x),p):kind==='dr-restrict'?x=>lerp(truth.predict(x),truth.average,p):kind==='dr-both-wrong'?x=>lerp(truth.predict(x),wrong.predict(x),p):truth.predict;
  curve(d,a,f,kind==='dr-both-wrong'?C.rust:C.teal,kind==='dr-fit'?p:1);
  d.math(kind==='dr-both-wrong'?R`\overline\phi=4\ \text{still}`:kind==='dr-restrict'?R`f(x)\longrightarrow4`:R`\mathbb E[\widehat\phi\mid X]`,d.W*.65,20,230,38,d.fs,kind==='dr-both-wrong'?C.rust:C.teal);
 }
}
function drOne(d,p,overlap=false){
 const {W,H,mobile:z}=d,e=overlap?lerp(.25,.03,p):.25,val=3+2/e,left=35,right=W-35,max=overlap?80:14,x=v=>left+(right-left)*v/max,y=H*.68;
 d.math(overlap?R`\widehat e=`+fmt(e):R`3+4(27-25)=11`,W/2,H*.20,W-12,55,z?25:35,C.blue);
 d.math(overlap?R`\widehat\phi=`+fmt(val):R`\text{prediction}+\text{correction}`,W/2,H*.41,W-10,45,z?19:27,C.teal);
 d.line(left,y,right,y,C.ink,1);(overlap?[0,20,40,60,80]:[0,3,6,9,12]).forEach(v=>d.text(v,x(v),y+23,d.small,C.muted));
 d.circle(x(3),y,5,C.white,C.muted);d.arrow(x(3),y-17,x(overlap?val:lerp(3,11,p)),y-17,C.blue);d.circle(x(overlap?val:lerp(3,11,p)),y,7,C.blue,C.blue);
 d.text(overlap?'One fixed residual, changing probability':'A pseudo-outcome is not an individual effect',W/2,H-16,z?12:17,C.muted);
}
function drRobust(d,p){
 const {W,H,mobile:z}=d,xs=[W*.20,W*.56,W*.86],ys=[H*.32,H*.53,H*.74],cases=[[true,true],[false,true],[true,false]],active=Math.min(2,Math.floor(p*3));
 d.text('Outcome means',xs[0],28,d.small);d.text('Propensity',xs[1],28,d.small);d.math(R`\mathbb E[\widehat\phi\mid X=0]`,W/2,H-23,W-15,40,d.fs,C.teal);
 cases.forEach(([a,b],i)=>{const y=ys[i],on=i<=active;if(i===active)d.rect(5,y-22,W-10,44,C.pale,'none');d.text(a?'Correct':'Incorrect',xs[0],y,d.fs,a?C.blue:C.rust,'middle',on?1:.2);d.text(b?'Correct':'Incorrect',xs[1],y,d.fs,b?C.blue:C.rust,'middle',on?1:.2);const v=mean(drRows(a,b).filter(o=>o.x===0).map(o=>o.phi));d.math(on?fmt(v):'\\cdots',xs[2],y,70,40,d.fs+6,C.teal,on?1:.2);});
}
function rMean(d,p){
 const a=outcomeAxes(d);outcomeCurves(d,a,1,false);dots(d,a,metaRows,o=>o.y,.25);
 curve(d,a,x=>lerp(metaMu(0,x),metaEll(x),p),C.ink);
 d.math(R`\mu_0`,d.W*.33,20,90,35,d.fs,C.rust);d.math(R`\ell`,d.W*.56,20,90,35,d.fs+3,C.ink);d.math(R`\mu_1`,d.W*.79,20,90,35,d.fs,C.blue);
}
function residualPanels(d,p,slopes=false){
 const {W,H,mobile:z}=d,width=W/3,progress=slopes?1:p;
 [0,.5,1].forEach((x0,j)=>{
  const left=j*width+(z?27:38),right=(j+1)*width-12,top=75,bottom=H-62,xmin=lerp(-.1,-1,progress),xmax=lerp(1.1,1,progress),ymin=lerp(18,-6,progress),ymax=lerp(36,6,progress);
  const a={x:x=>left+(right-left)*(x-xmin)/(xmax-xmin),y:y=>bottom-(bottom-top)*(y-ymin)/(ymax-ymin)};
  d.math('X='+x0,(left+right)/2,19,width-5,30,d.fs);
  d.math(progress===0?'Y':R`\widetilde Y`,left,53,45,28,z?13:18,C.muted);
  if(slopes)d.math(R`\tau=`+metaTau(x0),(left+right)/2,49,width-5,28,d.fs,C.teal);
  d.line(left,top,left,bottom,C.line,1);d.line(left,bottom,right,bottom,C.ink,1);
  [xmin,xmax].forEach(v=>d.text(fmt(v),a.x(v),bottom+17,z?13:15,C.muted));[ymin,ymax].forEach(v=>d.text(fmt(v),left-5,a.y(v),z?13:15,C.muted,'end'));
  d.math(progress===0?'D':R`\widetilde D`,(left+right)/2,H-15,90,30,d.small);
  if(progress===1)d.line(left,a.y(0),right,a.y(0),C.line,1,'3 3');
  metaRows.filter(o=>o.x===x0).forEach(o=>d.mark(o,a.x(o.d-progress*metaE(o.x)),a.y(o.y-progress*metaEll(o.x)),z?2.8:4,.6));
  if(slopes){const lo=-.85,hi=lerp(lo,.85,p);d.line(a.x(lo),a.y(lo*metaTau(x0)),a.x(hi),a.y(hi*metaTau(x0)),C.teal,2.5);}
 });
}
function rLossPlot(d,p,overlap=false){
 const e=overlap?lerp(.5,.03,p):.5,a=d.axes({xmin:overlap?-4:-3.2,xmax:overlap?4:3.2,ymin:0,ymax:overlap?4.3:3.4,xlabel:overlap?R`f(x)-\tau(x)`:'c',ylabel:overlap?R`\Delta L_R`:R`L_R`,xticks:overlap?[-4,0,4]:[-3,0,3],yticks:overlap?[0,2,4]:[0,1,2,3],bottom:d.mobile?70:65});
 if(overlap){curve(d,a,x=>.25*x*x,C.muted,1,'5 4',-4,4);curve(d,a,x=>rInformation(e)*x*x,C.teal,1,'',-4,4);d.math('e='+fmt(e),d.W*.65,20,170,36,d.fs,C.teal);}
 else{curve(d,a,c=>rLoss(x=>metaTau(x)+c),C.blue,1,'',-3.2,3.2);const c=2.5*(1-p),loss=rLoss(x=>metaTau(x)+c);d.line(a.x(c),a.y(1),a.x(c),a.y(loss),C.teal,2,'4 4');d.circle(a.x(c),a.y(loss),7,C.teal,C.teal);d.math('c='+fmt(c)+R`,\ L_R=`+fmt(loss),d.W*.62,20,250,36,d.fs,C.teal);}
}
function rQuotient(d,p){
 const {W,H,mobile:z}=d,e=lerp(.25,.02,p),quotient=6-1/e,weight=e*e,left=35,right=W-30,x=v=>left+(right-left)*(v+50)/60,y=H*.42;
 d.math(R`\widetilde Y/\widetilde D=`+fmt(quotient),W/2,H*.16,W-15,45,z?23:31,C.rust);
 d.line(left,y,right,y,C.ink,1);[-50,-25,0,10].forEach(v=>d.text(v,x(v),y+19,d.small,C.muted));d.circle(x(quotient),y,6,C.rust,C.rust);
 const by=H*.70;d.math(R`\widetilde D^2=`+fmt(weight,4),W*.35,by-26,W*.65,38,d.fs,C.blue);d.line(left,by,right,by,C.line,15);d.line(left,by,left+(right-left)*weight/.0625,by,C.blue,15);
 d.math(R`(\widetilde Y-6\widetilde D)^2=1`,W/2,H-24,W-12,40,z?18:25,C.teal);
}
function rOrthogonal(d,p){
 const a=d.axes({xmin:-.2,xmax:.2,ymin:5,ymax:6.1,xlabel:'t',ylabel:R`\overline\tau(0)`,xticks:[-.2,0,.2],yticks:[5,5.5,6],bottom:d.mobile?70:65});
 curve(d,a,t=>rLocal(0,t,t),C.blue,1,'',-.2,.2);d.line(a.x(-.2),a.y(6),a.x(.2),a.y(6),C.teal,1.5,'5 4');const t=.2*(1-p);d.circle(a.x(t),a.y(rLocal(0,t,t)),7,C.blue,C.blue);d.math('a=b=t',d.W*.65,20,180,36,d.fs,C.blue);
}
function rNotDR(d,p){
 const {W,H,mobile:z}=d,left=z?165:270,right=W-35,x=v=>left+(right-left)*(v-5.5)/.7,cases=[[0,0],[.1,0],[0,.1],[.1,.1]];
 d.line(x(6),35,x(6),H-30,C.teal,1.5,'4 4');d.math(R`\overline\tau(0)`,W*.73,15,130,30,d.fs,C.teal);
 cases.forEach(([a,b],i)=>{const y=H*(.20+.20*i),v=lerp(6,rLocal(0,a,b),p);d.math('a='+a+',\\ b='+b,left/2,y,left-12,40,d.fs);d.circle(x(v),y,6,b?C.rust:C.blue,b?C.rust:C.blue);d.text(fmt(v),x(v)+13,y,d.fs,b?C.rust:C.blue,'start');});
}
function rEvaluate(d,p){
 const {W,H,mobile:z}=d,left=z?116:195,right=W-48,x=v=>left+(right-left)*v/5.5,candidates=[metaTau,()=>4,()=>0],labels=[R`f=\tau`,'f=4','f=0'];
 candidates.forEach((fn,i)=>{const y=H*(.23+.25*i),v=rLoss(fn);d.math(labels[i],left/2,y,left-10,45,d.fs+2);d.line(left,y,x(1),y,C.blue,z?16:22);d.line(x(1),y,x(lerp(1,v,p)),y,i?C.rust:C.blue,z?16:22);d.text(fmt(v),x(v)+10,y,d.fs,i?C.rust:C.blue,'start');});
 d.line(left,H-42,right,H-42,C.ink,1);[0,1,3,5].forEach(v=>d.text(v,x(v),H-22,d.small,C.muted));d.text('Residual loss with known nuisance functions',W/2,17,d.small,C.muted);
}
export function drawMeta(d,kind,p){
 switch(kind){
  case 'meta-target':metaTarget(d,p);break;
  case 'meta-assignment':assignment(d,p);break;
  case 'meta-s-fit':sFit(d,p);break;
  case 'meta-s-contrast':metaTarget(d,p,true);break;
  case 'meta-s-additive':sEffect(d,p);break;
  case 'meta-s-shrink':sEffect(d,p,true);break;
  case 'meta-t-fit':twoPanels(d,'outcome',p);break;
  case 'meta-t-difference':tDifference(d,p);break;
  case 'meta-t-errors':tErrors(d,p);break;
  case 'meta-x-treated':xImpute(d,p,1);break;
  case 'meta-x-control':xImpute(d,p,0);break;
  case 'meta-x-fit':twoPanels(d,'effect',p);break;
  case 'meta-x-blend':xWeights(d,p);break;
  case 'meta-x-imbalance':xImbalance(d,p);break;
  case 'meta-structure':structure(d,p);break;
  case 'meta-evaluation':evaluation(d,p);break;
  case 'dr-average':case 'dr-pseudo':case 'dr-fit':case 'dr-restrict':case 'dr-both-wrong':drPlot(d,p,kind);break;
  case 'dr-one':drOne(d,p);break;
  case 'dr-overlap':drOne(d,p,true);break;
  case 'dr-robust':drRobust(d,p);break;
  case 'r-mean':rMean(d,p);break;
  case 'r-residuals':residualPanels(d,p);break;
  case 'r-slopes':residualPanels(d,p,true);break;
  case 'r-loss':rLossPlot(d,p);break;
  case 'r-quotient':rQuotient(d,p);break;
  case 'r-overlap':rLossPlot(d,p,true);break;
  case 'r-orthogonal':rOrthogonal(d,p);break;
  case 'r-not-dr':rNotDR(d,p);break;
  case 'r-evaluate':rEvaluate(d,p);break;
  default:throw new Error('Unknown meta-learner scene '+kind);
 }
}
