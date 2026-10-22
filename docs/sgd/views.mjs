import {foundationKinds,renderFoundation} from './batch-epoch.mjs';
import {constructionKinds,renderConstruction} from './construction.mjs';
import {predict,ramp,gradient,data,loss} from './model.mjs';
import {sgdKinds,renderSGD,C,clamp,line,text,set,canvas,chart,facts,observations,gradientBars,interpolate} from './sgd-detail.mjs';
function trainingPanel(svg,trace,index,mix,firstEpoch,color,width){
 const {w,h}=canvas(svg),s=trace[index],value=loss(interpolate(s.model,s.after,mix));
 const compact=h<150,l=37,r=w-12,t=compact?57:87,b=h-24,maxStep=firstEpoch?32:9600;
 const X=step=>l+step/maxStep*(r-l),Y=v=>b-(Math.log10(Math.max(.001,v))+3)/3*(b-t);
 let html=text(0,13,`Epoch ${s.epoch} / 300`,'small')+text(w,13,`Batch ${s.batch} / 32 · 64 rows`,'small','end');
 const slot=(w-2)/32;
 for(let j=0;j<32;j++)html+=`<rect x="${j*slot}" y="23" width="${Math.max(1,slot-2)}" height="${j===s.batch-1?12:7}" fill="${j===s.batch-1?C.rust:j<s.batch-1?color:'#e8e8e8'}"/>`;
 if(!compact)html+=text(0,57,mix<.25?'Select batch → calculate gradient':'Apply gradient → update parameters','small');
 for(const v of [.01,.1,1])html+=line(l,Y(v),r,Y(v),'#eee')+text(l-5,Y(v)+3,String(v),'small','end');
 let path='';for(let i=0;i<=index;i++)path+=(i?'L':'M')+X(trace[i].step)+','+Y(trace[i].loss);
 path+=`L${X(s.step+mix)},${Y(value)}`;
 html+=`<path d="${path}" stroke="${color}" fill="none" stroke-width="2"/><circle cx="${X(s.step+mix)}" cy="${Y(value)}" r="3" fill="${color}"/>`+line(l,b,r,b,'#767676')+text(l,b+18,'0','small')+text(r,b+18,`${maxStep.toLocaleString('en-US')} updates`,'small','end');
 set(svg,html,`Width ${width}. Epoch ${s.epoch}, batch ${s.batch} of 32, 64 observations. Training MSE ${value.toFixed(4)}; logarithmic loss axis.`);
}
export function render(plot,detail,scene,p,e){
 const u=clamp((p-.12)/.7),kind=scene.kind;
 if(constructionKinds.includes(kind))return renderConstruction(plot,detail,kind,p);
 if(foundationKinds.includes(kind))return renderFoundation(plot,detail,scene,p,e);
 if(sgdKinds.includes(kind))return renderSGD(plot,detail,scene,p,e);
 const c=chart(plot,{xmin:-1,xmax:1});
 let body=c.base+c.step(),detailTitle='Construction',plotTitle='Function',info={kind,progress:p};
 if(kind==='steepen'){
  const a=2*16**u;body+=c.path(x=>ramp(x,a),C.teal);
  body+=line(c.X(0),c.Y(-.12),c.X(1/a),c.Y(-.12),C.rust,'',2);
  facts(detail,[['Steepness a',a.toFixed(1),'Width stays at 2 neurons'],['Ramp width',(1/a).toFixed(3),'The second hinge is at 1/a']]);info.a=a;
 }
 if(['initialize','loss','full','update','initialize-wide'].includes(kind)){
  const wide=kind==='initialize-wide',color=wide?C.teal:C.blue;
  let model=wide?e.traces[12][0].model:e.firstUpdate.before;detailTitle='Training';
  if(kind==='update'){model=interpolate(e.firstUpdate.before,e.firstUpdate.after,u);body+=c.path(x=>predict(e.firstUpdate.before,x),C.gray,2,'4 4');}
  body+=observations(c,{batch:kind==='update'?e.firstUpdate.indices:null,residualModel:kind==='loss'?model:null,amount:kind==='loss'?u:1})+c.path(x=>predict(model,x),color);
  if(kind==='initialize'||wide)facts(detail,[['Hidden neurons',wide?'12':'2','One hidden layer'],['Trainable parameters',wide?'37':'7','Weights and biases are trainable']]);
  if(kind==='loss')facts(detail,[['Training MSE',e.traces[2][0].loss.toFixed(4),'All 2,048 observations'],['Residuals shown',String(Math.ceil(2048*u)),'Squared, then averaged']]);
  if(kind==='full'){gradientBars(detail,gradient(model,data));detailTitle='Backpropagated gradient';}
  if(kind==='update')facts(detail,[['Learning rate','0.025','Zero gradient: no change'],['Batch MSE',loss(model,e.firstUpdate.indices.map(i=>data[i])).toFixed(4),'One actual SGD update']]);
  info.width=wide?12:2;
 }
 if(['batches-two','train-two','batches-wide','train-wide'].includes(kind)){
  const width=kind.endsWith('wide')?12:2,color=width===2?C.blue:C.teal,trace=e.traces[width],firstEpoch=kind.startsWith('batches');
  const start=firstEpoch?0:31,end=firstEpoch?31:trace.length-1;
  const position=u*(end-start+1),index=Math.min(end,start+Math.floor(position)),fraction=u===1?1:position-Math.floor(position),mix=clamp((fraction-.25)/.65),s=trace[index];
  const model=interpolate(s.model,s.after,mix);
  body+=observations(c,{batch:s.indices})+c.path(x=>predict(s.model,x),C.gray,1.8,'3 4')+c.path(x=>predict(model,x),color);
  trainingPanel(detail,trace,index,mix,firstEpoch,color,width);
  plotTitle=`Width ${width} · update ${(s.step+(mix===1?1:0)).toLocaleString('en-US')} / 9,600`;
  detailTitle=`Full-data MSE · ${loss(model).toFixed(4)} (log scale)`;
  info={...info,width,step:s.step,epoch:s.epoch,batch:s.batch,mix,batchFirst:s.indices[0],predictionAtZero:predict(model,0)};
 }
 if(kind==='compare'||kind==='finish'){
  const narrow=e.runs.find(r=>r.width===2&&r.seed===1),wide=e.runs.find(r=>r.width===12&&r.seed===1);
  body+=observations(c)+c.path(x=>predict(narrow.model,x),C.blue)+c.path(x=>predict(wide.model,x),C.teal,3,'10 4');
  if(kind==='finish')body+=c.path(x=>ramp(x,64),C.rust,2.5,'2 4');
  facts(detail,[['Width 2 · MSE',narrow.loss.toFixed(4),'7 parameters'],['Width 12 · MSE',wide.loss.toFixed(4),'37 parameters']]);
  plotTitle='Predictions after 300 epochs';detailTitle='One hidden layer · same data and update count';
 }
 set(plot,body,`${scene.title} ${scene.caption}`);return {...info,detailTitle,plotTitle};
}
