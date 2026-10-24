import {foundationKinds,renderFoundation} from './batch-epoch.mjs';
import {constructionKinds,renderConstruction} from './construction.mjs';
import {renderInitialNetwork} from './network.mjs';
import {renderParameterProbe} from './parameter-probe.mjs';
import {renderStepTarget} from './step-target.mjs';
import {renderTwoHinges} from './two-hinges.mjs';
import {predict,ramp,gradient,data,mse} from './model.mjs';
import {sgdKinds,renderSGD,renderParameterUpdate,C,clamp,line,text,set,canvas,chart,facts,observations,gradientBars,interpolate} from './sgd-detail.mjs';
function trainingPanel(svg,trace,index,mix,firstEpoch,color,width){
 const {w,h}=canvas(svg),s=trace[index],value=mse(interpolate(s.model,s.after,mix));
 const projected=globalThis.innerWidth>=1000,compact=h<150,l=projected?48:37,r=w-12,t=compact?42:projected?100:87,b=h-(projected?32:24),maxStep=firstEpoch?32:9600;
 const X=step=>l+step/maxStep*(r-l),Y=v=>b-(Math.log10(Math.max(.001,v))+3)/3*(b-t);
 let html=text(0,projected?20:13,`Batch ${s.batch} / 32 · 64 observations`,'small');
 const slot=(w-2)/32;
 for(let j=0;j<32;j++)html+=`<rect x="${j*slot}" y="${projected?32:23}" width="${Math.max(1,slot-2)}" height="${j===s.batch-1?12:7}" fill="${j===s.batch-1?C.rust:j<s.batch-1?color:'#e8e8e8'}"/>`;
 if(!compact)html+=text(0,projected?70:57,mix<.25?'Batch → gradient':'Update all parameters','small');
 for(const v of (compact?[.01,1]:[.01,.1,1]))html+=line(l,Y(v),r,Y(v),'#eee')+text(l-5,Y(v)+3,String(v),'small','end');
 let path='';for(let i=0;i<=index;i++)path+=(i?'L':'M')+X(trace[i].step)+','+Y(trace[i].loss);
 path+=`L${X(s.step+mix)},${Y(value)}`;
 html+=`<path d="${path}" stroke="${color}" fill="none" stroke-width="2"/><circle cx="${X(s.step+mix)}" cy="${Y(value)}" r="3" fill="${color}"/>`+line(l,b,r,b,'#767676')+text(l,b+18,'0','small')+text(r,b+18,`${maxStep.toLocaleString('en-US')} updates`,'small','end');
 set(svg,html,`Width ${width}. Epoch ${s.epoch}, batch ${s.batch} of 32, 64 observations. Training MSE ${value.toFixed(4)}; logarithmic loss axis.`);
}
export function render(plot,detail,scene,p,e){
 const u=clamp((p-.12)/.7),kind=scene.kind;
 if(kind==='step-target')return renderStepTarget(plot,detail,scene,p);
 if(kind==='two-hinges')return renderTwoHinges(plot,detail,scene,p);
 if(constructionKinds.includes(kind))return renderConstruction(plot,detail,kind,p);
 if(foundationKinds.includes(kind))return renderFoundation(plot,detail,scene,p,e);
 if(sgdKinds.includes(kind))return renderSGD(plot,detail,scene,p,e);
 if(kind==='parameter-probe')return renderParameterProbe(plot,detail,scene,p,e);
 const c=chart(plot,{xmin:-1,xmax:1});
 let body=c.base+c.step(),detailTitle='Construction',plotTitle='Function',info={kind,progress:p};
 if(kind==='steepen'){
  const a=2*16**u;body+=c.path(x=>ramp(x,a),C.teal);
  body+=line(c.X(0),c.Y(-.12),c.X(1/a),c.Y(-.12),C.rust,'',2);
  facts(detail,[['Input slope k',a.toFixed(1),'Width stays at 2 neurons'],['Ramp width',(1/a).toFixed(3),'The second hinge is at 1/k']]);info.a=a;
 }
 if(['initialize','loss','full','update','initialize-wide'].includes(kind)){
  const wide=kind==='initialize-wide',color=wide?C.teal:C.blue;
  let model=wide?e.traces[12][0].model:e.firstUpdate.before;detailTitle='Training';
  if(kind==='update'){model=interpolate(e.firstUpdate.before,e.firstUpdate.after,u);body+=c.path(x=>predict(e.firstUpdate.before,x),C.gray,2,'4 4');}
  body+=observations(c,{batch:kind==='update'?e.firstUpdate.indices:null,residualModel:kind==='loss'?model:null,amount:kind==='loss'?u:1})+c.path(x=>predict(model,x),color);
  if(kind==='initialize'){info.displayedParameters=renderInitialNetwork(detail,model);detailTitle='Initial network · weights and biases';}
  if(wide)facts(detail,[['Hidden neurons','12','One hidden layer'],['Trainable parameters','37','Learn weights and biases']]);
  if(kind==='loss')facts(detail,[['Training MSE',e.traces[2][0].loss.toFixed(4),'All 2,048 observations'],['Residuals shown',String(Math.ceil(2048*u)),'Squared, then averaged']]);
  if(kind==='full'){gradientBars(detail,gradient(model,data));detailTitle='Full-data gradient';}
  if(kind==='update'){
   info.displayedParameters=renderParameterUpdate(detail,e.firstUpdate.before,e.firstUpdate.after,e.firstUpdate.gradient,u);
   info.updateProgress=u;
   plotTitle='Dashed: before · blue: current model';detailTitle='All seven parameters update together';
  }
  info.width=wide?12:2;info.completedUpdates=kind==='update'?u:0;info.outputBias=model.c;info.predictionAtZero=predict(model,0);info.model=model;
 }
 if(['batches-two','train-two','batches-wide','train-wide'].includes(kind)){
  const width=kind.endsWith('wide')?12:2,color=width===2?C.blue:C.teal,trace=e.traces[width],firstEpoch=kind.startsWith('batches');
  const firstStep=firstEpoch?(width===2?2:0):(width===2?33:32);
  const start=trace.findIndex(s=>s.step===firstStep),end=firstEpoch?31:trace.length-1;
  const position=(kind==='batches-two'?u**1.5:u)*(end-start+1),index=Math.min(end,start+Math.floor(position)),fraction=u===1?1:position-Math.floor(position),mix=clamp((fraction-.25)/.65),s=trace[index];
  const model=interpolate(s.model,s.after,mix);
  body+=observations(c,{batch:s.indices})+c.path(x=>predict(s.model,x),C.gray,1.8,'3 4')+c.path(x=>predict(model,x),color);
  trainingPanel(detail,trace,index,mix,firstEpoch,color,width);
  plotTitle=`${width} neurons · current batch in rust`;
  detailTitle=`Full-data MSE · ${mse(model).toFixed(4)} (log scale)`;
  info={...info,width,step:s.step,epoch:s.epoch,batch:s.batch,mix,batchFirst:s.indices[0],predictionAtZero:predict(model,0),model,completedUpdates:s.step+mix};
 }
 if(kind==='epoch-two'){
  const trace=e.traces[2],s=trace.find(s=>s.step===32),previous=trace.find(s=>s.step===31),select=clamp(u/.45),mix=clamp((u-.5)/.5),model=interpolate(s.model,s.after,mix);
  body+=observations(c)+c.path(x=>predict(s.model,x),C.gray,1.8,'3 4')+c.path(x=>predict(model,x),C.blue);
  for(const [indices,alpha] of [[previous.indices,1-select],[s.indices,select]]){
   body+=`<g opacity="${alpha}">`;
   for(const i of indices){const d=data[i];body+=`<circle cx="${c.X(d.x)}" cy="${c.Y(d.y)}" r="3.3" fill="${C.rust}" stroke="white" stroke-width=".6"/>`;}
   body+='</g>';
  }
  trainingPanel(detail,trace,32,mix,false,C.blue,2);
  plotTitle=select<1?'Same learned model · reshuffle the observations':'Epoch 2 · first batch in the new partition';
  detailTitle=mix===0?'32 updates complete · start a new pass':'First update of epoch 2';
  info={...info,width:2,step:32,epoch:2,batch:1,mix,batchFirst:s.indices[0],model,completedUpdates:32+mix,predictionAtZero:predict(model,0)};
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
