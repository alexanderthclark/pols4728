import {mse,predict} from './model.mjs';
import {C,clamp,canvas,chart,line,observations,set,text} from './sgd-detail.mjs';

const labels=['θ₁₁','θ₁₀','φ₁','θ₂₁','θ₂₀','φ₂','φ₀'];
const keys=['w','b','a'];
const range=.02;
const cache=new WeakMap();
const copy=model=>({c:model.c,units:model.units.map(unit=>({...unit}))});
const value=(model,index)=>index===6?model.c:model.units[Math.floor(index/3)][keys[index%3]];
const signed=(v,d=3)=>(v<0?'−':'+')+Math.abs(v).toFixed(d);

function changed(baseline,index,delta){
 const model=copy(baseline);
 if(index===6)model.c+=delta;
 else model.units[Math.floor(index/3)][keys[index%3]]+=delta;
 return model;
}

function results(experiment){
 const source=experiment.traces[2].at(-1).after;
 if(!cache.has(source)){
  const baselineModel=copy(source),baselineMSE=mse(baselineModel);
  const trialResults=labels.map((parameterName,parameterIndex)=>({
   parameterIndex,parameterName,baselineValue:value(baselineModel,parameterIndex),
   minus:mse(changed(baselineModel,parameterIndex,-range)),
   plus:mse(changed(baselineModel,parameterIndex,range)),
  }));
  cache.set(source,{baselineModel,baselineMSE,trialResults});
 }
 return cache.get(source);
}

// Every trial starts from the same trained model. These are previews, not updates.
export function parameterProbeState(experiment,progress){
 const p=clamp(progress),position=p*labels.length,parameterIndex=Math.min(labels.length-1,Math.floor(position));
 const local=p===1?1:position-parameterIndex;
 const stops=[0,-range,0,range,0],phase=Math.min(3,Math.floor(local*4)),fraction=local===1?1:local*4-phase;
 const ease=(1-Math.cos(Math.PI*fraction))/2;
 const delta=stops[phase]+(stops[phase+1]-stops[phase])*ease;
 const saved=results(experiment),model=changed(saved.baselineModel,parameterIndex,delta);
 return {...saved,model,parameterIndex,parameterName:labels[parameterIndex],delta,currentMSE:mse(model),completedUpdates:9600};
}

function renderTable(svg,frame){
 const {w,h}=canvas(svg),compact=h<240,tableTop=compact?49:104;
 const rowHeight=Math.max(10,Math.min(compact?17:34,(h-tableTop-8)/7));
 const x=7,minusX=w*.61,plusX=w-8,lossChange=frame.currentMSE-frame.baselineMSE;
 let body=text(x,14,`Baseline MSE ${frame.baselineMSE.toFixed(8)}`,'small','start',C.blue);
 body+=text(x,31,`Trial ${frame.currentMSE.toFixed(8)} · Δ ${signed(lossChange,8)}`,'small','start',C.rust);
 if(!compact){
  body+=text(x,58,`Change only ${frame.parameterName} by ${signed(frame.delta,4)}`,'','start',C.rust);
  body+=text(x,78,`${value(frame.baselineModel,frame.parameterIndex).toFixed(4)} → ${value(frame.model,frame.parameterIndex).toFixed(4)} · all others fixed`,'small');
 }
 body+=text(x,tableTop,'Parameter','small')+text(minusX,tableTop,'−0.02','small','end')+text(plusX,tableTop,'+0.02','small','end');
 body+=line(x,tableTop+5,w-5,tableTop+5,'#d9dfe2');
 frame.trialResults.forEach((result,index)=>{
  const y=tableTop+(index+1)*rowHeight,active=index===frame.parameterIndex;
  body+=`<g data-parameter="${result.parameterName}" data-minus-mse="${result.minus}" data-plus-mse="${result.plus}" data-active="${active}">`;
  if(active)body+=`<rect x="2" y="${y-rowHeight*.67}" width="${w-4}" height="${rowHeight-1}" rx="2" fill="#f6e9e3"/>`;
  body+=text(x,y,result.parameterName,'small','start',active?C.rust:C.ink);
  body+=text(minusX,y,result.minus.toFixed(8),'small','end')+text(plusX,y,result.plus.toFixed(8),'small','end')+'</g>';
 });
 set(svg,body,`Independent one-parameter trials after training. Baseline full-data MSE ${frame.baselineMSE.toFixed(8)}. Changing only ${frame.parameterName} by ${signed(frame.delta,4)} gives MSE ${frame.currentMSE.toFixed(8)}. Each table row reports MSE after subtracting 0.02 or adding 0.02 to that parameter, with all other parameters held fixed. `+frame.trialResults.map(row=>`${row.parameterName}: minus 0.02, ${row.minus.toFixed(8)}; plus 0.02, ${row.plus.toFixed(8)}.`).join(' '));
}

export function renderParameterProbe(plot,detail,scene,p,experiment){
 const frame=parameterProbeState(experiment,p),c=chart(plot,{xmin:-1,xmax:1});
 const body=c.base+c.step()+observations(c)+c.path(x=>predict(frame.baselineModel,x),C.blue,3)+c.path(x=>predict(frame.model,x),C.rust,2,'7 5');
 set(plot,body,`${scene.title} The blue curve is the trained model after 300 epochs. The dashed rust curve changes only ${frame.parameterName} by ${signed(frame.delta,4)}, holding the other six parameters fixed. This is a what-if trial; training remains at 9,600 updates.`);
 renderTable(detail,frame);
 return {kind:scene.kind,progress:p,width:2,...frame,plotTitle:`Blue: trained fit · rust: ${frame.parameterName} ${signed(frame.delta,3)}`,detailTitle:'One-parameter trials · full-data MSE'};
}
