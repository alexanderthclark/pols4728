import {mse,predict} from './model.mjs';
import {C,clamp,canvas,chart,line,observations,set} from './sgd-detail.mjs';

const labels=['θ₁₁','θ₁₀','φ₁','θ₂₁','θ₂₀','φ₂','φ₀'];
const parameterTex=[String.raw`\theta_{11}`,String.raw`\theta_{10}`,String.raw`\phi_1`,String.raw`\theta_{21}`,String.raw`\theta_{20}`,String.raw`\phi_2`,String.raw`\phi_0`];
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

// Use the lesson's bundled KaTeX for real mathematical glyphs and subscripts.
const mathCache=new Map();
function mathLabel(x,y,width,tex,color,size=14){
 if(!mathCache.has(tex))mathCache.set(tex,globalThis.katex.renderToString(tex,{displayMode:false,output:'htmlAndMathml',throwOnError:true,strict:'error'}));
 return `<foreignObject x="${x}" y="${y}" width="${width}" height="${size*1.7}" class="probe-math"><div xmlns="http://www.w3.org/1999/xhtml" style="font-size:${size}px;color:${color};line-height:1.2">${mathCache.get(tex)}</div></foreignObject>`;
}

function renderTable(svg,frame){
 const {w,h}=canvas(svg),compact=w<340||h<220,wide=globalThis.innerWidth>=1600,short=h<(wide?400:350),tableTop=compact?82:short?(wide?158:132):(wide?190:156);
 const rowHeight=Math.max(13,Math.min(compact?17:wide?52:44,(h-tableTop-10)/7));
 const x=7,minusX=w*.61,plusX=w-8,lossChange=frame.currentMSE-frame.baselineMSE;
 const label=(px,py,value,anchor='start',color=C.muted,size=compact?10:wide?22:18)=>
  `<text x="${px}" y="${py}" text-anchor="${anchor}" style="fill:${color};font-size:${size}px;font-variant-numeric:tabular-nums">${value}</text>`;
 const readoutSize=compact?10:wide?24:20,readoutTop=compact?14:wide?26:22,readoutGap=compact?16:wide?32:26;
 let body=label(x,readoutTop,`Baseline MSE: ${frame.baselineMSE.toFixed(8)}`,'start',C.blue,readoutSize);
 body+=label(x,readoutTop+readoutGap,`Current trial MSE: ${frame.currentMSE.toFixed(8)}`,'start',C.rust,readoutSize);
 body+=label(x,readoutTop+readoutGap*2,`Change in MSE: ${signed(lossChange,8)}`,'start',C.rust,readoutSize);
 if(!compact&&!short)body+=label(x,wide?134:110,'Fixed trials: resulting MSE','start',C.ink);
 body+=label(x,tableTop,'Parameter')+label(minusX,tableTop-(compact?14:wide?27:22),'MSE after','end')+label(plusX,tableTop-(compact?14:wide?27:22),'MSE after','end');
 body+=label(minusX,tableTop,'−0.02 nudge','end')+label(plusX,tableTop,'+0.02 nudge','end');
 body+=line(x,tableTop+5,w-5,tableTop+5,'#d9dfe2');
 frame.trialResults.forEach((result,index)=>{
  const y=tableTop+(index+1)*rowHeight,active=index===frame.parameterIndex;
  body+=`<g data-parameter="${result.parameterName}" data-minus-mse="${result.minus}" data-plus-mse="${result.plus}" data-active="${active}">`;
  if(active)body+=`<rect x="2" y="${y-rowHeight*.72}" width="${w-4}" height="${rowHeight-1}" rx="2" fill="#f6e9e3"/>`;
  body+=mathLabel(x,y-(compact?10:wide?22:18),w*.22,parameterTex[index],active?C.rust:C.ink,compact?11:wide?24:20);
  body+=label(minusX,y,result.minus.toFixed(8),'end')+label(plusX,y,result.plus.toFixed(8),'end')+'</g>';
 });
 set(svg,body,`Independent one-parameter trials after training. Baseline full-data MSE ${frame.baselineMSE.toFixed(8)}. Changing only ${frame.parameterName} by ${signed(frame.delta,4)} gives current trial MSE ${frame.currentMSE.toFixed(8)}, a change in MSE of ${signed(lossChange,8)} from baseline. The table contains separate fixed trials: the resulting MSE after subtracting 0.02 or adding 0.02 to each trained parameter, with all other parameters held fixed. Table entries are MSE values, not changes in MSE. `+frame.trialResults.map(row=>`${row.parameterName}: minus 0.02, MSE ${row.minus.toFixed(8)}; plus 0.02, MSE ${row.plus.toFixed(8)}.`).join(' '));
}

export function renderParameterProbe(plot,detail,scene,p,experiment){
 const frame=parameterProbeState(experiment,p),c=chart(plot,{xmin:-1,xmax:1});
 const trialTex=parameterTex[frame.parameterIndex]+String.raw`\mapsto `+parameterTex[frame.parameterIndex]+(frame.delta<0?'-':'+')+Math.abs(frame.delta).toFixed(4);
 const body=c.base+c.step()+observations(c)+c.path(x=>predict(frame.baselineModel,x),C.blue,3)+c.path(x=>predict(frame.model,x),C.rust,2,'7 5')+mathLabel(c.left,0,c.w-c.left-5,trialTex,C.rust,c.w<450?11:globalThis.innerWidth>=1600?26:22);
 set(plot,body,`${scene.title} The blue curve is the trained model after 300 epochs. The dashed rust curve changes only ${frame.parameterName} by ${signed(frame.delta,4)}, holding the other six parameters fixed. This is a what-if trial; training remains at 9,600 updates.`);
 renderTable(detail,frame);
 return {kind:scene.kind,progress:p,width:2,...frame,plotTitle:'Blue: trained fit · dashed rust: current trial',detailTitle:'Full-data MSE¹ · smaller is better'};
}
