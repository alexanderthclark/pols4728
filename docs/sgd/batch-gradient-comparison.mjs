import {data,gradient,predict} from './model.mjs';
import {mathText,richText} from './math.mjs';
import {C,canvas,chart,clamp,line,text,set,observations} from './sgd-detail.mjs';

const labels=['\\theta_{11}','\\theta_{10}','\\phi_1','\\theta_{21}','\\theta_{20}','\\phi_2','\\phi_0'];
const flatten=g=>g.units.flatMap(u=>[u.w,u.b,u.a]).concat(g.c);
const smooth=t=>{t=clamp(t);return t*t*(3-2*t);};
const cache=new WeakMap();
function comparison(e){
 if(!cache.has(e)){
  const model=e.firstUpdate.before,batches=e.traces[2].slice(0,32).map(s=>s.indices);
  // Every gradient is recomputed at this one model, including later batches.
  // Recorded later gradients would confound the batch with intervening updates.
  const all=batches.map(indices=>flatten(gradient(model,indices.map(i=>data[i]))));
  cache.set(e,{model,batches,all,gradients:{batch1:all[0],batch2:all[1],full:flatten(gradient(model,data))}});
 }
 return cache.get(e);
}
const dot=(x,y,r,fill,stroke=C.rust,width=1.5)=>`<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${width}"/>`;

function gradientPlot(svg,values,batchReveal,spreadReveal){
 const {w,h}=canvas(svg),large=globalThis.innerWidth>=1000;
 const left=large?65:40,right=w-(large?18:10),keyY=large?24:14;
 const top=large?100:50,bottom=h-(large?52:30),gap=(bottom-top)/7;
 const allValues=values.all.flat().concat(values.gradients.full),lo=Math.floor(Math.min(0,...allValues)*10)/10,hi=Math.ceil(Math.max(0,...allValues)*10)/10;
 const pad=(hi-lo)*.03,X=v=>left+(v-lo+pad)/(hi-lo+2*pad)*(right-left);
 const key=(x,label,source)=>{
  const marker=source==='full'?line(x,keyY-6,x,keyY+6,C.blue,'',3):dot(x,keyY,large?4:3,source==='batch1'?C.rust:'white');
  return marker+richText(x+(large?13:9),keyY+4,label,'small','start',source==='full'?C.blue:C.rust);
 };
 let body=key(left,String.raw`\(B_1\)`,'batch1')+key(left+(right-left)*.24,String.raw`\(B_2\)`,'batch2')+key(left+(right-left)*.48,'Full data','full');
 body+=text(left,large?59:39,!large&&spreadReveal>.5?'Weights fixed · faint: 30 other batches':'Same weights · no update','small');
 if(large)body+=`<g opacity="${spreadReveal}">`+text(right,83,'Faint dots: other 30 batches','small','end')+'</g>';
 body+=line(X(0),top-7,X(0),bottom+3,'#bbb','3 4');
 for(let i=0;i<7;i++){
  const y=top+(i+.5)*gap,offset=Math.min(large?6:3,gap*.19),radius=Math.min(large?4.5:3.2,gap*.18),tickHeight=Math.min(large?11:7,gap*.4);
  body+=line(left,y,right,y,'#ececec')+mathText(left-(large?14:9),y+4,labels[i],'small','end',C.ink);
  body+=`<g data-gradient-spread="${i}" opacity="${spreadReveal*.35}">`;
  values.all.slice(2).forEach((g,j)=>{body+=dot(X(g[i]),y+((j%3)-1)*Math.min(3,gap*.08),large?2.5:1.7,C.rust,C.rust,.5);});
  body+='</g>';
  const marker=(source,value,content,opacity=1)=>`<g data-gradient-source="${source}" data-parameter="${labels[i]}" data-gradient="${value}" opacity="${opacity}">${content}</g>`;
  body+=marker('full',values.gradients.full[i],line(X(values.gradients.full[i]),y-tickHeight,X(values.gradients.full[i]),y+tickHeight,C.blue,'',2.5));
  body+=marker('batch1',values.gradients.batch1[i],dot(X(values.gradients.batch1[i]),y-offset,radius,C.rust));
  body+=marker('batch2',values.gradients.batch2[i],dot(X(values.gradients.batch2[i]),y+offset,radius,'white'),batchReveal);
 }
 body+=line(left,bottom+3,right,bottom+3,'#999');
 for(let tick=Math.round(lo*10);tick<=Math.round(hi*10);tick++){
  const v=tick/10,x=X(v);body+=line(x,bottom+3,x,bottom+7,'#999')+text(x,bottom+(large?28:20),v===0?'0':v.toFixed(1),'small','middle');
 }
 if(large)body+=text((left+right)/2,h-3,'Gradient component','small','middle');
 set(svg,body,'Seven gradient components at the same initial parameters. Filled rust circles: batch 1. Open rust circles: batch 2. Blue ticks: all 2,048 observations. Faint rust dots reveal the other 30 batch gradients, all computed at these same weights. Every row uses the same horizontal scale. The first three components remain zero. The remaining components vary with the batch. No parameters have changed and no update has occurred.');
}

function fixedFit(svg,values,batchReveal){
 const large=globalThis.innerWidth>=1000,c=chart(svg,{xmin:-1,xmax:1,topInset:large?48:20,yticks:large?[0,.5,1]:[0,1]});
 let body=c.base+c.step()+observations(c)+c.path(x=>predict(values.model,x),C.blue).replace('<path ','<path data-curve="fixed-model" ');
 body+=text(c.left,large?22:12,'Predictions stay fixed','small');
 const shown=batchReveal<.5?1:2;
 body+=richText(c.left,large?48:30,`Selected: \\(B_${shown}\\) · 64 observations`,'small','start',C.rust);
 values.batches.slice(0,2).forEach((indices,j)=>{
  const alpha=j?batchReveal:1-batchReveal;
  body+=`<g data-selected-batch="${j+1}" opacity="${alpha}">`;
  for(const id of indices){const d=data[id];body+=`<g data-observation="${id}">`+dot(c.X(d.x),c.Y(d.y),large?4:2.7,j?'white':C.rust,C.rust,1.3)+'</g>';}
  body+='</g>';
 });
 set(svg,body,'The initial fitted curve stays fixed while the highlighted observations change from the first batch to the second batch. Both batches have 64 distinct observations from the same noiseless data. No weight or bias changes.');
}

export function renderGradientComparison(plot,detail,scene,p,e){
 const values=comparison(e),batchReveal=smooth((p-.18)/.32),spreadReveal=smooth((p-.6)/.25);
 gradientPlot(plot,values,batchReveal,spreadReveal);fixedFit(detail,values,batchReveal);
 return {kind:scene.kind,progress:p,model:values.model,gradients:values.gradients,batches:values.batches.slice(0,2),allBatchGradients:values.all,batchReveal,spreadReveal,completedUpdates:0,procedureStep:2,plotTitle:'Gradient components · a shared scale',detailTitle:'Same network · different observations'};
}
