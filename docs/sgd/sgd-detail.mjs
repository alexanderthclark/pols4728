import {predict,gradient,data,loss} from './model.mjs';

// Shared geometry keeps the detailed SGD sequence on the same axes and palette.
export const C={blue:'#234e70',rust:'#a54f32',ink:'#242424',muted:'#606060',teal:'#007f73',gray:'#aaa'};
export const clamp=x=>Math.max(0,Math.min(1,x));
export const fmt=x=>Math.abs(x)<1e-10?'0':Number(x.toFixed(3)).toString();
export const line=(x1,y1,x2,y2,color=C.ink,dash='',width=1)=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}" ${dash?`stroke-dasharray="${dash}"`:''}/>`;
export const text=(x,y,t,cls='',anchor='start',color='')=>`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}" ${color?`style="fill:${color}"`:''}>${t}</text>`;
export function set(svg,body,label){svg.innerHTML=body;svg.setAttribute('aria-label',label);}
export function canvas(svg){const r=svg.getBoundingClientRect();const w=Math.max(260,r.width),h=Math.max(70,r.height);svg.setAttribute('viewBox',`0 0 ${w} ${h}`);return {w,h};}
export function chart(svg,{xmin=-1,xmax=1.2,ymin=-.25,ymax=1.4,xticks=[-1,0,1],yticks=[0,.5,1]}={}){
 const {w,h}=canvas(svg),left=37,right=w-17,top=15,bottom=h-27;
 const X=x=>left+(x-xmin)/(xmax-xmin)*(right-left),Y=y=>bottom-(y-ymin)/(ymax-ymin)*(bottom-top);
 let base=`<defs><clipPath id="clip-${svg.id}"><rect x="${left}" y="${top}" width="${right-left}" height="${bottom-top}"/></clipPath></defs>`;
 for(const y of yticks)base+=line(left,Y(y),right,Y(y),'#e7e7e7')+text(left-8,Y(y)+4,fmt(y),'small','end');
 base+=line(left,bottom,right,bottom,'#767676');
 for(const x of xticks)base+=line(X(x),bottom,X(x),bottom+4,'#767676')+text(X(x),bottom+19,fmt(x),'small','middle');
 base+=text(right+8,bottom+19,'x','small');
 const path=(fn,color=C.blue,width=3,dash='',opacity=1)=>{let d='';for(let i=0;i<=420;i++){const x=xmin+(xmax-xmin)*i/420;d+=(i?'L':'M')+X(x).toFixed(2)+','+Y(fn(x)).toFixed(2);}return `<path d="${d}" fill="none" stroke="${color}" stroke-width="${width}" stroke-opacity="${opacity}" ${dash?`stroke-dasharray="${dash}"`:''} clip-path="url(#clip-${svg.id})"/>`;};
 const step=()=>`<path d="M${X(xmin)},${Y(0)}H${X(0)}M${X(0)},${Y(1)}H${X(xmax)}" fill="none" stroke="${C.muted}" stroke-width="2" stroke-dasharray="6 5"/><circle cx="${X(0)}" cy="${Y(0)}" r="3" fill="${C.muted}"/><circle cx="${X(0)}" cy="${Y(1)}" r="3" fill="white" stroke="${C.muted}" stroke-width="1.5"/>`;
 return {w,h,X,Y,path,step,base,left,right,top,bottom,xmin,xmax};
}
export function facts(svg,entries){const {w,h}=canvas(svg);let html='';if(h<150){const cols=entries.length;entries.forEach((e,i)=>{const x=6+i*w/cols;html+=text(x,18,e[0],'small')+text(x,47,e[1],cols>2?'':'large')+(e[2]&&cols<3?text(x,68,e[2],'small'):'');});}else{const space=Math.min(100,(h-20)/entries.length);entries.forEach((e,i)=>{const y=24+i*space;html+=text(10,y,e[0],'small')+text(10,y+37,e[1],'large')+(e[2]?text(10,y+59,e[2],'small'):'');});}set(svg,html,entries.map(e=>e.join(': ')).join('. '));}
export function observations(c,{batch=null,residualModel=null,amount=1}={}){
 // Dense observations share one SVG path; each highlighted batch remains explicit.
 let dots='',residuals='',selected='';const active=new Set(batch||[]);
 data.forEach((d,i)=>{if(d.x<c.xmin||d.x>c.xmax)return;const x=c.X(d.x),y=c.Y(d.y);
 dots+=`M${x.toFixed(2)},${y.toFixed(2)}v.1`;
 if(residualModel&&i<data.length*amount&&(!batch||active.has(i)))residuals+=`M${x},${y}V${c.Y(predict(residualModel,d.x))}`;
 if(active.has(i))selected+=`<circle cx="${x}" cy="${y}" r="3.3" fill="${C.rust}" stroke="white" stroke-width=".6"/>`;
 });
 return `<path d="${dots}" fill="none" stroke="#b4b4b4" stroke-width="3" stroke-linecap="round"/><path d="${residuals}" fill="none" stroke="${C.rust}" stroke-opacity="${batch?.65:.10}" stroke-width="1"/>`+selected;
}
export function gradientBars(svg,g1,g2=null){const {w,h}=canvas(svg),vals=g1.units.flatMap(u=>[u.w,u.b,u.a]).concat(g1.c),other=g2?g2.units.flatMap(u=>[u.w,u.b,u.a]).concat(g2.c):null,labels=['w₁','b₁','a₁','w₂','b₂','a₂','c'];
 const top=18,bottom=h-24,mid=(top+bottom)/2,scale=(bottom-top)/2/Math.max(.02,...vals.map(Math.abs),...(other||[]).map(Math.abs)),col=(w-30)/7;
 let body=line(10,mid,w-10,mid,'#999');vals.forEach((v,i)=>{const x=20+col*i;body+=line(x,mid,x,mid-v*scale,C.blue,'',5);if(other)body+=line(x+7,mid,x+7,mid-other[i]*scale,C.rust,'',5);body+=text(x+2,bottom+19,labels[i],'small','middle');});
 body+=text(10,10,other?'Blue: full data · rust: minibatch':'Gradient components (signed)','small');set(svg,body,'Gradient components for the seven trainable parameters. '+(other?'Full gradient in blue, minibatch estimate in rust.':''));
}
export function interpolate(a,b,t){return {c:a.c+(b.c-a.c)*t,units:a.units.map((v,i)=>Object.fromEntries(['w','b','a'].map(k=>[k,v[k]+(b.units[i][k]-v[k])*t])))};}

export const sgdKinds=['sgd-transition','sgd-plan','shuffle','batch','batch-loss','batch-gradient','parameter-step','next-batch'];
const labels=['w₁','b₁','a₁','w₂','b₂','a₂','c'];
const flatten=g=>g.units.flatMap(v=>[v.w,v.b,v.a]).concat(g.c);
const signed=(v,d=4)=>(v<0?'−':v>0?'+':'')+Math.abs(v).toFixed(d);
const lerp=(a,b,t)=>a+(b-a)*t;
const smooth=t=>{t=clamp(t);return t*t*(3-2*t);};
const arrow=(x1,y1,x2,y2,color=C.muted,width=1.5)=>{
 const a=Math.atan2(y2-y1,x2-x1),s=5;
 return line(x1,y1,x2,y2,color,'',width)+`<path d="M${x2-s*Math.cos(a-.55)},${y2-s*Math.sin(a-.55)}L${x2},${y2}L${x2-s*Math.cos(a+.55)},${y2-s*Math.sin(a+.55)}" fill="none" stroke="${color}" stroke-width="${width}"/>`;
};
const circle=(x,y,r,fill,stroke='none',width=1)=>`<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${width}"/>`;

function batchFacts(svg,indices,selected=64){
 const {w,h}=canvas(svg),cols=16,dx=Math.min((w-16)/cols,24),dy=Math.min((h-27)/4,17),x0=(w-cols*dx)/2,y0=3;
 let body='';
 indices.forEach((id,i)=>{
  const x=x0+(i%cols+.5)*dx,y=y0+(Math.floor(i/cols)+.5)*dy;
  body+=circle(x,y,Math.max(1.5,Math.min(dx,dy)*.32),i<selected?C.rust:'#ddd');
 });
 body+=text(w/2,h-4,`${selected} of 64 selected · first row of the shuffled data`,'small','middle');
 set(svg,body,`${selected} of the first batch’s 64 observations selected. This is one of 32 disjoint batches.`);
}

function chapterDiagram(svg,e,u){
 const {w,h}=canvas(svg),col=w/3,cy=h*.49,sy=Math.min(50,h*.28),r=2,first=e.firstUpdate;
 let body='';
 // These are the actual 64 observations, not a schematic random sample.
 first.indices.forEach((id,i)=>{
  const d=data[id],orderedX=col*.13+(i%8)*col*.095,orderedY=cy-sy/2+Math.floor(i/8)*sy/7;
  const x=lerp(orderedX,col*.15+(d.x+1)*col*.32,smooth(u*1.8)),y=lerp(orderedY,cy+sy*.4-d.y*sy*.8,smooth(u*1.8));
  body+=circle(x,y,r,C.rust);
 });
 body+=text(col*.46,h-8,'64 observations','small','middle');
 const gs=flatten(first.gradient),scale=sy/Math.max(...gs.map(Math.abs));
 body+=line(col*1.12,cy,col*1.88,cy,'#bbb');
 gs.forEach((g,i)=>{const x=col*(1.17+.105*i);body+=line(x,cy,x,cy-g*scale*smooth(u*2-.4),g<0?C.rust:C.blue,'',4);});
 body+=text(col*1.5,h-8,'1 gradient','small','middle');
 const m=interpolate(first.before,first.after,smooth(u*2-1));
 let before='',after='';
 for(let i=0;i<=100;i++){
  const x=-1+2*i/100,px=col*(2.1+.77*i/100),Y=v=>cy+sy/2-v*sy;
  before+=(i?'L':'M')+px+','+Y(predict(first.before,x));
  after+=(i?'L':'M')+px+','+Y(predict(m,x));
 }
 body+=`<path d="${before}" fill="none" stroke="#aaa" stroke-width="2" stroke-dasharray="3 3"/><path d="${after}" fill="none" stroke="${C.teal}" stroke-width="3"/>`;
 body+=text(col*2.5,h-8,'1 update','small','middle');
 body+=arrow(col*.91,cy,col*1.06,cy)+arrow(col*1.92,cy,col*2.05,cy);
 set(svg,body,'One actual batch of 64 observations produces a seven-component gradient, which changes the model in one SGD update.');
}

function planDiagram(svg,e,active){
 const {w,h}=canvas(svg),top=26,bottom=h-20,left=3,gridW=w*.56,gridH=bottom-top,dx=gridW/64,dy=gridH/32;
 let body=text(left,15,'2,048 observations','small');
 const order=e.traces[2].slice(0,32).flatMap(s=>s.indices);
 order.forEach((id,i)=>{
  const selected=i<64&&active>0;
  body+=`<rect x="${left+(i%64)*dx}" y="${top+Math.floor(i/64)*dy}" width="${Math.max(.7,dx*.74)}" height="${Math.max(.65,dy*.7)}" fill="${selected?C.rust:active===0?(data[id].y?C.rust:C.blue):'#c2c7ca'}"/>`;
 });
 const bx=w*.71,by=top+gridH*.1,bw=w*.25,step=Math.min(bw/8,gridH*.8/8),xx=bx+(bw-8*step)/2;
 e.firstUpdate.indices.forEach((id,i)=>body+=circle(xx+(i%8+.5)*step,by+(Math.floor(i/8)+.5)*step,Math.max(1.3,step*.31),active>0?C.rust:'#bbc4ca'));
 body+=arrow(w*.59,(top+bottom)/2,w*.68,(top+bottom)/2,C.muted);
 body+=text(w*.84,15,'64 in a batch','small','middle');
 body+=text(left,h-3,'32 disjoint batches','small')+text(w*.84,h-3,'1 update','small','middle');
 set(svg,body,'All 2,048 observations are shuffled into 32 disjoint batches. One batch of 64 supplies one gradient and one update.');
}

function stepRail(svg,active){
 const {w,h}=canvas(svg),names=['Shuffle','Take 64','Gradient','Update'],cy=Math.min(30,h*.45),x=i=>(i+.5)*w/4;
 let body=line(x(0),cy,x(3),cy,'#d6dbde','',1.5);
 names.forEach((name,i)=>{
  body+=circle(x(i),cy,10,i===active?C.blue:'white',i<=active?C.blue:'#a9b2b7',1.5);
  body+=`<text x="${x(i)}" y="${cy+4}" text-anchor="middle" style="font:11px Arial;fill:${i===active?'white':C.blue}">${i+1}</text>`;
  body+=text(x(i),cy+28,name,'small','middle',i===active?C.blue:C.muted);
 });
 set(svg,body,`Four batching steps: shuffle, take 64, compute the average gradient, update. Current step: ${names[active]}.`);
}

function shuffleDiagram(svg,e,u){
 const {w,h}=canvas(svg),left=22,right=w-6,top=23,bottom=h-22,dx=(right-left)/64,dy=(bottom-top)/32;
 const order=e.traces[2].slice(0,32).flatMap(s=>s.indices),position=Array(data.length);
 order.forEach((id,slot)=>position[id]=slot);
 let body=text(left,12,u<.01?'Ordered by x':'Same observations · shuffled order','small');
 data.forEach((d,id)=>{
  const dest=position[id],t=smooth(u),x=left+lerp(id%64,dest%64,t)*dx,y=top+lerp(Math.floor(id/64),Math.floor(dest/64),t)*dy;
  body+=`<rect data-observation="${id}" x="${x.toFixed(2)}" y="${y.toFixed(2)}" width="${Math.max(.7,dx*.76)}" height="${Math.max(.65,dy*.7)}" fill="${d.y?C.rust:'white'}" ${d.y?'':`stroke="${C.blue}" stroke-width="${Math.min(.65,dx*.16,dy*.16)}"`}/>`;
 });
 if(u>.8)body+=`<rect x="${left-2}" y="${top-2}" width="${right-left+4}" height="${dy+3}" fill="none" stroke="${C.ink}" stroke-width="1.2"/>`;
 [0,15,31].forEach(row=>body+=text(left-7,top+(row+.7)*dy,String(row+1),'small','end'));
 body+=text(left,h-3,'Each row: 64 observations','small')+text(right,h-3,'32 rows','small','end');
 set(svg,body,'All 2,048 observation tiles move from input order into the recorded shuffled order. Open blue squares have label zero; filled rust squares have label one. Each of the 32 rows contains 64 observations and forms one batch.');
}

function squaredErrors(svg,residuals,amount){
 const {w,h}=canvas(svg),left=3,right=w-5,top=5,bottom=h-19,dx=(right-left)/64,squares=residuals.map(v=>v*v),max=Math.max(...squares),mean=squares.reduce((s,v)=>s+v,0)/64,Y=v=>bottom-v/max*(bottom-top);
 let body=line(left,bottom,right,bottom,'#999');
 squares.forEach((v,i)=>{const x=left+(i+.5)*dx;body+=line(x,bottom,x,Y(v)*1, i<amount?C.rust:'#ddd','',Math.max(1,dx*.6));});
 body+=line(left,Y(mean),right,Y(mean),C.blue,'4 3',1.7);
 body+=text(left,h-3,'64 squared errors','small')+text(right,h-3,`mean ${mean.toFixed(4)}`,'small','end',C.blue);
 set(svg,body,`The 64 squared residuals sum to ${squares.reduce((s,v)=>s+v,0).toFixed(4)}. Dividing by 64 gives batch mean squared error ${mean.toFixed(4)}. The dashed line marks that mean.`);
}

function componentBars(svg,g,{total=64,reference=null}={}){
 const {w,h}=canvas(svg),values=flatten(g),refs=reference?flatten(reference):null,left=10,right=w-10,dx=(right-left)/7,top=32,bottom=h-29,mid=top+(bottom-top)*.36,max=Math.max(.75,...(refs||[]).map(Math.abs)),scale=(bottom-top)*.57/max;
 let body=line(left,mid,right,mid,'#a7afb3');
 body+=text(left+dx*1.5,13,'Neuron 1: inactive','small','middle')+text(left+dx*4.5,13,'Neuron 2','small','middle')+text(left+dx*6.5,13,'Bias','small','middle');
 values.forEach((v,i)=>{
  const x=left+dx*(i+.5),y=mid-v*scale;
  if(refs)body+=Math.abs(refs[i])<1e-10?circle(x-4,mid,2,'white',C.blue,1.2):line(x-4,mid,x-4,mid-refs[i]*scale,C.blue,'',4);
  if(Math.abs(v)<1e-10)body+=circle(x+4,mid,2,'white',C.rust,1.3);
  else body+=line(x+4,mid,x+4,y,C.rust,'',4)+circle(x+4,y,2,C.rust);
  body+=text(x,v>=0?y-7:y+15,fmt(v),'small','middle')+text(x,h-8,labels[i],'','middle');
 });
 set(svg,body,`Rust: batch gradient after accumulating ${total} of 64 observations. Blue: full-data gradient at the same weights. Batch values: ${labels.map((label,i)=>label+' '+values[i].toFixed(4)).join(', ')}. The inactive first neuron has three zero gradients.`);
}

function biasContributions(svg,residuals,count){
 const {w,h}=canvas(svg),left=2,right=w-5,top=4,bottom=h-22,mid=(top+bottom)/2,dx=(right-left)/64,values=residuals.map(v=>2*v/64),max=Math.max(...values.map(Math.abs)),scale=(bottom-top)/2/max;
 let body=line(left,mid,right,mid,'#aaa');
 values.forEach((v,i)=>{const x=left+(i+.5)*dx;body+=line(x,mid,x,mid-v*scale,i<count?C.rust:'#e0e0e0','',Math.max(1,dx*.6));});
 body+=text(left,h-4,'Each bar: 2 × residual / 64','small')+text(right,h-4,`${count} / 64`,'small','end');
 set(svg,body,`For the output bias, each observation contributes twice its residual divided by 64. ${count} contributions have accumulated so far.`);
}

function biasStep(svg,first,eta,u){
 const {w,h}=canvas(svg),left=28,right=w-16,y=Math.max(55,h*.57),X=v=>left+(v-.49)/.03*(right-left),c=lerp(first.before.c,first.after.c,u);
 let body=text(left,15,`Gradient: ${signed(first.gradient.c)}`,'small')+text(right,15,`η = ${eta}`,'small','end');
 body+=line(left,y,right,y,'#a2aaaf');
 for(const tick of [.49,.50,.51,.52])body+=line(X(tick),y-3,X(tick),y+4,'#a2aaaf')+text(X(tick),y+20,tick.toFixed(2),'small','middle');
 body+=line(X(.5),y-14,X(.5),y+5,C.blue,'3 3')+circle(X(.5),y,4,'white',C.blue,2);
 if(u>.015)body+=arrow(X(.5),y-12,X(c),y-12,C.teal,2);
 body+=circle(X(c),y,5,C.teal)+text(X(c),y-26,c.toFixed(4),'','middle',C.teal);
 body+=text(left,h-3,'A negative gradient increases c.','small');
 set(svg,body,`The output bias increases from ${first.before.c.toFixed(4)} to ${c.toFixed(4)}. The recorded update is minus learning rate ${eta} times gradient ${signed(first.gradient.c)}.`);
 return c;
}

function stepArithmetic(svg,first,eta){
 const {w,h}=canvas(svg),change=first.after.c-first.before.c,space=Math.min(28,(h-8)/3),x=3;
 const body=text(x,space-2,'change = − learning rate × gradient','small')+
 text(x,space*2-2,`= −${eta} × (${signed(first.gradient.c)}) = ${signed(change)}`,'', 'start',C.teal)+
 text(x,space*3-2,`new c = ${first.before.c.toFixed(4)} + ${change.toFixed(4)} = ${first.after.c.toFixed(4)}`,'','start',C.blue);
 set(svg,body,`Using unrounded values: new output bias equals ${first.before.c} minus ${eta} times ${first.gradient.c}, giving ${first.after.c}. Displayed values are rounded.`);
}

function nextBatchFacts(svg,first,second,g2,u){
 const {w,h}=canvas(svg),next=u>=.5,g=next?g2:first.gradient,batch=next?2:1,top=Math.min(24,h*.28),dx=w/32;
 let body=text(0,top,`Batch ${batch} of 32`,'')+text(w,top,next?`g(c) = ${signed(g.c)}`:'Update 1 complete','','end',next?C.rust:C.blue);
 for(let i=0;i<32;i++)body+=`<rect x="${i*dx}" y="${top+13}" width="${Math.max(1,dx-2)}" height="${i===batch-1?10:5}" fill="${i===batch-1?C.rust:i<batch-1?C.blue:'#ddd'}"/>`;
 body+=text(0,h-4,next?'New batch · updated model':'Keep the updated weights','small')+text(w,h-4,'64 observations','small','end');
 set(svg,body,next?`Batch 2 of 32. Its output-bias gradient is ${g.c.toFixed(4)}, evaluated at the model after the first update. The first and second batches share no observations.`:'Update 1 is complete. Keep the updated weights while selecting the next 64 observations.');
}

export function renderSGD(plot,detail,scene,p,e){
 const kind=scene.kind,u=clamp((p-.10)/.72),first=e.firstUpdate,rows=first.indices.map(i=>data[i]),residuals=rows.map(d=>predict(first.before,d.x)-d.y),eta=e.traces[2][0].eta;
 let plotTitle='One minibatch, one update',detailTitle='The batching procedure',info={kind,progress:p,batchSize:64};
 if(kind==='sgd-transition'){
  chapterDiagram(plot,e,u);stepRail(detail,Math.min(3,Math.floor(u*4)));
  plotTitle='Stochastic gradient descent';detailTitle='Learn the weights, one small batch at a time';
 }
 if(kind==='sgd-plan'){
  const active=Math.min(3,Math.floor(u*4));planDiagram(plot,e,active);stepRail(detail,active);info.activeStep=active;
  plotTitle='From the dataset to one minibatch';detailTitle='Within one epoch';
 }
 if(kind==='shuffle'){
  shuffleDiagram(plot,e,u);facts(detail,[['Data remain fixed','2,048','Only their order changes'],['One epoch','32 × 64','Each appears once']]);
  plotTitle='Shuffle the order once per epoch';detailTitle='Open: y = 0 · filled: y = 1';info.shuffleProgress=u;
 }
 if(kind==='batch'||kind==='batch-loss'){
  const c=chart(plot,{xmin:-1,xmax:1}),count=Math.min(64,Math.floor(u*64)+1),selected=kind==='batch'?first.indices.slice(0,count):first.indices;
  let body=c.base+c.step()+observations(c,{batch:selected})+c.path(x=>predict(first.before,x),C.blue);
  if(kind==='batch-loss')first.indices.slice(0,count).forEach(i=>{const d=data[i];body+=line(c.X(d.x),c.Y(d.y),c.X(d.x),c.Y(predict(first.before,d.x)),C.rust,'',1.4);});
  set(plot,body,`${scene.title} ${scene.caption}`);
  if(kind==='batch')batchFacts(detail,first.indices,count);else squaredErrors(detail,residuals,count);
  plotTitle=kind==='batch'?'Take the first 64 shuffled observations':'Evaluate this batch at the current weights';
  detailTitle=kind==='batch'?'Batch 1 of 32 · no replacement':`Batch MSE = ${first.batchBefore.toFixed(4)}`;
  info={...info,batch:1,selected:count,batchFirst:first.indices[0],batchMSE:first.batchBefore};
 }
 if(kind==='batch-gradient'){
  const count=Math.min(64,Math.floor(u*64)+1),partial=gradient(first.before,rows.slice(0,count)),factor=count/64;
  partial.c*=factor;partial.units.forEach(unit=>{for(const key of ['w','b','a'])unit[key]*=factor;});
  componentBars(plot,partial,{total:count,reference:gradient(first.before,data)});biasContributions(detail,residuals,count);
  plotTitle=count===64?'The average gradient of 64 observations':`Accumulating scaled gradients · ${count} / 64`;detailTitle=`Inside the output-bias gradient · sum = ${signed(partial.c)}`;
  info={...info,contributions:count,gradientC:partial.c,gradient:flatten(partial)};
 }
 if(kind==='parameter-step'){
  const c=biasStep(plot,first,eta,u);stepArithmetic(detail,first,eta);
  plotTitle='Follow one parameter: output bias c';detailTitle='Subtract the gradient, scaled by the learning rate';
  info={...info,eta,gradientC:first.gradient.c,outputBias:c,deltaC:first.after.c-first.before.c};
 }
 if(kind==='next-batch'){
  const second=e.traces[2][1],g2=gradient(second.model,second.indices.map(i=>data[i])),c=chart(plot,{xmin:-1,xmax:1}),next=u>=.5;
  let body=c.base+c.step()+observations(c)+c.path(x=>predict(second.model,x),C.blue);
  const oldAlpha=1-smooth(u*2),newAlpha=smooth((u-.3)/.5);
  [first.indices,second.indices].forEach((indices,j)=>{
   const alpha=j?newAlpha:oldAlpha;
   body+=`<g opacity="${alpha}">`;
   indices.forEach(i=>{const d=data[i];if(j)body+=line(c.X(d.x),c.Y(d.y),c.X(d.x),c.Y(predict(second.model,d.x)),C.rust,'',1);body+=circle(c.X(d.x),c.Y(d.y),3.3,C.rust,'white',.7);});
   body+='</g>';
  });
  set(plot,body,'The next 64 shuffled observations replace the first batch. None are repeated. Their residuals and gradient are recomputed using the model after update one.');
  nextBatchFacts(detail,first,second,g2,u);
  plotTitle='Keep the updated curve; select the next batch';detailTitle='New observations → a new gradient';
  info={...info,batch:next?2:1,batchFirst:(next?second.indices:first.indices)[0],outputBias:second.model.c,gradientC:next?g2.c:null,batchOverlap:0};
 }
 return {...info,plotTitle,detailTitle};
}
