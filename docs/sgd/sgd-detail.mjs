import {predict,gradient,data,loss} from './model.mjs';
import {mathText,richText,plainText} from './math.mjs';

// Shared geometry keeps the detailed SGD sequence on the same axes and palette.
export const C={blue:'#234e70',rust:'#a54f32',ink:'#242424',muted:'#606060',teal:'#007f73',gray:'#aaa'};
export const clamp=x=>Math.max(0,Math.min(1,x));
// Larger projected type needs matching breathing room inside each SVG.
const projected=()=>globalThis.innerWidth>=1000;
export const fmt=x=>Math.abs(x)<1e-10?'0':Number(x.toFixed(3)).toString();
export const line=(x1,y1,x2,y2,color=C.ink,dash='',width=1)=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}" ${dash?`stroke-dasharray="${dash}"`:''}/>`;
export const text=(x,y,t,cls='',anchor='start',color='')=>`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}" ${color?`style="fill:${color}"`:''}>${t}</text>`;
export function set(svg,body,label){svg.innerHTML=body;svg.setAttribute('aria-label',plainText(label));}
export function canvas(svg){const r=svg.getBoundingClientRect();const w=Math.max(260,r.width),h=Math.max(70,r.height);svg.setAttribute('viewBox',`0 0 ${w} ${h}`);return {w,h};}
export function chart(svg,{xmin=-1,xmax=1.2,ymin=-.25,ymax=1.4,xticks=[-1,0,1],yticks=[0,.5,1],topInset=0}={}){
 const {w,h}=canvas(svg),large=projected(),left=large?48:37,right=w-(large?25:17),top=(large?23:15)+topInset,bottom=h-(large?39:27);
 const X=x=>left+(x-xmin)/(xmax-xmin)*(right-left),Y=y=>bottom-(y-ymin)/(ymax-ymin)*(bottom-top);
 let base=`<defs><clipPath id="clip-${svg.id}"><rect x="${left}" y="${top}" width="${right-left}" height="${bottom-top}"/></clipPath></defs>`;
 for(const y of yticks)base+=line(left,Y(y),right,Y(y),'#e7e7e7')+text(left-8,Y(y)+4,fmt(y),'small','end');
 base+=line(left,bottom,right,bottom,'#767676');
 for(const x of xticks)base+=line(X(x),bottom,X(x),bottom+4,'#767676')+text(X(x),bottom+(large?27:19),fmt(x),'small','middle');
 base+=mathText(right+8,bottom+(large?27:19),'x','small');
 const path=(fn,color=C.blue,width=3,dash='',opacity=1)=>{let d='';for(let i=0;i<=420;i++){const x=xmin+(xmax-xmin)*i/420;d+=(i?'L':'M')+X(x).toFixed(2)+','+Y(fn(x)).toFixed(2);}return `<path d="${d}" fill="none" stroke="${color}" stroke-width="${width}" stroke-opacity="${opacity}" ${dash?`stroke-dasharray="${dash}"`:''} clip-path="url(#clip-${svg.id})"/>`;};
 const step=()=>`<path d="M${X(xmin)},${Y(0)}H${X(0)}M${X(0)},${Y(1)}H${X(xmax)}" fill="none" stroke="${C.muted}" stroke-width="2" stroke-dasharray="6 5"/><circle cx="${X(0)}" cy="${Y(0)}" r="3" fill="${C.muted}"/><circle cx="${X(0)}" cy="${Y(1)}" r="3" fill="white" stroke="${C.muted}" stroke-width="1.5"/>`;
 return {w,h,X,Y,path,step,base,left,right,top,bottom,xmin,xmax};
}
export function facts(svg,entries){const {w,h}=canvas(svg),large=projected();let html='';if(h<150){const cols=entries.length;entries.forEach((e,i)=>{const x=6+i*w/cols;html+=richText(x,18,e[0],'small')+richText(x,47,e[1],cols>2?'':'large','start',e[3]||'')+(e[2]&&cols<3?richText(x,68,e[2],'small'):'');});}else{const space=Math.min(large?135:100,(h-20)/entries.length);entries.forEach((e,i)=>{const y=24+i*space;html+=richText(10,y,e[0],'small')+richText(10,y+(large?44:37),e[1],'large','start',e[3]||'')+(e[2]?richText(10,y+(large?73:59),e[2],'small'):'');});}set(svg,html,entries.map(e=>e.slice(0,3).join(': ')).join('. '));}
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
export function gradientBars(svg,g1,g2=null){const {w,h}=canvas(svg),vals=g1.units.flatMap(u=>[u.w,u.b,u.a]).concat(g1.c),other=g2?g2.units.flatMap(u=>[u.w,u.b,u.a]).concat(g2.c):null;
 const large=projected(),top=large?32:18,bottom=h-(large?34:24),mid=(top+bottom)/2,scale=(bottom-top)/2/Math.max(.02,...vals.map(Math.abs),...(other||[]).map(Math.abs)),col=(w-30)/7;
 let body=line(10,mid,w-10,mid,'#999');vals.forEach((v,i)=>{const x=20+col*i;body+=line(x,mid,x,mid-v*scale,C.blue,'',5);if(other)body+=line(x+7,mid,x+7,mid-other[i]*scale,C.rust,'',5);body+=mathText(x+2,bottom+19,mathLabels[i],'small','middle');});
 body+=text(10,projected()?19:10,other?'Blue: full data · rust: minibatch':'Gradient components (signed)','small');set(svg,body,'Gradient components for the seven trainable parameters: four hidden-layer θ parameters and three output-layer φ parameters. '+(other?'Full gradient in blue, minibatch estimate in rust.':''));
}
export function interpolate(a,b,t){return {c:a.c+(b.c-a.c)*t,units:a.units.map((v,i)=>Object.fromEntries(['w','b','a'].map(k=>[k,v[k]+(b.units[i][k]-v[k])*t])))};}

export const sgdKinds=['sgd-plan','partition','shuffle','batch','point-loss','batch-loss','batch-gradient','parameter-step','next-batch'];
const labels=['θ₁₁','θ₁₀','φ₁','θ₂₁','θ₂₀','φ₂','φ₀'];
const mathLabels=['\\theta_{11}','\\theta_{10}','\\phi_1','\\theta_{21}','\\theta_{20}','\\phi_2','\\phi_0'];
const flatten=g=>g.units.flatMap(v=>[v.w,v.b,v.a]).concat(g.c);
const signed=(v,d=4)=>(v<0?'-':v>0?'+':'')+Math.abs(v).toFixed(d);
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
 body+=text(w/2,h-4,`${selected} of 64 observations selected`,'small','middle');
 set(svg,body,`${selected} of the first batch’s 64 observations selected. This is one of 32 disjoint batches.`);
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
 let body=richText(left,12,u<.01?String.raw`Ordered by \(x\)`:'Same observations · shuffled order','small');
 data.forEach((d,id)=>{
  const dest=position[id],t=smooth(u),x=left+lerp(id%64,dest%64,t)*dx,y=top+lerp(Math.floor(id/64),Math.floor(dest/64),t)*dy;
  body+=`<rect data-observation="${id}" x="${x.toFixed(2)}" y="${y.toFixed(2)}" width="${Math.max(.7,dx*.76)}" height="${Math.max(.65,dy*.7)}" fill="${d.y?C.rust:'white'}" ${d.y?'':`stroke="${C.blue}" stroke-width="${Math.min(.65,dx*.16,dy*.16)}"`}/>`;
 });
 if(u>.8)body+=`<rect x="${left-2}" y="${top-2}" width="${right-left+4}" height="${dy+3}" fill="none" stroke="${C.ink}" stroke-width="1.2"/>`;
 [0,15,31].forEach(row=>body+=text(left-7,top+(row+.7)*dy,String(row+1),'small','end'));
 body+=text(left,h-3,'Each row: 64 observations','small')+text(right,h-3,'32 rows','small','end');
 set(svg,body,'All 2,048 observation tiles move from input order into the recorded shuffled order. Open blue squares have label zero; filled rust squares have label one. Each of the 32 rows contains 64 observations and forms one batch.');
}

function partitionDiagram(svg,e,u){
 const {w,h}=canvas(svg),large=projected(),left=5,right=w-5,top=large?34:24,bottom=h-(large?35:23),t=smooth(u),cols=8,groupRows=4,compact=h<210||w<450;
 const labelSpace=large?29:15,labelFont=large?(globalThis.innerWidth>=1600&&globalThis.innerHeight>800?20:16):10;
 const gapX=8*t,gapY=7*t,gw=(right-left-gapX*(cols-1))/cols,gh=(bottom-top-gapY*(groupRows-1))/groupRows;
 const batches=e.traces[2].slice(0,32),order=batches.flatMap(s=>s.indices),dx=(right-left)/64,dy=(bottom-top)/32;
 let body=text(left,large?19:13,'2,048 observations','small');
 body+=`<rect x="${left}" y="${top}" width="${right-left}" height="${bottom-top}" fill="#f3f4f4" stroke="#aab1b6" stroke-opacity="${1-t}"/>`;
 batches.forEach((batch,j)=>{
  const x=left+(j%cols)*(gw+gapX),y=top+Math.floor(j/cols)*(gh+gapY);
  body+=`<g data-batch="${j+1}" data-count="${batch.indices.length}"><rect x="${x}" y="${y}" width="${gw}" height="${gh}" rx="2" fill="white" fill-opacity="${t}" stroke="${j===0?C.rust:'#9aa7b0'}" stroke-width="${j===0?1.7:1}" stroke-opacity="${t}"/>`;
  if(compact)body+=`<g opacity="${t}">`+mathText(x+gw/2,y+gh/2+3,`B_{${j+1}}`,'small','middle',j===0?C.rust:C.ink,{size:10,width:gw})+'</g>';
  else body+=`<g opacity="${t}">`+richText(x+4,y+labelSpace-4,`\\(B_{${j+1}}\\) · 64`,'small','start',j===0?C.rust:C.muted,{size:labelFont,width:gw-8})+'</g>';
  body+='</g>';
 });
 if(!compact)order.forEach((id,i)=>{
  const group=Math.floor(i/64),within=i%64,gx=left+(group%cols)*(gw+gapX),gy=top+Math.floor(group/cols)*(gh+gapY);
  const x=lerp(left+(i%64+.5)*dx,gx+4+(within%8+.5)*(gw-8)/8,t),y=lerp(top+(Math.floor(i/64)+.5)*dy,gy+labelSpace+(Math.floor(within/8)+.5)*(gh-labelSpace-3)/8,t);
  const mark=Math.max(.6,Math.min((gw-8)/8,(gh-labelSpace-3)/8)*.50);
  body+=`<rect data-observation="${id}" x="${x-mark/2}" y="${y-mark/2}" width="${mark}" height="${mark}" fill="${group===0?C.rust:C.blue}" opacity="${group===0?1:.5}"/>`;
 });
 body+=text(left,h-5,'32 batches of 64 · no overlap','small');
 if(w>420)body+=richText(right,h-5,String.raw`\(32\times64=2{,}048\)`,'small','end');
 set(svg,body,'Partition all 2,048 shuffled observations into 32 disjoint batches of 64. Every observation belongs to exactly one batch, with no omissions and no repeats. Batch 1 is outlined.');
}

function pointLoss(svg,detail,first,u){
 const id=first.indices[0],d=data[id],prediction=predict(first.before,d.x),residual=prediction-d.y,square=.5*residual*residual,c=chart(svg,{xmin:-1,xmax:1,topInset:projected()?28:18});
 const px=c.X(d.x),py=c.Y(prediction),ty=c.Y(d.y),reveal=smooth(u/.55);
 let body=c.base+c.step()+c.path(x=>predict(first.before,x),C.blue);
 body+=`<g opacity=".22">`+observations(c,{batch:first.indices})+'</g>';
 body+=circle(px,ty,5,C.rust,'white',1)+circle(px,ty,8,'none',C.ink,1);
 body+=`<g opacity="${reveal}">`+line(px,ty,px,py,C.rust,'',2)+circle(px,py,4,'white',C.blue,2)+'</g>';
 const labelX=Math.min(c.right-(projected()?170:118),px+12);
 body+=richText(labelX,ty-7,`target \\(y_i=${d.y}\\)`,'small','start',C.rust);
 body+=`<g opacity="${reveal}">`+text(labelX,py-9,`prediction ${prediction.toFixed(4)}`,'small','start',C.blue)+'</g>';
 body+=text(c.left,projected()?20:13,'Parameters fixed','small');
 body+=text(c.left,c.top+2,`Observation ${id+1} from batch 1`,'small');
 set(svg,body,`For observation ${id+1}, x is ${d.x.toFixed(4)} and the target is ${d.y}. The current model predicts ${prediction.toFixed(4)}. The vertical segment is the residual: prediction minus target. Half its square is the point loss ${square.toFixed(4)}.`);
 const {w,h}=canvas(detail),large=projected(),gap=Math.min(27,(h-6)/3),x=3;
 let arithmetic;
 if(large){
  arithmetic=richText(x,24,String.raw`Residual \(=\) prediction \(-\) target`,'small')+mathText(x,54,`${prediction.toFixed(4)}-${d.y}=${residual.toFixed(4)}`,'','start',C.rust,{width:w-6});
  arithmetic+=`<g opacity="${smooth((u-.35)/.45)}">`+richText(x,103,String.raw`Point loss \(=\frac12\times\text{residual}^2\)`,'small')+mathText(x,133,`\\tfrac12(${residual.toFixed(4)})^2=${square.toFixed(4)}`,'','start',C.rust,{width:w-6})+'</g>';
 }else{
  arithmetic=richText(x,gap-1,`prediction \\(-\\) target \\(=${prediction.toFixed(4)}-${d.y}\\)`,'','start','',{width:w-6});
  arithmetic+=richText(x,gap*2-1,`residual \\(=${residual.toFixed(4)}\\)`,'','start',C.rust);
  arithmetic+=`<g opacity="${smooth((u-.35)/.45)}">`+mathText(x,gap*3-1,`\\ell_i=\\tfrac12(${residual.toFixed(4)})^2=${square.toFixed(4)}`,'','start',C.rust,{width:w-6})+'</g>';
 }
 set(detail,arithmetic,`Prediction minus target is ${residual.toFixed(4)}. Taking half the squared residual gives point loss ${square.toFixed(4)}. Numbers are displayed rounded.`);
 return {pointIndex:id,pointX:d.x,pointTarget:d.y,pointPrediction:prediction,pointResidual:residual,pointLoss:square};
}

function squaredErrors(svg,residuals,amount,normalized){
 const {w,h}=canvas(svg),large=projected(),left=3,right=w-5,top=large?62:23,bottom=h-(large?55:23),dx=(right-left)/64,squares=residuals.map(v=>.5*v*v),max=Math.max(...squares),sum=squares.slice(0,amount).reduce((s,v)=>s+v,0),mean=squares.reduce((s,v)=>s+v,0)/64,Y=v=>bottom-v/max*(bottom-top);
 let body=richText(left,large?20:13,`Sum \\(S_B=${sum.toFixed(4)}\\)`,'small','start',C.rust)+text(large?left:right,large?44:13,`${amount} of 64 losses added`,'small',large?'start':'end');
 body+=line(left,bottom,right,bottom,'#999');
 squares.forEach((v,i)=>{const x=left+(i+.5)*dx;body+=line(x,bottom,x,Y(v),i<amount?C.rust:'#e2e2e2','',Math.max(1,dx*.6));});
 // Retain the same first observation as its loss becomes one term of the sum.
 body+=circle(left+dx*.5,Y(squares[0]),2.4,'white',C.ink,1);
 if(normalized)body+=line(left,Y(mean),right,Y(mean),C.blue,'4 3',1.7);
 if(large&&normalized)body+=richText(left,h-34,String.raw`Batch half-MSE \(=\text{sum}/64\)`,'small','start',C.blue)+mathText(left,h-9,`\\frac{${sum.toFixed(4)}}{64}=${mean.toFixed(4)}`,'','start',C.blue,{width:w-8});
 else body+=richText(left,h-9,normalized?`Half-MSE \\(L_B=\\frac{${sum.toFixed(4)}}{64}=${mean.toFixed(4)}\\)`:String.raw`Each point loss \(=\frac12\times\text{residual}^2\)`,'small','start',normalized?C.blue:C.muted,{width:w-8});
 set(svg,body,`${amount} half-squared errors have been added to the running sum S_B, now ${sum.toFixed(4)}. ${normalized?`All 64 half-squared errors sum to ${sum.toFixed(4)}. Divide this sum by 64 to obtain the mean batch loss L_B = ${mean.toFixed(4)}. The dashed line marks the mean.`:'The circled first bar is the point loss from the preceding scene. The mean will be calculated after all 64 terms have been added.'}`);
 return {batchSum:sum,lossesAdded:amount,batchHalfMSE:mean,batchMSE:2*mean,normalized};
}

function gradientSpaces(svg,indices,residuals,g){
 const {w,h}=canvas(svg),large=projected(),phone=globalThis.innerWidth<=700,diagramHeight=h-(phone?58:0),values=flatten(g),left=large?36:27,right=w-9,dx=(right-left)/values.length;
 const residualTop=large?38:27,residualBottom=diagramHeight*.30,residualMid=(residualTop+residualBottom)/2,residualScale=(residualBottom-residualTop)*.45/Math.max(...residuals.map(Math.abs)),rx=(right-left)/residuals.length;
 const parameterTop=diagramHeight*.59,parameterBottom=diagramHeight-(large?43:31),parameterMid=(parameterTop+parameterBottom)/2,parameterScale=(parameterBottom-parameterTop)*(large?.30:.34)/Math.max(.08,...values.map(Math.abs));
 let body='<g data-space="observations">'+text(3,large?20:13,'Observation space · 64 residuals','small','start',C.rust)+line(left,residualMid,right,residualMid,'#c5b5ae')+text(left-8,residualMid+3,'0','small','end');
 body+=text(right,large?20:13,'Parameters fixed','small','end');
 residuals.forEach((v,i)=>{
  const x=left+rx*(i+.5),y=residualMid-v*residualScale;
  body+=`<g data-observation="${indices[i]}" data-batch-position="${i+1}" data-residual="${v}">`+line(x,residualMid,x,y,C.rust,'',Math.max(1,rx*.62))+'</g>';
 });
 body+=text(left,residualBottom+(large?22:14),'1','small')+text((left+right)/2,residualBottom+(large?22:14),'Position in this batch','small','middle')+text(right,residualBottom+(large?22:14),'64','small','end')+'</g>';
 body+='<g data-space="parameters">'+richText(3,diagramHeight*.54,String.raw`Parameter space · 7 components: \(4\theta+3\phi\)`,'small','start',C.blue,{width:w-6})+line(left,parameterMid,right,parameterMid,'#a7afb3')+text(left-8,parameterMid+3,'0','small','end');
 values.forEach((v,i)=>{
  const x=left+dx*(i+.5),y=parameterMid-v*parameterScale,color=i===values.length-1?C.teal:C.blue;
  body+=`<g data-parameter="${labels[i]}" data-gradient="${v}">`;
  if(Math.abs(v)<1e-10)body+=circle(x,parameterMid,2,'white',color,1.3);
  else body+=line(x,parameterMid,x,y,color,'',4)+circle(x,y,2,color);
  body+=text(x,v>=0?y-9:y+(large?20:13),fmt(v),'small','middle',color)+mathText(x,diagramHeight-7,mathLabels[i],'','middle',color)+'</g>';
 });
 body+='</g>';
 if(phone){
  body+=line(3,h-53,right,h-53,'#e7e7e7');
  body+=richText(3,h-36,String.raw`\(\phi_0\) shifts every prediction equally`,'small','start',C.teal);
  body+=mathText(3,h-16,`\\frac{\\partial L_{B_1}}{\\partial\\phi_0}=\\text{mean residual}=${signed(g.c)}`,'small','start',C.teal,{width:w-6});
 }
 set(svg,body,`Two different spaces. The upper plot contains ${residuals.length} signed residuals, one for each observation in batch order: prediction minus target. The lower plot contains ${values.length} gradient components, one for each model parameter: four hidden-layer θ parameters and three output-layer φ parameters. A component tells how the batch loss changes when that parameter changes slightly, with the others held fixed. Each component combines errors across the batch according to that parameter's effect on the predictions; residuals are not assigned one-to-one to parameters. ${labels.map((label,i)=>label+' '+values[i].toFixed(4)).join(', ')}. All seven components are calculated at the same current model. No parameters have been updated yet.`);
}

function gradientFacts(svg,residuals,g){
 const {w,h}=canvas(svg),large=projected(),gap=Math.min(large?43:30,(h-6)/3),x=3,meanResidual=residuals.reduce((sum,value)=>sum+value,0)/residuals.length;
 const body=richText(x,gap-1,String.raw`Gradient \(=\) mean of the 64 residuals`,'small')+
 mathText(x,gap*2-1,`\\dfrac{r_1+\\cdots+r_{64}}{64}=${signed(meanResidual)}`,'','start',C.teal,{width:w-6});
 set(svg,body,`For the output bias φ₀, changing φ₀ shifts every prediction by the same amount. With half-MSE loss, its gradient component is the mean of all ${residuals.length} residuals: ${meanResidual}, displayed as ${signed(g.c)}. Other parameters affect predictions differently, so their components combine the residuals differently. The model is still fixed.`);
 return meanResidual;
}

// All seven entries share the curve's interpolation progress. The scalar output-bias
// illustration never changes the model; this table is the actual vector step.
export function renderParameterUpdate(svg,before,after,g,progress){
 const {w,h}=canvas(svg),large=projected(),oldValues=flatten(before),values=flatten(interpolate(before,after,progress)),gradients=flatten(g);
 const rowTop=Math.min(large?54:37,h*.22),gap=Math.min(large?42:38,(h-rowTop-(large?36:9))/6),columns=large?{label:4,before:w*.32,arrow:w*.405,current:w*.66,gradient:w-4}:{label:4,before:w*.30,arrow:w*.355,current:w*.60,gradient:w-4},header=large?21:12;
 let body=text(columns.label,header,large?'Param.':'Parameter','small')+text(columns.before,header,'Before','small','end')+text(columns.current,header,'Now','small','end')+text(columns.gradient,header,'Gradient','small','end')+line(0,header+(large?9:6),w,header+(large?9:6),'#ddd');
 const displayedParameters=values.map((value,i)=>({label:labels[i],before:oldValues[i],current:value,gradient:gradients[i],unchanged:gradients[i]===0}));
 displayedParameters.forEach((entry,i)=>{
  const y=rowTop+i*gap,color=entry.unchanged?C.muted:C.teal;
  body+=`<g data-parameter="${entry.label}" data-before="${entry.before}" data-current="${entry.current}" data-gradient="${entry.gradient}">`;
  body+=mathText(columns.label,y,mathLabels[i],'','start',C.blue)+text(columns.before,y,entry.before.toFixed(4),'small','end')+mathText(columns.arrow,y,'\\to','small','middle')+text(columns.current,y,entry.current.toFixed(4),'small','end',color)+text(columns.gradient,y,entry.unchanged?(large?'0':'0 · unchanged'):signed(entry.gradient),'small','end',color)+'</g>';
 });
 if(large)body+=richText(4,rowTop+6*gap+29,String.raw`0 gradient \(\to\) no change`,'small');
 set(svg,body,`All seven parameters take the same simultaneous update, at ${(progress*100).toFixed(0)} percent of the animation. Every gradient was calculated at the model before the update. ${displayedParameters.map(entry=>`${entry.label}: ${entry.before.toFixed(4)} to ${entry.current.toFixed(4)}, gradient ${signed(entry.gradient)}${entry.unchanged?', unchanged because its gradient is zero':''}`).join('. ')}. The curve uses these seven current values together.`);
 return displayedParameters;
}

function biasStep(svg,first,eta,u){
 const {w,h}=canvas(svg),large=projected(),left=large?38:28,right=w-(large?27:16),y=Math.max(large?85:55,h*.57),X=v=>left+(v-.49)/.03*(right-left),c=lerp(first.before.c,first.after.c,u);
 let body=richText(left,large?22:15,`Gradient: \\(${signed(first.gradient.c)}\\)`,'small')+mathText(right,large?22:15,`\\alpha=${eta}`,'small','end');
 body+=line(left,y,right,y,'#a2aaaf');
 for(const tick of [.49,.50,.51,.52])body+=line(X(tick),y-3,X(tick),y+4,'#a2aaaf')+text(X(tick),y+(large?30:20),tick.toFixed(2),'small','middle');
 body+=line(X(.5),y-14,X(.5),y+5,C.blue,'3 3')+circle(X(.5),y,4,'white',C.blue,2);
 if(u>.015)body+=arrow(X(.5),y-12,X(c),y-12,C.teal,2);
 body+=circle(X(c),y,5,C.teal)+text(X(c),y-26,c.toFixed(4),'','middle',C.teal);
 body+=text(left,h-3,'One component of the same update.','small');
 set(svg,body,`A magnified illustration of one component of the same seven-parameter update: output bias φ₀ would increase from ${first.before.c.toFixed(4)} to ${c.toFixed(4)}. The arithmetic is minus learning rate ${eta} times gradient ${signed(first.gradient.c)}. The model has not changed and no update has been counted yet; all seven parameters will take their step together in the next scene.`);
 return c;
}

function stepArithmetic(svg,first,eta){
 const {w,h}=canvas(svg),large=projected(),change=first.after.c-first.before.c,space=Math.min(large?42:28,(h-8)/3),x=3;
 const body=richText(x,space-2,String.raw`Change \(=-\) learning rate \(\times\) gradient`,'small','start','',{width:w-6})+
 mathText(x,space*2-2,`=-${eta}\\times(${signed(first.gradient.c)})=${signed(change)}`,'', 'start',C.teal,{width:w-6})+
 mathText(x,space*3-2,`\\phi_0:\ ${first.before.c.toFixed(4)}+${change.toFixed(4)}=${first.after.c.toFixed(4)}`,'','start',C.teal,{width:w-6});
 set(svg,body,`This is a preview of one component, not a separate model update. Using unrounded values: new output bias φ₀ equals ${first.before.c} minus ${eta} times ${first.gradient.c}, giving ${first.after.c}. Displayed values are rounded.`);
}

function nextBatchFacts(svg,second,model,u,updateProgress){
 const {w,h}=canvas(svg),large=projected(),top=Math.min(large?25:21,h*.26),dx=w/32,done=updateProgress===1,batchLoss=loss(second.model,second.indices.map(i=>data[i]));
 let body=text(0,top,'Batch 2 of 32','')+text(w,top,done?'Update 2 complete':u<.15?'Take batch 2':u<.3?'Compute its loss':u<.5?'Find its gradient':'Apply update 2','small','end',done?C.teal:C.rust);
 for(let i=0;i<32;i++)body+=`<rect x="${i*dx}" y="${top+11}" width="${Math.max(1,dx-2)}" height="${i===1?9:5}" fill="${i===1?C.rust:i===0?C.blue:'#ddd'}"/>`;
 const footer=u<.15?'No observations shared with batch 1.':u<.3?`Half-MSE at the new weights: ${batchLoss.toFixed(4)}`:u<.5?'New gradient: one component per parameter.':`All 7 together · half-MSE ${loss(model,second.indices.map(i=>data[i])).toFixed(4)}`;
 body+=text(0,h-4,footer,'small','start',u<.5?C.rust:C.blue);
 set(svg,body,`Batch 2 has 64 observations and shares none with batch 1. Its half-MSE loss ${batchLoss.toFixed(4)} and all seven gradient components are evaluated at the weights after update 1. ${done?'Update 2 is now complete; all seven parameters have taken their step together.':updateProgress>0?'Applying the second update to all seven parameters together; zero-gradient components stay unchanged.':u<.15?'Select the next 64 observations.':u<.3?'First evaluate their loss at the updated weights.':'Next obtain one gradient component per parameter.'} The current batch half-MSE is ${loss(model,second.indices.map(i=>data[i])).toFixed(4)}.`);
}

export function renderSGD(plot,detail,scene,p,e){
 const kind=scene.kind,u=clamp((p-.10)/.72),first=e.firstUpdate,rows=first.indices.map(i=>data[i]),residuals=rows.map(d=>predict(first.before,d.x)-d.y),eta=e.traces[2][0].eta;
 let plotTitle='One minibatch, one update',detailTitle='The batching procedure',info={kind,progress:p,batchSize:64,completedUpdates:0};
 if(kind==='sgd-plan'){
  const active=Math.min(3,Math.floor(u*4));planDiagram(plot,e,active);stepRail(detail,active);info.activeStep=active;
  plotTitle='From the dataset to one minibatch';detailTitle='Within one epoch';
 }
 if(kind==='shuffle'){
  shuffleDiagram(plot,e,u);facts(detail,[['Data remain fixed','2,048','Only their order changes'],['One epoch',String.raw`\(32\times64\)`,'Each appears once']]);
  plotTitle='Shuffle the order once per epoch';detailTitle=String.raw`Open: \(y=0\) · filled: \(y=1\)`;info.shuffleProgress=u;
 }
 if(kind==='partition'){
  const batches=e.traces[2].slice(0,32),indices=batches.flatMap(s=>s.indices),coverage=new Set(indices).size;
  partitionDiagram(plot,e,u);set(detail,'','');
  plotTitle='A partition: disjoint groups that cover all the data';detailTitle='';
  info={...info,partitionBatchCount:batches.length,partitionSize:indices.length,partitionCoverage:coverage,partitionOverlap:indices.length-coverage,batch:1,batchFirst:first.indices[0]};
 }
 if(kind==='point-loss'){
  info={...info,...pointLoss(plot,detail,first,u),batch:1,batchFirst:first.indices[0]};
  plotTitle='One point: prediction, target, and their difference';detailTitle='Half the squared residual is this point’s loss';
 }
 if(kind==='batch'||kind==='batch-loss'){
  const c=chart(plot,{xmin:-1,xmax:1,topInset:projected()?28:18}),count=kind==='batch'?Math.min(64,Math.floor(u*64)+1):1+Math.floor(clamp(u/.72)*63),normalized=u>=.84,selected=kind==='batch'?first.indices.slice(0,count):first.indices;
  let body=c.base+c.step()+observations(c,{batch:selected})+c.path(x=>predict(first.before,x),C.blue);
  if(kind==='batch-loss'){
   body+=text(c.left,projected()?20:13,'Parameters fixed','small');
   first.indices.slice(0,count).forEach(i=>{const d=data[i];body+=line(c.X(d.x),c.Y(d.y),c.X(d.x),c.Y(predict(first.before,d.x)),C.rust,'',1.4);});
   const d=data[first.indices[0]];body+=circle(c.X(d.x),c.Y(d.y),6,'none',C.ink,1.3);
  }
  if(kind==='batch'){
   const keyY=projected()?15:9,keyGap=projected()?164:112;
   body+=circle(c.left+3,keyY,3.3,C.rust,'white',.6)+text(c.left+13,keyY+4,'Selected batch','small','start',C.rust);
   body+=circle(c.left+keyGap,keyY,2,'#b4b4b4')+text(c.left+keyGap+10,keyY+4,'Other observations','small');
  }
  set(plot,body,`${scene.title} ${scene.reading??scene.caption}`);
  if(kind==='batch')batchFacts(detail,first.indices,count);else info={...info,...squaredErrors(detail,residuals,count,normalized)};
  plotTitle=kind==='batch'?'Take the first 64 shuffled observations':normalized?'Divide the sum by the batch size':`Add the losses from all 64 observations · ${count} of 64`;
  detailTitle=kind==='batch'?'Batch 1 of 32 · no replacement':normalized?'The sum becomes the mean batch loss':'Each bar is half a squared residual';
  info={...info,batch:1,selected:count,batchFirst:first.indices[0],batchHalfMSE:first.batchBefore,batchMSE:2*first.batchBefore};
 }
 if(kind==='batch-gradient'){
  gradientSpaces(plot,first.indices,residuals,first.gradient);const meanResidual=gradientFacts(detail,residuals,first.gradient);
  plotTitle='One residual per observation; one direction per parameter';detailTitle=String.raw`Example: \(\phi_0\) changes every prediction equally`;
  info={...info,residuals,residualCount:residuals.length,parameterCount:flatten(first.gradient).length,meanResidual,outputBiasGradient:first.gradient.c,gradientC:first.gradient.c,gradient:flatten(first.gradient),model:first.before};
 }
 if(kind==='parameter-step'){
  const c=biasStep(plot,first,eta,u);stepArithmetic(detail,first,eta);
  plotTitle=String.raw`Zoom in on \(\phi_0\): one component of the same update`;detailTitle='Preview the arithmetic; then apply all seven together';
  info={...info,eta,gradientC:first.gradient.c,outputBias:first.before.c,previewOutputBias:c,deltaC:first.after.c-first.before.c,model:first.before,illustrationOnly:true};
 }
 if(kind==='next-batch'){
  const second=e.traces[2][1],g2=gradient(second.model,second.indices.map(i=>data[i])),c=chart(plot,{xmin:-1,xmax:1}),updateProgress=smooth((u-.5)/.5),model=interpolate(second.model,second.after,updateProgress);
  let body=c.base+c.step()+observations(c)+c.path(x=>predict(second.model,x),C.gray,2,'4 4')+c.path(x=>predict(model,x),C.blue);
  const oldAlpha=1-smooth(u/.3),newAlpha=smooth(u/.3);
  [first.indices,second.indices].forEach((indices,j)=>{
   const alpha=j?newAlpha:oldAlpha;
   body+=`<g opacity="${alpha}">`;
   indices.forEach(i=>{const d=data[i];if(j)body+=line(c.X(d.x),c.Y(d.y),c.X(d.x),c.Y(predict(model,d.x)),C.rust,'',1);body+=circle(c.X(d.x),c.Y(d.y),3.3,C.rust,'white',.7);});
   body+='</g>';
  });
  set(plot,body,`The next 64 shuffled observations replace the first batch with no repeats. Compute their loss and gradient at the model after update 1, then apply update 2. ${updateProgress===1?'The second update is complete.':updateProgress>0?'The curve is moving through the second update.':'The weights stay fixed while the new gradient is calculated.'}`);
  nextBatchFacts(detail,second,model,u,updateProgress);
  plotTitle=updateProgress===1?'Two batches used; two updates complete':u<.15?'Select the next 64 unused observations':u<.3?'Evaluate this batch’s loss at the updated weights':u<.5?'Find the gradient of this batch loss':'Apply the second update';detailTitle='Continue from the weights reached by the first update';
  info={...info,model,batch:2,batchFirst:second.indices[0],outputBias:model.c,gradientC:u<.3?null:g2.c,batchOverlap:first.indices.filter(id=>second.indices.includes(id)).length,batchHalfMSE:loss(model,second.indices.map(i=>data[i])),batchHalfMSEBefore:loss(second.model,second.indices.map(i=>data[i])),batchHalfMSEAfter:loss(second.after,second.indices.map(i=>data[i])),batchMSE:2*loss(model,second.indices.map(i=>data[i])),batchMSEBefore:2*loss(second.model,second.indices.map(i=>data[i])),batchMSEAfter:2*loss(second.after,second.indices.map(i=>data[i])),updateProgress,completedUpdates:updateProgress===1?2:1,procedureStep:u<.15?0:u<.3?1:u<.5?2:3};
 }
 return {...info,plotTitle,detailTitle};
}
