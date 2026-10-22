import {constructionKinds,renderConstruction} from './construction.mjs';
import {predict,ramp,gradient,data,loss} from './model.mjs';
const C={blue:'#234e70',rust:'#a54f32',ink:'#242424',muted:'#606060',teal:'#007f73',gray:'#aaa'};
const clamp=x=>Math.max(0,Math.min(1,x));
const fmt=x=>Math.abs(x)<1e-10?'0':Number(x.toFixed(3)).toString();
const line=(x1,y1,x2,y2,color=C.ink,dash='',width=1)=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}" ${dash?`stroke-dasharray="${dash}"`:''}/>`;
const text=(x,y,t,cls='',anchor='start',color='')=>`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}" ${color?`style="fill:${color}"`:''}>${t}</text>`;
function set(svg,body,label){svg.innerHTML=body;svg.setAttribute('aria-label',label);}
function canvas(svg){const r=svg.getBoundingClientRect();const w=Math.max(260,r.width),h=Math.max(70,r.height);svg.setAttribute('viewBox',`0 0 ${w} ${h}`);return {w,h};}
function chart(svg,{xmin=-1,xmax=1.2,ymin=-.25,ymax=1.4,xticks=[-1,0,1],yticks=[0,.5,1]}={}){
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
function facts(svg,entries){const {w,h}=canvas(svg);let html='';if(h<150){const cols=entries.length;entries.forEach((e,i)=>{const x=6+i*w/cols;html+=text(x,18,e[0],'small')+text(x,47,e[1],cols>2?'':'large')+(e[2]&&cols<3?text(x,68,e[2],'small'):'');});}else{const space=Math.min(100,(h-20)/entries.length);entries.forEach((e,i)=>{const y=24+i*space;html+=text(10,y,e[0],'small')+text(10,y+37,e[1],'large')+(e[2]?text(10,y+59,e[2],'small'):'');});}set(svg,html,entries.map(e=>e.join(': ')).join('. '));}
function observations(c,{batch=null,residualModel=null,amount=1}={}){
 // Dense observations share one SVG path; each highlighted batch remains explicit.
 let dots='',residuals='',selected='';const active=new Set(batch||[]);
 data.forEach((d,i)=>{if(d.x<c.xmin||d.x>c.xmax)return;const x=c.X(d.x),y=c.Y(d.y);
 dots+=`M${x.toFixed(2)},${y.toFixed(2)}v.1`;
 if(residualModel&&i<data.length*amount&&(!batch||active.has(i)))residuals+=`M${x},${y}V${c.Y(predict(residualModel,d.x))}`;
 if(active.has(i))selected+=`<circle cx="${x}" cy="${y}" r="3.3" fill="${C.rust}" stroke="white" stroke-width=".6"/>`;
 });
 return `<path d="${dots}" fill="none" stroke="#b4b4b4" stroke-width="3" stroke-linecap="round"/><path d="${residuals}" fill="none" stroke="${C.rust}" stroke-opacity="${batch?.65:.10}" stroke-width="1"/>`+selected;
}
function gradientBars(svg,g1,g2=null){const {w,h}=canvas(svg),vals=g1.units.flatMap(u=>[u.w,u.b,u.a]).concat(g1.c),other=g2?g2.units.flatMap(u=>[u.w,u.b,u.a]).concat(g2.c):null,labels=['w₁','b₁','a₁','w₂','b₂','a₂','c'];
 const top=18,bottom=h-24,mid=(top+bottom)/2,scale=(bottom-top)/2/Math.max(.02,...vals.map(Math.abs),...(other||[]).map(Math.abs)),col=(w-30)/7;
 let body=line(10,mid,w-10,mid,'#999');vals.forEach((v,i)=>{const x=20+col*i;body+=line(x,mid,x,mid-v*scale,C.blue,'',5);if(other)body+=line(x+7,mid,x+7,mid-other[i]*scale,C.rust,'',5);body+=text(x+2,bottom+19,labels[i],'small','middle');});
 body+=text(10,10,other?'Blue: full data · rust: minibatch':'Gradient components (signed)','small');set(svg,body,'Gradient components for the seven trainable parameters. '+(other?'Full gradient in blue, minibatch estimate in rust.':''));
}
function interpolate(a,b,t){return {c:a.c+(b.c-a.c)*t,units:a.units.map((v,i)=>Object.fromEntries(['w','b','a'].map(k=>[k,v[k]+(b.units[i][k]-v[k])*t])))};}
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
 const c=chart(plot,{xmin:-1,xmax:1});
 let body=c.base+c.step(),detailTitle='Construction',plotTitle='Function',info={kind,progress:p};
 if(kind==='steepen'){
  const a=2*16**u;body+=c.path(x=>ramp(x,a),C.teal);
  body+=line(c.X(0),c.Y(-.12),c.X(1/a),c.Y(-.12),C.rust,'',2);
  facts(detail,[['Steepness a',a.toFixed(1),'Width stays at 2 neurons'],['Ramp width',(1/a).toFixed(3),'The second hinge is at 1/a']]);info.a=a;
 }
 if(['initialize','loss','full','batch','update','initialize-wide'].includes(kind)){
  const wide=kind==='initialize-wide',color=wide?C.teal:C.blue;
  let model=wide?e.traces[12][0].model:e.firstUpdate.before;detailTitle='Training';
  if(kind==='update'){model=interpolate(e.firstUpdate.before,e.firstUpdate.after,u);body+=c.path(x=>predict(e.firstUpdate.before,x),C.gray,2,'4 4');}
  body+=observations(c,{batch:['batch','update'].includes(kind)?e.firstUpdate.indices:null,residualModel:['loss','batch'].includes(kind)?model:null,amount:kind==='loss'?u:1})+c.path(x=>predict(model,x),color);
  if(kind==='initialize'||wide)facts(detail,[['Hidden neurons',wide?'12':'2','One hidden layer'],['Trainable parameters',wide?'37':'7','All weights and biases are learned']]);
  if(kind==='loss')facts(detail,[['Training MSE',e.traces[2][0].loss.toFixed(4),'All 2,048 observations'],['Residuals shown',String(Math.ceil(2048*u)),'Squared, then averaged']]);
  if(kind==='full'){gradientBars(detail,gradient(model,data));detailTitle='Backpropagated gradient';}
  if(kind==='batch'){gradientBars(detail,gradient(model,data),e.firstUpdate.gradient);detailTitle='Full gradient and batch gradient';}
  if(kind==='update')facts(detail,[['Learning rate','0.025','All seven parameters update'],['Batch MSE',loss(model,e.firstUpdate.indices.map(i=>data[i])).toFixed(4),'One actual SGD update']]);
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
