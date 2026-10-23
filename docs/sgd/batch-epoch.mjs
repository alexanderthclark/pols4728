import {C,clamp,line,text,set,canvas} from './sgd-detail.mjs';

// A deliberately small teaching example, separate from the recorded experiment.
// IDs remain attached to the same observations throughout every transformation.
export const foundationKinds=['sgd-transition','dataset-intro','batch-intro','batch-pass','epoch-complete','epoch-repeat','batch-scale'];
const ids=Array.from({length:12},(_,i)=>i+1);
const nextOrder=[8,2,11,5,1,9,4,12,6,3,10,7];
const lerp=(a,b,t)=>a+(b-a)*t;
const smooth=t=>{t=clamp(t);return t*t*(3-2*t);};
const number=(x,y,value,size=14,color=C.ink)=>`<text x="${x}" y="${y}" text-anchor="middle" style="font:${size}px Arial,sans-serif;fill:${color}">${value}</text>`;
const check=(x,y,size=7,color=C.teal)=>`<path d="M${x-size*.7},${y}l${size*.5},${size*.5}l${size},${-size}" fill="none" stroke="${color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>`;

function geometry(svg){
 const {w,h:rawHeight}=canvas(svg),h=Math.max(200,rawHeight),mobile=w<600;
 // The parent supplies at least 200 px for the foundation diagram. Retaining
 // that coordinate height also makes a smaller transient resize safe to render.
 svg.setAttribute('viewBox',`0 0 ${w} ${h}`);
 const gap=mobile?9:Math.min(17,Math.max(8,(h-140)*.08));
 const size=mobile?Math.min(31,(w-118)/4):Math.min(54,(h-114-2*gap)/3);
 const gridWidth=4*size+3*gap,gridHeight=3*size+2*gap;
 const gx=(w-gridWidth)/2,gy=mobile?32+(h-200)*.38:44+(h-114-gridHeight)/2;
 const grid=ids.map((_,i)=>({x:gx+(i%4)*(size+gap),y:gy+Math.floor(i/4)*(size+gap)}));
 let groups,positions;
 if(mobile){
  const boxHeight=Math.min(46,(h-72)/3),boxGap=Math.min(10,(h-3*boxHeight-60)/2),startY=31;
  groups=Array.from({length:3},(_,i)=>({x:7,y:startY+i*(boxHeight+boxGap),w:w-14,h:boxHeight}));
  const tileSize=Math.min(31,boxHeight-10),tileGap=Math.min(8,(w-115-4*tileSize)/3),startX=(w-4*tileSize-3*tileGap)/2+12;
  positions=ids.map((_,i)=>({x:startX+(i%4)*(tileSize+tileGap),y:groups[Math.floor(i/4)].y+(boxHeight-tileSize)/2,size:tileSize}));
 }else{
  const overall=Math.min(w-42,930),boxGap=Math.min(36,overall*.045),boxWidth=(overall-2*boxGap)/3,boxHeight=Math.min(196,h-105),startX=(w-overall)/2,startY=(h-boxHeight)/2-3;
  groups=Array.from({length:3},(_,i)=>({x:startX+i*(boxWidth+boxGap),y:startY,w:boxWidth,h:boxHeight}));
  const tileSize=Math.min(54,(boxHeight-65)/2),tileGap=13;
  positions=ids.map((_,i)=>{const b=groups[Math.floor(i/4)],slot=i%4;return {x:b.x+(b.w-2*tileSize-tileGap)/2+(slot%2)*(tileSize+tileGap),y:b.y+35+Math.floor(slot/2)*(tileSize+tileGap),size:tileSize};});
 }
 return {w,h,mobile,size,grid,groups,positions,gx,gy,gridWidth,gridHeight};
}

function tile(id,x,y,size,{used=false,active=false,opacity=1}={}){
 const stroke=used?C.teal:active?C.blue:'#aeb9c1',fill=used?'#edf5f3':active?'#f0f4f7':'white';
 return `<g data-observation="${id}" opacity="${opacity}" transform="translate(${x.toFixed(3)} ${y.toFixed(3)})"><rect width="${size}" height="${size}" rx="3" fill="${fill}" stroke="${stroke}" stroke-width="${active?1.6:1.1}"/>${number(size/2,size/2+(size<35?4:7),id,size<35?14:22,used?C.teal:C.ink)}</g>`;
}

function groupFrames(g,{completed=0,current=-1,opacity=1,fraction=0}={}){
 let body=`<g opacity="${opacity}">`;
 g.groups.forEach((b,i)=>{
  const done=i<completed,active=i===current,color=done?C.teal:active?C.blue:'#b7bfc4';
  body+=`<g data-batch="${i+1}" data-state="${done?'complete':active?'current':'waiting'}"><rect x="${b.x}" y="${b.y}" width="${b.w}" height="${b.h}" rx="5" fill="none" stroke="${color}" stroke-width="${active?2:1.2}" ${done||active?'':'stroke-dasharray="4 4"'}/>`;
  if(g.mobile){
   body+=text(b.x+9,b.y+b.h/2-3,'Batch','small','start',color)+text(b.x+9,b.y+b.h/2+11,String(i+1),'','start',color);
   if(done)body+=check(b.x+b.w-12,b.y+b.h/2,6);
  }else{
   body+=text(b.x+13,b.y+22,`Batch ${i+1}`,'','start',color);
   if(done)body+=check(b.x+b.w-18,b.y+17,7);
  }
  if(active&&fraction>0)body+=line(b.x+7,b.y+b.h-4,b.x+7+(b.w-14)*fraction,b.y+b.h-4,C.blue,'',2);
  body+='</g>';
 });
 return body+'</g>';
}

function tiles(g,{order=ids,fromOrder=order,grouping=1,motion=1,completed=0,current=-1}={}){
 const destination=new Map(order.map((id,i)=>[id,i])),origin=new Map(fromOrder.map((id,i)=>[id,i]));
 return ids.map(id=>{
  const slot=destination.get(id),from=g.positions[origin.get(id)],to=g.positions[slot],base=g.grid[id-1];
  const groupX=lerp(from.x,to.x,motion),groupY=lerp(from.y,to.y,motion),x=lerp(base.x,groupX,grouping),y=lerp(base.y,groupY,grouping),size=lerp(g.size,to.size,grouping);
  return tile(id,x,y,size,{used:Math.floor(slot/4)<completed,active:Math.floor(slot/4)===current});
 }).join('');
}

function footer(g,first,second='',color=C.ink){
 const firstY=g.h-(second?(g.mobile?23:36):(g.mobile?8:12)),secondY=g.h-(g.mobile?5:9);
 return text(g.w/2,firstY,first,'','middle',color)+(second?text(g.w/2,secondY,second,'small','middle'):'');
}

function datasetDiagram(svg,kind,u){
 const g=geometry(svg),{w,h,mobile}=g;
 let body=text(w/2,mobile?15:20,'Small teaching example · 12 observations','small','middle');
 body+=`<rect x="${g.gx-13}" y="${g.gy-12}" width="${g.gridWidth+26}" height="${g.gridHeight+24}" rx="6" fill="none" stroke="#b7bfc4" stroke-width="1.2"/>`;
 body+=tiles(g,{grouping:0,current:-1});
 if(kind==='dataset-intro'){
  const first=g.grid[0],amount=smooth(u);
  body+=`<rect x="${first.x-3}" y="${first.y-3}" width="${g.size+6}" height="${g.size+6}" rx="5" fill="none" stroke="${C.blue}" stroke-width="2" opacity="${amount}"/>`;
  body+=footer(g,'Each tile is one observation.','The same number follows the same observation.');
 }else{
  body+=footer(g,mobile?'Dataset → batches → epoch':'One dataset → small batches → one complete pass');
 }
 set(svg,body,kind==='dataset-intro'?'Small teaching example with twelve observations, numbered one through twelve. Each numbered tile is one observation; its number keeps its identity visible.':'Batches and epochs: start with one dataset, work through small batches, and complete a pass through every observation. The diagram uses a small example of twelve observations.');
 return {example:true,observationCount:12,batchSize:4,batchCount:3,ids:[...ids],order:[...ids],epoch:0,completedEpochs:0,used:0,completedBatches:0,currentBatch:0};
}

function groupingDiagram(svg,u){
 const g=geometry(svg),t=smooth(u),visible=smooth((u-.25)/.5);
 let body=text(g.w/2,g.mobile?15:20,'Batch size = 4 observations','small','middle');
 body+=groupFrames(g,{current:0,opacity:visible})+tiles(g,{grouping:t,current:u>.6?0:-1});
 body+=footer(g,'One batch = 4 observations used together.');
 set(svg,body,'The twelve observations separate into three batches of four. Batch one contains observations one through four and has a solid outline. The other two batches contain five through eight and nine through twelve. A batch is the group used together.');
 return {example:true,observationCount:12,batchSize:4,batchCount:3,ids:[...ids],order:[...ids],grouping:t,epoch:1,completedEpochs:0,used:0,completedBatches:0,currentBatch:1};
}

function passDiagram(svg,u,complete=false){
 const g=geometry(svg),position=complete?3:u*3,completed=Math.min(3,Math.floor(position+1e-9)),current=completed<3?completed:-1,fraction=current<0?0:position-completed;
 let body=text(g.w/2,g.mobile?15:20,complete?'Every observation used once':current<0?'All three batches have been used':`Use batch ${current+1}, then move to the next`,'small','middle');
 body+=groupFrames(g,{completed,current,fraction})+tiles(g,{completed,current});
 body+=complete?footer(g,'1 complete pass = 1 epoch','12 / 12 observations · 3 / 3 batches',C.teal):footer(g,`${completed*4} / 12 observations used`,`${completed} / 3 batches complete`);
 set(svg,body,complete?'All three batches are complete and checked. All twelve observations have been used exactly once. One complete pass through the dataset is one epoch.':`${completed} of three batches complete, so ${completed*4} of twelve observations have been used. ${current<0?'The complete pass is finished.':`Currently using batch ${current+1}; completed batches are checked.`}`);
 return {example:true,observationCount:12,batchSize:4,batchCount:3,ids:[...ids],order:[...ids],epoch:1,completedEpochs:completed===3?1:0,used:completed*4,completedBatches:completed,currentBatch:current+1,batchProgress:fraction};
}

function repeatDiagram(svg,u){
 const g=geometry(svg),started=u>.06,motion=smooth((u-.12)/.72),completed=started?0:3,current=motion===1?0:-1;
 let body=text(g.w/2,g.mobile?15:20,started?'Epoch 2 · reshuffle the same data':'Epoch 1 · one complete pass','small','middle');
 body+=groupFrames(g,{completed,current})+tiles(g,{order:nextOrder,fromOrder:ids,motion,completed,current});
 body+=started?footer(g,'0 / 12 used · 0 / 3 batches complete','New order. Keep what the model learned.'):footer(g,'12 / 12 used · 3 / 3 batches complete','Ready for another pass through the same data.');
 set(svg,body,started?'Epoch two begins. The same twelve numbered observations move to a new shuffled order and form three new batches of four. No observation is added or removed. The count used in this epoch resets to zero, while the model keeps what it learned.':'Epoch one is complete. All twelve observations have been used and all three batches are checked. The next scroll movement begins a second pass through the same dataset.');
 return {example:true,observationCount:12,batchSize:4,batchCount:3,ids:[...ids],order:motion===1?[...nextOrder]:[...ids],targetOrder:[...nextOrder],shuffleProgress:motion,epoch:started?2:1,completedEpochs:1,used:started?0:12,completedBatches:completed,currentBatch:current+1};
}

function scaleDiagram(svg,u){
 const {w,h,mobile}=geometry(svg),short=!mobile&&h<300,cols=mobile?4:8,rows=32/cols,margin=mobile?8:Math.max(20,w*.1),left=margin,right=w-margin,top=mobile?39:short?40:72,bottom=h-(mobile?42:60),gap=mobile?3:short?6:12,cellW=(right-left-(cols-1)*gap)/cols,cellH=(bottom-top-(rows-1)*gap)/rows,count=Math.min(32,Math.floor(u*32)+1);
 let body=short?text(w/2,20,'2,048 observations · 64 in each batch','small','middle'):text(w/2,mobile?15:20,'2,048 observations in the full dataset','small','middle')+text(w/2,mobile?32:46,'Each box is a batch of 64','small','middle');
 for(let i=0;i<32;i++){
  const x=left+(i%cols)*(cellW+gap),y=top+Math.floor(i/cols)*(cellH+gap),shown=i<count;
  body+=`<g data-batch="${i+1}" opacity="${shown?1:.16}"><rect x="${x}" y="${y}" width="${cellW}" height="${cellH}" rx="2" fill="${i===0?'#f0f4f7':'white'}" stroke="${i===0?C.blue:'#9fadb7'}" stroke-width="${i===0?1.8:1}"/>${number(x+cellW/2,y+cellH/2+(mobile?3:cellH<26?5:7),'64',mobile?10:cellH<26?16:22,i===0?C.blue:C.muted)}</g>`;
 }
 body+=footer({w,h,mobile},'32 batches × 64 = 2,048','Use all 32 batches → 1 epoch');
 set(svg,body,'The actual training dataset has 2,048 observations. With batch size 64, it divides exactly into 32 batches, shown as 32 boxes. Each box is a batch of 64 observations. Using all 32 batches completes one epoch.');
 return {example:false,observationCount:2048,batchSize:64,batchCount:32,shownBatches:count,epoch:0,completedEpochs:0,used:0,completedBatches:0,currentBatch:0};
}

export function renderFoundation(plot,detail,scene,p,e){
 const u=clamp((p-.1)/.7),kind=scene.kind;
 let info;
 if(kind==='sgd-transition'||kind==='dataset-intro')info=datasetDiagram(plot,kind,u);
 else if(kind==='batch-intro')info=groupingDiagram(plot,u);
 else if(kind==='batch-pass')info=passDiagram(plot,u);
 else if(kind==='epoch-complete')info=passDiagram(plot,u,true);
 else if(kind==='epoch-repeat')info=repeatDiagram(plot,u);
 else if(kind==='batch-scale')info=scaleDiagram(plot,u);
 else throw new Error(`Unknown batch-and-epoch scene: ${kind}`);
 set(detail,'','The batch and epoch explanation is contained in the main diagram.');
 const plotTitles={'sgd-transition':'Batches and epochs','dataset-intro':'One dataset, twelve observations','batch-intro':'Divide the observations into batches','batch-pass':'Use every batch once','epoch-complete':'One full pass through the dataset','epoch-repeat':'Another epoch uses the same observations','batch-scale':'The same idea at the experiment’s scale'};
 return {kind,progress:p,...info,plotTitle:plotTitles[kind],detailTitle:''};
}
