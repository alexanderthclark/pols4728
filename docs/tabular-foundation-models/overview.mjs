// Original schematics. Geometry follows scroll position in either direction;
// input numbers are illustrative, and no numerical model outputs are invented.
const C={ink:'#242424',blue:'#234E70',rust:'#A54F32',teal:'#007F73',muted:'#606060',rule:'#767676',paper:'#FFFFFF',pale:'#E8EFF4'};
const clamp=x=>Math.max(0,Math.min(1,Number.isFinite(x)?x:1));
const mix=(a,b,t)=>a+(b-a)*t;
const smooth=x=>{const t=clamp(x);return t*t*(3-2*t);};
const f=x=>Number(x.toFixed(2));
const esc=x=>String(x).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const text=(x,y,label,size=20,color=C.ink,anchor='middle',extra='')=>`<text x="${f(x)}" y="${f(y)}" text-anchor="${anchor}" font-size="${size}" fill="${color}" ${extra}>${esc(label)}</text>`;
const rect=(x,y,w,h,color=C.blue,fill=C.paper,width=1.5,extra='')=>`<rect x="${f(x)}" y="${f(y)}" width="${f(w)}" height="${f(h)}" rx="2" fill="${fill}" stroke="${color}" stroke-width="${width}" ${extra}/>`;
const line=(x1,y1,x2,y2,color=C.blue,width=1.6,extra='')=>`<line x1="${f(x1)}" y1="${f(y1)}" x2="${f(x2)}" y2="${f(y2)}" stroke="${color}" stroke-width="${width}" ${extra}/>`;
function arrow(x1,y1,x2,y2,color=C.blue,width=1.7){
  const a=Math.atan2(y2-y1,x2-x1), head=6;
  return line(x1,y1,x2,y2,color,width)+`<path d="M${f(x2-head*Math.cos(a-.55))},${f(y2-head*Math.sin(a-.55))} L${f(x2)},${f(y2)} L${f(x2-head*Math.cos(a+.55))},${f(y2-head*Math.sin(a+.55))}" fill="none" stroke="${color}" stroke-width="${width}"/>`;
}
function packet(x1,y1,x2,y2,p,color=C.teal,size=7){
  const t=smooth(p);
  return rect(mix(x1,x2,t)-size/2,mix(y1,y2,t)-size/2,size,size,color,color,1,'data-moving-input="true"');
}
function routedPacket(points,p,color=C.teal,size=7){
  const segments=points.slice(1).map((point,i)=>Math.hypot(point[0]-points[i][0],point[1]-points[i][1]));
  const length=segments.reduce((sum,n)=>sum+n,0);
  let distance=length*smooth(p), segment=0;
  while(segment<segments.length-1&&distance>segments[segment])distance-=segments[segment++];
  const start=points[segment],end=points[segment+1],t=segments[segment]?distance/segments[segment]:1;
  return rect(mix(start[0],end[0],t)-size/2,mix(start[1],end[1],t)-size/2,size,size,color,color,1,'data-moving-input="true"');
}
function modelFrame(mobile,stage){
  const b=mobile?{x:62,y:202,w:276,h:90}:{x:350,y:132,w:365,h:190};
  let s=rect(b.x,b.y,b.w,b.h,C.blue,C.paper,1.8);
  s+=text(b.x+b.w/2,mobile?187:109,stage===0?'Shared network':'Pretrained network',mobile?21:25,C.blue);
  const status=stage===0?'Weights learned here':'Weights fixed';
  s+=text(b.x+b.w/2,mobile?222:305,status,mobile?16:18,stage===0?C.rust:C.blue,'middle',`data-weight-status="${stage===0?'learning':'fixed'}"`);
  return {s,...b};
}
function network(mobile,p,learning=false){
  const xs=mobile?[111,200,289]:[410,532,654];
  const ys=mobile?[239,253,267]:[177,218,259];
  const radius=mobile?3.5:6;
  let s='';
  for(let k=0;k<2;k++)for(let i=0;i<3;i++)for(let j=0;j<3;j++){
    const strength=learning?mix(.15,.35+((i+j+k)%3)*.23,smooth(p)):.42+((i+j+k)%3)*.16;
    const width=learning?mix(.7,1+((i+2*j+k)%3)*.65,smooth(p)):1.15;
    s+=line(xs[k],ys[i],xs[k+1],ys[j],C.blue,width,`opacity="${f(strength)}"`);
  }
  xs.forEach((x,k)=>ys.forEach(y=>{s+=`<circle cx="${x}" cy="${y}" r="${radius}" fill="${k===2?C.teal:C.blue}"/>`;}));
  return s;
}
function miniTask(x,y,w,h,index,mobile){
  const cols=3,rows=4,cw=w/cols,rh=(h-20)/rows;
  let s=text(x+w/2,y-9,`Task ${index+1}`,mobile?16:19,C.muted);
  s+=rect(x,y,w,h,C.rule,C.paper,1);
  for(let col=1;col<cols;col++)s+=line(x+cw*col,y,x+cw*col,y+h,C.rule,.8);
  for(let row=0;row<=rows;row++)s+=line(x,y+20+rh*row,x+w,y+20+rh*row,C.rule,.8);
  for(let row=0;row<rows;row++)for(let col=0;col<cols;col++){
    const pad=mobile?5:7;
    const color=col===2?C.teal:C.blue;
    const valueWidth=cw-2*pad-(row+col+index)%2*(mobile?5:8);
    s+=rect(x+cw*col+pad,y+20+rh*row+rh*.4,valueWidth,Math.max(3,rh*.2),color,color,0);
  }
  // A withheld target is present in every pretraining prediction task.
  s+=rect(x+2*cw+1,y+h-rh+1,cw-2,rh-2,C.paper,C.paper,0);
  s+=text(x+2.5*cw,y+h-rh/2+5,'?',mobile?16:20,C.rust);
  return s;
}
function pretraining(p,mobile){
  let s='';
  if(mobile){
    s+=text(200,24,'Pretraining tables used earlier',20,C.ink);
    [16,151,286].forEach((x,i)=>{
      s+=miniTask(x,63,98,89,i,true);
      const route=i===1?[[200,158],[200,166],[326,166],[326,198]]:[[x+49,158],[i===0?86:312,198]];
      if(i===1)s+=`<path d="M200,158 V166 H326 V198" fill="none" stroke="${C.blue}" stroke-width="1.3"/>`+arrow(326,188,326,198,C.blue,1.3);
      else s+=arrow(...route[0],...route[1],C.blue,1.3);
      s+=routedPacket(route,clamp(p*1.6-i*.26),C.teal,6);
    });
  }else{
    s+=text(180,45,'Pretraining tables used earlier',21,C.ink);
    [94,211,328].forEach((yy,i)=>{
      const x=40;
      s+=miniTask(x,yy,130,88,i,false);
      const endY=190+i*33;
      s+=arrow(x+136,yy+44,346,endY,C.blue,1.3);
      s+=packet(x+136,yy+44,346,endY,clamp(p*1.6-i*.26),C.teal,7);
    });
  }
  s+=modelFrame(mobile,0).s+network(mobile,p,true);
  if(mobile){
    s+=arrow(200,297,200,315,C.teal);
    s+=text(200,337,'One set of learned weights',20,C.teal);
  }else{
    s+=arrow(722,226,776,226,C.teal);
    s+=text(830,214,'Ready for',23,C.teal)+text(830,246,'a new table',23,C.teal);
    s+=text(532,374,'One set of learned weights',24,C.blue);
  }
  return s;
}

const labeledRows=[['.2','.8','0'],['.4','.1','1'],['.8','.3','1'],['.7','.9','0']];
const queryValues=['.6','.2','?'];
function promptTable(x,y,w,rh,mobile,{small=false}={}){
  const cw=w/3,head=rh,fs=small?(mobile?14:17):(mobile?18:23);
  let s=`<g data-overview-input="whole-table">`+rect(x,y,w,6*rh,C.blue,C.paper,1.5);
  for(let j=1;j<3;j++)s+=line(x+j*cw,y,x+j*cw,y+6*rh,C.rule,.8);
  ['X₁','X₂',small&&mobile?'Y':'Target'].forEach((v,j)=>{s+=text(x+(j+.5)*cw,y+rh*.7,v,fs,C.muted);});
  const centers=[];
  labeledRows.forEach((row,i)=>{const yy=y+head+i*rh;s+=line(x,yy,x+w,yy,C.rule,.8);const points=[];row.forEach((v,j)=>{const xx=x+(j+.5)*cw,cy=yy+rh*.52;points.push([xx,cy]);s+=text(xx,yy+rh*.72,v,fs,j===2?C.teal:C.ink,'middle',j===2?'data-context-target="true"':'');});centers.push(points);});
  const qy=y+5*rh;s+=rect(x,qy,w,rh,C.blue,C.pale,1.5);s+=`<g data-overview-query="true">`;
  queryValues.forEach((v,j)=>{const xx=x+(j+.5)*cw;s+=text(xx,qy+rh*.72,v,fs,j===2?C.rust:C.ink,'middle',j===2?'data-query-target="hidden"':'data-query-feature="true"');});
  s+='</g></g>';centers.push(queryValues.map((v,j)=>[x+(j+.5)*cw,qy+rh*.52]));
  return {s,centers,bottom:y+6*rh,cw,queryY:qy};
}
function predictor(x,y,w,h,p,mobile,caption='Weights fixed'){
  let s=rect(x,y,w,h,C.blue,C.paper,1.7);const xs=[x+w*.19,x+w*.5,x+w*.81],ys=[y+h*.2,y+h*.37,y+h*.54];
  for(let k=0;k<2;k++)for(let i=0;i<3;i++)for(let j=0;j<3;j++)s+=line(xs[k],ys[i],xs[k+1],ys[j],C.blue,1.1,`opacity="${.3+(i+j)%3*.12}"`);
  xs.forEach((xx,k)=>ys.forEach(yy=>{s+=`<circle cx="${f(xx)}" cy="${f(yy)}" r="${mobile?3.2:5}" fill="${k===2?C.teal:C.blue}"/>`;}));
  return s+text(x+w/2,y+h*.88,caption,mobile?16:21,C.blue,'middle','data-weight-status="fixed"');
}
function inference(p,mobile){
  const t=mobile?promptTable(22,43,356,24,true):promptTable(28,118,280,34,false);
  let s=text(mobile?200:168,mobile?22:58,'Current task’s input',mobile?20:24,C.blue)+t.s;
  if(mobile){
    s+=text(200,203,'Known labels are inputs; the last target is hidden.',15,C.muted);
    s+=text(200,229,'Pretrained model',21,C.blue)+predictor(65,240,270,66,p,true);
    s+=`<path d="M200,192 H40 V272 H61" fill="none" stroke="${C.blue}" stroke-width="1.5"/>`+arrow(49,272,61,272)+routedPacket([[200,192],[40,192],[40,272],[61,272]],p,C.teal,6);
    s+=arrow(200,311,200,326,C.teal)+packet(200,312,200,323,p,C.teal,5)+`<g data-overview-output="schematic">`+text(200,346,'Prediction for the new row',18,C.teal,'middle','data-schematic-output="true"')+'</g>';
  }else{
    s+=text(168,91,'Features, labels, and query features',17,C.muted);
    s+=text(168,357,'Last row: new case; target unknown',20,C.muted);
    s+=text(558,105,'Pretrained model',26,C.blue)+text(558,136,'Earlier tables shaped these weights',19,C.muted);
    s+=predictor(430,159,255,170,p,false)+arrow(317,221,422,221)+packet(319,221,418,221,p,C.teal,8);
    s+=arrow(693,237,756,237,C.teal)+packet(695,237,752,237,p,C.teal,7);
    s+=`<g data-overview-output="schematic">`+text(813,216,'Prediction',24,C.teal,'middle','data-schematic-output="true"')+text(813,249,'for the',21,C.teal)+text(813,280,'new row',21,C.teal)+'</g>';
    s+=text(450,420,'Pretraining tables are not supplied again.',22,C.muted,'middle','data-pretraining-resupplied="false"');
  }
  return s;
}
function vectorGlyph(x,y,w,h,row,col,p,extra=''){
  const target=row===4&&col===2,color=col===2?(target?C.rust:C.teal):C.blue;
  let s=`<g ${extra}>`+rect(x,y,w,h,color,C.paper,target?2:1,target?'stroke-dasharray="3 2"':'');
  for(let k=0;k<3;k++){const len=w*(.24+((row+col+k)%3)*.12)+p*(target?4:2)*(k-1);s+=line(x+w*.15,y+h*(.28+k*.21),x+w*.15+len,y+h*(.28+k*.21),color,mobileSafeWidth(w));}
  return s+'</g>';
}
const mobileSafeWidth=w=>w<50?1.8:2.6;
function embeddedCube(x,y,w,h,depth,p,mobile){
  const cw=w/3,rh=h/5;let s=`<g data-table-tensor="true" data-vector-depth="schematic">`;
  // Offset front/back grids provide the third (embedding-vector) dimension.
  s+=`<path d="M${x},${y} L${x+depth},${y-depth} H${x+w+depth} V${y+h-depth} L${x+w},${y+h} V${y} Z" fill="${C.pale}" stroke="${C.rule}" stroke-width="1"/>`;
  for(let j=0;j<=3;j++)s+=line(x+j*cw,y,x+j*cw+depth,y-depth,C.rule,.8);
  for(let i=0;i<=5;i++)s+=line(x+w,y+i*rh,x+w+depth,y+i*rh-depth,C.rule,.8);
  const centers=[];
  for(let i=0;i<5;i++){const row=[];for(let j=0;j<3;j++){const xx=x+j*cw,yy=y+i*rh;row.push([xx+cw/2,yy+rh/2]);const role=i===4?(j===2?'query-target-placeholder':'query-feature'):(j===2?'known-label':'context-feature');s+=vectorGlyph(xx+2,yy+2,cw-4,rh-4,i,j,p,`data-cell-role="${role}" data-embedding-row="${i}" data-embedding-col="${j}"`);}centers.push(row);}
  s+='</g>';return {s,centers,query:centers[4][2],cw,rh};
}
function extractedVector(x,y,w,h,p,mobile){
  let s=`<g data-final-query-target="true">`+rect(x,y,w,h,C.rust,C.paper,2);
  const count=mobile?6:8;for(let j=0;j<count;j++){const xx=x+6+j*(w-12)/count;const hh=h*(.35+((j+Math.floor(p*2))%3)*.14);s+=rect(xx,y+(h-hh)/2,(w-12)/count-3,hh,C.rust,C.pale,1);}
  return s+'</g>';
}
function cellComputation(p,mobile){
  const input=mobile?promptTable(12,78,108,21,true,{small:true}):promptTable(18,132,216,30,false,{small:true});
  const cube=mobile?embeddedCube(165,103,126,112,24,p,true):embeddedCube(321,140,204,180,32,p,false);
  let s=input.s+cube.s;
  if(mobile){
    s+=text(66,53,'Table',18,C.blue)+text(139,128,'Embed',14,C.blue)+arrow(125,148,157,148)+packet(126,148,155,148,clamp(p*2),C.teal,5);
    s+=`<path d="M160,77 C170,22 344,22 321,77" fill="none" stroke="${C.blue}" stroke-width="1.5"/>`+arrow(325,66,321,77)+text(243,18,'Transform vectors',18,C.blue)+text(243,62,'repeat',14,C.muted);
    s+=text(228,239,'Embedding vectors',17,C.muted);
    s+=text(103,265,'Query target',17,C.rust)+text(103,289,'representation',17,C.rust);
    s+=`<path d="M295,204 H333 V255" fill="none" stroke="${C.rust}" stroke-width="1.5"/>`+arrow(333,244,333,255,C.rust)+routedPacket([[295,204],[333,204],[333,255]],clamp((p-.35)/.4),C.rust,5);
    s+=extractedVector(228,262,143,28,p,true);
    s+=`<g data-readout-source="query-target" data-overview-output="schematic" data-readout-visible="${p>=.75}">`;
    s+=`<path d="M228,276 H201 V329 H231" fill="none" stroke="${C.teal}" stroke-width="1.5"/>`+arrow(221,329,231,329,C.teal)+routedPacket([[228,276],[201,276],[201,329],[231,329]],clamp((p-.65)/.35),C.teal,5);
    s+=text(99,331,'Decode',20,C.blue)+text(99,347,'prediction head',14,C.muted);
    s+=text(306,318,'Pr(Y = 0)',18,C.teal,'middle','data-class-probability="0"')+text(306,344,'Pr(Y = 1)',18,C.teal,'middle','data-class-probability="1"');
    s+='</g>';
  }else{
    s+=text(126,91,'Whole input table',23,C.blue)+text(270,207,'Embed',18,C.blue)+arrow(243,229,307,229)+packet(245,229,304,229,clamp(p*2),C.teal,7);
    s+=`<path d="M333,113 C335,32 555,32 539,113" fill="none" stroke="${C.blue}" stroke-width="1.5"/>`+arrow(541,99,539,113);
    s+=text(438,31,'Transform representations',22,C.blue)+text(438,79,'repeat',18,C.muted);
    s+=text(429,359,'Each cell has an embedding vector',20,C.muted);
    s+=arrow(535,303,602,303,C.rust)+packet(537,303,598,303,clamp((p-.35)/.4),C.rust,7);
    s+=text(655,246,'Query target',19,C.rust)+text(655,273,'representation',19,C.rust)+extractedVector(610,285,97,40,p,false);
    s+=`<g data-readout-source="query-target" data-overview-output="schematic" data-readout-visible="${p>=.75}">`;
    s+=arrow(715,305,764,305,C.teal)+packet(717,305,760,305,clamp((p-.65)/.35),C.teal,7);
    s+=text(813,212,'Decode',25,C.blue)+text(813,241,'prediction head',18,C.muted);
    s+=text(813,291,'Pr(Y = 0)',22,C.teal,'middle','data-class-probability="0"')+text(813,332,'Pr(Y = 1)',22,C.teal,'middle','data-class-probability="1"');s+='</g>';
    s+=text(450,418,'Fixed pretrained weights · cell-based, TabPFN-style schematic',20,C.blue,'middle','data-weight-status="fixed"');
  }
  // Known labels take the table's embedding route too; query truth never enters.
  return s;
}
function permittedArrow(x1,y1,x2,y2,p,{source,reader,sourceRole='context',readerRole='context',queryId=''}={}){
  const attrs=`data-attention-source="${source}" data-attention-reader="${reader}" data-source-role="${sourceRole}" data-reader-role="${readerRole}"${queryId?` data-query-id="${queryId}"`:''}`;
  return `<g ${attrs}>`+arrow(x1,y1,x2,y2,readerRole==='context'?C.blue:C.teal,1.5)+packet(x1,y1,x2,y2,p,readerRole==='context'?C.blue:C.teal,4)+'</g>';
}
function maskQuery(x,y,w,h,values,id,mobile,p){
  const cw=w/3;let s=`<g data-mask-query="${id}">`+rect(x,y,w,h,C.blue,C.pale,1.4);values.forEach((v,j)=>{if(j)s+=line(x+j*cw,y,x+j*cw,y+h,C.rule,.8);s+=text(x+(j+.5)*cw,y+h*.7,v,mobile?19:24,j===2?C.rust:C.ink,'middle',j===2?'data-query-target="hidden"':'data-query-feature="true"');});
  const yy=y+h+10;s+=permittedArrow(x+cw*.5,yy,x+cw*2.5-5,yy,p,{source:`${id}-features`,reader:`${id}-target`,sourceRole:'query',readerRole:'query',queryId:id})+permittedArrow(x+cw*2.5,yy+5,x+cw*.5+5,yy+5,p,{source:`${id}-target`,reader:`${id}-features`,sourceRole:'query',readerRole:'query',queryId:id});return s+'</g>';
}
function attentionRoles(p,mobile){
  const box=mobile?{x:65,y:39,w:270,h:135}:{x:250,y:51,w:400,h:181};
  const nodes=mobile?[[120,83],[280,83],[200,137]]:[[350,103],[550,103],[450,182]],r=mobile?18:25;
  let s=text(mobile?200:450,mobile?23:29,'Labeled context rows',mobile?21:27,C.blue)+rect(box.x,box.y,box.w,box.h,C.blue);
  for(let i=0;i<3;i++)for(let j=0;j<3;j++)if(i!==j){const [ax,ay]=nodes[i],[bx,by]=nodes[j],a=Math.atan2(by-ay,bx-ax),offset=2;const dx=-Math.sin(a)*offset,dy=Math.cos(a)*offset;s+=permittedArrow(ax+(r+3)*Math.cos(a)+dx,ay+(r+3)*Math.sin(a)+dy,bx-(r+4)*Math.cos(a)+dx,by-(r+4)*Math.sin(a)+dy,clamp(p*2),{source:`c${i+1}`,reader:`c${j+1}`});}
  nodes.forEach(([xx,yy],i)=>{s+=`<circle cx="${xx}" cy="${yy}" r="${r}" fill="white" stroke="${C.blue}" stroke-width="1.5"/>`+text(xx,yy+(mobile?5:7),`C${i+1}`,mobile?17:23,C.blue);});
  if(mobile){
    s+=text(200,195,'Features + observed labels',17,C.muted);
    s+=text(97,220,'Query 1',18,C.blue)+text(303,220,'Query 2',18,C.blue);
    s+=`<g data-attention-source="context-block" data-attention-reader="q1" data-source-role="context" data-reader-role="query"><path d="M65,128 H6 V257 H8" fill="none" stroke="${C.teal}" stroke-width="1.5"/>`+arrow(0+1,257,8,257,C.teal)+routedPacket([[65,128],[6,128],[6,257],[8,257]],clamp((p-.35)/.65),C.teal,4)+'</g>';
    s+=`<g data-attention-source="context-block" data-attention-reader="q2" data-source-role="context" data-reader-role="query"><path d="M335,128 H394 V257 H392" fill="none" stroke="${C.teal}" stroke-width="1.5"/>`+arrow(399,257,392,257,C.teal)+routedPacket([[335,128],[394,128],[394,257],[392,257]],clamp((p-.35)/.65),C.teal,4)+'</g>';
    s+=maskQuery(12,236,170,41,['.6','.2','?'],'q1',true,p)+maskQuery(218,236,170,41,['.1','.7','?'],'q2',true,p);
    s+=text(200,305,'Within a query, feature attention is allowed.',14,C.muted)+text(200,326,'Distinct queries cannot read each other.',17,C.ink)+text(200,344,'Roles, not left-to-right generation.',14,C.muted);
  }else{
    s+=text(450,262,'Context rows can read one another',23,C.blue);
    s+=text(230,312,'Query 1',24,C.blue)+text(670,312,'Query 2',24,C.blue);
    s+=`<g data-attention-source="context-block" data-attention-reader="q1" data-source-role="context" data-reader-role="query"><path d="M250,170 H150 V322" fill="none" stroke="${C.teal}" stroke-width="1.5"/>`+arrow(150,310,150,322,C.teal)+routedPacket([[250,170],[150,170],[150,322]],clamp((p-.35)/.65),C.teal,5)+'</g>';
    s+=`<g data-attention-source="context-block" data-attention-reader="q2" data-source-role="context" data-reader-role="query"><path d="M650,170 H750 V322" fill="none" stroke="${C.teal}" stroke-width="1.5"/>`+arrow(750,310,750,322,C.teal)+routedPacket([[650,170],[750,170],[750,322]],clamp((p-.35)/.65),C.teal,5)+'</g>';
    s+=maskQuery(80,329,300,44,['.6','.2','?'],'q1',false,p)+maskQuery(520,329,300,44,['.1','.7','?'],'q2',false,p);
    s+=text(450,420,'Within each query row, features can exchange information.',19,C.muted)+text(450,451,'Distinct queries do not read each other; context never reads queries.',21,C.ink);
  }
  return s;
}
function shapeTable(x,y,w,rh,featureCount,contextCount,mobile,id){
  const cw=w/(featureCount+1);let s=`<g data-size-task="${id}" data-feature-count="${featureCount}" data-context-count="${contextCount}" data-query-feature-count="${featureCount}">`+rect(x,y,w,rh*(contextCount+2),C.blue);
  for(let j=1;j<=featureCount;j++)s+=line(x+j*cw,y,x+j*cw,y+rh*(contextCount+2),C.rule,.8);
  for(let j=0;j<=featureCount;j++)s+=text(x+(j+.5)*cw,y+rh*.72,j===featureCount?'Y':`X${j+1}`,mobile?16:rh<25?18:20,j===featureCount?C.teal:C.muted);
  for(let i=0;i<=contextCount;i++){const yy=y+(i+1)*rh,isQuery=i===contextCount;s+=line(x,yy,x+w,yy,isQuery?C.blue:C.rule,isQuery?1.5:.8);for(let j=0;j<=featureCount;j++){const value=j===featureCount?(isQuery?'?':String(i%2)):((i+j+1)%9/10).toFixed(1).replace(/^0/,'');s+=text(x+(j+.5)*cw,yy+rh*.72,value,mobile?16:rh<25?18:21,j===featureCount?(isQuery?C.rust:C.teal):C.ink,'middle',isQuery&&j===featureCount?'data-query-target="hidden"':'');}}
  return s+'</g>';
}
function tableSizes(p,mobile){
  let s='';
  if(mobile){
    s+=text(102,28,'Task A',21,C.blue)+shapeTable(12,42,180,24,2,2,true,'a');
    s+=text(122,177,'Task B',21,C.blue)+shapeTable(12,190,220,22,4,4,true,'b');
    s+=text(328,127,'Same model',19,C.blue)+predictor(270,143,116,86,p,true);
    s+=arrow(200,94,263,169)+packet(202,94,260,169,p,C.blue,5)+arrow(239,260,263,208,C.teal)+packet(241,258,261,212,p,C.teal,5);
    s+=text(328,263,'Separate calls',16,C.muted)+text(328,289,'Task A prediction',15,C.teal)+text(328,313,'Task B prediction',15,C.teal);
    s+=text(200,345,'Each query keeps its own task’s columns.',17,C.muted);
  }else{
    s+=text(170,38,'Task A: fewer rows and columns',23,C.blue)+shapeTable(30,59,280,31,2,3,false,'a');
    s+=text(220,268,'Task B: more rows and columns',23,C.blue)+shapeTable(30,290,380,20,4,5,false,'b');
    s+=text(650,115,'Same pretrained model',26,C.blue)+predictor(525,145,250,170,p,false);
    s+=arrow(318,139,517,198)+packet(320,140,514,197,p,C.blue,7)+arrow(419,354,517,276,C.teal)+packet(420,353,514,279,p,C.teal,7);
    s+=text(650,357,'Separate prediction calls',21,C.muted)+text(650,391,'One set of pretrained weights',21,C.blue);
    s+=text(450,453,'Within each task, context and query columns align.',19,C.muted);
  }
  return s;
}
/** SVG contents for a 900×460 desktop or 400×350 mobile viewBox. */
export function renderOverview(stage=0,progress=1,mobile=false){
  const p=clamp(progress),k=Math.max(0,Math.min(4,Math.trunc(Number.isFinite(stage)?stage:0)));
  const titles=['Earlier tables train shared network weights','The whole current task table is the prompt','Cell embeddings transform iteratively, then the query target is read out','Attention follows context and query roles, not left-to-right generation','Different task table sizes use the same fixed pretrained model'];
  const desc=[
    'Original schematic: pretraining tables used earlier teach shared network weights. Those earlier tables shape the network; the new task table is supplied later. Moving packets and changing connection widths are not recorded training traces.',
    'One unified input table contains feature values, four observed context labels, and a new row with known features and an unknown target. Known labels are inputs. The entire current task table enters the pretrained model, whose weights remain fixed. Earlier pretraining tables are not supplied again. The output is unnumbered.',
    'Original cell-based TabPFN-style schematic inspired by Molnar Figure 4.6. Feature cells, known label cells, and the unknown-target placeholder each become vectors, shown by the depth of a table-shaped tensor. Repeated attention and nonlinear transformations contextualize them. Only the final query-target representation is selected for the prediction head. Pr(Y=0) and Pr(Y=1) are symbolic class probabilities, not numerical model results.',
    'Arrows show allowed information flow into a reading row. Every labeled context row can read every other context row. Each query row reads labeled context; context rows never read query rows, and distinct query rows never read one another. Feature attention within each query row still exchanges information. This is not a left-to-right language-generation mask.',
    'Two schematic tasks differ in numbers of context rows and feature columns. Each task is a separate call to the same pretrained model with fixed weights. Within each task, the query has the same feature columns in the same order as its labeled context. This diagram states no numerical size limit or guarantee of predictive accuracy.'
  ];
  const content=[pretraining,inference,cellComputation,attentionRoles,tableSizes][k](p,mobile);
  return `<g data-overview="true" data-overview-stage="${k}" data-overview-progress="${p}" data-overview-mobile="${Boolean(mobile)}" data-weight-status="${k===0?'learning':'fixed'}"><title>${titles[k]}</title><desc>${desc[k]}</desc>${content}</g>`;
}
