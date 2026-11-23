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
function contextTable(mobile,stage,p){
  const x=mobile?16:28,y=mobile?44:132,w=mobile?210:260;
  const head=mobile?24:30,rh=mobile?23:31,cw=w/3;
  const switched=stage===2&&p>=.5;
  const label=stage===2?`Labeled context ${switched?'B':'A'}`:'Labeled context';
  let s=text(x+w/2,mobile?27:107,label,mobile?19:24,C.blue,'middle','data-context-label="true"');
  s+=rect(x,y,w,head+4*rh,C.rule,C.paper,1);
  s+=text(x+cw,y+(mobile?17:21),'Features',mobile?16:19,C.muted);
  s+=text(x+cw*2.5,y+(mobile?17:21),'Target',mobile?16:19,C.muted);
  s+=line(x+2*cw,y,x+2*cw,y+head+4*rh,C.rule,1);
  s+=line(x+cw,y+head,x+cw,y+head+4*rh,C.rule,.7);
  const data=[['.2','.8','0'],['.4','.1','1'],['.8','.3','1'],['.7','.9','0']];
  data.forEach((row,i)=>{
    const yy=y+head+i*rh;
    s+=line(x,yy,x+w,yy,C.rule,.7);
    row.forEach((v,j)=>{
      const changed=switched&&j===2&&(i===0||i===1);
      const value=changed?(v==='0'?'1':'0'):v;
      s+=text(x+(j+.5)*cw,yy+rh*.72,value,mobile?17:21,j===2?(changed?C.rust:C.teal):C.ink,'middle',j===2?'data-context-target="true"':'');
    });
  });
  return s;
}
function query(mobile){
  const x=mobile?246:28,y=mobile?88:339,w=mobile?138:260,h=mobile?40:37;
  let s=text(x+w/2,mobile?66:325,'New row',mobile?19:24,C.blue);
  s+=rect(x,y,w,h,C.blue,C.paper,1.3,'data-query="true"');
  ['.6','.2','?'].forEach((v,j)=>{
    if(j)s+=line(x+j*w/3,y,x+j*w/3,y+h,C.rule,.8);
    s+=text(x+(j+.5)*w/3,y+h*.68,v,mobile?19:23,j===2?C.rust:C.ink,'middle',j===2?'data-query-target="hidden"':'');
  });
  if(mobile)s+=text(x+w/2,151,'Target hidden',16,C.muted);
  else s+=text(x+w/2,402,'Target hidden',18,C.muted);
  return `<g data-overview-query="true">${s}</g>`;
}
function inputArrows(mobile){
  if(mobile)return `<path d="M121,167 H86 V198" fill="none" stroke="${C.blue}" stroke-width="1.7"/>`+arrow(86,188,86,198)+arrow(315,159,315,198);
  return arrow(294,219,346,219)+`<path d="M294,357 H328 V270 H346" fill="none" stroke="${C.blue}" stroke-width="1.7"/>`+arrow(336,270,346,270);
}
function output(mobile,stage,p){
  let s='';
  if(mobile){
    s+=arrow(200,297,200,315,C.teal);
    s+=text(200,337,stage===2?'Prediction can change':'Prediction for the new row',19,C.teal,'middle','data-schematic-output="true"');
    s+=packet(200,297,200,312,p,C.teal,6);
  }else{
    s+=arrow(722,226,776,226,C.teal);
    s+=packet(724,226,772,226,p,C.teal,8);
    s+=text(829,211,'Prediction',23,C.teal,'middle','data-schematic-output="true"');
    s+=text(829,241,stage===2?'can change':'for the',20,C.teal);
    if(stage!==2)s+=text(829,269,'new row',20,C.teal);
  }
  return `<g data-overview-output="schematic">${s}</g>`;
}
function pretraining(p,mobile){
  let s='';
  if(mobile){
    s+=text(200,24,'Earlier: many prediction tasks',20,C.ink);
    [16,151,286].forEach((x,i)=>{
      s+=miniTask(x,63,98,89,i,true);
      const route=i===1?[[200,158],[200,166],[326,166],[326,198]]:[[x+49,158],[i===0?86:312,198]];
      if(i===1)s+=`<path d="M200,158 V166 H326 V198" fill="none" stroke="${C.blue}" stroke-width="1.3"/>`+arrow(326,188,326,198,C.blue,1.3);
      else s+=arrow(...route[0],...route[1],C.blue,1.3);
      s+=routedPacket(route,clamp(p*1.6-i*.26),C.teal,6);
    });
  }else{
    s+=text(170,45,'Many prediction tasks',25,C.ink);
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
function inference(p,mobile){
  let s=contextTable(mobile,1,p)+query(mobile)+inputArrows(mobile)+modelFrame(mobile,1).s;
  s+=network(mobile,p);
  s+=mobile?routedPacket([[121,167],[86,167],[86,198],[111,239]],p,C.teal,6)+routedPacket([[315,159],[315,198],[289,253]],p,C.rust,6):packet(295,219,411,218,p,C.teal,8)+packet(328,270,410,259,p,C.rust,8);
  return s+output(mobile,1,clamp((p-.35)/.65));
}
function representations(mobile,p){
  const xs=mobile?[109,200,291]:[412,532,652];
  const top=mobile?232:165,w=mobile?76:96,h=mobile?48:108;
  const shift=smooth(p), phase=clamp(p*2);
  let s='';
  ['Encode','Exchange','Decode'].forEach((label,k)=>{
    const x=xs[k]-w/2;
    if(k)s+=arrow(xs[k-1]+w/2+2,top+h/2,x-3,top+h/2,C.blue,1.4);
    s+=rect(x,top,w,h,C.blue,C.paper,1.2);
    s+=text(xs[k],mobile?251:187,label,mobile?16:18,C.blue);
  });
  if(mobile){
    [0,1].forEach(i=>{
      s+=line(88,261+i*8,111+mix(3,13,shift*(1-i)),261+i*8,C.teal,2.5);
      s+=line(273,261+i*8,289+mix(4,15,shift*i),261+i*8,C.teal,2.5);
      const endY=mix(261+i*8,269-i*8,shift);
      s+=line(179,261+i*8,219,endY,C.blue,1.2);
      s+=`<circle cx="179" cy="${261+i*8}" r="2.5" fill="${C.teal}"/><circle cx="219" cy="${f(endY)}" r="2.5" fill="${C.rust}" data-context-representation="true"/>`;
    });
    // The representation itself moves: the fixed-weight model never moves.
    s+=packet(85,277,314,277,p,C.teal,5);
    return s;
  }
  // Small vector glyphs stand for representations, not a specified tokenization.
  const glyph=(x,y,offset,color)=>{
    let z='';
    for(let k=0;k<4;k++){
      const length=12+((k+offset)%3)*6;
      z+=line(x,y+k*9,x+length,y+k*9,color,3);
    }
    return z;
  };
  s+=glyph(381,207,0,C.teal)+glyph(416,221,1,C.rust);
  const left=504,right=559,baseY=211;
  [0,1,2].forEach(i=>{
    const aY=baseY+i*20,bY=baseY+((i+1)%3)*20;
    s+=line(left,aY,right,mix(aY,bY,shift),C.blue,1.2);
    s+=`<circle cx="${left}" cy="${aY}" r="4" fill="${C.teal}"/>`;
    s+=`<circle cx="${right}" cy="${f(mix(aY,bY,shift))}" r="4" fill="${C.rust}" data-context-representation="true"/>`;
  });
  s+=glyph(625,211,Math.round(phase*2),C.teal);
  // A progress-controlled representation passes through encode/exchange/decode.
  s+=packet(382,268,682,268,p,C.teal,7);
  s+=text(532,350,'Attention + nonlinear transformations',21,C.muted);
  return s;
}
function changingContext(p,mobile){
  let s=contextTable(mobile,2,p)+query(mobile)+inputArrows(mobile)+modelFrame(mobile,2).s;
  s+=representations(mobile,p);
  if(mobile){
    // Labeled context still enters the same model as its representations change.
    s+=routedPacket([[121,167],[86,167],[86,198]],p,C.teal,6);
  }else s+=packet(295,219,346,219,p,C.teal,8);
  return s+output(mobile,2,p);
}

/** SVG contents for a 900×460 desktop or 400×350 mobile viewBox. */
export function renderOverview(stage=0,progress=1,mobile=false){
  const p=clamp(progress), k=Math.max(0,Math.min(2,Math.trunc(Number.isFinite(stage)?stage:0)));
  const titles=['Many tasks teach shared network weights','Labeled context and a new row enter a fixed-weight network','Context-dependent representations produce the prediction'];
  const descriptions=[
    'Schematic: several separate tables feed a shared network during pretraining. Scroll moves task inputs toward the network and changes its learned connections. These learned weights are reused for a new table.',
    'Illustrative labeled rows and new-row features feed the same pretrained network. The query target remains a question mark. Scroll moves inputs through the network toward an unnumbered prediction; weights remain fixed.',
    'The query features stay the same while two example labels change from context A to B. Inside the fixed-weight network, encoding, information exchange, and decoding produce context-dependent representations. Scroll changes their geometry. An unnumbered prediction can change; there are no measured attention values or model outputs.'
  ];
  const content=k===0?pretraining(p,mobile):k===1?inference(p,mobile):changingContext(p,mobile);
  return `<g data-overview="true" data-overview-stage="${k}" data-overview-progress="${p}" data-overview-mobile="${Boolean(mobile)}"><title>${titles[k]}</title><desc>${descriptions[k]}</desc>${content}</g>`;
}
