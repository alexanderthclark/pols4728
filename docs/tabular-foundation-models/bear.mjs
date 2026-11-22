import {bearResults} from './bear-results.mjs';
// The bear example is simulated. SVG transitions interpolate visual geometry;
// they do not represent intermediate fitted models or measured attention.
const C = {ink:'#242424', blue:'#234E70', rust:'#A54F32', teal:'#007F73', muted:'#606060', rule:'#767676'};
const clamp = p => Math.max(0, Math.min(1, Number.isFinite(p) ? p : 0));
const mix = (a,b,p) => a+(b-a)*p;
const HIGH = 1/(1+Math.exp(-1));
const LOW = 1-HIGH;
const escape = value => String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
const text = (x,y,value,size=22,color=C.ink,anchor='middle') => `<text x="${x}" y="${y}" font-size="${size}" fill="${color}" text-anchor="${anchor}">${escape(value)}</text>`;
const line = (x1,y1,x2,y2,color=C.rule,width=2,attrs='') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}" ${attrs}/>`;
const rect = (x,y,w,h,color=C.rule,fill='white',r=3) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${fill}" stroke="${color}" stroke-width="1.5" rx="${r}"/>`;
const dot = (x,y,r=5,color=C.blue) => `<circle cx="${x}" cy="${y}" r="${r}" fill="${color}"/>`;
const group = (s,opacity=1) => `<g opacity="${opacity}">${s}</g>`;
const percent = value => `${(value*100).toFixed(1)}%`;
const growLine = (x1,y1,x2,y2,p,color=C.blue,width=2) => line(x1,y1,mix(x1,x2,p),mix(y1,y2,p),color,width);

export const bearAucCases = [
  {id:'signal_only', label:'Two useful columns', short:'Signal only', auc:.7300, color:C.blue},
  {id:'continuous_noise', label:'+ continuous noise', short:'+ continuous noise', auc:.6506, color:C.rust},
  {id:'binary_noise', label:'+ binary noise', short:'+ binary noise', auc:.7304, color:C.blue},
  {id:'two_bins', label:'Continuous noise; two bins', short:'Two histogram bins', auc:.7291, color:C.blue},
  {id:'interaction_feature', label:'Continuous noise + product', short:'+ interaction product', auc:.7277, color:C.blue}
].map(item => {
  const result = bearResults.cases.find(row => row.depth === 2 && row.id === item.id);
  if (!result) throw new Error(`Missing bear diagnostic: ${item.id}`);
  return {...item, auc:result.mean_auc};
});

function probabilityBar(x,y,w,h,p,label,small=false) {
  return rect(x,y,w,h,C.rule) + `<rect x="${x+1}" y="${y+1}" width="${(w-2)*p}" height="${h-2}" fill="${C.teal}" opacity=".18"/>`
    + line(x+w*p,y,x+w*p,y+h,C.teal,3)
    + text(x+w/2,y+h/2+(small?6:8),label,small?20:26,C.ink);
}

function margins(p,mobile) {
  const W=mobile?400:900, x=mobile?30:90, w=mobile?150:300, gap=mobile?40:120;
  const headings=['Bear type','Strategy'];
  let s=text(W/2,mobile?24:35,'Survival, averaged over the other variable',mobile?17:23,C.muted);
  headings.forEach((h,j)=>{
    const xx=x+j*(w+gap), yy=mobile?70:100;
    s+=text(xx+w/2,yy,h,mobile?21:28,C.blue);
    const labels=j===0?['Black','Brown']:['Fight','Lie down'];
    labels.forEach((label,k)=>{
      const y=yy+(mobile?35:55)+k*(mobile?85:105);
      s+=text(xx,y,label,mobile?18:22,C.ink,'start');
      // Reveal a fixed 50% bar; never attach a 50% label to another width.
      s+=group(probabilityBar(xx,y+12,w,mobile?34:42,.5,'50%',mobile),clamp(p*1.6));
    });
  });
  return s+text(W/2,mobile?318:415,'Neither variable helps on its own.',mobile?19:25,C.blue);
}

function joint(p,mobile) {
  const x=mobile?102:285, y=mobile?75:95, w=mobile?130:205, h=mobile?92:125;
  let s=text(x+w/2,y-23,'Fight',mobile?19:25)+text(x+w*1.5,y-23,'Lie down',mobile?19:25);
  for(let b=0;b<2;b++) {
    s+=text(x-(mobile?15:30),y+b*h+h/2+7,b?'Brown':'Black',mobile?18:24,C.ink,'end');
    for(let a=0;a<2;a++) {
      const target=b===a?HIGH:LOW, value=mix(.5,target,p), color=target===HIGH?C.teal:C.rust;
      const xx=x+a*w, yy=y+b*h;
      s+=rect(xx,yy,w,h,C.rule);
      s+=`<rect x="${xx+1}" y="${yy+h-(h-2)*value-1}" width="${w-2}" height="${(h-2)*value}" fill="${color}" opacity=".13"/>`;
      s+=line(xx+1,yy+h-(h-2)*value-1,xx+w-1,yy+h-(h-2)*value-1,color,2);
      s+=text(xx+w/2,yy+h/2+9,percent(value),mobile?27:39,color);
    }
  }
  return s+text(mobile?200:490,mobile?304:392,'Same marginal averages. Different joint outcomes.',mobile?16:23,C.blue);
}

function tree(p,mobile) {
  const center=mobile?200:450, rootY=mobile?30:42, branchY=mobile?110:154, leafY=mobile?227:326;
  const children=mobile?[105,295]:[235,665], leaves=mobile?[48,145,255,352]:[115,345,555,785];
  let s=text(center,rootY,'Bear type?',mobile?22:28,C.blue)+text(center,rootY+27,'50%',mobile?18:23,C.muted);
  children.forEach((x,j)=>{
    s+=line(center,rootY+37,x,branchY-28,C.blue,2)+text(x,branchY,j?'Brown':'Black',mobile?20:26,C.blue);
    s+=text(x,branchY+25,'50%',mobile?18:24,C.muted);
    s+=text(x,branchY+51,'Strategy?',mobile?18:23,C.ink);
    for(let k=0;k<2;k++) {
      const xx=leaves[j*2+k], q=clamp(p*1.5-k*.18), value=j===k?HIGH:LOW;
      s+=growLine(x,branchY+60,xx,leafY-25,q,C.blue);
      s+=group(text(xx,leafY,k?'Lie down':'Fight',mobile?16:22)+text(xx,leafY+(mobile?32:38),percent(value),mobile?24:32,value===HIGH?C.teal:C.rust),q);
    }
  });
  return s+text(center,mobile?320:428,'The second split reveals the interaction.',mobile?18:24,C.blue);
}

function thresholds(p,mobile) {
  const left=mobile?150:325, right=mobile?370:825, ys=mobile?[65,135,235]:[85,175,305], size=mobile?17:23;
  let s='';
  ['Bear type','Strategy','Noise column'].forEach((label,j)=>{
    const y=ys[j];
    s+=text(mobile?15:50,y+7,label,size,j===2?C.rust:C.blue,'start');
    s+=line(left,y,right,y,C.rule,2);
    if(j<2) {
      s+=dot(left,y,5,C.blue)+dot(right,y,5,C.blue);
      s+=line((left+right)/2,y-17,(left+right)/2,y+17,C.blue,3);
      s+=text((left+right)/2,y+42,'One possible cut',mobile?16:20,C.muted);
    } else {
      for(let k=0;k<17;k++) {
        const xx=mix(left,right,(k+.3)/17);
        s+=dot(xx,y,3.5,k%3===0?C.teal:C.rust);
      }
      const split=mix(left+15,right-15,p);
      s+=rect(left,y-13,Math.max(1,split-left),26,'none','#f7e8e1',0);
      for(let k=0;k<17;k++)s+=dot(mix(left,right,(k+.3)/17),y,3.5,k%3===0?C.teal:C.rust);
      s+=line(split,y-25,split,y+25,C.rust,3);
      s+=text((left+right)/2,y+53,'Many candidate cuts',mobile?16:22,C.rust);
    }
  });
  return s+text(mobile?200:450,mobile?333:425,'Schematic search; dots are not fitted data.',mobile?16:21,C.muted);
}

function auc(p,mobile) {
  const left=mobile?185:385, right=mobile?330:810, top=mobile?55:75, row=mobile?48:57, axis=mobile?305:385;
  const scale=value=>left+(value-.5)/.25*(right-left);
  let s=text(mobile?200:450,mobile?21:30,'Mean held-out AUC · 12 seeds',mobile?18:24,C.blue);
  for(const value of (mobile?[.5,.6,.7]:[.5,.6,.7,.75])) {
    const x=scale(value);
    s+=line(x,axis,x,axis+6,C.rule,1.5)+text(x,axis+26,value.toFixed(2),mobile?16:20,C.muted);
    if(value===.5)s+=line(x,top-14,x,axis,C.rule,1,'stroke-dasharray="4 5"');
  }
  s+=line(left,axis,right,axis,C.rule,1.5);
  bearAucCases.forEach((d,i)=>{
    const y=top+i*row, q=clamp(p*1.5-i*.12), x=scale(mix(.5,d.auc,q));
    s+=text(mobile?12:40,y+6,mobile?d.short:d.label,mobile?16:21,C.ink,'start');
    s+=line(left,y,x,y,d.color,2)+`<circle cx="${x}" cy="${y}" r="${mobile?5:6}" fill="${d.color}" data-case="${d.id}" data-auc="${d.auc}"/>`;
    s+=group(text(mobile?390:880,y+6,d.auc.toFixed(3),mobile?16:21,d.color,'end'),q);
  });
  return s;
}

function hypothesis(p,mobile) {
  const W=mobile?400:900;
  let s=text(W/2,mobile?24:34,'A hypothesis about TabPFN',mobile?20:27,C.blue);
  if(mobile) {
    const y=65;
    s+=rect(30,y,150,72,C.blue)+text(105,y+29,'Bear + strategy',18,C.blue)+text(105,y+55,'Joint pattern',17,C.ink);
    s+=rect(220,y,150,72,C.rule)+text(295,y+29,'Noise columns',18,C.muted)+text(295,y+55,'Irrelevant',17,C.muted);
    s+=growLine(105,y+80,175,205,p,C.blue,3)+growLine(295,y+80,225,205,p,C.rule,1.5);
    s+=rect(85,205,230,66,C.blue,'#f7fafc')+text(200,233,'Pretrained inference',21,C.blue)+text(200,257,'uses the whole context',17,C.muted);
    s+=text(200,320,'No measured attention or TabPFN scores.',16,C.muted);
  } else {
    s+=rect(45,105,230,95,C.blue)+text(160,143,'Bear + strategy',25,C.blue)+text(160,177,'Joint pattern',22);
    s+=rect(45,255,230,95,C.rule)+text(160,293,'Noise columns',25,C.muted)+text(160,327,'Irrelevant',22,C.muted);
    s+=growLine(290,152,440,210,p,C.blue,4)+growLine(290,303,440,247,p,C.rule,1.5);
    s+=rect(460,130,350,190,C.blue,'#f7fafc',8)+text(635,185,'Pretrained inference',28,C.blue)+text(635,232,'Context + learned task prior',22)+text(635,278,'Can use joint patterns',24,C.blue);
    s+=text(450,421,'A plausible explanation, not an identified internal mechanism.',23,C.muted);
  }
  return s;
}

function typeControl(p,mobile) {
  const names=mobile?['Bear','Strategy','Noise 1','Noise 2','Noise 3']:['Bear type','Strategy','Noise 1','Noise 2','Noise 3'];
  const values=[[0,0,.13,.89,.24],[0,1,.82,.35,.91],[1,0,.26,.77,.42],[1,1,.63,.19,.68]];
  const left=mobile?15:90, top=mobile?72:110, w=mobile?74:144, h=mobile?43:58;
  let s=text(mobile?200:450,mobile?25:40,p<.5?'Only the signal columns are binary.':'Now the distractors are binary too.',mobile?19:27,C.blue);
  names.forEach((name,j)=>s+=text(left+(j+.5)*w,top-18,name,mobile?16:23,j<2?C.blue:C.muted));
  for(let i=0;i<4;i++)for(let j=0;j<5;j++) {
    const x=left+j*w,y=top+i*h;
    s+=rect(x,y,w,h,j<2?C.blue:C.rule,j<2?'#f0f5f8':'white',0);
    const v=values[i][j];
    if(j<2)s+=text(x+w/2,y+h/2+7,v,mobile?20:27,C.blue);
    else {
      const offset=mix(0,-h*.65,clamp(p*2));
      s+=group(text(x+w/2,y+h/2+7+offset,v.toFixed(2),mobile?18:24,C.muted),1-clamp(p*2));
      s+=group(text(x+w/2,y+h/2+7+mix(h*.4,0,clamp((p-.25)/.75)),v>.5?1:0,mobile?20:27,C.rust),clamp((p-.25)/.75));
    }
  }
  s+=text(mobile?200:450,mobile?300:395,'Does TabPFN’s advantage survive?',mobile?20:27,C.blue);
  s+=text(mobile?200:450,mobile?329:435,'Proposed control; TabPFN has not been rerun.',mobile?16:21,C.muted);
  return s;
}

function render(stage,p,mobile) {
  const progress=clamp(p), draw=[margins,joint,tree,thresholds,auc,hypothesis,typeControl][stage];
  if(!draw)throw new RangeError('Unknown bear scene');
  return `<g data-bear-stage="${stage}" data-bear-progress="${progress}" data-bear-mobile="${mobile}">${draw(progress,mobile)}</g>`;
}

export function renderBear(stage,p) { return render(stage,p,false); }
export function renderBearMobile(stage,p) { return render(stage,p,true); }
