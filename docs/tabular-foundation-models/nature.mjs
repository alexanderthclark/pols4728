// Original teaching diagrams based on Hollmann et al., Nature (2025), Figs. 1, 2, 5.
// Numbers in the example table are constructed, not a TabPFN trace.
const C={ink:'#242424',blue:'#234E70',rust:'#A54F32',teal:'#007F73',muted:'#606060',rule:'#767676',light:'#d6dcdf'};
const clamp=n=>Math.max(0,Math.min(1,Number.isFinite(n)?n:1));
const f=n=>Number(n.toFixed(4));
const ease=n=>{const t=clamp(n);return t*t*(3-2*t);};
const txt=(x,y,s,size=21,color=C.ink,anchor='start',extra='')=>`<text x="${f(x)}" y="${f(y)}" font-size="${size}" fill="${color}" text-anchor="${anchor}" ${extra}>${s}</text>`;
const rect=(x,y,w,h,fill,extra='')=>`<rect x="${f(x)}" y="${f(y)}" width="${f(w)}" height="${f(h)}" fill="${fill}" ${extra}/>`;
const line=(x1,y1,x2,y2,color=C.rule,width=1.5,extra='')=>`<line x1="${f(x1)}" y1="${f(y1)}" x2="${f(x2)}" y2="${f(y2)}" stroke="${color}" stroke-width="${width}" ${extra}/>`;
const circle=(x,y,r,fill,extra='')=>`<circle cx="${f(x)}" cy="${f(y)}" r="${r}" fill="${fill}" ${extra}/>`;
const group=(s,opacity=1,extra='')=>`<g opacity="${f(opacity)}" ${extra}>${s}</g>`;
const rows=[['0','0','.13','.89','.24','1'],['0','1','.82','.35','.91','0'],['1','0','.26','.77','.42','0'],['1','1','.63','.19','.68','1']];
const query=['1','1','.72','.28','.55','?'];
const headers=['B','L','Z₁','Z₂','Z₃','Y'];

function arrow(x1,y1,x2,y2,t,color=C.blue) {
  const u=ease(t),x=x1+(x2-x1)*u,y=y1+(y2-y1)*u;
  const angle=Math.atan2(y2-y1,x2-x1),a=7;
  let s=line(x1,y1,x,y,color,2);
  if(u>.03)s+=`<path d="M ${f(x-a*Math.cos(angle-.5))} ${f(y-a*Math.sin(angle-.5))} L ${f(x)} ${f(y)} L ${f(x-a*Math.cos(angle+.5))} ${f(y-a*Math.sin(angle+.5))}" fill="none" stroke="${color}" stroke-width="2"/>`;
  return s;
}

function smallTable(x,y,w,h,shownRows,p,mobile) {
  const cw=w/6,size=mobile?16:21;
  let s=rect(x,y,w,h,'#eef1f3');
  headers.forEach((v,j)=>s+=txt(x+(j+.5)*cw,y+h*.72,v,size,j===5?C.teal:C.ink,'middle'));
  shownRows.forEach((row,i)=>{
    const r=ease(p*shownRows.length-i),yy=y+(i+1)*h;
    let cells=line(x,yy+h,x+w,yy+h,C.light);
    row.forEach((v,j)=>cells+=txt(x+(j+.5)*cw,yy+h*.72,v,size,j===5?C.teal:C.ink,'middle'));
    s+=group(cells,r,`transform="translate(${f((1-r)*24)} 0)"`);
  });
  return s;
}

function generator(p,mobile) {
  const g=ease(p/.42),table=ease((p-.34)/.6);
  let s='';
  const nodes=mobile?[[45,41,'B'],[135,41,'L'],[90,132,'Y'],[204,132,'U']]:[[100,80,'B'],[280,80,'L'],[190,220,'Y'],[385,220,'U']];
  const r=mobile?23:30;
  const [b,l,y,u]=nodes;
  s+=arrow(b[0]+10,b[1]+r-3,y[0]-9,y[1]-r,g,C.blue);
  s+=arrow(l[0]-10,l[1]+r-3,y[0]+9,y[1]-r,g,C.blue);
  s+=arrow(u[0]-r,u[1],y[0]+r,y[1],g,C.rust);
  nodes.forEach(([x,yy,label],i)=>{
    s+=circle(x,yy,r,'white',`stroke="${i===3?C.rust:i===2?C.teal:C.blue}" stroke-width="2"`);
    s+=txt(x,yy+(mobile?6:8),label,mobile?21:27,i===3?C.rust:i===2?C.teal:C.blue,'middle');
  });
  if(mobile) {
    s+=txt(8,93,'Brown',16,C.ink)+txt(132,93,'Lie down',16,C.ink);
    s+=txt(90,172,'Survived',16,C.teal,'middle')+txt(204,172,'Noise',16,C.rust,'middle');
    [265,315,365].forEach((x,i)=>s+=circle(x,41,21,'white',`stroke="${C.rule}" stroke-width="1.5"`)+txt(x,47,headers[i+2],18,C.muted,'middle'));
    s+=txt(315,91,'Disconnected',16,C.muted,'middle')+txt(315,113,'distractors',16,C.muted,'middle');
    s+=txt(20,213,'Illustrative rows from one rule',17);
    s+=smallTable(20,227,360,29,rows.slice(0,2),table,true);
    s+=txt(200,341,'Teaching graph, not a sampled TabPFN task.',16,C.muted,'middle');
  } else {
    s+=txt(100,32,'Brown',21,C.ink,'middle')+txt(280,32,'Lie down',21,C.ink,'middle');
    s+=txt(190,274,'Survived',21,C.teal,'middle')+txt(385,274,'Noise',21,C.rust,'middle');
    [100,230,360].forEach((x,i)=>s+=circle(x,341,26,'white',`stroke="${C.rule}" stroke-width="1.5"`)+txt(x,348,headers[i+2],23,C.muted,'middle'));
    s+=txt(230,393,'Disconnected distractors',21,C.muted,'middle');
    s+=txt(480,76,'Illustrative rows from one rule',23);
    s+=smallTable(480,102,390,48,rows,table,false);
    s+=txt(450,443,'Teaching graph, not a task extracted from TabPFN pretraining.',21,C.muted,'middle');
  }
  return s;
}

function architecture(p,mobile) {
  const x=mobile?25:150,y=mobile?44:77,w=mobile?330:600,h=mobile?41:53,hh=mobile?28:37,cw=w/6;
  const feature=ease(p/.48),sample=ease((p-.43)/.5),size=mobile?17:23;
  const all=[...rows,query];
  let s=txt(mobile?200:450,mobile?23:33,p<.5?'1. Combine features within each row':'2. Query reads the training context',mobile?17:25,C.ink,'middle');
  s+=rect(x,y,w,hh,'#eef1f3');
  headers.forEach((v,j)=>s+=txt(x+(j+.5)*cw,y+hh*.73,v,mobile?17:22,j===5?C.teal:C.ink,'middle'));
  all.forEach((row,i)=>{
    const yy=y+hh+i*h,cy=yy+(mobile?16:22),vy=yy+h-9;
    if(i===4)s+=rect(x,yy,w,h,'#eaf3f0',`stroke="${C.teal}" stroke-width="1.3"`);
    else s+=line(x,yy+h,x+w,yy+h,C.light);
    row.forEach((v,j)=>{
      const cx=x+(j+.5)*cw;
      s+=txt(cx,cy,v,size,j===5?C.teal:C.ink,'middle');
      // Unnumbered vector glyphs denote representations, not a measured trace.
      for(let k=0;k<3;k++)s+=rect(cx-8+k*6,vy-3,4,6,j===5?C.teal:C.blue,`opacity="${f(.2+.8*feature)}"`);
    });
    s+=line(x+cw*.5+10,vy,x+cw*.5+10+(w-cw-20)*feature,vy,C.blue,mobile?1.6:2,`opacity="${f(1-.73*sample)}"`);
    if(!mobile)s+=txt(125,cy,i===4?'Query':`Row ${i+1}`,20,i===4?C.teal:C.muted,'end');
  });
  // Information arrives at the query from context in the same column/group.
  // Training-to-training attention is also permitted, but omitted for clarity.
  for(let j=0;j<6;j++) {
    const cx=x+(j+.5)*cw,channel=cx+(mobile?18:32),targetY=y+hh+5*h-9;
    for(let i=0;i<4;i++) {
      const sy=y+hh+(i+1)*h-9,amount=ease(sample*4-i),endY=sy+(targetY-sy)*amount;
      const d=`M ${f(cx+10)} ${f(sy)} H ${f(channel)} V ${f(endY)}${amount>=.999?` H ${f(cx+10)}`:''}`;
      s+=`<path d="${d}" fill="none" stroke="${C.teal}" stroke-width="${mobile?1.2:1.5}" opacity="${f(sample*.65)}"/>`;
    }
    if(sample>.01)s+=circle(cx+9,targetY,2.5,C.teal,`opacity="${f(sample)}"`);
  }
  if(mobile) {
    s+=txt(200,307,'Last row: query; its Y stays hidden.',16,C.teal,'middle');
    s+=txt(200,332,'Paths show access, not attention weights.',16,C.muted,'middle');
  } else {
    s+=txt(450,415,'Query labels are withheld; training rows supply the context.',22,C.teal,'middle');
    s+=txt(450,451,'Ungrouped view · permitted information flow, not measured attention',20,C.muted,'middle');
  }
  return s;
}

function efficiency(p,mobile) {
  const t=ease(p),x=mobile?35:300,w=mobile?330:500,h=mobile?40:54;
  const ys=mobile?[87,208]:[123,260],axis=mobile?288:365;
  let s=txt(mobile?200:450,mobile?25:37,'Similar aggregate predictive performance',mobile?17:25,C.ink,'middle');
  const models=[['CatBoost (default)',1,C.blue],['TabPFN',.5,C.teal]];
  models.forEach(([name,fraction,color],i)=>{
    const yy=ys[i];
    s+=txt(mobile?x:65,mobile?yy-13:yy+32,name,mobile?19:23,color);
    s+=rect(x,yy,w*fraction,h,color,`data-model="${i===0?'catboost':'tabpfn'}" data-fraction="${fraction}"`);
    for(let k=1;k<20*fraction;k++)s+=line(x+w*k/20,yy+4,x+w*k/20,yy+h-4,'white',1,'opacity=".33"');
    s+=txt(mobile?365:880,mobile?yy-13:yy+34,i===0?'100%':'≈50%',mobile?20:25,C.ink,'end');
  });
  // Gray unused rows leave the diagram; the measured comparison stays fixed.
  for(let k=10;k<20;k++) {
    const lift=t*(25+(k-10)*2),fade=1-t;
    s+=rect(x+w*k/20+2,ys[1]-lift+3,w/20-4,h-6,'#d6dcdf',`opacity="${f(fade*.75)}" data-unused-row="${k}"`);
  }
  s+=line(x,axis,x+w,axis,C.rule);
  [0,.5,1].forEach(v=>s+=line(x+w*v,axis,x+w*v,axis+6,C.rule)+txt(x+w*v,axis+26,`${v*100}%`,mobile?16:21,C.muted,'middle'));
  s+=txt(mobile?200:550,mobile?342:435,'Training rows, relative to the full dataset',mobile?16:22,C.muted,'middle');
  if(!mobile)s+=txt(65,90,'Fig. 5a',21,C.muted);
  return s;
}

function render(stage,progress,mobile) {
  const n=Math.max(0,Math.min(2,Math.round(Number(stage)||0))),p=clamp(progress);
  const s=n===0?generator(p,mobile):n===1?architecture(p,mobile):efficiency(p,mobile);
  const title=['Synthetic task generation: a teaching analogue','TabPFN table attention: architectural schematic','Nature Figure 5a: relative training-data requirement'][n];
  return `<g font-family="inherit" data-nature-stage="${n}" data-nature-progress="${p}" data-nature-mobile="${mobile}"><title>${title}</title>${s}</g>`;
}

/** SVG contents for viewBox="0 0 900 460". */
export function renderNature(stage=0,progress=1) { return render(stage,progress,false); }
/** SVG contents for viewBox="0 0 400 350". */
export function renderNatureMobile(stage=0,progress=1) { return render(stage,progress,true); }
