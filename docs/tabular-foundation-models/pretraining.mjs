import {scmFamilies,scmTasks} from './pretraining-data.mjs?v=20261123-scm-pretraining';
import {bearComparisonResults} from './bear-comparison-results.mjs?v=20261123-context-ten-noise';
const C={ink:'#242424',blue:'#234E70',rust:'#A54F32',teal:'#007F73',muted:'#606060',rule:'#767676',pale:'#E8EFF4',paper:'#FFFFFF'};
const clamp=x=>Math.max(0,Math.min(1,Number.isFinite(x)?x:1));
const mix=(a,b,t)=>a+(b-a)*t;
const smooth=x=>{const t=clamp(x);return t*t*(3-2*t);};
const f=x=>Number(x.toFixed(3));
const esc=x=>String(x).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const text=(x,y,s,size=20,color=C.ink,anchor='middle',attrs='')=>`<text x="${f(x)}" y="${f(y)}" font-size="${size}" fill="${color}" text-anchor="${anchor}" ${attrs}>${esc(s)}</text>`;
const rect=(x,y,w,h,color=C.rule,fill=C.paper,attrs='')=>`<rect x="${f(x)}" y="${f(y)}" width="${f(w)}" height="${f(h)}" fill="${fill}" stroke="${color}" stroke-width="1.3" ${attrs}/>`;
const line=(x1,y1,x2,y2,color=C.rule,width=1.4,attrs='')=>`<line x1="${f(x1)}" y1="${f(y1)}" x2="${f(x2)}" y2="${f(y2)}" stroke="${color}" stroke-width="${width}" ${attrs}/>`;
function arrow(x1,y1,x2,y2,color=C.blue,width=1.5){const a=Math.atan2(y2-y1,x2-x1),h=5;return line(x1,y1,x2,y2,color,width)+`<path d="M${f(x2-h*Math.cos(a-.6))},${f(y2-h*Math.sin(a-.6))} L${f(x2)},${f(y2)} L${f(x2-h*Math.cos(a+.6))},${f(y2-h*Math.sin(a+.6))}" fill="none" stroke="${color}" stroke-width="${width}"/>`;}
function packet(a,b,p,color=C.teal,size=6){const t=smooth(p);return rect(mix(a[0],b[0],t)-size/2,mix(a[1],b[1],t)-size/2,size,size,color,color,'data-moving-packet="true"');}
function polyPacket(points,p,color=C.teal,size=6){const lengths=points.slice(1).map((v,i)=>Math.hypot(v[0]-points[i][0],v[1]-points[i][1]));let d=lengths.reduce((s,n)=>s+n,0)*smooth(p),i=0;while(i<lengths.length-1&&d>lengths[i])d-=lengths[i++];const t=lengths[i]?d/lengths[i]:1;return packet(points[i],points[i+1],t,color,size);}
function graph(id,x,y,w,h,p,mobile=false){
  const k=w/190,dy=h/88,r=mobile?11:17;
  const defs={
    chain:{nodes:[['X₁',15,38],['X₂',95,38],['Y',175,38]],edges:[[0,1],[1,2]]},
    latent:{nodes:[['H',95,7],['X₁',15,66],['X₂',66,66],['X₃',120,66],['Y',175,66]],edges:[[0,1],[0,2],[0,3],[0,4]]},
    nonlinear:{nodes:[['X₁',25,8],['X₂',145,8],['Y',85,69],['X₃',177,69]],edges:[[0,2],[1,2]]},
    interaction:{nodes:[['X₁',25,10],['X₂',165,10],['Y',95,69]],edges:[[0,2],[1,2]]}
  };
  const d=defs[id],nodes=d.nodes.map(([name,xx,yy])=>({name,x:x+xx*k,y:y+yy*dy}));let s=`<g data-scm-family="${id}">`;
  for(const [a,b] of d.edges){const A=nodes[a],B=nodes[b],ang=Math.atan2(B.y-A.y,B.x-A.x);const start=[A.x+(r+2)*Math.cos(ang),A.y+(r+2)*Math.sin(ang)],end=[B.x-(r+3)*Math.cos(ang),B.y-(r+3)*Math.sin(ang)];s+=arrow(...start,...end,C.blue,1.4)+packet(start,end,p,C.teal,mobile?4:6);}
  for(const n of nodes){const latent=n.name==='H',color=n.name==='Y'?C.teal:latent?C.muted:C.blue;s+=`<circle cx="${f(n.x)}" cy="${f(n.y)}" r="${r}" fill="white" stroke="${color}" stroke-width="1.6" ${latent?'stroke-dasharray="3 2"':''}/>`+text(n.x,n.y+(mobile?4.5:6.5),n.name,mobile?14:20,color);}
  return s+'</g>';
}
function gallery(p,mobile){
  let s=text(mobile?200:450,mobile?22:27,'Different rules, different tables',mobile?20:25);
  const positions=mobile?[[14,49],[214,49],[14,178],[214,178]]:[[31,90],[250,90],[469,90],[688,90]];
  scmFamilies.forEach((family,i)=>{const [x,y]=positions[i],w=mobile?171:180;s+=text(x+w/2,y,family.name,mobile?17:22,C.blue);s+=graph(family.id,x,y+17,w,mobile?64:100,clamp(p*1.5-i*.12),mobile);s+=text(x+w/2,y+(mobile?100:154),family.formula,mobile?14:16,C.muted);if(!mobile)s+=text(x+w/2,y+181,family.targetFormula,15,C.muted);});
  if(mobile){s+=text(200,301,'Synthetic: TabPFN (2025) / TabICLv2',15,C.blue)+text(200,324,'Collected real tables: TabDPT',15,C.rust)+text(200,345,'Illustrative SCMs; independent input noise',14,C.muted);}
  else{s+=line(90,351,810,351,C.rule,1)+text(450,385,'Synthetic tables: TabPFN (2025) / TabICLv2',22,C.blue)+text(450,418,'Collected real tables: TabDPT',22,C.rust)+text(450,452,'Original illustrative SCMs, not recovered checkpoint-training samples. Noise inputs are omitted.',16,C.muted);}
  return s;
}
const num=x=>Number.isInteger(x)?String(x):x.toFixed(2).replace(/^(-?)0\./,'$1.');
function table(task,x,y,w,rh,mobile,{compact=false}={}){
  const rowLabels=compact?['1','New']:['1','2','3','4','New'];
  const records=compact?[task.context[0],task.query]:[...task.context,task.query];
  const headers=['Row',...task.featureNames,'Y'],cw=w/headers.length,head=rh;
  let s=`<g data-model-input="true" data-task-family="${task.familyId}">`+rect(x,y,w,head+records.length*rh,C.blue);
  for(let j=1;j<headers.length;j++)s+=line(x+j*cw,y,x+j*cw,y+head+records.length*rh,C.rule,.8);
  headers.forEach((h,j)=>{s+=text(x+(j+.5)*cw,y+head*.7,h,mobile?(compact?14:17):(compact?16:21),C.muted);});
  records.forEach((row,i)=>{const yy=y+head+i*rh,isQuery=i===records.length-1;s+=line(x,yy,x+w,yy,isQuery?C.blue:C.rule,isQuery?1.5:.8);const cells=[rowLabels[i],...row.features.map(value=>mobile&&compact?'·':num(value)),isQuery?'?':String(row.target)];cells.forEach((v,j)=>{s+=text(x+(j+.5)*cw,yy+rh*.72,v,mobile?(compact?14:17):(compact?16:21),j===cells.length-1?(isQuery?C.rust:C.teal):C.ink,'middle',j===cells.length-1&&isQuery?'data-query-target="hidden"':'');});});
  return s+'</g>';
}
function predictionTask(p,mobile){
  const index=Math.min(3,Math.floor(p*4)),task=scmTasks[index],family=scmFamilies[index],local=p===1?1:p*4-index;
  let s=text(mobile?200:160,mobile?18:36,family.name,mobile?20:25,C.blue,'middle',`data-selected-family="${task.familyId}"`);
  if(mobile){
    s+=graph(family.id,109,38,182,44,local,true);
    s+=text(200,106,'Model input: context + new features',18,C.blue);
    s+=table(task,16,119,368,27,true);
    s+=text(200,305,'Kept outside the model input:',16,C.muted);
    s+=text(200,331,`Withheld target Y* = ${task.scoringTarget}`,20,C.rust,'middle','data-scoring-target="true"');
    s+=`<path d="M200,88 H6 V146 H12" fill="none" stroke="${C.blue}" stroke-width="1.3"/>`+arrow(6,146,12,146)+polyPacket([[200,88],[6,88],[6,146],[12,146]],local,C.teal,4);
  }else{
    s+=graph(family.id,40,112,240,132,local,false);
    s+=text(160,307,family.formula,20,C.muted)+text(160,340,'SCM generates the table',20,C.muted);
    s+=arrow(295,218,349,218)+packet([296,218],[345,218],local);
    s+=text(550,57,'Model input',25,C.blue);
    s+=table(task,360,82,380,43,false);
    s+=text(550,384,'First four rows: labeled context',21,C.muted)+text(550,418,'Last row: new features, hidden target',21,C.muted);
    s+=text(821,131,'For scoring',21,C.rust)+text(821,166,'only',21,C.rust);
    s+=line(766,190,876,190,C.rust,1.4,'stroke-dasharray="4 4"');
    s+=text(821,236,`Y* = ${task.scoringTarget}`,28,C.rust,'middle','data-scoring-target="true"');
    s+=text(821,276,'Outside',20,C.muted)+text(821,304,'the input',20,C.muted);
  }
  return s;
}
function network(x,y,w,h,p,mobile,learning){
  let s=rect(x,y,w,h,C.blue,C.paper,`data-weight-status="${learning?'learning':'fixed'}"`);
  const xs=[x+w*.2,x+w*.5,x+w*.8],ys=[y+h*.27,y+h*.45,y+h*.63];
  for(let k=0;k<2;k++)for(let i=0;i<3;i++)for(let j=0;j<3;j++){const weight=learning?mix(.7,1+((i+j+k)%3)*.7,p):1.3;s+=line(xs[k],ys[i],xs[k+1],ys[j],C.blue,weight,`opacity="${f(learning?mix(.2,.7,p):.45)}"`);}
  for(let k=0;k<3;k++)for(const yy of ys)s+=`<circle cx="${f(xs[k])}" cy="${f(yy)}" r="${mobile?3.5:5}" fill="${k===2?C.teal:C.blue}"/>`;
  return s+text(x+w/2,y+h*.91,learning?'Weights learned here':'Weights fixed',mobile?17:21,learning?C.rust:C.blue,'middle','data-weight-label="true"');
}
function learning(p,mobile){
  let s='';const tasks=scmTasks.slice(0,3);
  if(mobile){
    tasks.forEach((task,i)=>{const x=8+i*133;s+=text(x+59,21,`Task ${i+1}`,17,C.blue)+table(task,x,32,118,23,true,{compact:true});if(i===1){s+=`<path d="M200,106 V114 H360 V178 H340" fill="none" stroke="${C.blue}" stroke-width="1.5"/>`+arrow(350,178,340,178)+polyPacket([[200,106],[200,114],[360,114],[360,178],[340,178]],clamp(p*1.6-i*.2),C.teal,5);}else{const endX=i===0?80:320;s+=arrow(x+59,106,endX,151)+packet([x+59,106],[endX,151],clamp(p*1.6-i*.2),C.teal,5);}});
    s+=text(200,141,'One shared network',20,C.blue)+network(64,157,272,85,p,true,true);
    s+=arrow(200,246,200,266,C.teal)+packet([200,247],[200,266],clamp((p-.25)/.75),C.teal,5);
    s+=text(200,287,'Compare predictions with held-out targets',16,C.ink)+text(200,312,`q₁, q₂, q₃   versus   ${tasks.map(t=>t.scoringTarget).join(', ')}`,18,C.rust,'middle','data-training-comparison="true"');
    s+=`<path d="M338,304 H380 V198 H340" fill="none" stroke="${C.rust}" stroke-width="1.5"/>`+arrow(353,198,340,198,C.rust)+text(310,336,'Update weights',16,C.rust);
    s+=polyPacket([[338,304],[380,304],[380,198],[340,198]],clamp((p-.5)/.5),C.rust,5);
  }else{
    tasks.forEach((task,i)=>{const yy=38+i*126;s+=text(143,yy+1,`Task ${i+1} · ${scmFamilies[i].name}`,21,C.blue)+table(task,46,yy+14,195,29,false,{compact:true});s+=arrow(248,yy+54,355,164+i*42)+packet([248,yy+54],[355,164+i*42],clamp(p*1.5-i*.15));});
    s+=text(485,113,'One shared network',25,C.blue)+network(363,137,244,172,p,false,true);
    s+=arrow(615,218,684,218,C.teal)+packet([617,218],[680,218],clamp((p-.25)/.75));
    s+=text(776,159,'Predictions',24,C.teal)+text(776,191,'q₁, q₂, q₃',25,C.teal);
    s+=text(776,246,'Compare with',22,C.ink)+text(776,278,'withheld targets',22,C.ink)+text(776,314,tasks.map(t=>t.scoringTarget).join('     '),26,C.rust,'middle','data-training-comparison="true"');
    s+=`<path d="M776,330 V390 H485 V317" fill="none" stroke="${C.rust}" stroke-width="1.5"/>`+arrow(485,336,485,317,C.rust);
    s+=text(632,421,'Update the same shared weights',22,C.rust)+polyPacket([[776,330],[776,390],[485,390],[485,317]],clamp((p-.5)/.5),C.rust);
  }
  return s;
}
function bearInput(mobile){
  const x=mobile?14:24,y=mobile?52:89,w=mobile?372:390,rh=mobile?25:40,headers=['B','L','Z₁','…','Z₁₀','Y'];
  const rows=[['0','0','·','…','·','1'],['0','1','·','…','·','0'],['1','1','·','…','·','1'],['1','1','·','…','·','?']];
  let s=`<g data-bear-input="schematic">`+text(x+w/2,mobile?25:39,'Bear context + new encounter',mobile?20:26,C.blue)+text(x+w/2,mobile?45:68,'Shape of the input; noise values omitted',mobile?14:18,C.muted)+rect(x,y,w,5*rh,C.blue);
  const cw=w/6;headers.forEach((v,j)=>{if(j)s+=line(x+j*cw,y,x+j*cw,y+5*rh,C.rule,.8);s+=text(x+(j+.5)*cw,y+rh*.72,v,mobile?17:22,C.muted);});
  rows.forEach((row,i)=>{const yy=y+(i+1)*rh;s+=line(x,yy,x+w,yy,i===3?C.blue:C.rule,i===3?1.5:.8);row.forEach((v,j)=>{s+=text(x+(j+.5)*cw,yy+rh*.72,v,mobile?18:23,j===5?(i===3?C.rust:C.teal):C.ink,'middle',i===3&&j===5?'data-query-target="hidden"':'');});});return s+'</g>';
}
function transfer(p,mobile){
  const result=bearComparisonResults.cases.find(x=>x.model==='tabicl_v2'&&x.id==='continuous_noise');
  let s=bearInput(mobile);
  if(mobile){
    s+=`<path d="M200,181 H42 V242 H64" fill="none" stroke="${C.blue}" stroke-width="1.5"/>`+arrow(54,242,64,242)+polyPacket([[200,181],[42,181],[42,242],[64,242]],p,C.teal,5);
    s+=text(200,202,'Pretrained TabICLv2',20,C.blue)+network(68,214,264,60,1,true,false);
    s+=arrow(200,278,200,293,C.teal)+text(200,315,'Query survival probability',19,C.teal,'middle','data-schematic-prediction="true"');
    s+=text(200,345,`Held-out AUC ${result.mean_auc.toFixed(3)} · mean of 12 runs`,16,C.blue,'middle',`data-benchmark-auc="${result.mean_auc}"`);
  }else{
    s+=arrow(423,212,478,212)+packet([424,212],[475,212],p);
    s+=text(613,102,'Pretrained TabICLv2',26,C.blue)+network(490,128,245,170,1,false,false);
    s+=arrow(743,212,770,212,C.teal)+packet([745,212],[767,212],p,C.teal);
    s+=text(842,164,'Query',23,C.teal)+text(842,194,'survival',23,C.teal)+text(842,224,'probability',23,C.teal,'middle','data-schematic-prediction="true"');
    s+=line(100,354,800,354,C.rule,1)+text(450,391,`Separate held-out benchmark: mean test AUC ${result.mean_auc.toFixed(3)}`,25,C.blue,'middle',`data-benchmark-auc="${result.mean_auc}"`);
    s+=text(450,426,'8,000 training rows · 2,000 test rows · 12 simulation seeds',21,C.muted);
  }
  return s;
}
/** SVG contents; parent chooses 900×460 desktop or 400×350 mobile viewBox. */
export function renderPretraining(stage=0,progress=1,mobile=false){
  const k=Math.max(0,Math.min(3,Math.trunc(Number.isFinite(stage)?stage:0))),p=clamp(progress);
  const title=['Varied mechanisms and distinct pretraining data sources','A generated table becomes a context-and-query prediction task','Shared network weights learn from held-out targets across tasks','A broadly pretrained model makes a new bear prediction with fixed weights'][k];
  const desc=['Original illustrative SCMs show a chain, a shared latent cause, a nonlinear rule with a distractor, and an interaction. Packets follow their arrows as scrolling changes. Noise inputs are omitted from the graphs. Synthetic and collected real-table sources are different model recipes, not a claim that every model uses both.','Scroll selects four deterministic original SCM examples and their exact generated tables. The top four rows provide labeled context. The last row supplies query features but its target stays a question mark. The generated true target appears outside the model-input table solely for scoring. Displayed features are rounded.','Different generated tables supply separate prediction tasks to one shared network. Unnumbered model outputs are compared with actual generated held-out targets; a loss-driven update changes the shared weights. Connection widths and moving packets are schematic, not recorded neural training.','The input shape is schematic; shown bear rows and omitted noise values are not a recovered benchmark context. All ten noise columns remain part of the measured task. TabICLv2 uses its pretrained checkpoint with fixed weights. The unnumbered query prediction is distinct from the separate, actual twelve-seed held-out mean ROC AUC.'][k];
  const body=[gallery,predictionTask,learning,transfer][k](p,mobile);
  return `<g data-pretraining-stage="${k}" data-pretraining-progress="${p}" data-pretraining-mobile="${Boolean(mobile)}"><title>${title}</title><desc>${desc}</desc>${body}</g>`;
}
