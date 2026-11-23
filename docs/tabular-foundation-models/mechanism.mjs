import {miniPfnResults} from './mini-pfn-results.mjs';

// Real values from the accompanying trained educational model. This is not a
// trace from TabPFN. Scroll reveals computations; it does not retrain a model.
const C={ink:'#242424',blue:'#234E70',rust:'#A54F32',teal:'#007F73',muted:'#606060',rule:'#767676'};
const clamp=x=>Math.max(0,Math.min(1,Number.isFinite(x)?x:1));
const num=(x,d=3)=>Number(x).toFixed(d).replace('-','−');
const pct=x=>`${(100*x).toFixed(1)}%`;
const vec=v=>`[${num(v[0],2)}, ${num(v[1],2)}, …]`;
const text=(x,y,s,z=22,c=C.ink,a='middle')=>`<text x="${x}" y="${y}" font-size="${z}" fill="${c}" text-anchor="${a}">${s}</text>`;
const rect=(x,y,w,h,c=C.rule,fill='white',r=4)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" stroke="${c}" stroke-width="1.5" fill="${fill}"/>`;
const line=(x1,y1,x2,y2,c=C.blue,width=2)=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${c}" stroke-width="${width}"/>`;
const grow=(x1,y1,x2,y2,p,c=C.blue)=>line(x1,y1,x1+(x2-x1)*clamp(p),y1+(y2-y1)*clamp(p),c);
const group=(s,opacity=1)=>`<g opacity="${opacity}">${s}</g>`;
const A=miniPfnResults.examples.A, B=miniPfnResults.examples.B;
const update=miniPfnResults.first_update;
export const recordedTrainingTasks={
  first:{context:update.context,query:update.query,query_outcome:update.query_outcome},
  // Reproduced from sample_tasks(256, Generator().manual_seed(4729)), index 1.
  // This is another task in the first minibatch, not another optimizer step.
  second:{context:[[1,1,0],[1,0,0],[1,0,0],[1,0,0]],query:[0,1],query_outcome:1}
};

function table(example,x,y,w,h=36,query=true,small=false){
  const width=w/3,z=small?17:21,base=small?23:26;
  let s='';
  (w<250?['B','L','Y']:['Brown','Lie down','Survived']).forEach((v,j)=>s+=text(x+(j+.5)*width,y+base,v,small?16:20,j===2?C.teal:C.ink));
  example.context.forEach((row,i)=>{
    const yy=y+(i+1)*h;
    s+=line(x,yy,x+w,yy,C.rule,1);
    row.forEach((v,j)=>s+=text(x+(j+.5)*width,yy+base,v,z,j===2?C.teal:C.blue));
  });
  let bottom=y+5*h;
  if(query){
    s+=rect(x,bottom,w*2/3,h,C.blue,'#eef4f8',0);
    example.query.forEach((v,j)=>s+=text(x+(j+.5)*width,bottom+base,v,z,C.blue));
    s+=text(x+2.5*width,bottom+base,'?',z,C.rust);
    s+=`<path d="M${x} ${y} H${x+w} V${bottom} H${x+w*2/3} V${bottom+h} H${x} Z" fill="none" stroke="${C.blue}" stroke-width="2"/>`;
  }else s+=rect(x,y,w,5*h,C.blue,'none',0);
  return s;
}

function worlds(p,m){
  const high=1/(1+Math.exp(-1)), low=1-high;
  let s='';
  ['A','B'].forEach((name,j)=>{
    const x=m?j*200+59:j*425+150,y=m?100:115,w=m?65:125,h=m?70:95;
    const selected=j===0?clamp(p*2):0;
    s+=rect(m?j*200+7:j*425+38,m?28:30,m?186:390,m?245:325,selected>.1?C.blue:C.rule,'white',6);
    s+=text(m?j*200+100:j*425+233,m?56:70,`World ${name}`,m?22:30,selected>.1?C.blue:C.muted);
    s+=text(x+w/2,y-15,'Fight',m?16:22)+text(x+w*1.5,y-15,m?'Lie down':'Lie down',m?16:22);
    ['Black','Brown'].forEach((bear,b)=>{
      s+=text(x-8,y+(b+.5)*h+6,bear,m?16:22,C.ink,'end');
      for(let a=0;a<2;a++){
        const probability=(b===a)===(name==='A')?high:low;
        s+=rect(x+a*w,y+b*h,w,h,C.rule,probability>.5?'#edf6f3':'#faf0eb',0);
        s+=text(x+(a+.5)*w,y+(b+.5)*h+8,pct(probability),m?20:32,probability>.5?C.teal:C.rust);
      }
    });
  });
  s+=text(m?200:450,m?315:412,'Draw one world for the entire table.',m?20:27,C.blue);
  return s;
}

function task(p,m){
  const x=m?35:70,y=m?28:50,w=m?330:390,h=m?33:46;
  let s=table(recordedTrainingTasks.first,x,y,w,h,true,m);
  const targetX=x+w*5/6,targetY=y+5*h+h/2;
  const endX=m?200:710,endY=m?308:190;
  s+=text(m?200:265,m?18:365,'Blue outline: model input',m?17:22,C.blue);
  s+=group(rect(endX-(m?90:125),endY-28,m?180:250,58,C.rust,'#fff9f6')+text(endX,endY+8,`Withheld Y = ${update.query_outcome}`,m?21:27,C.rust),clamp(p*2));
  s+=grow(targetX,targetY,m?endX:endX-125,m?endY-32:endY,p,C.rust);
  if(!m)s+=text(710,292,'Used only to score the prediction',21,C.muted);
  return s;
}

function loss(p,m){
  // Prediction and loss are discrete recorded states, never interpolated values.
  const after=p>=.6;
  const probability=after?update.first_query_probability_after:update.first_query_probability_before;
  const meanLoss=after?update.batch_log_loss_after:update.batch_log_loss_before;
  const state=after?'After one Adam step':'Before the first update';
  const data=`data-update-state="${after?'after':'before'}" data-query-probability="${probability}" data-batch-log-loss="${meanLoss}" data-batch-size="${update.batch_size}"`;
  if(m){
    let s=text(200,22,state,21,C.blue)+rect(70,42,260,58,C.blue,'#f7fafc')+text(200,66,'Shared network weights',18,C.blue)+text(200,88,'Same Brown + Lie down query',16,C.muted);
    s+=text(200,139,`One query: q = ${num(probability,4)}`,26,C.teal);
    s+=text(200,176,`Observed outcome: Y = ${update.query_outcome}`,19,C.rust);
    s+=text(200,221,`Mean loss over ${update.batch_size} tasks`,19,C.rust)+text(200,255,num(meanLoss,4),29,C.rust);
    s+=group(`<path d="M308 250 H366 V70 H340" fill="none" stroke="${C.rust}" stroke-width="2"/>`,clamp(p*2));
    s+=text(200,305,'256 losses → one Adam update',19,C.rust)+text(200,337,'Mean losses measured on the same batch.',16,C.muted);
    return `<g ${data}>${s}</g>`;
  }
  let s=text(450,35,state,29,C.blue)+rect(45,105,230,100,C.blue,'#f7fafc')+text(160,146,'Shared network',27,C.blue)+text(160,180,'Same context + query',20,C.muted);
  s+=line(290,155,410,155)+text(530,126,'Brown + Lie down',22,C.blue)+text(530,176,`q = ${num(probability,4)}`,34,C.teal);
  s+=text(775,145,`Y = ${update.query_outcome}`,32,C.rust)+text(775,182,'Observed outcome',20,C.muted);
  s+=text(605,270,`Mean of ${update.batch_size} task losses = ${num(meanLoss,4)}`,28,C.rust);
  s+=group(`<path d="M620 303 V365 H160 V221" fill="none" stroke="${C.rust}" stroke-width="2.5"/>`,clamp(p*2));
  s+=text(450,415,'Average 256 losses; backpropagate; take one Adam step.',24,C.rust);
  return `<g ${data}>${s}</g>`;
}

function reversed(p,m){
  const second=p>=.5,e=second?recordedTrainingTasks.second:recordedTrainingTasks.first;
  const name=second?'Task 2':'Task 1';
  const pair=second?'Black + Lie down':'Brown + Lie down';
  let s=text(m?200:245,m?22:35,`${name} of 256`,m?21:28,C.blue);
  s+=table(e,m?35:50,m?40:65,m?330:390,m?32:43,true,m);
  if(m){s+=text(200,279,`Query: ${pair}`,21,C.blue)+text(200,308,`Withheld outcome: Y = ${e.query_outcome}`,19,C.rust)+text(200,339,'Both tasks enter the same batch update.',16,C.muted);}
  else{s+=grow(460,200,580,200,p<.5?p*2:(p-.5)*2)+text(720,145,'New query features',23,C.muted)+text(720,196,pair,27,C.blue)+text(720,260,`Withheld outcome: Y = ${e.query_outcome}`,23,C.rust)+text(450,415,'Each table contributes one query loss to the minibatch.',25,C.blue);}
  return `<g data-training-task="${second?2:1}">${s}</g>`;
}

function evaluation(p,m){
  const e=miniPfnResults.evaluation,rows=[['Ignore context',e.ignore_context_log_loss,C.muted],['Trained mini-PFN',e.model_log_loss,C.blue],['Exact Bayes',e.bayes_log_loss,C.teal]];
  const left=m?35:325,right=m?365:820,top=m?85:90,spacing=m?75:100;
  let s=text(m?200:450,m?24:30,`${e.n_tasks.toLocaleString('en-US')} new test tasks`,m?21:27,C.blue);
  rows.forEach(([label,value,c],i)=>{
    const yy=top+i*spacing,visibility=i===0?1:clamp((p-(i-1)*.3)/.35);
    const barWidth=(right-left)*value/.8;
    s+=group(text(m?left:45,m?yy-12:yy+15,label,m?18:24,c,'start')+rect(left,yy,barWidth,m?18:26,c,c,0)+text(m?390:880,m?yy+15:yy+21,num(value),m?18:24,c,'end'),visibility);
  });
  const y=m?291:382;s+=line(left,y,right,y,C.rule,1);
  [0,.4,.8].forEach(v=>s+=text(left+(right-left)*v/.8,y+26,num(v,1),m?16:20,C.muted));
  s+=text(m?200:450,m?345:448,'Log loss · lower is better',m?18:23,C.muted);
  return s;
}

function learning(p,m){
  const data=miniPfnResults.training, left=m?52:110,top=m?45:60,w=m?310:675,h=m?215:290;
  const yMax=.75,yMin=.58, maxStep=data.at(-1).step;
  const px=step=>left+w*step/maxStep, py=value=>top+h*(yMax-value)/(yMax-yMin);
  let s=text(m?200:450,m?21:28,'Validation log loss',m?19:26,C.blue);
  s+=line(left,top,left,top+h,C.rule,1.5)+line(left,top+h,left+w,top+h,C.rule,1.5);
  [.6,.65,.7,.75].forEach(v=>s+=text(left-10,py(v)+6,num(v,2),m?16:20,C.muted,'end'));
  [0,maxStep/2,maxStep].forEach(v=>s+=text(px(v),top+h+25,String(v),m?16:20,C.muted));
  const count=Math.min(data.length,1+Math.floor(p*(data.length-1))),shown=data.slice(0,count),last=shown.at(-1);
  s+=`<polyline points="${shown.map(d=>`${px(d.step)},${py(d.loss)}`).join(' ')}" fill="none" stroke="${C.blue}" stroke-width="3"/>`;
  shown.forEach(d=>s+=`<circle cx="${px(d.step)}" cy="${py(d.loss)}" r="4" fill="${C.blue}"/>`);
  s+=text(m?200:450,m?315:408,`Update ${last.step}: ${num(last.loss)}`,m?21:28,C.blue);
  s+=text(m?200:450,m?343:448,'One shared set of weights across tasks',m?17:23,C.muted);
  return s;
}

function frozen(p,m){
  const chosen=p<.5?'A':'B';
  let s='';
  if(m){
    s+=text(200,21,'Same query: Brown + Lie down',18,C.blue);
    [A,B].forEach((e,j)=>{
      const x=15+j*200;
      s+=text(x+85,48,`Context ${j?'B':'A'}`,19,chosen===(j?'B':'A')?C.blue:C.muted);
      s+=table(e,x,58,170,27,false,true);
    });
    s+=rect(135,220,130,54,C.blue,'#f7fafc')+text(200,255,'θ fixed',25,C.blue);
    s+=grow(chosen==='A'?100:300,202,200,214,1);
    s+=text(95,321,`A: ${num(A.probability)}`,22,C.teal)+text(305,321,`B: ${num(B.probability)}`,22,C.teal);
    s+=grow(200,282,chosen==='A'?95:305,297,clamp(p<.5?p*2:(p-.5)*2),C.teal);
  }else{
    [A,B].forEach((e,j)=>{
      const y=40+j*205;
      s+=text(170,y-12,`Context ${j?'B':'A'}`,22,chosen===(j?'B':'A')?C.blue:C.muted);
      s+=table(e,45,y,250,34,false,false);
      s+=text(770,y+72,`q${j?'B':'A'} = ${num(e.probability)}`,32,C.teal);
    });
    s+=rect(420,160,220,110,C.blue,'#f7fafc')+text(530,204,'Same network',27,C.blue)+text(530,246,'θ fixed',29,C.blue);
    const cy=chosen==='A'?122:327;s+=grow(310,cy,405,215,1)+grow(650,215,715,cy,clamp(p<.5?p*2:(p-.5)*2),C.teal);
    s+=text(530,390,'Only the labeled context changes.',24,C.muted);
  }
  return s;
}

function embedding(p,m){
  const row=A.context[0],vector=A.row_embeddings[0];
  let s=text(m?200:450,m?25:35,'Black + Fight + Survived',m?22:29,C.blue);
  const x=m?65:70,y=m?72:155,w=m?90:82;
  ['B','L','Y'].forEach((label,j)=>{
    s+=text(x+(j+.5)*w,y-12,label,m?18:24,j===2?C.teal:C.blue);
    s+=rect(x+j*w,y,w,m?48:62,j===2?C.teal:C.blue);
    s+=text(x+(j+.5)*w,y+(m?32:41),String(row[j]),m?25:32,j===2?C.teal:C.blue);
  });
  if(m){s+=grow(200,132,200,174,p*2)+rect(115,185,170,56,C.blue,'#f7fafc')+text(200,221,'Row MLP',24,C.blue)+grow(200,252,200,275,(p-.4)/.6);s+=group(text(200,305,vec(vector),25,C.blue)+text(200,337,'First 2 of 32 learned coordinates',17,C.muted),clamp((p-.4)/.6));}
  else{s+=grow(333,186,400,186,p*2)+rect(420,135,170,104,C.blue,'#f7fafc')+text(505,195,'Row MLP',29,C.blue)+grow(608,186,648,186,(p-.4)/.6);s+=group(text(765,196,vec(vector),28,C.blue)+text(745,251,'32 coordinates',23,C.muted),clamp((p-.4)/.6));s+=text(450,393,'The row’s features and known label enter together.',25,C.muted);}
  return s;
}

function attention(p,m){
  const active=Math.min(4,Math.floor(p*4+1e-8)),rowY=m?79:112,gap=m?39:55;
  const xs=m?[48,150,292]:[135,365,675];
  let s=text(xs[0],m?30:52,m?'[B,L,Y]':'Context row [B,L,Y]',m?17:23,C.blue)+text(xs[1],m?30:52,'Weight',m?17:23,C.blue)+text(xs[2],m?30:52,m?'Value (2 / 32)':'Value: first 2 of 32 coordinates',m?17:23,C.blue);
  const sum=[0,0];
  A.context.forEach((r,i)=>{
    const y=rowY+i*gap,c=i<active?C.blue:C.muted;
    if(i<active){sum[0]+=A.weighted_values[i][0];sum[1]+=A.weighted_values[i][1];}
    s+=text(xs[0],y,`[${r.join(',')}]`,m?18:25,c)+text(xs[1],y,num(A.attention[i]),m?18:25,c)+text(m?208:500,y,'×',m?18:25,c)+text(xs[2],y,vec(A.values[i]),m?18:25,c);
    const left=xs[1]-(m?25:50),width=m?50:100;
    s+=line(left,y+10,left+width*A.attention[i],y+10,c,3);
  });
  const pooled=active===4?A.pooled:sum;
  s+=line(m?25:70,m?231:325,m?375:830,m?231:325,C.rule,1.5);
  s+=text(m?200:450,m?269:371,`${active<4?'Partial sum':'Weighted sum'} (${active}/4)`,m?19:25,C.teal);
  s+=text(m?200:450,m?301:409,vec(pooled),m?25:31,C.teal);
  s+=text(m?200:450,m?341:450,'Attention weights are not feature importance.',m?16:21,C.muted);
  return s;
}

function decode(p,m){
  if(m){
    let s=text(200,21,'Concatenate two 32-number vectors',18,C.blue);
    s+=text(200,49,'Query representation (32)',17,C.muted)+text(200,76,vec(A.query_embedding),23,C.blue);
    s+=text(200,108,'Pooled context (32)',17,C.muted)+text(200,135,vec(A.pooled),23,C.blue);
    s+=text(200,167,'[query; context] · 64 inputs',18,C.blue)+grow(200,175,200,192,p*2);
    s+=rect(90,200,220,47,C.blue,'#f7fafc')+text(200,231,'Decoder MLP',24,C.blue);
    s+=group(text(200,280,`Logit = ${num(A.logit)}`,23,C.ink),clamp(p*2));
    s+=group(text(200,324,`Sigmoid: q = ${num(A.probability)}`,26,C.teal),clamp((p-.35)/.65));
    return s;
  }
  let s=text(160,80,'Query representation (32)',22,C.blue)+text(160,122,vec(A.query_embedding),27,C.blue);
  s+=text(160,196,'Pooled context (32)',22,C.blue)+text(160,238,vec(A.pooled),27,C.blue);
  s+=`<path d="M290 112 H320 V238 H290" fill="none" stroke="${C.blue}" stroke-width="2"/>`;
  s+=text(390,130,'Concatenate',22,C.blue)+text(390,164,'64 inputs',22,C.blue)+grow(330,195,450,195,p*2);
  s+=rect(470,145,190,100,C.blue,'#f7fafc')+text(565,203,'Decoder MLP',26,C.blue);
  s+=grow(677,195,725,195,p*2)+group(text(800,142,'Logit',24,C.muted)+text(800,199,num(A.logit),37,C.ink),clamp(p*2));
  s+=group(grow(800,222,800,282,(p-.35)/.65,C.teal)+text(680,267,'Sigmoid',23,C.muted)+text(780,335,`q = ${num(A.probability)}`,37,C.teal),clamp((p-.35)/.65));
  return s+text(450,424,'[Query representation; pooled context] keeps all 64 coordinates.',23,C.blue);
}

const stages=[worlds,task,loss,reversed,evaluation,learning,frozen,embedding,attention,decode];
export function renderMechanism(stage,progress=1,mobile=false){
  if(!stages[stage])throw new RangeError('Unknown mini-PFN mechanism scene');
  const p=clamp(progress);
  return `<g data-mechanism-stage="${stage}" data-mechanism-progress="${p}" data-mechanism-mobile="${mobile}" data-model="educational-mini-pfn">${stages[stage](p,mobile)}</g>`;
}
