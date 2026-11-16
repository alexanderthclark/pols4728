/* Exact, reversible visual states. The scroll engine owns time and navigation. */
const C = {ink:'#242424', prediction:'#234E70', expert:'#A54F32', corrected:'#007F73', muted:'#606060', rule:'#767676', pale:'#e7eaec'};
const clamp = x => Math.min(1,Math.max(0,x));
const lerp = (a,b,p) => a+(b-a)*p;
const esc = s => String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const f = (x,d=3) => Number(x).toFixed(d);
const avg = xs => xs.reduce((a,b)=>a+b,0)/xs.length;

export function regressionAt(rows, progress=0, shift=0) {
  const xs=rows.map(r=>lerp(r.q,r.x,progress)+shift), ys=rows.map(r=>r.y);
  const mx=avg(xs), my=avg(ys), variance=avg(xs.map(x=>x*x))-mx*mx;
  const slope=(avg(xs.map((x,i)=>x*ys[i]))-mx*my)/variance;
  return {slope,intercept:my-slope*mx,meanX:mx,meanY:my,variance};
}

function text(x,y,s,size=14,color=C.ink,anchor='start',extra='') {
  return `<text x="${x}" y="${y}" fill="${color}" font-size="${size}" text-anchor="${anchor}" font-family="Arial,Helvetica,sans-serif" ${extra}>${esc(s)}</text>`;
}
function multiline(x,y,s,maxChars=28,size=14,color=C.ink,anchor='start',lineHeight=20) {
  const lines=[];let current='';
  for(const word of s.split(' ')){if(current.length+word.length+1>maxChars&&current){lines.push(current);current=word;}else current+=(current?' ':'')+word;}
  if(current)lines.push(current);
  return lines.map((s,i)=>text(x,y+i*lineHeight,s,size,color,anchor)).join('');
}
function line(x1,y1,x2,y2,color=C.rule,width=1,dash='') {
  return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}"${dash?` stroke-dasharray="${dash}"`:''}/>`;
}
function point(x,y,color=C.prediction,size=4,shape='circle',hollow=false,opacity=1) {
  const fill=hollow?'white':color;
  if(shape==='square')return `<rect x="${x-size}" y="${y-size}" width="${size*2}" height="${size*2}" fill="${fill}" stroke="${color}" stroke-width="1.5" opacity="${opacity}"/>`;
  if(shape==='diamond')return `<path d="M ${x} ${y-size-1} l ${size+1} ${size+1} l ${-size-1} ${size+1} l ${-size-1} ${-size-1} Z" fill="${fill}" stroke="${color}" stroke-width="1.5" opacity="${opacity}"/>`;
  return `<circle cx="${x}" cy="${y}" r="${size}" fill="${fill}" stroke="${color}" stroke-width="1.5" opacity="${opacity}"/>`;
}
function arrow(x1,y1,x2,y2,color=C.corrected,width=1.7) {
  const dx=x2-x1,dy=y2-y1,l=Math.hypot(dx,dy);if(l<2)return '';
  const ux=dx/l,uy=dy/l,head=Math.min(6,l*.45);
  return line(x1,y1,x2,y2,color,width)+`<path d="M ${x2-ux*head-uy*head*.55} ${y2-uy*head+ux*head*.55} L ${x2} ${y2} L ${x2-ux*head+uy*head*.55} ${y2-uy*head-ux*head*.55}" fill="none" stroke="${color}" stroke-width="${width}"/>`;
}
function frame(w,h,description,contents){return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="100%" height="100%" role="img" aria-label="${esc(description)}" style="display:block;overflow:visible"><title>${esc(description)}</title>${contents}</svg>`;}
function mathHTML(tex,size=22){return `<div data-visual-math="${esc(tex)}" style="font-size:${size}px;line-height:1.45;text-align:center;min-width:0"></div>`;}
function title(w,h,p) {
  const compact=h<280,l=30,r=w-30,cy=h*(compact?.42:.50),separation=compact?33:48,range=r-l,start=l+.70*range,end=l+.55*range,xx=lerp(start,end,p);
  let s='';
  // A calibration diagram, not a quantitative data plot.
  if(!compact)s=text(w/2,35,'A prediction is a measurement',w<400?18:23,C.ink,'middle');
  for(let row=0;row<2;row++){
    const yy=cy+(row?separation:-separation);s+=line(l,yy,r,yy,C.ink,1.8);
    for(let j=0;j<=10;j++){const tx=lerp(l,r,j/10);s+=line(tx,yy,tx,yy+(j%5===0?14:8),C.ink,1.2);}
  }
  s+=text(l,cy-separation-22,'Automated measurement',14,C.prediction)+text(l,cy+separation+38,'Expert audit',14,C.expert);
  s+=point(start,cy-separation,C.prediction,7)+point(end,cy+separation,C.expert,7,'diamond',true);
  s+=line(start,cy-separation+12,start,cy+8,C.prediction,1,'3 4');
  s+=arrow(start,cy+12,xx,cy+12,C.corrected,2.4)+point(xx,cy+12,C.corrected,5);
  s+=multiline(w/2,h-(compact?12:36),'Use the audit to correct the estimate.',Math.floor(w/8),compact?13:16,C.ink,'middle');
  return frame(w,h,'Calibration diagram: an automated estimate moves toward the value supported by an expert audit.',s);
}
const procedures={
  llm:{name:'LLM-only analysis',steps:['Predict labels with the LLM','Expert-code a subset of texts','Evaluate and revise predictions','Use predicted labels in the analysis'],end:'Predicted labels enter directly',color:C.prediction},
  supervised:{name:'Classical supervised learning',steps:['Expert-code a training sample','Fit a prediction model','Cross-validate and tune','Use predictions in the analysis'],end:'Predictions enter directly',color:C.prediction},
  dsl:{name:'Design-based supervised learning',steps:['Predict labels for all texts','Randomly sample expert coding','Optionally improve predictions','Correct the estimating equations'],end:'Predictions + audited discrepancies',color:C.corrected}
};
function workflow(w,h,p,which='dsl') {
  const d=procedures[which]||procedures.dsl,x=21,top=38,bottom=h-56,gap=(bottom-top)/3;
  let s='';
  for(let i=0;i<3;i++)s+=line(x,top+i*gap+13,x,top+(i+1)*gap-13,C.pale,2);
  d.steps.forEach((label,i)=>{
    const yy=top+i*gap,col=i===3?d.color:C.ink;
    s+=point(x,yy,i===3?d.color:C.muted,11,'circle',true)+text(x,yy+4,i+1,12,col,'middle');
    s+=multiline(46,yy+5,label,Math.max(20,Math.floor((w-52)/8.1)),w<400?15:18,col,'start',22);
    if(which==='dsl'&&i===2)s+=text(w-4,yy+27,'Skipped here',12,C.muted,'end');
  });
  const yy=lerp(top,bottom,p),offset=8;
  s+=`<g transform="translate(${x-offset-4},${yy-offset-4})"><rect x="-3" y="-3" width="11" height="14" fill="white" stroke="${d.color}"/><rect width="11" height="14" fill="white" stroke="${d.color}"/><path d="M2 4h7m-7 3h7m-7 3h4" stroke="${d.color}" stroke-width="1"/></g>`;
  s+=text(46,h-14,d.end,w<400?12:14,d.color);
  return frame(w,h,`${d.name}: ${d.steps.map((s,i)=>`${i+1}. ${s}`).join('. ')}`,s);
}
function meanCorpus(w,h,p) {
  const cell=Math.min((w-40)/10,(h-90)/10),x0=(w-10*cell)/2,y0=45;
  let s=text(w/2,24,'10,000 statements',18,C.ink,'middle');
  for(let i=0;i<100;i++){
    const x=x0+(i%10)*cell,y=y0+Math.floor(i/10)*cell,shown=clamp(p*100-i+1);
    s+=`<rect x="${x+2}" y="${y+2}" width="${cell-4}" height="${cell-4}" rx="1" fill="${i<40?C.prediction:C.pale}" opacity="${lerp(.15,1,shown)}"/>`;
  }
  s+=text(w/2,h-20,'40% predicted support',18,C.prediction,'middle');
  s+=text(w/2,h-1,'Each square represents 100 statements.',12,C.muted,'middle');
  return frame(w,h,'100 squares represent 10,000 statements; 40 filled blue squares are predicted supporters.',s);
}
function meanAudit(w,h,p) {
  const left=w<400?98:125,top=74,right=w-10,bottom=h-55,cw=(right-left)/2,ch=(bottom-top)/2;
  let s=text(w/2,21,'1,000 randomly audited statements',w<400?15:19,C.ink,'middle');
  s+=text(left+cw,49,'LLM label',13,C.prediction,'middle');
  s+=text(left+cw*.5,top-8,'Support',13,C.prediction,'middle')+text(left+cw*1.5,top-8,'Other',13,C.prediction,'middle');
  s+=text(left-10,top+ch*.5-7,'Expert:',12,C.expert,'end')+text(left-10,top+ch*.5+11,'Support',13,C.expert,'end');
  s+=text(left-10,top+ch*1.5-7,'Expert:',12,C.expert,'end')+text(left-10,top+ch*1.5+11,'Other',13,C.expert,'end');
  const cells=[{x:0,y:0,n:270,label:'Agree',error:false},{x:1,y:0,n:30,label:'Missed support',error:true},{x:0,y:1,n:90,label:'False support',error:true},{x:1,y:1,n:610,label:'Agree',error:false}];
  for(const c of cells){const xx=left+c.x*cw,yy=top+c.y*ch;
    s+=`<rect x="${xx}" y="${yy}" width="${cw}" height="${ch}" fill="${c.error?'#fcf3ee':'#f5f6f6'}" stroke="white" stroke-width="3"/>`;
    s+=text(xx+cw/2,yy+ch*(h<260?.68:.47),c.n,h<260?20:w<400?24:34,c.error?C.expert:C.ink,'middle');
    if(h>=260)s+=multiline(xx+cw/2,yy+ch*.47+23,c.label,Math.floor(cw/7),12,C.muted,'middle',15);
    if(c.error&&p>.15)s+=line(xx+8,yy+ch-5,xx+8+(cw-16)*clamp((p-.15)/.7),yy+ch-5,C.expert,2);
  }
  s+=text(w/2,h-28,'Expert support: 300 / 1,000 = 30%',13,C.expert,'middle');
  s+=text(w/2,h-8,'LLM support: 360 / 1,000 = 36%',13,C.prediction,'middle');
  return frame(w,h,'Audit confusion matrix: 270 true positives, 30 false negatives, 90 false positives, and 610 true negatives. Expert support 30%; LLM support 36%.',s);
}
function meanCorrect(w,h,p) {
  const l=18,r=w-46,y1=h*.34,y2=h*.64,x=v=>l+v/.5*(r-l),value=lerp(.4,.34,p);
  let s=text(l,30,'Prediction + audit discrepancy',w<400?17:21);
  s+=text(l,y1-24,'All statements',14,C.prediction)+line(l,y1,x(.4),y1,C.prediction,13)+text(x(.4)+9,y1+5,'40%',15,C.prediction);
  s+=text(l,y2-24,'Corrected estimate',14,C.corrected)+line(l,y2,x(value),y2,C.corrected,13)+text(x(value)+9,y2+5,`${(100*value).toFixed(p===1?0:1)}%`,15,C.corrected);
  s+=arrow(x(.4),y1+22,x(value),y1+22,C.expert,2)+text(l,y2+38,'Audit: (30 − 90) / 1,000 = −0.06',14,C.expert);
  const axis=h-31;s+=line(l,axis,r,axis,C.rule);
  for(const v of [0,.1,.2,.3,.4,.5])s+=text(x(v),axis+21,`${Math.round(v*100)}%`,12,C.muted,'middle');
  return frame(w,h,`The full-corpus predicted mean of 40% is corrected downward by six percentage points to 34%. Current displayed estimate ${100*value}%.`,s);
}
function meanReason(w,h,p) {
  const compact=h<280,l=30,r=w-30,zero=w/2,spread=Math.min(94,w*.27),y=compact?h*.5+6:h*.48;
  let s=text(w/2,28,'The audit estimates the discrepancy',w<400?16:21,C.ink,'middle');
  s+=line(l,y,r,y,C.rule,1.3)+line(zero,y-(compact?39:75),zero,y+(compact?40:76),C.rule,1,'4 5');
  s+=text(l,y-(compact?53:94),'LLM overstates',13,C.expert)+text(r,y-(compact?53:94),'LLM understates',13,C.expert,'end');
  if(!compact)s+=text(zero,y+104,'Expert minus LLM',14,C.ink,'middle');
  for(let i=0;i<12;i++){
    const sign=i<9?-1:1,column=i<9?Math.floor(i/3)-1:0,xx=zero+sign*spread+column*17,yy=y+(i%3-1)*20;
    s+=point(lerp(zero,xx,p),yy,C.expert,5,sign<0?'square':'circle');
  }
  s+=text(zero-spread,y+(compact?49:68),'90 false positives',12,C.expert,'middle')+text(zero+spread,y+(compact?49:68),'30 false negatives',12,C.expert,'middle');
  s+=text(w/2,h-12,'One mark = 10 audited cases',13,C.muted,'middle');
  return frame(w,h,'Nine squares represent 90 false positives and three circles represent 30 false negatives. Each mark represents 10 audit cases. Net expert-minus-LLM difference is negative.',s);
}

function regression(w,h,p,source,kind) {
  const rows=source.rows,l=42,r=w-15,t=43,b=h-57,xmax=kind==='regression-shift'?6.5:6;
  const x=v=>l+v/xmax*(r-l),y=v=>b-v/45*(b-t),move=kind==='regression-reveal'?p:0,shift=kind==='regression-shift'?.5*p:0;
  let s=line(l,t,l,b,C.rule)+line(l,b,r,b,C.rule);
  for(const v of [0,2,4,6])s+=text(x(v),b+22,v,12,C.muted,'middle');
  for(const v of [0,20,40])s+=text(l-8,y(v)+4,v,12,C.muted,'end');
  s+=text(l,t-18,'Engagement',14,C.ink);
  rows.forEach(row=>{
    const audit=kind==='regression-audit',xx=audit?lerp(row.q,row.audited?row.x:row.q,p):lerp(row.q,row.x,move)+shift;
    const shape=row.z===1?'square':'circle',color=audit&&row.audited?C.expert:kind==='regression-reveal'&&p>.75?C.expert:C.prediction;
    if((audit&&row.audited)||kind==='regression-reveal')s+=line(x(row.q),y(row.y),x(xx),y(row.y),audit?C.expert:C.rule,1);
    s+=point(x(xx),y(row.y),color,w<400?3.5:4.5,shape,audit&&row.audited,audit&&!row.audited?lerp(1,.18,p):1);
  });
  if(kind!=='regression-audit'){
    const fit=regressionAt(rows,move,shift),xl=.2+shift,xr=5.8+shift;
    s+=line(x(xl),y(fit.intercept+fit.slope*xl),x(xr),y(fit.intercept+fit.slope*xr),kind==='regression-reveal'&&p>.75?C.ink:C.prediction,2.3,kind==='regression-auto'?'6 3':'');
    s+=text(r,24,`Slope ${f(fit.slope)}`,w<400?16:20,kind==='regression-reveal'&&p>.75?C.ink:C.prediction,'end');
  }else s+=text(r,24,'16 audited / 40 posts',14,C.expert,'end');
  const xlabel=kind==='regression-reveal'?(p===1?'Expert hostility':p===0?'Automated hostility':'Moving from automated to expert score'):kind==='regression-shift'?`Automated score + ${f(shift,2)}`:kind==='regression-audit'?'Hostility score':'Automated hostility';
  s+=text((l+r)/2,h-18,xlabel,13,C.ink,'middle');
  s+=text((l+r)/2,h-1,kind==='regression-shift'?'Hypothetical shift; actual mean correction = 0':'○ restrained   □ emphatic',11,C.muted,'middle');
  return frame(w,h,kind==='regression-audit'?'Forty posts, with expert coordinates revealed for sixteen audited posts. Other posts retain automated coordinates.':`Engagement against hostility score. ${xlabel}. Fitted slope ${f(regressionAt(rows,move,shift).slope)}.`,s);
}

function moment(w,h,p,source,kind) {
  const index={'moment-mean':0,'moment-square':1,'moment-cross':2}[kind],moment=source.moments[index],audit=source.rows.filter(r=>r.audited);
  const fn=index===0?(x,y)=>x:index===1?(x,y)=>x*x:(x,y)=>x*y;
  const values=audit.flatMap(r=>[fn(r.q,r.y),fn(r.x,r.y)]),max=Math.max(...values)*1.08;
  const names=['Score','Squared score','Score × engagement'];
  if(h<270){
    const l=18,r=w-36,mx=Math.max(moment.automated,moment.corrected)*1.18,x=v=>l+v/mx*(r-l),y1=62,y2=111,total=moment.automated+moment.correction*p;
    let s=text(l,21,`${names[index]}: average over all posts`,w<400?14:18);
    s+=text(l,y1-16,'Automated moment',13,C.prediction)+line(l,y1,x(moment.automated),y1,C.prediction,9)+text(x(moment.automated)+6,y1+4,f(moment.automated,index===0?0:2),12,C.prediction);
    s+=text(l,y2-16,'Corrected moment',13,C.corrected)+line(l,y2,x(total),y2,C.corrected,9)+text(x(total)+6,y2+4,f(total,index===1?4:index===2?2:3),12,C.corrected);
    s+=text(l,151,`Audit correction: ${moment.correction<0?'−':'+'}${f(Math.abs(moment.correction)*p,index===1?4:index===2?2:3)}`,16,C.expert);
    s+=text(l,h-12,'All 16 paired expert–LLM differences enter.',12,C.muted);
    return frame(w,h,`${names[index]}: automated full-corpus moment ${moment.automated}; audit correction ${moment.correction}; corrected moment ${moment.corrected}. Compact view shows the aggregate correction using all sixteen expert-audit pairs.`,s);
  }
  const l=39,r=w-24,t=43,b=h-113,x=v=>l+v/max*(r-l),rowGap=(b-t)/15;
  let s=text(l,21,'Paired values in the 16-post audit',w<400?14:18);
  for(let i=0;i<audit.length;i++){
    const row=audit[i],yy=t+i*rowGap,q=fn(row.q,row.y),expert=fn(row.x,row.y),now=lerp(q,expert,p);
    s+=line(x(q),yy,x(now),yy,C.expert,1.3)+point(x(q),yy,C.prediction,2.8,'circle',true)+point(x(now),yy,C.expert,3,'diamond');
    if(i%4===0||i===15)s+=text(l-9,yy+4,row.post,11,C.muted,'end');
  }
  s+=line(l,b+13,r,b+13,C.rule);
  const ticks=index===0?[0,2,4,6]:index===1?[0,10,20,30]:[0,80,160,240];
  ticks.filter(v=>v<=max).forEach(v=>s+=text(x(v),b+29,v,11,C.muted,'middle'));
  s+=text((l+r)/2,b+46,names[index],13,C.ink,'middle');
  s+=text(l,b+64,'○ Automated',12,C.prediction)+text(r,b+64,'◆ Expert',12,C.expert,'end');
  const correction=moment.correction*p,total=moment.automated+correction,digits=index===1?4:index===2?2:3;
  const math=`${f(moment.automated,index===0?0:2)} ${correction<0?'-':'+'} ${f(Math.abs(correction),digits)} = ${f(total,digits)}`;
  s+=text(w/2,h-8,math,w<400?18:24,C.corrected,'middle');
  return frame(w,h,`${names[index]}: automated full-corpus moment ${moment.automated}; audit correction ${moment.correction}; corrected moment ${moment.corrected}. All sixteen audited paired values are shown; row labels give post IDs.`,s);
}

function assemble(w,h,p,source) {
  const numerator=lerp(7,4.25,p),denominator=lerp(2.25,2.0625,p);
  if(h<260)return `<div style="width:100%;height:100%;display:flex;flex-direction:column;justify-content:center;gap:10px;text-align:center">${mathHTML(`\\text{Covariance: }${f(lerp(85,82.25,p),2)}-3(26)=${f(numerator,2)}`,17)}${mathHTML(`\\text{Variance: }${f(lerp(11.25,11.0625,p),4)}-9=${f(denominator,4)}`,16)}<div style="padding-top:8px;border-top:1px solid ${C.pale}">${mathHTML(`\\widehat\\beta_{\\mathrm{DSL}}=\\frac{${f(numerator,2)}}{${f(denominator,4)}}=${f(numerator/denominator)}`,23)}</div></div>`;
  return `<div style="width:100%;height:100%;display:flex;flex-direction:column;justify-content:center;gap:${h<300?13:22}px;text-align:center">
    <div style="font:13px Arial;color:${C.muted}">Corrected covariance</div>
    ${mathHTML(`${w<400?'':'M_{XY}-M_X\\overline Y='}${f(lerp(85,82.25,p),2)}-3(26)=${f(numerator,2)}`,w<400?20:23)}
    <div style="font:13px Arial;color:${C.muted}">Corrected variance</div>
    ${mathHTML(`${w<400?'':'M_{X^2}-M_X^2='}${f(lerp(11.25,11.0625,p),4)}-3^2=${f(denominator,4)}`,w<400?18:23)}
    <div style="padding-top:12px;border-top:1px solid ${C.pale}">${mathHTML(`\\widehat\\beta_{\\mathrm{DSL}}=\\frac{${f(numerator,2)}}{${f(denominator,4)}}=${f(numerator/denominator)}`,w<400?24:34)}</div>
    <div style="font:12px Arial;color:${C.muted}">The outcome mean uses all 40 posts.</div>
  </div>`;
}
function compare(w,h,p,source) {
  const l=w<400?127:165,r=w-48,t=38,b=h-59,x=v=>l+(v-1.4)/2*(r-l),gap=(b-t-20)/3;
  let s=line(l,b,r,b,C.rule)+line(x(2),t-22,x(2),b,C.ink,1,'4 4');
  const entries=[['All expert',source.fits.oracle.slope,C.ink,'diamond'],['LLM labels',source.fits.automated.slope,C.prediction,'circle'],['Expert audit',source.fits.expert_only.slope,C.expert,'square'],['DSL',lerp(source.fits.automated.slope,source.fits.dsl.slope,p),C.corrected,'diamond']];
  entries.forEach(([name,value,color,shape],i)=>{const yy=t+gap*i;s+=text(l-12,yy+5,name,w<400?13:17,color,'end')+point(x(value),yy,color,5,shape)+text(x(value)+9,yy-11,f(value),w<400?12:16,color);});
  for(const v of [1.5,2,2.5,3])s+=text(x(v),b+22,v.toFixed(1),12,C.muted,'middle');
  s+=text((l+r)/2,h-12,'Regression slope',13,C.ink,'middle');
  return frame(w,h,'Slope estimates: full-expert benchmark 2.000; automated labels 3.111; expert-only audit 1.706; DSL 2.061.',s);
}
function uncertainty(w,h,p,source) {
  const l=15,r=w-52,x=v=>l+v/1.4*(r-l),y1=h*.31,y2=h*.64;
  let s=text(l,23,'Estimated population standard error',w<400?15:20);
  for(const [label,value,color,yy] of [['Expert audit',source.population_se.expert_only,C.expert,y1],['DSL',source.population_se.dsl,C.corrected,y2]]){
    const xx=x(value),reveal=x(value*p);
    s+=text(l,yy-22,label,16,color)+line(l,yy,reveal,yy,color,11);
    s+=text(xx+9,yy+5,f(value,2),16,color);
  }
  const axis=h-42;s+=line(l,axis,r,axis,C.rule);
  for(const v of [0,.5,1])s+=text(x(v),axis+21,f(v,1),12,C.muted,'middle');
  s+=text(w/2,h-3,'Corpus sampling + expert-audit sampling',w<400?12:14,C.muted,'middle');
  return frame(w,h,'Estimated population regression standard errors, allowing both posts and audit selection to vary: expert-only 1.158; DSL 0.802.',s);
}
function closing(w,h,p) {
  const items=[['Predictions','All documents',C.prediction],['Expert discrepancies','Random audit',C.expert],['Corrected moments','Downstream analysis',C.corrected]];
  const x=w*.5,top=40,bottom=h-35,gap=(bottom-top)/2;let s='';
  items.forEach(([label,caption,color],i)=>{const yy=top+i*gap;s+=text(x,yy,label,w<400?20:27,color,'middle')+text(x,yy+24,caption,13,C.muted,'middle');if(i<2)s+=arrow(x,yy+35,x,yy+35+(gap-55)*clamp(p*2-i+1),C.rule,1.3);});
  return frame(w,h,'Use predictions from all documents, estimate discrepancies in the random expert audit, and form corrected moments for downstream analysis.',s);
}

/** All teaching motion is a deterministic function of progress in [0,1]. */
export function renderVisual(container, kind, progress, source, options={}) {
  const p=options.reduced?1:clamp(progress),w=Math.max(260,Math.floor(container.clientWidth||640));
  const h=Math.max(180,Math.min(500,Math.floor(container.clientHeight||(options.mobile?300:430))));
  const renderers={
    title:()=>title(w,h,p),workflow:()=>workflow(w,h,p,options.workflow),
    'mean-corpus':()=>meanCorpus(w,h,p),'mean-audit':()=>meanAudit(w,h,p),
    'mean-correct':()=>meanCorrect(w,h,p),'mean-reason':()=>meanReason(w,h,p),
    'regression-auto':()=>regression(w,h,p,source,kind),'regression-reveal':()=>regression(w,h,p,source,kind),
    'regression-audit':()=>regression(w,h,p,source,kind),'regression-shift':()=>regression(w,h,p,source,kind),
    'moment-mean':()=>moment(w,h,p,source,kind),'moment-square':()=>moment(w,h,p,source,kind),
    'moment-cross':()=>moment(w,h,p,source,kind),'slope-assemble':()=>assemble(w,h,p,source),
    'slope-compare':()=>compare(w,h,p,source),uncertainty:()=>uncertainty(w,h,p,source),closing:()=>closing(w,h,p)
  };
  container.innerHTML=(renderers[kind]||renderers.closing)();
  container.dataset.kind=kind;container.dataset.progress=p.toFixed(4);
  for(const el of container.querySelectorAll('[data-visual-math]')){
    const tex=el.getAttribute('data-visual-math');
    if(globalThis.katex)globalThis.katex.render(tex,el,{displayMode:true,throwOnError:true});
    else el.textContent=tex;
  }
}
