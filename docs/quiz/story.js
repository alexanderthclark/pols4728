import {coarsePrediction, randomModelSelection, sampleCarnies, carnieMoments, integratedCarnies, tutoringRows, tutoringSummary} from './model.mjs';
import {createMotionView} from './animate.js?v=6';
import {sceneProgress, activeScene} from './motion.mjs';

document.body.classList.remove('no-js');
const $ = id => document.getElementById(id);
const visual = $('visual');
const steps = [...document.querySelectorAll('.step')];
const mobile = window.matchMedia('(max-width: 700px) and (orientation: portrait)');
let compact = false;
const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
let motionPreference = null;
const steppedMotion = () => motionPreference ?? reduced.matches;
const colors = ['var(--blue)','var(--rust)','var(--teal)','var(--purple)','var(--ochre)','var(--sky)','var(--rose)','var(--muted)'];
const patterns = ['', '7 3', '2 3', '8 3 2 3', '10 4', '1 3', '5 2 1 2', '3 3', '8 2 2 2 2 2', '12 3 2 3'];
const rows = tutoringRows();
const answer = tutoringSummary();
const riskCache = new Map();
let draws = sampleCarnies();
let active = 0;
let serial = 0;
let scheduled = false;
let lastPosition = {index:0, fraction:0};
let mountedScene = -1;
let noiseFrom = 1;
let noiseLast = Number($('noise')?.value||100000);
const sceneInfo = [
 ['Question 1 · a','A fair guess at every observation','Each guess matches either fixed target label with probability ½. The particular 20 guesses shown are illustrative.'],
 ['Question 1 · b','Search can make noise look perfect','The bars give expected candidate counts, not observed counts. Selection uses validation labels; evaluation uses untouched test labels.'],
 ['Question 1 · c','Evaluate the whole selection procedure','Four outer folds illustrate repeated selection. The inner observations all serve model selection; no parameters require training.'],
 ['Question 2 · a','Ten training samples, three predictors','Thin lines are ten simulated fits. Dashed gray is the population mean y = x. Carnie 3 transforms the same ten fits as Carnie 1. All full-range panels share scales.'],
 ['Question 2 · b','Low noise does not ensure a stable slope','The bands show ±2 prediction standard deviations across training samples. Both mean predictions equal x; the second slope is much less stable.'],
 ['Question 2 · c','Average fixed-x quantities over the population','Bias and variance are calculated across training samples at each x, then averaged over uniform X. The two panels use different units.'],
 ['Question 2 · c','A small change can cross a reporting threshold','Detail view near x = 365 days. Carnie 3 turns tiny differences in fitted lines into 365-day jumps. These panels intentionally have different vertical scales.'],
 ['Question 2 · d','Compare total prediction error','All three total MSEs include the same 3.65 days² of outcome noise. Bars use a common linear scale; direct values retain the small errors.'],
 ['Question 2 · d','At very large noise, bounded reports can win','Total MSE includes fresh outcome noise. The horizontal scale is logarithmic; equal distances represent equal ratios.'],
 ['Question 3 · a(i)','One student per leaf removes every residual','Eight schematic students, with randomly assigned tutoring. Start at their overall mean, ȳ = 75. Each leaf then memorizes its student and the residual shrinks to zero.'],
 ['Question 3 · a(ii)','Hold out students before fitting the tree','The evaluation outcomes never set the fitted predictions. Random assignment makes the expected adjustment the same in both evaluation groups.'],
 ['Question 3 · b(i)','A 30-point gap contains a 10-point effect','Now use the 32-student observational example. Every square is one student. More tutored students have high baseline scores, adding a 20-point baseline difference.'],
 ['Question 3 · b(ii)','The outcome tree learns two group means','A split at 55 separates the low and high score groups. The 45-point prediction gap includes 40 baseline points and 5 points from different tutoring rates.'],
 ['Question 3 · b(iii)','Outcome adjustment alone leaves a 7.5-point gap','Marker area represents student count. Comparing outcome residuals by the original tutoring indicator yields 3.75 − (−3.75) = 7.5.'],
 ['Question 3 · b(iv)','Adjust tutoring on the same score groups','The remaining outcome and treatment variation satisfies R = 10T in all four groups. Marker area represents the number of students.'],
 ['Question 3 · b(iv)','Weight the four rows by their student counts','Each group contributes once per student. The residual regression without an intercept gives ΣTR / ΣT² = 60 / 6 = 10.']
];

const escapeText = value => String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
const fmt = (value, digits=2) => Number(value).toLocaleString('en-US',{maximumFractionDigits:digits,minimumFractionDigits:digits});
const tick = value => Math.abs(value)>=1000000 ? `${value/1000000}m` : Math.abs(value)>=1000 ? `${value/1000}k` : String(value);
const text = (x,y,label,attrs='') => `<text x="${x}" y="${y}" ${attrs}>${escapeText(label)}</text>`;
const line = (x1,y1,x2,y2,attrs='') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${attrs}/>`;
const rect = (x,y,w,h,attrs='') => `<rect x="${x}" y="${y}" width="${Math.max(0,w)}" height="${Math.max(0,h)}" ${attrs}/>`;
const circle = (x,y,r,attrs='') => `<circle cx="${x}" cy="${y}" r="${r}" ${attrs}/>`;
const path = (points,attrs='') => `<path d="${points.map((p,i)=>`${i?'L':'M'}${p[0]},${p[1]}`).join(' ')}" ${attrs}/>`;
const enter = (markup,start=0,end=1,dx=0,dy=12) => `<g data-enter="${start},${end},${dx},${dy}">${markup}</g>`;
function wipe(markup,x,y,width,height,start=0,end=1){
 const id=`wipe-${serial++}`;
 return `<defs><clipPath id="${id}">${rect(x,y,width,height,`data-wipe="${width},${start},${end}"`)}</clipPath></defs><g clip-path="url(#${id})">${markup}</g>`;
}
const specAttr = object => JSON.stringify(object).replaceAll('"','&quot;');
function probe(p,w,h,model,zoom=false,kind='prediction',stroke='var(--blue)'){
 const spec={model,kind,x0:p.x(zoom?358:0),x1:p.x(zoom?372:730),y0:p.y(0),unit:p.y(1)-p.y(0),lo:zoom?358:0,hi:zoom?372:730,start:zoom?.15:.65};
 const attrs=`data-probe-mark="${model===2?'square':'circle'}" fill="white" stroke="${stroke}" stroke-width="2"`;
 return `<g data-probe="${specAttr(spec)}">${p.clip(line(0,p.top,0,p.bottom,'data-guide="" stroke="var(--rule)" stroke-dasharray="2 3"'))}${model===2?rect(0,0,9,9,attrs):circle(0,0,4.5,attrs)}</g>`;
}
function formulaAttrs(w,extra=''){
 return `class="formula-line" ${w<380?`style="font-size:${w<270?16:17}px!important"`:''} ${extra}`;
}
function svg(w,h,title,body,attrs=''){
 return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" role="img" aria-label="${escapeText(title)}" ${attrs}><title>${escapeText(title)}</title>${body}</svg>`;
}
function plot(w,h,{top=30,left=58,right=20,bottom=46,xmin=0,xmax=730,ymin=0,ymax=730,xticks=[0,365,730],yticks=[0,365,730],xlabel='x (days)',ylabel='Predicted age (days)',xformat=tick,yformat=tick,xlabelAttrs=''}={}){
 const x=v=>left+(v-xmin)/(xmax-xmin)*(w-left-right);
 const y=v=>h-bottom-(v-ymin)/(ymax-ymin)*(h-top-bottom);
 const id=`plot-${serial++}`;
 let base=`<defs><clipPath id="${id}">${rect(left,top,w-left-right,h-top-bottom)}</clipPath></defs>`;
 base+=line(left,top,left,h-bottom,'class="plot-frame"')+line(left,h-bottom,w-right,h-bottom,'class="plot-frame"');
 xticks.forEach(v=>{base+=line(x(v),h-bottom,x(v),h-bottom+5,'class="plot-tick"')+text(x(v),h-bottom+20,xformat(v),'text-anchor="middle"');});
 yticks.forEach(v=>{base+=line(left-5,y(v),left,y(v),'class="plot-tick"')+text(left-10,y(v)+4,yformat(v),'text-anchor="end" data-y-tick=""');});
 if(xlabel)base+=text((left+w-right)/2,h-3,xlabel,`text-anchor="middle" class="axis-title" ${xlabelAttrs}`);
 if(ylabel)base+=text(left,15,ylabel,'class="axis-title"');
 return {x,y,top,bottom:h-bottom,base,clip:inner=>`<g clip-path="url(#${id})">${inner}</g>`};
}
function chance(w){
 let b='';const left=compact?68:92,cw=(w-left-8)/10;
 const target=[0,1,1,0,1,0,0,1,1,0,1,0,1,1,0,0,1,0,1,0];
 const guess =[1,1,0,0,1,1,0,0,1,0,0,0,1,0,0,1,1,1,0,1];
 for(let block=0;block<2;block++){
  const y=(compact?17:28)+block*(compact?86:105),targetY=y+(compact?27:35),guessY=y+(compact?54:68);
  b+=text(left,y,`Observations ${block*10+1}–${block*10+10}`,'class="small"');
  b+=text(0,targetY,'Target y')+text(0,guessY,'Guess X');
  for(let i=0;i<10;i++){
   const n=block*10+i,cx=left+(i+.5)*cw;
   b+=text(cx,targetY,target[n],'text-anchor="middle"');
   b+=enter(rect(cx-cw*.42,guessY-19,cw*.84,25,`fill="white" stroke="${guess[n]===target[n]?'var(--blue)':'var(--rule)'}" ${guess[n]===target[n]?'':'stroke-dasharray="2 2"'}`)+text(cx,guessY,guess[n],'text-anchor="middle"'),n*.035,n*.035+.11,0,-14);
  }
 }
 b+=enter(text(w/2,compact?205:280,'P(guess matches y) = ½',`text-anchor="middle" ${formulaAttrs(w)}`)+text(w/2,compact?237:315,'Expected test accuracy: 50%','text-anchor="middle" class="label"'),.8,1,0,12);
 return svg(w,compact?255:340,'Twenty illustrative fair binary guesses against a fixed target. Every guess has a one-half chance of matching.',b);
}
function selection(w){
 const n=Number($('validation')?.value||10),r=randomModelSelection(32768,n),h=compact?270:350,ph=compact?205:265;
 let binomial=1;const counts=[];
 for(let k=0;k<=n;k++){if(k)binomial*=(n-k+1)/k;counts.push(32768*binomial*2**(-n));}
 const max=Math.max(...counts)*1.2;
 const p=plot(w,ph,{top:compact?43:38,left:compact?45:57,right:15,bottom:43,xmin:-.5,xmax:n+.5,ymax:max,xticks:[0,Math.floor(n/2),n],yticks:[0,Math.round(max/2000)*1000],xlabel:`Correct validation predictions (of ${n})`,ylabel:'Expected candidate count'});
 let b=p.base;
 counts.forEach((v,k)=>{const bw=(p.x(1)-p.x(0))*.7;b+=rect(p.x(k)-bw/2,p.y(v),bw,p.y(0)-p.y(v),`data-search-bar="${p.y(0)},${p.y(0)-p.y(v)}" fill="${k===n?'var(--rust)':'var(--blue)'}"`);});
 b+=text(w-15,34,`Perfect: ${fmt(r.expectedPerfect,r.expectedPerfect<1?2:0)} expected`,`data-search-perfect="${r.expectedPerfect}" text-anchor="end"`);
 b+=line(10,compact?218:291,w-10,compact?218:291,'stroke="var(--rule)"');
 b+=text(w/2,compact?239:318,`${20-n} untouched test observations`,'text-anchor="middle"');
 b+=text(w/2,compact?261:343,'Expected test accuracy: 50%','text-anchor="middle" class="label"');
 return svg(w,h,`Searching 32768 random candidates on ${n} validation observations yields ${fmt(r.expectedPerfect)} expected perfect candidates, but expected test accuracy stays 50 percent.`,b);
}
function nested(w){
 const left=compact?54:75,cw=(w-left-8)/4;let b='';
 for(let row=0;row<4;row++){
  const y=(compact?20:42)+row*(compact?43:62),rh=compact?31:42;
  b+=text(0,y+(compact?20:25),`Fold ${row+1}`);
  let tiles='';for(let col=0;col<4;col++){
   tiles+=rect(left+col*cw,y,cw-5,rh,`fill="white" stroke="${col===row?'var(--rust)':'var(--blue)'}" ${col===row?'stroke-width="2"':'stroke-dasharray="4 3"'}`);
   tiles+=text(left+col*cw+(cw-5)/2,y+(compact?20:25),col===row?'Test':'Select','text-anchor="middle"');
  }b+=enter(tiles,row*.2,row*.2+.2,-18,0);
 }
 b+=text(w/2,compact?220:332,'Select inside → evaluate outside','text-anchor="middle" class="label"');
 b+=text(w/2,compact?247:361,'Average the outer held-out accuracies','text-anchor="middle"');
 return svg(w,compact?265:385,'Four outer folds: choose a candidate on the other fifteen observations, then evaluate on the five held-out observations.',b);
}
function curves(w,zoom=false){
 const view=$('curve-view')?.value||'all';
 const models=zoom?[0,2]:view==='all'?[0,1,2]:[{first:0,second:1,third:2}[view]??0];
 const fh=compact?(zoom?130:models.length===3?90:255):(zoom?175:models.length===3?128:310);
 const ymax=zoom?null:Math.max(730,...draws.firstSlopes.map(s=>s*730),...draws.secondSlopes.map(s=>s*730))*1.04;
 let b='';
 for(let panel=0;panel<models.length;panel++){
  const j=models[panel],ymin=zoom?(j===0?355:-8):0,ymaxj=zoom?(j===0?380:385):ymax;
  const top=compact?21:32,bottom=compact?34:panel===models.length-1?43:30;
  const p=plot(w,fh,{top,left:compact?43:58,right:16,bottom,xmin:zoom?358:0,xmax:zoom?372:730,ymin,ymax:ymaxj,xticks:compact&&!zoom&&panel<models.length-1?[]:zoom?[358,365,372]:[0,365,730],yticks:zoom?(j===0?(compact?[360,380]:[360,370,380]):[0,365]):compact?[0,730]:[0,365,730],xlabel:panel===models.length-1?'x (days)':'',ylabel:''});
  let inner=path([[p.x(zoom?358:0),p.y(zoom?358:0)],[p.x(zoom?372:730),p.y(zoom?372:730)]],'class="plot-reference"');
  const slopes=j===1?draws.secondSlopes:draws.firstSlopes;
  slopes.forEach((s,i)=>{
   let pts;
   if(j===2){
    const lo=zoom?358:0,hi=zoom?372:730;
    const cuts=[lo,...[365/s,730/s].filter(v=>s>0&&v>lo&&v<hi),hi].sort((a,c)=>a-c);
    pts=[[p.x(lo),p.y(coarsePrediction(s*lo))]];
    for(let n=1;n<cuts.length;n++){const x=cuts[n];pts.push([p.x(x),p.y(coarsePrediction(s*(x-1e-7)))]);pts.push([p.x(x),p.y(coarsePrediction(s*(x+1e-7)))]);}
   }else pts=[[p.x(zoom?358:0),p.y(s*(zoom?358:0))],[p.x(zoom?372:730),p.y(s*(zoom?372:730))]];
   inner+=wipe(path(pts,`class="data-line" stroke="${colors[i%8]}" stroke-dasharray="${patterns[i]}" ${i===0?'style="stroke-width:2.3"':''}`),p.x(zoom?358:0),top,p.x(zoom?372:730)-p.x(zoom?358:0),p.bottom-top,zoom?0:i*.043,zoom?.12:i*.043+.22);
  });
  const title=(compact?['Carnie 1 · train 0, 730','Carnie 2 · train 0, 1','Carnie 3 · whole years']:['Carnie 1 · training at 0 and 730','Carnie 2 · training at 0 and 1','Carnie 3 · whole-year reports'])[j];
  b+=`<g transform="translate(0 ${panel*fh})">${p.base}${p.clip(inner)}${probe(p,w,fh,j,zoom)}${text(compact?43:58,compact?13:16,title,'class="facet-title"')}</g>`;
 }
 return svg(w,fh*models.length,'Ten simulated fitted curves per displayed carnie. Carnie 3 floors and truncates Carnie 1’s fitted curves. '+(zoom?'Zoom around x equals 365; vertical scales differ.':'The full-range panels share axes.'),b);
}
function slope(w){
 const h=compact?260:330,p=plot(w,h,{left:compact?45:58,bottom:40,ymax:900,yticks:[0,365,730],ylabel:'Mean and ±2 SD (days)'});let b=p.base;
 for(const j of [1,0]){
  const s=2/Math.sqrt(j===0?73000:100),points=[[0,0],[730,730*(1+s)],[730,730*(1-s)],[0,0]].map(([x,y])=>[p.x(x),p.y(y)]);
  b+=p.clip(wipe(`<polygon points="${points.map(pt=>pt.join(',')).join(' ')}" fill="${j?'var(--rust)':'var(--blue)'}" fill-opacity=".1" stroke="${j?'var(--rust)':'var(--blue)'}" stroke-dasharray="${j?'7 4':''}"/>`,p.x(0),p.top,p.x(730)-p.x(0),p.bottom-p.top,0,.45));
  const spec={j,x0:p.x(0),x1:p.x(730),y0:p.y(0),unit:p.y(1)-p.y(0)};
  b+=p.clip(`<g data-sd="${specAttr(spec)}">${line(0,0,0,0,`data-sd-line="" stroke="${j?'var(--rust)':'var(--blue)'}" stroke-width="3"`)}${circle(0,0,4,`data-sd-mean="" fill="${j?'var(--rust)':'var(--blue)'}"`)}</g>`);
 }
 b+=path([[p.x(0),p.y(0)],[p.x(730),p.y(730)]],'stroke="var(--ink)" stroke-width="1.5" fill="none"');
 b+=text(p.x(185),p.y(865),'Carnie 2: broad band');
 b+=path([[p.x(410),p.y(830)],[p.x(470),p.y(770)],[p.x(570),p.y(684)]],'fill="none" stroke="var(--rust)" stroke-dasharray="4 3"');
 b+=text(p.x(205),p.y(150),'Carnie 1: narrow band');
 b+=path([[p.x(205)-6,p.y(150)-4],[p.x(95),p.y(150)-4],[p.x(450),p.y(450*(1-2/Math.sqrt(73000)))]] ,'fill="none" stroke="var(--blue)"');
 return svg(w,h,'Both OLS predictions are unbiased. At every positive x Carnie 2 has 730 times Carnie 1’s prediction variance.',b);
}
function moments(w){
 let b='';const fh=compact?125:180;
 for(let panel=0;panel<2;panel++){
  const top=compact?24:30,p=plot(w,fh,{top,left:compact?45:58,bottom:panel?35:24,ymax:panel?35000:730,yticks:panel?(compact?[0,35000]:[0,17500,35000]):compact?[0,730]:[0,365,730],xticks:panel?[0,365,730]:[],xlabel:panel?'x (days)':'',ylabel:panel?'Variance at fixed x (days²)':'Mean prediction (days)'});
  let inner='';
  for(const j of [0,2]){
   const points=Array.from({length:731},(_,x)=>[p.x(x),p.y(carnieMoments(x)[j][panel?'variance':'mean'])]);
   inner+=wipe(path(points,`fill="none" stroke="${j?'var(--rust)':'var(--blue)'}" stroke-width="2" stroke-dasharray="${j?'7 3':''}"`),p.x(0),top,p.x(730)-p.x(0),p.bottom-top,0,.55);
   inner+=probe(p,w,fh,j,false,panel?'variance':'mean',j?'var(--rust)':'var(--blue)');
  }
  b+=`<g transform="translate(0 ${panel*fh})">${p.base}${p.clip(inner)}</g>`;
 }
 b+=text(w/2,fh*2+(compact?17:20),'Solid: Carnie 1 · Dashed: Carnie 3','text-anchor="middle"');
 return svg(w,fh*2+(compact?25:35),'Carnie 1’s mean equals x and has low variance. Carnie 3 is biased downward and has variance spikes at the reporting thresholds.',b);
}
function getRisk(m){if(!riskCache.has(m))riskCache.set(m,integratedCarnies(m));return riskCache.get(m);}
function mse(w,large=false){
 const m=large?Number($('noise')?.value||100000):1,risk=getRisk(m);let b='';
 const left=compact?69:80,right=compact?86:90,h=compact?(large?245:220):320,usable=w-left-right;
 const max=Math.max(...risk.map(r=>r.mse),...(large?getRisk(noiseFrom).map(r=>r.mse):[]))*1.12;
 const xmax=large?Math.ceil(Math.log10(max)):max,scale=v=>left+(large?Math.log10(v):v)/xmax*usable;
 const axisY=compact?(large?198:177):254;
 b+=line(left,axisY,w-right,axisY,'class="plot-frame"');
 const logStep=usable<260?3:2;
 const ticks=large?Array.from({length:xmax+1},(_,i)=>i).filter(i=>i===0||i%logStep===0).filter(i=>i===0||xmax-i>=logStep||i===xmax):[0,15000,30000,45000].filter(v=>v<max);
 if(large&&!ticks.includes(xmax))ticks.push(xmax);
 ticks.forEach(v=>{const x=left+v/xmax*usable;b+=line(x,axisY,x,axisY+5,'class="plot-tick"')+text(x,axisY+22,large?`10${['⁰','¹','²','³','⁴','⁵','⁶','⁷','⁸','⁹','¹⁰'][v]||`^${v}`}`:tick(v),'text-anchor="middle"');});
 b+=text(compact?0:left,21,large?`Noise multiplier m = ${fmt(m,0)}`:'Total MSE (with noise)',`class="label" ${large?'data-noise-heading=""':''}`);
 const hatchId=`variance-${serial++}`,dotsId=`noise-${serial++}`;
 b+=`<defs><pattern id="${hatchId}" patternUnits="userSpaceOnUse" width="5" height="5"><rect width="5" height="5" fill="white"/><path d="M-1,1 L1,-1 M0,5 L5,0 M4,6 L6,4" stroke="var(--blue)" stroke-width="1.5"/></pattern><pattern id="${dotsId}" patternUnits="userSpaceOnUse" width="4" height="4"><rect width="4" height="4" fill="white"/><circle cx="2" cy="2" r="1" fill="var(--muted)"/></pattern></defs>`;
 risk.forEach((r,j)=>{
  const y=(compact?55:76)+j*(compact?43:67);b+=text(0,y+5,`Carnie ${j+1}`);
  if(large){const spec={j,left,usable,xmax,from:noiseFrom,to:m};b+=`<g data-noise-row="${specAttr(spec)}">${line(left,y,scale(r.mse),y,'data-noise-line="" stroke="var(--rule)" stroke-dasharray="3 3"')}${circle(scale(r.mse),y,5,`data-noise-dot="" fill="${colors[j]}"`)}${text(scale(r.mse)+9,y+5,fmt(r.mse,r.mse>100000?0:2),'data-noise-value=""')}</g>`;}
  else{
   let accum=0;for(const [k,[key,c]] of [['biasSquared','var(--rust)'],['variance',`url(#${hatchId})`],['noise',`url(#${dotsId})`]].entries()){b+=rect(scale(accum),y-10,scale(accum+r[key])-scale(accum),20,`data-component="${scale(accum+r[key])-scale(accum)},${k}" fill="${c}"`);accum+=r[key];}
   b+=enter(line(scale(r.mse),y-14,scale(r.mse),y+14,'stroke="var(--ink)"'),.85,1,0,0);
  }
  if(!large)b+=enter(text(scale(r.mse)+9,y+5,fmt(r.mse,r.mse>100000?0:2)),.85,1,8,0);
 });
 b+=text(w/2,h-(compact?3:13),large?'Total MSE (days², log scale)':'Total MSE (days², linear scale)','text-anchor="middle" class="axis-title"');
 if(!large){b+=text(compact?0:left,h+16,'Solid: squared bias · Stripes: variance','class="small"');b+=text(compact?0:left,h+33,'Dots: outcome noise','class="small"');}
 return svg(w,h+(large?0:45),`Total MSE at noise multiplier ${m}: ${risk.map((r,i)=>`Carnie ${i+1} ${fmt(r.mse)} days squared`).join('; ')}.`,b);
}
function memorize(w){
 const h=compact?280:350,ph=compact?199:270,p=plot(w,ph,{top:29,left:45,bottom:35,xmin:0,xmax:8,ymin:0,ymax:115,xticks:[1,4,7],yticks:[0,50,100],xlabel:'Student (schematic)',ylabel:'Observed Y / fitted prediction'});let b=p.base;
 const outcomes=Array.from({length:8},(_,i)=>50+40*(i>=4)+10*((i+1)%2));
 const mean=outcomes.reduce((sum,y)=>sum+y,0)/outcomes.length;
 b+=line(p.x(0),p.y(mean),p.x(8),p.y(mean),'data-memorization-mean="75" stroke="var(--rule)" stroke-width="1" stroke-dasharray="5 4"');
 b+=line(p.x(0)-5,p.y(mean),p.x(0),p.y(mean),'class="plot-tick"');
 b+=text(p.x(0)-10,p.y(mean)+4,'ȳ',`text-anchor="end" aria-label="Overall mean outcome, 75" style="font-family:'Course Math',serif;font-size:16px;font-style:italic"`);
 for(let i=1;i<=8;i++){
  const y=outcomes[i-1],x=p.x(i-.4);
  b+=rect(x-8,p.y(y)-8,16,16,'fill="white" stroke="var(--ink)" stroke-width="1.5"');
  b+=line(x,p.y(y),x,p.y(mean),`data-fit-residual="${p.y(mean)},${p.y(y)},${i}" stroke="var(--rust)" stroke-width="1.5" stroke-dasharray="3 2"`);
  b+=line(x-10,p.y(y),x+10,p.y(y),`data-fit="${p.y(mean)},${p.y(y)},${i}" stroke="var(--blue)" stroke-width="2"`);
 }
 const ky=compact?220:296;
 b+=rect(8,ky-9,10,10,'fill="white" stroke="var(--ink)"')+text(24,ky,'Observed Y');
 b+=line(w*.49,ky-4,w*.49+19,ky-4,'stroke="var(--blue)" stroke-width="2"')+text(w*.49+25,ky,'Leaf ŷ(X)');
 b+=line(8,ky+16,25,ky+16,'stroke="var(--rust)" stroke-dasharray="3 2"')+text(32,ky+20,'Residual R = Y − ŷ(X)');
 b+=enter(text(w/2,compact?267:337,'Residuals = 0 → effect estimate = 0',`text-anchor="middle" ${w<380?'style="font-size:13px"':'class="label"'}`),.85,1,0,10);
 return svg(w,h,'The initial prediction is the overall mean outcome, 75, marked by y-bar on the vertical axis and a dashed horizontal reference. A separate fitted leaf then matches every observed student’s score. Observed outcomes are squares, fitted leaves are horizontal segments, and dashed vertical residuals R = Y minus the fitted prediction shrink to zero. The proposed effect estimate equals zero.',b);
}
function holdout(w){
 const cx=w/2;let b='';
 b+=text(cx,compact?17:24,'Random split before fitting','text-anchor="middle" class="label"');
 const bw=(w-25)/2,by=compact?36:52,bh=compact?58:75;
 b+=rect(0,by,bw,bh,'fill="white" stroke="var(--blue)"')+rect(bw+25,by,bw,bh,'fill="white" stroke="var(--rust)" stroke-dasharray="6 3"');
 b+=text(bw/2,by+22,'Training students','text-anchor="middle"')+text(bw/2,by+44,'Fit ŷ(X) here','text-anchor="middle"');
 b+=text(bw+25+bw/2,by+22,'Evaluation students','text-anchor="middle"')+text(bw+25+bw/2,by+44,'R = Y − ŷ(X)','text-anchor="middle"');
 const ty=compact?110:145;
 b+=wipe(line(bw/2,ty,bw+25+bw/2,ty,'stroke="var(--blue)" stroke-width="1.5"'),bw/2,ty-5,bw+25,10,0,.45);
 b+=circle(bw/2,ty,4,`data-transfer="${bw/2},${bw+25+bw/2}" fill="var(--blue)"`);
 b+=text(cx,compact?138:173,'f(X): score without tutoring','text-anchor="middle" class="small"');
 b+=text(cx,compact?153:191,'Conditional on the fitted tree:','text-anchor="middle"');
 b+=text(cx,compact?174:216,'Tutored: E[f(X) − ŷ(X)] + 10','text-anchor="middle" class="label"');
 b+=text(cx,compact?207:254,'Untutored: E[f(X) − ŷ(X)]','text-anchor="middle" class="label"');
 b+=line(15,compact?227:279,w-15,compact?227:279,'stroke="var(--rule)"');
 b+=enter(text(cx,compact?258:317,'Expected difference = 10',`text-anchor="middle" ${formulaAttrs(w)}`),.7,1,0,10);
 return svg(w,compact?275:345,'Fit on training students and compute residuals R = Y minus the fitted prediction on evaluation students. Under randomized tutoring, the expected common baseline adjustment cancels, leaving ten points.',b);
}
function raw(w){
 let b='';const left=compact?8:20,gap=compact?24:35,pw=(w-left*2-gap)/2,cell=Math.min(compact?18:25,(pw-6)/4.9);
 for(let group=0;group<2;group++){
  const gx=left+group*(pw+gap),treated=group===1;
  b+=text(gx,compact?17:25,treated?'Tutored':'Untutored','class="label"')+text(gx,compact?35:45,'16 students');
  for(let i=0;i<16;i++){
   const high=treated?i>=4:i>=12,x=gx+(i%4)*cell*1.3,y=(compact?47:62)+Math.floor(i/4)*cell*1.3;
   b+=enter(rect(x,y,cell,cell,`fill="white" stroke="${high?'var(--rust)':'var(--blue)'}" stroke-width="1.5" ${high?'stroke-dasharray="3 2"':''}`)+text(x+cell/2,y+cell/2+4,high?'H':'L','text-anchor="middle"'),(group*16+i)*.021,(group*16+i)*.021+.12,0,-12);
  }
  const y=(compact?47:62)+4*cell*1.3;
  b+=text(gx,y+(compact?20:30),treated?'Mean Y = 90':'Mean Y = 60','class="label"');
 }
 b+=text(left,compact?188:248,'L: baseline 50 · H: baseline 90');
 b+=enter(text(w/2,compact?220:293,'Observed gap: 30 = 20 + 10',`text-anchor="middle" ${formulaAttrs(w)}`),.8,1,0,10);
 b+=text(w/2,compact?247:326,'Baseline composition + tutoring effect','text-anchor="middle"');
 return svg(w,compact?265:350,'Untutored students include twelve low and four high baseline students. Tutored students include four low and twelve high. The observed thirty-point gap includes a twenty-point composition difference.',b);
}
function tree(w){
 const left=w*.25,right=w*.75;let b='';
 b+=text(w/2,compact?22:30,'X < 55?',`text-anchor="middle" ${formulaAttrs(w)}`);
 const branchTop=compact?34:44,branchBottom=compact?76:106;
 b+=wipe(line(w/2-15,branchTop,left,branchBottom,'stroke="var(--rule)"')+line(w/2+15,branchTop,right,branchBottom,'stroke="var(--rule)"'),0,branchTop-4,w,branchBottom-branchTop+12,0,.3);
 b+=text(left,compact?63:83,'Yes','text-anchor="end"')+text(right,compact?63:83,'No');
 b+=text(left,compact?101:132,'Low: X = 40 or 50','text-anchor="middle"')+text(right,compact?101:132,'High: X = 60 or 70','text-anchor="middle"');
 b+=enter(text(left,compact?137:175,'ŷ = 52.5',`text-anchor="middle" ${formulaAttrs(w)}`),.25,.7,w/2-left,-35)+enter(text(right,compact?137:175,'ŷ = 97.5',`text-anchor="middle" ${formulaAttrs(w)}`),.25,.7,w/2-right,-35);
 b+=text(left,compact?167:211,'12 × 50 + 4 × 60','text-anchor="middle"')+text(right,compact?167:211,'4 × 90 + 12 × 100','text-anchor="middle"');
 b+=text(left,compact?189:236,'divided by 16','text-anchor="middle"')+text(right,compact?189:236,'divided by 16','text-anchor="middle"');
 b+=line(15,compact?213:264,w-15,compact?213:264,'stroke="var(--rule)"');
 b+=enter(text(w/2,compact?248:304,'Leaf gap = 45 = 40 + 5',`text-anchor="middle" ${formulaAttrs(w)}`),.8,1,0,10);
 return svg(w,compact?270:335,'Outcome tree splits at score 55. Low leaf prediction 52.5, high leaf 97.5. The gap is 45 because the high group has more tutoring.',b);
}
function residuals(w,both=false){
 const h=compact?270:340,left=compact?51:57,top=compact?32:35,bottom=compact?44:52;
 const p=plot(w,h,{top,left,right:compact&&!both?54:25,bottom,xmin:both?-1:-.4,xmax:both?1.25:1.4,ymin:-10,ymax:10,xticks:both?[-1,0,1]:[0,1],yticks:[-10,-5,0,5,10],xlabel:both?'Adjusted treatment T = D − ê(X)':'Tutoring indicator D',xlabelAttrs:both?'data-treatment-axis=""':'',ylabel:both?'Adjusted score R (points)':'',xformat:v=>both?String(v):v?'Tutored':'Untutored'});
 let b=p.base;
 if(!both)b+=text(left,15,'Adjusted score R (points)','data-outcome-axis="" class="axis-title"');
 b+=line(p.x(both?-1:-.4),p.y(0),p.x(both?1.25:1.4),p.y(0),`data-zero-line="" stroke="var(--rule)" stroke-dasharray="3 3"`);
 if(both)b+=p.clip(wipe(path([[p.x(-1),p.y(-10)],[p.x(1),p.y(10)]],'fill="none" stroke="var(--ink)" stroke-width="1.5"'),p.x(-1),top,p.x(1)-p.x(-1),p.bottom-top,.8,1));
 rows.forEach((r,i)=>{
  const x=p.x(both?r.T:r.D),y=p.y(r.R),rad=Math.sqrt(r.count)*3;
  const marks=circle(x,y,rad,`fill="white" stroke="${r.group==='Low'?'var(--blue)':'var(--rust)'}" stroke-width="2" ${r.group==='High'?'stroke-dasharray="3 2"':''}`)+text(x+(both?(r.T>0?-1:1)*17:17),y+4,`${r.group} · ${r.count}`,`text-anchor="${both&&r.T>0?'end':'start'}" class="small"`);
  b+=`<g ${both?`data-adjust-x="${p.x(r.D)-x}"`:`data-adjust-y="${i},${y}"`}>${marks}</g>`;
 });
 if(!both){for(const [d,mean] of [[0,-3.75],[1,3.75]]){b+=`<g data-outcome-mean="${d},${p.y(mean)}">${line(p.x(d)-24,p.y(mean),p.x(d)+24,p.y(mean),'stroke="var(--ink)" stroke-width="2"')}${text(p.x(d)-29,p.y(mean)+4,String(mean),'data-mean-value="" text-anchor="end"')}</g>`;}}
 else b+=enter(text(p.x(.15),p.y(8.7),'R = 10T','class="label"'),.85,1,0,10);
 return svg(w,h,both?'Subtract the score-group tutoring rate from D. The four adjusted points then lie on R equals ten times T.':'Subtract the outcome tree prediction from each score; then zoom to the residuals. Comparing original tutoring groups gives means minus 3.75 and plus 3.75.',b,`data-outcome-scale="${p.bottom},${p.bottom-p.top}"`);
}
function weighted(w){
 let b='';const cols=[0,w*.42,w*.70,w-1],labels=['Group','Count','Σ TR','Σ T²'];
 labels.forEach((v,i)=>b+=text(cols[i],compact?25:28,v,`${i?'text-anchor="end"':''} class="small"`));
 rows.forEach((r,i)=>{
  const y=(compact?64:76)+i*(compact?31:43),vals=[`${r.group}, D=${r.D}`,r.count,fmt(r.count*r.T*r.R,2),fmt(r.count*r.T*r.T,2)];
  b+=`<g data-weight-row="${i}">${line(0,y+10,w,y+10,'data-weight-rule="" stroke="var(--blue)" stroke-width="2"')}${vals.map((v,j)=>text(cols[j],y,v,`${j?'text-anchor="end"':''} ${j===1?'data-weight-count=""':j===2?'data-weight-tr=""':j===3?'data-weight-t2=""':''}`)).join('')}</g>`;
 });
 b+=line(0,compact?180:232,w,compact?180:232,'stroke="var(--rule)"');
 [text(cols[0],compact?204:260,'Total'),text(cols[1],compact?204:260,'32','data-total-count="" text-anchor="end"'),text(cols[2],compact?204:260,fmt(answer.numerator,0),'data-total-tr="" text-anchor="end"'),text(cols[3],compact?204:260,fmt(answer.denominator,0),'data-total-t2="" text-anchor="end"')].forEach(s=>b+=s);
 b+=text(w/2,compact?249:320,'τ̂ = 60 / 6 = 10',`data-total-ratio="" text-anchor="middle" ${formulaAttrs(w)}`);
 return svg(w,compact?270:350,'Count-weighted products sum to sixty for T times R and six for T squared. The regression slope is ten.',b);
}
function renderScene(scene,w){
 switch(scene){case 0:return chance(w);case 1:return selection(w);case 2:return nested(w);case 3:return curves(w);case 4:return slope(w);case 5:return moments(w);case 6:return curves(w,true);case 7:return mse(w);case 8:return mse(w,true);case 9:return memorize(w);case 10:return holdout(w);case 11:return raw(w);case 12:return tree(w);case 13:return residuals(w);case 14:return residuals(w,true);default:return weighted(w);}
}
const sceneMotion=createMotionView({getDraws:()=>draws,getRisk});
const stage=document.querySelector('.stage-shell');
$('story').prepend(stage);
function anchor(){
 const header=document.querySelector('.site-header').getBoundingClientRect().height;
 return mobile.matches ? header+stage.getBoundingClientRect().height+28 : header+Math.min(180,(innerHeight-header)*.28);
}
function fillStage(){
 const info=sceneInfo[active];
 $('stage-kicker').textContent=info[0];$('stage-count').textContent=`${String(active+1).padStart(2,'0')} / ${steps.length}`;
 $('stage-title').textContent=info[1];$('visual-caption').textContent=info[2];
 sceneMotion.remove(visual);
 compact=visual.getBoundingClientRect().height<350;
 visual.innerHTML=renderScene(active,Math.max(240,Math.floor(visual.getBoundingClientRect().width)))+'<p class="motion-readout" hidden></p>';
 sceneMotion.mount(visual,active);mountedScene=active;
 $('previous').disabled=active===0;$('next').disabled=active===steps.length-1;$('scene-select').value=String(active);
 document.querySelectorAll('.site-header nav a').forEach((a,i)=>{if(i===(active<3?0:active<9?1:2))a.setAttribute('aria-current','location');else a.removeAttribute('aria-current');});
}
function scrollState(){
 scheduled=false;
 const rects=steps.map(step=>step.getBoundingClientRect()),a=anchor();
 active=activeScene(rects.map(r=>r.top),a);
 for(const intro of document.querySelectorAll('.chapter-intro')){
  const box=intro.getBoundingClientRect();
  if(box.top<=a&&box.bottom>a)active=steps.indexOf(intro.parentElement.querySelector('.step'));
 }
 if(active!==mountedScene)fillStage();
 const r=rects[active],expanded=steps[active].querySelector('.calculation[open]');
 const progress=expanded?1:sceneProgress(r.top,r.height,a,steppedMotion());
 // Let long copy travel through the space below a pinned phone diagram.
 // All text remains in the native page; no nested scrolling is required.
 steps.forEach((step,i)=>{
  const copy=step.querySelector('.scene-copy');
  const available=innerHeight-(mobile.matches?a:parseFloat(getComputedStyle(copy).top))-12;
  const overflow=Math.max(0,copy.offsetHeight-available);
  copy.style.transform=i===active&&!expanded?`translateY(${-overflow*progress}px)`:'none';
 });
 sceneMotion.update(visual,progress);
 const readout=visual.querySelector('.motion-readout');
 if(readout)$('motion-readout').textContent=readout.textContent;
 $('scene-progress').value=progress;
 lastPosition={index:active,fraction:(a-r.top)/r.height};
 const start=$('story').offsetTop,range=$('story').offsetHeight-innerHeight;
 document.querySelector('.reading-progress').style.scale=`${Math.max(0,Math.min(1,(scrollY-start)/range))} 1`;
}
function schedule(){if(!scheduled){scheduled=true;requestAnimationFrame(scrollState);}}
function navigate(n,history=true){
 n=Math.max(0,Math.min(steps.length-1,n));
 if(history)window.history.pushState(null,'',`#step-${n}`);
 window.scrollTo({top:scrollY+steps[n].getBoundingClientRect().top-anchor()+1,behavior:'instant'});
 scrollState();
}
$('previous').addEventListener('click',()=>navigate(active-1));$('next').addEventListener('click',()=>navigate(active+1));
$('scene-select').addEventListener('change',e=>navigate(Number(e.target.value)));
window.addEventListener('keydown',e=>{
 if(e.altKey||e.ctrlKey||e.metaKey||e.shiftKey||/^(INPUT|SELECT|TEXTAREA|BUTTON|SUMMARY)$/.test(e.target.tagName)||e.target.isContentEditable||scrollY<$('story').offsetTop-100)return;
 if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();navigate(active+(e.key==='ArrowRight'?1:-1));}
});
window.addEventListener('scroll',schedule,{passive:true});
window.addEventListener('popstate',()=>{const m=location.hash.match(/^#step-(\d+)$/);if(m)navigate(Number(m[1]),false);else schedule();});
// Chapter links start the associated calculation after its setup has been read.
let resizeTimer;window.addEventListener('resize',()=>{
 const saved={...lastPosition};clearTimeout(resizeTimer);
 resizeTimer=setTimeout(()=>{
  const wasReading=scrollY>=$('story').offsetTop-100;
  active=saved.index;fillStage();
  if(wasReading){const r=steps[active].getBoundingClientRect();window.scrollTo({top:scrollY+r.top+saved.fraction*r.height-anchor(),behavior:'instant'});}
  scrollState();
 },120);
});
function updateMotionPreference(){
 $('motion-mode').setAttribute('aria-pressed',String(steppedMotion()));
 $('motion-mode').textContent=steppedMotion()?'Use continuous motion':'Use stepped motion';
 scrollState();
}
$('motion-mode').addEventListener('click',()=>{motionPreference=!steppedMotion();updateMotionPreference();});
reduced.addEventListener('change',updateMotionPreference);
document.querySelectorAll('details').forEach(detail=>detail.addEventListener('toggle',()=>{
 if(detail.matches('.calculation[open]')){
  detail.closest('.scene-copy').style.transform='none';
  const rect=detail.getBoundingClientRect();window.scrollTo({top:scrollY+rect.top-anchor(),behavior:'instant'});
 }
 schedule();
}));
function selectionDetail(){
 const n=Number($('validation')?.value||10),r=randomModelSelection(32768,n);
 if($('validation-value'))$('validation-value').textContent=n;
 if($('selection-detail'))$('selection-detail').textContent=`Use ${n} observations for selection and ${20-n} for the final test. Expected perfect validation candidates: ${fmt(r.expectedPerfect,r.expectedPerfect<1?2:0)}. Probability of at least one: ${r.perfectProbability>.999999?'greater than 99.9999%':fmt(100*r.perfectProbability,2)+'%'}. Expected test accuracy stays 50%; its standard deviation is ${fmt(100*r.testSD,1)} percentage points.`;
}
function noiseDetail(){const m=Number($('noise')?.value||100000),r=getRisk(m);if($('noise-detail'))$('noise-detail').textContent=`At m = ${fmt(m,0)}, total MSEs are ${r.map((v,i)=>`Carnie ${i+1}: ${fmt(v.mse)}`).join('; ')} days². Lowest MSE: Carnie ${r.indexOf(r.reduce((a,b)=>a.mse<b.mse?a:b))+1}.`;}
function refresh(){selectionDetail();noiseDetail();fillStage();scrollState();}
$('validation')?.addEventListener('input',refresh);$('noise')?.addEventListener('change',()=>{noiseFrom=noiseLast;noiseLast=Number($('noise').value);refresh();});$('curve-view')?.addEventListener('change',refresh);$('resample')?.addEventListener('click',()=>{draws=sampleCarnies();refresh();});
const bar=document.createElement('div');bar.className='reading-progress';bar.setAttribute('aria-hidden','true');document.body.append(bar);
refresh();
const initial=location.hash.match(/^#step-(\d+)$/);if(initial)requestAnimationFrame(()=>navigate(Number(initial[1]),false));
else requestAnimationFrame(scrollState);
updateMotionPreference();
