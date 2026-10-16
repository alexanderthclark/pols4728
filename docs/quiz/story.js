import {coarsePrediction, randomModelSelection, sampleCarnies, carnieMoments, integratedCarnies, tutoringRows, tutoringSummary} from './model.mjs';
import {createMotionView} from './animate.js';

document.body.classList.remove('no-js');
const $ = id => document.getElementById(id);
const visual = $('visual');
const steps = [...document.querySelectorAll('.step')];
const mobile = window.matchMedia('(max-width: 900px), (max-height: 760px)');
const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
const colors = ['var(--blue)','var(--rust)','var(--teal)','var(--purple)','var(--ochre)','var(--sky)','var(--rose)','var(--muted)'];
const patterns = ['', '7 3', '2 3', '8 3 2 3', '10 4', '1 3', '5 2 1 2', '3 3', '8 2 2 2 2 2', '12 3 2 3'];
const rows = tutoringRows();
const answer = tutoringSummary();
const riskCache = new Map();
let draws = sampleCarnies();
let active = 0;
let serial = 0;
let scheduled = false;
let navigationTarget = null;
let navigationTimer;
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
 ['Question 3 · a(i)','One student per leaf removes every residual','These eight illustrative students have distinct pre-tutoring scores. In-sample memorization gives ŷ(Xᵢ) = Yᵢ even when D is excluded.'],
 ['Question 3 · a(ii)','Hold out students before fitting the tree','The evaluation outcomes never set the fitted predictions. Random assignment makes the expected adjustment the same in both evaluation groups.'],
 ['Question 3 · b(i)','A 30-point gap contains a 10-point effect','Every square is one of the quiz’s 32 students. Tutored students contain more high-score students, adding a 20-point baseline difference.'],
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
function probe(p,w,h,model,zoom=false,kind='prediction'){
 const spec={model,kind,x0:p.x(zoom?358:0),x1:p.x(zoom?372:730),y0:p.y(0),unit:p.y(1)-p.y(0),lo:zoom?358:0,hi:zoom?372:730,start:zoom?.15:.65};
 return `<g data-probe="${specAttr(spec)}">${p.clip(line(0,30,0,h-35,'data-guide="" stroke="var(--rule)" stroke-dasharray="2 3"'))}${model===2?rect(0,0,9,9,'data-probe-mark="square" fill="white" stroke="var(--blue)" stroke-width="2"'):circle(0,0,4.5,'data-probe-mark="circle" fill="white" stroke="var(--blue)" stroke-width="2"')}</g>`;
}

function svg(w,h,title,body){
 return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" role="img" aria-label="${escapeText(title)}"><title>${escapeText(title)}</title>${body}</svg>`;
}
function plot(w,h,{top=30,left=58,right=20,bottom=46,xmin=0,xmax=730,ymin=0,ymax=730,xticks=[0,365,730],yticks=[0,365,730],xlabel='x (days)',ylabel='Predicted age (days)',xformat=tick,yformat=tick}={}){
 const x=v=>left+(v-xmin)/(xmax-xmin)*(w-left-right);
 const y=v=>h-bottom-(v-ymin)/(ymax-ymin)*(h-top-bottom);
 const id=`plot-${serial++}`;
 let base=`<defs><clipPath id="${id}">${rect(left,top,w-left-right,h-top-bottom)}</clipPath></defs>`;
 base+=line(left,top,left,h-bottom,'class="plot-frame"')+line(left,h-bottom,w-right,h-bottom,'class="plot-frame"');
 xticks.forEach(v=>{base+=line(x(v),h-bottom,x(v),h-bottom+5,'class="plot-tick"')+text(x(v),h-bottom+20,xformat(v),'text-anchor="middle"');});
 yticks.forEach(v=>{base+=line(left-5,y(v),left,y(v),'class="plot-tick"')+text(left-10,y(v)+4,yformat(v),'text-anchor="end"');});
 if(xlabel)base+=text((left+w-right)/2,h-3,xlabel,'text-anchor="middle" class="axis-title"');
 if(ylabel)base+=text(left,15,ylabel,'class="axis-title"');
 return {x,y,base,clip:inner=>`<g clip-path="url(#${id})">${inner}</g>`};
}
function chance(w){
 let b=''; const left=92,cw=(w-left-10)/10;
 const target=[0,1,1,0,1,0,0,1,1,0,1,0,1,1,0,0,1,0,1,0];
 const guess =[1,1,0,0,1,1,0,0,1,0,0,0,1,0,0,1,1,1,0,1];
 for(let block=0;block<2;block++){
  const y=28+block*105;
  b+=text(left,y,`Observations ${block*10+1}–${block*10+10}`,'class="small"');
  b+=text(0,y+35,'Target y')+text(0,y+68,'Guess X');
  for(let i=0;i<10;i++){
   const n=block*10+i,cx=left+(i+.5)*cw;
   b+=text(cx,y+35,target[n],'text-anchor="middle"');
   b+=enter(rect(cx-cw*.42,y+47,cw*.84,28,`fill="white" stroke="${guess[n]===target[n]?'var(--blue)':'var(--rule)'}" ${guess[n]===target[n]?'':'stroke-dasharray="2 2"'}`)+text(cx,y+66,guess[n],'text-anchor="middle"'),n*.035,n*.035+.11,0,-20);
  }
 }
 b+=enter(text(w/2,280,'P(guess matches y) = ½','text-anchor="middle" class="formula-line"')+text(w/2,315,'Expected test accuracy: 50%','text-anchor="middle" class="label"'),.8,1,0,16);
 return svg(w,340,'Twenty illustrative fair binary guesses against a fixed target. Every guess has a one-half chance of matching.',b);
}
function selection(w){
 const n=Number($('validation')?.value||10),r=randomModelSelection(32768,n),h=350;
 let binomial=1;const counts=[];
 for(let k=0;k<=n;k++){if(k)binomial*= (n-k+1)/k;counts.push(32768*binomial*2**(-n));}
 const max=Math.max(...counts)*1.2;
 const p=plot(w,265,{top:38,left:57,right:15,bottom:47,xmin:-.5,xmax:n+.5,ymax:max,xticks:[0,Math.floor(n/2),n],yticks:[0,Math.round(max/2000)*1000],xlabel:`Correct validation predictions (of ${n})`,ylabel:'Expected number of candidates'});
 let b=p.base;
 counts.forEach((v,k)=>{const bw=(p.x(1)-p.x(0))*.7;b+=rect(p.x(k)-bw/2,p.y(v),bw,p.y(0)-p.y(v),`data-search-bar="${p.y(0)},${p.y(0)-p.y(v)}" fill="${k===n?'var(--rust)':'var(--blue)'}"`);});
 b+=text(w-17,32,`Perfect: ${fmt(r.expectedPerfect,r.expectedPerfect<1?2:0)} expected`,`data-search-perfect="${r.expectedPerfect}" text-anchor="end"`);
 b+=line(25,291,w-25,291,'stroke="var(--rule)"');
 b+=text(25,318,`${20-n} untouched test observations`,'class="label"');
 b+=text(w-25,343,'Expected test accuracy: 50%','text-anchor="end" class="label"');
 return svg(w,h,`Searching 32768 random candidates on ${n} validation observations yields ${fmt(r.expectedPerfect)} expected perfect candidates, but expected test accuracy stays 50 percent.`,b);
}
function nested(w){
 const left=75,cw=(w-left-12)/4;let b='';
 for(let row=0;row<4;row++){
  const y=42+row*62;b+=text(0,y+25,`Fold ${row+1}`);
  let tiles='';for(let col=0;col<4;col++){
   tiles+=rect(left+col*cw,y,cw-7,42,`fill="white" stroke="${col===row?'var(--rust)':'var(--blue)'}" ${col===row?'stroke-width="2"':'stroke-dasharray="4 3"'}`);
   tiles+=text(left+col*cw+(cw-7)/2,y+25,col===row?'Test':'Select','text-anchor="middle"');
  }b+=enter(tiles,row*.2,row*.2+.2,-24,0);
 }
 b+=text(w/2,332,'Select inside → evaluate outside','text-anchor="middle" class="label"');
 b+=text(w/2,361,'Average the outer held-out accuracies','text-anchor="middle"');
 return svg(w,385,'Four outer folds: choose a candidate on the other fifteen observations, then evaluate on the five held-out observations.',b);
}
function curves(w,zoom=false){
 const view=$('curve-view')?.value||'all';
 const models=zoom?[0,2]:view==='all'?[0,1,2]:[{first:0,second:1,third:2}[view]??0];
 const fh=zoom?175:models.length===3?(mobile.matches?155:128):310;
 const ymax=zoom?null:Math.max(730,...draws.firstSlopes.map(s=>s*730),...draws.secondSlopes.map(s=>s*730))*1.04;
 let b='';
 for(let panel=0;panel<models.length;panel++){
  const j=models[panel],ymin=zoom?(j===0?355:-8):0,ymaxj=zoom?(j===0?380:385):ymax;
  const p=plot(w,fh,{top:32,bottom:panel===models.length-1?43:30,xmin:zoom?358:0,xmax:zoom?372:730,ymin,ymax:ymaxj,xticks:zoom?[358,365,372]:[0,365,730],yticks:zoom?(j===0?[360,370,380]:[0,365]):[0,365,730],xlabel:panel===models.length-1?'x (days)':'',ylabel:''});
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
   inner+=wipe(path(pts,`class="data-line" stroke="${colors[i%8]}" stroke-dasharray="${patterns[i]}" ${i===0?'style="stroke-width:2.3"':''}`),p.x(zoom?358:0),32,p.x(zoom?372:730)-p.x(zoom?358:0),fh-32-(panel===models.length-1?43:30),zoom?0:i*.043,zoom?.12:i*.043+.22);
  });
  const title=['Carnie 1 · training at 0 and 730','Carnie 2 · training at 0 and 1','Carnie 3 · whole-year reports'][j];
  b+=`<g transform="translate(0 ${panel*fh})">${p.base}${p.clip(inner)}${probe(p,w,fh,j,zoom)}${text(58,16,title,'class="facet-title"')}</g>`;
 }
 return svg(w,fh*models.length,'Ten simulated fitted curves per displayed carnie. Carnie 3 floors and truncates Carnie 1’s fitted curves. '+(zoom?'Zoom around x equals 365; vertical scales differ.':'The full-range panels share axes.'),b);
}
function slope(w){
 const h=330,p=plot(w,h,{ymax:900,yticks:[0,365,730],ylabel:'Mean and ±2 SD (days)'});let b=p.base;
 for(const j of [1,0]){
  const s=2/Math.sqrt(j===0?73000:100),points=[[0,0],[730,730*(1+s)],[730,730*(1-s)],[0,0]].map(([x,y])=>[p.x(x),p.y(y)]);
  b+=p.clip(wipe(`<polygon points="${points.map(pt=>pt.join(',')).join(' ')}" fill="${j?'var(--rust)':'var(--blue)'}" fill-opacity=".1" stroke="${j?'var(--rust)':'var(--blue)'}" stroke-dasharray="${j?'7 4':''}"/>`,p.x(0),30,p.x(730)-p.x(0),254,0,.45));
  const spec={j,x0:p.x(0),x1:p.x(730),y0:p.y(0),unit:p.y(1)-p.y(0)};
  b+=p.clip(`<g data-sd="${specAttr(spec)}">${line(0,0,0,0,`data-sd-line="" stroke="${j?'var(--rust)':'var(--blue)'}" stroke-width="3"`)}${circle(0,0,4,`data-sd-mean="" fill="${j?'var(--rust)':'var(--blue)'}"`)}</g>`);
 }
 b+=path([[p.x(0),p.y(0)],[p.x(730),p.y(730)]],'stroke="var(--ink)" stroke-width="1.5" fill="none"');
 b+=text(p.x(350),p.y(625),'Carnie 2: broad band','text-anchor="middle"');
 b+=text(p.x(420),p.y(290),'Carnie 1: narrow band','text-anchor="middle"');
 return svg(w,h,'Both OLS predictions are unbiased. At every positive x Carnie 2 has 730 times Carnie 1’s prediction variance.',b);
}
function moments(w){
 let b='';const fh=180;
 for(let panel=0;panel<2;panel++){
  const p=plot(w,fh,{top:30,bottom:panel?42:30,ymax:panel?35000:730,yticks:panel?[0,17500,35000]:[0,365,730],xlabel:panel?'x (days)':'',ylabel:panel?'Variance at fixed x (days²)':'Mean prediction (days)'});
  let inner='';
  for(const j of [0,2]){
   const points=Array.from({length:731},(_,x)=>[p.x(x),p.y(carnieMoments(x)[j][panel?'variance':'mean'])]);
   inner+=wipe(path(points,`fill="none" stroke="${j?'var(--rust)':'var(--blue)'}" stroke-width="2" stroke-dasharray="${j?'7 3':''}"`),p.x(0),30,p.x(730)-p.x(0),fh-(panel?42:30)-30,0,.55);
   inner+=probe(p,w,fh,j,false,panel?'variance':'mean');
  }
  b+=`<g transform="translate(0 ${panel*fh})">${p.base}${p.clip(inner)}</g>`;
 }
 b+=text(58,fh*2+20,'Solid: Carnie 1 · Dashed: Carnie 3');
 return svg(w,fh*2+35,'Carnie 1’s mean equals x and has low variance. Carnie 3 is biased downward and has variance spikes at the reporting thresholds.',b);
}
function getRisk(m){if(!riskCache.has(m))riskCache.set(m,integratedCarnies(m));return riskCache.get(m);}
function mse(w,large=false){
 const m=large?Number($('noise')?.value||100000):1,risk=getRisk(m);let b='';
 const left=80,right=90,top=45,h=320,usable=w-left-right;
 const max=Math.max(...risk.map(r=>r.mse),...(large?getRisk(noiseFrom).map(r=>r.mse):[]))*1.12;
 const xmin=large?0:0,xmax=large?Math.ceil(Math.log10(max)):max;
 const scale=v=>left+(large?Math.log10(v):v)/xmax*usable;
 const axisY=254;
 b+=line(left,axisY,w-right,axisY,'class="plot-frame"');
 const logStep=usable<260?3:2;
 const ticks=large?Array.from({length:xmax+1},(_,i)=>i).filter(i=>i===0||i%logStep===0).filter(i=>i===0||xmax-i>=logStep||i===xmax):[0,15000,30000,45000].filter(v=>v<max);
 if(large&&!ticks.includes(xmax))ticks.push(xmax);
 ticks.forEach(v=>{const x=left+v/xmax*usable;b+=line(x,axisY,x,axisY+5,'class="plot-tick"')+text(x,axisY+22,large?`10${['⁰','¹','²','³','⁴','⁵','⁶','⁷','⁸','⁹','¹⁰'][v]||`^${v}`}`:tick(v),'text-anchor="middle"');});
 b+=text(left,21,large?`Noise multiplier m = ${fmt(m,0)}`:'Total MSE (with noise)',`class="label" ${large?'data-noise-heading=""':''}`);
 const hatchId=`variance-${serial++}`,dotsId=`noise-${serial++}`;
 b+=`<defs><pattern id="${hatchId}" patternUnits="userSpaceOnUse" width="5" height="5"><rect width="5" height="5" fill="white"/><path d="M-1,1 L1,-1 M0,5 L5,0 M4,6 L6,4" stroke="var(--blue)" stroke-width="1.5"/></pattern><pattern id="${dotsId}" patternUnits="userSpaceOnUse" width="4" height="4"><rect width="4" height="4" fill="white"/><circle cx="2" cy="2" r="1" fill="var(--muted)"/></pattern></defs>`;
 risk.forEach((r,j)=>{
  const y=76+j*67;b+=text(0,y+5,`Carnie ${j+1}`);
  if(large){const spec={j,left,usable,xmax,from:noiseFrom,to:m};b+=`<g data-noise-row="${specAttr(spec)}">${line(left,y,scale(r.mse),y,'data-noise-line="" stroke="var(--rule)" stroke-dasharray="3 3"')}${circle(scale(r.mse),y,5,`data-noise-dot="" fill="${colors[j]}"`)}${text(scale(r.mse)+9,y+5,fmt(r.mse,r.mse>100000?0:2),'data-noise-value=""')}</g>`;}
  else{
   let accum=0;for(const [k,[key,c]] of [['biasSquared','var(--rust)'],['variance',`url(#${hatchId})`],['noise',`url(#${dotsId})`]].entries()){b+=rect(scale(accum),y-10,scale(accum+r[key])-scale(accum),20,`data-component="${scale(accum+r[key])-scale(accum)},${k}" fill="${c}"`);accum+=r[key];}
   b+=enter(line(scale(r.mse),y-14,scale(r.mse),y+14,'stroke="var(--ink)"'),.85,1,0,0);
  }
  if(!large)b+=enter(text(scale(r.mse)+9,y+5,fmt(r.mse,r.mse>100000?0:2)),.85,1,8,0);
 });
 b+=text((left+w-right)/2,h-13,large?'Total MSE (days², logarithmic scale)':'Total MSE (days², linear scale)','text-anchor="middle" class="axis-title"');
 if(!large){b+=text(left,h+16,'Solid: squared bias · Stripes: variance','class="small"');b+=text(left,h+33,'Dots: outcome noise','class="small"');}
 return svg(w,h+(large?0:45),`Total MSE at noise multiplier ${m}: ${risk.map((r,i)=>`Carnie ${i+1} ${fmt(r.mse)} days squared`).join('; ')}.`,b);
}
function memorize(w){
 const h=350,p=plot(w,h,{top:35,left:48,bottom:80,xmin:0,xmax:8,ymin:0,ymax:115,xticks:[1,4,7],yticks:[0,50,100],xlabel:'Distinct student score (schematic)',ylabel:'Exam score / fitted leaf prediction'});let b=p.base;
 for(let i=1;i<=8;i++){
  const d=i%2,y=50+40*(i>=5)+10*d,x=p.x(i-.4);
  b+=rect(x-8,p.y(y)-8,16,16,'fill="white" stroke="var(--ink)" stroke-width="1.5"');
  b+=circle(x,p.y(y),3,`fill="${d?'var(--rust)':'var(--blue)'}"`);
  b+=line(x,p.y(y),x,p.y(75),`data-fit-residual="${p.y(75)},${p.y(y)},${i}" stroke="var(--rust)" stroke-width="1.5" stroke-dasharray="3 2"`);
  b+=line(x-12,p.y(y),x+12,p.y(y),`data-fit="${p.y(75)},${p.y(y)},${i}" stroke="var(--blue)" stroke-width="2"`);
 }
 b+=enter(text(w/2,h-15,'Residuals = 0 → effect estimate = 0','text-anchor="middle" class="label"'),.85,1,0,10);
 return svg(w,h,'A separate fitted leaf matches every observed student’s score. Every in-sample residual equals zero, so the proposed effect estimate equals zero.',b);
}
function holdout(w){
 const cx=w/2;let b='';
 b+=text(cx,24,'Random split before fitting','text-anchor="middle" class="label"');
 const bw=(w-35)/2;
 b+=rect(0,52,bw,75,'fill="white" stroke="var(--blue)"')+rect(bw+35,52,bw,75,'fill="white" stroke="var(--rust)" stroke-dasharray="6 3"');
 b+=text(bw/2,81,'Training students','text-anchor="middle"')+text(bw/2,107,'Fit ŷ(X) here','text-anchor="middle"');
 b+=text(bw+35+bw/2,81,'Evaluation students','text-anchor="middle"')+text(bw+35+bw/2,107,'Compute R here','text-anchor="middle"');
 b+=wipe(line(bw/2,145,bw+35+bw/2,145,'stroke="var(--blue)" stroke-width="1.5"'),bw/2,140,bw+35,10,0,.45);
 b+=circle(bw/2,145,4,`data-transfer="${bw/2},${bw+35+bw/2}" fill="var(--blue)"`);
 b+=text(cx,173,'Conditional on the fitted tree:','text-anchor="middle"');
 b+=text(cx,216,'Tutored: E[f(X) − ŷ(X)] + 10','text-anchor="middle" class="label"');
 b+=text(cx,254,'Untutored: E[f(X) − ŷ(X)]','text-anchor="middle" class="label"');
 b+=line(35,279,w-35,279,'stroke="var(--rule)"');
 b+=enter(text(cx,317,'Expected difference = 10','text-anchor="middle" class="formula-line"'),.7,1,0,12);
 return svg(w,345,'Fit on training students and compare residuals on evaluation students. Under randomized tutoring, the expected common baseline adjustment cancels, leaving ten points.',b);
}
function raw(w){
 let b='';const left=20,gap=35,pw=(w-left*2-gap)/2,cell=Math.min(25,(pw-10)/4);
 for(let group=0;group<2;group++){
  const gx=left+group*(pw+gap),treated=group===1;
  b+=text(gx,25,treated?'Tutored':'Untutored','class="label"')+text(gx,45,'16 students');
  for(let i=0;i<16;i++){
   const high=treated?i>=4:i>=12,x=gx+(i%4)*cell*1.3,y=62+Math.floor(i/4)*cell*1.3;
   b+=enter(rect(x,y,cell,cell,`fill="white" stroke="${high?'var(--rust)':'var(--blue)'}" stroke-width="1.5" ${high?'stroke-dasharray="3 2"':''}`)+text(x+cell/2,y+cell/2+4,high?'H':'L','text-anchor="middle"'),(group*16+i)*.021,(group*16+i)*.021+.12,0,-14);
  }
  const y=62+4*cell*1.3;
  b+=text(gx,y+30,treated?'Mean Y = 90':'Mean Y = 60','class="label"');
 }
 b+=text(left,248,'L: baseline 50 · H: baseline 90');
 b+=enter(text(w/2,293,'Observed gap: 30 = 20 + 10','text-anchor="middle" class="formula-line"'),.8,1,0,10);
 b+=text(w/2,326,'Baseline composition + tutoring effect','text-anchor="middle"');
 return svg(w,350,'Untutored students include twelve low and four high baseline students. Tutored students include four low and twelve high. The observed thirty-point gap includes a twenty-point composition difference.',b);
}
function tree(w){
 const left=w*.25,right=w*.75;let b='';
 b+=text(w/2,30,'X < 55?','text-anchor="middle" class="formula-line"');
 b+=wipe(line(w/2-15,44,left,106,'stroke="var(--rule)"')+line(w/2+15,44,right,106,'stroke="var(--rule)"'),0,40,w,80,0,.3);
 b+=text(left,83,'Yes','text-anchor="end"')+text(right,83,'No');
 b+=text(left,132,'Low: X = 40 or 50','text-anchor="middle"')+text(right,132,'High: X = 60 or 70','text-anchor="middle"');
 b+=enter(text(left,175,'ŷ = 52.5','text-anchor="middle" class="formula-line"'),.25,.7,w/2-left,-50)+enter(text(right,175,'ŷ = 97.5','text-anchor="middle" class="formula-line"'),.25,.7,w/2-right,-50);
 b+=text(left,211,'12 × 50 + 4 × 60','text-anchor="middle"')+text(right,211,'4 × 90 + 12 × 100','text-anchor="middle"');
 b+=text(left,236,'divided by 16','text-anchor="middle"')+text(right,236,'divided by 16','text-anchor="middle"');
 b+=line(25,264,w-25,264,'stroke="var(--rule)"');
 b+=enter(text(w/2,304,'Leaf gap = 45 = 40 + 5','text-anchor="middle" class="formula-line"'),.8,1,0,10);
 return svg(w,335,'Outcome tree splits at score 55. Low leaf prediction 52.5, high leaf 97.5. The gap is 45 because the high group has more tutoring.',b);
}
function residuals(w,both=false){
 const p=plot(w,340,{top:35,left:57,right:25,bottom:52,xmin:both?-1:-.4,xmax:both?1.25:1.4,ymin:-10,ymax:10,xticks:both?[-1,0,1]:[0,1],yticks:[-10,-5,0,5,10],xlabel:both?'Treatment after subtracting ê(X)':'Tutoring indicator D',ylabel:both?'Adjusted score R (points)':'',xformat:v=>both?String(v):v?'Tutored':'Untutored'});
 let b=p.base;
 if(!both)b+=text(57,15,'Adjusted score R (points)','data-outcome-axis="" class="axis-title"');
 b+=line(p.x(both?-1:-.4),p.y(0),p.x(both?1.25:1.4),p.y(0),`data-zero-line="" stroke="var(--rule)" stroke-dasharray="3 3"`);
 if(both)b+=p.clip(wipe(path([[p.x(-1),p.y(-10)],[p.x(1),p.y(10)]],'fill="none" stroke="var(--ink)" stroke-width="1.5"'),p.x(-1),35,p.x(1)-p.x(-1),253,.8,1));
 rows.forEach((r,i)=>{
  const x=p.x(both?r.T:r.D),y=p.y(r.R),rad=Math.sqrt(r.count)*3;
  const marks=circle(x,y,rad,`fill="white" stroke="${r.group==='Low'?'var(--blue)':'var(--rust)'}" stroke-width="2" ${r.group==='High'?'stroke-dasharray="3 2"':''}`)+text(x+(both?(r.T>0?-1:1)*17:17),y+4,`${r.group} · ${r.count}`,`text-anchor="${both&&r.T>0?'end':'start'}" class="small"`);
  b+=`<g ${both?`data-adjust-x="${p.x(r.D)-x}"`:`data-adjust-y="${i},${y}"`}>${marks}</g>`;
 });
 if(!both){for(const [d,mean] of [[0,-3.75],[1,3.75]]){b+=`<g data-outcome-mean="${d},${p.y(mean)}">${line(p.x(d)-24,p.y(mean),p.x(d)+24,p.y(mean),'stroke="var(--ink)" stroke-width="2"')}${text(p.x(d)-29,p.y(mean)+4,String(mean),'data-mean-value="" text-anchor="end"')}</g>`;}}
 else b+=enter(text(p.x(.15),p.y(8.7),'R = 10T','class="label"'),.85,1,0,10);
 return svg(w,340,both?'Subtract the score-group tutoring rate from D. The four adjusted points then lie on R equals ten times T.':'Subtract the outcome tree prediction from each score; then zoom to the residuals. Comparing original tutoring groups gives means minus 3.75 and plus 3.75.',b);
}
function weighted(w){
 let b='';const cols=[0,w*.35,w*.58,w*.8],labels=['Group','Count','Count × TR','Count × T²'];
 labels.forEach((v,i)=>b+=text(cols[i],28,v,'class="small"'));
 rows.forEach((r,i)=>{
  const y=76+i*43,vals=[`${r.group}, D=${r.D}`,r.count,fmt(r.count*r.T*r.R,2),fmt(r.count*r.T*r.T,2)];
  b+=`<g data-weight-row="${i}">${line(0,y+12,w,y+12,'data-weight-rule="" stroke="var(--blue)" stroke-width="2"')}${vals.map((v,j)=>text(cols[j],y,v)).join('')}</g>`;
 });
 b+=line(0,232,w,232,'stroke="var(--rule)"');
 [text(cols[0],260,'Total'),text(cols[1],260,'32','data-total-count=""'),text(cols[2],260,fmt(answer.numerator,0),'data-total-tr=""'),text(cols[3],260,fmt(answer.denominator,0),'data-total-t2=""')].forEach(s=>b+=s);
 b+=text(w/2,320,'τ̂ = 60 / 6 = 10','data-total-ratio="" text-anchor="middle" class="formula-line"');
 return svg(w,350,'Count-weighted products sum to sixty for T times R and six for T squared. The regression slope is ten.',b);
}
function renderScene(scene,w){
 switch(scene){case 0:return chance(w);case 1:return selection(w);case 2:return nested(w);case 3:return curves(w);case 4:return slope(w);case 5:return moments(w);case 6:return curves(w,true);case 7:return mse(w);case 8:return mse(w,true);case 9:return memorize(w);case 10:return holdout(w);case 11:return raw(w);case 12:return tree(w);case 13:return residuals(w);case 14:return residuals(w,true);default:return weighted(w);}
}
const sceneMotion=createMotionView({getDraws:()=>draws,getRisk});
function motionControls(){return `<div class="motion-controls" role="group" aria-label="Figure animation"><button type="button" data-motion-toggle>Play</button><button type="button" data-motion-replay>Replay</button><progress max="1" value="0" aria-label="Animation progress"></progress><p class="motion-readout"></p></div>`;}
function fillStage(){
 const info=sceneInfo[active];
 $('stage-kicker').textContent=info[0];$('stage-count').textContent=`${active+1} / ${steps.length}`;$('stage-title').textContent=info[1];$('visual-caption').textContent=info[2];
 sceneMotion.remove(visual);
 if(!mobile.matches){visual.innerHTML=renderScene(active,Math.max(280,Math.floor(visual.getBoundingClientRect().width)))+motionControls();sceneMotion.mount(visual,active);}
 $('previous').disabled=active===0;$('next').disabled=active===steps.length-1;$('scene-select').value=String(active);
 document.querySelectorAll('.site-header nav a').forEach((a,i)=>{if(i===(active<3?0:active<9?1:2))a.setAttribute('aria-current','location');else a.removeAttribute('aria-current');});
}
function fillInline(){
 if(!mobile.matches){document.querySelectorAll('.inline-visual').forEach(figure=>sceneMotion.remove(figure));return;}
 steps.forEach((step,i)=>{
  let figure=step.querySelector('.inline-visual');
  if(!figure){figure=document.createElement('figure');figure.className='inline-visual';step.querySelector('h3').after(figure);}
  const w=Math.max(260,Math.floor(step.getBoundingClientRect().width));
  sceneMotion.remove(figure);figure.innerHTML=renderScene(i,w)+motionControls()+`<figcaption>${escapeText(sceneInfo[i][2])}</figcaption>`;sceneMotion.mount(figure,i);
 });
}
function setScene(n){n=Math.max(0,Math.min(steps.length-1,n));if(n!==active){active=n;fillStage();}}
function navigate(n,history=true){
 n=Math.max(0,Math.min(steps.length-1,n));navigationTarget=n;clearTimeout(navigationTimer);active=n;fillStage();
 if(history)window.history.pushState(null,'',`#step-${n}`);
 steps[n].scrollIntoView({behavior:reduced.matches?'instant':'smooth',block:'start'});
 navigationTimer=setTimeout(()=>{navigationTarget=null;scrollState();},1500);
}
function scrollState(){
 scheduled=false;
 const threshold=window.innerHeight*(mobile.matches?.3:.42);
 let index=0;
 for(let i=0;i<steps.length;i++){if(steps[i].getBoundingClientRect().top<=threshold)index=i;else break;}
 if(navigationTarget===null)setScene(index);
 const progress=document.querySelector('.reading-progress');
 if(progress){const start=$('story').offsetTop,range=$('story').offsetHeight-window.innerHeight;progress.style.scale=`${Math.max(0,Math.min(1,(window.scrollY-start)/range))} 1`;}
}
if(!$('scene-select').options.length)sceneInfo.forEach((s,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${i+1}. ${s[0]}`;$('scene-select').append(o);});
$('previous').addEventListener('click',()=>navigate(active-1));$('next').addEventListener('click',()=>navigate(active+1));$('scene-select').addEventListener('change',e=>navigate(Number(e.target.value)));
window.addEventListener('keydown',e=>{if(e.altKey||e.ctrlKey||e.metaKey||e.shiftKey||/^(INPUT|SELECT|TEXTAREA|BUTTON|SUMMARY)$/.test(e.target.tagName)||e.target.isContentEditable)return;if(window.scrollY<$('story').offsetTop-100)return;if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();navigate(active+(e.key==='ArrowRight'?1:-1));}});
window.addEventListener('scroll',()=>{if(!scheduled){scheduled=true;requestAnimationFrame(scrollState);}},{passive:true});
window.addEventListener('scrollend',()=>{if(navigationTarget!==null){navigationTarget=null;clearTimeout(navigationTimer);scrollState();}});
for(const event of ['wheel','touchstart'])window.addEventListener(event,()=>{navigationTarget=null;clearTimeout(navigationTimer);},{passive:true});
window.addEventListener('popstate',()=>{const m=location.hash.match(/^#step-(\d+)$/);if(m)navigate(Number(m[1]),false);});
let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{fillStage();fillInline();},120);});
function selectionDetail(){
 const n=Number($('validation')?.value||10),r=randomModelSelection(32768,n);
 if($('validation-value'))$('validation-value').textContent=n;
 if($('selection-detail'))$('selection-detail').textContent=`Use ${n} observations for selection and ${20-n} for the final test. Expected perfect validation candidates: ${fmt(r.expectedPerfect,r.expectedPerfect<1?2:0)}. Probability of at least one: ${r.perfectProbability>.999999?'greater than 99.9999%':fmt(100*r.perfectProbability,2)+'%'}. Expected test accuracy stays 50%; its standard deviation is ${fmt(100*r.testSD,1)} percentage points.`;
}
function noiseDetail(){const m=Number($('noise')?.value||100000),r=getRisk(m);if($('noise-detail'))$('noise-detail').textContent=`At m = ${fmt(m,0)}, total MSEs are ${r.map((v,i)=>`Carnie ${i+1}: ${fmt(v.mse)}`).join('; ')} days². Lowest MSE: Carnie ${r.indexOf(r.reduce((a,b)=>a.mse<b.mse?a:b))+1}.`;}
function refresh(){selectionDetail();noiseDetail();fillStage();fillInline();}
$('validation')?.addEventListener('input',refresh);$('noise')?.addEventListener('change',()=>{noiseFrom=noiseLast;noiseLast=Number($('noise').value);refresh();});$('curve-view')?.addEventListener('change',refresh);$('resample')?.addEventListener('click',()=>{draws=sampleCarnies();refresh();});
const bar=document.createElement('div');bar.className='reading-progress';bar.setAttribute('aria-hidden','true');document.body.append(bar);
refresh();
const initial=location.hash.match(/^#step-(\d+)$/);if(initial)requestAnimationFrame(()=>navigate(Number(initial[1]),false));
