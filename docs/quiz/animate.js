import {coarsePrediction, carnieMoments, tutoringRows} from './model.mjs';
import {clamp01, easeInOut, phase, lerp} from './motion.mjs';

/** Compile figure renderers once, then render any reversible scroll position. */
export function createMotionView({getDraws,getRisk}){
 const rows=tutoringRows();
 const controllers=new Map();
 const fmt=(v,d=2)=>Number(v).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
 const exact=v=>Number(v).toLocaleString('en-US',{maximumFractionDigits:4});

 function renderer(root,scene){
  const all=selector=>[...root.querySelectorAll(selector)];
  const enters=all('[data-enter]').map(el=>({el,v:el.dataset.enter.split(',').map(Number)}));
  const wipes=all('[data-wipe]').map(el=>({el,v:el.dataset.wipe.split(',').map(Number)}));
  const bars=all('[data-search-bar]').map(el=>({el,v:el.dataset.searchBar.split(',').map(Number)}));
  const probes=all('[data-probe]').map(el=>({el,s:JSON.parse(el.dataset.probe),guide:el.querySelector('[data-guide]'),mark:el.querySelector('[data-probe-mark]')}));
  const deviations=all('[data-sd]').map(el=>({s:JSON.parse(el.dataset.sd),line:el.querySelector('[data-sd-line]'),mean:el.querySelector('[data-sd-mean]')}));
  const components=all('[data-component]').map(el=>({el,v:el.dataset.component.split(',').map(Number)}));
  const noise=all('[data-noise-row]').map(el=>({s:JSON.parse(el.dataset.noiseRow),line:el.querySelector('[data-noise-line]'),dot:el.querySelector('[data-noise-dot]'),value:el.querySelector('[data-noise-value]')}));
  const fits=all('[data-fit],[data-fit-residual]').map(el=>({el,v:(el.dataset.fit||el.dataset.fitResidual).split(',').map(Number),residual:el.hasAttribute('data-fit-residual')}));
  const outcomePoints=all('[data-adjust-y]').map(el=>({el,v:el.dataset.adjustY.split(',').map(Number)}));
  const outcomeMeans=all('[data-outcome-mean]').map(el=>({el,v:el.dataset.outcomeMean.split(',').map(Number),value:el.querySelector('[data-mean-value]')}));
  const treatmentPoints=all('[data-adjust-x]').map(el=>({el,dx:Number(el.dataset.adjustX)}));
  const outcomeTicks=scene===13?all('[data-y-tick]'):[];
  const scale=root.querySelector('[data-outcome-scale]');
  const [outcomeBottom,outcomeSpan]=(scale?.dataset.outcomeScale||'288,253').split(',').map(Number);
  const outcomeAxis=root.querySelector('[data-outcome-axis]');
  const zeroLine=root.querySelector('[data-zero-line]');
  const treatmentAxis=root.querySelector('[data-treatment-axis]');
  const weightedRows=scene===15?rows.map((row,i)=>({row,el:root.querySelector(`[data-weight-row="${i}"]`)})).map(({row,el})=>({row,rule:el.querySelector('[data-weight-rule]'),count:el.querySelector('[data-weight-count]'),tr:el.querySelector('[data-weight-tr]'),t2:el.querySelector('[data-weight-t2]')})):[];
  const totalCount=root.querySelector('[data-total-count]');
  const totalTR=root.querySelector('[data-total-tr]');
  const totalT2=root.querySelector('[data-total-t2]');
  const totalRatio=root.querySelector('[data-total-ratio]');
  const noiseHeading=root.querySelector('[data-noise-heading]');
  const perfect=root.querySelector('[data-search-perfect]');
  const transfer=root.querySelector('[data-transfer]');
  const readout=root.querySelector('.motion-readout');
  const progress=root.querySelector('progress');
  return value=>{
   const draws=getDraws(),t=clamp01(value);root.dataset.motionProgress=String(t);if(progress)progress.value=t;
   enters.forEach(({el,v:[start,end,dx,dy]})=>{const q=easeInOut(phase(t,start,end));el.style.opacity=String(q);el.setAttribute('transform',`translate(${dx*(1-q)} ${dy*(1-q)})`);});
   wipes.forEach(({el,v:[width,start,end]})=>el.setAttribute('width',width*easeInOut(phase(t,start,end))));
   const searched=Math.round(32768*easeInOut(phase(t,0,.85)));
   bars.forEach(({el,v:[baseline,height]})=>{const h=height*searched/32768;el.setAttribute('height',h);el.setAttribute('y',baseline-h);});
   if(perfect){const n=Number(perfect.dataset.searchPerfect)*searched/32768;perfect.textContent=`Perfect: ${fmt(n,n<1?2:0)} expected`;}
   let probeX=0;
   probes.forEach(({el,s,guide,mark})=>{
    const q=easeInOut(phase(t,s.start,1)),x=lerp(s.lo,s.hi,q),cx=lerp(s.x0,s.x1,q);probeX=x;
    const predicted=s.kind==='prediction'?(s.model===2?coarsePrediction(draws.firstSlopes[0]*x):(s.model===1?draws.secondSlopes[0]:draws.firstSlopes[0])*x):carnieMoments(x)[s.model][s.kind];
    const cy=s.y0+s.unit*predicted;el.style.opacity=t<s.start?'0':'1';guide.setAttribute('x1',cx);guide.setAttribute('x2',cx);
    if(mark.dataset.probeMark==='square'){mark.setAttribute('x',cx-4.5);mark.setAttribute('y',cy-4.5);}else{mark.setAttribute('cx',cx);mark.setAttribute('cy',cy);}
   });
   deviations.forEach(({s,line,mean})=>{const q=easeInOut(phase(t,.45,1)),x=730*q,cx=lerp(s.x0,s.x1,q),sd=x/Math.sqrt(s.j?100:73000);line.setAttribute('x1',cx);line.setAttribute('x2',cx);line.setAttribute('y1',s.y0+s.unit*(x-2*sd));line.setAttribute('y2',s.y0+s.unit*(x+2*sd));mean.setAttribute('cx',cx);mean.setAttribute('cy',s.y0+s.unit*x);});
   components.forEach(({el,v:[width,k]})=>el.setAttribute('width',width*easeInOut(phase(t,[0,.3,.63][k],[.3,.63,.86][k]))));
   let noiseM=1;
   if(noise.length){const s=noise[0].s;noiseM=Math.round(Math.exp(lerp(Math.log(s.from),Math.log(s.to),easeInOut(t))));const risks=getRisk(noiseM);noise.forEach(({s,line,dot,value})=>{const r=risks[s.j],x=s.left+Math.log10(r.mse)/s.xmax*s.usable;line.setAttribute('x2',x);dot.setAttribute('cx',x);value.setAttribute('x',x+9);value.textContent=fmt(r.mse,r.mse>100000?0:2);});noiseHeading.textContent=`Noise multiplier m = ${fmt(noiseM,0)}`;}
   fits.forEach(({el,v:[from,to,i],residual})=>{const y=lerp(from,to,easeInOut(phase(t,(i-1)*.095,(i-1)*.095+.18)));el.setAttribute('y2',y);if(!residual)el.setAttribute('y1',y);});
   if(transfer){const [from,to]=transfer.dataset.transfer.split(',').map(Number);transfer.setAttribute('cx',lerp(from,to,easeInOut(phase(t,0,.5))));}
   const subtract=easeInOut(phase(t,0,.63)),ymax=lerp(110,10,easeInOut(phase(t,.63,.95))),outcomeY=y=>outcomeBottom-(y+10)/(ymax+10)*outcomeSpan;
   outcomePoints.forEach(({el,v:[i,finalY]})=>{const r=rows[i];el.setAttribute('transform',`translate(0 ${outcomeY(r.Y-subtract*r.yhat)-finalY})`);el.querySelector('text').style.opacity=String(phase(t,.8,.98));});
   outcomeMeans.forEach(({el,v:[d,finalY],value})=>{const y=(d?90:60)-subtract*(d?86.25:63.75);el.setAttribute('transform',`translate(0 ${outcomeY(y)-finalY})`);value.textContent=fmt(y,2);});
   if(scene===13){outcomeTicks.forEach((el,i)=>{el.textContent=fmt(-10+(ymax+10)*i/4,0);});zeroLine.setAttribute('y1',outcomeY(0));zeroLine.setAttribute('y2',outcomeY(0));outcomeAxis.textContent=t<.63?'Subtracting ŷ(X) from Y (points)':'Adjusted score R (points)';}
   const treatmentProgress=easeInOut(phase(t,.08,.78));
   treatmentPoints.forEach(({el,dx})=>el.setAttribute('transform',`translate(${dx*(1-treatmentProgress)} 0)`));
   if(treatmentAxis)treatmentAxis.textContent=treatmentProgress<1?'Subtracting ê(X) from D':'Adjusted treatment T = D − ê(X)';
   let included=32;
   if(scene===15){
    included=Math.round(32*easeInOut(phase(t,.02,.95)));let remaining=included,sumTR=0,sumT2=0;
    weightedRows.forEach(({row:r,rule,count,tr,t2})=>{
     const n=Math.min(r.count,Math.max(0,remaining));remaining-=r.count;
     const productTR=n*r.T*r.R,productT2=n*r.T*r.T;sumTR+=productTR;sumT2+=productT2;
     rule.style.opacity=n>0&&n<r.count?'1':n===r.count?'.2':'0';
     if(count)count.textContent=n;if(tr)tr.textContent=exact(productTR);if(t2)t2.textContent=exact(productT2);
    });
    totalCount.textContent=included;totalTR.textContent=exact(sumTR);totalT2.textContent=exact(sumT2);
    totalRatio.textContent=included?`τ̂ = ${exact(sumTR)} / ${exact(sumT2)} = 10`:'τ̂ = —';
   }
   const status=[
    t<.8?'Reveal one illustrative fair guess at a time.':'Each unused guess still has a ½ chance of matching.',
    `Search ${fmt(searched,0)} of 32,768 candidates; bars show expectations.`,
    t<.8?`Select inside outer fold ${Math.min(4,Math.floor(t/.2)+1)}; evaluate outside it.`:'Average the four held-out results.',
    t<.65?`Draw independent training sample ${Math.min(10,Math.floor(t/.043)+1)} of 10.`:`Follow sample 1 at x = ${fmt(probeX,0)} days.`,
    `Compare ±2 SD at x = ${fmt(730*easeInOut(phase(t,.45,1)),0)} days.`,
    t<.65?'Draw the mean and variance across fixed ages.':`Inspect both panels at x = ${fmt(probeX,0)} days.`,
    t<.15?'Reuse the same ten fitted lines.':`x = ${fmt(probeX,1)} · fit ${fmt(draws.firstSlopes[0]*probeX,1)} → report ${fmt(coarsePrediction(draws.firstSlopes[0]*probeX),0)} days`,
    t<.3?'Add average squared bias.':t<.63?'Add prediction variance.':t<.86?'Add fresh outcome noise.':'The total is expected prediction MSE.',
    `Change noise variance to m = ${fmt(noiseM,0)}; watch the ranking.`,
    t<.85?'Give each student a leaf; move its prediction onto Y.':'Every prediction matches Y; every residual is zero.',
    t<.5?'Carry the fitted prediction to held-out students.':'The same expected adjustment cancels across groups.',
    t<.8?'Count the low- and high-score students in each group.':'The 30-point gap combines 20 baseline points and 10 effect points.',
    t<.7?'Split at 55 and form the two outcome means.':'The leaf gap includes unequal tutoring rates.',
    t<.63?`Subtract ${fmt(subtract*100,0)}% of each fitted outcome.`:t<.95?'Zoom into the completed residuals; axis labels follow the scale.':'Outcome residual means: −3.75 and +3.75.',
    t<.8?`Subtract ${fmt(easeInOut(phase(t,.08,.78))*100,0)}% of each score-group tutoring rate.`:'After treatment adjustment, every point lies on R = 10T.',
    `Include ${included} of 32 students in the weighted sums.`
   ][scene];
   if(readout)readout.textContent=status;
  };
 }
 function remove(root){controllers.delete(root);}
 function mount(root,scene){
  const render=renderer(root,scene);controllers.set(root,render);render(0);
 }
 function update(root,progress){controllers.get(root)?.(progress);}
 return {mount,update,remove};
}
