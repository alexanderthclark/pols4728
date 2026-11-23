import {miniPfnResults as run} from './mini-pfn-results.mjs';

/** Exact inference under the teaching model's two-world prior. */
export const MATCH_SURVIVAL = 1 / (1 + Math.exp(-1));
export const MISMATCH_SURVIVAL = 1 - MATCH_SURVIVAL;
export const observedContext = run.examples.A.context.map(row => [...row]);
export const reversedContext = run.examples.B.context.map(row => [...row]);
const binary = x => [0, 1, false, true].includes(x);

export function survivalProbability(world, brown, lieDown) {
  if (!['A', 'B'].includes(world) || !binary(brown) || !binary(lieDown)) {
    throw new RangeError('Use world A or B, with binary brown and lieDown indicators.');
  }
  const matches = Number(brown) === Number(lieDown);
  return matches === (world === 'A') ? MATCH_SURVIVAL : MISMATCH_SURVIVAL;
}

function normalizedPrior(prior) {
  if (!Array.isArray(prior) || prior.length !== 2 || prior.some(p => !Number.isFinite(p) || p < 0) || !(prior[0] + prior[1] > 0) || !Number.isFinite(prior[0] + prior[1])) {
    throw new RangeError('prior must contain two finite nonnegative weights with a finite positive sum.');
  }
  return prior.map(p => p / (prior[0] + prior[1]));
}

function stateFromLogLikelihood(logLikelihood, query, prior, n, rowLikelihoods = []) {
  const normalized = normalizedPrior(prior);
  const logWeights = normalized.map((p, i) => p === 0 ? -Infinity : Math.log(p) + logLikelihood[i]);
  const maxLog = Math.max(...logWeights);
  const weights = logWeights.map(p => Math.exp(p - maxLog));
  const posterior = weights.map(p => p / (weights[0] + weights[1]));
  const matchPrediction = posterior[0] * MATCH_SURVIVAL + posterior[1] * MISMATCH_SURVIVAL;
  const mismatchPrediction = posterior[0] * MISMATCH_SURVIVAL + posterior[1] * MATCH_SURVIVAL;
  const worldPredictions = ['A', 'B'].map(w => survivalProbability(w, ...query));
  const prediction = posterior.reduce((sum, p, i) => sum + p * worldPredictions[i], 0);
  const logLikelihoodRatio = logLikelihood[0] - logLikelihood[1];
  return {
    n, prior: normalized, posterior, query: [...query], prediction, matchPrediction, mismatchPrediction,
    worldPredictions, rowLikelihoods, logLikelihood,
    likelihood: logLikelihood.map(Math.exp),
    logLikelihoodRatio, likelihoodRatio: Math.exp(logLikelihoodRatio),
  };
}

/** Outcomes are independent conditional on the fixed features and one shared world. */
export function contextState(context, query = [1, 1], prior = [.5, .5]) {
  if (!Array.isArray(context) || context.some(row => !Array.isArray(row) || row.length !== 3 || row.some(x => !binary(x)))) {
    throw new RangeError('Each context row must contain binary bear, strategy, and observed outcome values.');
  }
  if (!Array.isArray(query) || query.length !== 2 || query.some(x => !binary(x))) {
    throw new RangeError('The query must contain binary bear and strategy values.');
  }
  const rowLikelihoods = context.map(([b, l, y]) => ['A', 'B'].map(world => {
    const p = survivalProbability(world, b, l);
    return Number(y) === 1 ? p : 1 - p;
  }));
  const logLikelihood = [0, 1].map(i => rowLikelihoods.reduce((sum, row) => sum + Math.log(row[i]), 0));
  return stateFromLogLikelihood(logLikelihood, query, prior, context.length, rowLikelihoods);
}

/** Compatibility helper: n A-supporting observations, each with likelihood ratio e. */
export function toyState(n, prior = [.5, .5]) {
  if (!Number.isSafeInteger(n) || n < 0) throw new RangeError('n must be a nonnegative safe integer.');
  return stateFromLogLikelihood([n * Math.log(MATCH_SURVIVAL), n * Math.log(MISMATCH_SURVIVAL)], [1, 1], prior, n);
}

export function expectedLogLoss(q, r) {
  if (![q, r].every(x => Number.isFinite(x) && x >= 0 && x <= 1)) throw new RangeError('q and r must be probabilities.');
  if (q === 0) return r === 0 ? 0 : Infinity;
  if (q === 1) return r === 1 ? 0 : Infinity;
  return -r * Math.log(q) - (1 - r) * Math.log(1 - q);
}

const exactA = contextState(observedContext);
const exactB = contextState(reversedContext);
const C = {ink:'#242424',blue:'#234E70',rust:'#A54F32',teal:'#007F73',muted:'#606060',rule:'#d6dcdf'};
const clamp = n => Math.max(0, Math.min(1, n));
const mix = (a, b, t) => a + (b - a) * t;
const ease = t => t * t * (3 - 2 * t);
const f = n => Number(n.toFixed(4));
const pct = n => `${(100 * n).toFixed(1)}%`;
const text = (x,y,s,size=21,color=C.ink,anchor='start',extra='') => `<text x="${f(x)}" y="${f(y)}" font-size="${size}" fill="${color}" text-anchor="${anchor}" ${extra}>${s}</text>`;
const line = (x1,y1,x2,y2,color=C.rule,width=1.5,extra='') => `<line x1="${f(x1)}" y1="${f(y1)}" x2="${f(x2)}" y2="${f(y2)}" stroke="${color}" stroke-width="${width}" ${extra}/>`;
const rect = (x,y,w,h,fill,extra='') => `<rect x="${f(x)}" y="${f(y)}" width="${f(w)}" height="${f(h)}" fill="${fill}" ${extra}/>`;
const dot = (x,y,r,color,extra='') => `<circle cx="${f(x)}" cy="${f(y)}" r="${r}" fill="${color}" ${extra}/>`;
const names = row => `${row[0] ? 'Brown' : 'Black'} · ${row[1] ? 'Lie down' : 'Fight'}`;
const worldName = i => i === 0 ? 'A' : 'B';

function scrollState(stage, progress) {
  const raw = clamp(Number.isFinite(progress) ? progress : 1);
  const t = ease(raw);
  // Interpolated visual states join exact posteriors after completed rows.
  const reveal = stage === 0 ? 0 : stage === 1 ? t : stage === 2 ? 1 + 3 * raw : 4;
  const lo = Math.min(4, Math.floor(reveal));
  const hi = Math.min(4, lo + 1);
  const partial = ease(reveal - lo);
  const left = contextState(observedContext.slice(0, lo));
  const right = contextState(observedContext.slice(0, hi));
  const pa = mix(left.posterior[0], right.posterior[0], partial);
  return {raw, t, reveal, posterior:[pa,1-pa], prediction:pa * MATCH_SURVIVAL + (1-pa) * MISMATCH_SURVIVAL};
}

function posteriorBars(weights, mobile, yTop=0) {
  let s = '';
  const x = mobile ? 58 : 555, width = mobile ? 240 : 245;
  const ys = mobile ? [228+yTop,265+yTop] : [95+yTop,172+yTop];
  weights.forEach((w,i) => {
    const y = ys[i], col = i === 0 ? C.blue : C.rust;
    s += text(mobile?20:x, mobile?y+16:y-13, mobile?worldName(i):`World ${worldName(i)}`, mobile?17:21, col);
    s += rect(x,y,width,mobile?19:25,'#eef1f3');
    s += rect(x,y,width*w,mobile?19:25,col);
    s += text(mobile?380:860,y+(mobile?16:21),pct(w),mobile?17:21,C.ink,'end','font-variant-numeric="tabular-nums"');
  });
  return s;
}

function evidence(stage, progress, mobile) {
  const st = scrollState(stage, progress);
  const rows = observedContext;
  let s = '';
  const title = stage === 0 ? 'Survival probabilities in each world' : 'Likelihood of each observed outcome';
  s += text(mobile?20:20,mobile?20:30,title,mobile?17:23);
  const xs = mobile ? [12,185,268,357] : [20,230,335,450];
  const hy = mobile ? 49 : 77;
  ['Bear · strategy','Y','A','B'].forEach((label,i) => s += text(xs[i],hy,label,mobile?16:20,i>1?(i===2?C.blue:C.rust):C.ink,i===0?'start':'middle'));
  rows.forEach((row,i) => {
    const y = (mobile?78:123) + i * (mobile?31:52);
    const amount = clamp(st.reveal-i);
    if(stage>0) s += rect(mobile?5:12,y-(mobile?21:29),mobile?389:482,mobile?28:44,'#eaf3f0',`opacity="${f(amount*.7)}"`);
    s += text(xs[0],y,names(row),mobile?16:20);
    const seen = stage>0 && amount>.12;
    s += text(xs[1],y,seen?String(row[2]):'?',mobile?18:23,seen?C.teal:C.muted,'middle');
    for(let j=0;j<2;j++) {
      const value = stage === 0 ? survivalProbability(worldName(j),row[0],row[1]) : exactA.rowLikelihoods[i][j];
      s += text(xs[j+2],y,stage===0||seen?pct(value):'—',mobile?16:21,j===0?C.blue:C.rust,'middle');
    }
    s += line(mobile?8:12,y+(mobile?9:17),mobile?392:494,y+(mobile?9:17));
  });
  const note = stage===0 ? 'Labels not yet incorporated into the prior.' : '1 = survived; 0 = did not survive.';
  s += text(mobile?12:20,mobile?199:330,note,mobile?15:19,C.muted);
  s += text(mobile?20:555,mobile?220:43,stage===0?'Prior probability':'Posterior probability',mobile?16:22);
  // Mobile title sits above the first bar without sharing its label baseline.
  s += posteriorBars(st.posterior,mobile);
  if(mobile) {
    const label = stage===0 ? 'Prior odds A : B = 1 : 1' : stage===1 ? 'One row: likelihood ratio = e' : st.reveal>=4 ? 'Four rows: odds A : B = e⁴ ≈ 54.6 : 1' : `Incorporating row ${Math.min(4,Math.floor(st.reveal)+1)} of 4`;
    s += text(200,318,label,stage===2?15:16,C.ink,'middle');
    if(stage===2)s += text(200,344,st.reveal>=4?'Both observed deaths also favor A.':'A death can favor A: use its likelihood.',16,C.teal,'middle');
  } else {
    const incorporating = st.reveal<4 && st.reveal%1>1e-8;
    s += text(555,245,stage===0?'Before reading the labels':incorporating?`Incorporating row ${Math.floor(st.reveal)+1}`:`${Math.min(4,Math.floor(st.reveal+1e-8))} labels incorporated`,20,C.muted);
    const eq = stage===0 ? 'Prior odds A : B = 1 : 1' : stage===1 ? 'Likelihood ratio = p / (1 − p) = e' : st.reveal>=4 ? 'Likelihood ratio = p⁴ / (1 − p)⁴ = e⁴' : `Multiply by the likelihood of row ${Math.min(4,Math.floor(st.reveal)+1)}`;
    s += text(450,403,eq,24,C.ink,'middle');
    s += text(450,441,stage===2?'Survival in a match and death in a mismatch both support A.':stage===0?'A favors matching pairs; B reverses the probabilities.':'The observed outcome was 73.1% likely in A and 26.9% likely in B.',20,C.muted,'middle');
    if(stage>0 && st.raw>.02 && st.raw<.98) {
      const y = 123 + Math.min(3,Math.floor(st.reveal))*52;
      const u = stage===1?st.raw:(3*st.raw)%1;
      s += dot(mix(505,545,u),mix(y,108,u),6,C.teal);
    }
  }
  return {s,st};
}

function predictive(progress,mobile) {
  const st=scrollState(3,progress), pa=exactA.posterior[0], pb=1-pa;
  let s='';
  if(mobile) {
    s+=text(20,20,'Same posterior for both queries',18);
    [pa,pb].forEach((w,i)=>{const y=39+i*38,col=i===0?C.blue:C.rust;s+=text(20,y+16,worldName(i),17,col)+rect(58,y,240,19,'#eef1f3')+rect(58,y,240*w,19,col)+text(380,y+16,pct(w),17,C.ink,'end');});
  } else {
    s+=text(40,33,'Belief about the rule',23);
    [pa,pb].forEach((w,i)=>{const y=100+i*90,col=i===0?C.blue:C.rust;s+=text(40,y-14,`World ${worldName(i)}`,21,col)+rect(40,y,250,25,'#eef1f3')+rect(40,y,250*w,25,col)+text(375,y+22,pct(w),21,C.ink,'end');});
    s+=text(40,301,'Posterior weight is',22,C.muted)+text(40,331,'not an outcome probability.',22,C.muted);
  }
  [[1,1],[1,0]].forEach((query,i)=>{
    const x=mobile?20:435, y=mobile?145+i*100:92+i*180, w=mobile?360:405;
    const pA=survivalProbability('A',...query),pB=1-pA;
    const contrib=[pa*pA,pb*pB],prediction=contrib[0]+contrib[1];
    const grow=ease(clamp(st.raw*2-i));
    s+=text(x,y,names(query),mobile?17:23);
    s+=text(mobile?380:850,y,pct(prediction),mobile?23:31,C.teal,'end');
    s+=rect(x,y+19,w,mobile?24:32,'#eef1f3');
    s+=rect(x,y+19,w*contrib[0]*grow,mobile?24:32,C.blue);
    s+=rect(x+w*contrib[0]*grow,y+19,w*contrib[1]*grow,mobile?24:32,C.rust);
    s+=text(x,y+(mobile?68:89),`${pct(pa)} × ${pct(pA)} + ${pct(pb)} × ${pct(pB)}`,mobile?15:20,C.muted);
  });
  s+=text(mobile?200:450,mobile?342:440,'Same world weights; different conditional predictions.',mobile?15:22,C.ink,'middle');
  return {s,st};
}

function comparison(progress,mobile) {
  const st=scrollState(4,progress),x0=mobile?108:255,w=mobile?270:590;
  const yy=mobile?[96,217]:[132,267];
  let s='';
  if(mobile) {
    s+=dot(20,17,5,C.teal)+text(31,23,'Exact Bayes',16,C.teal)+rect(200,12,10,10,C.rust)+text(218,23,'Trained Mini-PFN',16,C.rust);
  } else {
    s+=text(450,31,'Same query: Brown · Lie down',24,C.ink,'middle');
    s+=dot(270,60,6,C.teal)+text(285,67,'Exact Bayes',20,C.teal)+rect(510,54,12,12,C.rust)+text(533,67,'Trained Mini-PFN',20,C.rust);
  }
  s+=rect(x0+w*MISMATCH_SURVIVAL,mobile?40:85,w*(MATCH_SURVIVAL-MISMATCH_SURVIVAL),mobile?232:232,'#f0f3f4');
  [['A',exactA],['B',exactB]].forEach(([name,exact],i)=>{
    const y=yy[i],actual=run.examples[name].probability,position=mix(exact.prediction,actual,st.t);
    s+=text(mobile?12:30,y-3,`Context ${name}`,mobile?16:22);
    s+=text(mobile?12:30,y+23,run.examples[name].context.map(row=>row[2]).join(' · '),mobile?15:20,C.muted);
    s+=line(x0,y,x0+w,y);
    s+=line(x0+w*exact.prediction,y,x0+w*position,y,C.rust,3);
    s+=dot(x0+w*exact.prediction,y,6,C.teal);
    s+=rect(x0+w*actual-5,y-5,10,10,C.rust,`opacity="${f(.2+.8*st.t)}" data-learned-probability="${actual}"`);
    s+=text(x0+w*exact.prediction,y-20,pct(exact.prediction),mobile?17:22,C.teal,'middle');
    s+=text(x0+w*actual,y+35,pct(actual),mobile?17:22,C.rust,'middle');
  });
  const base=mobile?286:345;
  s+=line(x0,base,x0+w,base,C.muted,1.4);
  [0,.5,1].forEach(q=>s+=line(x0+w*q,base,x0+w*q,base+6,C.muted)+text(x0+w*q,base+26,`${q*100}%`,mobile?15:19,C.muted,'middle'));
  s+=text(mobile?200:450,mobile?325:407,mobile?'Exact mixtures stay in 26.9%–73.1%.':'Any exact mixture of these worlds lies between 26.9% and 73.1%.',mobile?16:21,C.muted,'middle');
  s+=text(mobile?200:450,mobile?347:442,mobile?'Neural approximation can go outside.':'The learned 74.0% and 23.6% reflect approximation error.',mobile?16:22,C.ink,'middle');
  return {s,st};
}

function lossCurve(progress,mobile) {
  const st=scrollState(5,progress),r=exactA.prediction,q=mix(.12,r,st.t),loss=expectedLogLoss(q,r);
  const x0=mobile?47:90,x1=mobile?378:845,yTop=mobile?51:83,yBottom=mobile?262:350,maxLoss=3.1;
  const px=v=>x0+(x1-x0)*v,py=v=>yBottom-(yBottom-yTop)*v/maxLoss;
  let s=text(mobile?20:40,mobile?20:29,mobile?`q = ${q.toFixed(3)}; expected loss = ${loss.toFixed(3)}`:'Expected log loss for the same query and context',mobile?17:24);
  if(!mobile)s+=text(845,61,`Candidate q = ${q.toFixed(3)}; loss = ${loss.toFixed(3)}`,22,C.blue,'end');
  s+=line(x0,yTop,x0,yBottom,C.muted)+line(x0,yBottom,x1,yBottom,C.muted);
  [0,1,2,3].forEach(v=>s+=line(x0-4,py(v),x0,py(v),C.muted)+text(x0-10,py(v)+5,String(v),mobile?15:19,C.muted,'end'));
  [0,.5,1].forEach(v=>s+=line(px(v),yBottom,px(v),yBottom+5,C.muted)+text(px(v),yBottom+24,v.toFixed(1),mobile?15:19,C.muted,'middle'));
  const pts=Array.from({length:193},(_,i)=>{const v=.02+i*.005;return `${f(px(v))},${f(py(expectedLogLoss(v,r)))}`;});
  s+=`<polyline points="${pts.join(' ')}" fill="none" stroke="${C.blue}" stroke-width="${mobile?2.5:3}"/>`;
  s+=line(px(r),py(expectedLogLoss(r,r)),px(r),yBottom,C.teal,1.5,'stroke-dasharray="5 5"');
  s+=dot(px(r),py(expectedLogLoss(r,r)),mobile?5:7,C.teal);
  s+=line(px(q),py(loss),px(q),yBottom,C.blue,1.5,'stroke-dasharray="3 4"');
  s+=rect(px(q)-5,py(loss)-5,10,10,C.rust);
  s+=text(mobile?212:468,mobile?312:407,`Minimum at q = r = ${pct(r)}`,mobile?19:26,C.teal,'middle');
  s+=text(mobile?200:450,mobile?342:444,'Expected over new outcomes, not one observed label.',mobile?15:21,C.muted,'middle');
  return {s,st:{...st,candidate:q,expectedLoss:loss}};
}

function render(stage,progress,mobile) {
  stage=Math.max(0,Math.min(5,Math.round(Number(stage)||0)));
  const result=stage<=2?evidence(stage,progress,mobile):stage===3?predictive(progress,mobile):stage===4?comparison(progress,mobile):lossCurve(progress,mobile);
  const {s,st}=result;
  return `<g font-family="inherit" data-bayes-stage="${stage}" data-bayes-mobile="${mobile}" data-evidence-count="${st.reveal}" data-posterior-a="${st.posterior[0]}" data-prediction="${st.prediction}"${stage===5?` data-candidate-q="${st.candidate}" data-expected-loss="${st.expectedLoss}"`:''}><title>Exact Bayesian inference for the same four bear encounters</title>${s}</g>`;
}

/** SVG contents for viewBox="0 0 900 460"; stages 0–5. */
export function renderBayes(stage=0,progress=1) { return render(stage,progress,false); }
/** SVG contents for viewBox="0 0 400 350"; stages 0–5. */
export function renderBayesMobile(stage=0,progress=1) { return render(stage,progress,true); }
