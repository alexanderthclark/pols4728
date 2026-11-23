import {bearComparisonResults as results} from './bear-comparison-results.mjs?v=20261123-context-ten-noise';

// Every horizontal position is a measured AUC. Scrolling assembles the
// individual simulation runs, then reveals their fixed mean.
const noiseCount=results.metadata.n_noise;
const C={ink:'#242424',muted:'#606060',rule:'#767676',grid:'#e0e4e6'};
const models=[
  {id:'tree_depth2',name:'Depth-two tree',detail:'One greedy fitted tree',color:'#234E70'},
  {id:'random_forest',name:'Random forest',detail:'500 fully grown trees',color:'#A54F32'},
  {id:'tabicl_v2',name:'TabICLv2',detail:'Pretrained tabular model',color:'#007F73'}
];
const clamp=p=>Math.max(0,Math.min(1,Number.isFinite(p)?p:0));
const smooth=p=>{const t=clamp(p);return t*t*(3-2*t);};
const f=n=>Number(n.toFixed(4));
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const text=(x,y,s,size=18,color=C.ink,anchor='start',attrs='')=>`<text x="${f(x)}" y="${f(y)}" font-size="${size}" fill="${color}" text-anchor="${anchor}" ${attrs}>${esc(s)}</text>`;
const line=(x1,y1,x2,y2,color,width=1,attrs='')=>`<line x1="${f(x1)}" y1="${f(y1)}" x2="${f(x2)}" y2="${f(y2)}" stroke="${color}" stroke-width="${width}" ${attrs}/>`;
const circle=(x,y,r,color,filled,attrs='')=>`<circle cx="${f(x)}" cy="${f(y)}" r="${r}" fill="${filled?color:'#fff'}" stroke="${color}" stroke-width="2" ${attrs}/>`;
const caseFor=(model,id)=>results.cases.find(row=>row.model===model&&row.id===id);

export function renderBearComparison(stage,p,mobile=false){
  const current=Math.max(0,Math.min(2,Math.trunc(stage))),progress=clamp(p);
  const L=mobile?{left:24,right:376,top:70,bottom:276,axis:291,ticks:312,label:338,first:103,gap:77}
    :{left:260,right:810,top:87,bottom:369,axis:389,ticks:415,label:447,first:133,gap:100};
  const scale=auc=>L.left+(auc-.46)/(.78-.46)*(L.right-L.left);
  // On narrow screens the row headings sit above each marker band. Leave
  // those headings clear rather than running vertical guides through text.
  const guide=(x,color,attrs='')=>mobile
    ? [0,1,2].map(i=>line(x,L.first+i*L.gap-12,x,L.first+i*L.gap+22,color,1,attrs)).join('')
    : line(x,L.top,x,L.bottom,color,1,attrs);
  const oracle=results.metadata.p_survival_match;
  const seedCount=results.metadata.seeds.length;
  let s=`<g data-bear-comparison-stage="${current}" data-comparison-progress="${progress}" data-scale-min="0.46" data-scale-max="0.78">`
    +'<title>Measured bear-survival prediction with continuous distractors</title>'
    +`<desc>All models use the same ${results.metadata.n_train} training and ${results.metadata.n_test} test rows per seed. Each input contains bear type, response, and ${noiseCount} independent uniform noise columns. Small circles show individual simulation runs; large circles and numerical labels show their mean test ROC AUC. Dashed line: population oracle AUC ${oracle}. Scrolling reveals the measured runs and their fixed mean without changing any horizontal AUC position.</desc>`;
  const title=`${noiseCount} continuous distractor columns`;
  s+=text(mobile?200:450,mobile?21:30,title,mobile?18:24,C.ink,'middle');
  const legendY=mobile?48:62,legendLeft=mobile?53:290;
  s+=circle(legendLeft,legendY-5,2.5,C.muted,true)+text(legendLeft+12,legendY,'Individual runs',mobile?17:18,C.muted)
    +circle(mobile?247:520,legendY-5,6,C.muted,true)+text(mobile?261:536,legendY,'Mean',mobile?17:18,C.muted);
  for(const tick of [.5,.6,.7]){
    const x=scale(tick);
    s+=guide(x,C.grid)+line(x,L.axis,x,L.axis+5,C.rule)
      +text(x,L.ticks,tick.toFixed(1),mobile?17:18,C.muted,'middle');
  }
  s+=line(L.left,L.axis,L.right,L.axis,C.rule,1.2)
    +text((L.left+L.right)/2,L.label,'Test ROC AUC · higher is better',mobile?17:19,C.ink,'middle')
    +guide(scale(oracle),C.muted,'stroke-dasharray="5 5" data-mark="oracle"');
  if(!mobile)s+=text(894,447,'Dashed: oracle .731',17,C.muted,'end');
  for(let i=0;i<=current;i++){
    const model=models[i],noisy=caseFor(model.id,'continuous_noise');
    if(!noisy)throw new Error(`Missing measured bear comparison for ${model.id}`);
    const y=L.first+i*L.gap,active=i===current;
    const meanReveal=active?smooth((progress-.68)/.28):1;
    const xMean=scale(noisy.mean_auc);
    const records=results.records.filter(row=>row.model===model.id&&row.id==='continuous_noise')
      .sort((a,b)=>a.seed-b.seed);
    s+=`<g data-model="${model.id}" data-noise-auc="${noisy.mean_auc}">`
      +text(24,mobile?y-22:y-5,model.name,mobile?19:24,model.color)
      +(mobile?'':text(24,y+21,model.detail,18,C.muted));
    s+='<g data-mark="seed-results">';
    records.forEach((row,index)=>{
      const reveal=active?smooth((progress-index*.045)/.12):1;
      const offset=(results.metadata.seeds.indexOf(row.seed)-(seedCount-1)/2)*(mobile?.65:.9);
      // The vertical arrival is visual assembly only; the AUC coordinate never moves.
      const markY=y+14+offset-18*(1-reveal);
      s+=`<g opacity="${f(reveal)}">`
        +circle(scale(row.auc),markY,mobile?1.8:2.4,model.color,true,`data-seed="${row.seed}" data-case="${row.id}" data-auc="${row.auc}"`)
        +'</g>';
    });
    s+='</g>'
      +`<g opacity="${f(meanReveal)}" data-mark="mean-result">`
      +circle(xMean,y,(mobile?6:7)*meanReveal,model.color,true,`data-mark="noise-mean" data-endpoint="${meanReveal===1}"`)
      +text(mobile?376:894,mobile?y-22:y+7,meanReveal===1?noisy.mean_auc.toFixed(3):'',mobile?18:22,model.color,'end','data-mark="mean-label" font-variant-numeric="tabular-nums"')
      +'</g></g>';
  }
  return s+'</g>';
}
