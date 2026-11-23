import {bearComparisonResults as results} from './bear-comparison-results.mjs';

// Only endpoint markers and scores are measurements. Scroll interpolates the
// filled marker from the signal-only position to the noise-column position.
const C={ink:'#242424',muted:'#606060',rule:'#767676',grid:'#e0e4e6'};
const models=[
  {id:'tree_depth2',name:'Depth-two tree',detail:'One greedy fitted tree',color:'#234E70'},
  {id:'random_forest',name:'Random forest',detail:'500 trees',color:'#A54F32'},
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
  const t=smooth((progress-.12)/.72),moving=t>0&&t<1;
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
    +'<title>Measured bear-survival prediction with and without irrelevant columns</title>'
    +`<desc>All models use the same ${results.metadata.n_train} training and ${results.metadata.n_test} test rows per seed. Open circles are mean test ROC AUC using only bear type and response. Filled circles add three independent uniform noise columns. Small marks are individual simulation seeds. Dashed line: population oracle AUC ${oracle}. Animation between endpoints is not an additional fitted result.</desc>`;
  const title=mobile?'2 signal columns; then add 3 noise columns':'Keep the same encounters. Add three columns of pure noise.';
  s+=text(mobile?200:450,mobile?21:30,title,mobile?18:24,C.ink,'middle');
  const legendY=mobile?48:62,legendLeft=mobile?30:267;
  s+=circle(legendLeft,legendY-5,5,C.muted,false)+text(legendLeft+13,legendY,'Signal only',mobile?17:18,C.muted)
    +circle(mobile?193:470,legendY-5,5,C.muted,true)+text(mobile?206:483,legendY,'+ continuous noise',mobile?17:18,C.muted);
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
    const model=models[i],clean=caseFor(model.id,'signal_only'),noisy=caseFor(model.id,'continuous_noise');
    if(!clean||!noisy)throw new Error(`Missing measured bear comparison for ${model.id}`);
    const y=L.first+i*L.gap,active=i===current,transition=active?t:1;
    const xClean=scale(clean.mean_auc),xNoise=scale(noisy.mean_auc),x=xClean+(xNoise-xClean)*transition;
    const value=active&&moving?'':transition===0?clean.mean_auc.toFixed(3):`${clean.mean_auc.toFixed(3)} / ${noisy.mean_auc.toFixed(3)}`;
    s+=`<g data-model="${model.id}" data-clean-auc="${clean.mean_auc}" data-noise-auc="${noisy.mean_auc}">`
      +text(mobile?24:24,mobile?y-22:y-5,model.name,mobile?19:24,model.color)
      +(mobile?'':text(24,y+21,model.detail,18,C.muted))
      +line(xClean,y,x,y,model.color,3)
      +circle(xClean,y,mobile?6:7,model.color,false,'data-mark="clean-mean"')
      +circle(x,y,mobile?5:6,model.color,true,`data-mark="noise-mean" data-endpoint="${transition===1}"`)
      +text(mobile?376:894,mobile?y-22:y+7,value,mobile?18:22,model.color,'end','data-mark="mean-label" font-variant-numeric="tabular-nums"');
    const reveal=active?smooth((progress-.76)/.24):1;
    const records=results.records.filter(row=>row.model===model.id);
    s+=`<g opacity="${f(reveal)}" data-mark="seed-results">`;
    for(const row of records){
      const offset=(results.metadata.seeds.indexOf(row.seed)-(seedCount-1)/2)*(mobile?.65:.9);
      const markY=y+14+offset;
      s+=circle(scale(row.auc),markY,mobile?1.8:2.4,model.color,row.id==='continuous_noise',`data-seed="${row.seed}" data-case="${row.id}" data-auc="${row.auc}"`);
    }
    s+='</g></g>';
  }
  return s+'</g>';
}
