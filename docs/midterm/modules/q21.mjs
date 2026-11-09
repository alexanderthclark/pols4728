import {m,equation,svg,mathLabel,renderMath,COLORS as C} from '../shared.mjs';
import {original} from './q21-original.mjs';
const T=String.raw;
export const FIRST=[{a:1,b:1,p:12,n:2},{a:0,b:1,p:6,n:6},{a:0,b:0,p:2,n:12}];
export const SECOND=[{a:.9,b:.1,p:10,n:0},{a:.1,b:.9,p:10,n:0},{a:.8,b:.8,p:0,n:10},{a:0,b:0,p:0,n:10}];
const score=(r,model)=>model==='e'?(r.a+r.b)/2:r[model];
export function rocPoints(rows,model){
  const thresholds=[...new Set(rows.map(r=>score(r,model)))].sort((a,b)=>b-a);
  return [[0,0],...thresholds.map(t=>{const chosen=rows.filter(r=>score(r,model)>=t);return[chosen.reduce((s,r)=>s+r.n,0)/20,chosen.reduce((s,r)=>s+r.p,0)/20];})];
}
export const auc=pts=>pts.slice(1).reduce((v,p,i)=>v+(p[0]-pts[i][0])*(p[1]+pts[i][1])/2,0);
const curves={a:rocPoints(FIRST,'a'),b:rocPoints(FIRST,'b'),e:rocPoints(FIRST,'e'),ap:rocPoints(SECOND,'a'),bp:rocPoints(SECOND,'b'),ep:rocPoints(SECOND,'e')};
const color={a:C.blue,b:C.orange,e:C.green,ap:C.blue,bp:C.orange,ep:C.green};
const names={a:'A',b:'B',e:'E',ap:"A'",bp:"B'",ep:"E'"};
const fmt=x=>Number.isInteger(x)?String(x):String(Number(x.toFixed(3)));
const pt=([x,y])=>[54+266*x,298-266*y];
const text=(x,y,s,extra='')=>`<text x="${x}" y="${y}" font-size="14" fill="${C.ink}" ${extra}>${s}</text>`;
const line=(x1,y1,x2,y2,stroke=C.line,extra='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${stroke}" ${extra}/>`;
const path=(points,stroke,width=3,extra='')=>`<polyline points="${points.map(pt).map(p=>p.join(',')).join(' ')}" fill="none" stroke="${stroke}" stroke-width="${width}" stroke-linejoin="round" ${extra}/>`;
function chart({series=[],current=null,point=null,area=null,strip=-1,compare=false}={}){
  let s='';
  for(const v of [0,.5,1]){
    const [x,y]=pt([v,v]);s+=line(x,32,x,298)+line(54,y,320,y)+text(x,320,fmt(v),'text-anchor="middle"')+text(42,y+5,fmt(v),'text-anchor="end"');
  }
  s+=`<rect x="54" y="32" width="266" height="266" stroke="${C.ink}" stroke-width="1.2" fill="none"/>`;
  s+=text(187,348,'False positive rate','text-anchor="middle"')+text(15,165,'True positive rate','transform="rotate(-90 15 165)" text-anchor="middle"');
  if(area){
    const pts=curves[area];
    pts.slice(1).forEach((v,i)=>{
      const prev=pts[i];if(v[0]===prev[0])return;
      const poly=[[prev[0],0],prev,v,[v[0],0]].map(pt).map(p=>p.join(',')).join(' ');
      s+=`<polygon points="${poly}" fill="${color[area]}" opacity="${strip===i?.28:strip<0?.18:.045}"/>`;
      const [x]=pt([v[0],0]);s+=line(x,pt(v)[1],x,298,color[area],'stroke-dasharray="3 4" opacity=".55"');
    });
  }
  for(const key of series){
    const faded=current&&key!==current;
    s+=path(curves[key],color[key],faded?2.5:3,`${key==='b'||key==='bp'?'stroke-dasharray="7 5"':''} opacity="${faded?.55:1}"`);
  }
  if(current&&!series.includes(current)){
    const pts=curves[current];let end=point?pts.findIndex(p=>Math.abs(p[0]-point[0])<1e-9&&Math.abs(p[1]-point[1])<1e-9):pts.length-1;
    if(end<0)end=pts.length-1;
    s+=path(pts.slice(0,end+1),color[current],3.5,current==='b'||current==='bp'?'stroke-dasharray="7 5"':'');
  }
  if(compare){
    for(const [x,ya,yb] of [[.1,.6,.225],[.4,.6+.4/3,.9]]){
      const [xx]=pt([x,0]);s+=line(xx,pt([0,Math.max(ya,yb)])[1],xx,298,C.muted,'stroke-dasharray="3 4"');
      for(const [y,col] of [[ya,C.blue],[yb,C.orange]]){const [cx,cy]=pt([x,y]);s+=`<circle cx="${cx}" cy="${cy}" r="5" fill="${col}" stroke="white" stroke-width="1.5"/>`;}
    }
  }
  if(point){
    const [x,y]=pt(point);s+=line(x,y,x,298,color[current]||C.ink,'stroke-dasharray="3 4" opacity=".55"')+line(54,y,x,y,color[current]||C.ink,'stroke-dasharray="3 4" opacity=".55"');
    s+=`<circle cx="${x}" cy="${y}" r="6.5" fill="white" stroke="${color[current]||C.ink}" stroke-width="3"/>`;
  }
  return svg(s,'0 0 356 364','ROC curve in the unit square, with false positive rate on the horizontal axis and true positive rate on the vertical axis.');
}
function legend(keys){return `<div class="roc-legend">${keys.map(k=>`<span><i style="border-color:${color[k]};${k==='b'||k==='bp'?'border-top-style:dashed':''}"></i>${m(names[k])}</span>`).join('')}</div>`;}
function scoreTable(rows,{model=null,t=null,average=false,allScores=false,activeRows=null}={}){
  const counter=rows===SECOND;
  const headerModel=model?`s_{${names[model+(counter?'p':'')]}}`:null;
  const cols=allScores?`<th>${m(counter?"s_{A'}":'s_A')}</th><th>${m(counter?"s_{B'}":'s_B')}</th>${average?`<th class="roc-average">${m(counter?"s_{E'}":'s_E')}</th>`:''}`:`<th>${m(headerModel||'s')}</th>`;
  return `<table class="roc-scoretable"><thead><tr><th>Group</th>${cols}<th><span class="roc-pos-symbol" aria-hidden="true"></span> Pos.</th><th><span class="roc-neg-symbol" aria-hidden="true"></span> Neg.</th></tr></thead><tbody>${rows.map((r,i)=>{
    const on=activeRows?activeRows.includes(i):model&&t!==null?score(r,model)>=t:false;
    const cells=allScores?`<td>${fmt(r.a)}</td><td>${fmt(r.b)}</td>${average?`<td class="roc-average">${m(`\\frac{${fmt(r.a)}+${fmt(r.b)}}2=${fmt(score(r,'e'))}`)}</td>`:''}`:`<td>${fmt(score(r,model||'a'))}</td>`;
    return `<tr class="${on?'admitted':''}"><th><span class="roc-row-marker">${on?'✓':''}</span>${i+1}</th>${cells}<td>${r.p}</td><td>${r.n}</td></tr>`;
  }).join('')}</tbody></table>`;
}
function counts(rows,model,t){const chosen=rows.filter(r=>score(r,model)>=t);return{tp:chosen.reduce((s,r)=>s+r.p,0),fp:chosen.reduce((s,r)=>s+r.n,0)};}
function counterPanel(tp,fp){return `<div class="roc-counts"><div><span>Negatives admitted</span>${m(`\\mathrm{FPR}=\\frac{${fp}}{20}=${fmt(fp/20)}`)}</div><div><span>Positives admitted</span>${m(`\\mathrm{TPR}=\\frac{${tp}}{20}=${fmt(tp/20)}`)}</div></div>`;}
function sceneCount(rows,model,t,{ghost=[],full=false,note='',title='',phaseLabel=''}={}){
  const {tp,fp}=counts(rows,model,t),key=model+(rows===SECOND?'p':'');
  const graphKey=full?'Lines show complete curves. The ring marks the current threshold.':ghost.length?'Faint lines show complete reference curves. The ring ends the current model’s path.':'The ring ends the path at the current threshold.';
  return `<div class="roc-scene-head"><h3 class="viz-title">${title||`${names[key]}: admit scores at least ${fmt(t)}`}</h3><span class="roc-threshold">${m(`t=${fmt(t)}`)}</span></div><div class="roc-workbench"><div class="roc-data">${scoreTable(rows,{model,t})}<p class="roc-key">✓ Shaded rows meet the threshold.<br>${graphKey}</p>${phaseLabel?`<p class="roc-phase">${phaseLabel}</p>`:''}</div><div class="roc-graph">${legend([...ghost,key])}${chart({series:full?[...ghost,key]:ghost,current:key,point:[fp/20,tp/20]})}</div></div>${counterPanel(tp,fp)}${note?`<p class="viz-note">${note}</p>`:''}`;
}
function dots(n,kind){return Array.from({length:n},()=>`<span class="roc-case ${kind}"></span>`).join('');}
const strips={
  b:[{tex:T`\frac{0+0.9}{2}\times0.4=0.18`,label:'First strip · width 0.4'},{tex:T`\frac{0.9+1}{2}\times0.6=0.57`,label:'Second strip · width 0.6'}],
  e:[{tex:T`\frac{0+0.6}{2}\times0.1=0.03`,label:'First strip · width 0.1'},{tex:T`\frac{0.6+0.9}{2}\times0.3=0.225`,label:'Middle strip · width 0.3'},{tex:T`\frac{0.9+1}{2}\times0.6=0.57`,label:'Last strip · width 0.6'}]
};
function areaScene(model,phase){
  const items=strips[model],done=phase>=items.length;
  return `<h3 class="viz-title">Area = width × average height</h3><div class="roc-workbench roc-area-workbench"><div class="roc-data roc-area-list">${items.map((v,i)=>`<div class="roc-area-item ${phase===i?'active':''}"><span>${v.label}</span>${equation(v.tex)}</div>`).join('')}</div><div class="roc-graph">${legend([model])}${chart({series:[model],area:model,strip:done?-1:phase})}</div></div><div class="roc-area-total">${equation(model==='b'?T`\mathrm{AUC}_B=0.18+0.57=\boxed{0.75}`:T`\mathrm{AUC}_E=0.03+0.225+0.57=\boxed{0.825}`)}</div><p class="viz-note">${done?'All strips are shaded; add their areas.':'The darker strip matches the highlighted calculation.'} Straight segments follow the exam's AUC convention.</p>`;
}
function averageScene(rows){
  const counter=rows===SECOND;
  return `<h3 class="viz-title">Average each pair of scores</h3>${equation(counter?T`s_{E'}=\frac{s_{A'}+s_{B'}}2`:T`s_E=\frac{s_A+s_B}2`)}${scoreTable(rows,{allScores:true,average:true})}<div class="roc-order">${counter?`${m('0.8')}<span>ten negatives</span><b>→</b>${m('0.5')}<span>all twenty positives</span><b>→</b>${m('0')}<span>ten negatives</span>`:`${m('1')}<span>group 1</span><b>→</b>${m('0.5')}<span>group 2</span><b>→</b>${m('0')}<span>group 3</span>`}</div><p class="viz-note">Read from highest to lowest score. Average scores first; then construct a new ROC curve.</p>`;
}
function renderScene(stage,phase,container){
  container.classList.add('roc-visual');
  container.dataset.rocPhase=String(phase);
  let html='';
  if(stage===0){
    html=`<h3 class="viz-title">The same 40 observations, scored twice</h3>${scoreTable(FIRST,{allScores:true})}<div class="roc-class-totals"><div><span>20 actual positives</span><div class="roc-cases">${dots(20,'positive')}</div></div><div><span>20 actual negatives</span><div class="roc-cases">${dots(20,'negative')}</div></div></div>${equation(T`\mathrm{FPR}=\frac{\text{negatives admitted}}{20}\qquad\mathrm{TPR}=\frac{\text{positives admitted}}{20}`)}<p class="viz-note">Each point uses its own class total: 20, not 40.</p>`;
  }else if(stage===1)html=sceneCount(FIRST,'a',1,{title:'The given curve starts with group 1',note:'Above a threshold of 1, no cases enter: the starting point is (0, 0).'});
  else if(stage===2)html=sceneCount(FIRST,'b',1,{ghost:['a'],title:'Two groups share B’s highest score',note:'12 + 6 = 18 positives; 2 + 6 = 8 negatives.'});
  else if(stage===3)html=sceneCount(FIRST,'b',0,{ghost:['a'],full:true,title:'At zero, every observation enters',note:'B now connects (0, 0), (0.4, 0.9), and (1, 1).'});
  else if(stage===4)html=areaScene('b',phase);
  else if(stage===5){
    html=`<h3 class="viz-title">Look at height, not just total area</h3><div class="roc-workbench"><div class="roc-data roc-comparisons"><div><span>At false positive rate 0.1</span>${equation(T`\mathrm{TPR}_A=0.6`)}${equation(T`\mathrm{TPR}_B=0.225`)}<p>${m('A')} is higher.</p></div><div><span>At false positive rate 0.4</span>${equation(T`\mathrm{TPR}_A\approx0.733`)}${equation(T`\mathrm{TPR}_B=0.9`)}<p>${m('B')} is higher.</p></div></div><div class="roc-graph">${legend(['a','b'])}${chart({series:['a','b'],compare:true})}</div></div>${equation(T`\mathrm{AUC}_A=\mathrm{AUC}_B=0.75`)}<p class="viz-note">Compare the straight segments specified in the question. Neither curve stays above the other.</p>`;
  }else if(stage===6)html=averageScene(FIRST);
  else if(stage===7)html=sceneCount(FIRST,'e',1,{ghost:['a','b'],title:'First, admit the highest ensemble score'});
  else if(stage===8)html=sceneCount(FIRST,'e',.5,{ghost:['a','b'],title:'The new middle score creates a new vertex',note:'Group 2 adds six positives and six negatives.'});
  else if(stage===9)html=sceneCount(FIRST,'e',0,{ghost:['a','b'],full:true,title:'The last group completes the curve',note:'E is nowhere below A or B in this dataset.'});
  else if(stage===10)html=areaScene('e',phase);
  else if(stage===11){
    html=`<h3 class="viz-title">New observations, new score overlap</h3>${scoreTable(SECOND,{allScores:true})}<div class="roc-counter-setup"><div>${m("A'")}<span>scores group 1 highest</span></div><div>${m("B'")}<span>scores group 2 highest</span></div></div><p class="viz-note">Groups 1 and 2 are all positive. Groups 3 and 4 are all negative.</p>`;
  }else if(stage===12)html=sceneCount(SECOND,'a',.9,{title:'First, ten positives enter',note:'True positive rate rises; false positive rate stays at zero.'});
  else if(stage===13)html=sceneCount(SECOND,'a',.8,{title:'Next, ten negatives enter',note:'False positive rate rises; true positive rate stays at one-half.'});
  else if(stage===14)html=sceneCount(SECOND,'a',phase===0?.1:0,{full:phase>0,title:phase===0?'Then the other ten positives enter':'Finally, the remaining ten negatives enter',phaseLabel:phase===0?'Move up to (0.5, 1).':'Move right to (1, 1).'});
  else if(stage===15)html=sceneCount(SECOND,phase===0?'a':'b',.9,{ghost:phase===0?[]:['ap'],full:true,title:phase===0?`${m("A'")} selects positive group 1 first`:`${m("B'")} selects positive group 2 first`,note:'Different identities, identical counts at every threshold: the two full ROC curves coincide.'})+equation(T`\mathrm{AUC}_{A'}=\mathrm{AUC}_{B'}=(0.5)(0.5)+(0.5)(1)=0.75`);
  else if(stage===16)html=averageScene(SECOND);
  else if(stage===17)html=sceneCount(SECOND,'e',.8,{ghost:['ap'],title:'Averaging puts ten negatives first',note:'The first move is right along the bottom: (0, 0) → (0.5, 0).'});
  else if(stage===18)html=sceneCount(SECOND,'e',.5,{ghost:['ap'],title:'All twenty positives are tied at 0.5',note:'Both positive groups enter together. The curve jumps from (0.5, 0) to (0.5, 1).'});
  else{
    html=`<h3 class="viz-title">The final negatives finish a worse curve</h3><div class="roc-workbench"><div class="roc-data">${scoreTable(SECOND,{model:'e',t:0})}<div class="roc-final-comparison">${equation(T`\mathrm{AUC}_{A'}=\mathrm{AUC}_{B'}=0.75`)}${equation(T`\mathrm{AUC}_{E'}=(0.5)(1)=0.50`)}</div></div><div class="roc-graph">${legend(['ap','ep'])}${chart({series:['ap','ep'],current:'ep',point:[1,1],area:'ep'})}</div></div><p class="viz-note">The left half contributes zero area. The right half contributes 0.5. Averaging helped in the first dataset and hurt in this one.</p>`;
  }
  container.innerHTML=html;renderMath(container);
}
export const question21={
  id:21,kicker:'Ranking & ensembles',title:'Build the ROC curve, one group at a time.',
  dek:'Follow the same observations from scores to counts to ROC points. Then see exactly what averaging changes.',
  prompt:original.prompt,parts:original.parts,
  sections:[{label:'(a) Build and compare',step:0},{label:'(b) Why averaging helps',step:6},{label:'(c) Why it can hurt',step:11}],
  steps:[
    {title:'Keep the two denominators separate.',body:`<p>There are 20 positives and 20 negatives. The table groups observations with identical score pairs.</p><p>At each threshold, count how many of each class are predicted positive. Divide each count by 20.</p>`,caption:'Setup · Horizontal = negatives admitted / 20. Vertical = positives admitted / 20.'},
    {title:'Start with the given classifier.',body:`<p>Predict positive when the score is at least ${m('t')}. For ${m('A')} at ${m('t=1')}, only group 1 enters.</p><p>Its 2 negatives and 12 positives produce ${m('(2/20,12/20)=(0.1,0.6)')}.</p>`,caption:'Setup · No admitted cases gives (0, 0); all admitted cases gives (1, 1).'},
    {title:'Now count for B.',body:`<p>For ${m('B')}, groups 1 and 2 both have score 1. They enter together.</p><p>Add their counts: 18 positives and 8 negatives. Plot ${m('(8/20,18/20)=(0.4,0.9)')}.</p>`,caption:'Part (a) · Tied scores enter together.'},
    {title:'Complete B’s curve.',body:`<p>Lower the threshold to zero. Group 3 enters, bringing both counts to 20.</p><p>The last point is ${m('(1,1)')}. Connect it to the interior point and the origin with straight segments, as the question instructs.</p>`,caption:'Part (a) · Three vertices determine B’s complete ROC curve.'},
    {title:'Find the area under B.',body:`<p>Split at false positive rate 0.4. In each strip, multiply the width by the average of the two endpoint heights.</p><p>The areas are 0.18 and 0.57. Their sum is ${m('0.75')}.</p>`,caption:'Part (a) · Scroll to highlight each strip, then the complete area.'},
    {title:'Equal area is not dominance.',body:`<p>At low false positive rates, ${m('A')} is higher. Farther right, ${m('B')} is higher.</p><p>Both have AUC 0.75, but neither lies at least as high at every false positive rate.</p>`,caption:'Part (a) · AUC summarizes total area; dominance compares height everywhere.'},
    {title:'Average the scores first.',body:`<p>Compute ${m('(s_A+s_B)/2')} separately for each row. The ensemble scores are 1, 0.5, and 0.</p><p>Group 2 now has its own middle score. This changes the order in which groups can enter.</p>`,caption:'Part (b) · Average scores—not ROC curves or AUCs.'},
    {title:'At 1, the ensemble admits group 1.',body:`<p>The first group contributes 12 positives and 2 negatives.</p><p>The ensemble’s first interior point is ${m('(0.1,0.6)')}, the same interior point as ${m('A')}.</p>`,caption:'Part (b) · First threshold: 1.'},
    {title:'At 0.5, group 2 joins.',body:`<p>Six more positives and six more negatives enter. Cumulative counts become 18 and 8.</p><p>The next point is ${m('(0.4,0.9)')}. The ensemble can now reach both useful interior vertices in sequence.</p>`,caption:'Part (b) · Second threshold: 0.5. Add to the previous counts.'},
    {title:'At zero, the last group joins.',body:`<p>Group 3 brings the totals to 20 positives and 20 negatives, completing the curve at ${m('(1,1)')}.</p><p>In this dataset, the ensemble curve is nowhere below either individual curve.</p>`,caption:'Part (b) · Third threshold: 0. All 40 cases are admitted.'},
    {title:'Add the ensemble’s three strips.',body:`<p>The widths are 0.1, 0.3, and 0.6. Their areas are 0.03, 0.225, and 0.57.</p><p>Total AUC is ${m('0.825')}, exceeding both individual AUCs of 0.75.</p>`,caption:'Part (b) · The gain comes from the new score ordering.'},
    {title:'Start fresh with the second dataset.',body:`<p>Do not carry over the old score groups. The new dataset again contains 20 positives and 20 negatives.</p><p>${m("A'")} ranks one positive group first; ${m("B'")} ranks the other first. Both assign 0.8 to the same negative group.</p>`,caption:'Part (c) · Track group identity as well as each model’s ranking.'},
    {title:'At 0.9, A′ finds ten positives.',body:`<p>Only group 1 enters: ten positives and no negatives.</p><p>Move straight up from ${m('(0,0)')} to ${m('(0,0.5)')}.</p>`,caption:'Part (c) · Adding only positives moves the ROC curve up.'},
    {title:'At 0.8, ten negatives enter.',body:`<p>Group 3 now enters. The positive count stays at ten; the negative count rises to ten.</p><p>Move right to ${m('(0.5,0.5)')}.</p>`,caption:'Part (c) · Adding only negatives moves the ROC curve right.'},
    {title:'Finish A′: up, then right.',body:`<p>At 0.1, the remaining ten positives enter: move up to ${m('(0.5,1)')}.</p><p>At zero, the remaining ten negatives enter: move right to ${m('(1,1)')}. Scroll through these two additions.</p>`,caption:'Part (c) · The complete path is up, right, up, right.'},
    {title:'B′ swaps the positive groups.',body:`<p>Hold the threshold at 0.9. Scroll from ${m("A'")}, admitting group 1, to ${m("B'")}, admitting group 2. Both admit ten positives and no negatives, so the ring stays at ${m('(0,0.5)')}.</p><p>The class counts also match at every other threshold. The complete reference curves coincide, and each AUC is 0.75.</p>`,caption:'Part (c) · The table switches groups; the current point and complete ROC curve stay the same.'},
    {title:'Now average the new scores.',body:`<p>Both positive groups get ${m('(0.9+0.1)/2=0.5')}. The high-scoring negative group keeps ${m('(0.8+0.8)/2=0.8')}.</p><p>The ensemble now ranks those ten negatives ahead of every positive.</p>`,caption:'Part (c) · This change in ordering is the reason averaging hurts.'},
    {title:'The ensemble takes a wrong first step.',body:`<p>At threshold 0.8, group 3 enters by itself: ten negatives, zero positives.</p><p>The first ROC move is right along the bottom, to ${m('(0.5,0)')}.</p>`,caption:'Part (c) · High false positive rate before finding a single positive.'},
    {title:'All the positives arrive at once.',body:`<p>Both positive groups have score 0.5, so lowering the threshold to 0.5 admits all twenty together.</p><p>The negative count stays at ten. Move vertically to ${m('(0.5,1)')}.</p>`,caption:'Part (c) · Ties cannot be split by a deterministic score threshold.'},
    {title:'Finish the curve and compare.',body:`<p>At zero, the last ten negatives enter and the curve ends at ${m('(1,1)')}.</p><p>Its AUC is 0.50, below 0.75 for each individual model. Here averaging puts ten negatives ahead of every positive. Equal individual AUCs therefore do not guarantee an improvement from averaging.</p>`,caption:'Part (c) · No universal improvement: inspect what score averaging does to the rankings.'}
  ],
  takeaway:`First dataset: ${m(T`0.75,\;0.75\;\longrightarrow\;0.825`)}. Second dataset: ${m(T`0.75,\;0.75\;\longrightarrow\;0.50`)}. Average the scores, order the groups, count each class, and plot the resulting ROC points.`,
  render(stage,container){renderScene(stage,stage===4?2:stage===10?3:stage===14||stage===15?1:0,container);},
  update(progress,container,stage){
    const count=stage===4?3:stage===10?4:stage===14||stage===15?2:1;
    if(count===1)return;
    const phase=Math.min(count-1,Math.floor(Math.max(0,progress)*count/0.75));
    if(Number(container.dataset.rocPhase)!==phase)renderScene(stage,phase,container);
  }
};
