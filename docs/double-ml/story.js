import { rows, groups, summary, clamp, lerp, smoothstep, scrollState } from './model.mjs';

// Every pose is a pure function of scroll position. No timer advances the lesson.
const C = { ink:'#242424', blue:'#234E70', rust:'#A54F32', teal:'#007F73', muted:'#606060', rule:'#767676' };
const steps = [...document.querySelectorAll('.step')];
const diagram = document.querySelector('#diagram');
const equation = document.querySelector('#equation');
const caption = document.querySelector('#stage-caption');
const kicker = document.querySelector('#stage-kicker');
const motionPreference = matchMedia('(prefers-reduced-motion: reduce)');
const text = (x,y,value,extra='') => `<text x="${x}" y="${y}" ${extra}>${value}</text>`;
const line = (x1,y1,x2,y2,extra='') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${C.rule}" stroke-width="1.5" ${extra}/>`;
const rect = (x,y,w,h,extra='') => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="3" fill="white" stroke="${C.rule}" stroke-width="1.5" ${extra}/>`;
const arrow = (x1,y1,x2,y2,p=1,color=C.blue) => line(x1,y1,lerp(x1,x2,p),lerp(y1,y2,p),`style="stroke:${color}" stroke-width="2.5" marker-end="url(#arrow)"`);
const phase = (p,start=0,end=1) => smoothstep((p-start)/(end-start));
const n = v => Number(v.toFixed(2));
const color = row => row.tier === 'low' ? C.blue : C.rust;
const dot = (row,x,y,r=6,opacity=1) => `<circle data-student="${row.id}" cx="${n(x)}" cy="${n(y)}" r="${r}" fill="${row.d?color(row):'white'}" stroke="${color(row)}" stroke-width="2.1" opacity="${opacity}"><title>Starting score ${row.x}; ${row.d?'tutored':'untutored'}; final score ${row.y}</title></circle>`;
const group = (content,opacity=1) => `<g opacity="${clamp(opacity)}">${content}</g>`;
const legend = () => text(46,25,'○ Untutored    ● Tutored','class="small"') + text(465,25,'Blue: lower X','class="tiny" style="fill:#234E70"') + text(465,50,'Rust: higher X','class="tiny" style="fill:#A54F32"');
const eq = value => `<span>${value}</span>`;
const s = 'text-anchor="middle"';
const frame = (title,description,content) => `<title id="diagram-title">${title}</title><desc id="diagram-description">${description}</desc><defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 1L8 5L0 9" fill="none" stroke="${C.blue}" stroke-width="1.4"/></marker><clipPath id="plot-clip"><rect x="85" y="65" width="560" height="340"/></clipPath></defs>${content}`;

function effect(p) {
  const a = phase(p,.08,.65), x = value => 85+value*5;
  let svg = text(85,58,'Final score','class="small"');
  for (const [i,baseline] of [50,90].entries()) {
    const y=155+i*155;
    svg += text(85,y-42,baseline===50?'Starting score below 60':'Starting score at least 60','class="small"');
    svg += rect(85,y-14,baseline*5,28,`style="fill:${C.blue};stroke:none"`);
    svg += rect(x(baseline),y-14,50*a,28,`style="fill:${C.rust};stroke:none"`);
    svg += `<circle cx="${x(baseline)}" cy="${y}" r="8" fill="white" stroke="${C.ink}" stroke-width="2"/>`;
    svg += `<circle data-effect="treated" cx="${n(x(baseline+10*a))}" cy="${y}" r="8" fill="${C.rust}"/>`;
    svg += text(x(baseline)-8,y+43,String(baseline),'class="small" text-anchor="end"');
    svg += group(text(x(baseline)+25,y-28,'+10',`${s} style="fill:${C.rust}"`)+text(x(baseline+10)+8,y+43,String(baseline+10),'class="small"'),a);
  }
  svg += line(85,404,635,404);
  for(const v of [0,50,100,110])svg+=line(x(v),404,x(v),412)+text(x(v),442,String(v),`${s} class="small"`);
  return {svg, title:'A ten-point effect for either starting group',desc:'Tutoring adds ten points to a baseline final score of fifty or ninety. The treatment marker moves ten points to the right as you scroll.',equation:eq('Y = 50 + 40 · 1{X ≥ 60} + 10D'),caption:'The constructed rule holds X fixed: changing D from 0 to 1 adds 10 points.'};
}

function memorize(p) {
  const a=phase(p,.04,.75), xs=[35,42,49,56,63,70,77,84], ds=[0,1,1,0,1,0,0,1];
  const people=xs.map((x,i)=>({id:`memorize-${i}`,x,d:ds[i],y:50+40*Number(x>=60)+10*ds[i],tier:x<60?'low':'high'}));
  const xp=i=>110+i*72, yp=v=>255-(v-40)*2.6;
  let svg=text(75,34,'Observed final scores and tree predictions','class="small"')+line(80,75,80,270)+line(80,270,660,270);
  for(const v of [50,75,100])svg+=text(65,yp(v)+7,String(v),'class="small" text-anchor="end"');
  let path='';
  people.forEach((r,i)=>{
    const fitted=lerp(75,r.y,a), x=xp(i), y=yp(fitted);
    path+=i?`H${x-36}V${y}H${x+(i===7?28:36)}`:`M${x-28},${y}H${x+36}`;
    if(i<7)svg+=line(x+36,65,x+36,270,`stroke-dasharray="4 6" opacity="${a*.5}"`);
    svg+=line(x,yp(r.y),x,y,`style="stroke:${C.teal}" stroke-width="3"`)+dot(r,x,yp(r.y),7);
    svg+=text(x,297,String(r.x),`${s} class="tiny"`);
    const ry=395-(r.y-fitted)*1.7;
    svg+=line(x,395,x,ry,`style="stroke:${C.teal}" stroke-width="3"`)+`<circle data-residual="${i}" cx="${x}" cy="${n(ry)}" r="5" fill="${C.teal}"/>`;
  });
  svg+=`<path d="${path}" fill="none" stroke="${C.blue}" stroke-width="3"/>`+text(660,324,'Starting score X','class="tiny" text-anchor="end"');
  svg+=line(80,395,660,395)+text(75,357,'What remains after subtracting the fitted value','class="small"')+text(63,402,'0','class="small" text-anchor="end"');
  svg+=group(text(360,463,'Every training residual becomes zero.',`${s} class="small"`),phase(p,.65,.85));
  return {svg,title:'A tree memorizes the students used to fit it',desc:'Eight distinct starting scores allow eight separate tree leaves. The fitted steps move to the observed scores, and all eight residuals shrink to zero.',equation:eq('Rᵢ = Yᵢ − m̂(Xᵢ) → 0'),caption:'One student per leaf. Zero training residuals reflect memorization, even with randomized tutoring.'};
}

function holdout(p) {
  const split=phase(p,0,.3), fitted=phase(p,.28,.56), predict=phase(p,.55,.88);
  let svg=text(175,58,'Training students',s)+text(545,58,'Held-out students',s)+line(360,80,360,445,'stroke-dasharray="5 7"');
  for(let i=0;i<16;i++){
    const right=i>=8, j=i%8, targetX=(right?475:105)+(j%4)*46, targetY=115+Math.floor(j/4)*45;
    const x=lerp(215+(i%8)*40,targetX,split),y=lerp(115+Math.floor(i/8)*45,targetY,split);
    svg+=dot({id:`split-${i}`,x:35+i*3,d:i%2,tier:i%3?'low':'high',y:50},x,y,8);
  }
  svg+=arrow(175,192,175,246,fitted)+group(rect(63,256,224,78)+text(175,286,'Fit m̂(X)',s)+text(175,315,'Then freeze it',`${s} class="small"`),fitted);
  svg+=arrow(287,295,425,295,predict)+group(rect(430,249,230,91)+text(545,282,'Predict Y',s)+text(545,315,'using only X',`${s} class="small"`),predict);
  svg+=group(text(545,389,'Y − held-out prediction',`${s} class="small"`)+text(360,459,'Their own Y never trained their prediction.',`${s} class="small"`),predict);
  return {svg,title:'Fit on one group, predict the other',desc:'Students separate into training and held-out groups. A fitted tree is frozen and sends predictions to students whose outcomes did not train it.',equation:eq('E[held-out residual gap] = 10'),caption:'For randomized tutoring, the held-out residual gap targets 10 in expectation. One split can be noisy.'};
}

const plot={left:100,right:635,top:80,bottom:394};
const py=(v,min,max)=>plot.bottom-(v-min)/(max-min)*(plot.bottom-plot.top);
const px=v=>plot.left+(v+1)/2*(plot.right-plot.left);
function rankInGroup(row){return rows.filter(r=>r.tier===row.tier&&r.d===row.d).findIndex(r=>r.id===row.id)}
function offset(row){const i=rankInGroup(row),count=groups.find(g=>g.tier===row.tier&&g.d===row.d).count;return (i-(count-1)/2)*7}
function rawPoint(row){return {x:(row.d?530:220)+offset(row),y:py(row.y,40,110)}}
function gridPoint(row){const at=rows.filter(r=>r.x===row.x).findIndex(r=>r.id===row.id);return{x:125+[40,50,60,70].indexOf(row.x)*155+(at%2)*22-11,y:116+Math.floor(at/2)*45}}
function composition(p){
  const a=phase(p,.08,.7);let svg=legend();
  [40,50,60,70].forEach((v,i)=>{const x=125+i*155;svg+=group(text(x,76,`X = ${v}`,s)+text(x,320,v<60?'6 ○ + 2 ●':'2 ○ + 6 ●',`${s} class="small"`),1-a)});
  svg+=group(line(100,65,100,394)+line(100,394,635,394),a);
  for(const v of [50,60,90,100])svg+=group(text(85,py(v,40,110)+6,String(v),'class="small" text-anchor="end"'),a);
  rows.forEach(r=>{const f=gridPoint(r),t=rawPoint(r);svg+=dot(r,lerp(f.x,t.x,a),lerp(f.y,t.y,a),6)});
  svg+=group(text(220,435,'Untutored',s)+text(530,435,'Tutored',s),a);
  svg+=group(line(145,py(60,40,110),295,py(60,40,110),`style="stroke:${C.ink}" stroke-dasharray="6 4"`)+line(455,py(90,40,110),605,py(90,40,110),`style="stroke:${C.ink}" stroke-dasharray="6 4"`)+text(220,py(60,40,110)-17,'Mean 60',`${s} class="small"`)+text(530,py(90,40,110)+42,'Mean 90',`${s} class="small"`),phase(p,.6,.87));
  return{svg,title:'The same 32 students regroup by tutoring status',desc:'At each low starting score six students are untutored and two tutored; the pattern reverses at each high starting score. Regrouping them shows average outcomes sixty and ninety.',equation:eq('90 − 60 = 30 = 10 + 20'),caption:'The raw gap mixes a 10-point tutoring effect with a 20-point difference in baseline outcomes.'};
}

function outcome(p){
  const a=phase(p,.05,.52), bars=phase(p,.55,.88);
  let svg=group(rect(246,22,228,53)+text(360,57,'Predict Y from X',s)+line(360,75,190,136)+line(360,75,530,136)+text(180,118,'X &lt; 55',`${s} class="small"`)+text(540,118,'X ≥ 55',`${s} class="small"`),a);
  svg+=group(text(190,173,'52.5',`${s} class="large" style="fill:${C.blue}"`)+text(530,173,'97.5',`${s} class="large" style="fill:${C.rust}"`),a);
  rows.forEach(r=>{const f=rawPoint(r),members=rows.filter(v=>v.tier===r.tier),i=members.findIndex(v=>v.id===r.id);const t={x:(r.tier==='low'?130:470)+(i%4)*36,y:220+Math.floor(i/4)*28};svg+=dot(r,lerp(f.x,t.x,a),lerp(f.y,t.y,a),6)});
  const bx=90,scale=4.4;
  [[50,2.5,C.blue,369],[90,7.5,C.rust,430]].forEach(([base,extra,c,y])=>{
    svg+=group(rect(bx,y-15,base*scale,22,`style="fill:${c};stroke:none"`)+rect(bx+base*scale,y-15,extra*scale*bars,22,`style="fill:${C.teal};stroke:none"`)+text(650,y+3,`${base} + ${extra}`,`class="small" text-anchor="end"`),bars);
  });
  svg+=group(text(90,341,'Baseline + expected tutoring contribution','class="small"'),bars);
  return{svg,title:'The outcome model includes expected tutoring',desc:'A split at fifty-five yields predictions fifty-two point five and ninety-seven point five. Each adds the baseline outcome to ten times the local tutoring probability.',equation:eq('m̂(X) = baseline(X) + 10 · Pr(D = 1 | X)'),caption:'The model predicts observed Y. Its predictions include 2.5 or 7.5 points associated with the tutoring rate.'};
}

function axes({min=-10,max=10,horizontal='D',subtract=0}={}){
  let svg=line(100,70,100,394)+line(100,394,635,394);
  const ticks=max<=10?[-10,-5,0,5,10]:[-10,0,10,50,100].filter(v=>v>=min&&v<=max);
  ticks.forEach(v=>{const y=py(v,min,max);svg+=line(95,y,635,y,`opacity="${v===0?1:.22}" ${v===0?'':'stroke-dasharray="3 6"'}`)+text(84,y+6,String(v),'class="small" text-anchor="end"')});
  if(horizontal==='T'){
    [-.75,-.25,.25,.75].forEach(v=>svg+=text(px(v),425,String(v),`${s} class="tiny"`));
    svg+=line(px(0),70,px(0),394,'stroke-dasharray="4 5"');
  }else if(horizontal==='D')svg+=text(220,425,'0: untutored',`${s} class="small"`)+text(530,425,'1: tutored',`${s} class="small"`);
  return svg;
}

function residual(p){
  const a=phase(p,.04,.55), zoom=phase(p,.57,.82), min=-10, max=lerp(110,10,zoom);let svg=legend();
  svg+=axes({min,max});
  rows.forEach(r=>{const f=rawPoint(r),x=f.x,y=py(r.y-a*r.m,min,max);svg+=`<g clip-path="url(#plot-clip)">${line(x,py(r.y,min,max),x,y,`style="stroke:${color(r)}" opacity="${.3*(1-zoom)}"`)}</g>`+dot(r,x,y,6)});
  const show=phase(p,.82,.95),ctrl=py(-3.75,-10,10),treat=py(3.75,-10,10);
  svg+=group(line(145,ctrl,295,ctrl,`style="stroke:${C.ink}" stroke-dasharray="6 4"`)+text(220,ctrl+39,'Mean −3.75',`${s} class="small"`)+line(455,treat,605,treat,`style="stroke:${C.ink}" stroke-dasharray="6 4"`)+text(530,treat-17,'Mean 3.75',`${s} class="small"`),show);
  svg+=text(360,472,a<.99?'Subtract each outcome prediction ↓':zoom<.99?'Zoom in on the remaining differences':'Outcome residual R = Y − m̂(X)',`${s} class="small"`);
  return{svg,title:'Move each final score to its outcome residual',desc:'The same thirty-two students move vertically as the outcome predictions are subtracted. The residual means finish at minus three point seven five and three point seven five, a gap of seven point five.',equation:eq(`Rᵢ = Yᵢ − m̂(Xᵢ)${a>.99?'　 ·　 gap = 7.5':''}`),caption:'Adjusting only Y gives 7.5, because tutoring still varies systematically with X.'};
}

function treatment(p){
  const pan=phase(p,.02,.22),a=phase(p,.25,.78);let svg=legend()+axes({horizontal:'none'});
  [0,1].forEach(d=>svg+=group(text(lerp(d?530:220,px(d),pan),425,String(d),`${s} class="small"`),1-a));
  [-.75,-.25,.25,.75].forEach(v=>svg+=group(text(px(v),425,String(v),`${s} class="tiny"`),a));
  svg+=line(px(0),70,px(0),394,`stroke-dasharray="4 5" opacity="${pan}"`);
  rows.forEach(r=>{const f=rawPoint(r),startX=lerp(f.x,px(r.d),pan),endX=px(r.t),x=startX-267.5*a*r.e,y=py(r.r,-10,10);svg+=line(startX,y,endX,y,`style="stroke:${color(r)}" opacity="${pan*.4*(1-a)}" marker-end="url(#arrow)"`)+dot(r,x,y,6)});
  svg+=text(360,469,'Tutoring residual T = D − ê(X)',`${s} class="small"`);
  svg+=group(text(192,86,'ê = ¼ for lower X','class="small" style="fill:#234E70"')+text(402,115,'ê = ¾ for higher X','class="small" style="fill:#A54F32"'),1-phase(p,.48,.72));
  return{svg,title:'Now move tutoring to its residual',desc:'The horizontal axis first makes room for negative values. Keeping outcome residuals fixed, students move left as the predicted tutoring probability is subtracted.',equation:eq('Tᵢ = Dᵢ − ê(Xᵢ)'),caption:p<.25?'First, make room on the horizontal axis for negative residuals. The outcome residuals stay fixed.':'Subtract the predicted tutoring probability: ¼ for lower X, ¾ for higher X. Every student moves left.'};
}

function slope(p){
  const a=phase(p,.06,.55), compress=phase(p,.52,.82);let svg=legend()+axes({horizontal:'T'});
  const drawX=lerp(-.85,.85,a);svg+=line(px(-.85),py(-8.5,-10,10),px(drawX),py(10*drawX,-10,10),`style="stroke:${C.teal}" stroke-width="3"`);
  rows.forEach(r=>svg+=dot(r,px(r.t),py(r.r,-10,10),6,lerp(1,.23,compress)));
  groups.forEach(g=>{const x=px(g.t),y=py(g.r,-10,10);svg+=group(`<circle cx="${x}" cy="${y}" r="9" fill="${g.d?color(g):'white'}" stroke="${color(g)}" stroke-width="2.4"/>`+text(x+(g.t<0?18:-18),y+(g.t<0?-17:33),`×${g.count}`,`class="small" text-anchor="${g.t<0?'start':'end'}"`),compress)});
  svg+=group(text(385,95,'R = 10T','class="large" style="fill:#007F73"')+text(390,129,'Slope = 10','class="small"'),phase(p,.42,.66));
  svg+=text(360,469,'Same 32 students; four residual pairs',`${s} class="small"`);
  return{svg,title:'The adjusted relationship has slope ten',desc:'The four distinct residual pairs lie on R equals ten times T. Counts of twelve, four, four, and twelve retain all thirty-two students when the points are combined.',equation:eq('θ̂ = Σ TᵢRᵢ / Σ Tᵢ² = 60 / 6 = 10'),caption:'Fit one line through the origin using every student. Multiplicities keep the 32-student weighting visible.'};
}

function general(p){
  const a=phase(p,.05,.38), b=phase(p,.35,.67), c=phase(p,.65,.9);let svg=text(80,57,'Controls X',`${s} class="small"`)+rect(35,180,90,90)+text(80,232,'X',`${s} class="large"`);
  svg+=arrow(125,210,235,115,a)+arrow(125,240,235,310,a);
  svg+=group(rect(242,73,205,82)+text(345,105,'Predict Y',s)+text(345,137,'m̂(X)',`${s} class="small"`)+rect(242,272,205,82)+text(345,304,'Predict D',s)+text(345,336,'ê(X)',`${s} class="small"`),a);
  svg+=arrow(447,114,525,114,b)+arrow(447,313,525,313,b);
  svg+=group(text(608,105,'R = Y − m̂(X)',`${s} class="small"`)+text(608,304,'T = D − ê(X)',`${s} class="small"`),b);
  svg+=group(line(608,135,608,206)+line(608,206,360,420)+line(608,333,360,420)+rect(248,403,225,66)+text(360,445,'Estimate the effect',`${s} class="small"`),c);
  svg+=group(text(160,384,'Two supporting predictions',`${s} class="small"`),a);
  return{svg,title:'Predictions support the effect estimate',desc:'Controls feed two prediction functions. Subtracting those predictions produces outcome and treatment residuals, which enter a common estimating rule for the effect.',equation:eq('Σ Tᵢ(Rᵢ − θTᵢ) = 0'),caption:'Orthogonality makes the population rule locally insensitive to small nuisance errors. Accurate enough predictions still matter.'};
}

function crossfit(p){
  const a=phase(p,.04,.4),b=phase(p,.38,.73),pool=phase(p,.72,.95);
  let svg=text(360,35,'Two folds; each student predicted out of sample',`${s} class="small"`);
  for(const [i,letter] of ['A','B'].entries()){
    const y=115+i*150;
    svg+=rect(42,y-47,210,94)+text(147,y-10,`Train on ${letter}`,s)+text(147,y+24,'both models',`${s} class="small"`);
    const visible=i?b:a;svg+=arrow(254,y,440,y,visible);
    svg+=group(rect(449,y-47,229,94)+text(563,y-10,`Predict ${i?'A':'B'}`,s)+text(563,y+24,'held-out R and T',`${s} class="small"`),visible);
    for(let j=0;j<6;j++){
      const x=lerp(483+j*29,270+j*31,pool),dy=lerp(y+62,408+i*24,pool);
      svg+=group(`<circle data-fold="${letter}-${j}" cx="${n(x)}" cy="${n(dy)}" r="6" fill="${i?C.rust:C.blue}"/>`,visible);
    }
  }
  svg+=group(text(360,477,'Pool the held-out pairs → estimate the effect',`${s} class="small"`),pool);
  return{svg,title:'Cross-fitting swaps the training roles',desc:'First train both models on A and predict B. Then train fresh models on B and predict A. The held-out residual pairs from both groups move together into the final estimate.',equation:eq('Train A → predict B; train B → predict A'),caption:'This is the cross-fitting extension of a single holdout. Each student’s predictions come from other students.'};
}

const renderers={effect,memorize,holdout,composition,outcome,residual,treatment,slope,general,crossfit};
let positions=[], scheduled=false, lastIndex=-1;
const nav=document.querySelector('.lesson-nav');
const previous=document.querySelector('#previous'), next=document.querySelector('#next');
let current={index:0,progress:0};

function measure(){positions=steps.map(step=>({id:step.dataset.scene,top:step.getBoundingClientRect().top+scrollY,height:step.offsetHeight}));requestFrame()}
function requestFrame(){if(!scheduled){scheduled=true;requestAnimationFrame(update)}}
function update(){
  scheduled=false;
  current=scrollState(scrollY,positions,innerHeight,{anchor:innerWidth<1000?.67:.57,reducedMotion:motionPreference.matches});
  if(current.index<0)return;
  const scene=renderers[current.id](current.motionProgress);
  diagram.innerHTML=frame(scene.title,scene.desc,scene.svg);
  diagram.dataset.scene=current.id;diagram.dataset.progress=current.progress.toFixed(5);
  kicker.textContent=scene.title;equation.innerHTML=scene.equation;equation.removeAttribute('aria-label');caption.textContent=scene.caption;
  document.querySelector('#scroll-progress').style.width=`${current.overall*100}%`;
  steps.forEach((step,i)=>{
    const reveal=step.querySelector('.reveal');
    const revealProgress=i<current.index?1:i>current.index?0:phase(current.progress,.38,.62);
    reveal.style.opacity=revealProgress;reveal.style.transform=`translateY(${18*(1-revealProgress)}px)`;
    step.classList.toggle('is-active',i===current.index);
  });
  if(lastIndex!==current.index){document.querySelector('#step-count').textContent=`${current.index+1} / ${steps.length}`;lastIndex=current.index}
  const lesson=document.querySelector('#lesson').getBoundingClientRect();
  nav.classList.toggle('is-visible',lesson.top<innerHeight*.65&&lesson.bottom>innerHeight*.5);
  previous.disabled=current.index===0&&current.progress<.03;
  next.textContent=current.index===steps.length-1?'Reading ↓':'Next →';
}
function go(direction){
  const at=current.index+direction;
  if(at>=steps.length){document.querySelector('#references').scrollIntoView();return}
  const target=positions[Math.max(0,at)];
  window.scrollTo({top:Math.max(0,target.top-innerHeight*(innerWidth<1000?.67:.57)+target.height*.04),behavior:'auto'});
}
previous.addEventListener('click',()=>go(-1));next.addEventListener('click',()=>go(1));
document.addEventListener('keydown',event=>{
  if(event.altKey||event.ctrlKey||event.metaKey||/INPUT|TEXTAREA|SELECT/.test(event.target.tagName))return;
  if(event.key==='ArrowRight'||event.key==='ArrowLeft'){event.preventDefault();go(event.key==='ArrowRight'?1:-1)}
});
addEventListener('scroll',requestFrame,{passive:true});addEventListener('resize',measure);addEventListener('load',measure);motionPreference.addEventListener('change',requestFrame);
document.documentElement.classList.add('js');
document.fonts.ready.then(measure);
measure();
