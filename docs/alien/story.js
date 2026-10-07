import {illustration,fitRule,predict,summarize,simulate} from './model.mjs';
import {predictionLayout,crowdLayout,decompositionSegments} from './geometry.mjs';

const $=id=>document.getElementById(id);
const svg=$('diagram'),art=$('scene-art'),population=$('population'),annotations=$('annotations');
const steps=[...document.querySelectorAll('.step')];
const labels=['One alien learns from five games','The closest loss and win locate a cutoff','Test a game outside the training sample','The same procedure, different training samples','Fixed input: x₀ = +0.5 · true outcome: 1','1,000 independent training samples','Prediction axis · each mark is one alien','Bias: average prediction minus truth','Variance: squared distances from the mean','MSE: squared distances from the truth','MSE = squared bias + variance + noise','The same experiment with new settings'];
const short=['Five games','Learn a cutoff','Test the rule','Different samples','One question','1,000 aliens','Expectation','Bias','Variance','MSE','Decomposition','Explore'];
const workedCaptions=['Final scores are already known. y = 1 is a win; y = 0 is a loss.','The nearest observations leave a gap from −4 to +1. The midpoint is −1.5.','The hatched interval marks disagreements with the true rule.','Each row shows one alien’s five observations and its fitted cutoff.','A circle / 1 means predict win. A square / 0 means predict loss.','Exactly 1,000 marks, in training-sample order: 468 predict loss; 532 predict win.','The same 1,000 marks, regrouped by prediction. Their weighted mean is 0.532.','The arrow runs from the true outcome to the mean. Its signed length is −0.468.','Square each distance to the mean, then average over the 1,000 aliens.','The 468 loss predictions have squared error 1; the 532 win predictions have error 0.','Bar lengths use squared-error units. Outcome noise is zero at this fixed input.','Every mark still represents one independently trained alien.'];
let scene=-1,selected=0,seed=10,allWinsIndex,workedRules,workedStats,explorerRules,explorerStats;
let particleNodes=[];
const exampleRule=fitRule(illustration);
const fmt=(v,d=3)=>Math.abs(v)<1e-12?(0).toFixed(d):v.toFixed(d);
const signed=v=>`${v<0?'−':'+'}${Number.isFinite(v)?Math.abs(v):'∞'}`;
const text=(x,y,value,size=18,anchor='start',className='')=>`<text x="${x}" y="${y}" font-size="${size}" text-anchor="${anchor}" class="${className}">${value}</text>`;
const line=(x1,y1,x2,y2,className='axis',extra='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" class="${className}" ${extra}/>`;
const glyph=(x,y,scale=1)=>`<use href="#alien-glyph" transform="translate(${x} ${y}) scale(${scale})" class="glyph"/>`;
const mark=(x,y,win,r=6)=>win?`<circle cx="${x}" cy="${y}" r="${r}" fill="var(--blue)"/>`:`<rect x="${x-r}" y="${y-r}" width="${r*2}" height="${r*2}" fill="var(--rust)"/>`;
function dimensions(){const mobile=matchMedia('(max-width:760px)').matches;return {mobile,W:mobile?(innerWidth>500?600:400):800,H:mobile?250:550,fs:mobile?14:19};}
function svgDescription(title,description){$('diagram-title').textContent=title;$('diagram-description').textContent=description;}
function tableScene(d){
 const {mobile:m,W,fs}=d;const left=m?88:215,right=m?390:W-20,row=m?38:66,top=m?72:117;
 let s=glyph(m?18:48,m?136:233,m?.8:1.6);
 const columns=m?[left,left+75,left+151,right-14]:[left+20,left+167,left+309,right-20];
 ['Knicks','Spurs','x','y'].forEach((label,i)=>s+=text(columns[i],top-row*.75,label,m?12:17,i===3?'middle':'start','muted'));
 illustration.forEach(([k,spurs],i)=>{const y=top+i*row;const diff=k-spurs;const win=diff>0;s+=line(left,y+row*.42,right,y+row*.42);s+=text(columns[0],y,k,fs);s+=text(columns[1],y,spurs,fs);s+=text(columns[2],y,signed(diff),fs,'start',win?'blue':'rust');s+=text(columns[3],y,Number(win),fs,'middle');});
 return s;
}
function trainingScene(d,truth){
 const {mobile:m,W,fs}=d;const left=m?55:120,right=m?378:W-35,map=x=>left+(x+6)/18*(right-left),axis=m?226:376,cut=map(-1.5),zero=map(0),pointY=m?171:285;
 let s='';
 s+=text(left,m?32:63,'Observed point differentials',m?13:19,'start','muted');
 s+=line(left,axis,right,axis);
 [-4,0,4,8,12].forEach(x=>{s+=line(map(x),axis,map(x),axis+6);s+=text(map(x),axis+(m?23:32),x,fs,'middle','muted');});
 s+=text((left+right)/2,m?288:499,'Knicks points − Spurs points',m?13:18,'middle','muted');
 if(truth){s+=`<rect x="${cut}" y="${m?80:133}" width="${zero-cut}" height="${axis-(m?80:133)}" fill="url(#error-hatch)"/>`;s+=line(zero,m?76:128,zero,axis,'reference');s+=text(zero+9,m?70:116,'True: 0',m?12:18);}
 const groups=new Map();illustration.forEach(([k,spurs])=>{const diff=k-spurs;const count=groups.get(diff)||0;groups.set(diff,count+1);const y=pointY-count*(m?29:44);s+=mark(map(diff),y,diff>0,m?5:8);if(!truth||diff!==1||count===1)s+=text(map(diff),y-(m?12:19),signed(diff),m?12:17,'middle',diff>0?'blue':'rust');});
 s+=line(cut,m?93:128,cut,axis,'model');
 s+=text(cut-(m?7:13),m?77:116,'Learned: −1.5',m?12:18,'end','blue');
 if(!truth){const y=m?117:194;s+=line(map(-4),y,map(1),y,'axis');s+=line(map(-4),y-5,map(-4),y+5);s+=line(map(1),y-5,map(1),y+5);s+=text(map(-4)-(m?3:8),y-(m?12:19),'Closest loss',m?11:16,'end','rust');s+=text(map(1)+(m?3:8),y-(m?12:19),'Closest win',m?11:16,'start','blue');}
 if(truth){const x=map(-1);s+=`<polygon points="${x},${axis-7} ${x-6},${axis-17} ${x+6},${axis-17}" fill="var(--rust)"/>`;s+=line(x,m?207:343,x,axis-17,'measure');s+=text(x-(m?5:10),m?210:345,'New game: −1',m?12:18,'end','rust');s+=text(left,m?267:453,'Alien: win (1)   ·   Actual: loss (0)',m?13:20);}
 else{s+=text(left,m?267:453,'Predict loss',m?13:18);s+=text(right,m?267:453,'Predict win',m?13:18,'end');}
 return s;
}
function rowIndices(){
 const indices=[selected,(selected+1)%workedRules.length];
 const third=indices.includes(allWinsIndex)?(selected+2)%workedRules.length:allWinsIndex;
 return [...indices,third];
}
function sampleScene(d,fixed){
 const {mobile:m,W,fs}=d;const left=m?83:175,right=m?319:672,map=x=>left+(x+101)/202*(right-left),test=map(.5),rows=m?[85,151,217]:[149,281,413];
 let s=text(left,m?24:51,'Training observations and learned cutoffs',m?12:18,'start','muted');
 if(fixed){s+=line(test,m?48:90,test,m?237:454,'reference');s+=text(test+(m?7:12),m?43:80,'Same x₀ = +0.5',m?12:18);}
 rowIndices().forEach((i,j)=>{const rule=workedRules[i],y=rows[j],c=rule.cutoff;
 s+=glyph(m?23:55,y-(m?39:74),m?.47:.9);s+=text(m?32:73,y+(m?17:31),`Alien ${i+1}`,m?11:17,'middle');
 s+=line(left,y,right,y);
 const placed=[],radius=m?3.5:6;
 rule.differences.forEach(x=>{
  const px=map(x);let rank=0;
  while(placed.some(point=>point.rank===rank&&Math.abs(point.x-px)<2*radius+1))rank++;
  placed.push({x:px,rank});s+=mark(px,y-rank*(m?9:15),x>0,radius);
 });
 if(Number.isFinite(c)){s+=line(map(c),y-(m?28:48),map(c),y+12,'model');s+=text(left,y-(m?30:55),`Cutoff ${signed(c)}`,m?12:18,'start','blue');}
 else {
  s+=text(left,y-(m?30:55),`Cutoff ${signed(c)} · always ${rule.constant?'win':'loss'}`,m?12:18,'start','blue');
  const edge=rule.constant?left:right,direction=rule.constant?-1:1;
  s+=line(edge+direction*3,y,edge+direction*(m?17:29),y,'model','marker-end="url(#arrow-blue)"');
 }
 if(fixed){const p=predict(rule,.5),bx=m?360:741;s+=line(right+5,y,bx-(m?14:23),y,'axis', 'marker-end="url(#arrow-ink)"');s+=mark(bx,y,p,m?13:22);s+=text(bx,y+(m?4:6),p,m?15:25,'middle','inverse');}
 });
 [-100,0,100].forEach(x=>s+=text(map(x),m?259:485,x,fs,'middle','muted'));
 s+=text((left+right)/2,m?285:529,'Knicks points − Spurs points',m?13:18,'middle','muted');
 return s;
}
function stackGeometry(d,compact=false,stats=workedStats){
 const m=d.mobile;return {left:m?80:160,right:m?320:640,baseline:m?(compact?162:188):(compact?295:326),columns:Math.max(20,Math.ceil(Math.max(stats.wins,stats.losses)/27)),gap:m?(compact?2.6:3.1):(compact?6.3:7.1)};
}
function populationScene(d,stats,mode){
 const {mobile:m,W,fs}=d;const compact=mode==='decomposition'||mode==='explore',g=stackGeometry(d,compact,stats),axis=g.baseline+(m?22:42),mean=g.left+(g.right-g.left)*stats.mean,truth=g.left+(g.right-g.left)*stats.truth;
 let s='';
 s+=glyph(g.left-(m?9:17),m?9:18,m?.45:.85)+glyph(g.right-(m?9:17),m?9:18,m?.45:.85);
 s+=text(g.left,m?58:95,`${stats.losses} predict 0`,m?14:20,'middle','rust');s+=text(g.right,m?58:95,`${stats.wins} predict 1`,m?14:20,'middle','blue');
 s+=line(g.left,axis,g.right,axis);
 [0,1].forEach(v=>{const x=g.left+(g.right-g.left)*v;s+=line(x,axis,x,axis+6);s+=text(x,axis+(m?22:31),v,m?17:23,'middle');});
 s+=line(truth,axis-15,truth,axis+8,'reference');
 if(mode!=='mse'){
  const edgeMean=stats.mean<.18||stats.mean>.82,top=edgeMean?(m?90:143):(m?70:112);s+=line(mean,top,mean,axis,'model');
  const meanAnchor=stats.mean<.15?'start':stats.mean>.85?'end':'middle';
  s+=text(mean,top-(m?6:12),`Mean ${fmt(stats.mean)}`,m?13:20,meanAnchor,'blue');
 }
 if(mode==='mean'){
  s+=text((g.left+g.right)/2,m?272:469,'Fitted prediction at the fixed input',m?13:18,'middle','muted');
  s+=text(g.right,axis+(m?41:57),'True outcome = 1',m?12:17,'end');
 }
 if(mode==='bias'){
  const y=axis-(m?12:23);s+=line(truth-3,y,mean+3,y,'measure','marker-end="url(#arrow-rust)"');
  s+=text((truth+mean)/2,axis+(m?46:72),`Bias ${fmt(stats.bias)}`,m?14:21,'middle','rust');
  s+=text(g.right,m?82:124,'True outcome: 1',m?12:17,'middle');
 }
 if(mode==='variance'){
  const y=axis-(m?10:19);s+=line(g.left,y,mean,y,'measure');s+=line(mean,y-9,g.right,y-9,'model');
  s+=text((g.left+mean)/2,axis+(m?45:67),`−${fmt(stats.mean)}`,m?13:20,'middle','rust');
  s+=text((g.right+mean)/2,axis+(m?45:67),`+${fmt(1-stats.mean)}`,m?13:20,'middle','blue');
  s+=text((g.left+g.right)/2,m?283:514,'Distances from the same mean',m?12:18,'middle','muted');
 }
 if(mode==='mse'){
  const other=truth===g.right?g.left:g.right,y=axis-(m?11:22);
  s+=line(other,y,truth,y,'measure','marker-end="url(#arrow-rust)"');
  s+=text((g.left+g.right)/2,axis+(m?47:72),'Distance to truth: 1',m?14:21,'middle','rust');
  s+=text(truth,m?83:125,`True outcome: ${stats.truth}`,m?12:17,'middle');
  s+=text(truth===g.right?g.left:g.right,m?283:516,`${stats.truth?stats.losses:stats.wins} errors of 1; ${stats.truth?stats.wins:stats.losses} errors of 0`,m?12:18,truth===g.right?'start':'end','muted');
 }
 if(compact){
  const x=m?43:95,width=m?314:610,y=m?258:459,height=m?14:23;
  const {segments,totalWidth}=decompositionSegments(stats,{x,width,max:1});
  s+=line(x,y+height+6,x+width,y+height+6);
  s+=`<rect x="${x}" y="${y}" width="${segments[0].width}" height="${height}" fill="var(--blue)"/><rect x="${segments[1].x}" y="${y}" width="${segments[1].width}" height="${height}" fill="var(--rust)"/>`;
  s+=text(x,y-(m?8:16),'Squared-error components',m?12:18);
  s+=text(x+width,y-(m?8:16),`MSE ${fmt(stats.mse)}`,m?12:18,'end');
  s+=text(x,y+height+(m?22:32),'0',m?11:16,'middle','muted');
  s+=text(x+width,y+height+(m?22:32),'1',m?11:16,'middle','muted');
  if(totalWidth>width*.05)s+=line(x+totalWidth,y-3,x+totalWidth,y+height+6,'axis');
  if(segments[0].width>(m?45:75))s+=text(segments[0].x+segments[0].width/2,y+height+(m?22:34),m?'Bias²':`Bias² ${fmt(stats.biasSquared)}`,m?12:16,'middle','blue');
  if(segments[1].width>(m?65:110))s+=text(segments[1].x+segments[1].width/2,y+height+(m?22:34),m?'Variance':`Variance ${fmt(stats.variance)}`,m?12:16,'middle','rust');
 }
 return s;
}
function setParticles(d,stats,crowd=false){
 const m=d.mobile;const positions=crowd?crowdLayout(1000,{left:m?35:120,top:m?70:105,columns:40,gap:m?8.4:14}):predictionLayout(stats.predictions,stackGeometry(d,scene>=10,stats));
 const r=m?1:2.65;
 particleNodes.forEach((node,i)=>{const p=positions[i];node.setAttribute('class',`particle ${stats.predictions[i]?'win':'loss'}`);node.setAttribute('width',r*2);node.setAttribute('height',r*2);node.setAttribute('x',-r);node.setAttribute('y',-r);node.setAttribute('rx',stats.predictions[i]?r:0);node.style.transform=`translate(${m?p.x*d.W/400:p.x}px, ${m?p.y*.8:p.y}px)`;});
 population.style.opacity='1';
}
function setMath(value){$('figure-math').innerHTML=value;}
function fitMobileArtwork(d){
 const sx=d.W/400;
 for(const node of art.querySelectorAll('*')){
  const square=node.tagName==='rect'&&node.getAttribute('width')===node.getAttribute('height');
  const oldX=Number(node.getAttribute('x')),oldY=Number(node.getAttribute('y')),width=Number(node.getAttribute('width')),height=Number(node.getAttribute('height'));
  for(const attr of ['x','x1','x2','cx'])if(node.hasAttribute(attr))node.setAttribute(attr,Number(node.getAttribute(attr))*sx);
  for(const attr of ['y','y1','y2','cy'])if(node.hasAttribute(attr))node.setAttribute(attr,Number(node.getAttribute(attr))*.8);
  if(square){node.setAttribute('x',(oldX+width/2)*sx-width/2);node.setAttribute('y',(oldY+height/2)*.8-height/2);}
  else if(node.tagName==='rect')node.setAttribute('width',width*sx);
  if(node.tagName==='text')node.setAttribute('font-size',Math.max(18,Number(node.getAttribute('font-size'))||18));
  if(node.tagName==='use')node.setAttribute('transform',node.getAttribute('transform').replace(/translate\(([-\d.]+) ([-\d.]+)\)/,(_,x,y)=>`translate(${Number(x)*sx} ${Number(y)*.8})`));
  if(node.tagName==='polygon')node.setAttribute('points',node.getAttribute('points').split(' ').map(p=>{const[x,y]=p.split(',');return`${Number(x)*sx},${Number(y)*.8}`;}).join(' '));
  if(node.tagName==='rect'&&node.getAttribute('fill')==='url(#error-hatch)')node.setAttribute('height',height*.8);
 }
}
function render(){
 if(!workedRules)return;const d=dimensions(),s=scene===11?explorerStats:workedStats;
 svg.setAttribute('viewBox',`0 0 ${d.W} ${d.H}`);
 $('stage-label').textContent=labels[scene];$('stage-count').textContent=`${scene+1} / ${steps.length}`;$('short-title').textContent=short[scene];$('caption').textContent=workedCaptions[scene];
 $('previous').disabled=scene===0;$('next').disabled=scene===steps.length-1;$('sample-control').hidden=!(scene===3||scene===4);
 annotations.innerHTML='';population.style.opacity='0';
 if(scene===0){art.innerHTML=tableScene(d);setMath('<span><i>x</i> = Knicks points − Spurs points</span>');svgDescription('Five games observed by one alien','The Knicks scores are 105, 105, 111, 107, and 94; Spurs scores are 95, 104, 115, 106, and 90. Differentials: plus 10, plus 1, minus 4, plus 1, plus 4.');}
 if(scene===1||scene===2){art.innerHTML=trainingScene(d,scene===2);setMath(scene===1?'<span>Cutoff = (−4 + 1) / 2 = −1.5</span>':'<span>At −1: prediction 1 ≠ true outcome 0</span>');svgDescription(scene===1?'A midpoint cutoff of minus one point five':'The fitted cutoff disagrees with the true cutoff',scene===1?'The closest observed loss is minus four and win is plus one. Place the boundary halfway between them.':'The true rule changes at zero. The fitted rule changes at minus one point five, and incorrectly calls a one-point Knicks loss a win.');}
 if(scene===3||scene===4){art.innerHTML=sampleScene(d,scene===4);const r=workedRules[selected];setMath(scene===3?`<span>Selected alien ${selected+1}: cutoff ${signed(r.cutoff)}</span>`:'<span>Same input +0.5 · true outcome 1</span>');svgDescription('Three aliens with different training samples',rowIndices().map(i=>{const rule=workedRules[i],cutoff=Number.isFinite(rule.cutoff)?String(rule.cutoff):rule.constant?'negative infinity, always predicts one':'positive infinity, always predicts zero';return `Alien ${i+1}: differentials ${rule.differences.join(', ')}; cutoff ${cutoff}${scene===4?'; predicts '+predict(rule,.5):''}.`;}).join(' '));}
 if(scene===5){art.innerHTML=glyph(d.mobile?182:380,d.mobile?5:12,d.mobile?.55:1)+text(d.mobile?200:d.W/2,d.mobile?58:93,'Same learner, 1,000 independent samples',d.mobile?13:19,'middle','muted');setParticles(d,s,true);setMath('<span>468 predict 0 &nbsp; + &nbsp; 532 predict 1</span>');svgDescription('One thousand alien predictions before aggregation','Exactly 1,000 marks represent independent training samples. Each predicts at plus zero point five: 468 zeros and 532 ones.');}
 if(scene>=6){const mode=['mean','bias','variance','mse','decomposition','explore'][scene-6];art.innerHTML=populationScene(d,s,mode);setParticles(d,s,false);
  const math={mean:'<span>(468 × 0 + 532 × 1) / 1,000 = 0.532</span>',bias:'<span>Bias ≈ 0.532 − 1 = −0.468</span>',variance:'<span class="compact">Variance ≈ 0.468(0.532)² + 0.532(0.468)² ≈ 0.249</span>',mse:'<span>MSE = (468 × 1 + 532 × 0) / 1,000 = 0.468</span>',decomposition:'<span>0.468 ≈ <span class="math-blue">0.219</span> + <span class="math-rust">0.249</span> + 0</span>',explore:`<span class="compact">MSE ${fmt(s.mse)} ≈ bias² ${fmt(s.biasSquared)} + variance ${fmt(s.variance)} + 0</span>`};
  if(d.mobile){math.mean='<span>Mean = 532 / 1,000 = 0.532</span>';math.variance='<span>Variance ≈ 0.249</span>';math.mse='<span>MSE = 468 / 1,000 = 0.468</span>';math.explore=`<span>${fmt(s.mse)} ≈ ${fmt(s.biasSquared)} + ${fmt(s.variance)} + 0</span>`;}
  setMath(math[mode]);svgDescription(`${mode}: the same population of fitted predictions`,`${s.losses} predictions of zero; ${s.wins} predictions of one. Mean ${fmt(s.mean)}, true outcome ${s.truth}, bias ${fmt(s.bias)}, squared bias ${fmt(s.biasSquared,6)}, variance ${fmt(s.variance,6)}, MSE ${fmt(s.mse)}. Each mark is one alien. The decomposition bar, when shown, uses a zero-to-one squared-error scale.`);
  if(scene===11){$('stage-label').textContent=`${$('games').value} games per alien · x₀ = ${signed(Number($('test-point').value))} · truth ${s.truth}`;$('caption').textContent=`1,000 aliens: ${s.losses} predict 0; ${s.wins} predict 1. Mean ${fmt(s.mean)}; bias ${fmt(s.bias)}.`;}
 }
 if(d.mobile)fitMobileArtwork(d);
}
function setScene(i){i=Math.max(0,Math.min(steps.length-1,i));if(i===scene)return;scene=i;render();}
function topOffset(){return matchMedia('(max-width:760px)').matches?document.querySelector('.stage-shell').offsetHeight+50:56;}
function navigate(i){i=Math.max(0,Math.min(steps.length-1,i));setScene(i);window.scrollTo({top:steps[i].getBoundingClientRect().top+scrollY-topOffset(),behavior:'instant'});}
document.querySelector('.start').addEventListener('click',event=>{event.preventDefault();navigate(0);});
$('previous').addEventListener('click',()=>navigate(scene-1));$('next').addEventListener('click',()=>navigate(scene+1));
$('sample-alien').addEventListener('change',()=>{selected=Number($('sample-alien').value);render();});
let scheduled=false;
function track(){scheduled=false;if(!workedRules)return;const m=matchMedia('(max-width:760px)').matches,line=m?document.querySelector('.stage-shell').getBoundingClientRect().bottom+90:innerHeight*.5;let i=0;for(let j=0;j<steps.length;j++)if(steps[j].getBoundingClientRect().top<=line)i=j;setScene(i);}
addEventListener('scroll',()=>{if(!scheduled){scheduled=true;requestAnimationFrame(track);}},{passive:true});
addEventListener('resize',()=>{if(workedRules){render();track();}});
function updateExperiment(newSamples=false){
 if(!workedRules)return;const n=Number($('games').value),x=Number($('test-point').value);if(newSamples)seed++;
 if(newSamples||explorerRules[0].games.length!==n)explorerRules=simulate(1000,n,seed);
 explorerStats=summarize(explorerRules,x);
 $('explorer-status').textContent=`1,000 independently trained aliens; ${n} games each; fixed input ${signed(x)}. Mean prediction ${fmt(explorerStats.mean)}, bias ${fmt(explorerStats.bias)}, MSE ${fmt(explorerStats.mse)}.`;
 if(scene!==11)navigate(11);else render();
}
$('games').addEventListener('change',()=>updateExperiment());$('test-point').addEventListener('change',()=>updateExperiment());$('resample').addEventListener('click',()=>updateExperiment(true));
$('reset').addEventListener('click',()=>{if(!workedRules)return;$('games').value='5';$('test-point').value='0.5';explorerRules=workedRules;explorerStats=workedStats;$('explorer-status').textContent='Restored the worked example: 468 loss predictions and 532 win predictions.';render();});
try{
 const response=await fetch('./samples.json');if(!response.ok)throw new Error('Training samples did not load.');const fixture=await response.json();
 workedRules=fixture.games.map(games=>({games,...fitRule(games)}));allWinsIndex=workedRules.findIndex(rule=>rule.constant===1);workedStats=summarize(workedRules);explorerRules=workedRules;explorerStats=workedStats;
 const ns='http://www.w3.org/2000/svg';particleNodes=workedRules.map((rule,i)=>{const node=document.createElementNS(ns,'rect');node.dataset.alien=i;population.append(node);return node;});
 $('sample-alien').innerHTML=Array.from({length:1000},(_,i)=>`<option value="${i}">Alien ${i+1}</option>`).join('');
 setScene(0);track();
}catch(error){$('caption').textContent='The figures could not load. The complete worked explanation remains readable; reload to retry.';console.error(error);}
