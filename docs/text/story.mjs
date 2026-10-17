import {chapters} from './lesson.mjs';
import {speechCartoon} from './speech-cartoon.mjs';
import {documents,tokenize,vocabulary,counts,tfidf,termStats,cosine,cbowExample,cbowStep} from './model.mjs';

const clamp=(x,a=0,b=1)=>Math.max(a,Math.min(b,x));
const mix=(a,b,t)=>a+(b-a)*t;
const fmt=(n,d=2)=>Math.abs(n)<1e-9?'0':n.toFixed(d);
const vec=a=>'[ '+a.map(n=>fmt(n)).join(', ')+' ]';
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const svg=(body,label)=>`<svg viewBox="0 0 680 390" role="img" aria-label="${esc(label)}">${body}</svg>`;
const txt=(x,y,t,extra='')=>`<text x="${x}" y="${y}" font-size="26" ${extra}>${esc(t)}</text>`;
const line=(x1,y1,x2,y2,color='#767676',extra='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="2" ${extra}/>`;
const html=(el,s)=>{if(el.innerHTML!==s)el.innerHTML=s;};
const state={term:'budget',doc:0,motion:!matchMedia('(prefers-reduced-motion: reduce)').matches};
const gradient=cbowStep();

function matrix(values,{focus=null,doc=null,decimals=0,weighted=false,caption=null}={}){
  return `<table class="matrix"><caption class="visual-title">${caption??(weighted?'TF–IDF weights':'Document–term counts')}</caption><thead><tr><th scope="col">Text</th>${vocabulary.map(t=>`<th scope="col">${t}</th>`).join('')}</tr></thead><tbody>${values.map((row,r)=>`<tr ${r===doc?'class="active"':''}><th scope="row">D${r+1}</th>${row.map((v,c)=>`<td class="${vocabulary[c]===focus?'focus ':''}${v===0?'zero':''}">${decimals?fmt(v,decimals):v}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}
function tokenDiagram(p){
  const tokens=tokenize(documents[0].text),q=clamp(p-1),seen={};
  const positions=[[40,95],[155,95],[365,95],[40,160],[155,160],[40,225],[155,225],[365,225]];
  const present=[...new Set(tokens)].sort();
  const marks=tokens.map((word,i)=>{const row=present.indexOf(word);const dup=seen[word]??0;seen[word]=dup+1;const x=mix(positions[i][0],50+dup*190,q),y=mix(positions[i][1],90+row*48,q);return txt(x,y,word,`fill="${word==='budget'?'#234e70':'#242424'}" data-token="${i}"`);}).join('');
  return svg(txt(20,35,q<.5?'D1 · lowercase tokens':'D1 · gather identical terms')+marks+txt(20,345,q<.5?'8 tokens · punctuation removed':'The original order is being discarded.'),'Tokens from D1 move into vocabulary columns as you scroll');
}
function bagVisual(stage,p){
  if(stage<2)return speechCartoon(state.motion?clamp(p):stage,{reducedMotion:!state.motion,id:'lesson-speech'});
  stage-=2;p-=2;
  if(stage===0)return `<div class="corpus">${documents.map((d,i)=>`<div class="doc"><span>D${i+1}</span><span>${d.text}</span></div>`).join('')}</div>`;
  if(stage===1||(p<1.94&&stage===2))return tokenDiagram(p);
  if(stage===2)return matrix(counts);
  if(stage===3)return `<div class="flow"><div class="flow-box">Senators support voters</div><span aria-hidden="true">↓</span><div class="vector">senators: 1 · support: 1 · voters: 1</div><span aria-hidden="true">↑</span><div class="flow-box">Voters support senators</div></div><p class="visual-note">Different roles. Identical count vectors.</p>`;
  if(stage===4){const t=1;return `<h4 class="visual-title">“Upon” per 1,000 words</h4>`+svg(
    txt(25,92,'Hamilton')+txt(25,218,'Madison')+
    line(210,275,630,275)+[0,1,2,3].map(v=>line(210+v*140,270,210+v*140,281)+txt(210+v*140,319,String(v),'text-anchor="middle"')).join('')+
    `<rect x="210" y="60" height="44" width="${420*t}" fill="#234e70"/><rect x="210" y="186" height="44" width="${140/6*t}" fill="#a54f32"/>`+
    txt(210+420*t,43,'≈ 3.0','text-anchor="end"')+txt(230+140/6*t,216,'≈ 0.17'), 'Reported approximate rates of upon: Hamilton 3 per thousand, Madison 0.17 per thousand')+`<p class="visual-note">Mosteller &amp; Wallace (1963), comparison writings.<br>Rates are normalized for text length.</p>`;}
  if(stage===5)return `<div class="study-pipeline"><div class="study-labels"><span>Revelation</span><span>Gospel of John</span></div><span aria-hidden="true">↓</span><div class="flow-box">500-word Greek samples<br>75 common-word counts</div><span aria-hidden="true">↓</span><div>Compare frequency profiles</div></div><p class="visual-note">Schematic of Erwin &amp; Oakes (2012).<br>No new authorship estimate is shown.</p>`;
  return `<div class="flow"><div class="flow-row"><div class="flow-box">Authorship<br><span class="small">writing habits</span></div><div class="flow-box">Topic<br><span class="small">distinctive terms</span></div></div><div class="math">Same words → different features</div><div class="flow-box">What should count as evidence?</div></div>`;
}

function termControls(){return `<div class="term-controls"><label for="term">Term</label><select id="term">${vocabulary.map(t=>`<option ${t===state.term?'selected':''}>${t}</option>`).join('')}</select><label for="document">Text</label><select id="document">${documents.map((_,i)=>`<option value="${i}" ${i===state.doc?'selected':''}>D${i+1}</option>`).join('')}</select></div>`;}
function tfVisual(stage,p){
  const s=termStats(state.term,state.doc),base=`<div data-term-controls>${termControls()}</div>`;
  let content='';
  if(stage<2){content=matrix(counts,{focus:state.term,doc:state.doc})+`<div class="metric-line"><span>tf = ${s.tf}</span>${stage===1?`<span>df = ${s.df} of ${s.N}</span>`:''}</div>`;}
  else if(stage===2){content=`<h4 class="visual-title">${esc(state.term)} · D${state.doc+1}</h4><div class="labelled-equation"><div class="math">idf = ln(${s.N} / ${s.df}) = ${fmt(s.idf,3)}</div><div class="metric-line">${s.df} of ${s.N} documents contain this term</div></div><div class="math">tf = ${s.tf} &nbsp; → &nbsp; weight = ${fmt(s.weight,3)}</div>`;}
  else if(stage===3){const q=state.motion?clamp((p-2.5)/.5):1;const vals=counts.map((r,i)=>r.map((v,j)=>mix(v,tfidf[i][j],q)));const scale=mix(1,s.idf,q);content=matrix(vals,{focus:state.term,doc:state.doc,decimals:2,weighted:true,caption:q<1?`Reweighting counts · ${Math.round(q*100)}%`:'TF–IDF weights'})+`<div class="math">${s.tf} × ${fmt(scale,3)} = ${fmt(s.tf*scale,3)}</div><p class="visual-note">${q<1?'The current scale interpolates from 1 to IDF.':'The completed matrix uses raw-count TF × IDF.'}</p>`;}
  else{content=`<h4 class="visual-title">TF–IDF cosine similarity</h4><div class="bars">${[1,2,3].map(i=>{const v=cosine(tfidf[0],tfidf[i]);return `<div class="bar-row"><span>D1 with D${i+1}</span><div class="bar-track"><div class="bar-fill" style="width:${v*100}%"></div></div><span class="bar-value">${fmt(v,3)}</span></div>`;}).join('')}</div><div class="math">cos θ = (x · y) / (‖x‖ ‖y‖)</div><p class="visual-note">0 means no weighted terms overlap; 1 means the same direction.<br>Cosine adjusts length, but does not learn synonyms.</p>`;}
  return base+`<div data-term-result>${content}</div>`;
}

function sentence(){return `<div class="token-line">${cbowExample.sentenceTokens.map((w,i)=>`<span class="token ${i===1?'target':i===0||i===2?'context':''}">${i===1?'[ '+w+' ]':w}</span>`).join('')}</div>`;}
function wordVisual(stage,p){
  if(stage===0)return svg(line(140,280,570,280)+line(140,280,140,40)+`<path d="M 140 280 L 380 280" stroke="#234e70" stroke-width="5"/><path d="M 140 280 L 140 40" stroke="#a54f32" stroke-width="5"/>`+txt(380,324,'senate','text-anchor="middle"')+txt(165,55,'parliament')+txt(330,190,'[1, 0] · [0, 1] = 0','text-anchor="middle"')+txt(340,375,'Vocabulary axes do not encode related usage.','text-anchor="middle"'),'Two orthogonal vocabulary axes have dot product zero');
  if(stage===1)return sentence()+`<div class="math">P(senate | the, debated)</div><p class="visual-note">Solid outline: context · dashed outline: target<br>Radius 1 uses one word on each side.</p>`;
  if(stage===2){const q=state.motion?clamp((p-1.5)/.5):1;return svg(
    txt(90,55,'the','text-anchor="middle"')+txt(500,55,'debated','text-anchor="middle"')+
    txt(90,105,vec(cbowExample.inputVectors.the),'text-anchor="middle"')+txt(500,105,vec(cbowExample.inputVectors.debated),'text-anchor="middle"')+
    line(90,130,mix(90,310,q),mix(130,225,q),'#234e70')+line(500,130,mix(500,310,q),mix(130,225,q),'#234e70')+
    txt(310,270,'average = [ 0.50, 0.50 ]','text-anchor="middle"')+txt(310,345,'h = (vthe + vdebated) / 2','text-anchor="middle"'), 'The context input vectors one zero and zero one are averaged to one half one half');}
  if(stage===3)return `<h4 class="visual-title">P(word | the, debated)</h4><div class="bars">${cbowExample.words.map(w=>`<div class="bar-row ${w==='senate'?'target':''}"><span>${w}${w==='senate'?' ←':''}</span><div class="bar-track"><div class="bar-fill" style="width:${cbowExample.probabilities[w]*100}%"></div></div><span class="bar-value">${fmt(cbowExample.probabilities[w]*100,1)}%</span></div>`).join('')}</div><div class="math">P(w | C) = exp(u<sub>w</sub> · h) / ∑<sub>j</sub> exp(u<sub>j</sub> · h)</div><p class="visual-note">Target senate: score ${fmt(cbowExample.scores.senate,2)} · probability 21.0% · loss 1.559</p>`;
  if(stage===4){const q=state.motion?clamp((p-3.5)/.5):1;const before=gradient.before,after=cbowStep(0.5*q).after;const prob=after.probabilities.senate,loss=after.loss;return `<h4 class="visual-title">One gradient step · progress ${Math.round(q*100)}%</h4><div class="math">θ<sub>new</sub> = θ − 0.5 ∇L(θ)</div><div class="bars"><div class="bar-row"><span>Before</span><div class="bar-track"><div class="bar-fill" style="width:${before.probabilities.senate*100}%"></div></div><span class="bar-value">21.0%</span></div><div class="bar-row target"><span>Updated</span><div class="bar-track"><div class="bar-fill" style="width:${prob*100}%"></div></div><span class="bar-value">${fmt(prob*100,1)}%</span></div></div><div class="metric-line">Loss: ${fmt(loss,3)}</div><p class="visual-note">h: ${vec(after.averageVector)}<br>Probabilities are recomputed along the gradient step.<br>The completed state is one calculated gradient update.</p>`;}
  return `<div class="flow"><div class="flow-row"><div class="flow-box">the [senate] debated</div></div><div class="flow-row"><div class="flow-box">the [parliament] debated</div></div><div class="math">Shared context → related training signals</div><p class="visual-note">A schematic mechanism, not an empirical embedding plot.<br>Similarity reflects corpus usage, including its biases.</p></div>`;
}

const root=document.querySelector('#chapters');
root.innerHTML=chapters.map((c,i)=>`<section class="track" id="${c.id}" data-chapter="${i}" aria-label="${c.title}"><div class="scene"><div class="scene-heading"><h2>${String(i+1).padStart(2,'0')} / ${c.title}</h2><span>${c.subtitle}</span></div><div class="scene-body"><div class="scene-copy"><h3></h3><p class="explanation"></p><p class="small qualification"></p></div><div class="visual"></div></div><div class="scene-bottom"><span class="counter" aria-live="polite"></span><button type="button" class="previous" aria-label="Previous stage">← Previous</button><button type="button" class="next" aria-label="Next stage">Next →</button><button type="button" class="motion" aria-pressed="${state.motion}">${state.motion?'Motion on':'Motion off'}</button></div><div class="progress" aria-hidden="true"><span></span></div></div></section>`).join('');

const views=[...document.querySelectorAll('.track')].map((track,i)=>({track,chapter:chapters[i],visual:track.querySelector('.visual'),copy:track.querySelector('.scene-copy'),counter:track.querySelector('.counter'),bar:track.querySelector('.progress span'),stage:-1,p:0,segment:0}));
let queued=false,active=0;
function layout(){const h=innerHeight-56;document.documentElement.style.setProperty('--scene',`${h}px`);views.forEach(v=>{v.segment=h*1.1;v.track.style.height=`${h+(v.chapter.stages.length-1)*v.segment}px`;});render();}
function updateVisual(v,stage,p){
  // Do not replace focused form controls during a scroll frame.
  if(v.chapter.id==='tf-idf'&&v.visual.querySelector('[data-term-controls]')){
    const temp=document.createElement('div');temp.innerHTML=tfVisual(stage,p);
    html(v.visual.querySelector('[data-term-result]'),temp.querySelector('[data-term-result]').innerHTML);
  }else html(v.visual,v.chapter.id==='bag-of-words'?bagVisual(stage,p):v.chapter.id==='tf-idf'?tfVisual(stage,p):wordVisual(stage,p));
}
function render(){queued=false;const y=scrollY;
  for(const [i,v] of views.entries()){
    const rawPosition=clamp((y-v.track.offsetTop+56)/v.segment,0,v.chapter.stages.length-1);
    // Scroll positions may round to whole pixels; stage links still reach exact endpoints.
    const raw=Math.abs(rawPosition-Math.round(rawPosition))<.002?Math.round(rawPosition):rawPosition;
    const p=state.motion?raw:Math.round(raw),stage=Math.round(raw),s=v.chapter.stages[stage];v.p=raw;
    const r=v.track.getBoundingClientRect();if(r.top<=innerHeight*.5&&r.bottom>innerHeight*.5)active=i;
    if(v.stage!==stage){v.copy.querySelector('h3').textContent=s.title;v.copy.querySelector('.explanation').textContent=s.text;v.copy.querySelector('.qualification').textContent=s.note;v.counter.textContent=`${stage+1} / ${v.chapter.stages.length}`;v.stage=stage;}
    // Render the active chapter and completed boundaries; no time-driven animation.
    updateVisual(v,stage,p);
    v.bar.style.width=`${raw/(v.chapter.stages.length-1)*100}%`;
    v.track.dataset.stage=stage;v.track.dataset.progress=raw.toFixed(4);
  }
}
function schedule(){if(!queued){queued=true;requestAnimationFrame(render);}}
function jump(i,stage){
  if(stage<0){if(i===0){document.querySelector('#title').scrollIntoView();return;}i--;stage=views[i].chapter.stages.length-1;}
  if(stage>=views[i].chapter.stages.length){if(i===views.length-1){document.querySelector('#compare').scrollIntoView();return;}i++;stage=0;}
  const v=views[i];scrollTo({top:v.track.offsetTop-56+stage*v.segment,behavior:'instant'});schedule();
}
root.addEventListener('click',e=>{const track=e.target.closest('.track');if(!track)return;const i=Number(track.dataset.chapter),v=views[i];if(e.target.closest('.next'))jump(i,Math.round(v.p)+1);if(e.target.closest('.previous'))jump(i,Math.round(v.p)-1);if(e.target.closest('.motion')){state.motion=!state.motion;document.querySelectorAll('.motion').forEach(b=>{b.textContent=state.motion?'Motion on':'Motion off';b.setAttribute('aria-pressed',String(state.motion));});render();}});
root.addEventListener('change',e=>{if(e.target.id==='term')state.term=e.target.value;if(e.target.id==='document')state.doc=Number(e.target.value);const v=views[1];updateVisual(v,Math.round(v.p),state.motion?v.p:Math.round(v.p));});
document.addEventListener('keydown',e=>{if(e.altKey||e.ctrlKey||e.metaKey||e.target.closest('input,select,textarea,button,a,summary'))return;if(e.key==='ArrowRight'||e.key==='ArrowLeft'){const r=views[active].track.getBoundingClientRect();if(r.top<=60&&r.bottom>56){e.preventDefault();jump(active,Math.round(views[active].p)+(e.key==='ArrowRight'?1:-1));}}});
addEventListener('scroll',schedule,{passive:true});addEventListener('resize',layout);
matchMedia('(prefers-reduced-motion: reduce)').addEventListener('change',e=>{state.motion=!e.matches;document.querySelectorAll('.motion').forEach(b=>{b.textContent=state.motion?'Motion on':'Motion off';b.setAttribute('aria-pressed',String(state.motion));});render();});
layout();document.fonts.ready.then(schedule);
if(location.hash&&document.querySelector(location.hash))requestAnimationFrame(()=>document.querySelector(location.hash).scrollIntoView());
