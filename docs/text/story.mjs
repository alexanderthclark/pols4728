import {chapters} from './lesson.mjs';
import {updateAiExample} from './ai-example.mjs';
import {bagVisual,tfVisual,wordVisual,inspectTerm} from './visuals.mjs';
import {prose} from './math.mjs';
import {documents,vocabulary} from './model.mjs';

const clamp=(x,a=0,b=1)=>Math.max(a,Math.min(b,x));
const media=matchMedia('(prefers-reduced-motion: reduce)');
const state={motion:!media.matches&&new URLSearchParams(location.search).get('motion')!=='reduce'};
const html=(el,value)=>{if(el.innerHTML!==value)el.innerHTML=value;};
const root=document.querySelector('#chapters');
root.innerHTML=chapters.map((c,i)=>`<section class="track" id="${c.id}" data-chapter="${i}" aria-label="${c.title}"><div class="scene"><div class="scene-heading"><h2><span class="chapter-number">${i===0?'Opening':String(i).padStart(2,'0')}</span>${c.title}</h2><span class="chapter-subtitle">${c.subtitle}</span></div><div class="scene-body"><div class="scene-copy"><h3></h3><p class="explanation"></p><p class="small qualification"></p></div><div class="visual"></div></div><div class="scene-bottom"><span class="counter" aria-live="polite"></span><button type="button" class="previous" aria-label="Previous stage">← Previous</button><button type="button" class="next" aria-label="Next stage">Next →</button><button type="button" class="motion" aria-pressed="${state.motion}">${state.motion?'Motion on':'Motion off'}</button></div><div class="progress" aria-hidden="true"><span></span></div></div></section>`).join('');
const views=[...root.querySelectorAll('.track')].map((track,i)=>({track,chapter:chapters[i],visual:track.querySelector('.visual'),copy:track.querySelector('.scene-copy'),counter:track.querySelector('.counter'),bar:track.querySelector('.progress span'),stage:-1,p:0,segment:0,lastVisual:''}));
let queued=false,active=0;
const header=()=>document.querySelector('.topbar').getBoundingClientRect().height;
function layout(){
  const height=innerHeight-header();
  document.documentElement.style.setProperty('--scene',`${height}px`);
  views.forEach(v=>{v.segment=height*1.05;v.track.style.height=`${height+(v.chapter.stages.length-1)*v.segment}px`;});
  render();
}
function render(){
  queued=false;
  for(const [i,v] of views.entries()){
    const position=clamp((scrollY-v.track.offsetTop+header())/v.segment,0,v.chapter.stages.length-1);
    const raw=Math.abs(position-Math.round(position))<.002?Math.round(position):position;
    const stage=Math.round(raw),p=state.motion?raw:stage,s=v.chapter.stages[stage];v.p=raw;
    const bounds=v.track.getBoundingClientRect();
    if(bounds.top<=innerHeight*.5&&bounds.bottom>innerHeight*.5)active=i;
    // Only visible scenes need animation work; entering one always refreshes it.
    if(bounds.bottom<0||bounds.top>innerHeight)continue;
    if(v.stage!==stage){
      v.copy.querySelector('h3').textContent=s.title;
      html(v.copy.querySelector('.explanation'),prose(s.text,s.links));
      html(v.copy.querySelector('.qualification'),prose(s.note));
      v.counter.textContent=`${stage+1} / ${v.chapter.stages.length}`;v.stage=stage;
      v.track.dataset.stage=stage;
    }
    const fade=state.motion?clamp((Math.abs(raw-stage)-.41)/.09):0;
    v.copy.style.opacity=1-fade;
    v.copy.style.transform=`translateY(${Math.sign(stage-raw)*fade*10}px)`;
    if(v.chapter.id==='ai-writing')updateAiExample(v.visual,stage,p,{reducedMotion:!state.motion});
    else{
      const draw=v.chapter.id==='bag-of-words'?bagVisual:v.chapter.id==='tf-idf'?tfVisual:wordVisual;
      html(v.visual,draw(stage,p,state.motion));
    }
    v.bar.style.width=`${raw/(v.chapter.stages.length-1)*100}%`;
    v.track.dataset.progress=raw.toFixed(4);
  }
  document.querySelectorAll('.topbar nav a').forEach(a=>{
    if(a.hash===`#${views[active].chapter.id}`)a.setAttribute('aria-current','step');else a.removeAttribute('aria-current');
  });
}
function schedule(){if(!queued){queued=true;requestAnimationFrame(render);}}
function jump(i,stage){
  const from=i;
  if(stage<0){if(i===0){document.querySelector('#title').scrollIntoView();return;}i--;stage=views[i].chapter.stages.length-1;}
  if(stage>=views[i].chapter.stages.length){if(i===views.length-1){document.querySelector('#compare').scrollIntoView();return;}i++;stage=0;}
  const v=views[i];scrollTo({top:v.track.offsetTop-header()+stage*v.segment,behavior:'instant'});
  if(i!==from)v.track.querySelector(stage===0?'.next':'.previous').focus({preventScroll:true});
  schedule();
}
function motion(value){state.motion=value;root.querySelectorAll('.motion').forEach(b=>{b.textContent=value?'Motion on':'Motion off';b.setAttribute('aria-pressed',String(value));});render();}
root.addEventListener('click',e=>{
  const track=e.target.closest('.track');if(!track)return;
  const i=Number(track.dataset.chapter),v=views[i];
  if(e.target.closest('.next'))jump(i,Math.round(v.p)+1);
  if(e.target.closest('.previous'))jump(i,Math.round(v.p)-1);
  if(e.target.closest('.motion'))motion(!state.motion);
});
document.addEventListener('keydown',e=>{
  if(e.altKey||e.ctrlKey||e.metaKey||e.target.closest('input,select,textarea,button,a,summary'))return;
  if(e.key==='ArrowRight'||e.key==='ArrowLeft'){
    const r=views[active].track.getBoundingClientRect();
    if(r.top<=header()+4&&r.bottom>header()){e.preventDefault();jump(active,Math.round(views[active].p)+(e.key==='ArrowRight'?1:-1));}
  }
});
const explorer=document.querySelector('#term-explorer');
explorer.innerHTML=`<div class="term-controls"><label for="term">Term</label><select id="term">${vocabulary.map(t=>`<option ${t==='budget'?'selected':''}>${t}</option>`).join('')}</select><label for="document">Text</label><select id="document">${documents.map((_,i)=>`<option value="${i}">D${i+1}</option>`).join('')}</select></div><div class="explorer-result" aria-live="polite">${inspectTerm()}</div>`;
explorer.addEventListener('change',()=>html(explorer.querySelector('.explorer-result'),inspectTerm(explorer.querySelector('#term').value,Number(explorer.querySelector('#document').value))));
addEventListener('scroll',schedule,{passive:true});addEventListener('resize',layout);
media.addEventListener('change',e=>motion(!e.matches));
layout();document.fonts.ready.then(schedule);
if(location.hash){const target=document.getElementById(location.hash.slice(1));if(target)requestAnimationFrame(()=>target.scrollIntoView());}
