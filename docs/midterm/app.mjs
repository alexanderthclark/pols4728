import {questions as first} from './modules/q19-q20.mjs';
import {questions as second} from './modules/q21-q22.mjs';
import {questions as third} from './modules/q23-q25.mjs';
import {renderMath,esc} from './shared.mjs';
const questions=[...first,...second,...third];
if('scrollRestoration' in history)history.scrollRestoration='manual';
const reducedMotion=matchMedia('(prefers-reduced-motion: reduce)');
const chapters=document.querySelector('#chapters');
document.querySelector('#quick-nav').innerHTML=questions.map(q=>`<a href="#q${q.id}" aria-label="Question ${q.id}: ${esc(q.title)}">${q.id}</a>`).join('');
document.querySelector('#chapter-list').innerHTML=questions.map(q=>`<li><a href="#q${q.id}"><span class="number">${q.id}</span><span>${q.title}</span><span class="arrow" aria-hidden="true">↗</span></a></li>`).join('');
chapters.innerHTML=questions.map((q,qi)=>`<section class="chapter" id="q${q.id}" aria-labelledby="title-${q.id}">
  <div class="chapter-opening"><div class="question-number" aria-hidden="true">${q.id}</div><div><p class="eyebrow">Question ${q.id} · ${q.kicker}</p><h2 id="title-${q.id}">${q.title}</h2><p class="dek">${q.dek}</p></div></div>
  <details class="original"><summary>Read the original question</summary><div class="original-content">${q.prompt}<ol>${q.parts.map(p=>`<li>${p}</li>`).join('')}</ol></div></details>
  ${q.sections?`<nav class="chapter-parts" aria-label="Parts of question ${q.id}">${q.sections.map(p=>`<a href="#q${q.id}-step-${p.step+1}">${p.label}</a>`).join('')}</nav>`:''}
  <div class="scrolly"><aside class="visual-stage" aria-label="Visual explanation for question ${q.id}"><div class="stage-meta"><span>${q.kicker}</span><span class="stage-index">01 / ${String(q.steps.length).padStart(2,'0')}</span></div><div class="visual"></div><p class="stage-caption"></p><div class="stage-progress" aria-hidden="true">${q.steps.map(()=>'<span></span>').join('')}</div></aside><div class="steps">${q.steps.map((s,i)=>`<article id="q${q.id}-step-${i+1}" class="step ${i===0?'active':''}" data-step="${i}"><span class="step-count">${String(i+1).padStart(2,'0')} / ${String(q.steps.length).padStart(2,'0')}</span><h3>${s.title}</h3><div class="step-body">${s.body}</div><div class="reading-figure" aria-label="Visual for ${esc(s.title)}"></div></article>`).join('')}</div></div>
  <div class="takeaway"><p class="eyebrow">What to carry forward</p><div class="takeaway-body">${q.takeaway}</div>${qi<questions.length-1?`<a class="next" href="#q${questions[qi+1].id}">Continue to question ${questions[qi+1].id} <span aria-hidden="true">→</span></a>`:'<a class="next" href="#top">Return to the question index ↑</a>'}</div>
</section>`).join('');
renderMath(document);
const scenes=questions.map(q=>{
  const section=document.querySelector(`#q${q.id}`);
  return {q,section,scrolly:section.querySelector('.scrolly'),stage:section.querySelector('.visual-stage'),visual:section.querySelector('.visual'),steps:[...section.querySelectorAll('.step')],active:-1};
});
function setStage(scene,index){
  if(scene.active===index)return;
  scene.active=index;
  scene.stage.dataset.last=String(index===scene.q.steps.length-1);
  scene.q.render(index,scene.visual,0);
  renderMath(scene.visual);
  scene.steps.forEach((el,i)=>el.classList.toggle('active',i===index));
  scene.stage.querySelector('.stage-index').textContent=`${String(index+1).padStart(2,'0')} / ${String(scene.q.steps.length).padStart(2,'0')}`;
  scene.stage.querySelector('.stage-caption').textContent=scene.q.steps[index].caption;
  scene.stage.querySelectorAll('.stage-progress span').forEach((el,i)=>el.classList.toggle('on',i<=index));
}
scenes.forEach(s=>setStage(s,0));
let ticking=false;
let currentNav=null;
function stepAnchor(scene){
  return innerWidth<=760?Math.min(innerHeight*.78,58+scene.stage.getBoundingClientRect().height+125):innerHeight*.52;
}
function update(){
  ticking=false;
  let latest=null;
  for(const scene of scenes){
    const sectionBox=scene.section.getBoundingClientRect();
    if(sectionBox.top<=innerHeight*.45&&sectionBox.bottom>100)latest=scene.q.id;
    if(document.body.classList.contains('reading'))continue;
    const box=scene.scrolly.getBoundingClientRect();
    if(box.bottom<0||box.top>innerHeight)continue;
    const anchor=stepAnchor(scene);
    let index=0;
    scene.steps.forEach((el,i)=>{if(el.getBoundingClientRect().top<=anchor)index=i;});
    setStage(scene,index);
    if(scene.q.update&&!document.body.classList.contains('reading')){
      const stepBox=scene.steps[index].getBoundingClientRect();
      const progress=Math.min(1,Math.max(0,(anchor-stepBox.top)/Math.max(1,stepBox.height*.72)));
      scene.q.update(reducedMotion.matches?1:progress,scene.visual,index);
    }
  }
  if(currentNav!==latest){
    currentNav=latest;
    document.querySelectorAll('#quick-nav a').forEach(el=>{if(el.getAttribute('href')===`#q${latest}`)el.setAttribute('aria-current','location');else el.removeAttribute('aria-current');});
  }

}
function schedule(){if(!ticking){ticking=true;requestAnimationFrame(update);}}
function jumpToStep(hash){
  const match=/^#q(19|2[0-5])-step-(\d+)$/.exec(hash);
  if(!match)return false;
  const scene=scenes.find(s=>s.q.id===Number(match[1])),index=Number(match[2])-1;
  if(!scene?.steps[index])return false;
  if(document.body.classList.contains('reading')){
    scene.steps[index].scrollIntoView({behavior:'instant',block:'start'});
  }else{
    setStage(scene,index);
    scene.q.update?.(reducedMotion.matches?1:0,scene.visual,index);
    const top=scrollY+scene.steps[index].getBoundingClientRect().top-stepAnchor(scene)+16;
    scrollTo({top,behavior:'instant'});
  }
  schedule();
  return true;
}
document.addEventListener('click',event=>{
  if(event.defaultPrevented||event.button!==0||event.metaKey||event.ctrlKey||event.shiftKey||event.altKey)return;
  const link=event.target.closest?.('a[href]');
  const hash=link?.getAttribute('href');
  if(!/^#q(19|2[0-5])-step-\d+$/.test(hash||''))return;
  event.preventDefault();
  history.pushState(null,'',hash);
  jumpToStep(hash);
});
addEventListener('hashchange',()=>jumpToStep(location.hash));
addEventListener('scroll',schedule,{passive:true});addEventListener('resize',schedule);document.querySelectorAll('.original').forEach(el=>el.addEventListener('toggle',schedule));
let readingRendered=false;
function buildReading(){
  if(readingRendered)return;
  scenes.forEach(scene=>scene.steps.forEach((step,i)=>{const el=step.querySelector('.reading-figure');scene.q.render(i,el,1);renderMath(el);if(scene.q.update)scene.q.update(1,el,i);}));
  readingRendered=true;
}
document.querySelector('#reading-toggle').addEventListener('click',()=>{
  const oldId=currentNav;
  const reading=!document.body.classList.contains('reading');
  if(reading){
    scenes.forEach(scene=>{scene.visual.innerHTML='';scene.active=-1;});
    buildReading();
  }else{
    document.querySelectorAll('.reading-figure').forEach(el=>el.innerHTML='');
    readingRendered=false;
  }
  document.body.classList.toggle('reading',reading);
  const button=document.querySelector('#reading-toggle');button.setAttribute('aria-pressed',String(reading));button.textContent=reading?'Scroll view':'Reading view';button.setAttribute('aria-label',button.textContent);
  if(oldId)document.querySelector(`#q${oldId}`).scrollIntoView({behavior:'instant',block:'start'});
  schedule();
});
addEventListener('beforeprint',buildReading);
update();
// Restore deep links after dynamic chapter creation and font layout.
document.fonts.ready.then(()=>{
  if(!jumpToStep(location.hash)&&/^#q(19|2[0-5])$/.test(location.hash))document.querySelector(location.hash)?.scrollIntoView({behavior:'instant'});
  schedule();
});
