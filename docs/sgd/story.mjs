import {epochState,renderEpochCounter} from './epoch-counter.mjs';
import {scenes as allScenes,foundationScenes} from './scenes.mjs';
const scenes=document.body.dataset.chapter==='batches'?foundationScenes:allScenes;
import {render} from './views.mjs';
import {constructionKinds} from './construction.mjs';
const $=id=>document.getElementById(id),stage=document.querySelector('.stage'),steps=[...document.querySelectorAll('.step')];
const media=matchMedia('(prefers-reduced-motion: reduce)');
let experiment,index=0,pending=false,last='',frame;
const typesetCache=new Map();
const procedure=$('procedure'),procedureItems=[...procedure.children];
function showProcedure(scene,p,frame){
 const plan=scene.layout==='plan',active=plan?Math.min(3,Math.floor(Math.max(0,Math.min(1,(p-.1)/.72))*4)):(frame.procedureStep??scene.procedureStep);
 procedure.hidden=!plan&&!Number.isInteger(active);
 procedureItems.forEach((item,i)=>{
  item.classList.toggle('is-current',i===active);
  item.classList.toggle('is-pending',plan&&i>active);
  item.classList.toggle('is-done',i<active);
  if(i===active)item.setAttribute('aria-current','step');else item.removeAttribute('aria-current');
  item.setAttribute('aria-hidden',String(plan&&i>active));
 });
}
function formula(tex){if(!tex)return '';if(!typesetCache.has(tex))typesetCache.set(tex,window.katex.renderToString(tex,{displayMode:true,output:'htmlAndMathml',throwOnError:true,strict:'error'}));return typesetCache.get(tex);}
function draw(){pending=false;if(!experiment)return;const anchor=stage.getBoundingClientRect().top;let selected=0;steps.forEach((s,i)=>{if(s.getBoundingClientRect().top<=anchor+.5)selected=i;});index=selected;const rect=steps[index].getBoundingClientRect(),scene=scenes[index];let p=Math.max(0,Math.min(1,(anchor-rect.top)/rect.height));p=Math.round(p*200)/200;if(media.matches)p=1;const mobile=innerWidth<=700,signature=[index,p,mobile,innerWidth,innerHeight].join(':');if(signature===last)return;last=signature;
 if(stage.dataset.scene!==scene.id){$('heading').textContent=scene.title;$('caption').textContent=scene.caption;$('note').textContent=scene.note;$('part').textContent=scene.part;$('count').textContent=`${String(index+1).padStart(2,'0')} / ${scenes.length}`;$('status').textContent=scene.title;stage.dataset.scene=scene.id;}
 const tex=mobile&&scene.compact?scene.compact:scene.tex;$('equation').hidden=!tex;if($('equation').dataset.tex!==tex){$('equation').innerHTML=formula(tex);$('equation').dataset.tex=tex;}
 stage.dataset.construction=String(constructionKinds.includes(scene.kind));
 stage.dataset.training=String(scene.kind.startsWith('batches')||scene.kind.startsWith('train-')||scene.kind==='epoch-two');
 stage.dataset.layout=scene.layout||'standard';
 const counter=$('epoch-counter'),showCounter=document.body.dataset.chapter!=='batches'&&(scene.part.startsWith('02')||['initialize-wide','batches-wide','train-wide'].includes(scene.kind));
 if(showCounter&&counter.hidden)renderEpochCounter(counter,epochState(0,experiment.config.epochs,experiment.config.batchesPerEpoch));
 counter.hidden=!showCounter;stage.dataset.counter=String(showCounter);
 showProcedure(scene,p,{});
 frame=render($('plot'),$('detail'),scene,p,experiment);
 showProcedure(scene,p,frame);
 if(showCounter)renderEpochCounter(counter,epochState(frame.completedUpdates,experiment.config.epochs,experiment.config.batchesPerEpoch));
 $('detail-label').textContent=frame.detailTitle;$('plot-label').textContent=frame.plotTitle;$('progress').style.width=`${100*(index+p)/scenes.length}%`;
 // Only the changing caption moves, and it remains still for most of the beat.
 const arrival=media.matches?1:Math.min(1,p/.1);$('caption').style.transform=`translateY(${(1-arrival)*6}px)`;$('caption').style.opacity=String(.65+.35*arrival);
 stage.dataset.progress=String(p);stage.dataset.frame=JSON.stringify(frame);
}
function queue(){if(!pending){pending=true;requestAnimationFrame(draw);}}
function go(delta){const n=Math.max(0,Math.min(scenes.length-1,index+delta));const rect=steps[n].getBoundingClientRect();window.scrollTo({top:window.scrollY+rect.top+rect.height*.15,behavior:media.matches?'instant':'smooth'});}
addEventListener('scroll',queue,{passive:true});addEventListener('resize',queue);media.addEventListener('change',()=>{last='';queue();});document.addEventListener('keydown',e=>{if(e.altKey||e.ctrlKey||e.metaKey||e.shiftKey||/INPUT|TEXTAREA|SELECT|BUTTON/.test(e.target.tagName))return;if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();go(e.key==='ArrowRight'?1:-1);}});
$('reading').addEventListener('toggle',()=>{if($('reading').open)$('reading').scrollIntoView();});document.querySelector('.skip').addEventListener('click',()=>{$('reading').open=true;});
try{if(document.body.dataset.chapter==='batches'){experiment={};}else{const response=await fetch('./experiment.json');if(!response.ok)throw new Error('Could not load experiment');experiment=await response.json();}document.body.classList.add('enhanced');$('steps').setAttribute('aria-hidden','true');$('steps').setAttribute('inert','');stage.hidden=false;if(location.hash){const target=document.getElementById(decodeURIComponent(location.hash.slice(1)));target?.scrollIntoView({behavior:'instant',block:'start'});}draw();document.fonts.ready.then(()=>{last='';queue();});}catch(error){console.error(error);stage.hidden=true;document.body.classList.remove('enhanced');$('steps').removeAttribute('aria-hidden');$('steps').removeAttribute('inert');}
