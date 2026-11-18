import {chapters} from './lesson.mjs?v=20261108-peloton-1';
import {updateAiExample} from './ai-example.mjs?v=20261108-peloton-1';
import {bagVisual,tfVisual,topicVisual,authorshipVisual} from './visuals.mjs?v=20261108-peloton-1';
import {updateVisual} from './visual-update.mjs?v=20261108-peloton-1';
import {prose} from './math.mjs?v=20261108-peloton-1';
import {initializeExplorer} from './explorer.mjs?v=20261108-peloton-1';
import {topicModelsVisual} from './topic-models.mjs?v=20261108-peloton-1';

// Bag of words and TF–IDF share one pinned workspace and one visual node.
export function storyGroups(source=chapters){
  const groups=[];
  for(const chapter of source){
    if(chapter.id==='tf-idf'&&groups.at(-1)?.chapters.at(-1).id==='bag-of-words')groups.at(-1).chapters.push(chapter);
    else groups.push({chapters:[chapter],application:!!chapter.application});
  }
  return groups.map(group=>{
    let offset=0;
    const steps=group.chapters.flatMap(chapter=>{
      const start=offset;offset+=chapter.stages.length;
      return chapter.stages.map((scene,localStage)=>({chapter,scene,localStage,offset:start}));
    });
    return {...group,steps};
  });
}

export function initializeStory() {
  const clamp=(x,a=0,b=1)=>Math.max(a,Math.min(b,x));
  const media=matchMedia('(prefers-reduced-motion: reduce)');
  const state={motion:!media.matches&&new URLSearchParams(location.search).get('motion')!=='reduce'};
  const html=(el,value)=>{if(el.innerHTML!==value)el.innerHTML=value;};
  const root=document.querySelector('#chapters'),applications=document.querySelector('#application-chapters');
  const roots=[root,applications],groups=storyGroups();
  const sceneMarkup=(group,i)=>{
    const chapter=group.chapters[0];
    const anchors=group.chapters.slice(1).map(c=>`<span class="chapter-anchor" data-chapter-id="${c.id}" aria-hidden="true"></span>`).join('');
    return `<section class="track" data-chapter-id="${chapter.id}" data-group="${i}" aria-label="${chapter.title}">${anchors}<div class="scene"><div class="scene-heading"><h2><span class="chapter-number"></span><span class="chapter-title"></span></h2><span class="chapter-subtitle"></span></div><div class="scene-body"><div class="scene-copy"><h3></h3><p class="explanation"></p><p class="small qualification"></p></div><div class="visual"></div></div><div class="scene-bottom"><span class="counter" aria-live="polite"></span><button type="button" class="previous" aria-label="Previous stage">← Previous</button><button type="button" class="next" aria-label="Next stage">Next →</button><button type="button" class="motion" aria-pressed="${state.motion}">${state.motion?'Motion on':'Motion off'}</button></div><div class="progress" aria-hidden="true"><span></span></div></div></section>`;
  };
  root.innerHTML=groups.map((g,i)=>g.application?'':sceneMarkup(g,i)).join('');
  applications.innerHTML=groups.map((g,i)=>g.application?sceneMarkup(g,i):'').join('');
  const views=roots.flatMap(parent=>[...parent.querySelectorAll('.track')]).map(track=>({track,group:groups[Number(track.dataset.group)],visual:track.querySelector('.visual'),copy:track.querySelector('.scene-copy'),counter:track.querySelector('.counter'),bar:track.querySelector('.progress span'),stage:-1,p:0,segment:0,lastVisual:''}));
  let queued=false,active=0,disposed=false;
  const header=()=>document.querySelector('.topbar').getBoundingClientRect().height;
  function layout(){
    const height=innerHeight-header();
    document.documentElement.style.setProperty('--scene',`${height}px`);
    views.forEach(v=>{
      v.segment=height*1.05;
      v.track.style.height=`${height+(v.group.steps.length-1)*v.segment}px`;
      v.track.querySelectorAll('.chapter-anchor').forEach(anchor=>{
        const first=v.group.steps.findIndex(step=>step.chapter.id===anchor.dataset.chapterId);
        anchor.style.top=`${first*v.segment}px`;
      });
    });
    render();
  }
  function render(){
    if(disposed)return;
    queued=false;
    for(const [i,v] of views.entries()){
      const position=clamp((scrollY-v.track.offsetTop+header())/v.segment,0,v.group.steps.length-1);
      const raw=Math.abs(position-Math.round(position))<.002?Math.round(position):position;
      const stage=Math.round(raw),entry=v.group.steps[stage],{chapter,scene,localStage,offset}=entry;
      const p=(state.motion?raw:stage)-offset;v.p=raw;
      const bounds=v.track.getBoundingClientRect();
      if(bounds.top<=innerHeight*.5&&bounds.bottom>innerHeight*.5)active=i;
      if(bounds.bottom<0||bounds.top>innerHeight)continue;
      if(v.stage!==stage){
        const chapterIndex=chapters.indexOf(chapter);
        v.track.querySelector('.chapter-number').textContent=chapterIndex===0?'Opening':String(chapterIndex).padStart(2,'0');
        v.track.querySelector('.chapter-title').textContent=chapter.title;
        v.track.querySelector('.chapter-subtitle').textContent=chapter.subtitle;
        v.track.setAttribute('aria-label',chapter.title);
        v.copy.querySelector('h3').textContent=scene.title;
        html(v.copy.querySelector('.explanation'),prose(scene.text,scene.links));
        html(v.copy.querySelector('.qualification'),prose(scene.note));
        v.counter.textContent=`${localStage+1} / ${chapter.stages.length}`;v.stage=stage;
        v.track.dataset.stage=localStage;
        v.track.dataset.activeChapter=chapter.id;
        v.track.dataset.visual=scene.visual;
      }
      const fade=state.motion?clamp((Math.abs(raw-stage)-.41)/.09):0;
      v.copy.style.opacity=1-fade;
      v.copy.style.transform=`translateY(${Math.sign(stage-raw)*fade*10}px)`;
      if(chapter.id==='ai-writing')updateAiExample(v.visual,localStage,p,{reducedMotion:!state.motion});
      else{
        const draw={'bag-of-words':bagVisual,'tf-idf':tfVisual,authorship:authorshipVisual,bertopic:topicVisual,'topic-models':topicModelsVisual}[chapter.id];
        const markup=draw(localStage,p,state.motion);
        if(markup!==v.lastVisual){updateVisual(v.visual,markup);v.lastVisual=markup;}
      }
      v.bar.style.width=`${chapter.stages.length===1?100:clamp(raw-offset,0,chapter.stages.length-1)/(chapter.stages.length-1)*100}%`;
      v.track.dataset.progress=raw.toFixed(4);
    }
    const v=views[active],chapter=v.group.steps[Math.round(v.p)].chapter;
    document.querySelectorAll('.topbar nav a').forEach(a=>{
      if(a.hash===`#${chapter.id}`)a.setAttribute('aria-current','step');else a.removeAttribute('aria-current');
    });
  }
  function schedule(){if(!disposed&&!queued){queued=true;requestAnimationFrame(render);}}
  function jump(i,stage){
    const from=i;
    if(stage<0&&views[i].group.application&&!views[i-1]?.group.application){document.querySelector('#compare').scrollIntoView();return;}
    if(stage>=views[i].group.steps.length&&!views[i].group.application&&views[i+1]?.group.application){document.querySelector('#compare').scrollIntoView();return;}
    if(stage<0){if(i===0){document.querySelector('#title').scrollIntoView();return;}i--;stage=views[i].group.steps.length-1;}
    if(stage>=views[i].group.steps.length){if(i===views.length-1){document.querySelector('#reading').scrollIntoView();return;}i++;stage=0;}
    const v=views[i];scrollTo({top:v.track.offsetTop-header()+stage*v.segment,behavior:'instant'});
    if(i!==from)v.track.querySelector(stage===0?'.next':'.previous').focus({preventScroll:true});
    schedule();
  }
  function motion(value){state.motion=value;roots.forEach(parent=>parent.querySelectorAll('.motion').forEach(b=>{b.textContent=value?'Motion on':'Motion off';b.setAttribute('aria-pressed',String(value));}));render();}
  function onClick(e){
    const track=e.target.closest('.track');if(!track)return;
    const i=Number(track.dataset.group),v=views[i];
    if(e.target.closest('.next'))jump(i,Math.round(v.p)+1);
    if(e.target.closest('.previous'))jump(i,Math.round(v.p)-1);
    if(e.target.closest('.motion'))motion(!state.motion);
  }
  function keydown(e){
    if(e.altKey||e.ctrlKey||e.metaKey||e.target.closest('input,select,textarea,button,a,summary'))return;
    if(e.key==='ArrowRight'||e.key==='ArrowLeft'){
      const r=views[active].track.getBoundingClientRect();
      if(r.top<=header()+4&&r.bottom>header()){e.preventDefault();jump(active,Math.round(views[active].p)+(e.key==='ArrowRight'?1:-1));}
    }
  }
  const explorer=document.querySelector('#term-explorer');
  let disposeExplorer=()=>{};
  const onMotionChange=e=>motion(!e.matches);
  function dispose(){
    disposed=true;
    roots.forEach(parent=>parent.removeEventListener('click',onClick));
    disposeExplorer();
    document.removeEventListener('keydown',keydown);
    removeEventListener('scroll',schedule);removeEventListener('resize',layout);
    media.removeEventListener('change',onMotionChange);
  }
  try{
    disposeExplorer=initializeExplorer(explorer);
    layout();
    roots.forEach(parent=>parent.addEventListener('click',onClick));
    document.addEventListener('keydown',keydown);
    addEventListener('scroll',schedule,{passive:true});addEventListener('resize',layout);
    media.addEventListener('change',onMotionChange);
    document.fonts.ready.then(schedule);
    return dispose;
  }catch(error){dispose();throw error;}
}
