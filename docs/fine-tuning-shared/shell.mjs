export const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
export const sceneAt=(position,count)=>clamp(Math.round(position),0,count-1);
export const positionForScroll=(scrollY,top,beat,count)=>clamp((scrollY-top)/beat,0,count-1);

export function mountStory(lesson){
  const root=document.querySelector('[data-story]');
  const pin=document.querySelector('.story-pin');
  const svg=document.querySelector('#lesson-visual');
  const label=document.querySelector('#scene-count');
  const heading=document.querySelector('#scene-heading');
  const body=document.querySelector('#scene-body');
  const equation=document.querySelector('#scene-equation');
  const progress=document.querySelector('.progress-fill');
  const previous=document.querySelector('#previous-scene');
  const next=document.querySelector('#next-scene');
  const chooser=document.querySelector('#scene-select');
  const motionToggle=document.querySelector('#motion-toggle');
  const space=document.querySelector('.scroll-space');
  const motion=matchMedia('(prefers-reduced-motion: reduce)');
  const compact=matchMedia('(max-width: 700px)');
  const count=lesson.scenes.length;
  let top=0,beat=1,lastCaption=-1,queued=false,lastPosition=-1,manualReduced=false;
  document.body.classList.add('enhanced');
  function measure(){
    const preserve=lastPosition>=0&&root.getBoundingClientRect().top<80;
    const heldPosition=lastPosition;
    beat=Math.max(460,innerHeight*.88);
    const header=parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--header'))||60;
    top=root.getBoundingClientRect().top+scrollY-header;
    space.style.height=`${beat*(count-1)}px`;
    svg.setAttribute('viewBox',compact.matches?'0 0 500 480':'0 0 1000 560');
    if(preserve)scrollTo({top:top+heldPosition*beat,behavior:'instant'});
    lastCaption=-1;
    schedule(true);
  }
  function draw(){
    queued=false;
    const position=positionForScroll(scrollY,top,beat,count);
    const current=sceneAt(position,count);
    const reduced=motion.matches||manualReduced;
    const drawPosition=reduced?current:position;
    svg.innerHTML=lesson.render(drawPosition,{compact:compact.matches,short:innerHeight<680,reducedMotion:reduced});
    svg.dataset.motion=reduced?'steps':'continuous';
    motionToggle.setAttribute('aria-pressed',String(reduced));
    motionToggle.disabled=motion.matches;
    svg.dataset.position=position.toFixed(6);
    svg.dataset.renderPosition=drawPosition.toFixed(6);
    root.dataset.scene=String(current);
    root.dataset.position=position.toFixed(6);
    if(current!==lastCaption){
      const scene=lesson.scenes[current];
      const short=compact.matches&&innerHeight<680;
      heading.textContent=short?(scene.shortTitle||scene.title):scene.title;
      body.innerHTML=short?(scene.shortBody||scene.body):scene.body;
      equation.innerHTML=scene.equation||'';
      label.textContent=`${String(current+1).padStart(2,'0')} / ${count}`;
      svg.setAttribute('aria-label',scene.title+'. '+body.textContent);
      chooser.value=String(current);
      previous.disabled=current===0;
      next.disabled=current===count-1;
      lastCaption=current;
    }
    progress.style.width=`${position/(count-1)*100}%`;
    lastPosition=position;
    root.dataset.lastPosition=String(lastPosition);
  }
  function schedule(){if(!queued){queued=true;requestAnimationFrame(draw)}}
  function go(index,focus=false){
    const target=clamp(index,0,count-1);
    root.dataset.navTarget=String(target);
    scrollTo({top:top+target*beat,behavior:'instant'});
    schedule();
    if(focus)heading.focus({preventScroll:true});
    history.replaceState(null,'',`#${lesson.scenes[target].id}`);
  }
  previous.addEventListener('click',()=>go(sceneAt(lastPosition,count)-1,true));
  next.addEventListener('click',()=>go(sceneAt(lastPosition,count)+1,true));
  chooser.addEventListener('change',()=>go(Number(chooser.value),true));
  motionToggle.addEventListener('click',()=>{manualReduced=!manualReduced;schedule()});
  document.querySelector('.begin').addEventListener('click',event=>{event.preventDefault();go(0,true)});
  document.addEventListener('keydown',event=>{
    if(event.altKey||event.metaKey||event.ctrlKey||event.target.closest('input,select,textarea,summary,details[open]'))return;
    const box=root.getBoundingClientRect();
    if(box.top>innerHeight*.3||box.bottom<innerHeight*.4)return;
    if(event.key==='ArrowRight'||event.key==='ArrowLeft'){
      event.preventDefault();go(sceneAt(lastPosition,count)+(event.key==='ArrowRight'?1:-1),true);
    }
  });
  addEventListener('scroll',schedule,{passive:true});
  addEventListener('resize',measure,{passive:true});
  motion.addEventListener('change',()=>schedule(true));
  compact.addEventListener('change',measure);
  measure();
  const id=decodeURIComponent(location.hash.slice(1));
  const initial=lesson.scenes.findIndex(scene=>scene.id===id);
  if(initial>=0)requestAnimationFrame(()=>go(initial));
  document.fonts?.ready.then(()=>{measure();if(initial>=0)go(initial)});
  addEventListener('pageshow',()=>{if(initial>=0)requestAnimationFrame(()=>go(initial))},{once:true});
  document.querySelector('#reading').addEventListener('click',event=>{
    const link=event.target.closest('[data-jump]');if(!link)return;
    event.preventDefault();go(Number(link.dataset.jump),true);
  });
  window.lessonStory={go,measure,get position(){return lastPosition},get beat(){return beat},get top(){return top},count};
}
