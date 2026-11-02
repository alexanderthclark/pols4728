(() => {
  'use strict';
  const clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,v));
  const reducedQuery=matchMedia('(prefers-reduced-motion: reduce)');
  let motion=!reducedQuery.matches,queued=false;
  const chapters=[...document.querySelectorAll('.scroll-chapter')].map(el=>{
    const n=Number(el.dataset.stages),scene=el.querySelector('.scene'),refs=Object.fromEntries([...el.querySelectorAll('[data-scene]')].map(x=>[x.dataset.scene,x]));
    const dots=el.querySelector('.scene-dots');
    const chapter={el,n,scene,refs,last:-1,stage:-1,dots:[]};
    for(let i=0;i<n;i++){
      const b=document.createElement('button');b.type='button';b.title=`Stage ${i+1}`;b.setAttribute('aria-label',`Stage ${i+1}`);
      b.addEventListener('click',()=>{
        const top=el.getBoundingClientRect().top+scrollY;
        const span=el.offsetHeight-scene.offsetHeight;
        scrollTo({top:top+i/(n-1)*span-(innerWidth<=650?60:76),behavior:motion?'smooth':'instant'});
      });dots.append(b);chapter.dots.push(b);
    }
    el.querySelector('.motion-toggle').addEventListener('click',()=>{motion=!motion;chapters.forEach(c=>c.last=-1);update();});
    return chapter;
  });
  function resize(){
    chapters.forEach(c=>{
      const span=c.el.offsetHeight-c.scene.offsetHeight;
      c.el.querySelectorAll('.scroll-marker').forEach((marker,i)=>marker.style.top=`${i/(c.n-1)*span}px`);
      c.last=-1;
    });update();
  }
  function update(){
    queued=false;
    const header=innerWidth<=650?60:76;
    chapters.forEach(c=>{
      const rect=c.el.getBoundingClientRect();
      const progress=clamp((header-rect.top)/(c.el.offsetHeight-c.scene.offsetHeight))*(c.n-1);
      const p=motion?progress:Math.round(progress);
      if(Math.abs(p-c.last)<.0001)return;c.last=p;
      const data=window.ImbalanceScenes[c.el.dataset.chapter](p);
      const stage=data.stage??Math.round(p);
      c.refs.title.textContent=data.title;c.refs.description.textContent=data.description;c.refs.eyebrow.textContent=data.eyebrow;
      if(c.refs.controls)c.refs.controls.hidden=!data.controls;
      c.refs.canvas.innerHTML=data.svg;
      if(data.equationMath){
        c.refs.equation.innerHTML=window.LogisticMath[data.equationMath].html+'<span class="equation-caption"></span>';
        c.refs.equation.querySelector('.equation-caption').textContent=data.equation;
      }else c.refs.equation.textContent=data.equation;
      c.refs.note.innerHTML=data.note||'Scroll back to retrace the argument.';
      c.refs.count.textContent=`${String(stage+1).padStart(2,'0')} / ${String(c.n).padStart(2,'0')}`;
      c.el.querySelector('.scene-progress span').style.width=`${progress/(c.n-1)*100}%`;
      const toggle=c.el.querySelector('.motion-toggle');toggle.textContent=motion?'Motion on':'Motion off';toggle.setAttribute('aria-pressed',String(motion));
      c.dots.forEach((b,i)=>{b.classList.toggle('active',i===stage);if(i===stage)b.setAttribute('aria-current','step');else b.removeAttribute('aria-current');});
      c.el.dataset.progress=p.toFixed(4);c.el.dataset.stage=stage;
    });
  }
  function schedule(){if(!queued){queued=true;requestAnimationFrame(update);}}
  addEventListener('scroll',schedule,{passive:true});addEventListener('resize',resize);
  reducedQuery.addEventListener('change',e=>{motion=!e.matches;chapters.forEach(c=>c.last=-1);update();});
  resize();
  // A direct stage URL needs its marker positioned before scrolling to it.
  const anchor=document.getElementById(location.hash.slice(1));
  if(anchor?.classList.contains('scroll-marker'))requestAnimationFrame(()=>anchor.scrollIntoView({behavior:'instant',block:'start'}));
})();
