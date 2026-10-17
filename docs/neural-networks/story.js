(function(){
  'use strict';
  const State=window.NNStoryState,Views=window.NNViews,Narrative=window.NNStoryNarrative;
  const steps=Array.from(document.querySelectorAll('.step[data-scene]'));
  const byId=id=>document.getElementById(id);
  const reduced=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  byId('story').classList.add('is-enhanced');
  // Keep the full reading route in the document while its sections supply
  // scroll distances. The pinned caption is the sole interactive explanation.
  byId('scroll-steps').setAttribute('aria-hidden','true');
  byId('scroll-steps').setAttribute('inert','');
  let currentIndex=0,renderSignature='',pending=0,inspectionX=null,lastFrame=null;
  const chapter=scene=>scene.startsWith('width-')?'02 · Width and joints':scene.startsWith('sine-')||scene==='universal'?'03 · Approximation':'01 · Shallow networks';
  function update(){
    pending=0;
    const mobile=window.matchMedia('(max-width: 760px)').matches;
    const stage=byId('story-stage');
    // Align each scene's progress with the top of the whole pinned panel.
    const focus=stage.getBoundingClientRect().top;
    let index=0;
    for(let i=0;i<steps.length;i++){if(steps[i].getBoundingClientRect().top<=focus)index=i;}
    currentIndex=index;
    const rect=steps[index].getBoundingClientRect(),scene=steps[index].dataset.scene;
    let progress=Math.max(0,Math.min(1,(focus-rect.top)/Math.max(rect.height,1)));
    // Quantization prevents needless SVG work on fractional-pixel scroll events.
    progress=Math.round(progress*100)/100;
    if(reduced)progress=scene==='affine'?(progress<.65?0:1):scene==='width-many'||scene==='sine-wide'?Math.round(progress*10)/10:1;
    const signature=scene+':'+progress+':'+(mobile?'compact':'desktop')+':'+(inspectionX===null?'story':inspectionX);
    if(signature!==renderSignature){
      renderSignature=signature;
      const frame=State.frame(scene,progress);
      frame.compact=mobile;
      const domain=frame.kind==='sine'?[0,2*Math.PI]:[-2,2];
      if(inspectionX!==null){
        frame.x=Math.max(domain[0],Math.min(domain[1],inspectionX));
        frame.inspectInput=true;
      }
      const input=byId('input-x');
      input.min=String(domain[0]);input.max=String(domain[1]);input.value=String(frame.x);
      input.setAttribute('aria-valuetext','x = '+frame.x.toFixed(2));
      byId('input-x-value').textContent=frame.x.toFixed(2);
      lastFrame=frame;
      Views.render(byId('function-plot'),byId('network-plot'),frame);
      byId('stage-name').textContent=chapter(scene);
      byId('stage-count').textContent=String(index+1).padStart(2,'0')+' / '+steps.length;
      byId('stage-caption').textContent=frame.caption;
      const cue=Narrative.cue(frame,{reduced});
      frame.narrativeKey=cue.key;frame.narrativeOpacity=cue.opacity;
      const caption=byId('narrative-caption');
      if(caption.dataset.cue!==cue.key){
        const source=cue.source==='scene'?steps[index]:byId(cue.source);
        // The hidden reading route retains its original heading anchors.
        caption.innerHTML=source.innerHTML.replace(/\sid="[^"]*"/g,'');
        caption.dataset.cue=cue.key;
      }
      caption.style.opacity=String(cue.opacity);
      caption.setAttribute('aria-hidden',cue.opacity===0?'true':'false');
      if(cue.opacity===0)caption.setAttribute('inert','');else caption.removeAttribute('inert');
      window.NNFormula.render(byId('stage-equation'),frame);
      const activationModel=frame.model||(frame.kind==='relu'?window.NNMath.createShallow(0,[{t:0,a:1}]):null);
      const counter=byId('active-neurons');
      frame.hiddenCount=activationModel?activationModel.width:0;
      frame.activeCount=activationModel?window.NNMath.contributions(activationModel,frame.x).filter(unit=>unit.activation>1e-8).length:0;
      counter.hidden=!activationModel;
      counter.textContent=activationModel?frame.activeCount+' of '+frame.hiddenCount+' neurons active · x = '+frame.x.toFixed(2):'';
      if(byId('previous'))byId('previous').disabled=index===0;
      if(byId('next'))byId('next').disabled=index===steps.length-1;
      if(byId('story-status')&&byId('story-status').dataset.cue!==cue.key){byId('story-status').dataset.scene=scene;byId('story-status').dataset.cue=cue.key;byId('story-status').textContent=cue.source==='bias-omitted'?'By convention, intercepts are often omitted from network diagrams.':steps[index].querySelector('h2,h3')?.textContent||scene;}
      steps.forEach((s,i)=>{s.classList.toggle('is-active',i===index);if(i===index)s.setAttribute('aria-current','step');else s.removeAttribute('aria-current');});
      const story=byId('story');if(story)story.dataset.scene=scene;
    }
    const scrolled=window.scrollY||0,total=document.documentElement.scrollHeight-window.innerHeight;
    if(byId('progress'))byId('progress').style.width=Math.max(0,Math.min(100,total?scrolled/total*100:0))+'%';
  }
  function queue(){if(!pending)pending=window.requestAnimationFrame(update);}
  function go(delta){
    const target=steps[Math.max(0,Math.min(steps.length-1,currentIndex+delta))];
    const rect=target.getBoundingClientRect();
    const offset=byId('story-stage').getBoundingClientRect().top;
    // Land after the entering fade so the selected scene is readable.
    window.scrollTo({top:(window.scrollY||0)+rect.top-offset+rect.height*.12,behavior:reduced?'auto':'smooth'});
  }
  if(byId('previous'))byId('previous').addEventListener('click',()=>go(-1));
  if(byId('next'))byId('next').addEventListener('click',()=>go(1));
  byId('input-x').addEventListener('input',e=>{
    const value=Number(e.target.value);
    if(Number.isFinite(value)){inspectionX=value;queue();}
  });
  document.addEventListener('keydown',e=>{
    if(e.altKey||e.ctrlKey||e.metaKey||/INPUT|TEXTAREA|SELECT/.test(e.target?.tagName||''))return;
    if(e.key==='ArrowRight'){e.preventDefault();go(1);}else if(e.key==='ArrowLeft'){e.preventDefault();go(-1);}
  });
  window.addEventListener('scroll',queue,{passive:true});window.addEventListener('resize',queue);
  document.fonts?.ready.then(queue);
  update();
  window.NNScrollStory=Object.freeze({getFrame:()=>lastFrame,refresh:update});
})();
