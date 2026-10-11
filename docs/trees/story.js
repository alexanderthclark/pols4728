(function (root, factory) {
  'use strict';
  const api = factory(root.DeliveryView || (typeof require === 'function' ? require('./delivery-view.js') : null), root.DeliveryExample || (typeof require === 'function' ? require('./delivery.js') : null), root.GrowthView || (typeof require === 'function' ? require('./growth-view.js') : null));
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root.document) api.init(root.document, root);
})(typeof window !== 'undefined' ? window : globalThis, function (D, B, G) {
  'use strict';
  const sceneHeadings = [
    'Predict delivery time', 'Fit a tree to the residuals', 'Add the first correction',
    'Fit the new residuals', 'Add the second correction', 'Keep fitting the residuals'
  ];
  const sceneEquations = [
    'F₀(x) = ȳ = 12', 'h₁ ← fit(y − F₀)', 'F₁ = F₀ + h₁',
    'h₂ ← fit(y − F₁)', 'F₂ = F₁ + h₂', ''
  ];
  const sceneTakeaways = [
    'Observed times are 6, 11, or 17 minutes. Hover a point for its exact value.', '', 'Add the same leaf value to every prediction in its interval.',
    'The next tree fits new targets, so its best cut can change.',
    'One leaf raises both right-hand intervals. Some errors grow while total squared error falls.', ''
  ];
  function scenePresentation(scene, state={}) {
    if(scene===6)return G.transitionPresentation();
    if(scene>=7)return G.presentation(scene-7,state);
    let equation=sceneEquations[scene],takeaway=sceneTakeaways[scene];
    if(scene===0 && !(state.process>0))equation='';
    if(scene===4 && state.process>=3)takeaway='Two cutpoints make three intervals. The heights still leave errors to correct.';
    if(scene===1 && state.process>=2)takeaway='Each leaf predicts the mean residual of the observations it contains.';
    if(scene===5){
      const current=D.continuationState(state.continuation),round=Math.max(3,current.round);
      const sub=n=>String(n).split('').map(d=>'₀₁₂₃₄₅₆₇₈₉'[Number(d)]).join('');
      equation=current.round===2 || current.amount===0?`h${sub(round)} ← fit(y − F${sub(round-1)})`:`F${sub(round)} = F${sub(round-1)} + h${sub(round)}`;
      takeaway=state.continuation>=1
        ?'This greedy sequence never reaches exact zero in finitely many rounds. Training fit alone does not establish accuracy on new data.'
        :'Later trees revisit existing cuts, changing the heights without adding intervals.';
    }
    return {heading:sceneHeadings[scene],equation,takeaway};
  }
  const fmt = n => Math.abs(n) < 1e-9 ? '0' : String(Number(n.toFixed(2))).replace('-', '−');
  function renderScene(scene, state = {}, compact = false) {
    return scene < 6 ? D.render(scene, state, compact) : scene === 6 ? G.renderTransition(compact) : G.render(scene-7,state,compact);
  }
  function readout(scene, state = {}) {
    return scene < 6 ? D.readout(scene, state) : scene === 6 ? G.transitionReadout() : G.readout(scene-7,state);
  }

  function init(doc,win) {
    const get=id=>doc.getElementById(id);
    doc.documentElement.dataset.enhanced='true';
    const state={process:0,candidateIndex:null,continuation:0,growth:0};
    const steps=Array.from(doc.querySelectorAll('.step'));
    const diagram=get('diagram'), shell=diagram.closest('.stage-shell');
    const reduced=win.matchMedia('(prefers-reduced-motion: reduce)');
    const compact=win.matchMedia('(max-width: 800px)');
    let scene=0, frame=0, animation=0, targetScene=null, targetTimeout=0, processResumeTarget=null, growthResumeTarget=null, scrollPosition=win.scrollY, scrollSyncPending=true;
    function render(transition=false) {
      shell.dataset.scene=String(scene);
      shell.dataset.layout='visual';
      get('stage-topic').textContent=scenePresentation(scene,state).heading;
      get('stage-topic').hidden=true;
      const presentation=scenePresentation(scene,state);
      get('visual-heading').hidden=false;
      get('visual-heading').textContent=presentation.heading;
      get('stage-equation').textContent=presentation.equation;
      get('stage-equation').hidden=scene===6;
      get('stage-count').textContent=`${String(scene+1).padStart(2,'0')} / ${String(steps.length).padStart(2,'0')}`;
      get('stage-caption').textContent=presentation.takeaway;
      get('stage-caption').hidden=scene===6;
      get('continue-boosting').hidden=scene!==9;
      const growthScene=scene===7 || scene===8;
      get('growth-controls').hidden=!growthScene;
      if(growthScene){
        const maximum=G.max(scene-7), complete=Math.floor(state.growth+1e-8), unit=scene===7?'split':'stump';
        get('growth-label').textContent=state.growth>complete+1e-8
          ?`Drawing ${unit} ${complete+1} of ${maximum}`:`${complete} / ${maximum} ${unit}s`;
        get('growth-back').textContent=`Previous ${unit}`;
        get('growth-next').textContent=`Next ${unit}`;
        get('growth-back').disabled=state.growth<=0;
        get('growth-next').disabled=state.growth>=maximum;
      }
      const processScene=scene>=0 && scene<=4, fitScene=scene===1 || scene===3;
      get('process-controls').hidden=!processScene;
      get('candidate-controls').hidden=!fitScene;
      get('continuation-controls').hidden=scene!==5;
      if(scene===5){
        const current=D.continuationState(state.continuation),complete=current.amount>=1;
        const round=complete?current.round:current.round-1;
        get('continuation-label').textContent=complete || current.amount===0?`After ${round} trees`:`Adding tree ${current.round}`;
        get('continuation-back').disabled=state.continuation<=0;
        get('continuation-next').disabled=state.continuation>=1;
      }
      if(processScene){
        const maximum=D.processMax(scene),labels=D.processLabels(scene), stage=Math.min(maximum,Math.ceil(state.process));
        const label=`${stage} / ${maximum} · ${labels[stage]}`;
        if(get('process-label').textContent!==label)get('process-label').textContent=label;
        get('process-progress').value=state.process;
        get('process-progress').max=maximum;
        get('process-progress').setAttribute('aria-valuetext',labels[stage]);
        get('process-back').disabled=state.process<=0;
        get('process-next').disabled=state.process>=maximum;
      }
      if(fitScene){
        const round=scene===1?1:2,candidates=B.candidates(round);
        const winner=candidates.reduce((best,row,i)=>row.sse<candidates[best].sse?i:best,0);
        const index=state.candidateIndex===null?winner:state.candidateIndex,row=candidates[index];
        get('candidate-cut').value=index;
        get('candidate-value').textContent=`${fmt(row.threshold)} km`;
        get('candidate-description').textContent=`Leaf means: ${fmt(row.left)}, ${fmt(row.right)} min. Residual SSE: ${fmt(row.sse)} min². ${index===winner?'Lowest among the seven candidate splits.':`Best split: ${fmt(candidates[winner].threshold)} km, SSE ${fmt(candidates[winner].sse)} min².`}`;
      }
      get('calculation').textContent=readout(scene,state);
      get('calculation').hidden=scene===6;
      get('previous').disabled=scene===0;
      get('next').disabled=scene===steps.length-1;
      steps.forEach((s,i)=>s.classList.toggle('active',i===scene));
      diagram.setAttribute('aria-label',scene===6?'Transition to two predictors':'Current tree and its predictions');
      diagram.innerHTML=renderScene(scene,state,compact.matches);
      if(transition && scene>1 && !reduced.matches && diagram.animate) diagram.animate([{opacity:.25},{opacity:1}],{duration:280,easing:'ease-out'});
    }
    function stopAnimation(keepProcessPosition=false){
      win.cancelAnimationFrame(animation);animation=0;
      if(!keepProcessPosition){processResumeTarget=null;growthResumeTarget=null;}
      get('process-replay').textContent=processResumeTarget===null?'Replay sequence':'Resume sequence';
      get('growth-replay').textContent=growthResumeTarget===null?'Replay':'Resume';
      get('calculation').setAttribute('aria-live','polite');
    }
    function closeCandidates(){get('candidate-details').open=false;state.candidateIndex=null;}
    function animateProcess(target,restart=false){
      stopAnimation();closeCandidates();
      if(restart)state.process=0;
      if(reduced.matches){state.process=target;render();return;}
      const from=state.process,start=win.performance.now();
      const duration=Math.max(650,Math.abs(target-from)*1450);
      processResumeTarget=target;
      get('process-replay').textContent='Pause';
      get('calculation').setAttribute('aria-live','off');
      function tick(now){
        const t=Math.min(1,(now-start)/duration);
        state.process=Math.round((from+(target-from)*t)*1000)/1000;
        render();
        if(t<1)animation=win.requestAnimationFrame(tick);
        else {stopAnimation();render();}
      }
      render();animation=win.requestAnimationFrame(tick);
    }
    function activate(index){
      if(index===scene) return;
      stopAnimation(); closeCandidates(); scene=index; shell.scrollTop=0; state.process=0; state.continuation=0; state.growth=0;render(true);
    }
    function readingLine(){
      // One viewport-centered trigger for the full-width stage at every screen size.
      return win.innerHeight*.46;
    }
    function onScroll(){
      if(frame) return;
      frame=win.requestAnimationFrame(()=>{
        frame=0;
        get('progress').style.width=`${100*Math.min(1,Math.max(0,win.scrollY/(doc.documentElement.scrollHeight-win.innerHeight||1)))}%`;
        if(targetScene!==null) return;
        const moved=Math.abs(win.scrollY-scrollPosition)>.1, force=scrollSyncPending;
        scrollPosition=win.scrollY;scrollSyncPending=false;
        const line=readingLine();
        let nearest=0;
        steps.forEach((s,i)=>{if(s.getBoundingClientRect().top<=line) nearest=i;});
        const changed=nearest!==scene;
        activate(nearest);
        if((moved || changed || force) && scene>=0 && scene<=8 && scene!==6){
          stopAnimation();closeCandidates();
          const box=steps[scene].getBoundingClientRect();
          // Give the initial scatterplot time to arrive before revealing the mean.
          const hold=scene===0?.4:.08,duration=scene===0?.5:.77;
          const fraction=Math.max(0,Math.min(1,((line-box.top)/Math.max(1,box.height)-hold)/duration));
          if(scene>=7){
            const progress=Math.round(fraction*G.max(scene-7)*1000)/1000;
            state.growth=reduced.matches?Math.floor(progress):progress;
          }else if(scene===5){
            state.continuation=fraction;
            if(reduced.matches){
              const current=D.continuationState(fraction);
              state.continuation=D.continuationProgress(current.round-(current.amount>=1?0:1));
            }
          }else {
            const maximum=D.processMax(scene),process=Math.min(maximum,Math.round(fraction*maximum*1000)/1000);
            state.process=reduced.matches?Math.floor(process):process;
          }
          get('calculation').setAttribute('aria-live','off');
          render();
        }
      });
    }
    function go(index){
      index=Math.max(0,Math.min(steps.length-1,index));
      targetScene=index;win.clearTimeout(targetTimeout);activate(index);
      const offset=win.innerHeight*.18;
      const top=win.scrollY+steps[index].getBoundingClientRect().top-offset;
      win.scrollTo({top,behavior:reduced.matches?'instant':'smooth'});
      // Prevent an in-between section from replacing a button's target while scrolling.
      targetTimeout=win.setTimeout(()=>{targetScene=null;scrollPosition=win.scrollY;scrollSyncPending=false;onScroll();},reduced.matches?40:900);
    }
    get('previous').addEventListener('click',()=>go(scene-1));
    get('next').addEventListener('click',()=>go(scene+1));
    for(const [id,direction] of [['continuation-back',-1],['continuation-next',1]])get(id).addEventListener('click',()=>{
      stopAnimation();
      const current=D.continuationState(state.continuation);
      const complete=current.amount>=1;
      const round=direction>0?current.round+(complete?1:0):current.round-1;
      state.continuation=D.continuationProgress(Math.max(2,Math.min(8,round)));
      render();
    });
    function animateGrowth(target,restart=false){
      stopAnimation();
      if(restart)state.growth=0;
      if(reduced.matches){state.growth=target;render();return;}
      const from=state.growth,start=win.performance.now(),duration=Math.max(650,Math.abs(target-from)*2200);
      growthResumeTarget=target;
      get('growth-replay').textContent='Pause';
      get('calculation').setAttribute('aria-live','off');
      function tick(now){
        const t=Math.min(1,(now-start)/duration);
        state.growth=Math.round((from+(target-from)*t)*1000)/1000;
        render();
        if(t<1)animation=win.requestAnimationFrame(tick);
        else {stopAnimation();render();}
      }
      render();animation=win.requestAnimationFrame(tick);
    }
    get('growth-next').addEventListener('click',()=>animateGrowth(Math.min(G.max(scene-7),Math.floor(state.growth+1e-6)+1)));
    get('growth-back').addEventListener('click',()=>animateGrowth(Math.max(0,Math.ceil(state.growth-1e-6)-1)));
    get('growth-replay').addEventListener('click',()=>{
      if(animation){stopAnimation(true);render();}
      else if(growthResumeTarget!==null)animateGrowth(growthResumeTarget);
      else animateGrowth(G.max(scene-7),true);
    });
    get('process-next').addEventListener('click',()=>animateProcess(Math.min(D.processMax(scene),Math.floor(state.process+1e-6)+1)));
    get('process-back').addEventListener('click',()=>animateProcess(Math.max(0,Math.ceil(state.process-1e-6)-1)));
    get('process-replay').addEventListener('click',()=>{if(animation){stopAnimation(true);render();}else if(processResumeTarget!==null)animateProcess(processResumeTarget);else animateProcess(D.processMax(scene),true);});
    get('process-progress').addEventListener('input',event=>{stopAnimation();closeCandidates();state.process=Number(event.target.value);render();});
    get('candidate-details').addEventListener('toggle',()=>{
      if(scene!==1 && scene!==3)return;
      if(!get('candidate-details').open && state.candidateIndex===null)return;
      stopAnimation();
      if(get('candidate-details').open){
        const rows=B.candidates(scene===1?1:2);
        state.candidateIndex=rows.reduce((best,row,i)=>row.sse<rows[best].sse?i:best,0);
      }else state.candidateIndex=null;
      render();
    });
    get('candidate-cut').addEventListener('input',event=>{stopAnimation();state.candidateIndex=Number(event.target.value);render();});
    doc.addEventListener('keydown',event=>{
      if(event.target.closest('input,select,button,a,summary,details,[role="button"]'))return;
      if(event.key==='ArrowRight'){event.preventDefault();go(scene+1);}
      if(event.key==='ArrowLeft'){event.preventDefault();go(scene-1);}
    });
    win.addEventListener('scroll',onScroll,{passive:true});
    win.addEventListener('resize',()=>{scrollSyncPending=true;render();onScroll();},{passive:true});
    reduced.addEventListener('change',()=>{stopAnimation();scrollSyncPending=true;render();onScroll();});
    render();onScroll();
  }
  return {renderScene,readout,scenePresentation,init};
});
