import { evaluateNetwork, foldedValue, INPUT_DOMAIN, DEFAULT_THRESHOLD } from './model.mjs';
import { curveGeometry, foldedCurveGeometry } from './geometry.mjs';
import { LinkedFigures, renderNetwork, modeLabels } from './views.js';

const $=selector=>document.querySelector(selector);
const steps=[...document.querySelectorAll('.step')];
const stage=$('.visual-stage');
const reducedMotion=matchMedia('(prefers-reduced-motion: reduce)');
const mobile=matchMedia('(max-width: 820px)');
const formatter=new Intl.NumberFormat('en-US',{maximumFractionDigits:2});
const number=value=>formatter.format(Math.abs(value)<1e-9?0:Math.sign(value)*Math.round((Math.abs(value)+1e-12)*100)/100).replace('-','−');
const names=[
  'One ten-joint curve','Three first-layer ramps','The fold q(x)',
  'Fold the input line','One joint, three locations','Reuse the whole pattern',
  'The same curve, fewer parameters','Explore the linked joints',
];
const subtitles=[
  '22 deep parameters · at least 31 shallow parameters','One hidden layer · one joint per ReLU',
  'Slope +1, then −1, then +1 on 0 ≤ x ≤ 3','The middle interval runs backward',
  'h₂,₂ = a[q − τ] · one threshold in the folded coordinate',
  'Three downstream joints · each reused on three intervals',
  'Two hidden layers · three units in each','Select an input or move the middle threshold',
];
const notes=[
  'Nine copied joints (rust) and one surviving turn of the fold (blue).',
  'Each ramp changes slope once, at x = 0, 1, or 2.',
  'The same q runs from 0 to 1 three times. The middle pass is reversed.',
  'Solid rows represent input intervals. Dotted guides link two drawings of the same endpoint at x = 1 or 2.',
  '',
  'Each dashed threshold cuts all three passes: nine copied joint locations.',
  '10 true joints · 22 dense parameters. A shallow network needs at least 10 units and 31 parameters for this curve.',
  '',
];
const state={scene:0,threshold:DEFAULT_THRESHOLD,x:.25,explorer:'combined'};
let navigationTarget=null,scrollFrame=0;
const thresholdControls=$('#threshold-controls');
const surfaceControls=$('#surface-controls');
const probeControls=$('#probe-controls');
const reuseContent=$('#reuse-joint .step-content');
const foldContent=$('#fold-input .step-content');
const exploreContent=$('#explore .step-prose');
reuseContent.append(thresholdControls);
exploreContent.prepend(surfaceControls,probeControls);
thresholdControls.hidden=false;surfaceControls.hidden=false;probeControls.hidden=false;
const dots=steps.map((step,index)=>{
  const button=document.createElement('button');
  button.type='button';button.setAttribute('aria-label','Frame '+(index+1)+': '+step.querySelector('h1,h2').textContent);
  button.addEventListener('click',()=>goTo(index));
  $('#progress-dots').append(button);
  return button;
});
const figures=new LinkedFigures($('#surface'),$('#feature-plot'),x=>setPoint(x,true));
function currentMode() {
  return ['combined','h11','fold','fold','unit2','combined','combined',state.explorer][state.scene];
}
function modeValue(mode,values) {return mode==='downstream'?foldedValue(values.fold,{threshold:state.threshold}):values[mode];}
function updateFigures(animate=false) {
  const mode=currentMode();
  $('.feature-view').hidden=state.scene<5 || (state.scene===7 && mode!=='combined');
  figures.update({scene:state.scene,mode,threshold:state.threshold,x:state.x,showProbe:state.scene!==0,cover:state.scene===0},{animate});
  const values=evaluateNetwork(state.x,{threshold:state.threshold});
  $('#input-values').textContent=number(state.x);
  $('#fold-values').previousElementSibling.textContent=state.scene===1?'Activations h₁':'Folded q';
  $('#fold-values').textContent=state.scene===1?'('+values.h1.map(number).join(', ')+')':number(values.fold);
  $('#response-label').textContent=state.scene===1?'First feature h₁,₁':mode==='fold'?'Fold q(x)':'Response '+modeLabels[mode];
  $('#response-value').textContent=number(modeValue(mode,values));
  renderNetwork($('#network'),{scene:state.scene,mode,threshold:state.threshold,x:state.x});
  $('#probe-x').value=state.x;$('#probe-x-value').textContent=number(state.x);
  $('#threshold').value=state.threshold;$('#threshold-value').textContent=number(state.threshold);
  let note=notes[state.scene];
  if(state.scene===4) note='The joint at q = '+number(state.threshold)+' appears at x = '+[state.threshold,2-state.threshold,2+state.threshold].map(number).join(', ')+'. Move τ to move all three.';
  if(state.scene===7) {
    if(mode==='combined') note=curveGeometry(mode,{threshold:state.threshold}).bends.length+' true joints. Moving τ links three of them; the other thresholds and the fold stay fixed.';
    else if(mode==='downstream') note=foldedCurveGeometry({threshold:state.threshold}).bends.length+' joints in g(q). The horizontal axis here is q; the input control still selects x.';
    else if(mode==='fold') note='This fold stays fixed when τ moves. Its turns are at x = 1 and 2.';
    else note='One threshold, three copied joints at x = '+[state.threshold,2-state.threshold,2+state.threshold].map(number).join(', ')+'.';
  }
  $('#figure-note').textContent=note;
}
function setPoint(x,announce=false) {
  state.x=Math.max(INPUT_DOMAIN[0],Math.min(INPUT_DOMAIN[1],Math.round(x*100)/100));
  updateFigures();
  if(announce) {
    const values=evaluateNetwork(state.x,{threshold:state.threshold});
    $('#interaction-status').textContent='Input '+number(state.x)+', folded coordinate '+number(values.fold)+'. '+modeLabels[currentMode()]+' is '+number(modeValue(currentMode(),values))+'.';
  }
}
function setScene(scene,{animate=true}={}) {
  if(scene<0 || scene>=steps.length) return;
  const changed=state.scene!==scene;
  state.scene=scene;
  stage.classList.toggle('mode-cover',scene===0);
  stage.classList.toggle('mode-feature',scene===1);
  stage.classList.toggle('mode-combined',scene===6);
  stage.classList.toggle('mode-fold',scene>=1&&scene<=3);
  stage.classList.toggle('mode-foldmap',scene===3||scene===5);
  $('#response-value').parentElement.hidden=scene>=1&&scene<=3;
  $('#probe-x').min=scene===3?'0':String(INPUT_DOMAIN[0]);
  $('#probe-x').max=scene===3?'3':String(INPUT_DOMAIN[1]);
  if(scene===3||scene===5) state.x=Math.max(0,Math.min(3,state.x));
  $('#stage-title').textContent=names[scene];$('#stage-subtitle').textContent=subtitles[scene];
  $('#stage-count').textContent=(scene+1)+' / '+steps.length;
  $('#previous').disabled=scene===0;$('#next').disabled=scene===steps.length-1;
  const probeDestination=scene===7?exploreContent:foldContent;
  if(probeControls.parentElement!==probeDestination) probeDestination.append(probeControls);
  const destination=scene===7?exploreContent:reuseContent;
  if(thresholdControls.parentElement!==destination) {
    if(scene===7) destination.insertBefore(thresholdControls,probeControls);
    else destination.append(thresholdControls);
  }
  dots.forEach((button,index)=>{
    button.classList.toggle('is-active',index===scene);
    button.setAttribute('aria-current',index===scene?'step':'false');
  });
  steps.forEach((step,index)=>step.classList.toggle('is-active',index===scene));
  updateFigures(animate&&changed);
}
function readingOffset() {
  const mastheadBottom=$('.masthead').getBoundingClientRect().bottom;
  return mobile.matches?mastheadBottom+stage.offsetHeight+12:mastheadBottom;
}
function goTo(scene,{animate=true,behavior=reducedMotion.matches?'instant':'smooth'}={}) {
  if(scene<0 || scene>=steps.length) return;
  setScene(scene,{animate});
  const top=Math.max(0,steps[scene].getBoundingClientRect().top+window.scrollY-readingOffset());
  navigationTarget={top,until:performance.now()+1400};
  history.replaceState(null,'','#'+steps[scene].id);
  window.scrollTo({top,behavior});
}
function inspectScroll() {
  scrollFrame=0;
  if(navigationTarget) {
    if(Math.abs(window.scrollY-navigationTarget.top)<3 || performance.now()>navigationTarget.until) navigationTarget=null;
    else return;
  }
  const offset=readingOffset(),line=offset+(window.innerHeight-offset)*.45;
  const best=steps.map((step,index)=>{
    const bounds=step.getBoundingClientRect();
    const distance=bounds.top<=line&&bounds.bottom>=line?0:Math.min(Math.abs(bounds.top-line),Math.abs(bounds.bottom-line));
    return {index,distance};
  }).sort((a,b)=>a.distance-b.distance)[0].index;
  if(best!==state.scene) setScene(best);
}
function scheduleScroll() {if(!scrollFrame) scrollFrame=requestAnimationFrame(inspectScroll);}
$('#previous').addEventListener('click',()=>goTo(state.scene-1));
$('#next').addEventListener('click',()=>goTo(state.scene+1));
$('#replay-fold').addEventListener('click',()=>figures.replayFold());
$('#surface-select').addEventListener('change',event=>{state.explorer=event.target.value;updateFigures(true);});
$('#threshold').addEventListener('input',event=>{state.threshold=Number(event.target.value);updateFigures();});
$('#threshold').addEventListener('change',()=>{
  $('#interaction-status').textContent='Threshold '+number(state.threshold)+'. Copied joints at '+[state.threshold,2-state.threshold,2+state.threshold].map(number).join(', ')+'.';
});
$('#probe-x').addEventListener('input',event=>setPoint(Number(event.target.value)));
$('#probe-x').addEventListener('change',()=>setPoint(state.x,true));
window.addEventListener('scroll',scheduleScroll,{passive:true});
window.addEventListener('resize',()=>{updateFigures();scheduleScroll();},{passive:true});
document.addEventListener('keydown',event=>{
  if(event.altKey||event.ctrlKey||event.metaKey||event.shiftKey||event.target.closest('input,select,textarea,summary')) return;
  if(event.key==='ArrowRight') {event.preventDefault();goTo(state.scene+1);}
  if(event.key==='ArrowLeft') {event.preventDefault();goTo(state.scene-1);}
});
history.scrollRestoration='manual';
setScene(0,{animate:false});
const initial=steps.findIndex(step=>step.id===decodeURIComponent(location.hash.slice(1)));
if(initial>0) {
  const openFrame=()=>goTo(initial,{animate:false,behavior:'instant'});
  if(document.readyState==='complete') openFrame();
  else window.addEventListener('load',openFrame,{once:true});
}
