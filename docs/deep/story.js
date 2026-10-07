import { evaluateNetwork, surfaceValue, INPUT_DOMAIN } from './model.mjs';
import { curveGeometry } from './geometry.mjs';
import { LinkedFigures, renderNetwork, modeLabels } from './views.js';

const $=selector=>document.querySelector(selector);
const steps=[...document.querySelectorAll('.step')];
const stage=$('.visual-stage');
const reducedMotion=matchMedia('(prefers-reduced-motion: reduce)');
const mobile=matchMedia('(max-width: 820px)');
const formatter=new Intl.NumberFormat('en-US',{maximumFractionDigits:2});
const number=value=>formatter.format(Math.abs(value)<1e-9?0:Math.sign(value)*Math.round((Math.abs(value)+1e-12)*100)/100).replace('-','−');
const names=[
  'The curve we will build','Three fixed first-layer features','The shallow readout',
  'Before the new ReLU','Three crossings become new bends','Move the new bends',
  'The combined output','Explore the same network',
];
const subtitles=[
  'One input x · one numeric response y','One hidden layer · hinges at 0, 1, and 2',
  'Weights change slopes between the fixed hinges','The first new unit receives all three features',
  'First-layer hinges stay fixed','The threshold changes only the second new unit’s bias',
  'Two hidden layers · three units in each','Select an input on the curve or use the control',
];
const notes=[
  'Height is response y. The horizontal axis is the input x.',
  'Each ramp has one hinge. Together they define four linear intervals.',
  'Gray guides mark the first-layer hinges. Crossing zero does not change the slope.',
  'Open rust circles mark zero crossings at 0.5, 1.5, and 2.5.',
  'Rust dots are new bends; the faint dashed curve is the sum before ReLU.',
  '', '', 'Gray guides remain at the first layer’s hinges.',
];
const state={scene:0,threshold:.3,x:.75,explorer:'combined'};
let navigationTarget=null,scrollFrame=0;

const thresholdControls=$('#threshold-controls');
const surfaceControls=$('#surface-controls');
const probeControls=$('#probe-controls');
const moveContent=$('#move-bends .step-content');
const exploreContent=$('#explore .step-prose');
moveContent.append(thresholdControls);
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
  return ['combined','h11','shallow','preactivation','unit1','unit2','combined',state.explorer][state.scene];
}
function updateFigures(animate=false) {
  const mode=currentMode();
  figures.update({scene:state.scene,mode,threshold:state.threshold,x:state.x,showProbe:state.scene!==0,cover:state.scene===0},{animate});
  const values=evaluateNetwork(state.x,{threshold:state.threshold});
  $('#input-values').textContent=number(state.x);
  $('#first-values').textContent='('+values.h1.map(number).join(', ')+')';
  $('#response-label').textContent=state.scene===1?'First feature h₁,₁':mode==='preactivation'?'Before ReLU q₁':'Response '+modeLabels[mode];
  $('#response-value').textContent=number(values[mode]);
  renderNetwork($('#network'),{scene:state.scene,mode,threshold:state.threshold,x:state.x});
  $('#probe-x').value=state.x;
  $('#probe-x-value').textContent=number(state.x);
  $('#threshold').value=state.threshold;
  $('#threshold-value').textContent=number(state.threshold);
  const geometry=curveGeometry(mode,{threshold:state.threshold});
  let note=notes[state.scene];
  if(state.scene===5) note='New bends at '+geometry.secondPreactivationRoots[1].map(number).join(', ')+'. First-layer hinges have not moved.';
  if(state.scene===6 || (state.scene===7 && mode==='combined')) note=geometry.bends.length+' bends · '+(geometry.bends.length+1)+' linear intervals in the displayed window.';
  if(state.scene===7 && mode==='shallow') note='The threshold changes only the second hidden layer. This shallow readout stays fixed.';
  $('#figure-note').textContent=note;
}
function setPoint(x,announce=false) {
  state.x=Math.max(INPUT_DOMAIN[0],Math.min(INPUT_DOMAIN[1],Math.round(x*100)/100));
  updateFigures();
  if(announce) $('#interaction-status').textContent='Input '+number(state.x)+'. '+modeLabels[currentMode()]+' is '+number(surfaceValue(currentMode(),state.x,{threshold:state.threshold}))+'.';
}
function setScene(scene,{animate=true}={}) {
  if(scene<0 || scene>=steps.length) return;
  const changed=state.scene!==scene;
  state.scene=scene;
  stage.classList.toggle('mode-cover',scene===0);
  stage.classList.toggle('mode-feature',scene===1);
  stage.classList.toggle('mode-combined',scene===6);
  $('#stage-title').textContent=names[scene];$('#stage-subtitle').textContent=subtitles[scene];
  $('#stage-count').textContent=(scene+1)+' / '+steps.length;
  $('#previous').disabled=scene===0;$('#next').disabled=scene===steps.length-1;
  // The three supporting curves appear when the final output recombines them.
  // Earlier frames concentrate on one curve and its crossings.
  $('.feature-view').hidden=scene<6;
  const destination=scene===7?exploreContent:moveContent;
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
$('#surface-select').addEventListener('change',event=>{
  state.explorer=event.target.value;updateFigures(true);
});
$('#threshold').addEventListener('input',event=>{
  state.threshold=Number(event.target.value);updateFigures();
});
$('#threshold').addEventListener('change',()=>{
  $('#interaction-status').textContent='Threshold '+number(state.threshold)+'. The first layer has not changed.';
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
