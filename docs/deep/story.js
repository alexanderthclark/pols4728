import { evaluateNetwork, surfaceValue } from './model.mjs';
import { LinkedFigures, renderNetwork, modeLabels } from './views.js';

const $=selector=>document.querySelector(selector);
const steps=[...document.querySelectorAll('.step')];
const stage=$('.visual-stage');
const reducedMotion=matchMedia('(prefers-reduced-motion: reduce)');
const mobile=matchMedia('(max-width: 820px)');
const formatter=new Intl.NumberFormat('en-US',{maximumFractionDigits:2});
const number=value=>formatter.format(Math.abs(value)<1e-9?0:value).replace('-','−');
const names=[
  'The surface we will build','Three first-layer features','The shallow readout',
  'The next weighted sum','A local feature','Reshape a local feature',
  'The combined output','Explore the same network',
];
const subtitles=[
  'Two inputs · one numeric response','One hidden layer · three fixed ReLU units',
  'One hidden layer · an affine output','Each new unit receives all three activations',
  'Two hidden layers · inspect the first new unit','Two hidden layers · inspect the second new unit',
  'Two hidden layers · three units in each layer','First-layer features stay fixed',
];
const notes=[
  'Height is response y. The horizontal axes are the two inputs.',
  'Solid crease and dashed activation boundary coincide for each ramp.',
  'Dashed: zero crossing. The surface continues below zero outside it.',
  'The first new unit’s weighted sum, before applying its ReLU.',
  'Dashed: activation boundary. The response outside it is exactly zero.',
  'A different incoming weight reshapes the footprint of the new feature.',
  'Solid lines mark changes in slope in the combined output.',
  'Select a point in the input plane or use the input controls.',
];
const state={scene:0,pinch:2,point:[.3,.2],feature:'h11',explorer:'combined'};
let navigationTarget=null,scrollFrame=0;

// Keep controls beside the argument they change. In the explorer, the same
// weight control is moved rather than duplicated, preserving one parameter.
const weightControls=$('#weight-controls');
const surfaceControls=$('#surface-controls');
const probeControls=$('#probe-controls');
const reshapeContent=$('#reshape-feature .step-content');
const exploreContent=$('#explore .step-content');
reshapeContent.append(weightControls);
exploreContent.querySelector('.step-prose').prepend(surfaceControls,probeControls);
weightControls.hidden=false;surfaceControls.hidden=false;probeControls.hidden=false;

const dots=steps.map((step,index)=>{
  const button=document.createElement('button');
  button.type='button';button.setAttribute('aria-label',`Frame ${index+1}: ${step.querySelector('h1,h2').textContent}`);
  button.addEventListener('click',()=>goTo(index));
  $('#progress-dots').append(button);
  return button;
});
const figures=new LinkedFigures($('#surface'),$('#input-plane'),point=>setPoint(point,true));

function currentMode() {
  return ['combined',state.feature,'shallow','preactivation','unit1','unit2','combined',state.explorer][state.scene];
}
function updateReadout() {
  const mode=currentMode();
  const values=evaluateNetwork(...state.point,{pinch:state.pinch});
  $('#input-values').textContent=`(${state.point.map(number).join(', ')})`;
  $('#first-values').textContent=`(${values.h1.map(number).join(', ')})`;
  $('#response-label').textContent=mode==='preactivation'?'Before ReLU':`Response ${modeLabels[mode]}`;
  $('#response-value').textContent=number(values[mode]);
  const hasSecond=state.scene>=3 && !(state.scene===7 && mode==='shallow');
  renderNetwork($('#network'),{scene:state.scene,mode,pinch:state.pinch,point:state.point});
  $('#network').dataset.hiddenLayers=hasSecond?'2':'1';
  $('#probe-x1').value=state.point[0];$('#probe-x2').value=state.point[1];
  $('#probe-x1-value').textContent=number(state.point[0]);$('#probe-x2-value').textContent=number(state.point[1]);
}
function updateFigures(animate=false) {
  const mode=currentMode();
  figures.update({scene:state.scene,mode,pinch:state.pinch,point:state.point,showProbe:state.scene!==0,cover:state.scene===0},{animate});
  updateReadout();
  $('#figure-note').textContent=state.pinch<.625 && (mode==='unit2'||mode==='combined')
    ? 'At this weight, the positive response extends beyond the displayed input window.'
    : notes[state.scene];
}
function updateWeight() {
  const weight=-state.pinch;
  $('#pinch-weight').value=weight;
  $('#pinch-value').textContent=number(weight);
  document.querySelectorAll('[data-pinch-formula]').forEach(element=>{
    element.textContent=number(state.pinch);
    element.closest('math').setAttribute('aria-label',`h two comma two equals ReLU of one minus ${state.pinch} times h one comma one minus h one comma two minus h one comma three`);
  });
}
function setPoint(point,announce=false) {
  state.point=point.map(value=>Math.max(-1.6,Math.min(1.6,Math.round(value*20)/20)));
  updateFigures();
  if(announce) $('#interaction-status').textContent=`Input (${state.point.map(number).join(', ')}). ${modeLabels[currentMode()]} is ${number(surfaceValue(currentMode(),...state.point,{pinch:state.pinch}))}.`;
}
function setScene(scene,{animate=true}={}) {
  if(scene<0 || scene>=steps.length) return;
  const changed=state.scene!==scene;
  state.scene=scene;
  stage.classList.toggle('mode-cover',scene===0);
  stage.classList.toggle('mode-feature',scene===1);
  stage.classList.toggle('mode-local',scene===4||scene===5);
  stage.classList.toggle('mode-combined',scene===6);
  stage.classList.toggle('exploring',scene===7);
  $('#stage-title').textContent=names[scene];$('#stage-subtitle').textContent=subtitles[scene];
  $('#stage-count').textContent=`${scene+1} / ${steps.length}`;
  $('#previous').disabled=scene===0;$('#next').disabled=scene===steps.length-1;
  $('#feature-controls').hidden=scene!==1;
  const destination=scene===7?exploreContent.querySelector('.step-prose'):reshapeContent;
  if(weightControls.parentElement!==destination) {
    if(scene===7) destination.insertBefore(weightControls,probeControls);
    else destination.append(weightControls);
  }
  dots.forEach((button,index)=>{
    button.classList.toggle('is-active',index===scene);
    button.setAttribute('aria-current',index===scene?'step':'false');
  });
  steps.forEach((step,index)=>{
    step.classList.toggle('is-active',index===scene);step.classList.toggle('is-past',index<scene);
  });
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
  history.replaceState(null,'',`#${steps[scene].id}`);
  window.scrollTo({top,behavior});
}
function inspectScroll() {
  scrollFrame=0;
  if(navigationTarget) {
    if(Math.abs(window.scrollY-navigationTarget.top)<3 || performance.now()>navigationTarget.until) navigationTarget=null;
    else return;
  }
  const offset=readingOffset();
  const line=offset+(window.innerHeight-offset)*.45;
  const best=steps.map((step,index)=>{
    const bounds=step.getBoundingClientRect();
    const distance=bounds.top<=line&&bounds.bottom>=line?0:Math.min(Math.abs(bounds.top-line),Math.abs(bounds.bottom-line));
    return {index,distance};
  }).sort((a,b)=>a.distance-b.distance)[0].index;
  if(best!==state.scene) setScene(best);
}
function scheduleScroll() { if(!scrollFrame) scrollFrame=requestAnimationFrame(inspectScroll); }

$('#previous').addEventListener('click',()=>goTo(state.scene-1));
$('#next').addEventListener('click',()=>goTo(state.scene+1));
$('#feature-select').addEventListener('change',event=>{
  state.feature=event.target.value;updateFigures(true);
});
$('#surface-select').addEventListener('change',event=>{
  state.explorer=event.target.value;updateFigures(true);
});
$('#pinch-weight').addEventListener('input',event=>{
  state.pinch=-Number(event.target.value);updateWeight();updateFigures();
});
$('#pinch-weight').addEventListener('change',()=>{
  $('#interaction-status').textContent=`Weight on the first feature is ${number(-state.pinch)}. The first layer has not changed.`;
});
for(const [selector,index] of [['#probe-x1',0],['#probe-x2',1]]) {
  $(selector).addEventListener('input',event=>{
    const point=[...state.point];point[index]=Number(event.target.value);setPoint(point);
  });
  $(selector).addEventListener('change',()=>setPoint(state.point,true));
}
window.addEventListener('scroll',scheduleScroll,{passive:true});
window.addEventListener('resize',()=>{updateFigures();scheduleScroll();},{passive:true});
document.addEventListener('keydown',event=>{
  if(event.altKey||event.ctrlKey||event.metaKey||event.shiftKey||event.target.closest('input,select,textarea,summary')) return;
  if(event.key==='ArrowRight') {event.preventDefault();goTo(state.scene+1);}
  if(event.key==='ArrowLeft') {event.preventDefault();goTo(state.scene-1);}
});
history.scrollRestoration='manual';
updateWeight();setScene(0,{animate:false});
const initial=steps.findIndex(step=>step.id===decodeURIComponent(location.hash.slice(1)));
if(initial>0) {
  const openFrame=()=>goTo(initial,{animate:false,behavior:'instant'});
  if(document.readyState==='complete') openFrame();
  else window.addEventListener('load',openFrame,{once:true});
}
