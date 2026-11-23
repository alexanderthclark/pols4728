import {renderBearComparison} from './bear-comparison.mjs?v=20261123-noise-only';
import {renderAtus} from './atus.mjs';
import {renderBear,renderBearMobile} from './bear.mjs';
import {renderDgp} from './dgp.mjs';
import {renderMechanism} from './mechanism.mjs';
import {renderNature,renderNatureMobile} from './nature.mjs';
import {scenes,sources} from './scenes.mjs?v=20261123-noise-only';
import {renderVisual,renderMobile} from './visuals.mjs';
import * as bayes from './bayes.mjs';
const $=id=>document.getElementById(id), story=$('story'),diagram=$('diagram');
const media=matchMedia('(prefers-reduced-motion: reduce)');
const reduced=()=>media.matches||new URLSearchParams(location.search).get('motion')==='reduce';
const clamp=x=>Math.max(0,Math.min(1,x));
$('scene-count').textContent=`${scenes.length} scenes · about 15 minutes`;
story.style.height=`${scenes.length*125+10}svh`;
let last=-1,pending=false;
function jump(index){window.scrollTo({top:story.offsetTop+(index+.76)/scenes.length*(story.offsetHeight-innerHeight),behavior:'instant'});}
const chapters=[];scenes.forEach((s,i)=>{if(!chapters.some(c=>c.name===s.chapter))chapters.push({name:s.chapter,index:i});});
for(const c of chapters){const a=document.createElement('a');a.href=`#scene-${c.index+1}`;a.textContent=c.name;a.addEventListener('click',e=>{e.preventDefault();jump(c.index);history.replaceState(null,'',a.hash);});$('route').append(a);}
document.querySelector('.begin').addEventListener('click',e=>{e.preventDefault();jump(0);history.replaceState(null,'','#scene-1');});
for(const source of sources){const li=document.createElement('li'),a=document.createElement('a'),span=document.createElement('span');a.href=source.url;a.textContent=source.title;span.className='source-note';span.textContent=source.note;li.append(a,span);$('source-list').append(li);}
function render(){pending=false;const overall=clamp((scrollY-story.offsetTop)/(story.offsetHeight-innerHeight));const position=Math.min(scenes.length-.00001,overall*scenes.length),index=Math.floor(position),scene=scenes[index];const raw=clamp((position-index-.06)/.62),p=reduced()?1:raw*raw*(3-2*raw),mobile=innerWidth<=760;
if(index!==last){$('scene-title').textContent=scene.title;$('caption').textContent=scene.body;$('chapter').textContent=scene.chapter;$('counter').textContent=`${index+1} / ${scenes.length}`;$('note').textContent=scene.note;$('reading-step').href=`reading.html#scene-${index+1}`;$('equation').innerHTML=globalThis.katex.renderToString(scene.equation,{displayMode:true,throwOnError:true,output:'htmlAndMathml'});diagram.setAttribute('aria-label',`${scene.title} ${scene.body} ${scene.note}`);last=index;}
diagram.setAttribute('viewBox',mobile?'0 0 400 350':'0 0 900 460');const visualIndex=scene.visualIndex;
diagram.innerHTML=scene.kind==='bearComparison'?renderBearComparison(scene.comparisonStage,p,mobile):scene.kind==='atus'?renderAtus(scene.atusStage,p,mobile):scene.kind==='nature'?(mobile?renderNatureMobile(scene.natureStage,p):renderNature(scene.natureStage,p)):scene.kind==='mechanism'?renderMechanism(scene.mechanismStage,p,mobile):scene.kind==='bayes'?(mobile?bayes.renderBayesMobile(scene.bayesStage,p):bayes.renderBayes(scene.bayesStage,p)):scene.kind==='dgp'?renderDgp(p,mobile):scene.kind==='bear'?(mobile?renderBearMobile(scene.bearStage,p):renderBear(scene.bearStage,p)):(mobile?renderMobile(visualIndex,p):renderVisual(visualIndex,p));diagram.dataset.scene=String(index+1);diagram.dataset.progress=p.toFixed(4);$('progress').style.width=`${overall*100}%`;document.documentElement.dataset.motion=reduced()?'reduced':'scroll';}
function schedule(){if(!pending){pending=true;requestAnimationFrame(render);}}
addEventListener('scroll',schedule,{passive:true});addEventListener('resize',schedule);media.addEventListener('change',schedule);
function hashJump(){const n=Number(location.hash.replace('#scene-',''));if(n>=1&&n<=scenes.length)requestAnimationFrame(()=>jump(n-1));}
addEventListener('hashchange',hashJump);document.fonts.ready.then(()=>{hashJump();schedule();});render();
