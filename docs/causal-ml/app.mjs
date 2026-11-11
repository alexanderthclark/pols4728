import {lessons} from './lessons.mjs';
import {draw} from './visuals.mjs';
import {clamp} from './model.mjs';
const lessonKey=document.body.dataset.lesson,lesson=lessons[lessonKey];
const $=id=>document.getElementById(id),story=$('story'),diagram=$('diagram'),media=matchMedia('(prefers-reduced-motion:reduce)'),query=new URLSearchParams(location.search);
const isReduced=()=>media.matches||query.get('motion')==='reduce';
const math=(s,displayMode=false)=>globalThis.katex.renderToString(s,{displayMode,throwOnError:true,output:'htmlAndMathml'});
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;');
const prose=s=>s.split(/(\$[^$]+\$)/g).map(x=>x.startsWith('$')?math(x.slice(1,-1)):esc(x)).join('');
let pending=false,lastIndex=-1,lastMobile=null,lastSignature='';
function metrics(){const top=story.getBoundingClientRect().top+scrollY-parseFloat(getComputedStyle($('stage')).top);return{top,range:story.offsetHeight-$('stage').offsetHeight};}
function jump(index,fraction=.17){const{top,range}=metrics();scrollTo({top:top+range*(index+fraction)/lesson.scenes.length,behavior:'instant'});}
function update(){pending=false;const{top,range}=metrics(),overall=clamp((scrollY-top)/range),position=Math.min(lesson.scenes.length-.000001,overall*lesson.scenes.length),index=Math.floor(position),frac=position-index,scene=lesson.scenes[index],mobile=innerWidth<=760,p=isReduced()?1:clamp((frac-.08)/.65),signature=`${index}|${Math.round(p*1000)}|${mobile}`;
 if(index!==lastIndex||mobile!==lastMobile){$('scene-title').textContent=scene.title;$('caption').innerHTML=prose(scene.body);$('chapter').textContent=scene.chapter;$('counter').textContent=`${String(index+1).padStart(2,'0')} / ${lesson.scenes.length}`;$('equation').innerHTML=math(mobile?(scene.mobileEquation||scene.equation):scene.equation,true);$('note').textContent=scene.note;$('scene-reading').href=`reading.html#scene-${index+1}`;diagram.setAttribute('aria-label',`${scene.title}. ${scene.body.replace(/\$/g,'')}`);lastIndex=index;lastMobile=mobile;}
 if(signature!==lastSignature){const result=draw(scene.kind,p,mobile,lessonKey);diagram.setAttribute('viewBox',result.viewBox);diagram.innerHTML=result.svg;lastSignature=signature;}
 $('progress').style.width=`${overall*100}%`;$('stage').dataset.scene=String(index);$('stage').dataset.progress=p.toFixed(4);$('stage').dataset.motion=isReduced()?'reduced':'scroll';
 const eq=$('equation').firstElementChild;if(eq){const ratio=$('equation').clientWidth/eq.scrollWidth;if(ratio<.99)eq.style.fontSize=`${Math.max(.7,ratio)}em`;}
}
function schedule(){if(!pending){pending=true;requestAnimationFrame(update);}}
story.style.height=`calc(${lesson.scenes.length*112}svh + 100svh)`;
document.querySelectorAll('[data-jump]').forEach(a=>a.addEventListener('click',event=>{event.preventDefault();const index=Number(a.dataset.jump);jump(index);history.replaceState(null,'',`#step-${index+1}`);}));
addEventListener('scroll',schedule,{passive:true});addEventListener('resize',()=>{lastSignature='';schedule();});media.addEventListener('change',schedule);
function hashJump(){const match=location.hash.match(/^#step-(\d+)$/);if(match)jump(clamp(Number(match[1])-1,0,lesson.scenes.length-1));}
addEventListener('hashchange',hashJump);
async function start(){await document.fonts.ready;hashJump();update();}
start();
