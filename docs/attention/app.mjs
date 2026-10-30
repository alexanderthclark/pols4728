import {createSocksScenes} from './socks-scenes.mjs?v=attention-sequence-2';
import {createExtensionScenes} from './extension-scenes.mjs?v=attention-sequence-2';

const C={ink:'#242424',blue:'#234e70',rust:'#a54f32',teal:'#007f73',purple:'#7b4f9d',muted:'#606060',rule:'#dededb'};
const svg=document.querySelector('#diagram');
const reduced=matchMedia('(prefers-reduced-motion: reduce)');
const reducedMotion=()=>reduced.matches||new URLSearchParams(location.search).get('motion')==='reduce';
const clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,v));
const lerp=(a,b,t)=>a+(b-a)*t;
const ease=t=>(t=clamp(t))*t*(3-2*t);
const pct=v=>`${Math.round(v*100)}%`;
const num=v=>Math.abs(v)<.0005?'0':Number.isInteger(v)?String(v):v.toFixed(2);
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
const mathCache=new Map();
function tex(s){if(!mathCache.has(s))mathCache.set(s,katex.renderToString(s,{throwOnError:true,output:'htmlAndMathml',strict:'error'}));return mathCache.get(s)}
const m=(s,x,y,w=160,h=44,size=26,color=C.ink)=>`<foreignObject x="${x-w/2}" y="${y-h/2}" width="${w}" height="${h}"><div xmlns="http://www.w3.org/1999/xhtml" class="mathbox" style="font-size:${size}px;color:${color}">${tex(s)}</div></foreignObject>`;
const text=(s,x,y,size=22,color=C.ink,anchor='middle',extra='')=>`<text x="${x}" y="${y}" font-size="${size}" style="fill:${color}" text-anchor="${anchor}" ${extra}>${esc(s)}</text>`;
const line=(x1,y1,x2,y2,color=C.rule,width=1,extra='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}" ${extra}/>`;
const rect=(x,y,w,h,fill='#fff',stroke='none',rx=0,extra='')=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${fill}" stroke="${stroke}" rx="${rx}" ${extra}/>`;
const circle=(x,y,r,fill,extra='')=>`<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" ${extra}/>`;
const defs=`<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="none" stroke="${C.muted}" stroke-width="1.4"/></marker><pattern id="blocked" width="9" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" x2="0" y1="0" y2="9" stroke="#ddd" stroke-width="3"/></pattern></defs>`;

function renderMathNodes(root=document){root.querySelectorAll('[data-math]').forEach(el=>el.innerHTML=tex(el.dataset.math))}
renderMathNodes();
const drawing={m,text,line,rect,circle,C,ease,lerp,clamp,num};
const scenes=[...createSocksScenes(drawing),...createExtensionScenes(drawing)];
// The complete text remains available independently of visual scene changes.
document.querySelector('#transcript').innerHTML=scenes.map((s,i)=>`<section><h3>${i+1}. ${s.title}</h3><p>${s.caption}</p><div class="transcript-eq">${tex(s.equation)}</div><p>${s.detail}</p></section>`).join('');
renderMathNodes(document.querySelector('#transcript'));
const story=document.querySelector('#story');
story.style.height=`${(scenes.length+1)*100}svh`;
let last=-1,scheduled=false;
function render(){
 scheduled=false;
 const r=story.getBoundingClientRect(),range=r.height-innerHeight;
 const total=clamp(-r.top/range)*scenes.length;
 const index=Math.min(scenes.length-1,Math.floor(total));
 const local=clamp(total-index);
 const s=scenes[index];
 const t=reducedMotion()&&!s.discretePhases?1:clamp(local/.66);
 if(last!==index){
  document.querySelector('#chapter').textContent=s.part;
  document.querySelector('#counter').textContent=`${index+1} / ${scenes.length}`;
  document.querySelector('#scene-label').textContent=s.label;
  document.querySelector('#scene-title').textContent=s.title;
  document.querySelector('#caption').textContent=s.caption;
  document.querySelector('#equation').innerHTML=tex(s.equation);
  document.querySelector('#detail').innerHTML=s.detail;
  renderMathNodes(document.querySelector('#detail'));
  document.querySelector('#visual-caption').textContent=s.visual;
  svg.setAttribute('aria-label',`${s.title} ${s.caption} ${s.visual}`);
  last=index;
 }
 document.querySelector('#progress').style.width=`${clamp(total/scenes.length)*100}%`;
 svg.innerHTML=defs+s.draw(t);
 story.dataset.scene=index;story.dataset.phase=t.toFixed(3);
 const a=reducedMotion()?1:Math.min(1,.72+local*4);
 document.querySelector('#narration').style.opacity=a;
}
function schedule(){if(!scheduled){scheduled=true;requestAnimationFrame(render)}}
addEventListener('scroll',schedule,{passive:true});addEventListener('resize',schedule);reduced.addEventListener('change',schedule);
function revealReading(){if(location.hash==='#reading')document.querySelector('#reading').open=true;}
addEventListener('hashchange',revealReading);revealReading();
// Hash jumps and native keyboard scrolling follow the same reversible sequence.
render();
