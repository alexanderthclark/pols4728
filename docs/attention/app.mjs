import {createIntuitionScenes} from './intuition-scenes.mjs?v=attention-dimensions-17';
import {createTrainedScenes} from './trained-scenes.mjs?v=attention-trained-model-6';
import {buildExplorer} from './trained-explorer.mjs?v=attention-trained-model-5';
import {createSocksScenes} from './socks-scenes.mjs?v=attention-notation-8';
import {createExtensionScenes} from './extension-scenes.mjs?v=attention-notation-8';
import {createInterpretationScenes} from './interpretation-scenes.mjs?v=attention-notation-8';
import {createSentenceScenes} from './sentence-scenes.mjs?v=attention-intuition-13';

const C={ink:'#242424',blue:'#234e70',rust:'#a54f32',teal:'#007f73',purple:'#7b4f9d',muted:'#606060',rule:'#dededb'};
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
const intuitionScenes=createIntuitionScenes(drawing);
const coreScenes=createSentenceScenes(drawing);
const numericalScenes=createInterpretationScenes(drawing);
const mechanicsScenes=[...createSocksScenes(drawing),...createExtensionScenes(drawing)];
// The complete text remains available independently of visual scene changes.
const transcript=scenes=>scenes.map((s,i)=>`<section>${s.title?`<h3>${i+1}. ${s.title}</h3>`:""}${s.caption?`<p>${s.caption}</p>`:""}${(s.quotes||[]).map(q=>`<blockquote><p>${esc(q.text)}</p><cite>${esc(q.reading)}</cite></blockquote>`).join('')}<div class="transcript-eq">${tex(s.equation)}</div><p>${s.detail}</p></section>`).join('');
document.querySelector('#transcript').innerHTML='<h3>One token, different contexts</h3>'+transcript(intuitionScenes)+'<h3>Sentiment, the residual stream, and attention</h3>'+transcript(coreScenes)+'<h3>A numerical attention classifier</h3>'+transcript(numericalScenes)+'<h3>The earlier Q/K/V calculation</h3>'+transcript(mechanicsScenes);
renderMathNodes(document.querySelector('#transcript'));
const quoteMarkup=quotes=>(quotes||[]).map(q=>`<blockquote><p>${esc(q.text)}</p><cite>${esc(q.reading)}</cite></blockquote>`).join('');
function makeStory(id,prefix,scenes){
 const story=document.querySelector('#'+id);
 const field=name=>document.getElementById(prefix+name);
 const svg=field('diagram');
 story.style.height=`${(scenes.length+1)*100}svh`;
 let last=-1;
 return function renderStory(){
 if(!story.getClientRects().length)return;
 const r=story.getBoundingClientRect(),range=r.height-innerHeight;
 const total=clamp(-r.top/range)*scenes.length;
 const index=Math.min(scenes.length-1,Math.floor(total));
 const local=clamp(total-index);
 const s=scenes[index];
 const t=reducedMotion()&&!s.discretePhases?1:clamp(local/.66);
 if(last!==index){
  field('chapter').textContent=s.part;
  field('counter').textContent=`${index+1} / ${scenes.length}`;
  field('scene-label').textContent=s.label;
  field('scene-label').hidden=!s.label;
  field('scene-title').textContent=s.title;
  field('scene-title').hidden=!s.title;
  field('caption').textContent=s.caption;
  field('caption').hidden=!s.caption;
  const examples=field('examples');
  if(examples){examples.innerHTML=quoteMarkup(s.quotes);examples.hidden=!s.quotes?.length;}
  story.dataset.hasQuotes=String(Boolean(s.quotes?.length));
  field('equation').innerHTML=tex(s.equation);
  field('detail').innerHTML=s.detail;
  field('detail').hidden=!s.detail;
  renderMathNodes(field('detail'));
  field('visual-caption').textContent=s.visual;
  svg.setAttribute('aria-label',`${s.title} ${s.caption} ${s.visual}`);
  last=index;
 }
 field('progress').style.width=`${clamp(total/scenes.length)*100}%`;
 // Each pinned stage owns its SVG paint definitions.
 svg.innerHTML=(defs+s.draw(t)).replaceAll('id="arrow"',`id="${id}-arrow"`).replaceAll('url(#arrow)',`url(#${id}-arrow)`).replaceAll('id="blocked"',`id="${id}-blocked"`).replaceAll('url(#blocked)',`url(#${id}-blocked)`);
 story.dataset.scene=index;story.dataset.phase=t.toFixed(3);
 const a=reducedMotion()?1:Math.min(1,.72+local*4);
 field('narration').style.opacity=a;
 };
}
const renderStories=[makeStory('intuition','intuition-',intuitionScenes),makeStory('story','',coreScenes),makeStory('numerical-story','numerical-',numericalScenes),makeStory('mechanics-story','mechanics-',mechanicsScenes)];
let scheduled=false;
function render(){scheduled=false;renderStories.forEach(draw=>draw())}
function schedule(){if(!scheduled){scheduled=true;requestAnimationFrame(render)}}
addEventListener('scroll',schedule,{passive:true});addEventListener('resize',schedule);reduced.addEventListener('change',schedule);
function revealReading(){
 const target=document.getElementById(location.hash.slice(1));
 if(!target)return;
 for(let node=target;node;node=node.parentElement)if(node.tagName==='DETAILS')node.open=true;
 requestAnimationFrame(()=>target.scrollIntoView({block:'start'}));
 schedule();
}
addEventListener('toggle',schedule,true);
addEventListener('hashchange',revealReading);revealReading();
// Hash jumps and native keyboard scrolling follow the same reversible sequence.
render();


// Precomputed outputs keep this lesson usable without a model download.
fetch(new URL('./data/infinite-jest-attention.json',import.meta.url))
 .then(response=>{if(!response.ok)throw new Error('Model results unavailable');return response.json();})
 .then(data=>{
  const scenes=createTrainedScenes(drawing,data);
  renderStories.splice(2,0,makeStory('trained-story','trained-',scenes));
  buildExplorer(document.querySelector('#trained-explorer'),data);
  const reading=document.createElement('div');
  reading.innerHTML='<h3>Measured results from DistilBERT</h3>'+transcript(scenes);
  document.querySelector('#transcript').append(reading);
  render();
 })
 .catch(error=>{
  document.querySelector('#trained-error').hidden=false;
  document.querySelector('#trained-story').hidden=true;
  console.error(error);schedule();
 });
