import {chapters} from './lesson.mjs?v=20261108-topics-1';
import {speechCartoon} from './speech-cartoon.mjs?v=20261108-topics-1';
import {latex, escapeText as esc} from './math.mjs?v=20261108-topics-1';
import {documents,vocabulary,counts,termStats,displayTerms} from './model.mjs?v=20261108-topics-1';
import {applicationVisual} from './applications.mjs?v=20261108-topics-1';
import {shortSentencesVisual} from './short-sentences.mjs?v=20261108-topics-1';
import {quoteFlowVisual} from './quote-flow.mjs?v=20261108-topics-1';

const clamp=x=>Math.max(0,Math.min(1,x));
const fmt=(n,d=2)=>Math.abs(n)<1e-9?'0':n.toFixed(d);
const svg=(body,label)=>`<svg viewBox="0 0 680 390" role="img" aria-label="${esc(label)}">${body}</svg>`;
const txt=(x,y,t,extra='')=>`<text x="${x}" y="${y}" ${extra.includes('font-size=')?'':'font-size="26"'} ${extra}>${esc(t)}</text>`;
const line=(x1,y1,x2,y2,color='#767676',extra='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="2" ${extra}/>`;
const equation=tex=>`<div class="math">${latex(tex,true)}</div>`;
const note=text=>`<p class="visual-note">${text}</p>`;
export function matrix(values,{focus=null,doc=null,decimals=0,weighted=false,caption=null,presence=false,interactive=false}={}){
  const terms=interactive?vocabulary:displayTerms,col=vocabulary.indexOf(focus);
  return `<table class="matrix${interactive?' explorer-matrix':''}"><caption class="visual-title">${caption??(weighted?'TF–IDF weights':'Document–term counts')}${interactive?' · all 25 terms':' · selected columns'}</caption><thead><tr><th scope="col">Document</th>${terms.map(t=>`<th scope="col" class="${t===focus?'term-focus':''}">${t}</th>`).join('')}</tr></thead><tbody>${values.map((row,r)=>`<tr ${r===doc?'class="active"':''}><th scope="row">${documents[r].label}</th>${terms.map(term=>{
    const c=vocabulary.indexOf(term),v=row[c],value=decimals?fmt(v,decimals):v,selected=r===doc&&c===col;
    return `<td class="${c===col?'focus ':''}${presence&&c===col&&v>0?'present ':''}${v===0?'zero':''}">${interactive?`<button type="button" class="matrix-cell" data-doc="${r}" data-col="${c}" data-term="${term}" tabindex="${selected?0:-1}" aria-pressed="${selected}" aria-label="${documents[r].label}, ${term}, count ${v}">${value}</button>`:value}</td>`;
  }).join('')}</tr>`).join('')}</tbody></table>`;
}

export function bagVisual(stage,p,motion=true){
  const scene=chapters.find(chapter=>chapter.id==='bag-of-words').stages[stage].visual;
  if(scene==='speech'||scene==='speech-counts')return speechCartoon(motion?clamp(p):stage,{reducedMotion:!motion,id:'lesson-speech',countLayout:scene==='speech-counts'?'matrix':'list'});
  const q=motion?clamp((p-stage+.5)*2):1;
  if(scene.startsWith('short-'))return shortSentencesVisual(scene,q);
  if(['corpus','tokens','lemma','stopwords','matrix'].includes(scene))return quoteFlowVisual(scene,q);
  if(scene==='order')return `<div class="flow"><div class="flow-box">Senators support voters</div><span aria-hidden="true">↓</span><div class="count-vector">senators: 1 · support: 1 · voters: 1</div><span aria-hidden="true">↑</span><div class="flow-box">Voters support senators</div></div>`;
  throw new RangeError(`Unknown bag-of-words scene: ${scene}`);
}

export function authorshipVisual(stage,p,motion=true){
  const t=motion?clamp((p-stage+.5)*2):1;
  return `<h4 class="visual-title">“Upon” per 1,000 words</h4>`+svg(
    txt(25,92,'Hamilton')+txt(25,218,'Madison')+
    line(210,275,630,275)+[0,1,2,3].map(v=>line(210+v*140,270,210+v*140,281)+txt(210+v*140,319,String(v),'text-anchor="middle"')).join('')+
    `<rect x="210" y="60" height="44" width="${420*t}" fill="#234e70"/><rect x="210" y="186" height="44" width="${140/6*t}" fill="#a54f32"/>`+
    txt(630,43,'≈ 3.0',`text-anchor="end" opacity="${t}"`)+txt(260,216,'≈ 0.17',`opacity="${t}"`), 'Approximate reported rates of upon: Hamilton 3 per thousand, Madison 0.17 per thousand')+note('Mosteller &amp; Wallace (1963) · comparison writings');
}

export function tfVisual(stage,p,motion=true){
  const phase=chapters.find(chapter=>chapter.id==='tf-idf').stages[stage].visual;
  return quoteFlowVisual(phase,motion?clamp((p-stage+.5)*2):1);
}

export function topicVisual(stage,p,motion=true){
  return applicationVisual(stage+6,p+6,motion);
}

export function termEquations(term='citizen',doc=0){
  const s=termStats(term,doc);
  return `<p class="explorer-selection">${esc(term)} in ${documents[doc].label}</p>${equation(String.raw`c_{t,d}=${s.tf},\qquad\mathrm{df}_t=${s.df},\qquad N=${s.N}`)}${equation(String.raw`w_{t,d}=${s.tf}\ln\!\left(\frac{${s.N}}{${s.df}}\right)${s.weight?String.raw`\approx`:"="}${fmt(s.weight,3)}`)}`;
}

export function inspectTerm(term='citizen',doc=0){
  return `<div class="explorer-scroll">${matrix(counts,{focus:term,doc,interactive:true})}</div><div class="explorer-result explore-equations" aria-live="polite" aria-atomic="true">${termEquations(term,doc)}</div>`;
}
