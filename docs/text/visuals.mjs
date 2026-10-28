import {speechCartoon} from './speech-cartoon.mjs?v=20261028-counts-2';
import {latex, escapeText as esc} from './math.mjs?v=20261028-counts-2';
import {documents,tokenize,vocabulary,counts,tfidf,termStats,cosine} from './model.mjs?v=20261028-counts-2';

const clamp=x=>Math.max(0,Math.min(1,x));
const mix=(a,b,t)=>a+(b-a)*t;
const fmt=(n,d=2)=>Math.abs(n)<1e-9?'0':n.toFixed(d);
const svg=(body,label)=>`<svg viewBox="0 0 680 390" role="img" aria-label="${esc(label)}">${body}</svg>`;
const txt=(x,y,t,extra='')=>`<text x="${x}" y="${y}" ${extra.includes('font-size=')?'':'font-size="26"'} ${extra}>${esc(t)}</text>`;
const line=(x1,y1,x2,y2,color='#767676',extra='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="2" ${extra}/>`;
const equation=tex=>`<div class="math">${latex(tex,true)}</div>`;
const note=text=>`<p class="visual-note">${text}</p>`;

export function matrix(values,{focus=null,doc=null,decimals=0,weighted=false,caption=null,presence=false}={}){
  const col=vocabulary.indexOf(focus);
  return `<table class="matrix"><caption class="visual-title">${caption??(weighted?'TF–IDF weights':'Document–term counts')}</caption><thead><tr><th scope="col">Text</th>${vocabulary.map(t=>`<th scope="col" class="${t===focus?'term-focus':''}">${t}</th>`).join('')}</tr></thead><tbody>${values.map((row,r)=>`<tr ${r===doc?'class="active"':''}><th scope="row">D${r+1}</th>${row.map((v,c)=>`<td class="${c===col?'focus ':''}${presence&&c===col&&v>0?'present ':''}${v===0?'zero':''}">${decimals?fmt(v,decimals):v}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}

/** Words travel to the coordinates they will occupy in the shared count matrix. */
export function countAssembly(p){
  const q=clamp(p-1),arrival=clamp((q-.65)/.35),tokens=tokenize(documents[0].text);
  const positions=[[30,92],[170,92],[430,92],[30,170],[170,170],[30,248],[170,248],[430,248]];
  const table=vocabulary.map((word,c)=>txt(112+c*78,150,word,`font-size="28" transform="rotate(-55 ${112+c*78} 150)"`)).join('')+
    counts.map((row,r)=>txt(27,202+r*43,`D${r+1}`,'font-size="24"')+row.map((v,c)=>txt(112+c*78,202+r*43,v,'text-anchor="middle" font-size="28"')).join('')).join('')+
    [168,217,260,303,346].map(y=>line(20,y,640,y,'#d9d9d5')).join('');
  const flying=tokens.map((word,i)=>{
    const t=clamp((q-i*.035)/.62),column=vocabulary.indexOf(word),x=mix(positions[i][0],112+column*78,t),y=mix(positions[i][1],202,t);
    return txt(x,y,word,`fill="${word==='budget'?'#234e70':'#242424'}" opacity="${1-clamp((q-.5)/.3)}" data-token="${i}"`);
  }).join('');
  return svg(`<g opacity="${arrival}" data-matrix-reveal="${arrival}">${table}</g>${flying}`,'D1 tokens move into shared vocabulary columns. The completed matrix has four document rows and seven term columns.')+
    note(q<.5?'D1 · 8 tokens, with repetitions retained':'Four documents · the same seven vocabulary coordinates');
}

export function bagVisual(stage,p,motion=true){
  if(stage<2)return speechCartoon(motion?clamp(p):stage,{reducedMotion:!motion,id:'lesson-speech'});
  stage-=2;p-=2;
  if(stage===0)return `<div class="corpus">${documents.map((d,i)=>`<div class="doc"><span>D${i+1}</span><span>${d.text}</span></div>`).join('')}</div>`;
  if(stage===1||stage===2)return countAssembly(motion?p:stage);
  if(stage===3)return `<div class="flow"><div class="flow-box">Senators support voters</div><span aria-hidden="true">↓</span><div class="count-vector">senators: 1 · support: 1 · voters: 1</div><span aria-hidden="true">↑</span><div class="flow-box">Voters support senators</div></div>${note('Different roles. Identical count vectors.')}`;
  if(stage===4){const t=motion?clamp((p-3.5)*2):1;return `<h4 class="visual-title">“Upon” per 1,000 words</h4>`+svg(
    txt(25,92,'Hamilton')+txt(25,218,'Madison')+
    line(210,275,630,275)+[0,1,2,3].map(v=>line(210+v*140,270,210+v*140,281)+txt(210+v*140,319,String(v),'text-anchor="middle"')).join('')+
    `<rect x="210" y="60" height="44" width="${420*t}" fill="#234e70"/><rect x="210" y="186" height="44" width="${140/6*t}" fill="#a54f32"/>`+
    txt(630,43,'≈ 3.0',`text-anchor="end" opacity="${t}"`)+txt(260,216,'≈ 0.17',`opacity="${t}"`), 'Approximate reported rates of upon: Hamilton 3 per thousand, Madison 0.17 per thousand')+note('Mosteller &amp; Wallace (1963) · comparison writings');}
  if(stage===5)return `<div class="study-pipeline"><div class="study-labels"><span>Revelation</span><span>Gospel of John</span></div><span aria-hidden="true">↓</span><div class="flow-box">500-word Greek samples<br>75 common-word counts</div><span aria-hidden="true">↓</span><div>Compare frequency profiles</div></div>${note('Erwin &amp; Oakes (2012) · study design, not a replication')}`;
  return `<div class="purpose"><div><span class="purpose-label">Authorship</span><p>Common words<br>Writing habits</p></div><div><span class="purpose-label">Topic</span><p>Distinctive terms<br>Subject matter</p></div></div>${note('The research question determines what to keep.')}`;
}

export function tfVisual(stage,p,motion=true){
  const term=stage===0?'the':'budget',s=termStats(term,0);
  if(stage<2)return matrix(counts,{focus:term,doc:0,presence:stage===1})+`<div class="metric-line">${latex(String.raw`\operatorname{tf}(\mathrm{${term}},D_1)=${s.tf}`)}${stage===1?latex(String.raw`\operatorname{df}(\mathrm{${term}})=${s.df}`):''}</div>`+note(stage===0?'“The” appears in every document.':'Each outlined cell contributes one document, regardless of its count.');
  if(stage===2)return `<h4 class="visual-title">Document rarity becomes a weight</h4><div class="idf-comparison">${['the','budget','passed'].map(t=>{const s=termStats(t,0);return `<div><span>${t}</span>${latex(String.raw`\ln\!\left(\frac{4}{${s.df}}\right)=${fmt(s.idf,3)}`)}</div>`;}).join('')}</div>`+equation(String.raw`\operatorname{idf}(t)=\ln\!\left(\frac{N}{\operatorname{df}(t)}\right)`);
  if(stage===3){const q=motion?clamp((p-2.5)*2):1,vals=counts.map((r,i)=>r.map((v,j)=>mix(v,tfidf[i][j],q))),scale=mix(1,s.idf,q);return matrix(vals,{focus:term,doc:0,decimals:2,weighted:true,caption:q<1?'From counts to TF–IDF weights':'TF–IDF weights'})+equation(String.raw`2\times ${fmt(scale,3)}=${fmt(2*scale,3)}`)+note(q<1?'Scroll to reweight each vocabulary column.':'D1 · budget: count × inverse document frequency');}
  return `<h4 class="visual-title">TF–IDF cosine similarity</h4><div class="bars">${[1,2,3].map(i=>{const v=cosine(tfidf[0],tfidf[i]);return `<div class="bar-row"><span>D1 with D${i+1}</span><div class="bar-track"><div class="bar-fill" style="width:${v*100}%"></div></div><span class="bar-value">${fmt(v,3)}</span></div>`;}).join('')}</div>`+equation(String.raw`\cos\theta=\frac{x^{\mathsf T}y}{\lVert x\rVert\,\lVert y\rVert}`)+note('0: no weighted overlap · 1: the same direction');
}

export function inspectTerm(term='budget',doc=0){
  const s=termStats(term,doc);
  return matrix(counts,{focus:term,doc})+`<div class="explore-equations">${equation(String.raw`\operatorname{tf}=${s.tf},\qquad\operatorname{df}=${s.df},\qquad N=${s.N}`)}${equation(String.raw`\text{TF–IDF}=${s.tf}\ln\!\left(\frac{${s.N}}{${s.df}}\right)=${fmt(s.weight,3)}`)}</div>`;
}
