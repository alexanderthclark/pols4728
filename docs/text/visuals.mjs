import {speechCartoon} from './speech-cartoon.mjs?v=20261028-preprocessing-3';
import {latex, escapeText as esc} from './math.mjs?v=20261028-preprocessing-3';
import {documents,tokenize,vocabulary,counts,tfidf,termStats,cosine} from './model.mjs?v=20261028-preprocessing-3';

const clamp=x=>Math.max(0,Math.min(1,x));
const mix=(a,b,t)=>a+(b-a)*t;
const fmt=(n,d=2)=>Math.abs(n)<1e-9?'0':n.toFixed(d);
const svg=(body,label)=>`<svg viewBox="0 0 680 390" role="img" aria-label="${esc(label)}">${body}</svg>`;
const txt=(x,y,t,extra='')=>`<text x="${x}" y="${y}" ${extra.includes('font-size=')?'':'font-size="26"'} ${extra}>${esc(t)}</text>`;
const line=(x1,y1,x2,y2,color='#767676',extra='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="2" ${extra}/>`;
const equation=tex=>`<div class="math">${latex(tex,true)}</div>`;
const note=text=>`<p class="visual-note">${text}</p>`;
const preprocessingSvg=(body,label)=>`<svg class="preprocessing-diagram" viewBox="0 0 680 300" role="img" aria-label="${esc(label)}">${body}</svg>`;
const tokenSlots=[[20,198,110],[150,198,185],[355,198,145],[520,198,120],[20,270,145],[185,270,110],[315,270,145],[480,270,160]];

/** A token is an occurrence: repeated words remain separate as boundaries become visible. */
export function tokenizationVisual(q){
  q=clamp(q);
  const raw=documents[0].text.split(' '),words=tokenize(documents[0].text);
  const starts=[[20,90],[85,90],[270,90],[405,90],[470,90],[20,140],[85,140],[205,140]];
  return preprocessingSvg(txt(20,30,'D1 · separate the word tokens','font-size="25"')+words.map((word,i)=>{
    const [x,y,width]=tokenSlots[i],px=mix(starts[i][0],x+10,q),py=mix(starts[i][1],y,q);
    return `<rect x="${x}" y="${y-36}" width="${width}" height="50" fill="none" stroke="#767676" opacity="${q}"/>`+
      txt(px,py,q<.5?raw[i]:word,`font-size="32" data-token="${i}"`);
  }).join(''),'The eight words in D1 separate into eight tokens; capital letters become lowercase and periods are removed.')+note('8 tokens · repeated words remain separate');
}

/** Illustrative dictionary forms, not a lemmatizer applied to the worked corpus. */
export function lemmatizationVisual(q){
  q=clamp(q);
  const forms=[['debated','debate'],['debating','debate'],['senators','senator']];
  return preprocessingSvg(txt(30,30,'Word form','font-size="25"')+txt(440,30,'Dictionary form','font-size="25"')+forms.map(([form,lemma],i)=>{
    const y=98+i*80,end=mix(265,400,q);
    return txt(30,y,form,'font-size="36"')+line(265,y-10,end,y-10,'#234e70')+
      `<path d="M${end-10} ${y-17}L${end} ${y-10}L${end-10} ${y-3}" fill="none" stroke="#234e70" stroke-width="2" opacity="${q}"/>`+
      txt(440,y,lemma,`font-size="36" fill="#234e70" opacity="${q}" data-lemma="${lemma}"`);
  }).join(''),'Lemmatization maps debated and debating to debate, and senators to senator.')+note('Optional grouping · our worked matrix keeps the original forms');
}

/** Show exactly which occurrences a possible stop-word choice would discard. */
export function stopWordsVisual(q){
  q=clamp(q);
  const words=tokenize(documents[0].text),kept=words.filter(word=>word!=='the');
  const starts=[[20,115],[135,115],[350,115],[535,115],[20,190],[190,190],[310,190],[495,190]];
  const ends=[[20,115],[240,115],[435,115],[20,190],[220,190]];
  let keepIndex=0;
  return preprocessingSvg(txt(20,30,'Illustrative choice: remove “the”','font-size="25"')+words.map((word,i)=>{
    const removed=word==='the',[x,y]=starts[i],target=removed?[x,y+40]:ends[keepIndex++];
    const px=mix(x,target[0],q),py=mix(y,target[1],q);
    return `<g data-token="${i}" data-kept="${!removed}" opacity="${removed?1-q:1}">`+
      txt(px,py,word,`font-size="32" fill="${removed?'#a54f32':'#242424'}"`)+
      (removed?`<path d="M${px-3} ${py-12}h57" stroke="#a54f32" stroke-width="2" opacity="${q}"/>`:'')+'</g>';
  }).join('')+txt(20,275,q<1?'8 tokens → 5 retained · 3 removed':`${kept.length} retained tokens · 3 instances of “the” removed`,'font-size="25"'),'Removing the stop word the discards three tokens and leaves parliament, debated, budget, budget, and passed.')+note('A possible choice only · our worked matrix retains “the”');
}

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
  const q=motion?clamp((p-stage+.5)*2):1;
  if(stage===2)return `<div class="corpus">${documents.map((d,i)=>`<div class="doc"><span>D${i+1}</span><span>${d.text}</span></div>`).join('')}</div>`;
  if(stage===3)return tokenizationVisual(q);
  if(stage===4)return lemmatizationVisual(q);
  if(stage===5)return stopWordsVisual(q);
  if(stage===6)return countAssembly(1+q);
  if(stage===7)return `<div class="flow"><div class="flow-box">Senators support voters</div><span aria-hidden="true">↓</span><div class="count-vector">senators: 1 · support: 1 · voters: 1</div><span aria-hidden="true">↑</span><div class="flow-box">Voters support senators</div></div>${note('Different roles. Identical count vectors.')}`;
  if(stage===8){const t=q;return `<h4 class="visual-title">“Upon” per 1,000 words</h4>`+svg(
    txt(25,92,'Hamilton')+txt(25,218,'Madison')+
    line(210,275,630,275)+[0,1,2,3].map(v=>line(210+v*140,270,210+v*140,281)+txt(210+v*140,319,String(v),'text-anchor="middle"')).join('')+
    `<rect x="210" y="60" height="44" width="${420*t}" fill="#234e70"/><rect x="210" y="186" height="44" width="${140/6*t}" fill="#a54f32"/>`+
    txt(630,43,'≈ 3.0',`text-anchor="end" opacity="${t}"`)+txt(260,216,'≈ 0.17',`opacity="${t}"`), 'Approximate reported rates of upon: Hamilton 3 per thousand, Madison 0.17 per thousand')+note('Mosteller &amp; Wallace (1963) · comparison writings');}
  return `<div class="purpose"><div><span class="purpose-label">Who wrote it?</span><p>Common words<br>Writing habits</p></div><div><span class="purpose-label">What is it about?</span><p>Distinctive terms<br>TF–IDF weights</p></div></div>${note('Keep the count matrix · change what receives weight')}`;
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
