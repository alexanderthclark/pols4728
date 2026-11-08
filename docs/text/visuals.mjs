import {chapters} from './lesson.mjs?v=20261108-cosine-5';
import {speechCartoon} from './speech-cartoon.mjs?v=20261108-cosine-5';
import {latex, escapeText as esc} from './math.mjs?v=20261108-cosine-5';
import {documents,tokenize,vocabulary,counts,tfidf,termStats,cosine,displayTerms,stopWords,lemmaForms} from './model.mjs?v=20261108-cosine-5';
import {applicationVisual} from './applications.mjs?v=20261108-cosine-5';
import {shortSentencesVisual} from './short-sentences.mjs?v=20261108-cosine-5';

const clamp=x=>Math.max(0,Math.min(1,x));
const mix=(a,b,t)=>a+(b-a)*t;
const fmt=(n,d=2)=>Math.abs(n)<1e-9?'0':n.toFixed(d);
const svg=(body,label)=>`<svg viewBox="0 0 680 390" role="img" aria-label="${esc(label)}">${body}</svg>`;
const txt=(x,y,t,extra='')=>`<text x="${x}" y="${y}" ${extra.includes('font-size=')?'':'font-size="26"'} ${extra}>${esc(t)}</text>`;
const line=(x1,y1,x2,y2,color='#767676',extra='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="2" ${extra}/>`;
const equation=tex=>`<div class="math">${latex(tex,true)}</div>`;
const note=text=>`<p class="visual-note">${text}</p>`;
const preprocessingSvg=(body,label)=>`<svg class="preprocessing-diagram" viewBox="0 0 680 300" role="img" aria-label="${esc(label)}">${body}</svg>`;
// Wrap tokens in a fixed view box; every occurrence keeps its own identity.
function tokenPositions(words){
  let x=14,y=74;
  return words.map(word=>{const width=word.length*15+22;if(x+width>670){x=14;y+=52;}const point=[x,y,width];x+=width+9;return point;});
}

export function tokenizationVisual(q){
  q=clamp(q);
  const raw=documents[0].text.split(' '),words=tokenize(documents[0].text),slots=tokenPositions(raw);
  return preprocessingSvg(txt(14,25,'Schumpeter · each occurrence is a token','font-size="26"')+words.map((word,i)=>{
    const [x,y,width]=slots[i];
    return `<rect x="${x}" y="${y-30}" width="${width}" height="42" fill="none" stroke="#767676" opacity="${q}"/>`+
      txt(x+7,y,q<.5?raw[i]:word,`font-size="29" data-token="${i}"`);
  }).join(''),'Schumpeter’s quotation yields 21 lowercase alphabetic tokens; the and as each occur twice.')+note('21 tokens · “the” and “as” each occur twice');
}

export function lemmatizationVisual(q){
  q=clamp(q);
  const forms=[['drops','drop'],['enters','enter'],['making','make']];
  return preprocessingSvg(txt(30,30,'Word form','font-size="25"')+txt(440,30,'Dictionary form','font-size="25"')+forms.map(([form,lemma],i)=>{
    const y=98+i*80,end=mix(265,400,q);
    return txt(30,y,form,'font-size="36"')+line(265,y-10,end,y-10,'#234e70')+
      `<path d="M${end-10} ${y-17}L${end} ${y-10}L${end-10} ${y-3}" fill="none" stroke="#234e70" stroke-width="2" opacity="${q}"/>`+
      txt(440,y,lemma,`font-size="36" fill="#234e70" opacity="${q}" data-lemma="${lemma}"`);
  }).join(''),'Lemmatization maps drops to drop, enters to enter, and making to make.');
}

export function stopWordsVisual(q){
  q=clamp(q);
  const words=tokenize(documents[0].text).map(word=>lemmaForms[word]??word);
  const kept=words.filter(word=>!stopWords.has(word)),starts=tokenPositions(words),ends=tokenPositions(kept);
  let keepIndex=0;
  return preprocessingSvg(txt(14,25,'Schumpeter · remove the chosen stop words','font-size="25"')+words.map((word,i)=>{
    const removed=stopWords.has(word),[x,y]=starts[i],target=removed?[x,y+25]:ends[keepIndex++];
    const px=mix(x,target[0],q),py=mix(y,target[1],q);
    return `<g data-token="${i}" data-kept="${!removed}" opacity="${removed?1-q:1}">`+
      txt(px+7,py,word,`font-size="29" fill="${removed?'#a54f32':'#242424'}"`)+
      (removed?`<path d="M${px+4} ${py-10}h${word.length*15}" stroke="#a54f32" stroke-width="2" opacity="${q}"/>`:'')+'</g>';
  }).join(''),'After lemmatization and the chosen stop-word removal, Schumpeter has 11 tokens; 10 occurrences are removed.')+note('21 tokens → 11 retained · Whitman retains 15');
}

export function matrix(values,{focus=null,doc=null,decimals=0,weighted=false,caption=null,presence=false,interactive=false}={}){
  const terms=interactive?vocabulary:displayTerms,col=vocabulary.indexOf(focus);
  return `<table class="matrix${interactive?' explorer-matrix':''}"><caption class="visual-title">${caption??(weighted?'TF–IDF weights':'Document–term counts')}${interactive?' · all 25 terms':' · selected columns'}</caption><thead><tr><th scope="col">Document</th>${terms.map(t=>`<th scope="col" class="${t===focus?'term-focus':''}">${t}</th>`).join('')}</tr></thead><tbody>${values.map((row,r)=>`<tr ${r===doc?'class="active"':''}><th scope="row">${documents[r].label}</th>${terms.map(term=>{
    const c=vocabulary.indexOf(term),v=row[c],value=decimals?fmt(v,decimals):v,selected=r===doc&&c===col;
    return `<td class="${c===col?'focus ':''}${presence&&c===col&&v>0?'present ':''}${v===0?'zero':''}">${interactive?`<button type="button" class="matrix-cell" data-doc="${r}" data-col="${c}" data-term="${term}" tabindex="${selected?0:-1}" aria-pressed="${selected}" aria-label="${documents[r].label}, ${term}, count ${v}">${value}</button>`:value}</td>`;
  }).join('')}</tr>`).join('')}</tbody></table>`;
}

/** The displayed occurrences travel to their shared coordinates; other columns are omitted. */
export function countAssembly(p){
  const q=clamp(p-1),arrival=clamp((q-.65)/.35);
  const tokens=displayTerms.filter(term=>documents[0].tokens.includes(term));
  const table=displayTerms.map((word,c)=>txt(175+c*80,145,word,`font-size="27" transform="rotate(-55 ${175+c*80} 145)"`)).join('')+
    counts.map((row,r)=>txt(10,202+r*65,documents[r].label,'font-size="24"')+displayTerms.map((term,c)=>txt(175+c*80,202+r*65,row[vocabulary.indexOf(term)],'text-anchor="middle" font-size="30"')).join('')).join('')+
    [165,225,290].map(y=>line(10,y,660,y,'#d9d9d5')).join('');
  const flying=tokens.map((word,i)=>{
    const t=clamp((q-i*.06)/.62),column=displayTerms.indexOf(word),x=mix(30+i*210,175+column*80,t),y=mix(70,202,t);
    return txt(x,y,word,`font-size="31" fill="#234e70" opacity="${1-clamp((q-.5)/.3)}" data-token="${i}"`);
  }).join('');
  return svg(`<g opacity="${arrival}" data-matrix-reveal="${arrival}">${table}</g>${flying}`,'Schumpeter tokens enter selected vocabulary columns. Two document rows use six displayed columns from the full 25-term vocabulary.');
}

export function bagVisual(stage,p,motion=true){
  const scene=chapters.find(chapter=>chapter.id==='bag-of-words').stages[stage].visual;
  if(scene==='speech'||scene==='speech-counts')return speechCartoon(motion?clamp(p):stage,{reducedMotion:!motion,id:'lesson-speech'});
  const q=motion?clamp((p-stage+.5)*2):1;
  if(scene.startsWith('short-'))return shortSentencesVisual(scene,q);
  if(scene==='corpus')return `<div class="corpus">${documents.map((d,i)=>`<figure class="doc"><blockquote>“${d.text}”</blockquote><figcaption>${d.label} <span>· ${d.source}</span></figcaption></figure>`).join('')}</div>`;
  if(scene==='tokens')return tokenizationVisual(q);
  if(scene==='lemma')return lemmatizationVisual(q);
  if(scene==='stopwords')return stopWordsVisual(q);
  if(scene==='matrix')return countAssembly(1+q);
  if(scene==='order')return `<div class="flow"><div class="flow-box">Senators support voters</div><span aria-hidden="true">↓</span><div class="count-vector">senators: 1 · support: 1 · voters: 1</div><span aria-hidden="true">↑</span><div class="flow-box">Voters support senators</div></div>`;
  if(scene==='federalist'){const t=q;return `<h4 class="visual-title">“Upon” per 1,000 words</h4>`+svg(
    txt(25,92,'Hamilton')+txt(25,218,'Madison')+
    line(210,275,630,275)+[0,1,2,3].map(v=>line(210+v*140,270,210+v*140,281)+txt(210+v*140,319,String(v),'text-anchor="middle"')).join('')+
    `<rect x="210" y="60" height="44" width="${420*t}" fill="#234e70"/><rect x="210" y="186" height="44" width="${140/6*t}" fill="#a54f32"/>`+
    txt(630,43,'≈ 3.0',`text-anchor="end" opacity="${t}"`)+txt(260,216,'≈ 0.17',`opacity="${t}"`), 'Approximate reported rates of upon: Hamilton 3 per thousand, Madison 0.17 per thousand')+note('Mosteller &amp; Wallace (1963) · comparison writings');}
  return `<div class="purpose"><div><span class="purpose-label">Who wrote it?</span><p>Common words<br>Writing habits</p></div><div><span class="purpose-label">What is it about?</span><p>Distinctive terms<br>TF–IDF weights</p></div></div>`;
}

/** A term's IDF is shared; its score belongs to a term–document pair. */
export function wordDocumentVisual(){
  const rows=documents.map((document,doc)=>{
    const s=termStats('citizen',doc);
    const calculation=String.raw`${s.tf}\ln(${s.N}/${s.df})${s.tf?String.raw`\approx`:'='}${fmt(s.weight,3)}`;
    return `<tr data-doc="${doc}"><th scope="row">${document.label}</th><td>${s.tf}</td><td data-weight="${s.weight}">${latex(calculation)}</td></tr>`;
  }).join('');
  return `<div class="pair-example">${equation(String.raw`w_{t,d}=c_{t,d}\ln\!\left(\frac{N}{\mathrm{df}_t}\right)`)}<table class="pair-table"><caption>“citizen” across the two documents</caption><thead><tr><th scope="col">Document</th><th scope="col">Count</th><th scope="col">TF–IDF score</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

export function tfVisual(stage,p,motion=true){

  const term=stage===0?'political':'citizen',s=termStats(term,0);
  if(stage<2)return matrix(counts,{focus:term,doc:0,presence:stage===1})+`<div class="metric-line">${latex(String.raw`c_{\mathrm{${term}},\mathrm{Schumpeter}}=${s.tf}`)}${stage===1?latex(String.raw`\mathrm{df}_{\mathrm{${term}}}=${s.df}`):''}</div>`;
  if(stage===2)return `<div class="idf-comparison">${['political','citizen','america'].map(t=>{const s=termStats(t,0);return `<div><span>${t}</span>${latex(String.raw`\ln\!\left(\frac{${s.N}}{${s.df}}\right)${s.idf?String.raw`\approx`:"="}${fmt(s.idf,3)}`)}</div>`;}).join('')}</div>`+equation(String.raw`\mathrm{idf}_t=\ln\!\left(\frac{N}{\mathrm{df}_t}\right)`);
  if(stage===3)return wordDocumentVisual();
  if(stage===4){const q=motion?clamp((p-3.5)*2):1,vals=counts.map((r,i)=>r.map((v,j)=>mix(v,tfidf[i][j],q))),scale=mix(1,s.idf,q);return matrix(vals,{focus:term,doc:0,decimals:2,weighted:true,caption:q<1?'From counts to TF–IDF weights':'TF–IDF weights'})+equation(String.raw`1\times ${fmt(scale,3)}=${fmt(scale,3)}`)+note(q<1?'Scroll to reweight each vocabulary column.':'Schumpeter · citizen: count × inverse document frequency');}
  const similarity=cosine(tfidf[0],tfidf[1]);
  return `<h4 class="visual-title">Schumpeter and Whitman</h4><div class="similarity-result">${latex(String.raw`\cos\theta=${fmt(similarity,3)}`)}</div>`+equation(String.raw`\cos\theta=\frac{x^{\mathsf T}y}{\lVert x\rVert\,\lVert y\rVert}`);
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
