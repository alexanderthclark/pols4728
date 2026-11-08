import {documents, tokenize, isRetainedToken, displayTerms, vocabulary, counts, tfidf, cosine, termStats, referenceCount, focusTerms} from './model.mjs?v=20261108-peloton-1';
import {escapeText as esc, mathLabel} from './math.mjs?v=20261108-peloton-1';

const clamp = value => Math.max(0, Math.min(1, value));
const mix = (a, b, q) => q<=0?a:q>=1?b:a+(b-a)*q;
const blue = '#234e70', rust = '#a54f32', ink = '#242424', muted = '#606060', rule = '#deded9';
const phases = ['corpus', 'tokens', 'stopwords', 'fullclass', 'matrix', 'tf', 'df', 'idf', 'word-document', 'weighted', 'cosine'];
const columns = displayTerms.map((word, index) => ({word, column: vocabulary.indexOf(word), x: 200 + index * 80}));
const rowY = [220, 290], quoteY = [110, 392], authorY = [28, 310];
const format = (value, weighted) => Math.abs(value) < 1e-10 ? '0' : weighted ? String(Number(value.toFixed(2))) : String(Math.round(value));
const text = (key, x, y, value, attributes = '') => `<text data-key="${key}" x="${x}" y="${y}" ${attributes}>${esc(value)}</text>`;
const math = (key, x, y, w, h, tex, size=30) => `<g data-key="${key}">${mathLabel(x,y,w,h,tex,size)}</g>`;
const group = (key, opacity, body) => `<g data-key="${key}" opacity="${opacity}" aria-hidden="${opacity === 0}">${body}</g>`;
const relation = value => Math.abs(value)<1e-10 ? '=0' : String.raw`\approx${Number(value.toFixed(3))}`;
const rawSimilarity = cosine(counts[0],counts[1]), weightedSimilarity = cosine(tfidf[0],tfidf[1]);
const sampleTerms = [focusTerms.common,focusTerms.distinctive];

// Actual advance widths from fonts/lmroman10-regular.otf (1,000 units/em).
// Separate SVG words keep their natural glyph shapes; their advances and one
// normal space determine placement. A small clearance also separates boxes.
const lmAsciiAdvances=[333,278,374,833,500,833,778,278,389,389,500,778,278,333,278,500,500,500,500,500,500,500,500,500,500,500,278,278,778,778,778,472,778,750,708,722,764,681,653,785,750,361,514,778,625,917,750,778,681,778,736,556,722,750,750,1028,750,750,611,278,500,278,556,750,500,500,556,444,556,444,306,500,556,278,306,528,278,833,556,500,556,528,392,394,389,556,528,722,528,528,444,500,278,500,556];
const lmExtraAdvances={'·':778,'’':278,'‘':278,'“':472,'”':472,'–':500,'—':1000};
function wordAdvance(word,fontSize=30){
  const units=[...word].reduce((sum,character)=>{
    const code=character.codePointAt(0);
    return sum+(code>=32&&code<=126?lmAsciiAdvances[code-32]:lmExtraAdvances[character]??750);
  },0);
  return units*fontSize/1000;
}
function wordSlots(words,top,fontSize=30,lineHeight=35){
  let x=14,y=top;
  const gap=fontSize*.333+1;
  return words.map(word=>{
    const width=wordAdvance(word,fontSize);
    if(x+width>665){x=14;y+=lineHeight;}
    const slot={x,y,width};x+=width+gap;return slot;
  });
}

// Source quotations remain separate from normalized tokens: contractions and
// punctuation need not have a one-to-one relationship with whitespace words.
const sourceWords=documents.map((document,row)=>{
  const words=document.text.split(/\s+/),slots=wordSlots(words,quoteY[row]);
  return words.map((word,index)=>({word,...slots[index]}));
});
const previewTokens=documents.map((document,row)=>{
  const words=tokenize(document.text), start=wordSlots(words,quoteY[row]);
  const retained=wordSlots(words.filter(isRetainedToken),quoteY[row]);
  let retainedIndex=0;
  return words.map((word,index)=>({word,index,row,removed:!isRetainedToken(word),start:start[index],end:isRetainedToken(word)?retained[retainedIndex++]:start[index]}));
});

function classSummary(row, reveal, assembly) {
  const document=documents[row], top=row*282, color=row?rust:blue;
  const lines=Array.from({length:7},(_,i)=>`<path data-key="document-line-${row}-${i}" d="M46 ${top+119+i*15}h${82+(i%3)*14}" stroke="${color}" stroke-width="2" opacity=".5"/>`).join('');
  return group(`whole-class-${row}`,reveal*(1-assembly),
    `<rect data-key="document-page-${row}" x="30" y="${top+95}" width="146" height="147" fill="#fff" stroke="${color}" stroke-width="2"/>${lines}`+
    text(`whole-count-${row}`,235,top+144,document.retainedTokenCount.toLocaleString('en-US'),`font-size="48" fill="${color}"`)+
    text(`whole-label-${row}`,235,top+187,'retained words in the full class',`font-size="28" fill="${ink}"`)+
    text(`raw-total-${row}`,235,top+224,`${document.rawTokenCount.toLocaleString('en-US')} tokens before filtering`,`font-size="24" fill="${muted}"`));
}

function calculation(phase) {
  const common=termStats(focusTerms.common,0), distinct=termStats(focusTerms.distinctive,0);
  if(phase==='matrix')return {
    primary:String.raw`\cos(\mathbf c_C,\mathbf c_Y)${relation(rawSimilarity)}`,
    secondary:`Count vectors · all ${vocabulary.length.toLocaleString('en-US')} vocabulary coordinates`,
  };
  if(phase==='tf')return {
    primary:String.raw`c_{\mathrm{${focusTerms.common}},C}=${common.tf},\qquad c_{\mathrm{${focusTerms.distinctive}},C}=${distinct.tf}`,
    secondary:`“${focusTerms.common}” has the higher count in Cycling.`,
  };
  if(phase==='df')return {
    primary:String.raw`\mathrm{df}_{\mathrm{${focusTerms.distinctive}}}=${distinct.df},\qquad N=${referenceCount}`,
    secondary:'Document frequency uses the whole reference corpus.',
  };
  if(phase==='idf')return {
    primary:String.raw`\mathrm{idf}_t=\ln\!\left(\frac{N}{\mathrm{df}_t}\right)`,
    secondaryMath:sampleTerms.map(term=>{
      const s=termStats(term,0);
      return String.raw`\mathrm{${term}}:\quad\ln(${s.N}/${s.df})${relation(s.idf)}`;
    }),
  };
  if(phase==='word-document')return {
    primary:String.raw`w_{t,d}=c_{t,d}\ln\!\left(\frac{N}{\mathrm{df}_t}\right)`,
    secondaryMath:sampleTerms.map(term=>{
      const s=termStats(term,0);
      return String.raw`w_{\mathrm{${term}},\mathrm{Cycling}}=${s.tf}\ln(${s.N}/${s.df})${relation(s.weight)}`;
    }),
  };
  if(phase==='weighted'){
    const earth=termStats('earth',1);
    return {
      primary:String.raw`w_{\mathrm{earth},Y}=${earth.tf}\ln(${earth.N}/${earth.df})${relation(earth.weight)}`,
      secondary:'Same full-class counts; corpus-wide rarity changes the weights.',
    };
  }
  if(phase==='cosine')return {
    primary:String.raw`\cos(\mathbf c_C,\mathbf c_Y)${relation(rawSimilarity)}`,
    secondaryMath:[String.raw`\cos(\mathbf w_C,\mathbf w_Y)${relation(weightedSimilarity)}`],
    secondary:`Counts → TF–IDF · all ${vocabulary.length.toLocaleString('en-US')} coordinates`,
  };
  return {primary:'',secondary:''};
}

/**
 * The same two classes persist through excerpt preprocessing, a deliberate
 * expansion to full-transcript counts, and corpus-based TF–IDF weighting.
 */
export function quoteFlowVisual(phase,progress=1) {
  const phaseIndex=phases.indexOf(phase);
  if(phaseIndex<0)throw new RangeError(`Unknown transcript phase: ${phase}`);
  const q=clamp(progress), tokenAmount=phaseIndex<1?0:phase==='tokens'?q:1;
  const removal=phaseIndex<2?0:phase==='stopwords'?q:1;
  const whole=phaseIndex<3?0:phase==='fullclass'?q:1;
  const assembly=phaseIndex<4?0:phase==='matrix'?q:1;
  const previewOpacity=1-clamp(whole*2), summaryReveal=clamp((whole-.3)/.7);
  const weightAmount=phaseIndex<9?0:phase==='weighted'?q:1;
  const values=counts.map((row,r)=>row.map((count,c)=>mix(count,tfidf[r][c],weightAmount)*assembly));
  const focus=phase==='tf'?focusTerms.common:phase==='weighted'?'earth':['df','idf','word-document'].includes(phase)?focusTerms.distinctive:null;
  const isWeighted=phaseIndex>=9;
  const tableTitle=phase==='weighted'&&q<1?'Full-class counts → TF–IDF weights':isWeighted?'Full-class TF–IDF weights':'Full-class word counts';

  const authors=documents.map((document,row)=>text(`author-${row}`,14,mix(authorY[row],rowY[row],assembly),document.label,
    `class="q-author" font-size="31" fill="${row?rust:blue}"`)).join('');
  const sources=documents.map((document,row)=>{
    const words=document.source.split(/\s+/), slots=wordSlots(words,57+row*282,19,25);
    return group(`source-${row}`,1-assembly,words.map((word,i)=>text(`source-word-${row}-${i}`,slots[i].x,slots[i].y,word,`font-size="19" fill="${muted}"`)).join(''));
  }).join('');
  const quotes=group('source-quotations',(1-tokenAmount)*previewOpacity,sourceWords.map((words,row)=>words.map(({word,x,y},index)=>text(`quote-${row}-${index}`,x,y,word,`font-size="30" fill="${ink}"`)).join('')).join(''));
  const tokens=previewTokens.flat().map(({row,index,word,removed,start,end})=>{
    const x=mix(start.x,end.x,removal),y=mix(start.y,end.y,removal);
    const opacity=tokenAmount*previewOpacity*(removed?1-removal:1);
    return `<g data-key="word-${row}-${index}" data-doc="${row}" data-token="${index}" data-kept="${!removed}" opacity="${opacity}">
      <rect data-key="box-${row}-${index}" x="${x-3}" y="${y-26}" width="${wordAdvance(word)+6}" height="34" fill="none" stroke="${rule}" stroke-width="1" opacity="${1-removal}"/>
      ${text(`text-${row}-${index}`,x,y,word,`font-size="30" fill="${removed&&removal>0?muted:ink}"`)}
      <path data-key="strike-${row}-${index}" d="M${x-2} ${y-9}h${wordAdvance(word)+4}" stroke="${muted}" stroke-width="2" opacity="${removed?removal:0}"/>
    </g>`;
  }).join('');
  const grid=group('matrix',assembly,
    text('matrix-title',340,34,tableTitle,`text-anchor="middle" font-size="27" fill="${ink}"`)+
    columns.map(({word,x})=>text(`column-${word}`,x,171,word,`transform="rotate(-55 ${x} 171)" font-size="28" fill="${word===focus?blue:ink}"`)).join('')+
    [188,255,325].map((y,index)=>`<path data-key="rule-${index}" d="M14 ${y}H665" stroke="${rule}" fill="none"/>`).join('')+
    counts.map((row,r)=>columns.map(({word,column,x})=>{
      const value=values[r][column],highlighted=word===focus||['tf','idf'].includes(phase)&&sampleTerms.includes(word);
      // This outline indicates whether one of the two displayed classes contains
      // the word; the separate corpus dots, not these rows, establish its df.
      const present=phase==='df'&&word===focus&&row[column]>0;
      return `<g data-key="cell-${r}-${column}" data-doc="${r}" data-term="${word}" data-value="${value}">
        <rect data-key="cell-fill-${r}-${column}" x="${x-34}" y="${rowY[r]-30}" width="68" height="51" fill="${highlighted?'#edf2f5':'white'}" stroke="${present?blue:'none'}" stroke-width="2"/>
        ${text(`cell-value-${r}-${column}`,x,rowY[r],format(value,isWeighted),`text-anchor="middle" font-size="${mix(40,26,weightAmount)}" fill="${Math.abs(value)<1e-10?muted:highlighted?blue:ink}"`)}
      </g>`;
    }).join('')).join(''));

  const calc=calculation(phase),hasCalculation=phaseIndex>=4;
  const eqBody=phase==='df'
    ? text('reference-label',340,363,`${termStats(focusTerms.distinctive,0).df} of ${referenceCount} classes contain “${focusTerms.distinctive}”`,`text-anchor="middle" font-size="26" fill="${ink}"`)+
      Array.from({length:referenceCount},(_,i)=>`<circle data-key="reference-document-${i}" data-present="${i<termStats(focusTerms.distinctive,0).df}" cx="${29+(i%16)*41}" cy="${390+Math.floor(i/16)*16}" r="5.5" fill="${i<termStats(focusTerms.distinctive,0).df?blue:'#fff'}" stroke="${i<termStats(focusTerms.distinctive,0).df?blue:rule}" stroke-width="1.5"/>`).join('')+
      math('primary',8,516,664,49,calc.primary,29)+text('calculation-note',340,588,calc.secondary,`text-anchor="middle" font-size="23" fill="${muted}"`)
    : math('primary',8,354,664,60,calc.primary,31)+
      (calc.secondaryMath?calc.secondaryMath.map((tex,index)=>math(`secondary-${index}`,8,427+index*58,664,48,tex,29)).join(''):text('calculation-note',340,467,calc.secondary||'',`text-anchor="middle" font-size="25" fill="${muted}"`))+
      (phase==='cosine'?text('comparison-note',340,536,calc.secondary,`text-anchor="middle" font-size="25" fill="${muted}"`):'');
  const equations=group('calculation',hasCalculation?(phase==='matrix'?assembly:1):0,eqBody);
  const scope=group('preview-scope',1-assembly,text('preview-scope-text',340,586,whole>.5?'Count every retained word in both full transcripts.':'Short excerpts shown · each full class is one document',`text-anchor="middle" font-size="24" fill="${muted}"`));
  const description=phaseIndex<4
    ? `${documents.map(document=>`${document.label}: ${document.text}`).join(' ')} These are short previews, not the texts used as whole documents. ${phase==='fullclass'?documents.map(d=>`${d.label} has ${d.retainedTokenCount} retained words in its full transcript.`).join(' '):phase==='stopwords'?'Selected stop words, short tokens, and terms outside the corpus vocabulary are removed from the previews.':phase==='tokens'?'Preview words become lowercase alphabetic tokens.':''}`
    : `${isWeighted?'TF–IDF weights':'Word counts'} for the full Cycling and Yoga transcripts. Six displayed columns: ${columns.map(({word,column})=>`${word}: ${format(values[0][column],isWeighted)}, ${format(values[1][column],isWeighted)}`).join('; ')}. IDF is computed from ${referenceCount} reference class transcripts. ${phase==='cosine'?`Full-vocabulary cosine similarity changes from ${rawSimilarity} for counts to ${weightedSimilarity} for TF–IDF.`:''}`;
  return `<div class="quote-flow" data-visual-family="quotations" data-phase="${phase}" data-fullclass-progress="${whole}" data-assembly-progress="${assembly}"><svg class="quote-workspace" viewBox="0 0 680 600" role="group" aria-label="${esc(description)}">${grid}<g data-key="authors">${authors}</g><g data-key="sources">${sources}</g>${quotes}<g data-key="words">${tokens}</g>${classSummary(0,summaryReveal,assembly)}${classSummary(1,summaryReveal,assembly)}${scope}${equations}</svg></div>`;
}
