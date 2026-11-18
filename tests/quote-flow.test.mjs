import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import test from 'node:test';
import {documents, tokenize, vocabulary, counts, tfidf, displayTerms, termStats, referenceCount, focusTerms} from '../docs/text/model.mjs';

const require=createRequire(import.meta.url);
globalThis.katex=require('../docs/word2vec/vendor/katex/katex.min.js');
const {quoteFlowVisual}=await import('../docs/text/quote-flow.mjs');
const phases=['corpus','tokens','stopwords','fullclass','matrix','tf','df','idf','word-document','weighted','cosine'];
const attrs=tag=>Object.fromEntries([...tag.matchAll(/([\w-]+)="([^"]*)"/g)].map(([,key,value])=>[key,value]));
const tokenTexts=markup=>[...markup.matchAll(/<text\b([^>]*data-key="text-(\d)-(\d+)"[^>]*)>([^<]+)<\/text>/g)].map(([,attributes,row,index,word])=>({...attrs(attributes),row:Number(row),index:Number(index),word}));
const cells=markup=>[...markup.matchAll(/<g\b([^>]*data-key="cell-\d+-\d+"[^>]*)>/g)].map(([,attributes])=>attrs(attributes));
const tex=markup=>[...markup.matchAll(/<annotation encoding="application\/x-tex">([\s\S]*?)<\/annotation>/g)].map(([,value])=>value.replaceAll('&amp;','&').replaceAll('&lt;','<').replaceAll('&gt;','>'));

test('exact excerpt previews become tokens, then the lesson explicitly expands to full transcripts',()=>{
  const corpus=quoteFlowVisual('corpus',1),stop=quoteFlowVisual('stopwords',1),whole=quoteFlowVisual('fullclass',1);
  for(const [row,document] of documents.entries()){
    const words=[...corpus.matchAll(new RegExp(`<text[^>]*data-key="quote-${row}-\\d+"[^>]*>([^<]+)<\\/text>`,'g'))].map(([,word])=>word);
    assert.equal(words.join(' '),document.text);
    assert.deepEqual(tokenTexts(quoteFlowVisual('tokens',1)).filter(token=>token.row===row).map(token=>token.word),tokenize(document.text));
    const kept=[...stop.matchAll(/<g data-key="word-(\d+)-(\d+)"[^>]*data-kept="true" opacity="1">([\s\S]*?)<\/g>/g)];
    assert.deepEqual(kept.filter(([,r])=>Number(r)===row).map(([, , ,body])=>body.match(/data-key="text-\d+-\d+"[^>]*>([^<]+)</)[1]),document.tokens);
    assert.ok(whole.includes(document.retainedTokenCount.toLocaleString('en-US')));
    assert.ok(document.retainedTokenCount>document.tokens.length*20,'A full class must not be represented by its preview counts.');
    assert.equal(counts[row].reduce((sum,value)=>sum+value,0),document.retainedTokenCount);
  }
  assert.match(whole,/Count every retained word in both full transcripts/);
  assert.match(whole,/data-key="source-quotations" opacity="0"/);
  assert.ok(cells(quoteFlowVisual('matrix',0)).every(cell=>Number(cell['data-value'])===0));
  for(const phase of phases){
    const markup=quoteFlowVisual(phase,1);
    assert.equal(tokenTexts(markup).length,documents.reduce((sum,d)=>sum+tokenize(d.text).length,0));
    assert.match(markup,/data-key="author-0"[^>]*>Cycling</);
    assert.match(markup,/data-key="author-1"[^>]*>Yoga</);
  }
});

test('wide preview words have actual font advances and clear spacing, with matching token boxes',()=>{
  const corpus=quoteFlowVisual('corpus',1);
  const words=[...corpus.matchAll(/<text\b([^>]*data-key="quote-\d+-\d+"[^>]*)>([^<]+)<\/text>/g)].map(([,attributes,word])=>({...attrs(attributes),word}));
  // Reference advances read from the bundled LM font at 30px, not a character
  // count estimate. These pairs previously collided in the rendered preview.
  for(const [word,advance] of [['minimum',125.01],['Now',59.16],['And',55.86]]){
    const i=words.findIndex(token=>token.word===word),current=words[i],next=words[i+1];
    assert.equal(next.y,current.y);
    assert.ok(Number(next.x)-Number(current.x)-advance>=10,`${word} needs a visible following space`);
  }
  const tokens=quoteFlowVisual('tokens',1),minimum=tokenTexts(tokens).find(token=>token.word==='minimum');
  const box=tokens.match(new RegExp(`<rect[^>]*data-key="box-${minimum.row}-${minimum.index}"[^>]*>`))[0];
  assert.ok(Math.abs(Number(attrs(box).width)-131.01)<1e-9);
  const strike=tokens.match(new RegExp(`<path[^>]*data-key="strike-${minimum.row}-${minimum.index}"[^>]*>`))[0];
  assert.ok(Math.abs(Number(attrs(strike).d.split('h')[1])-129.01)<1e-9);
  assert.doesNotMatch(corpus,/textLength=/);
  assert.doesNotMatch(quoteFlowVisual('matrix',1),/data-key="matrix-scope"/);
});

test('six fixed full-class columns persist and the count ranking changes under corpus-based TF–IDF',()=>{
  const expectedKeys=cells(quoteFlowVisual('matrix',1)).map(cell=>cell['data-key']);
  for(const phase of ['matrix','tf','df','idf','word-document','weighted','cosine']){
    const displayed=cells(quoteFlowVisual(phase,1));
    assert.deepEqual(displayed.map(cell=>cell['data-key']),expectedKeys);
    assert.equal(displayed.length,12);
    for(const cell of displayed){
      const row=Number(cell['data-doc']),column=vocabulary.indexOf(cell['data-term']);
      assert.equal(Number(cell['data-value']),['weighted','cosine'].includes(phase)?tfidf[row][column]:counts[row][column]);
    }
    for(const term of displayTerms)assert.ok(displayed.some(cell=>cell['data-term']===term));
  }
  for(const q of [0,.25,.5,.75,1])for(const cell of cells(quoteFlowVisual('weighted',q))){
    const row=Number(cell['data-doc']),column=vocabulary.indexOf(cell['data-term']);
    const expected=q===1?tfidf[row][column]:counts[row][column]+(tfidf[row][column]-counts[row][column])*q;
    assert.equal(Number(cell['data-value']),expected);
  }
  const resistance=termStats('resistance',0),confidence=termStats('confidence',0);
  assert.ok(resistance.tf>confidence.tf);
  assert.ok(resistance.weight<confidence.weight);
});

test('document frequency uses 127 reference classes rather than the two displayed rows',()=>{
  const markup=quoteFlowVisual('df',1),dots=[...markup.matchAll(/<circle\b[^>]*data-key="reference-document-\d+"[^>]*>/g)].map(([tag])=>attrs(tag));
  assert.equal(dots.length,referenceCount);
  assert.equal(dots.filter(dot=>dot['data-present']==='true').length,termStats(focusTerms.distinctive,0).df);
  assert.equal(referenceCount,127);
  assert.equal(termStats('confidence',0).df,25);
  assert.ok(tex(markup).includes(String.raw`\mathrm{df}_{\mathrm{confidence}}=25,\qquad N=127`));
  const equations=tex(quoteFlowVisual('idf',1));
  assert.ok(equations.some(value=>value.includes(String.raw`\ln(127/110)`)));
  assert.ok(equations.some(value=>value.includes(String.raw`\ln(127/25)`)));
  assert.match(quoteFlowVisual('cosine',1),/Full-vocabulary cosine similarity changes from 0\.2318/);
  const comparison=tex(quoteFlowVisual('cosine',1));
  assert.ok(comparison.includes(String.raw`\cos(\mathbf c_C,\mathbf c_Y)\approx0.232`));
  assert.ok(comparison.includes(String.raw`\cos(\mathbf w_C,\mathbf w_Y)\approx0.022`));
});

test('preprocessing retraces token positions and full-class rows stay fixed while weights change',()=>{
  const positions=markup=>tokenTexts(markup).map(({row,index,x,y,word})=>({row,index,x,y,word}));
  for(const [previous,next] of [['tokens','stopwords'],['stopwords','fullclass'],['fullclass','matrix']]){
    assert.deepEqual(positions(quoteFlowVisual(previous,1)),positions(quoteFlowVisual(next,0)),`${previous} → ${next}`);
  }
  const matrixGeometry=markup=>[...markup.matchAll(/<text\b([^>]*data-key="(?:author-|column-|cell-value-)[^"]+"[^>]*)>/g)].map(([,attributes])=>{
    const a=attrs(attributes);return{key:a['data-key'],x:a.x,y:a.y};
  });
  const expected=matrixGeometry(quoteFlowVisual('matrix',1));
  for(const phase of phases.slice(5))assert.deepEqual(matrixGeometry(quoteFlowVisual(phase,.5)),expected);
  assert.notDeepEqual(positions(quoteFlowVisual('stopwords',0)),positions(quoteFlowVisual('stopwords',.5)));
  assert.notDeepEqual(matrixGeometry(quoteFlowVisual('matrix',0)),matrixGeometry(quoteFlowVisual('matrix',.5)));
});

test('every transcript phase has finite geometry, valid mathematics, and deterministic reverse states',()=>{
  for(const phase of phases){
    const progress=[0,.2,.4,.6,.8,1],forward=progress.map(q=>quoteFlowVisual(phase,q));
    assert.deepEqual([...progress].reverse().map(q=>quoteFlowVisual(phase,q)).reverse(),forward);
    for(const markup of forward){
      assert.doesNotMatch(markup,/NaN|Infinity|undefined|katex-error/);
      for(const expression of tex(markup))assert.doesNotThrow(()=>globalThis.katex.renderToString(expression,{throwOnError:true}));
    }
  }
});
