import { vocabulary as vocab, initial, forward, step, interpolateModel } from './model.js';
import { networkScene, renderNetwork } from './network.js?v=linear-scores';
import { analogyVectors, cosineScene, renderCosine } from './cosine.js';
import { modernScene, renderModern } from './modern.js?v=improvements';

const BLUE='#234e70', RUST='#a54f32', INK='#242424', MUTED='#606060', RULE='#d9d9d5';
const story=document.querySelector('#story'), svg=document.querySelector('#diagram');
const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches||new URLSearchParams(location.search).get('motion')==='reduce';
const scenes=[
  {chapter:'01 · The prediction task',title:'Learn from the words around a word.',eq:String.raw`\text{center word}\;\longrightarrow\;\text{context words}`,caption:'Skip-gram word2vec predicts nearby words from a center word.',detail:'Here we use one word on each side. Larger windows include more surrounding words.¹',alt:'The sentence voters support reform; support is the center and the other two words are its context.'},
  {chapter:'01 · The prediction task',title:'One input word gives two training examples.',eq:String.raw`x=w_c,\qquad y=w_o`,caption:'Each nearby word gives a separate training example.',detail:'Both examples have the same input. We will follow the pair with context word “reform.”',alt:'Two separate training examples in columns labelled center word x (input) and context word y (output): support to voters, and support to reform. The second row is highlighted for the walkthrough.'},
  {chapter:'02 · The input matrix',title:'A one-hot vector identifies the word.',eq:String.raw`d_c\in\{0,1\}^{|V|},\qquad |V|=6`,caption:'',detail:'',alt:'The six-word vocabulary is aligned with a one-hot column; only the second entry, support, is one.'},
  {chapter:'02 · The input matrix',title:'The input matrix stores one vector per word.',eq:String.raw`W_{\mathrm{in}}\in\mathbb{R}^{k\times|V|},\qquad k=2`,caption:'Each word labels a column. Each column contains two trainable coordinates.',detail:'Six words, two coordinates: 2 rows and 6 columns. These are illustrative starting weights.',alt:'W in is a two by six matrix with vocabulary words labeling columns. Support is the second column.'},
  {chapter:'02 · The input matrix',title:'Multiplication becomes a lookup.',eq:String.raw`h_c=W_{\mathrm{in}}d_c`,caption:'The 1 selects the “support” column. That column is the input embedding.',detail:'The hidden layer uses the identity (linear) activation: it passes these coordinates through unchanged.',alt:'The support column, 0.6 and 0.4, moves out of W in as the vector h c.'},
  {chapter:'03 · The output matrix',title:'A second matrix has a different job.',eq:String.raw`\begin{gathered}W_{\mathrm{out}}\in\mathbb{R}^{|V|\times k}\\u_o^{\mathsf T}=\text{row }o\text{ of }W_{\mathrm{out}}\end{gathered}`,caption:'Each row of the output matrix holds a vector for a candidate context word.',detail:'Both matrices contain every word. They are separate learned parameters, not transposes of one another.',alt:'W in has one column per word. W out has one row per word, denoted u o transpose. The highlighted reform row is labelled u reform transpose. The two matrices have separate numeric values.'},
  {chapter:'03 · The output matrix',title:'One row produces one score.',eq:String.raw`z_{co}=u_o^{\mathsf T}h_c`,caption:'The “reform” row of the output matrix takes a dot product with the “support” input vector.',detail:'A score measures compatibility for prediction. It is not yet a probability.',alt:'The vector u reform transpose is the reform row of W out, with entries 0.4 and 0.6. It multiplies the support embedding 0.6 and 0.4; the products sum to 0.48.'},
  {chapter:'03 · The output matrix',title:'Score every word. Then normalize.',eq:String.raw`p(w_o\mid w_c)=\frac{\exp(z_{co})}{\sum_j\exp(z_{cj})}`,caption:'Softmax exponentiates each score and divides by the sum of all six exponentiated scores.',detail:'The resulting probabilities add up to 1. Every vocabulary word is an output candidate.',alt:'All six dot products become scores and then a probability distribution; reform has probability 0.225.'},
  {chapter:'04 · Learning both matrices',title:'The sentence supplies the answer.',eq:String.raw`\ell=-\log p(w_o\mid w_c)`,caption:'The observed context word is “reform.” Its probability determines this pair’s loss.',detail:'The context word comes from the text. Training adjusts the matrices to make observed contexts more probable.',alt:'Reform is the observed context word. Its probability is 0.225 and the negative log probability is about 1.492.'},
  {chapter:'04 · Learning both matrices',title:'One error changes both matrices.',eq:String.raw`W\leftarrow W-\eta\,\nabla_W\ell,\quad\eta=0.3`,caption:'One gradient step changes the selected input column and every output row.',detail:'Under full softmax, all candidate words participate. Other input columns stay fixed on this pair.',alt:'The support column and every output row change by gradient descent. Reform probability rises from 0.225 to 0.269.'},
  {chapter:'04 · Learning both matrices',title:'Repeat across words and contexts.',eq:String.raw`\min_{W_{\mathrm{in}},W_{\mathrm{out}}}\;\sum_{(c,o)}-\log p(w_o\mid w_c)`,caption:'Each new pair selects its own input column. Both matrices accumulate what the corpus teaches.',detail:'This short sequence demonstrates updates on a tiny corpus. Useful embeddings require much more text.',alt:'Different center-context pairs select different input columns while all output rows keep learning.'},
  {chapter:'05 · Reading the representation',title:'The same word has two learned vectors.',eq:String.raw`\text{input embedding}\;\neq\;\text{output embedding}`,caption:'“Support” has a column in the input matrix and a separate row in the output matrix.',detail:'Here we keep the whole input matrix as the embedding table: one fixed vector per word, reused across its occurrences.',alt:'Support is highlighted in both matrices. Its input and output vectors are separately learned. The input matrix supplies the embeddings retained after training.'},
  {chapter:'05 · Reading the representation',title:'Similar contexts can yield similar vectors.',eq:String.raw`\operatorname{cos}(h_a,h_b)=\frac{h_a^{\mathsf T}h_b}{\|h_a\|\,\|h_b\|}`,caption:'To examine similarity, compare two input embeddings. A small angle gives a large cosine.',detail:'Illustrative geometry: not output from our tiny model. Similar usage can reflect meaning or grammatical role; it need not mean synonymy.',alt:'Two input vectors, voters and citizens, approach similar directions. A small angle between them corresponds to high cosine similarity.'},
  {chapter:'05 · Reading the representation',title:'Relationships can have directions, too.',eq:String.raw`h_{\text{king}}-h_{\text{man}}+h_{\text{woman}}\approx h_{\text{queen}}`,caption:'Subtract “man” from “king,” then add “woman.” This gives a query vector to compare with candidate words.',detail:'Illustrative projection. Such patterns can emerge in trained embeddings; they are not guaranteed identities.',alt:'Translating the man-to-king difference from woman produces the query vector, ending near but not exactly at queen. The next scene compares their directions.'},
  {chapter:'05 · Reading the representation',title:'The geometry is learned through prediction.',eq:String.raw`d_c\;\xrightarrow{\;W_{\mathrm{in}}\;}\;h_c\;\xrightarrow{\;W_{\mathrm{out}}\;}\;z_c\;\longrightarrow\;p`,caption:'We supplied words and their contexts. Training supplied the coordinates.',detail:'The two matrices make the learning task work; the input vectors become a representation we can reuse.',alt:'The complete path from a one-hot word through W in to an embedding, then through W out to scores and probabilities.'}
];

// Requested clarity addition: orient readers before the matrix walkthrough.
scenes.splice(2,0,networkScene);
scenes.splice(15,0,cosineScene);
scenes.push(modernScene);

const f0=forward(initial), first=step(initial);
// Exact gradient updates on every adjacent directed pair in four toy sentences.
const sentences=[[0,1,2],[3,4,5],[0,1,5],[3,4,2]];
const pairs=sentences.flatMap(words=>words.flatMap((c,i)=>[i-1,i+1].filter(j=>j>=0&&j<words.length).map(j=>[c,words[j]])));
const corpus=[first];for(const [c,o] of pairs)corpus.push(step(corpus.at(-1),c,o,.22));
const final=corpus.at(-1);
const clamp=(x,a=0,b=1)=>Math.max(a,Math.min(b,x));
const ease=x=>{x=clamp(x);return x*x*(3-2*x)};
const mix=(a,b,t)=>a+(b-a)*t;
const fmt=(v,n=2)=>(Math.abs(v)<.0001?0:v).toFixed(n);
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;');
const mathCache=new Map();
function latex(tex){if(!mathCache.has(tex))mathCache.set(tex,katex.renderToString(tex,{throwOnError:true,output:'htmlAndMathml'}));return mathCache.get(tex)}
function m(tex,x,y,w=200,h=42,size=22,color=INK){return `<foreignObject x="${x-w/2}" y="${y-h/2}" width="${w}" height="${h}" style="overflow:visible;color:${color};font-size:${size}px"><div xmlns="http://www.w3.org/1999/xhtml" class="mathbox">${latex(tex)}</div></foreignObject>`}
function text(s,x,y,size=16,color=INK,anchor='middle',extra=''){return `<text x="${x}" y="${y}" fill="${color}" font-size="${size}" text-anchor="${anchor}" ${extra}>${esc(s)}</text>`}
function line(x1,y1,x2,y2,color=RULE,width=1.5,arrow=false,dash=''){return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}" ${arrow?'marker-end="url(#arrow-'+(color===BLUE?'blue':color===RUST?'rust':'ink')+')"':''} ${dash?`stroke-dasharray="${dash}"`:''}/>`}
function rect(x,y,w,h,stroke=RULE,fill='white',r=0,width=1){return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" stroke="${stroke}" fill="${fill}" stroke-width="${width}"/>`}
function group(content,opacity=1,transform=''){return `<g opacity="${clamp(opacity)}" transform="${transform}">${content}</g>`}
function brackets(x,y,w,h,color=INK){return `<path d="M${x+5} ${y}h-10v${h}h10 M${x+w-5} ${y}h10v${h}h-10" fill="none" stroke="${color}" stroke-width="1.5"/>`}
function inMatrix(model,x,y,cw=62,ch=42,selected=1,{dim=false,title=true,changed=false}={}){
 let s=title?m(String.raw`W_{\mathrm{in}}`,x+cw*3,y-61,170,36,25,BLUE)+text('one column per word',x+cw*3,y-35,13,BLUE):'';
 s+=rect(x+selected*cw+2,y-3,cw-4,2*ch+6,BLUE,'#edf3f7',0,2);
 for(let j=0;j<6;j++){s+=text(vocab[j],x+(j+.5)*cw,y-13,cw<55?12.5:13,j===selected?BLUE:MUTED);for(let d=0;d<2;d++)s+=group(m(fmt(model.Win[d][j]),x+(j+.5)*cw,y+(d+.5)*ch,cw,ch,cw<55?17:20,j===selected?BLUE:INK),dim&&j!==selected?.25:1)}
 s+=brackets(x-5,y-4,6*cw+10,2*ch+8,BLUE);
 if(changed)s+=text('only this column changes',x+3*cw,y+2*ch+25,12,BLUE);
 return s;
}
function outMatrix(model,x,y,cw=60,ch=35,selected=2,{dim=false,title=true,changed=false}={}){
 let s=title?m(String.raw`W_{\mathrm{out}}`,x+cw,y-52,170,36,25,RUST)+text('one row per word',x+cw,y-24,13,RUST):'';
 if(changed)s+=rect(x-3,y-3,cw*2+6,ch*6+6,RUST,'#fcf2ee',0,2);
 s+=rect(x-2,y+selected*ch,2*cw+4,ch,RUST,'#faeee8',0,2);
 for(let j=0;j<6;j++){s+=text(vocab[j],x-16,y+(j+.5)*ch+4,13,j===selected?RUST:MUTED,'end');for(let d=0;d<2;d++)s+=group(m(fmt(model.Wout[j][d]),x+(d+.5)*cw,y+(j+.5)*ch,cw,ch,19,j===selected?RUST:INK),dim&&j!==selected?.3:1)}
 return s+brackets(x-5,y-3,2*cw+10,6*ch+6,RUST);
}
function vector(h,x,y,{color=BLUE,label=String.raw`h_c`,horizontal=false,size=23}={}){
 const a=horizontal?String.raw`\begin{bmatrix}${fmt(h[0])}&${fmt(h[1])}\end{bmatrix}`:String.raw`\begin{bmatrix}${fmt(h[0])}\\${fmt(h[1])}\end{bmatrix}`;
 return m(label,x,y-55,200,36,size,color)+m(a,x,y,200,92,size,color);
}
function chip(word,x,y,{color=INK,fill='white',w=142,h=54}={}){return rect(x-w/2,y-h/2,w,h,color,fill,3,1.5)+text(word,x,y+6,19,color)}
function stem(n){return n?'400 450':'1000 420'}
function both(model,n,{selected=1,out=2,changed=false}={}){
 return n?inMatrix(model,44,85,52,30,selected,{changed})+outMatrix(model,214,246,55,28,out,{changed}):inMatrix(model,70,165,64,44,selected,{changed})+outMatrix(model,745,117,65,38,out,{changed});
}
function pairLabel(c,o,n,y){return text('CENTER',n?112:390,y-23,10,BLUE)+text('OBSERVED CONTEXT',n?290:610,y-23,10,RUST)+chip(vocab[c],n?112:390,y+10,{color:BLUE,w:n?125:155})+chip(vocab[o],n?290:610,y+10,{color:RUST,w:n?125:155})+line(n?185:480,y+10,n?211:519,y+10,MUTED,1.5,true)}

function render(i,t,n){
 const w=n?400:1000;let s='';const mid=w/2;
 if(i===0){
  const y=n?150:170,xs=n?[66,200,334]:[250,500,750];
  s+=rect(xs[0]-57,y-49,xs[2]-xs[0]+114,100,RULE,'#fafaf8');
  xs.forEach((x,j)=>s+=chip(['voters','support','reform'][j],x,y,{w:n?112:185,color:j===1?BLUE:INK,fill:j===1?'#edf3f7':'white'}));
  s+=text('center',xs[1],y-75,14,BLUE);
  const p=ease(t*1.5);for(const x of [xs[0],xs[2]])s+=line(xs[1],y+55,mix(xs[1],x,p),y+55+60*p,RUST,2,true);
  s+=group(text('nearby word',xs[0],y+145,13,RUST)+text('nearby word',xs[2],y+145,13,RUST),p);
  s+=text('a context window of one word on each side',mid,n?365:360,13,MUTED);
 }
 if(i===1){
  const p=ease(t*1.7),selected=ease((t-.62)/.38);
  const cx=n?104:340,ox=n?294:665,cy=225,half=n?65:95;
  s+=text('voters support reform',mid,45,19,MUTED);
  s+=m(String.raw`\text{center word }x`,cx,92,200,38,n?22:25,BLUE);
  s+=m(String.raw`\text{context word }y`,ox,92,200,38,n?22:25,RUST);
  s+=text('input',cx,119,13,MUTED)+text('output',ox,119,13,MUTED);
  s+=group(rect(n?23:225,250,n?354:555,70,RUST,'#fff8f4',3,1.5),selected);
  for(let j=0;j<2;j++){
   const y=j?285:170,inputY=mix(cy,y,p);
   s+=line(cx+half+8,inputY,ox-half-10,y,j?RUST:MUTED,1.5,true);
   if(j===0||p>0)s+=chip('support',cx,inputY,{color:BLUE,w:2*half});
   s+=chip(j?'reform':'voters',ox,y,{color:j?RUST:INK,w:2*half,fill:j?'#faeee8':'white'});
  }
  s+=group(m(String.raw`(x,y)=(\text{support},\text{reform})`,mid,366,n?390:650,45,n?21:27,RUST)+text('Follow this example',mid,404,14,RUST),selected);
 }
 if(i===2){
  const x=n?185:500, y=n?100:65,ch=n?42:43;
  s+=chip('support',n?90:230,n?45:190,{color:BLUE,w:n?125:180,h:42});
  if(!n)s+=line(330,190,425,190,BLUE,1.5,true);
  s+=m('d_c',x+48,y-30,120,32,24,BLUE);
  for(let j=0;j<6;j++){s+=text(vocab[j],x-12,y+j*ch+23,14,j===1?BLUE:MUTED,'end');if(j===1)s+=rect(x+16,y+j*ch,64,ch,BLUE,'#edf3f7',0,2);s+=m(j===1?'1':'0',x+48,y+(j+.5)*ch,60,ch,24,j===1?BLUE:MUTED)}
  s+=brackets(x+8,y,80,6*ch,BLUE);
  if(n)s+=text('vocabulary order stays fixed',200,405,13,MUTED);
 }
 if(i===3||i===4){
  const x=n?44:85,y=n?115:154,cw=n?52:75,ch=n?35:48,p=ease(t);
  s+=inMatrix(initial,x,y,cw,ch,1,{dim:i===4});
  if(i===3)s+=m(String.raw`\underbrace{\phantom{000000000000}}_{|V|=6\text{ words}}`,x+3*cw,y+2*ch+42,6*cw,50,n?17:20,BLUE);
  else{ s+=m(String.raw`d_c^{\mathsf T}`,x-(n?24:40),y+2*ch+32,60,40,n?16:21,BLUE);for(let j=0;j<6;j++)s+=m(j===1?'1':'0',x+(j+.5)*cw,y+2*ch+32,cw,40,n?19:24,j===1?BLUE:MUTED); }
  if(i===3){s+=m(String.raw`k=2`,n?30:36,y+ch,65,40,n?14:17,BLUE);s+=text('each column is an embedding',mid,n?335:340,n?15:18,BLUE)}
  if(i===4){const sx=x+1.5*cw,sy=y+ch,ex=n?200:800,ey=n?340:195;
   s+=line(sx,sy+ch,mix(sx,ex,p),mix(sy+ch,ey+48,p),BLUE,1.5,false,'5 4');
   s+=group(rect(mix(sx,ex,p)-cw/2,mix(sy,ey,p)-ch,cw,2*ch,BLUE,'white',0,2)+m(String.raw`\begin{matrix}0.60\\0.40\end{matrix}`,mix(sx,ex,p),mix(sy,ey,p),100,100,n?22:27,BLUE),p);
   s+=group(m('h_c',ex,ey-ch-25,100,36,26,BLUE),p);
   if(!n)s+=text('the input vector for “support”',ex,310,15,BLUE);
  }
 }
 if(i===5){
  s+=both(initial,n);if(!n){s+=line(535,208,590,208,BLUE,1.7,true);s+=vector(f0.h,640,208,{size:23});}
  if(n)s+=m(String.raw`h_c=\begin{bmatrix}0.60\\0.40\end{bmatrix}`,93,303,175,100,21,BLUE);
  s+=group(n?rect(210,242,118,176,RUST,'none',0,2):rect(739,111,142,240,RUST,'none',0,2),ease(t));
  s+=line(n?331:884,n?316:212,n?339:902,n?316:212,RUST,1.5);
  s+=m(String.raw`u_{\text{reform}}^{\mathsf T}`,n?367:945,n?316:212,n?66:105,38,n?17:23,RUST);
 }
 if(i===6){
  const p=ease(t),a=n?95:280,b=n?292:720,y=n?100:125;
  s+=m(String.raw`\text{reform's row in }W_{\mathrm{out}}`,a,y-85,n?190:360,26,n?12:18,RUST);
  s+=vector([.4,.6],a,y,{color:RUST,label:String.raw`u_{\text{reform}}^{\mathsf T}`,horizontal:true,size:n?23:30});
  s+=vector(f0.h,b,y,{size:n?23:30,label:String.raw`h_{\text{support}}`});
  s+=m(String.raw`\cdot`,mid,y,40,40,24);
  const px1=n?115:375,px2=n?285:625,py=n?250:255;
  s+=line(a,y+55,px1,py-35,RUST,1.5,true);s+=line(b,y+55,px2,py-35,BLUE,1.5,true);
  s+=group(m(String.raw`0.4\times0.6`,px1,py,n?170:240,48,n?20:29)+m('+',mid,py,45,40,24)+m(String.raw`0.6\times0.4`,px2,py,n?170:240,48,n?20:29),clamp(p*2));
  s+=group(m(String.raw`0.24+0.24=0.48`,mid,n?345:345,n?360:600,54,n?29:34,RUST),clamp(p*2-0.4));
 }
 if(i===7||i===8){
  const p=ease(t),start=n?112:400,base=n?228:655,max=n?130:260,y0=n?85:95,ch=n?44:42;
  s+=text('candidate',n?60:220,y0-25,13,MUTED)+m('z_{co}',start+10,y0-25,100,30,20,RUST)+text('probability',base+max/2,y0-25,13,BLUE);
  for(let j=0;j<6;j++){let y=y0+j*ch;if(i===8&&j===2)s+=rect(n?8:130,y-8,n?384:810,ch-1,RUST,'#fff8f4',0,1.5);
   s+=text(vocab[j],n?93:320,y+18,n?13:16,j===2?RUST:INK,'end');s+=m(fmt(f0.z[j]),start+15,y+13,90,40,20,RUST);
   s+=line(start+65,y+13,base-10,y+13,RULE,1.2,true);
   const width=(f0.p[j]/.3)*max*(i===7?p:1);s+=rect(base,y+1,width,24,'none',j===2?RUST:BLUE);
   s+=text(fmt(f0.p[j],3),base+width+7,y+18,n?12:14,j===2?RUST:BLUE,'start');
  }
  s+=i===7?m(String.raw`\sum_o p(w_o\mid w_c)=1`,mid,n?402:387,380,42,n?19:23,BLUE):m(String.raw`\ell=-\log(0.225)\approx1.492`,mid,n?406:388,n?390:700,44,n?21:26,RUST);
 }
 if(i===9){
  const p=ease(t),model=interpolateModel(initial,first,p),f=forward(model);
  s+=both(model,n,{changed:true});
  if(!n)s+=m(String.raw`p(\text{reform}\mid\text{support})`,510,345,520,42,23)+m(String.raw`${fmt(f0.p[2],3)}\;\longrightarrow\;${fmt(f.p[2],3)}`,510,389,500,42,26,RUST);
  else s+=text('0.225 → '+fmt(f.p[2],3)+'   probability of reform',200,440,12,RUST);
 }
 if(i===10){
  const u=clamp(t)*pairs.length,k=Math.min(pairs.length-1,Math.floor(u)),a=u-k;
  const [c,o]=pairs[k],model=interpolateModel(corpus[k],corpus[k+1],ease(a));
  s+=both(model,n,{selected:c,out:o,changed:true});
  s+=text(`pair ${k+1} / ${pairs.length}:  ${vocab[c]} → ${vocab[o]}`,n?200:500,n?440:370,n?14:20,BLUE);
 }
 if(i===11){
  const p=ease(t);s+=inMatrix(final,n?44:70,n?85:165,n?52:64,n?30:44,1);
  s+=group(outMatrix(final,n?214:745,n?246:117,n?55:65,n?28:38,1),1-.7*p);
  if(!n)s+=text('keep the input embeddings',300,330,20,BLUE);
  else s+=text('keep these columns as embeddings',200,184,13,BLUE);
  s+=group(rect(n?37:62,n?78:157,n?326:400,n?74:104,BLUE,'none',0,2.5),p);
 }
 if(i===12){
  const p=ease(t),ox=n?65:220,oy=n?355:335,ax=n?240:510,ay=n?45:105;
  s+=line(ox,oy,n?365:825,oy,MUTED,1.2,true)+line(ox,oy,ox,35,MUTED,1.2,true);
  s+=text('illustrative projection',n?210:540,n?425:400,12,MUTED);
  const a=[ox+ax,oy-ay],b=[ox+ax*.86,oy-mix(ay*2.2,ay*1.12,p)];
  s+=line(ox,oy,a[0],a[1],BLUE,2.7,true)+line(ox,oy,b[0],b[1],BLUE,2.7,true);
  s+=m(String.raw`h_{\text{voters}}`,a[0]+(n?-10:10),a[1]+30,180,40,n?20:26,BLUE)+m(String.raw`h_{\text{citizens}}`,b[0]-(n?12:0),b[1]-26,200,42,n?20:26,BLUE);
  const theta1=Math.atan2(ay,ax),theta2=Math.atan2(oy-b[1],b[0]-ox),r=n?82:120;
  s+=`<path d="M${ox+r*Math.cos(theta1)} ${oy-r*Math.sin(theta1)} A${r} ${r} 0 0 0 ${ox+r*Math.cos(theta2)} ${oy-r*Math.sin(theta2)}" stroke="${BLUE}" stroke-width="1.8" fill="none"/>`;
  s+=m(String.raw`\theta`,ox+r+22,oy-r*.6,60,40,23,BLUE);
  s+=text('compare input with input',n?210:600,55,n?14:18,BLUE);
 }
 if(i===13){
  const p=ease(t),ox=n?35:180,oy=n?365:350,scale=n?140:155;
  const pt=([x,y])=>[ox+x*scale,oy-y*scale],man=pt(analogyVectors.man),woman=pt(analogyVectors.woman),king=pt(analogyVectors.king),queen=pt(analogyVectors.queen);
  const query=[woman[0]+king[0]-man[0],woman[1]+king[1]-man[1]];
  for(const [name,a] of [['man',man],['woman',woman],['king',king],['queen',queen]]){s+=line(ox,oy,a[0],a[1],BLUE,1.5,true);s+=m(`h_{\\text{${name}}}`,a[0],a[1]+(name==='man'||name==='woman'?30:-23),160,42,n?20:26,BLUE)}
  s+=line(man[0],man[1],king[0],king[1],INK,2.5,true,'5 4');
  const a=[mix(man[0],woman[0],p),mix(man[1],woman[1],p)],b=[mix(king[0],query[0],p),mix(king[1],query[1],p)];
  s+=line(a[0],a[1],b[0],b[1],INK,3,true);
  if(!n)s+=m(String.raw`h_{\text{king}}-h_{\text{man}}`,790,155,340,50,25)+text('a difference between vectors',790,200,15,MUTED);
  s+=text('illustrative projection · relationships are approximate',mid,n?429:408,n?10:13,MUTED);
 }
 if(i===14){
  if(n){s+=m(String.raw`d_c`,70,65,90,45,27)+line(115,65,165,65,BLUE,2,true)+m(String.raw`W_{\mathrm{in}}`,250,65,150,50,30,BLUE);
   s+=line(250,100,250,160,BLUE,2,true)+vector(final.Win.map(row=>row[1]),250,228,{label:'h_c',size:28});
   s+=text('a learned input embedding',200,320,17,BLUE);s+=text('keep one for every word',200,355,14,MUTED);
  }else{s+=m('d_c',100,160,120,60,35)+line(165,160,245,160,BLUE,2,true)+m(String.raw`W_{\mathrm{in}}`,340,160,170,60,36,BLUE)+line(430,160,505,160,BLUE,2,true)+m('h_c',560,160,100,60,35,BLUE)+line(620,160,690,160,RUST,2,true)+m(String.raw`W_{\mathrm{out}}`,790,160,180,60,36,RUST)+line(880,160,940,160,RUST,2,true)+m('p',970,160,60,60,35);
   s+=text('lookup',340,225,16,BLUE)+text('embedding',560,225,16,BLUE)+text('score, then softmax',790,225,16,RUST);s+=line(560,250,560,300,BLUE,2,true)+text('a representation for later tasks',560,345,22,BLUE)}
 }
 return s;
}

function renderScene(i,t,n){
 if(i===17)return renderModern();
 if(i===2)return renderNetwork(t,n,{m,text,line,rect,group,ease,clamp,fmt,vocab,initial,forward,BLUE,RUST,INK,MUTED,RULE});
 if(i===15)return renderCosine(t,n,{m,text,line,ease,mix,fmt,BLUE,RUST,MUTED});
 return render(i>15?i-2:i>2?i-1:i,t,n);
}

const markers=`<defs>${[['blue',BLUE],['rust',RUST],['ink',MUTED]].map(([name,col])=>`<marker id="arrow-${name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M1 1L9 5L1 9" fill="none" stroke="${col}" stroke-width="1.5"/></marker>`).join('')}</defs>`;
let active=-1,lastKey='',pending=false;
function update(){
 pending=false;const bounds=story.getBoundingClientRect(),range=story.offsetHeight-innerHeight;
 const progress=clamp(-bounds.top/range),position=progress*(scenes.length-.001),i=Math.min(scenes.length-1,Math.floor(position)),local=position-i;
 let t=clamp((local-.12)/.64);if(reduced)t=t<.5?0:1;
 const n=innerWidth<=700,key=`${i}:${Math.round(t*300)}:${n}`;
 if(key===lastKey)return;lastKey=key;
 if(active!==i){document.querySelector('.stage').classList.toggle('is-list',i===17);document.querySelector('#modern-alternatives').hidden=i!==17;const sc=scenes[i];document.querySelector('#chapter').textContent=sc.chapter;document.querySelector('#counter').textContent=`${i+1} / ${scenes.length}`;document.querySelector('#scene-title').textContent=sc.title;document.querySelector('#equation').innerHTML=latex(sc.eq);document.querySelector('#caption').textContent=sc.caption;document.querySelector('#detail').textContent=sc.detail;svg.setAttribute('aria-label',sc.alt);document.querySelector('#scope').innerHTML=i===17?'Embeddings in practice':'Skip-gram word2vec<sup><a href="#cbow" aria-label="Footnote 1: alternative architecture">1</a></sup>';active=i;}
 if(i!==17){svg.setAttribute('viewBox',`0 0 ${stem(n)}`);svg.innerHTML=markers+renderScene(i,t,n);}
 document.querySelector('#progress').style.width=`${progress*100}%`;
 story.dataset.scene=String(i);story.dataset.phase=t.toFixed(4);story.dataset.motion=reduced?'reduced':'full';
}
function schedule(){if(!pending){pending=true;requestAnimationFrame(update)}}
function init(){
 // Preserve the original scroll distance per scene when adding the network overview.
 story.style.height=`${100+1300*scenes.length/15}svh`;
 document.querySelectorAll('[data-math]').forEach(el=>el.innerHTML=latex(el.dataset.math));
 document.querySelector('#modern-alternatives').innerHTML=renderModern();
 document.querySelector('#transcript').innerHTML=scenes.map(sc=>`<section><h3>${sc.title}</h3>${sc===modernScene?renderModern(4):`<p>${sc.caption}</p><p>${sc.detail.replace('¹','')}</p><p>${latex(sc.eq)}</p>`}</section>`).join('');
 addEventListener('scroll',schedule,{passive:true});addEventListener('resize',()=>{lastKey='';schedule()});update();
}
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init);else init();

// Exposed immutable metadata is useful for deterministic browser verification.
window.word2vecLesson={sceneCount:scenes.length,reduced,initialProbability:f0.p[2],updatedProbability:forward(first).p[2]};
