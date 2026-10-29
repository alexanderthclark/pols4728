import {createSocksScenes} from './socks-scenes.mjs';
import {tokens,X,WQ,WK,WV,dk,handWeights,uniformWeights,matmul,attention} from './model.mjs';

const C={ink:'#242424',blue:'#234e70',rust:'#a54f32',teal:'#007f73',purple:'#7b4f9d',muted:'#606060',rule:'#dededb'};
const colors=[C.muted,C.blue,C.rust,C.teal];
const full=attention(), masked=attention({causal:true});
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
function matrix(data,{x=100,y=120,cw=58,ch=60,selected=-1,heat=false,mask=false,reveal=1,labels=false,decimals=false}={}){
 let out=''; const rows=data.length, cols=data[0].length;
 if(labels)tokens.forEach((v,i)=>out+=text(v,x-18,y+i*ch+ch/2+8,22,selected===i?C.rust:C.muted,'end'));
 for(let i=0;i<rows;i++)for(let j=0;j<cols;j++){
 const v=data[i][j], blocked=mask&&j>i;
 const opacity=clamp(reveal*rows-i);
 out+=`<g opacity="${opacity}">`;
 let fill=heat?`rgba(35,78,112,${.04+.78*(Number.isFinite(v)?v:0)})`:(i===selected?'#f6ebe6':'#f6f6f3');
 if(blocked)fill='url(#blocked)';
 out+=rect(x+j*cw+2,y+i*ch+2,cw-4,ch-4,fill,'none',4);
 if(blocked&&v===-Infinity)out+=m('-\\infty',x+j*cw+cw/2,y+i*ch+ch/2,cw,ch,25,C.muted);
 else out+=text(Number.isFinite(v)?(decimals?v.toFixed(2):num(v)):'−∞',x+j*cw+cw/2,y+i*ch+ch/2+8,24,heat&&v>.52?'#fff':blocked?'#777':C.ink);
 out+='</g>';
 }
 out+=`<path d="M${x-3} ${y}h-8v${rows*ch}h8 M${x+cols*cw+3} ${y}h8v${rows*ch}h-8" fill="none" stroke="${C.ink}" stroke-width="1.4"/>`;
 return out;
}
function tokenRows(t){
 let out=text('The senator thanked staff',320,50,26,C.muted);
 const y0=130,cw=70,ch=68,x=288;
 out+=m('X',393,96,100,45,32,C.blue);
 out+=text('three features',393,465,18,C.muted);
 tokens.forEach((word,i)=>{
  const y=lerp(55,y0+i*ch+ch/2,ease(t));
  out+=text(word,164,y+8,26,i===2?C.rust:C.ink,'end');
  out+=m(`x_${i+1}`,225,y,70,40,23,C.muted);
 });
 out+=matrix(X,{x,y:y0,cw,ch,selected:2,reveal:clamp(t*1.8)});
 out+=m('T=4,\\quad d=3',320,505,300,40,23,C.muted);
 return out;
}
function cubic(p0,p1,p2,p3,t){return (1-t)**3*p0+3*(1-t)**2*t*p1+3*(1-t)*t*t*p2+t**3*p3}
function mix(weights,t,{values=X,target='x',showWeights=true}={}){
 let out=text('source positions',114,45,18,C.muted)+text('destination',506,45,18,C.muted);
 const result=matmul([weights],values)[0];
 tokens.forEach((word,j)=>{
  const sy=104+j*92,ty=252,weight=weights[j];
  out+=text(word,104,sy-20,25,colors[j]);
  out+=m(`${target}_${j+1}`,104,sy+12,120,42,26,colors[j]);
  out+=`<path d="M170 ${sy}C300 ${sy} 335 ${ty} 450 ${ty}" fill="none" stroke="${colors[j]}" stroke-width="${2+weight*14}" opacity="${weight===0?.1:.15+.6*ease(t)}"/>`;
  const p=ease(t); if(weight>0)out+=circle(cubic(170,300,335,450,p),cubic(sy,sy,ty,ty,p),5+weight*7,colors[j]);
  if(showWeights)out+=rect(253,sy-18,65,30,'#fff')+text(pct(weight),286,sy+4,22,colors[j]);
 });
 out+=rect(451,214,163,78,'#fff',C.rust,8)+text('thanked',533,200,25,C.rust)+m(target==='v'?'y_3':'(AX)_3',533,250,155,55,29,C.rust);
 out+=m(`\\begin{bmatrix}${result.map(num).join('&')}\\end{bmatrix}`,337,462,590,64,30,C.blue);
 out+=text('the resulting row',337,505,18,C.muted);
 return out;
}
function multiply(t){
 const A=Array.from({length:4},()=>[...uniformWeights]); A[2]=[...handWeights];
 let out=m('A',158,58,100,44,32,C.blue)+m('X',423,58,100,44,32,C.blue);
 out+=matrix(A,{x:46,y:100,cw:56,ch:66,selected:2,heat:true});
 out+=m('\\times',301,230,50,50,33,C.muted);
 out+=matrix(X,{x:336,y:100,cw:58,ch:66,selected:Math.min(3,Math.floor(t*4))});
 out+=m('(AX)_3=\\sum_{j=1}^{4}a_{3j}x_j',320,420,610,65,30,C.blue);
 out+=text('one destination row × all source rows',320,485,21,C.muted);
 out+=rect(44,100+2*66,228,66,'none',C.rust,3,'stroke-width="2"');
 return out;
}
function projections(t){
 let out=matrix(X,{x:38,y:155,cw:44,ch:58,selected:2})+m('X',104,122,80,40,30,C.ink);
 out+=text('same input',104,430,20,C.muted);
 out+=`<path d="M188 264C255 264 235 124 316 124 M188 264C255 264 235 390 316 390" fill="none" stroke="${C.rule}" stroke-width="2"/>`;
 out+=rect(230,91,114,60,'#fff',C.blue,4)+m('W_Q',287,121,105,50,28,C.blue);
 out+=rect(230,356,114,60,'#fff',C.teal,4)+m('W_K',287,386,105,50,28,C.teal);
 out+=line(344,121,400,121,C.muted,1.5,'marker-end="url(#arrow)"')+line(344,386,400,386,C.muted,1.5,'marker-end="url(#arrow)"');
 out+=m('Q',494,29,80,40,30,C.blue)+m('K',494,292,80,40,30,C.teal);
 out+=matrix(full.Q,{x:439,y:54,cw:55,ch:45,selected:2,reveal:clamp(t*2)})+matrix(full.K,{x:439,y:318,cw:55,ch:45,selected:2,reveal:clamp(t*2)});
 out+=text('queries',596,135,19,C.blue)+text('keys',592,402,19,C.teal);
 return out;
}
function geometry(t){
 const o={x:85,y:380},unit=112;
 let out='';
 out+=line(o.x,o.y,405,o.y,C.rule,1)+line(o.x,o.y,o.x,60,C.rule,1);
 out+=text('0',o.x-18,o.y+22,19,C.muted);
 [1,2].forEach(v=>{out+=line(o.x+v*unit,o.y-4,o.x+v*unit,o.y+4,C.muted)+text(v,o.x+v*unit,o.y+25,18,C.muted);out+=line(o.x-4,o.y-v*unit,o.x+4,o.y-v*unit,C.muted)+text(v,o.x-17,o.y-v*unit+6,18,C.muted)});
 full.K.forEach(([a,b],i)=>{
  const x=o.x+a*unit,y=o.y-b*unit;
  out+=line(o.x,o.y,x,y,colors[i],i===3?4:2,i===2?'stroke-dasharray="6 5"':'');
  out+=circle(x,y,6,'#fff',`stroke="${colors[i]}" stroke-width="2"`);
  out+=m(`k_${i+1}`,x+(i===1?28:32),y+(i===2?0:0),60,42,25,colors[i]);
 });
 const qx=1,qy=2;
 const active=Math.min(3,Math.floor(t*4));
 out+=line(o.x,o.y,o.x+qx*unit,o.y-qy*unit,C.rust,4,'marker-end="url(#arrow)"')+m('q_3',o.x+qx*unit-10,o.y-qy*unit-28,70,45,30,C.rust);
 out+=text('query · key',533,78,20,C.muted);
 full.K.forEach((k,i)=>{const dot=qx*k[0]+qy*k[1];const y=146+i*67;if(i===active)out+=rect(432,y-28,196,50,'#f6ebe6','none',4);out+=m(`q_3 k_${i+1}^{\\top}`,487,y,100,40,22,C.muted)+text(num(dot),595,y+8,25,colors[i]);});
 out+=m('q_3=[1\\;2]',320,455,300,50,27,C.rust)+text('Longer, better-aligned keys can score more highly.',320,506,19,C.muted);
 return out;
}
function scores(t){
 let out=text('source key',375,37,20,C.muted);
 tokens.forEach((word,j)=>out+=text(word,252+j*88,81,22,colors[j],'middle','class="column-label"'));
 out+=matrix(full.S,{x:208,y:108,cw:88,ch:74,selected:2,labels:true,reveal:clamp(t*1.5)});
 out+=m('S=QK^\\top/\\sqrt{2}',362,453,400,55,32,C.blue);
 out+=text('one query per row; one key per column',320,506,20,C.muted);
 return out;
}
function bars(probabilities,t,{causal=false,fromScores=false}={}){
 const phase=fromScores?(t<.3?0:t<.62?1:2):2;
 const values=phase===0?full.S[2]:phase===1?full.S[2].map(Math.exp):probabilities;
 const max=phase===0?3:phase===1?18:1;
 const heading=phase===0?'1. Start with the scaled scores.':phase===1?'2. Exponentiate each score.':causal?'3. Normalize over the allowed sources.':'3. Divide by the sum of exponentials.';
 let out=text(heading,320,46,22,C.muted);
 const base=385,left=82,step=137,w=83,maxH=245;
 values.forEach((v,j)=>{
  const h=v/max*maxH;
  if(causal&&j===3)out+=rect(left+j*step,105,w,280,'url(#blocked)','none',4);
  out+=rect(left+j*step,base-h,w,h,colors[j],'none',4,`opacity="${v===0?.25:.85}"`);
  out+=text(phase===2?pct(v):num(v),left+j*step+w/2,base-h-14,27,colors[j]);
  out+=text(tokens[j],left+j*step+w/2,421,23,colors[j]);
 });
 out+=line(60,385,615,385,C.rule,1.5);
 out+=line(60,140,615,140,C.rule,1,'stroke-dasharray="2 6"')+text(phase===2?'100%':max,57,132,16,C.muted);
 const formula=phase===0?'s_3=[0,\\sqrt{2},\\sqrt{2},2\\sqrt{2}]':phase===1?`\\sum_j e^{s_{3j}}=${full.S[2].map(Math.exp).reduce((a,b)=>a+b,0).toFixed(2)}`:'\\sum_j a_{3j}=1';
 out+=m(formula,320,484,600,50,29,C.blue);
 return out;
}
function attentionGrid(t,causal=false){
 const A=causal?masked.A:full.A;
 let out=text('read from →',373,38,20,C.muted);
 tokens.forEach((word,j)=>out+=text(word,252+j*88,81,22,colors[j],'middle','class="column-label"'));
 out+=matrix(A,{x:208,y:108,cw:88,ch:74,selected:2,labels:true,heat:true,mask:causal,decimals:true,reveal:clamp(t*1.6)});
 out+=m(causal?'A_{\\mathrm{causal}}':'A',364,454,230,55,33,C.blue);
 out+=text('Every row sums to 1 (before rounding).',320,506,20,C.muted);
 return out;
}
function values(t){
 const V=full.V;
 let out=m('X',139,69,90,44,32,C.ink)+matrix(X,{x:52,y:108,cw:58,ch:65,selected:2});
 out+=line(250,235,397,235,C.muted,1.5,'marker-end="url(#arrow)"')+m('W_V',320,202,110,50,29,C.teal);
 out+=m('V',501,69,90,44,32,C.teal)+matrix(V,{x:443,y:108,cw:58,ch:65,selected:2,reveal:clamp(t*1.5)});
 out+=text('input features',139,412,21,C.muted)+text('value features',501,412,21,C.muted);
 out+=m('v_j=x_jW_V',320,479,400,60,31,C.teal);
 return out;
}
function output(t){
 let out=m('Y=AV',142,74,250,55,34,C.rust)+matrix(full.Y,{x:84,y:122,cw:58,ch:67,selected:2});
 out+=line(220,260,386,260,C.muted,1.5,'marker-end="url(#arrow)"')+m('W_O',307,223,130,50,28,C.blue);
 out+=m('YW_O',490,74,250,55,34,C.blue)+matrix(full.head,{x:403,y:122,cw:58,ch:67,selected:2,reveal:clamp(t*1.5)});
 out+=m('AVW_O=AX(W_VW_O)',320,467,610,60,30,C.blue);
 return out;
}
function maskScores(t){
 const progress=ease(t);
 let out=text('future sources',463,38,20,C.muted);
 tokens.forEach((word,j)=>out+=text(word,252+j*88,81,22,colors[j],'middle','class="column-label"'));
 const data=full.S.map((r,i)=>r.map((v,j)=>j>i&&progress>.45?-Infinity:v));
 out+=matrix(data,{x:208,y:108,cw:88,ch:74,selected:2,labels:true,mask:progress>.45});
 for(let i=0;i<4;i++)for(let j=i+1;j<4;j++)out+=rect(210+j*88,110+i*74,84,70,'url(#blocked)','none',4,`opacity="${progress}"`);
 if(progress>.45)for(let i=0;i<4;i++)for(let j=i+1;j<4;j++)out+=m('-\\infty',252+j*88,145+i*74,80,60,28,C.muted);
 out+=m('j>i\\;\\Longrightarrow\\;s_{ij}+M_{ij}=-\\infty',320,456,620,60,27,C.rust);
 out+=text('Mask scores before applying softmax.',320,510,21,C.muted);
 return out;
}
function finalDiagram(t){
 let out='';
 const x0=54,y=200;
 out+=m('X',x0,250,80,60,38,C.ink);
 out+=line(93,250,180,250,C.muted,1.7,'marker-end="url(#arrow)"');
 out+=rect(190,191,170,118,'#edf5f2','none',8)+m('W_V',275,234,160,60,34,C.teal)+text('map features',275,282,22,C.teal);
 out+=line(366,250,410,250,C.muted,1.7,'marker-end="url(#arrow)"');
 out+=rect(422,191,170,118,'#f0f5f7','none',8)+m('A(X)',507,234,160,60,34,C.blue)+text('mix tokens',507,282,22,C.blue);
 out+=`<path d="M55 214V112H507V181" fill="none" stroke="${C.rust}" stroke-width="2" stroke-dasharray="6 5"/>`;
 out+=text('queries and keys set the weights',320,75,22,C.rust);
 out+=m('Y=\\underbrace{A(X)}_{\\text{across tokens}}\\,\\underbrace{XW_V}_{\\text{within tokens}}',320,416,620,95,31,C.blue);
 out+=text('The attention weights depend on the input.',320,506,21,C.muted);
 return out;
}

const generalScenes=[
 {part:'01 · Mixing tokens',label:'A representation for each position',title:'One token. One row.',caption:'Start with four tokens and three features per token. Stack their representations into a matrix.',equation:'X\\in\\mathbb R^{T\\times d}',detail:'The numbers are illustrative features. Here, T = 4 and d = 3; the features are not labeled semantic properties.',visual:'Each row stays attached to its token position.',draw:tokenRows},
 {part:'01 · Mixing tokens',label:'Begin with the simplest mixture',title:'Read a little from everyone.',caption:'To build a new row for “thanked,” first give every source the same weight. The result is their average.',equation:'(AX)_3=\\tfrac14\\sum_{j=1}^{4}x_j',detail:'For now, values equal the input rows and there is no causal mask. All four positions can contribute.',visual:'The four paths carry equal weight.',draw:t=>mix(uniformWeights,t)},
 {part:'01 · Mixing tokens',label:'Change the weights',title:'A different mixture changes the result.',caption:'Let “thanked” read more from “senator.” The source rows stay fixed; only their contributions change.',equation:'(AX)_3=\\sum_{j=1}^{4}a_{3j}x_j',detail:'These first weights are chosen by hand. Soon, queries and keys will generate them from the input.',visual:'A thicker path means a larger contribution, not a different source vector.',draw:t=>mix(uniformWeights.map((v,i)=>lerp(v,handWeights[i],ease(t))),t)},
 {part:'01 · Mixing tokens',label:'Read the multiplication',title:'Attention multiplies from the left.',caption:'Row 3 of the attention matrix tells destination 3 how to combine all four source rows. The same operation works for every destination.',equation:'\\underbrace{A}_{T\\times T}\\underbrace{X}_{T\\times d}',detail:'In aᵢⱼ, i is the destination and j is the source. Every attention row sums to one.',visual:'Highlighted row: the weights used to construct the new row for “thanked.”',draw:multiply},
 {part:'02 · Queries and keys',label:'Let the input determine the weights',title:'Two views of the same token.',caption:'A query describes what a destination will match against. A key is the representation a source offers for that comparison.',equation:'\\begin{aligned}Q&=XW_Q\\\\K&=XW_K\\end{aligned}',detail:'The two maps are learned parameters, shared across positions. Our small example uses fixed illustrative maps and omits biases.',visual:'Right multiplication transforms features without mixing token positions.',draw:projections},
 {part:'02 · Queries and keys',label:'Compare one query with every key',title:'A match is a dot product.',caption:'Compare the query for “thanked” with each source key. A large positive dot product gives that source a high score.',equation:'s_{3j}=\\frac{q_3k_j^\\top}{\\sqrt{d_k}}',detail:'The query stays fixed while the comparisons are highlighted. Dot products depend on length as well as direction; these are not cosine similarities.',visual:'At the final query, “staff” has the largest dot product. The table shows unscaled dot products.',draw:geometry},
 {part:'02 · Queries and keys',label:'All comparisons at once',title:'Every query meets every key.',caption:'Multiplying Q by the transpose of K computes every destination–source score. Divide by the square root of the key dimension.',equation:'S=\\frac{QK^\\top}{\\sqrt{d_k}}',detail:'Here dₖ = 2. The scale controls how dot-product magnitudes grow with dimension. Scores can be negative and need not sum to one.',visual:'Rows are queries; columns are keys. The selected destination is still “thanked.”',draw:scores},
 {part:'02 · Queries and keys',label:'Turn scores into a mixture',title:'Softmax makes the weights.',caption:'Exponentiate the scores, then divide by their sum within this row. Larger scores receive more weight, and the weights add to one.',equation:'a_{ij}=\\frac{e^{s_{ij}}}{\\sum_{\\ell=1}^{T}e^{s_{i\\ell}}}',detail:'The scroll reveals scores, exponentials, then weights. Each panel labels its scale. The final probabilities are over source positions, not vocabulary words.',visual:'For “thanked,” the largest weight is now on “staff.”',draw:t=>bars(full.A[2],t,{fromScores:true})},
 {part:'02 · Queries and keys',label:'A separate mixture for every destination',title:'Each row asks its own question.',caption:'Apply softmax independently to every row. Each destination now has its own distribution over the same four sources.',equation:'A=\\operatorname{softmax}_{\\mathrm{row}}(S)',detail:'Attention is generally asymmetric. Different query and key maps, and separate normalization for each destination, allow aᵢⱼ and aⱼᵢ to differ.',visual:'The first query is zero, so its four scores are tied and its weights are uniform.',draw:t=>attentionGrid(t)},
 {part:'03 · Values',label:'Choose the information to carry',title:'Keys match. Values travel.',caption:'A key helps determine the weight. A value is the vector that actually enters the weighted sum. It comes from another learned map.',equation:'V=XW_V',detail:'This map transforms each row from three input features to two value features. It does not mix positions.',visual:'The same weights can now carry transformed features.',draw:values},
 {part:'03 · Values',label:'The weighted value output',title:'Use the weights to gather values.',caption:'For each destination, multiply every source value by its attention weight and add the contributions.',equation:'\\begin{aligned}Y&=AV\\\\y_i&=\\sum_j a_{ij}v_j\\end{aligned}',detail:'With values equal to X, this was AX. With a learned value map, it becomes AXWᵥ. Queries and keys still control the weights.',visual:'For “thanked,” the computed output is approximately [0.81, 0.95].',draw:t=>mix(full.A[2],t,{values:full.V,target:'v'})},
 {part:'03 · Values',label:'An optional output map',title:'Write the mixture in another feature space.',caption:'An output map can turn each mixed value vector back into d features. It changes what a head writes, while preserving its attention weights.',equation:'AVW_O=AX(W_VW_O)',detail:'This is the paper’s distinction between QK and OV: where a head reads, and what information it writes. We stop at the attention contribution.',visual:'Left multiplication mixes positions. Right multiplication transforms features.',draw:output},
 {part:'04 · Causal masking',label:'Restrict the available sources',title:'What if future tokens are hidden?',caption:'So far, “thanked” can read “staff,” a later position. In causal attention, a destination can read only earlier positions and itself.',equation:'j\\le i\\quad\\text{is allowed}',detail:'This restriction is useful when later tokens are not yet available. “Causal” here means no reading ahead in the sequence.',visual:'The future source “staff” currently receives about 65% of this row’s attention.',draw:t=>mix(full.A[2],t,{values:full.V,target:'v'})},
 {part:'04 · Causal masking',label:'Mask before normalization',title:'Remove the forbidden scores.',caption:'Add negative infinity wherever the source lies to the right of the destination. Those entries become exactly zero after softmax.',equation:'A=\\operatorname{softmax}_{\\mathrm{row}}(S+M)',detail:'Mᵢⱼ = 0 when j ≤ i, and −∞ otherwise. The diagonal remains available: a position may read itself.',visual:'The upper triangle is blocked. The diagonal and lower triangle remain available.',draw:maskScores},
 {part:'04 · Causal masking',label:'Renormalize the sources that remain',title:'The remaining weights still sum to one.',caption:'With “staff” removed, the weight is redistributed over the three allowed positions. “Senator” and “thanked” now receive about 45% each.',equation:'\\begin{aligned}a_{3j}^{\\mathrm{causal}}&=\\frac{e^{s_{3j}}}{\\sum_{\\ell\\le3}e^{s_{3\\ell}}}\\\\&\\quad (j\\le3)\\end{aligned}',detail:'Simply zeroing a weight after softmax would leave the row short of one. Masking the scores changes the denominator too.',visual:'All changes shown are recomputed from the same original scores.',draw:t=>bars(masked.A[2],t,{causal:true})},
 {part:'04 · Causal masking',label:'The complete causal pattern',title:'Every position respects the same rule.',caption:'The first position can read only itself. Later rows can read progressively more of the sequence. No row reads a future token.',equation:'A_{ij}=0\\quad\\text{for }j>i',detail:'Each allowed row is normalized separately. The final row is unchanged because all four positions are already available to it.',visual:'The mask changes which sources are available, not the direction of multiplication.',draw:t=>attentionGrid(t,true)},
 {part:'05 · One attention layer',label:'Put the pieces together',title:'A mixture chosen by the input.',caption:'Queries and keys choose where to read. Values choose what to carry. Left multiplication combines information across token positions.',equation:'Y=A(X)XW_V',detail:'Once A is fixed, the weighted sum is linear in the values. The full attention operation is generally nonlinear because A depends on X.',visual:'The same mechanism applies with or without a causal mask.',draw:finalDiagram}
];

function renderMathNodes(root=document){root.querySelectorAll('[data-math]').forEach(el=>el.innerHTML=tex(el.dataset.math))}
renderMathNodes();
// Keep mathematical labels in prose typeset from LaTeX too.
generalScenes[3].detail='In <span data-math="a_{ij}"></span>, <span data-math="i"></span> is the destination and <span data-math="j"></span> is the source. Every attention row sums to one.';
generalScenes[6].detail='Here <span data-math="d_k=2"></span>. The scale controls how dot-product magnitudes grow with dimension. Scores can be negative and need not sum to one.';
generalScenes[8].detail='Attention is generally asymmetric. Different query and key maps, and separate normalization for each destination, allow <span data-math="a_{ij}"></span> and <span data-math="a_{ji}"></span> to differ.';
generalScenes[10].detail='With values equal to <span data-math="X"></span>, this was <span data-math="AX"></span>. With a learned value map, it becomes <span data-math="AXW_V"></span>. Queries and keys still control the weights.';
generalScenes[13].detail='<span data-math="M_{ij}=0"></span> when <span data-math="j\\le i"></span>, and <span data-math="-\\infty"></span> otherwise. The diagonal remains available: a position may read itself.';
generalScenes[0].detail='The numbers are illustrative features. Here, <span data-math="T=4"></span> and <span data-math="d=3"></span>; the features are not labeled semantic properties.';
// A complete one-head task comes before the general matrix calculation.
const socksScenes=createSocksScenes({m,text,line,rect,circle,C,ease,lerp,clamp,num});
generalScenes[0].title='Now follow four tokens.';
generalScenes[0].caption='The two-token model used one-dimensional queries, keys, and values. Now unpack the same operations with larger vectors, then add a causal mask.';
generalScenes[7].discretePhases=true;
generalScenes.forEach(scene=>{scene.part=scene.part.replace(/^\d{2}/, n=>String(Number(n)+3).padStart(2,'0'));});
const scenes=[...socksScenes,...generalScenes];
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
