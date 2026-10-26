import {vectors as V,candidates,unit,add,scale,dot,norm,cosine,query,rank,cosmulScore,rankCosmul} from './math.js';
import {points,projectionMetadata} from './projection.js';
const BLUE='#234e70',RUST='#a54f32',INK='#242424',MUTED='#606060',RULE='#d9d9d5';
const scenes=[
{chapter:'06 · Stored vectors',title:'Start with a learned vector for each word.',eq:String.raw`w\longmapsto h_w\in\mathbb R^k`,caption:'Training is finished. A lookup now returns the stored vector for a word.',detail:'Our arithmetic uses invented 2D vectors. The pretrained Google News model has 300 coordinates per word.',alt:'Three vocabulary entries map to fixed, illustrative two-coordinate vectors.'},
{chapter:'07 · Comparing directions',title:'Normalize length. Keep direction.',eq:String.raw`\hat h_w=\frac{h_w}{\lVert h_w\rVert}`,caption:'Cosine similarity compares directions. Rescaling an arrow to length one does not turn it.',detail:'The angle between water and dirt stays the same while their lengths change.',alt:'Water and dirt arrows shrink onto the unit circle without changing their angle.'},
{chapter:'08 · Composing a query',title:'Add directions, tip to tail.',eq:String.raw`q=\hat h_{\mathrm{water}}+\hat h_{\mathrm{dirt}}`,caption:'Move the second arrow to the tip of the first. The sum runs from the origin to the new endpoint.',detail:'The named-word query normalizes each source vector before adding it.',alt:'The normalized dirt vector moves to the tip of the normalized water vector; a new arrow shows their sum.'},
{chapter:'08 · Composing a query',title:'Find the nearest direction.',eq:String.raw`s(x)=\hat h_x^{\mathsf T}\frac{q}{\lVert q\rVert}`,caption:'Compare every candidate with the query direction and rank the cosine scores.',detail:'Here, puddle ranks first. This is a result of our toy coordinates, not a claim about pretrained vectors.',alt:'The summed query shrinks to unit length. Candidates are ranked by exact cosine similarity.'},
{chapter:'09 · What gets normalized?',title:'Addition depends on the lengths you keep.',eq:String.raw`h_{\mathrm{dirt}}+h_{\mathrm{rain}}\quad\text{versus}\quad\hat h_{\mathrm{dirt}}+\hat h_{\mathrm{rain}}`,caption:'The raw sum gives the longer dirt vector more influence. Normalizing each source changes the answer.',detail:'In this toy example: raw addition ranks mud first; adding unit vectors ranks puddle first. Both exclude dirt and rain here.',alt:'Two side-by-side sums compare raw vectors and normalized vectors. The longer dirt vector pulls the raw query toward mud.'},
{chapter:'10 · Signed directions',title:'Subtraction reverses an arrow.',eq:String.raw`q=\hat h_{\mathrm{water}}+\hat h_{\mathrm{dirt}}-\hat h_{\mathrm{clean}}-\hat h_{\mathrm{dry}}`,caption:'A negative word contributes the opposite of its vector. The query is still a direction to search.',detail:'Subtracting clean or dry is not logical negation. An analogy uses the same signed-sum operation.',alt:'Water and dirt are added, then the clean and dry vectors are reversed and joined tip to tail.'},
{chapter:'11 · Another scoring rule',title:'Multiply similarities, not coordinates.',eq:String.raw`a_x=\tfrac{1+\cos(h_x,h_{\mathrm{water}})}2,\quad b_x=\tfrac{1+\cos(h_x,h_{\mathrm{dirt}})}2`,caption:'Multiplicative scoring rewards a candidate that is similar to both positive words.',detail:'The factors are shifted cosine similarities. Their product is a score, not a probability.',alt:'For each candidate, two shifted cosine similarities are multiplied to form its multiplicative score.'},
{chapter:'12 · From angles to distances',title:'Unit vectors connect cosine and distance.',eq:String.raw`\lVert\hat h_a-\hat h_b\rVert^2=2\bigl(1-\cos(h_a,h_b)\bigr)`,caption:'On the unit sphere, smaller Euclidean distance means larger cosine similarity.',detail:'This equality holds in the original embedding space. It explains the normalization before the map is fitted.',alt:'A chord joins two unit-vector endpoints. Its squared length equals twice one minus their cosine similarity.'},
{chapter:'13 · Making a map',title:'A map emphasizes local neighborhoods.',eq:String.raw`\hat h_i\in\mathbb R^k\quad\xrightarrow{\;\mathrm{t\text{-}SNE}\;}\quad y_i\in\mathbb R^2`,caption:'t-SNE searches for a two-dimensional arrangement that represents neighborhood relationships.',detail:'Actual fit to 18 synthetic 6D unit vectors; perplexity 8. This transition reveals the fitted positions, not optimization steps.',alt:'Eighteen synthetic vector entries move from a list into their fitted t-SNE positions. Three groups use different shapes.'},
{chapter:'13 · Reading a map',title:'The map’s axes do not carry meaning.',eq:String.raw`y_i\neq\text{the original word vector}`,caption:'Rotating the whole map preserves every distance in it. The horizontal direction has no inherent interpretation.',detail:'Inspect neighborhoods. Do not read cluster areas or gaps as faithful original distances, or do vector arithmetic on the map.',alt:'The same fitted t-SNE map rotates while keeping all pairwise map distances fixed.'}
];
const clamp=x=>Math.max(0,Math.min(1,x)),ease=x=>{x=clamp(x);return x*x*(3-2*x)},mix=(a,b,t)=>a+(b-a)*t,fmt=(x,n=3)=>(Math.abs(x)<.00001?0:x).toFixed(n);
const lerp=(a,b,t)=>a.map((x,i)=>mix(x,b[i],t));
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;');
const cache=new Map();function latex(tex){if(!cache.has(tex))cache.set(tex,katex.renderToString(tex,{throwOnError:true,output:'htmlAndMathml'}));return cache.get(tex)}
function m(tex,x,y,w=260,h=40,size=22,color=INK){return `<foreignObject x="${x-w/2}" y="${y-h/2}" width="${w}" height="${h}" style="overflow:visible;color:${color};font-size:${size}px"><div xmlns="http://www.w3.org/1999/xhtml" class="mathbox">${latex(tex)}</div></foreignObject>`}
function text(s,x,y,size=16,color=INK,anchor='middle'){return `<text x="${x}" y="${y}" fill="${color}" font-size="${size}" text-anchor="${anchor}">${esc(s)}</text>`}
function line(a,b,color=RULE,width=1.5,arrow=false,dash=''){return `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="${color}" stroke-width="${width}" ${arrow?`marker-end="url(#${color===BLUE?'blue':color===RUST?'rust':'ink'})"`:''} ${dash?`stroke-dasharray="${dash}"`:''}/>`}
function circle(x,y,r,fill='white',stroke=RULE,width=1){return `<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${width}"/>`}
function group(s,op=1){return `<g opacity="${clamp(op)}">${s}</g>`}
function plane(n,kind='unit'){
 const o=kind==='unit'?(n?[160,248]:[270,240]):(n?[100,295]:[140,295]),k=kind==='unit'?(n?112:145):(n?87:140);
 return {o,k,p:v=>[o[0]+k*v[0],o[1]-k*v[1]]};
}
function axes(g,n,unitCircle=false){let s=line([g.o[0]-20,g.o[1]],[n?370:570,g.o[1]])+line([g.o[0],n?395:380],[g.o[0],45]); if(unitCircle)s+=circle(...g.o,g.k,'none',RULE,1.5);s+=text('0',g.o[0]-13,g.o[1]+17,12,MUTED);return s}
function arrow(g,v,color=BLUE,label='',from=[0,0],labelOffset=[12,-10],size=18){const a=g.p(from),b=g.p(add(from,v));return line(a,b,color,2.6,true)+(label?m(label,b[0]+labelOffset[0],b[1]+labelOffset[1],150,34,size,color):'')}
function scoreRows(rows,x,y,width=260,step=47,highlight=0,phase=1){let s='';rows.forEach((row,j)=>{let yy=y+j*step;s+=text(row.word,x,yy,16,j===highlight?RUST:INK,'start');s+=text(fmt(row.score),x+width,yy,15,j===highlight?RUST:MUTED,'end');s+=line([x,yy+10],[x+width,yy+10],RULE,1);s+=line([x,yy+10],[x+Math.max(0,row.score)*width*phase,yy+10],j===highlight?RUST:BLUE,3)});return s}
const qWater=query([V.water,V.dirt]),qRaw=query([V.dirt,V.rain],[],{normalizeSources:false}),qUnit=query([V.dirt,V.rain]);
const ranking=rank(qWater,candidates,['water','dirt']);
function render(i,t,n){let s='',w=n?400:1000,mid=w/2,p=ease(t);
 if(i===0){
  const rows=['water','dirt','rain'],xs=n?[72,270]:[275,730];
  s+=text('word',xs[0],n?78:75,14,MUTED)+text('stored coordinates',xs[1],n?78:75,14,MUTED);
  rows.forEach((word,j)=>{let y=(n?137:140)+j*85;s+=text(word,xs[0],y+6,23,BLUE);s+=line([xs[0]+(n?42:75),y],[mix(xs[0]+(n?42:75),xs[1]-(n?65:120),p),y],BLUE,1.6,true);s+=group(m(String.raw`\begin{bmatrix}${V[word].join('&')}\end{bmatrix}`,xs[1],y,n?170:250,50,n?23:28),p)});
  s+=text('fixed toy vectors · no training in this lesson',mid,n?417:390,n?12:15,MUTED);
 }
 if(i===1){
  const g=plane(n,'raw');s+=axes(g,n)+circle(...g.o,g.k,'none',RULE);
  for(const [word,col,off] of [['water',BLUE,[-15,-23]],['dirt',RUST,[5,-24]]]){s+=arrow(g,lerp(V[word],unit(V[word]),p),col,(p===1?String.raw`\hat h_{\mathrm{${word}}}`:String.raw`h_{\mathrm{${word}}}`),[0,0],off,n?18:23)}
  const a=Math.atan2(V.dirt[1],V.dirt[0]),b=Math.atan2(V.water[1],V.water[0]);const r=g.k*.37;
  s+=`<path d="M${g.o[0]+r*Math.cos(a)} ${g.o[1]-r*Math.sin(a)} A${r} ${r} 0 0 0 ${g.o[0]+r*Math.cos(b)} ${g.o[1]-r*Math.sin(b)}" fill="none" stroke="${MUTED}"/>`;
  s+=m(String.raw`\theta`,g.o[0]+r*.95,g.o[1]-r*.82,60,30,20,MUTED);
  s+=m(String.raw`\cos(\theta)=${fmt(cosine(V.water,V.dirt))}`,n?205:775,n?405:150,n?330:310,45,n?23:27,BLUE);
  if(!n)s+=text('The angle stays fixed.',775,220,20)+text('Only vector lengths change.',775,250,17,MUTED);
 }
 if(i===2){
  const g=plane(n,'raw'),a=unit(V.water),b=unit(V.dirt),from=scale(a,p);s+=axes(g,n);s+=arrow(g,a,BLUE,String.raw`\hat h_{\mathrm{water}}`,[0,0],[-25,-24]);s+=arrow(g,b,RUST,String.raw`\hat h_{\mathrm{dirt}}`,from,[15,-25]);s+=group(arrow(g,qWater,INK,'q',[0,0],[12,18]),p);
  if(!n)s+=m(String.raw`q=\begin{bmatrix}${qWater.map(v=>fmt(v,2)).join(String.raw`\\`)}\end{bmatrix}`,780,195,280,100,30,BLUE)+text('Translation does not change a vector.',780,300,16,MUTED);
  else s+=m(String.raw`q=(${qWater.map(v=>fmt(v,2)).join(',')})^{\mathsf T}`,205,405,350,42,22,BLUE);
 }
 if(i===3){
  const g=n?{o:[110,175],k:80,p:v=>[110+80*v[0],175-80*v[1]]}:plane(false,'unit');s+=circle(...g.o,g.k,'none',RULE);s+=arrow(g,lerp(qWater,unit(qWater),p),BLUE,'q',[0,0],[24,-16]);
  for(const word of ['puddle','mud','river']){const pos=g.p(unit(V[word]));s+=line(g.o,pos,RULE,1.1);s+=circle(...pos,word==='puddle'?5:3,word==='puddle'?RUST:'white',word==='puddle'?RUST:MUTED)}
  s+=text('nearest direction',n?285:770,n?95:70,14,MUTED);
  s+=scoreRows(ranking.slice(0,n?4:5),n?210:650,n?133:120,n?155:260,n?41:52,0,p);
  s+=text('Source words are excluded.',n?200:770,n?377:402,n?13:15,MUTED);
 }
 if(i===4){
  const queries=[qRaw,qUnit];for(let j=0;j<2;j++){
   const cx=n?102+j*198:275+j*450,cy=n?280:290,k=n?45:100;
   const g={o:[cx-(n?60:110),cy],k,p:v=>[cx-(n?60:110)+k*v[0],cy-k*v[1]]};
   const a=j?unit(V.dirt):V.dirt,b=j?unit(V.rain):V.rain;
   s+=text(j?'normalize each, then add':'add the raw vectors',cx,n?72:55,n?13:19,j?BLUE:RUST);
   s+=arrow(g,a,BLUE,'',[0,0]);s+=arrow(g,b,RUST,'',scale(a,p));s+=group(arrow(g,queries[j],INK),p);const da=g.p(scale(a,.5)),ra=g.p(add(scale(a,p),scale(b,.5)));s+=text('dirt',da[0],da[1]+20,n?12:16,BLUE)+text('rain',ra[0]+21,ra[1],n?12:16,RUST);
   const end=g.p(queries[j]);s+=m(j?String.raw`q_{\mathrm{unit}}`:String.raw`q_{\mathrm{raw}}`,end[0],end[1]-20,n?110:180,32,n?19:25);
   s+=text(j?'puddle':'mud',cx,n?350:365,22,j?BLUE:RUST)+text('nearest candidate',cx,n?378:394,12,MUTED);
  }
  if(!n)s+=line([500,90],[500,370],RULE);
 }
 if(i===5){
  const g=n?{o:[65,330],k:80,p:v=>[65+80*v[0],330-80*v[1]]}:{o:[145,325],k:130,p:v=>[145+130*v[0],325-130*v[1]]};
  const terms=[unit(V.water),unit(V.dirt),scale(unit(V.clean),-1),scale(unit(V.dry),-1)];let current=[0,0];
  const names=[String.raw`\hat h_{\mathrm{water}}`,String.raw`\hat h_{\mathrm{dirt}}`,String.raw`-\hat h_{\mathrm{clean}}`,String.raw`-\hat h_{\mathrm{dry}}`];
  terms.forEach((term,j)=>{const k=clamp(p*4-j),col=j%2?RUST:BLUE;if(k>0)s+=arrow(g,scale(term,k),col,names[j],current,[j===2?30:j===0?-15:0,j===2?0:j===3?26:-22],n?15:18);current=add(current,scale(term,k))});
  if(p>0)s+=arrow(g,current,INK,'q',[0,0],[20,14]);
  if(n)s+=text('water + dirt − clean − dry',200,410,15,MUTED);
  else s+=text('Negative terms point in reverse.',785,150,19)+text('The result is a search direction.',785,185,17,MUTED);
 }
 if(i===6){
  const top=rankCosmul([V.water,V.dirt],[],candidates,['water','dirt']).slice(0,n?3:4),xs=n?[55,157,245,345]:[195,425,640,855];
  ['candidate','water factor','dirt factor','product'].forEach((s0,j)=>s+=text(s0,xs[j],n?90:65,n?12:15,MUTED));
  top.forEach((row,j)=>{const y=(n?150:125)+j*(n?75:65),a=(1+cosine(V[row.word],V.water))/2,b=(1+cosine(V[row.word],V.dirt))/2;s+=text(row.word,xs[0],y+5,n?17:21,j===0?RUST:INK);s+=m(fmt(a),xs[1],y,85,35,n?19:23,BLUE)+m(fmt(b),xs[2],y,85,35,n?19:23,BLUE);s+=group(m(fmt(row.score),xs[3],y,85,35,n?19:23,j===0?RUST:INK),p);s+=text('×',(xs[1]+xs[2])/2,y+5,18,MUTED)});
  s+=m(String.raw`s_{\mathrm{mul}}(x)=\frac{a_xb_x}{1+10^{-6}}`,mid,n?402:389,n?360:600,44,n?22:26);
 }
 if(i===7){
  const g=plane(n,'unit'),a=unit(V.water),b=unit(V.dirt),pa=g.p(a),pb=g.p(b),d2=dot(add(a,scale(b,-1)),add(a,scale(b,-1)));
  s+=axes(g,n,true)+arrow(g,a,BLUE,String.raw`\hat h_a`,[0,0],[-15,-22])+arrow(g,b,BLUE,String.raw`\hat h_b`,[0,0],[15,-18]);s+=line(pa,lerp(pa,pb,p),RUST,3,false,'6 3');
  s+=m(String.raw`\lVert\hat h_a-\hat h_b\rVert^2=${fmt(d2)}`,n?200:775,n?410:180,n?380:420,50,n?22:26,RUST);
  if(!n)s+=m(String.raw`2(1-${fmt(cosine(a,b))})=${fmt(d2)}`,775,250,420,50,24,MUTED);
 }
 if(i===8||i===9){
  const width=n?350:610,height=n?320:315,center=n?[200,230]:[370,220],angle=i===9?p*Math.PI*.38:0;
  const max=Math.max(...points.map(pt=>Math.hypot(...pt.position))),k=Math.min(width,height)/2/(max||1);
  const target=pt=>[center[0]+k*(pt.position[0]*Math.cos(angle)-pt.position[1]*Math.sin(angle)),center[1]+k*(pt.position[0]*Math.sin(angle)+pt.position[1]*Math.cos(angle))];
  points.forEach((pt,j)=>{const [tx,ty]=target(pt),x=i===8?mix(n?70+(j%6)*52:170+(j%6)*70,tx,p):tx,y=i===8?mix(n?125+Math.floor(j/6)*65:130+Math.floor(j/6)*70,ty,p):ty;let marker='';const col=pt.group==='A'?BLUE:pt.group==='B'?RUST:MUTED;
   if(pt.group==='A')marker=circle(x,y,5,col,col);else if(pt.group==='B')marker=`<rect x="${x-4.5}" y="${y-4.5}" width="9" height="9" fill="${col}"/>`;else marker=`<path d="M${x} ${y-6}l6 11h-12z" fill="${col}"/>`;
   s+=marker;if(pt.id.endsWith('1'))s+=text(pt.id,x+(n?13:18),y-9,n?12:15,col);
  });
  if(!n){s+=text('18 synthetic vectors',805,140,21)+text('6 dimensions → 2 dimensions',805,180,17,MUTED);s+=text(i===8?'Fitted positions':'Same points, same map distances',805,250,18,BLUE)}
  s+=text(i===8?'List positions → fitted t-SNE map':'Only the display orientation changes.',n?200:370,n?427:405,n?13:15,MUTED);
 }
 return s;
}
const markers=`<defs>${[['blue',BLUE],['rust',RUST],['ink',INK]].map(([id,c])=>`<marker id="${id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M1 1L9 5L1 9" fill="none" stroke="${c}" stroke-width="1.5"/></marker>`).join('')}</defs>`;
const story=document.querySelector('#exploration-story'),svg=document.querySelector('#exploration-diagram'),reduced=matchMedia('(prefers-reduced-motion: reduce)').matches||new URLSearchParams(location.search).get('motion')==='reduce';
let active=-1,lastKey='',pending=false;
function update(){pending=false;const bounds=story.getBoundingClientRect(),progress=clamp(-bounds.top/(story.offsetHeight-innerHeight)),position=progress*(scenes.length-.001),i=Math.min(scenes.length-1,Math.floor(position)),local=position-i;let t=clamp((local-.12)/.64);if(reduced)t=t<.5?0:1;const n=innerWidth<=700,key=`${i}:${Math.round(t*300)}:${n}`;if(key===lastKey)return;lastKey=key;
 if(active!==i){const sc=scenes[i];document.querySelector('#exploration-chapter').textContent=sc.chapter;document.querySelector('#exploration-counter').textContent=`${i+1} / ${scenes.length}`;document.querySelector('#exploration-scene-title').textContent=sc.title;document.querySelector('#exploration-equation').innerHTML=latex(sc.eq);document.querySelector('#exploration-caption').textContent=sc.caption;document.querySelector('#exploration-detail').textContent=sc.detail;svg.setAttribute('aria-label',sc.alt);document.querySelector('#exploration-scope').textContent=i>=8?'Synthetic vectors · fitted t-SNE map':'Illustrative vectors · exact calculations';active=i}
 svg.setAttribute('viewBox',`0 0 ${n?'400 460':'1000 440'}`);svg.innerHTML=markers+render(i,t,n);document.querySelector('#exploration-progress').style.width=`${progress*100}%`;story.dataset.scene=String(i);story.dataset.phase=t.toFixed(4);story.dataset.motion=reduced?'reduced':'full';}
function schedule(){if(!pending){pending=true;requestAnimationFrame(update)}}
function init(){document.querySelector('#exploration-transcript').innerHTML=scenes.map(sc=>`<section><h3>${sc.title}</h3><p>${sc.caption}</p><p>${sc.detail}</p><p>${latex(sc.eq)}</p></section>`).join('');addEventListener('scroll',schedule,{passive:true});addEventListener('resize',()=>{lastKey='';schedule()});update()}
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init);else init();
