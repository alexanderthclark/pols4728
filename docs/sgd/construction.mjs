import {relu,data} from './model.mjs';
import {mathText} from './math.mjs';
const colors={blue:'#234e70',rust:'#a54f32',teal:'#007f73',gray:'#767676'};
const clamp=x=>Math.max(0,Math.min(1,x));
const mix=(a,b,t)=>a+(b-a)*t;
const smooth=t=>{t=clamp(t);return t*t*(3-2*t);};
const number=x=>Math.abs(x)<.005?'0':Number(x.toFixed(2)).toString();
const text=(x,y,value,color='#242424',size=12,anchor='start')=>`<text x="${x}" y="${y}" text-anchor="${anchor}" style="fill:${color};font-size:${size}px">${value}</text>`;
const mathCache=new Map();
function math(x,y,tex,color='#242424',size=12,anchor='start'){
 if(!mathCache.has(tex))mathCache.set(tex,globalThis.katex.renderToString(tex,{displayMode:false,output:'htmlAndMathml',throwOnError:true,strict:'error'}));
 const width=size*9,left=anchor==='middle'?x-width/2:anchor==='end'?x-width:x;
 return `<foreignObject x="${left}" y="${y-size*.95}" width="${width}" height="${size*1.8}" class="construction-math"><div xmlns="http://www.w3.org/1999/xhtml" style="font-size:${size}px;color:${color};line-height:1.2;text-align:${anchor==='middle'?'center':anchor==='end'?'right':'left'}">${mathCache.get(tex)}</div></foreignObject>`;
}
const line=(x1,y1,x2,y2,color,width=1,dash='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}" stroke-dasharray="${dash}"/>`;
export const constructionKinds=['pair','input-first','input-second','flip','add','steepen','sgd-opening'];
export function constructionState(kind,p){
 const u=clamp((p-.1)/.72);
 const s={kind,u,w1:1,b1:0,w2:1,b2:0,a1:1,a2:1,c:0,merge:0,reveal:0,reframe:0,symbol:0,dataReveal:0};
 if(kind==='input-first')s.w1=1+u;
 if(kind==='input-second'){s.w1=2;s.w2=1+clamp(u*2);s.b2=-clamp(u*2-1);}
 if(['flip','add','steepen','sgd-opening'].includes(kind)){s.w1=s.w2=2;s.b2=-1;s.a2=kind==='flip'?1-2*u:-1;}
 if(kind==='add'){s.merge=clamp(u/.38);s.reveal=clamp((u-.38)/.62);}
 if(kind==='steepen'){
  s.merge=s.reveal=1;
  s.symbol=smooth((p-.04)/.18);
  s.reframe=smooth((p-.10)/.28);
  s.w1=s.w2=2*16**smooth((p-.40)/.48);
 }
 if(kind==='sgd-opening'){
  s.merge=s.reveal=s.symbol=s.reframe=1;s.w1=s.w2=32;
  s.dataReveal=smooth((p-.08)/.34);
 }
 s.a=s.w1;
 return s;
}
function network(svg,s){
 const rect=svg.getBoundingClientRect(),w=Math.max(260,rect.width),h=Math.max(80,rect.height),compact=h<160,wide=globalThis.innerWidth>=1600;
 svg.setAttribute('viewBox',`0 0 ${w} ${h}`);
 const x0=compact?20:wide?28:24,x1=w*.45,x2=w-(compact?27:wide?42:34),y0=h/2,y1=compact?34:h*.28,y2=compact?h-34:h*.72,r=compact?12:wide?32:26;
 const parameterSize=compact?10:wide?24:18;
 let html=`<defs><marker id="network-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10" fill="none" stroke="#767676" stroke-width="1.5"/></marker></defs>`;
 const connection=(xa,ya,xb,yb,color)=>`<line x1="${xa}" y1="${ya}" x2="${xb}" y2="${yb}" stroke="${color}" stroke-width="1.4" marker-end="url(#network-arrow)"/>`;
 for(const [j,y,color] of [[1,y1,colors.blue],[2,y2,colors.rust]]){
  html+=connection(x0+10,y0,x1-r-3,y,color)+connection(x1+r+3,y,x2-r-3,y0,color);
  html+=`<circle cx="${x1}" cy="${y}" r="${r}" fill="white" stroke="${color}" stroke-width="1.5"/>`+math(x1,y+(wide&&!compact?8:6),`h_${j}`,color,compact?11:wide?26:20,'middle');
  const wy=compact?(j===1?12:h-4):(y+y0)/2+(j===1?-(wide?21:16):(wide?33:27));
  const weightX=(x0+x1)/2;
  if(s.symbol<1)html+=`<g opacity="${1-s.symbol}">`+math(weightX,wy,String.raw`\theta_{${j}1}=${number(s['w'+j])}`,color,parameterSize,'middle')+'</g>';
  if(s.symbol>0)html+=`<g opacity="${s.symbol}">`+math(weightX,wy,String.raw`\theta_{${j}1}=k`,color,parameterSize,'middle')+'</g>';
  html+=math(x1,y+(j===1?-r-(wide&&!compact?16:12):r+(compact?14:wide?30:24)),String.raw`\theta_{${j}0}=${number(s['b'+j])}`,color,parameterSize,'middle');
  html+=math((x1+x2)/2,wy,String.raw`\phi_${j}=${number(s['a'+j])}`,color,parameterSize,'middle');
 }
 html+=`<circle cx="${x0}" cy="${y0}" r="10" fill="white" stroke="#767676"/><circle cx="${x2}" cy="${y0}" r="${r}" fill="white" stroke="${colors.teal}" stroke-width="1.5"/>`;
 html+=math(x0,y0+(compact?4:wide?8:6),'x','#242424',compact?13:wide?24:18,'middle')+math(x2,y0+(compact?4:wide?9:7),String.raw`\sum`,colors.teal,compact?13:wide?32:26,'middle')+math(x2,y0+r+(compact?20:wide?38:30),String.raw`\phi_0=${number(s.c)}`,colors.teal,parameterSize,'middle');
 svg.innerHTML=html;svg.setAttribute('aria-label',`Two hidden neurons: input weight θ₁₁ = ${s.w1}, hidden bias θ₁₀ = ${s.b1}; input weight θ₂₁ = ${s.w2}, hidden bias θ₂₀ = ${s.b2}. Output weights φ₁ = ${s.a1}, φ₂ = ${s.a2}; output bias φ₀ = ${s.c}. Hidden-layer parameters form θ; output-layer parameters form φ.`);
}
const shownIndices=Array.from({length:32},(_,i)=>Math.round(i*(data.length-1)/31));
function learningContext(svg,s,p){
 if(!s.dataReveal)return;
 const {width,height}=svg.getBoundingClientRect(),w=Math.max(260,width),h=Math.max(80,height),compact=h<180,size=compact?12:globalThis.innerWidth>=1600?26:20;
 const interval=globalThis.katex.renderToString(String.raw`[-1,1]`,{output:'htmlAndMathml',throwOnError:true});
 const learn=smooth((p-.48)/.27);
 svg.innerHTML=`<g opacity="${1-s.dataReveal}">${svg.innerHTML}</g><foreignObject x="0" y="0" width="${w}" height="${h}" opacity="${s.dataReveal}" class="learning-context"><div xmlns="http://www.w3.org/1999/xhtml" style="font-size:${size}px;line-height:1.35;height:100%;display:flex;flex-direction:column;justify-content:center;gap:${compact?'.3em':'.9em'}"><p>2,048 inputs, evenly spaced on ${interval}.</p><p>Outcomes from the same step function. No noise.</p><p style="opacity:${learn};color:${colors.blue}">Stochastic gradient descent (SGD) will fit the weights and biases to these data.</p><p style="font: ${compact?10:14}px/1.3 Arial,sans-serif;color:#606060">32 of 2,048 observations shown.</p></div></foreignObject>`;
 svg.setAttribute('aria-label','Fit the network to 2,048 finite, noiseless observations from the same step-function data-generating process. Inputs are evenly spaced between −1 and 1. Thirty-two representative observations are drawn. SGD will learn the weights and biases from these observations.');
}
export function renderConstruction(plot,detail,kind,p){
 const s=constructionState(kind,p),rect=plot.getBoundingClientRect(),w=Math.max(260,rect.width),h=Math.max(170,rect.height),compact=w<500,wide=globalThis.innerWidth>=1600;
 plot.setAttribute('viewBox',`0 0 ${w} ${h}`);
 const top=compact?30:wide?50:42,bottom=h-(compact?44:wide?68:60),gap=compact?22:wide?64:54,margin=compact?24:wide?44:36,panel=(w-gap)/2;
 const united=[margin,w-12];
 const bounds=[[mix(margin,united[0],s.merge),mix(panel-10,united[1],s.merge)],[mix(panel+gap+margin,united[0],s.merge),mix(w-12,united[1],s.merge)]];
 const xmin=mix(-.3,-1,s.reframe),xmax=mix(1.2,1,s.reframe),ymin=mix(-1.6,-.2,s.reframe),ymax=mix(2.6,1.3,s.reframe);
 const X=(x,j)=>bounds[j][0]+(x-xmin)/(xmax-xmin)*(bounds[j][1]-bounds[j][0]);
 const Y=y=>bottom-(y-ymin)/(ymax-ymin)*(bottom-top);
 const h1=x=>relu(s.w1*x+s.b1),h2=x=>relu(s.w2*x+s.b2),f=x=>s.c+h1(x)+s.a2*h2(x);
 let html=`<defs><clipPath id="construction-clip-${plot.id}"><rect x="0" y="${top}" width="${w}" height="${bottom-top}"/></clipPath></defs>`;
 function curve(fn,j,color,opacity=1,dash='',end=xmax,width=3,role=''){let d='';const xs=[xmin,0,...Array.from({length:241},(_,i)=>xmin+(xmax-xmin)*i/240),-s.b2/s.w2,end].filter(x=>x>=xmin&&x<=end).sort((a,b)=>a-b);for(const x of xs)d+=(d?'L':'M')+X(x,j)+','+Y(fn(x));return `<path data-curve="${role}" d="${d}" fill="none" stroke="${color}" stroke-width="${width}" stroke-opacity="${opacity}" stroke-dasharray="${dash}"/>`;}
 const weighted=['flip','add','steepen','sgd-opening'].includes(kind);
 for(let j=0;j<2;j++){
  const axisOpacity=j===1?1-s.merge:1;
  html+=`<g opacity="${axisOpacity}">`;
  for(const y of [-1,0,1,2])html+=`<g opacity="${y===-1||y===2?1-smooth(s.reframe/.4):1}">`+line(bounds[j][0],Y(y),bounds[j][1],Y(y),y===0?'#767676':'#e8e8e8')+text(bounds[j][0]-6,Y(y)+(compact?4:wide?7:6),y,colors.gray,compact?9:wide?20:16,'end')+'</g>';
  if(s.reframe>0)html+=`<g opacity="${s.reframe}">`+line(X(-1,j),bottom,X(-1,j),bottom+4,colors.gray)+text(X(-1,j),bottom+(compact?17:wide?30:24),'−1',colors.gray,compact?9:wide?20:16,'middle')+'</g>';
  for(const x of [0,1])html+=line(X(x,j),bottom,X(x,j),bottom+4,colors.gray)+text(X(x,j),bottom+(compact?17:wide?30:24),x,colors.gray,compact?9:wide?20:16,'middle');
  html+=math(mix(bounds[j][1],(bounds[j][0]+bounds[j][1])/2,s.reframe),bottom+(compact?17:wide?30:24)+s.reframe*(compact?14:wide?24:20),'x',colors.gray,compact?10:wide?20:16,'end')+'</g>';
 }
 const showOutput=s.reveal>0,componentOpacity=(showOutput?.55:1)*(1-s.reframe);
 if(kind==='input-first')html+=curve(x=>relu(x),0,colors.gray,.3,'4 4');
 if(kind==='input-second')html+=curve(x=>relu(x),1,colors.gray,.3,'4 4');
 if(kind==='flip')html+=curve(h2,1,colors.rust,.25,'4 4');
 html+=`<g clip-path="url(#construction-clip-${plot.id})">`+curve(h1,0,colors.blue,componentOpacity,showOutput?'5 4':'',xmax,3,'first')+curve(x=>(weighted?s.a2:1)*h2(x),1,colors.rust,componentOpacity,showOutput?'5 4':'',xmax,3,'second')+'</g>';
 const titles=weighted?[String.raw`\phi_1h_1(x)`,String.raw`\phi_2h_2(x)`]:['h_1(x)','h_2(x)'];
 const titleY=compact?16:wide?30:24,titleSize=compact?11:wide?24:20;
 html+=`<g opacity="${1-s.reframe}">`+math(bounds[0][0],titleY,titles[0],colors.blue,titleSize);
 if(kind==='flip'){
  const x=bounds[1][0],second=x+(compact?37:wide?84:70),swatch=compact?12:wide?26:22,space=compact?4:6,lineY=titleY-titleSize*.28;
  html+='<g class="contribution-key">'+line(x,lineY,x+swatch,lineY,colors.rust,3,'4 4')+mathText(x+swatch+space,titleY,'h_2','','start',colors.rust,{size:titleSize});
  html+=line(second,lineY,second+swatch,lineY,colors.rust,3)+mathText(second+swatch+space,titleY,String.raw`\phi_2h_2`,'','start',colors.rust,{size:titleSize})+'</g>';
 }else html+=math(s.merge?bounds[0][0]+(compact?80:wide?170:140):bounds[1][0],titleY,titles[1],colors.rust,titleSize);
 html+='</g>';
 if(showOutput){
  html+=curve(f,0,colors.teal,s.reveal*(1-s.dataReveal),'',xmax,3.5,'sum')+`<g opacity="${1-s.dataReveal}">`+math(bounds[0][0]+(compact?162:wide?340:280)*(1-s.reframe),compact?16:wide?30:24,String.raw`\hat f(x)`,colors.teal,compact?11:wide?24:20)+'</g>';
 }
 if(s.symbol>0)html+=`<g opacity="${s.symbol*(1-s.dataReveal)}">`+math(w-12,compact?16:wide?30:24,`k=${number(s.w1)}`,colors.teal,compact?11:wide?24:20,'end')+'</g>';
 if(s.dataReveal>0)html+=`<g opacity="${s.dataReveal}">`+math(bounds[0][0],compact?16:wide?30:24,'f(x)','#606060',compact?11:wide?24:20)+'</g>';
 if(showOutput)html+=`<path data-target="step" d="M${X(xmin,0)},${Y(0)}H${X(0,0)}M${X(0,0)},${Y(1)}H${X(xmax,0)}" fill="none" stroke="#606060" stroke-width="2" stroke-dasharray="6 5"/><circle cx="${X(0,0)}" cy="${Y(0)}" r="3" fill="#606060"/><circle cx="${X(0,0)}" cy="${Y(1)}" r="3" fill="white" stroke="#606060" stroke-width="1.5"/>`;
 const displayedObservations=kind==='sgd-opening'?shownIndices.map(index=>({index,...data[index]})):[];
 if(s.dataReveal>0)html+=`<g data-observations="finite-data" opacity="${s.dataReveal}">`+displayedObservations.map(d=>`<circle data-observation="${d.index}" data-x="${d.x}" data-y="${d.y}" cx="${X(d.x,0)}" cy="${Y(d.y)}" r="${compact?3:5}" fill="${colors.blue}" stroke="white" stroke-width="1"/>`).join('')+'</g>';
 plot.innerHTML=html;plot.setAttribute('aria-label',`${weighted?'Weighted contributions':'Hidden activations'} of two ReLU neurons. ${kind==='flip'?'Dashed rust shows the unchanged hidden activation h₂; solid rust shows its weighted contribution φ₂h₂. ':''}${showOutput?'Their sum is shown in teal against the dashed gray target step: zero for x ≤ 0 and one for x > 0.':''}`);
 network(detail,s);
 if(kind==='sgd-opening'){
  learningContext(detail,s,p);
  plot.setAttribute('aria-label','The same step-function DGP and 32 representative observations from its finite training set of 2,048 noiseless data points. The hand-chosen network fades out; fitting has not begun.');
 }
 return {...s,completedUpdates:0,displayedObservations,trainingDataCount:kind==='sgd-opening'?data.length:0,detailTitle:kind==='sgd-opening'?'Finite training data':'Network',plotTitle:s.dataReveal>.5?'Same step-function DGP':weighted?'Two-neuron approximation':'Build the two hidden neurons'};
}
