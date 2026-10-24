import {relu} from './model.mjs';
const colors={blue:'#234e70',rust:'#a54f32',teal:'#007f73',gray:'#767676'};
const clamp=x=>Math.max(0,Math.min(1,x));
const mix=(a,b,t)=>a+(b-a)*t;
const number=x=>Math.abs(x)<.005?'0':Number(x.toFixed(2)).toString();
const text=(x,y,value,color='#242424',size=12,anchor='start')=>`<text x="${x}" y="${y}" text-anchor="${anchor}" style="fill:${color};font-size:${size}px">${value}</text>`;
const line=(x1,y1,x2,y2,color,width=1,dash='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}" stroke-dasharray="${dash}"/>`;
export const constructionKinds=['pair','input-first','input-second','flip','add','output-bias','target'];
export function constructionState(kind,p){
 const u=clamp((p-.1)/.72);
 const s={kind,u,w1:1,b1:0,w2:1,b2:0,a1:1,a2:1,c:0,merge:0,reveal:0};
 if(kind==='input-first')s.w1=1+u;
 if(kind==='input-second'){s.w1=2;s.w2=1+clamp(u*2);s.b2=-clamp(u*2-1);}
 if(['flip','add','output-bias','target'].includes(kind)){s.w1=s.w2=2;s.b2=-1;s.a2=kind==='flip'?1-2*u:-1;}
 if(kind==='add'){s.merge=clamp(u/.38);s.reveal=clamp((u-.38)/.62);}
 if(['output-bias','target'].includes(kind)){s.merge=1;s.reveal=1;s.c=kind==='output-bias'?.3*u:.3*(1-u);}
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
  html+=`<circle cx="${x1}" cy="${y}" r="${r}" fill="white" stroke="${color}" stroke-width="1.5"/>`+text(x1,y+(wide&&!compact?8:6),`h${j===1?'₁':'₂'}`,color,compact?11:wide?26:20,'middle');
  const sub=j===1?'₁':'₂',wy=compact?(j===1?12:h-4):(y+y0)/2+(j===1?-(wide?21:16):(wide?33:27));
  html+=text((x0+x1)/2,wy,`θ${sub}₁ = ${number(s['w'+j])}`,color,parameterSize,'middle');
  html+=text(x1,y+(j===1?-r-(wide&&!compact?16:12):r+(compact?14:wide?30:24)),`θ${sub}₀ = ${number(s['b'+j])}`,color,parameterSize,'middle');
  html+=text((x1+x2)/2,wy,`φ${sub} = ${number(s['a'+j])}`,color,parameterSize,'middle');
 }
 html+=`<circle cx="${x0}" cy="${y0}" r="10" fill="white" stroke="#767676"/><circle cx="${x2}" cy="${y0}" r="${r}" fill="white" stroke="${colors.teal}" stroke-width="1.5"/>`;
 html+=text(x0,y0+(compact?4:wide?8:6),'x','#242424',compact?13:wide?24:18,'middle')+text(x2,y0+(compact?4:wide?9:7),'Σ',colors.teal,compact?13:wide?32:26,'middle')+text(x2,y0-r-(compact?9:wide?18:14),`φ₀ = ${number(s.c)}`,colors.teal,parameterSize,'end');
 if(!compact)html+=text(x1,h-8,'Hidden layer',colors.gray,wide?20:16,'middle')+text(w-3,h-8,'Output',colors.gray,wide?20:16,'end');
 svg.innerHTML=html;svg.setAttribute('aria-label',`Two hidden neurons: input weight θ₁₁ = ${s.w1}, hidden bias θ₁₀ = ${s.b1}; input weight θ₂₁ = ${s.w2}, hidden bias θ₂₀ = ${s.b2}. Output weights φ₁ = ${s.a1}, φ₂ = ${s.a2}; output bias φ₀ = ${s.c}. Hidden-layer parameters form θ; output-layer parameters form φ.`);
}
export function renderConstruction(plot,detail,kind,p){
 const s=constructionState(kind,p),rect=plot.getBoundingClientRect(),w=Math.max(260,rect.width),h=Math.max(170,rect.height),compact=w<500,wide=globalThis.innerWidth>=1600;
 plot.setAttribute('viewBox',`0 0 ${w} ${h}`);
 const top=compact?30:wide?50:42,bottom=h-(compact?30:wide?44:36),gap=compact?22:wide?64:54,margin=compact?24:wide?44:36,panel=(w-gap)/2;
 const united=[margin,w-12];
 const bounds=[[mix(margin,united[0],s.merge),mix(panel-10,united[1],s.merge)],[mix(panel+gap+margin,united[0],s.merge),mix(w-12,united[1],s.merge)]];
 const X=(x,j)=>bounds[j][0]+(x+.3)/1.5*(bounds[j][1]-bounds[j][0]);
 const Y=y=>bottom-(y+1.6)/4.2*(bottom-top);
 const h1=x=>relu(s.w1*x+s.b1),h2=x=>relu(s.w2*x+s.b2),f=x=>s.c+h1(x)+s.a2*h2(x);
 let html='';
 function curve(fn,j,color,opacity=1,dash='',end=1.2,width=3){let d='';const xs=[-.3,0,...Array.from({length:241},(_,i)=>-.3+1.5*i/240),-s.b2/s.w2,end].filter(x=>x>=-.3&&x<=end).sort((a,b)=>a-b);for(const x of xs)d+=(d?'L':'M')+X(x,j)+','+Y(fn(x));return `<path d="${d}" fill="none" stroke="${color}" stroke-width="${width}" stroke-opacity="${opacity}" stroke-dasharray="${dash}"/>`;}
 const weighted=['flip','add','output-bias','target'].includes(kind);
 for(let j=0;j<2;j++){
  const axisOpacity=j===1?1-s.merge:1;
  html+=`<g opacity="${axisOpacity}">`;
  for(const y of [-1,0,1,2])html+=line(bounds[j][0],Y(y),bounds[j][1],Y(y),y===0?'#767676':'#e8e8e8')+text(bounds[j][0]-6,Y(y)+(compact?4:wide?7:6),y,colors.gray,compact?9:wide?20:16,'end');
  for(const x of [0,1])html+=line(X(x,j),bottom,X(x,j),bottom+4,colors.gray)+text(X(x,j),bottom+(compact?17:wide?30:24),x,colors.gray,compact?9:wide?20:16,'middle');
  html+=text(bounds[j][1],bottom+(compact?17:wide?30:24),'x',colors.gray,compact?10:wide?20:16,'end')+'</g>';
 }
 const showOutput=s.reveal>0,componentOpacity=showOutput?.32:1;
 if(kind==='input-first')html+=curve(x=>relu(x),0,colors.gray,.3,'4 4');
 if(kind==='input-second')html+=curve(x=>relu(x),1,colors.gray,.3,'4 4');
 if(kind==='flip')html+=curve(h2,1,colors.rust,.25,'4 4');
 html+=curve(h1,0,colors.blue,componentOpacity,showOutput?'5 4':'')+curve(x=>(weighted?s.a2:1)*h2(x),1,colors.rust,componentOpacity,showOutput?'5 4':'');
 const titles=weighted?['φ₁h₁(x)','φ₂h₂(x)']:['h₁(x)','h₂(x)'];
 html+=text(bounds[0][0],compact?16:wide?30:24,titles[0],colors.blue,compact?11:wide?24:20)+text(s.merge?bounds[0][0]+(compact?80:wide?170:140):bounds[1][0],compact?16:wide?30:24,titles[1],colors.rust,compact?11:wide?24:20);
 if(showOutput){
  const cut=s.reveal<.86?mix(-.3,1.2,s.reveal/.86):1.2;
  html+=curve(f,0,colors.teal,1,'',cut,3.5)+text(bounds[0][0]+(compact?162:wide?340:280),compact?16:wide?30:24,'sum + φ₀',colors.teal,compact?11:wide?24:20);
  if(kind==='add'){
   const x=s.reveal<.86?cut:.75,v1=h1(x),v2=s.a2*h2(x),out=f(x);
   html+=line(X(x,0),Y(0),X(x,0),Y(v1),colors.blue,3)+line(X(x,0)+5,Y(v1),X(x,0)+5,Y(out),colors.rust,3);
   html+=`<circle cx="${X(x,0)}" cy="${Y(out)}" r="4" fill="${colors.teal}"/>`;
   html+=text(bounds[0][0]+10,bottom-8,`${number(v1)} ${v2<0?'−':'+'} ${number(Math.abs(v2))} = ${number(out)}`,colors.teal,compact?12:wide?30:24);
  }
 }
 if(showOutput)html+=`<path data-target="step" d="M${X(-.3,0)},${Y(0)}H${X(0,0)}M${X(0,0)},${Y(1)}H${X(1.2,0)}" fill="none" stroke="#606060" stroke-width="2" stroke-dasharray="6 5"/><circle cx="${X(0,0)}" cy="${Y(0)}" r="3" fill="#606060"/><circle cx="${X(0,0)}" cy="${Y(1)}" r="3" fill="white" stroke="#606060" stroke-width="1.5"/>`;
 plot.innerHTML=html;plot.setAttribute('aria-label',`${weighted?'Weighted contributions':'Hidden activations'} of two ReLU neurons. ${showOutput?'Their sum is shown in teal against the dashed gray target step: zero for x ≤ 0 and one for x > 0.':''}`);
 network(detail,s);
 return {...s,detailTitle:'Hidden parameters θ → output parameters φ',plotTitle:weighted?'Apply the output weights, then add':'Build the two hidden neurons'};
}
