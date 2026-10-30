import { socksAttention, SOCKS_WO } from './socks-model.mjs?v=attention-sequence-2';

// Continue the same construction: wider heads, output updates, and a new
// prediction setting. Every mathematical label uses the shared renderer.
export function createExtensionScenes({m,text,line,rect,circle,C,ease,lerp,clamp,num}) {
  const L=String.raw;
  const forward=socksAttention();
  const reverse=socksAttention({reverse:true});
  const causal=socksAttention({reverse:true,causal:true});
  const color=token=>token==='socks'?C.blue:C.rust;
  const group=(body,opacity=1)=>`<g opacity="${clamp(opacity)}">${body}</g>`;
  const near=(a,b)=>Math.abs(a-b)<1e-10;
  const value=v=>typeof v==='string'?v:v===-Infinity?L`-\infty`:near(v,Math.log(3))?L`\log3`:near(v,.5)?L`\tfrac12`:near(v,-.5)?L`-\tfrac12`:near(v,.25)?L`\tfrac14`:near(v,.75)?L`\tfrac34`:num(v);
  const arrow=(x1,y1,x2,y2,stroke=C.muted,opacity=1)=>line(x1,y1,x2,y2,stroke,1.5,`opacity="${opacity}" marker-end="url(#arrow)"`);
  const curve=(x1,y1,x2,y2)=>`M${x1} ${y1}C${x1} ${lerp(y1,y2,.55)} ${x2} ${lerp(y1,y2,.55)} ${x2} ${y2}`;
  const cubic=(a,b,c,d,t)=>(1-t)**3*a+3*(1-t)**2*t*b+3*(1-t)*t*t*c+t**3*d;
  function matrix(data,{x,y,cw=82,ch=82,heat=false,selected=null,size=29,reveal=1}={}) {
    let out='';
    data.forEach((row,i)=>row.forEach((v,j)=>{
      const blocked=v===-Infinity;
      const fill=blocked?'url(#blocked)':heat?`rgba(35,78,112,${.04+.7*Number(v)})`:'#f5f5f2';
      out+=group(rect(x+j*cw+3,y+i*ch+3,cw-6,ch-6,fill,'none',4)+m(value(v),x+(j+.5)*cw,y+(i+.5)*ch,cw-6,ch-6,size,heat&&v>.7?'#fff':C.ink),reveal);
    }));
    out+=`<path d="M${x-4} ${y}h-7v${data.length*ch}h7 M${x+data[0].length*cw+4} ${y}h7v${data.length*ch}h-7" fill="none" stroke="${C.ink}" stroke-width="1.4"/>`;
    if(selected)out+=rect(x+selected[1]*cw+1,y+selected[0]*ch+1,cw-2,ch-2,'none',C.rust,4,'stroke-width="2.5"');
    return out;
  }
  function icon(token,x,y,size=100,opacity=1) {
    const stroke=color(token);
    const sock=`<path d="M-22-45H15V-5L40 13Q50 22 39 35Q30 46 17 38L-17 16Q-26 10-26-2Z" fill="#fff" stroke="${stroke}" stroke-width="3" stroke-linejoin="round"/><path d="M-23-31H15 M-23-25H15 M-24 2Q-10 1-8 20 M25 7Q15 18 14 35" fill="none" stroke="${stroke}" stroke-width="2"/>`;
    const shoe=`<path d="M-46 8L-35-22Q-23-15-12-18L13 4L36 11Q48 15 48 28H-49Z" fill="#fff" stroke="${stroke}" stroke-width="3" stroke-linejoin="round"/><path d="M-48 21H47 M-20-8L-8-16 M-10 0L2-9 M1 7L12-1 M-37-19L-40 10" fill="none" stroke="${stroke}" stroke-width="2" stroke-linecap="round"/>`;
    return `<g transform="translate(${x} ${y}) scale(${size/100})" opacity="${opacity}">${token==='socks'?sock:shoe}</g>`;
  }
  function vectorHead(t) {
    const p=ease(t);
    let out=m(L`X\in\mathbb R^{T\times d}`,320,35,460,50,30,C.ink);
    out+=m(L`Q=XW_Q`,165,100,300,52,32,C.blue)+m(L`K=XW_K`,475,100,300,52,32,C.rust);
    out+=m(L`Q\in\mathbb R^{T\times d_k}`,165,151,300,45,26,C.blue);
    out+=m(L`K\in\mathbb R^{T\times d_k}`,475,151,300,45,26,C.rust);
    out+=line(165,179,270,208,C.blue,1.5)+line(475,179,370,208,C.rust,1.5);
    out+=circle(lerp(165,270,p),lerp(179,208,p),5,C.blue)+circle(lerp(475,370,p),lerp(179,208,p),5,C.rust);
    out+=m(L`S=\frac{QK^{\mathsf T}}{\sqrt{d_k}}\in\mathbb R^{T\times T}`,320,245,600,85,33,C.ink);
    out+=m(L`A=\operatorname{softmax}_{\rm row}(S)`,320,326,590,60,32,C.blue);
    out+=m(L`A\in\mathbb R^{T\times T}`,320,375,450,45,26,C.muted);
    out+=line(48,411,592,411,C.rule,1.5);
    out+=m(L`V=XW_V`,160,452,300,50,31,C.teal)+m(L`Y=AV`,465,452,300,50,31,C.blue);
    out+=m(L`V\in\mathbb R^{T\times d_v}`,160,502,310,46,26,C.teal);
    out+=m(L`Y\in\mathbb R^{T\times d_v}`,465,502,310,46,26,C.blue);
    return out;
  }
  function projectOutput(t) {
    const p=ease(t),arrival=clamp((t-.45)/.5);
    let out=text('Return the mixture to the input width.',320,45,25,C.muted);
    out+=m('Y',112,96,130,50,34,C.blue)+matrix(forward.Y,{x:74,y:130,cw:76,ch:77});
    out+=m(L`\times`,245,208,70,60,34,C.muted);
    out+=m('W_O',445,96,150,50,34,C.teal)+matrix(SOCKS_WO,{x:325,y:169,cw:80,ch:77});
    out+=m(L`2\times1`,112,342,160,50,26,C.muted)+m(L`1\times3`,445,284,190,50,26,C.muted);
    out+=`<path d="${curve(112,287,440,484.5)}" fill="none" stroke="${C.blue}" stroke-width="1.6" opacity=".28"/>`;
    const cy=lerp(287,484.5,.55);
    out+=circle(cubic(112,112,440,440,p),cubic(287,cy,cy,484.5,p),6,C.blue);
    out+=m(L`\Delta X`,127,449,180,60,35,C.blue);
    out+=matrix(forward.deltaX,{x:240,y:375,cw:80,ch:73,reveal:.28+.72*arrival,selected:t>.55?[1,2]:null});
    out+=m(L`=`,208,449,40,60,32,C.muted);
    return out;
  }
  function residual(t) {
    const p=ease(t),active=p>.35;
    const starts=[20,248,476],cy=174,cw=48,ch=82;
    let out=text('Add the attention update to the original rows.',320,48,25,C.muted);
    out+=m('X',92,118,145,55,33,C.blue)+m(L`\Delta X`,320,118,170,55,33,C.teal)+m(L`X'`,548,118,145,55,33,C.rust);
    out+=matrix(forward.X,{x:starts[0],y:cy,cw,ch,size:26,selected:[1,2]});
    out+=matrix(forward.deltaX,{x:starts[1],y:cy,cw,ch,size:26,selected:[1,2]});
    out+=matrix(forward.updatedX,{x:starts[2],y:cy,cw,ch,size:26,selected:[1,2],reveal:.2+.8*p});
    out+=m('+',206,256,60,60,35,C.muted)+m('=',434,256,60,60,35,C.muted);
    // Trace the two selected coordinates into the updated coordinate. The
    // displayed numbers stay exact while scroll reverses the transport.
    [[140,C.blue],[368,C.teal]].forEach(([sx,stroke])=>{
      out+=`<path d="M${sx} 347C${sx} 372 596 372 596 347" fill="none" stroke="${stroke}" stroke-width="1.5" opacity=".25"/>`;
      out+=circle(cubic(sx,sx,596,596,p),cubic(347,372,372,347,p),4.5,stroke,`opacity="${.8*Math.sin(Math.PI*p)}"`);
    });
    out+=line(48,379,592,379,C.rule,1.5);
    out+=text('The shoes row, third coordinate',320,420,25,C.muted);
    out+=m(L`-1+\tfrac12=-\tfrac12`,320,483,600,80,39,active?C.rust:C.muted);
    return out;
  }
  function newSetting(t) {
    const p=ease(t);
    let out=text('Predict the next item after “shoes”.',320,52,26,C.muted);
    out+=rect(365,105,235,298,'url(#blocked)','none',8);
    out+=icon('shoes',160,194,125)+icon('socks',482,194,125,.72-.47*p);
    out+=text('shoes',160,293,30,C.rust)+text('socks',482,293,30,C.blue);
    out+=m('x_1',160,350,140,55,32,C.rust)+m('x_2',482,350,140,55,32,C.muted);
    out+=line(321,111,321,406,C.rust,2,'stroke-dasharray="6 8"');
    out+=arrow(160,389,160,438,C.rust);
    out+=rect(55,446,210,58,'#f6ebe6','none',7)+text('predict next',160,483,26,C.rust);
    out+=text('still in the future',482,457,25,C.muted);
    out+=text('unavailable here',482,493,25,C.muted);
    return out;
  }
  function maskScores(t) {
    const masked=t>=.5;
    const scores=masked?causal.maskedScores:reverse.S;
    let out=m(masked?L`\widetilde S`:L`S`,353,49,160,58,36,C.blue);
    out+=text('shoes',287,112,26,C.rust)+text('socks',417,112,26,C.blue);
    out+=text('shoes',190,199,26,C.rust,'end')+text('socks',190,303,26,C.blue,'end');
    out+=matrix(scores,{x:222,y:147,cw:130,ch:104,selected:masked?[0,1]:null,size:34});
    out+=m(L`\widetilde S_{ij}=\begin{cases}s_{ij},&j\leq i\\-\infty,&j>i\end{cases}`,320,443,600,135,32,C.blue);
    return out;
  }
  function renormalize(t) {
    const future=t<.5,selected=future?[0,1]:[0,0];
    let out=text('Same sequence: shoes, socks.',320,43,25,C.muted);
    out+=m('A',160,101,180,54,33,C.blue)+m(L`A_{\rm causal}`,480,101,240,54,33,C.blue);
    [70,390].forEach(x=>{
      out+=text('shoes',x+45,151,24,C.rust)+text('socks',x+135,151,24,C.blue);
      out+=m('i=1',x-37,224,65,50,22,C.muted)+m('i=2',x-37,314,65,50,22,C.muted);
    });
    out+=matrix(reverse.A,{x:70,y:179,cw:90,ch:90,heat:true,selected,size:32});
    out+=matrix(causal.A,{x:390,y:179,cw:90,ch:90,heat:true,selected,size:32});
    out+=arrow(286,269,354,269,C.muted);
    out+=m(future?L`a_{12}:\quad\tfrac34\longrightarrow0`:L`a_{11}:\quad\tfrac14\longrightarrow1`,320,421,560,65,36,C.rust);
    out+=m(future?L`e^{-\infty}=0`:L`\operatorname{softmax}([0\;-\infty])=[1\;0]`,320,494,610,64,future?32:29,C.blue);
    return out;
  }
  function conclusion(t) {
    const p=clamp(t)*3;
    const rows=[
      {top:29,label:'Queries and keys choose where.',formula:L`Q,\ K\ \longrightarrow\ A`,color:C.blue},
      {top:182,label:'Values supply what.',formula:L`Y=AV`,color:C.teal},
      {top:335,label:'The output map returns an update.',formula:L`\Delta X=YW_O`,color:C.rust}
    ];
    let out='';
    rows.forEach((row,i)=>{
      const emphasis=1-Math.min(1,Math.abs(p-(i+.5)));
      out+=rect(35,row.top,570,133,`rgba(35,78,112,${.015+.04*emphasis})`,'none',7);
      out+=text(row.label,320,row.top+35,26,C.muted);
      out+=m(row.formula,320,row.top+88,580,68,36,row.color);
    });
    out+=m(L`X'=X+\Delta X`,320,511,570,58,32,C.ink);
    return out;
  }
  return [
    {part:'04 · From mixtures to updates',label:'Let the matching features be vectors',title:'Queries and keys can be vectors.',caption:'A query can compare several features with each key. A scaled dot product supplies the score; the resulting attention matrix still mixes source rows from the left.',equation:L`s_{ij}=\frac{q_i k_j^{\mathsf T}}{\sqrt{d_k}}`,detail:L`The dot product sums matching-coordinate products; division by <span data-math="\sqrt{d_k}"></span> controls score scale. Our scalar example had <span data-math="d_k=1"></span>.`,visual:'The same matching and mixing, with wider feature vectors.',draw:vectorHead},
    {part:'04 · From mixtures to updates',label:'Map the output back to the input width',title:'Turn a mixture into an update.',caption:'With socks first, the attention output has one coordinate per token. Choose an output projection that maps it back into three coordinates, matching the width of the input.',equation:L`\Delta X=YW_O=AVW_O`,detail:'Here the output projection writes the mixed value into the third coordinate and zero into the other two. The same map applies to every row.',visual:'The shoes output of one half becomes the third coordinate of its update.',draw:projectOutput},
    {part:'04 · From mixtures to updates',label:'Add the projected update',title:'Keep the input, and add what attention found.',caption:'Add the projected output to the original input, coordinate by coordinate. The shoes row receives an update of one half in its third coordinate.',equation:L`X'=X+\Delta X`,detail:'Our earlier task read out Y directly. Transformer attention can instead add this projected update. After the update, coordinates need not retain their original meanings.',visual:'The right-hand matrix contains updated features. This illustrates the attention update alone.',draw:residual},
    {part:'05 · When the future is unavailable',label:'A different prediction setting',title:'What if the next item is still unknown?',caption:'Now predict the next item after “shoes.” The first prediction may use that token, but it must not read the second token, “socks.”',equation:L`\text{At position }i,\quad j\leq i`,detail:'This is a new use for attention. We return to the original input rows; the earlier task scoring rule is not used here.',visual:'A later token may be present in a training sequence while being unavailable to this prediction.',draw:newSetting},
    {part:'05 · When the future is unavailable',label:'Exclude future sources before softmax',title:'Mask the score for the future.',caption:'In the reversed sequence, the first query strongly matches the second key. Replace that future score with negative infinity before applying softmax.',equation:L`\widetilde S=S+M`,detail:'The mask adds zero to allowed entries and negative infinity to future entries. A token may still read itself and earlier tokens.',visual:'Only the top-right entry is forbidden. Scroll to replace its score; all allowed scores stay unchanged.',draw:maskScores,discretePhases:true},
    {part:'05 · When the future is unavailable',label:'Recompute each probability row',title:'Remove the future, then renormalize.',caption:'The future source receives zero weight. The first token is now its own only allowed source, so its weight becomes one. The second row stays unchanged.',equation:L`A_{\rm causal}=\operatorname{softmax}_{\rm row}(\widetilde S)`,detail:'Masking changes which sources are available, then softmax normalizes over those sources. These are the complete before-and-after matrices.',visual:'The highlight first follows the forbidden contribution, then the weight that grows to one.',draw:renormalize,discretePhases:true},
    {part:'06 · The whole attention operation',label:'One operation, three roles',title:'Attention routes information.',caption:'Queries and keys determine the weights. Values supply what is mixed. An output projection can turn the mixture into an update to the input rows.',equation:L`X'=X+AVW_O`,detail:'Use a causal mask when the prediction must exclude future sources. The matching, mixing, and output projection otherwise keep the same form.',visual:'Where to read. What to carry. How to return the result to the input feature space.',draw:conclusion}
  ];
}
