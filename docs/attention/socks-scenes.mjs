import { socksAttention } from './socks-model.mjs?v=attention-sequence-2';

// Introductory construction: chosen weights, exact arithmetic, and no causal mask.
// The supplied drawing helpers keep math in the lesson's shared LaTeX renderer.
export function createSocksScenes({m,text,line,rect,circle,C,ease,lerp,clamp,num}) {
  const forward=socksAttention();
  const reverse=socksAttention({reverse:true});
  const noPosition=socksAttention({position:false});
  const uniform=socksAttention({uniform:true});
  const blue=C.blue, rust=C.rust;
  const identityColor=token=>token==='socks'?blue:rust;
  const svgGroup=(content,opacity=1)=>`<g opacity="${clamp(opacity)}">${content}</g>`;
  const array=(data)=>`\\begin{bmatrix}${data.map(row=>row.map(value=>typeof value==='string'?value:num(value)).join('&')).join('\\\\')}\\end{bmatrix}`;
  function icon(token,x,y,size=100,opacity=1) {
    const color=identityColor(token);
    const sock=`<path d="M-22-45H15V-5L40 13Q50 22 39 35Q30 46 17 38L-17 16Q-26 10-26-2Z" fill="#fff" stroke="${color}" stroke-width="3" stroke-linejoin="round"/><path d="M-23-31H15 M-23-25H15 M-24 2Q-10 1-8 20 M25 7Q15 18 14 35" fill="none" stroke="${color}" stroke-width="2"/>`;
    const shoe=`<path d="M-46 8L-35-22Q-23-15-12-18L13 4L36 11Q48 15 48 28H-49Z" fill="#fff" stroke="${color}" stroke-width="3" stroke-linejoin="round"/><path d="M-48 21H47 M-20-8L-8-16 M-10 0L2-9 M1 7L12-1 M-37-19L-40 10" fill="none" stroke="${color}" stroke-width="2" stroke-linecap="round"/>`;
    return `<g transform="translate(${x} ${y}) scale(${size/100})" opacity="${opacity}">${token==='socks'?sock:shoe}</g>`;
  }
  function slotLabels(y=358,withPosition=false) {
    let out=line(92,y-24,548,y-24,C.rule,1.5)+text('first',175,y+12,24,C.muted)+text('second',465,y+12,24,C.muted);
    if(withPosition)out+=m('p_1=+1',175,y+62,180,50,29,blue)+m('p_2=-1',465,y+62,180,50,29,rust);
    return out;
  }
  function sequence(t,{swap=false,position=false,y=205,size=130}={}) {
    const p=swap?ease(t):0;
    const sx=lerp(175,465,p),hx=lerp(465,175,p);
    let out=icon('socks',sx,y-70*Math.sin(Math.PI*p),size)+icon('shoes',hx,y+70*Math.sin(Math.PI*p),size);
    out+=text('socks',sx,y-70*Math.sin(Math.PI*p)+82,26,blue)+text('shoes',hx,y+70*Math.sin(Math.PI*p)+82,26,rust);
    out+=slotLabels(358,position);
    return out;
  }
  function cellMatrix(data,{x=245,y=175,cw=78,ch=88,rowTokens=null,heat=false,highlight=-1,reveal=1}={}) {
    let out='';
    for(let i=0;i<data.length;i++) {
      const alpha=clamp(reveal*data.length-i);
      let row='';
      if(rowTokens)row+=text(rowTokens[i],x-24,y+i*ch+ch/2+9,25,identityColor(rowTokens[i]),'end');
      for(let j=0;j<data[i].length;j++) {
        const value=data[i][j];
        const fill=heat?`rgba(35,78,112,${.05+.72*Number(value)})`:j===highlight?'#f6ebe6':'#f5f5f2';
        row+=rect(x+j*cw+3,y+i*ch+3,cw-6,ch-6,fill,'none',5);
        row+=m(typeof value==='string'?value:num(value),x+j*cw+cw/2,y+i*ch+ch/2,cw-4,ch-4,29,heat&&Number(value)>.7?'#fff':C.ink);
      }
      out+=svgGroup(row,alpha);
    }
    const w=data[0].length*cw,h=data.length*ch;
    return out+`<path d="M${x-4} ${y}h-8v${h}h8 M${x+w+4} ${y}h8v${h}h-8" fill="none" stroke="${C.ink}" stroke-width="1.5"/>`;
  }
  function task(t) {
    const p=ease(t);
    let out=text('The same two items. A different order.',320,53,25,C.muted)+sequence(t,{swap:true});
    if(p<.12)out+=m('y_{\\mathrm{target}}=0',320,475,430,60,34,blue);
    else if(p>.88)out+=m('y_{\\mathrm{target}}=-1',320,475,430,60,34,rust);
    else out+=text('Order is the information we need.',320,475,25,C.muted);
    return out;
  }
  function identities(t) {
    const p=ease(t),sy=lerp(150,300,p),hy=lerp(300,150,p);
    let out=text('Represent identity alone',320,52,25,C.muted);
    out+=icon('socks',98,sy,80)+m('[1\\;0]',220,sy,190,55,32,blue);
    out+=icon('shoes',98,hy,80)+m('[0\\;1]',220,hy,190,55,32,rust);
    out+=`<path d="M310 150Q375 150 412 225 M310 300Q375 300 412 225" fill="none" stroke="${C.rule}" stroke-width="2"/>`;
    out+=m('[\\tfrac12\\;\\tfrac12]',495,225,230,90,32,C.ink)+text('same mean',495,300,23,C.muted);
    out+=m('\\tfrac12\\bigl([1\\;0]+[0\\;1]\\bigr)',320,430,590,70,32,C.blue);
    out+=text('Reordering the rows cannot change their average.',320,506,23,C.muted);
    return out;
  }
  function positionalRows(t) {
    // Show actual same-shape addition, including every zero. The highlighted
    // entries are computations, not separate column blocks being concatenated.
    const E=forward.X.map(row=>[row[0],row[1],0]);
    const P=forward.X.map(row=>[0,0,row[2]]);
    const phase=t<.32?0:t<.65?1:2;
    const examples=[{i:0,j:0,label:'socks · first coordinate'},
      {i:0,j:2,label:'socks · third coordinate'},
      {i:1,j:2,label:'shoes · third coordinate'}];
    const {i,j,label}=examples[phase];
    const starts=[34,244,454],cw=50,ch=67,top=170;
    let out=text('Rows: socks, then shoes',320,36,23,C.muted);
    const matrices=[{name:'E',label:'token embeddings',data:E,color:blue},
      {name:'P',label:'position vectors',data:P,color:rust},
      {name:'X',label:'their sum',data:forward.X,color:C.teal}];
    matrices.forEach(({name,label,data,color},n)=>{
      const x=starts[n];
      out+=m(name,x+75,84,150,58,35,color);
      out+=text(label,x+75,125,22,C.muted);
      out+=cellMatrix(data,{x,y:top,cw,ch});
      // A matching cell in every matrix makes entrywise addition visible.
      out+=rect(x+j*cw+1,top+i*ch+1,cw-2,ch-2,'none',color,4,'stroke-width="3"');
    });
    out+=m('+',214,237,42,70,34,C.ink)+m('=',424,237,42,70,34,C.ink);
    out+=text('Two rows and three columns in every matrix.',320,356,23,C.muted);
    out+=text(label,320,413,24,C.muted);
    const ev=E[i][j],pv=P[i][j],xv=forward.X[i][j];
    out+=m(`${ev}+${pv<0?'('+pv+')':pv}=${xv}`,320,473,590,76,42,C.ink);
    return out;
  }
  function retrievalGoal(t) {
    const match=ease(clamp(t*2)),carry=ease(clamp(t*2-1));
    let out=text('Choose the source and the information.',320,43,24,C.muted);
    out+=icon('socks',155,201,110)+text('socks',155,292,27,blue);
    out+=icon('shoes',485,201,110)+text('shoes',485,292,27,rust);
    out+=text('Look for socks',485,109,24,rust);
    out+=`<path d="M415 199H225" fill="none" stroke="${C.muted}" stroke-width="2" stroke-dasharray="7 5" marker-end="url(#arrow)"/>`;
    out+=circle(lerp(415,225,match),199,6,rust);
    out+=text('match identity',320,163,22,C.muted);
    out+=m('+1',155,377,160,70,40,blue);
    out+=text('position it can supply',155,440,22,C.muted);
    out+=m('?',485,377,160,70,40,rust);
    out+=text('weighted result',485,440,22,C.muted);
    out+=`<path d="M225 377H415" fill="none" stroke="${blue}" stroke-width="3" opacity="${.15+.6*carry}" marker-end="url(#arrow)"/>`;
    out+=circle(lerp(225,415,carry),377,6,blue,`opacity="${carry}"`);
    out+=text('Next, set how much each source contributes.',320,510,23,C.muted);
    return out;
  }
  function queriesKeys(t) {
    let out=text('A deliberate, simple choice of maps',320,45,24,C.muted);
    out+=m('W_Q='+array([[0],[1],[0]]),172,159,280,174,29,blue);
    out+=m('W_K='+array([['\\log 3'],[0],[0]]),478,159,290,174,29,rust);
    out+=text('query reads “is shoes”',172,278,23,blue)+text('key reads “is socks”',478,278,23,rust);
    out+=svgGroup(m('Q='+array(forward.Q),172,382,270,130,32,blue)+m('K='+array([['\\log 3'],[0]]),478,382,290,130,32,rust),ease(t));
    out+=text('The same maps apply to both token positions.',320,510,23,C.muted);
    return out;
  }
  function matching(t) {
    const p=ease(t);
    let out=icon('shoes',104,248,115)+text('shoes',104,347,27,rust)+m('q_2=1',104,402,180,60,30,rust);
    out+=icon('socks',510,125,100)+text('socks',510,207,25,blue)+m('k_1=\\log 3',510,259,230,55,28,blue);
    out+=icon('shoes',510,344,100)+text('shoes',510,423,25,rust)+m('k_2=0',510,470,210,55,28,rust);
    const paths=[{y:146,score:'1\\cdot\\log 3=\\log 3',color:blue},{y:345,score:'1\\cdot0=0',color:rust}];
    paths.forEach(({y,score,color},i)=>{
      out+=`<path d="M175 245C290 245 300 ${y} 429 ${y}" fill="none" stroke="${color}" stroke-width="${i===0?4:2}" opacity=".55"/>`;
      const dotx=lerp(175,429,p),doty=lerp(245,y,p);
      out+=circle(dotx,doty,6,color);
      out+=rect(213,y-45,200,52,'#fff')+m(score,313,y-18,220,62,26,color);
    });
    out+=text('destination query',112,63,22,C.muted)+text('source keys',509,63,22,C.muted);
    return out;
  }
  function softmax(t) {
    const phase=t<.32?0:t<.65?1:2;
    const values=phase===0?[Math.log(3),0]:phase===1?[3,1]:[.75,.25];
    const labels=phase===0?['\\log 3','0']:phase===1?['3','1']:['\\tfrac34','\\tfrac14'];
    const max=phase===0?1.2:phase===1?4:1;
    const phaseTitles=['1. Compare scores.','2. Exponentiate.','3. Divide by the total.'];
    let out=text(phaseTitles[phase],320,60,27,C.muted);
    [175,465].forEach((x,i)=>{
      const h=values[i]/max*210;
      out+=rect(x-57,350-h,114,h,i===0?blue:rust,'none',5);
      out+=m(labels[i],x,350-h-34,160,55,35,i===0?blue:rust);
      out+=text(i===0?'socks':'shoes',x,395,27,i===0?blue:rust);
    });
    out+=line(76,350,564,350,C.rule,1.5);
    out+=m(phase===0?'s_2=[\\log 3\\;0]':phase===1?'[e^{\\log3}\\;e^0]=[3\\;1]':'a_{21}=\\frac3{3+1},\\quad a_{22}=\\frac1{3+1}',320,473,615,76,phase===2?29:31,C.blue);
    return out;
  }
  function allAttention(t) {
    let out=text('read from',386,42,24,C.muted);
    out+=text('socks',302,103,27,blue)+text('shoes',436,103,27,rust);
    out+=cellMatrix(forward.A,{x:236,y:141,cw:132,ch:105,rowTokens:forward.tokens,heat:true,reveal:clamp(t*1.6)});
    out+=m('A',367,405,180,60,36,C.blue);
    out+=m('AX='+array(forward.AX),320,480,600,90,29,C.ink);
    return out;
  }
  function carryPosition(t) {
    let out=text('Input rows',167,71,24,C.muted)+text('Values to carry',505,71,24,C.muted);
    out+=m('X',166,123,110,55,32,C.blue)+cellMatrix(forward.X,{x:47,y:171,cw:76,ch:92,highlight:2});
    out+=m('W_V='+array([[0],[0],[1]]),365,249,190,180,26,C.teal);
    out+=m('V',520,123,110,55,32,C.teal)+cellMatrix(forward.V,{x:473,y:171,cw:92,ch:92,reveal:clamp(t*1.6)});
    out+=m('V=XW_V='+array(forward.V),320,451,590,110,34,C.teal);
    return out;
  }
  function weightedValues(t) {
    const p=ease(t);
    let out=text('source values',113,50,23,C.muted)+text('destination outputs',491,50,23,C.muted);
    const ys=[171,343],xs=128,xt=492;
    forward.tokens.forEach((token,j)=>{
      out+=icon(token,xs,ys[j]-30,70)+text(token,xs,ys[j]+30,25,identityColor(token));
      out+=m(`v_${j+1}=${forward.V[j][0]>0?'+1':'-1'}`,xs,ys[j]+79,190,55,29,identityColor(token));
    });
    for(let i=0;i<2;i++)for(let j=0;j<2;j++){
      const w=forward.A[i][j],sy=ys[j],ty=ys[i];
      out+=`<path d="M200 ${sy}C305 ${sy} 335 ${ty} 414 ${ty}" fill="none" stroke="${identityColor(forward.tokens[j])}" stroke-width="${2+8*w}" opacity="${i===1?.65:.23}"/>`;
      const x=lerp(200,414,p),y=lerp(sy,ty,p);
      out+=circle(x,y,3+5*w,identityColor(forward.tokens[j]),`opacity="${i===1?1:.5}"`);
    }
    out+=rect(253,274,98,44,'#fff')+m('\\tfrac34',302,297,90,60,29,blue);
    out+=rect(253,354,98,44,'#fff')+m('\\tfrac14',302,377,90,60,29,rust);
    out+=text('socks',xt,ys[0]-39,26,blue)+text('shoes',xt,ys[1]-39,26,rust);
    out+=m('y_1=0',xt,ys[0]+7,230,65,34,blue)+m('y_2=\\tfrac12',xt,ys[1]+7,230,65,34,rust);
    out+=m('\\tfrac34(+1)+\\tfrac14(-1)=\\tfrac12',320,503,620,65,30,rust);
    return out;
  }
  function prediction(t) {
    let out=text('Both destination rows contribute.',320,54,25,C.muted);
    out+=cellMatrix(forward.Y,{x:92,y:139,cw:100,ch:98})+m('Y',142,102,100,50,35,C.blue);
    out+=line(223,236,303,236,C.muted,1.5,'marker-end="url(#arrow)"');
    out+=m('\\bar y=\\tfrac14',393,235,240,90,35,C.teal)+text('average the two rows',393,307,23,C.muted);
    out+=svgGroup(m('\\hat y=2\\bar y-\\tfrac12=0',320,416,600,92,38,C.blue),ease(t));
    out+=text('The task score matches the socks-first target.',320,501,23,C.muted);
    return out;
  }
  function swapped(t) {
    const p=ease(t),done=p>.9;
    let out=text('Change only the order.',320,47,25,C.muted);
    const sx=lerp(175,465,p),hx=lerp(465,175,p),y=147;
    out+=icon('socks',sx,y-43*Math.sin(Math.PI*p),86)+icon('shoes',hx,y+43*Math.sin(Math.PI*p),86);
    out+=text('first',175,240,24,C.muted)+text('second',465,240,24,C.muted);
    out+=m('p_1=+1',175,285,190,50,29,blue)+m('p_2=-1',465,285,190,50,29,rust);
    if(done){
      out+=m('Y='+array(reverse.Y),170,384,290,128,33,C.rust);
      out+=m('\\hat y=2(-\\tfrac14)-\\tfrac12=-1',432,402,390,110,28,C.rust);
    } else {
      out+=text('Queries and keys stay tied to identity.',320,380,24,C.muted);
      out+=text('Values stay tied to position.',320,428,24,C.muted);
    }
    out+=text('The maps and the final scoring rule are unchanged.',320,510,23,C.muted);
    return out;
  }
  function ablations(t) {
    const uniformStage=t>.5;
    const model=uniformStage?uniform:noPosition;
    let out=text(uniformStage?'Keep positions, but average uniformly.':'Remove the position feature.',320,57,25,C.muted);
    out+=icon('socks',175,155,90)+icon('shoes',465,155,90);
    out+=m(uniformStage?'A='+array(uniform.A):'V='+array(noPosition.V),320,304,590,145,33,C.muted);
    out+=m(`\\hat y=${num(model.prediction)}`,320,441,420,70,38,C.rust);
    out+=text('Both orders now get the same score.',320,509,25,C.muted);
    return out;
  }
  return [
    {part:'01 · Why order matters',label:'Begin with a small task',title:'Socks before shoes.',caption:'Give “socks, shoes” a score of zero and the reversed order a score of minus one. The items are identical; their order changes the target.',equation:'[\\text{socks},\\text{shoes}]\\;\\longmapsto\\;0',detail:'This is a deliberately small construction. We will choose every weight and compute every output; nothing here is a claim about a pretrained model.',visual:'Scroll to exchange the two items while their positions stay fixed.',draw:task},
    {part:'01 · Why order matters',label:'Averaging loses the distinction',title:'Identity alone is not enough.',caption:'Represent socks and shoes with two identity coordinates. Averaging those vectors gives the same result in either order.',equation:'\\tfrac12\\bigl([1\\;0]+[0\\;1]\\bigr)=[\\tfrac12\\;\\tfrac12]',detail:'A model that sees only this average cannot give the two sequences different predictions.',visual:'The row vectors move with the tokens, but their mean stays fixed.',draw:identities},
    {part:'01 · Why order matters',label:'Choose one construction',title:'Let’s take one simple route.',caption:'Position vectors can be learned and added to token embeddings of the same width. Here, we choose fixed vectors that use the third coordinate.',equation:'X=E+P',detail:'Add matching entries. <span data-math="E"></span>, <span data-math="P"></span>, and <span data-math="X"></span> all have two rows and three columns. From here on, <span data-math="X"></span> includes position.',visual:'We’ll use this choice for the rest of the socks-and-shoes example.',draw:positionalRows,discretePhases:true},
    {part:'02 · Construct one attention head',label:'Give the head a purpose',title:'What should shoes retrieve?',caption:'For our chosen solution, let shoes look for socks and read information about their position. Matching identifies the source; the value supplies the information to combine.',equation:'\\text{source}\\;\\longrightarrow\\;\\text{position}',detail:'This is the retrieval rule we want to implement. The next steps choose matrices and calculate how much each source contributes.',visual:'Identify socks, then use the position they carry. The weighted result is still to be calculated.',draw:retrievalGoal},
    {part:'02 · Construct one attention head',label:'Choose two feature maps',title:'Let shoes look for socks.',caption:'The query marks shoes as the destination asking for information. The key marks socks as the source to favor. Separate maps let us express this directed match.',equation:'Q=XW_Q,\\qquad K=XW_K',detail:'These shared maps implement our chosen rule. The factor <span data-math="\\log 3"></span> will give socks three times the weight of shoes. Training would learn the maps instead.',visual:'The query and key dimensions are both one in this example.',draw:queriesKeys},
    {part:'02 · Construct one attention head',label:'The shoes query compares both sources',title:'Compare the query with each key.',caption:'The shoes query is one. Its score against socks is log three; its score against shoes is zero.',equation:'s_{ij}=q_i k_j',detail:'The query/key dimension is one, so division by its square root changes nothing. No causal mask is used here.',visual:'The query belongs to the destination; each key belongs to a possible source.',draw:matching},
    {part:'02 · Construct one attention head',label:'Turn scores into weights',title:'Three parts socks, one part shoes.',caption:'Exponentiating the two scores gives three and one. Dividing by their total gives the shoes row weights of three quarters and one quarter.',equation:'\\operatorname{softmax}([\\log3\\;0])=[\\tfrac34\\;\\tfrac14]',detail:'The weights sum to one. They describe how much the shoes destination reads from each source.',visual:'The scroll shows exact scores, exponentials, and normalized weights in turn.',draw:softmax,discretePhases:true},
    {part:'02 · Construct one attention head',label:'Now include the socks destination',title:'Every destination gets a row.',caption:'The socks query is zero, so its scores tie and its weights are equal. Stack both distributions into the attention matrix.',equation:'A='+array([['\\tfrac12','\\tfrac12'],['\\tfrac34','\\tfrac14']]),detail:'Rows are destinations and columns are sources. Multiplying this matrix from the left mixes the rows of the input.',visual:'The full-row mixture is AX; the next step chooses which feature to carry.',draw:allAttention},
    {part:'02 · Construct one attention head',label:'Choose what the sources contribute',title:'Carry the position as the value.',caption:'The key identifies which source is socks. The value tells us where that source occurred. Choose a value map that carries only the position coordinate.',equation:'V=XW_V',detail:'Our source preference follows identity, while the information being mixed is position. The same input row supplies both through different maps.',visual:'The attention weights stay fixed while the value map selects the carried feature.',draw:carryPosition},
    {part:'02 · Construct one attention head',label:'Apply the weighted sum',title:'Shoes receives a trace of where socks was.',caption:'The socks destination averages the two positions to zero. The shoes destination puts more weight on socks and receives one half.',equation:'Y=AV',detail:'The matrix multiplication happens from the left. Each destination row is a weighted sum of source values.',visual:'The highlighted calculation is the shoes destination, which reads 75% from socks.',draw:weightedValues},
    {part:'03 · Read the task score',label:'A separate, simple readout',title:'Turn the two outputs into one score.',caption:'Average both destination outputs, then apply a fixed scoring rule. For socks before shoes, the resulting prediction is zero.',equation:'\\hat y=2\\bar y-\\tfrac12',detail:'The averaging and scoring rule are the task readout. They are specified separately from the attention operation.',visual:'Attention gives outputs zero and one half; their average is one quarter.',draw:prediction},
    {part:'03 · Read the task score',label:'Use exactly the same parameters',title:'Reverse the order. Keep the rule.',caption:'Socks now supplies the value minus one. Shoes still reads mostly from socks, so its output becomes minus one half and the final prediction becomes minus one.',equation:'\\hat y=-1',detail:'Queries and keys move with token identity. The position values remain attached to the first and second slots.',visual:'Only the token order changes; all projection maps and the scoring rule remain fixed.',draw:swapped},
    {part:'03 · Read the task score',label:'What makes the distinction possible',title:'We need position and selective mixing.',caption:'Remove position, or make both attention rows uniform, and the two orders collapse to the same prediction. The useful signal needs both ingredients.',equation:'\\hat y=-\\tfrac12\\quad\\text{for both orders}',detail:'This construction demonstrates how attention can move information that makes order distinguishable. It does not imply that every learned head follows this rule.',visual:'Scroll to compare removing position with replacing attention by uniform weights.',draw:ablations,discretePhases:true}
  ];
}
