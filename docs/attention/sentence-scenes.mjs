// Schematic architecture and symbolic weights, not fitted-model measurements.
export const SENTENCE_EXAMPLES = Object.freeze([
  Object.freeze({text: "Good and Infinite Jest don't belong in the same sentence", reading: 'Negative'}),
  Object.freeze({text: "Infinite Jest wasn't just good. It was spectacular.", reading: 'Positive'}),
]);
export const WORD_TOKENS = Object.freeze(['Good', 'and', 'Infinite', 'Jest', "don't", 'belong', 'in', 'the', 'same', 'sentence']);

export function createSentenceScenes({m, line, rect, C}) {
  const {blue, teal, rust, ink, muted, rule} = C;
  const clamp = v => Math.max(0, Math.min(1, v));
  const ease = v => { const x=clamp(v); return x*x*(3-2*x); };
  const lerp = (a,b,t) => a+(b-a)*t;
  const esc = v => String(v).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
  const label = (s,x,y,size=25,color=ink,anchor='middle') => `<text class="sentence-label" x="${x}" y="${y}" text-anchor="${anchor}" style="font-size:${size}px;fill:${color}">${esc(s)}</text>`;
  const math = (s,x,y,w=190,h=55,size=29,color=ink) => m(s,x,y,w,h,size,color);
  const arrow = (x1,y1,x2,y2,color=blue,width=2) => line(x1,y1,x2,y2,color,width,'marker-end="url(#arrow)"');
  const path = (d,color=blue,width=2,end=true) => `<path d="${d}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linecap="round" stroke-linejoin="round"${end?' marker-end="url(#arrow)"':''}/>`;
  const box = (x,y,w,h,color=rule,fill='#fff',width=1.4) => rect(x,y,w,h,fill,color,5,`stroke-width="${width}"`);
  const dot = (x,y,color=blue,r=6) => `<circle cx="${x}" cy="${y}" r="${r}" fill="${color}"/>`;
  const join = (x,y,color) => `<circle cx="${x}" cy="${y}" r="12" fill="#fff" stroke="${color}" stroke-width="2"/><path d="M${x-6} ${y}H${x+6}M${x} ${y-6}V${y+6}" stroke="${color}" stroke-width="2"/>`;
  const opacity = (amount,content) => `<g opacity="${clamp(amount)}">${content}</g>`;
  function featureMatrix(x,y,rows=10,columns=5,color=blue,cell=6,gap=2) {
    let out='';
    for(let i=0;i<rows;i++)for(let j=0;j<columns;j++)out+=rect(x+j*(cell+gap),y+i*(cell+gap),cell,cell,'#fff',color,0);
    return out;
  }
  function locationMarker(x=20,y=25) {
    return path(`M${x+65} ${y+91}V${y+9}`,blue,2)
      +path(`M${x+65} ${y+77}H${x+16}V${y+60}`,teal,1.8,false)
      +path(`M${x+16} ${y+36}V${y+27}H${x+53}`,teal,1.8)
      +box(x-2,y+36,37,24,teal,'#edf7f4',2)
      +featureMatrix(x+7,y+41,3,3,teal,4,2)+join(x+65,y+27,teal)
      +label('This head',x+45,y+119,20,teal);
  }
  const streamX=422, branchX=184;
  const attentionFrame={x:40,y:350,w:288,h:112};
  const inset={x:242,y:384,cell:5,gap:1.5,labelX:274,labelY:366};
  // Arrowheads inherit their path's semantic color; the open spine remains visible.
  function route(points,color,width=2.2,head=true) {
    let out=path(points.map(([x,y],i)=>`${i?'L':'M'}${x} ${y}`).join(''),color,width,false);
    if(head){const [x,y]=points.at(-1),[px,py]=points.at(-2),d=Math.hypot(x-px,y-py),ux=(x-px)/d,uy=(y-py)/d;
      out+=path(`M${x-8*ux+4*uy} ${y-8*uy-4*ux}L${x} ${y}L${x-8*ux-4*uy} ${y-8*uy+4*ux}`,color,width,false);}
    return out;
  }
  function travelingPoint(points,t) {
    const lengths=points.slice(1).map(([x,y],i)=>Math.hypot(x-points[i][0],y-points[i][1]));
    let distance=clamp(t)*lengths.reduce((a,b)=>a+b,0);
    for(let i=0;i<lengths.length;i++){
      if(distance<=lengths[i]||i===lengths.length-1){const p=distance/lengths[i];return [lerp(points[i][0],points[i+1][0],p),lerp(points[i][1],points[i+1][1],p)];}
      distance-=lengths[i];
    }
  }
  function tokenState(y,subscript) {
    let out=line(streamX+7,y,457,y,'#b0c4d3',1.3)+rect(460,y-28,64,56,'#f3f7fa','none',5);
    for(let i=0;i<10;i++)out+=rect(470,y-23+i*5,44,2.5,'#b7cbd9','none',1);
    return out+math(`X_${subscript}`,564,y,76,49,30,blue);
  }
  function architecture(t,{writeUpdate=false,showMotion=true}={}) {
    const sx=streamX;
    let out=path(`M${sx} 490V127`,'#eef3f7',15,false)
      +route([[sx,490],[sx,341]],blue,3)+route([[sx,315],[sx,203]],blue,3)+route([[sx,177],[sx,127]],blue,3)
      +route([[sx,490],[branchX,490],[branchX,462]],teal)
      +route([[branchX,350],[branchX,328],[sx-13,328]],teal)
      +route([[sx,304],[branchX,304],[branchX,288]],rust)
      +route([[branchX,214],[branchX,190],[sx-13,190]],rust);
    // The pulse follows a continuous route, disappearing inside each operation.
    // Drawing cards afterward keeps the moving mark away from their labels.
    if(showMotion){const [x,y]=travelingPoint([[sx,490],[branchX,490],[branchX,350],[branchX,328],[sx,328],[sx,304],[branchX,304],[branchX,214],[branchX,190],[sx,190],[sx,127]],ease(t));
      out+=dot(x,y,x===sx?blue:y>315?teal:rust,5);}
    out+=box(attentionFrame.x,attentionFrame.y,attentionFrame.w,attentionFrame.h,'#9bc8c0','#f5faf8',1.4)
      +box(40,214,288,74,'#d6b7a7','#fcf7f3',1.4)
      +label('MLP',62,244,28,rust,'start')+label('shared across tokens',62,273,21,muted,'start')
      +join(sx,328,teal)+join(sx,190,rust)
      +'<g aria-label="Residual stream">'+label('Residual',461,370,24,blue,'start')+label('stream',461,399,24,blue,'start')+'</g>';
    out+=writeUpdate?label('Attention update',62,381,25,teal,'start')+math('AXW_VW_O',184,424,250,52,30,teal)
      :label('Attention',62,390,28,teal,'start')+label('one head shown',62,420,21,muted,'start')
        +'<g class="attention-matrix-inset" aria-label="Symbolic ten by ten attention matrix A">'
        +featureMatrix(inset.x,inset.y,10,10,teal,inset.cell,inset.gap)+math('A',inset.labelX,inset.labelY,45,34,25,teal)+'</g>';
    out+=tokenState(453,0)+tokenState(276,1)+tokenState(155,2)
      +label('Embeddings + positions',sx,529,24,blue)
      +box(307,73,230,54,'#b0c4d3','#f5f8fb',1.4)
      +(writeUpdate?math('r=\\operatorname{mean}_i(X_2)_i',sx,100,225,48,23,blue):label('Mean pooling',sx,109,26,blue))
      +route([[sx,73],[sx,53]],blue,2)
      +math(writeUpdate?'z=b+r\\beta':'z',sx,29,260,44,writeUpdate?25:32,blue)
      +label(writeUpdate?'Class score':'Sentiment readout',61,36,22,muted,'start');
    return out;
  }
  function compareSentences(t) {
    const active=clamp(t)<.5?0:1;
    return label('Sentiment toward Infinite Jest',320,42,28)
      +box(34,77,572,148,active===0?blue:rule,'#fff',active===0?2.6:1.4)
      +label('Good and Infinite Jest',62,119,28,ink,'start')+label("don't belong in the same sentence",62,161,27,ink,'start')
      +label('Intended reading: negative',62,204,22,rust,'start')
      +box(34,267,572,148,active===1?blue:rule,'#fff',active===1?2.6:1.4)
      +label("Infinite Jest wasn't just good.",62,309,28,ink,'start')+label('It was spectacular.',62,351,28,ink,'start')
      +label('Intended reading: positive',62,394,22,teal,'start')+label('The surrounding phrase changes the reading.',320,489,25,muted);
  }
  function expandedHead(t=1) {
    const p=ease(t),x=lerp(attentionFrame.x,197,p),y=lerp(attentionFrame.y,86,p),w=lerp(attentionFrame.w,405,p),h=lerp(attentionFrame.h,385,p);
    let out=box(x,y,w,h,teal,'#fff',1.6)
      +opacity(1-p/.2,label('Attention',62,390,28,teal,'start')+label('one head shown',62,420,21,muted,'start'))
      +opacity((p-.8)/.2,label('One attention head',399.5,121,28,teal))
      +math('A',lerp(inset.labelX,286,p),lerp(inset.labelY,157,p),100,51,lerp(25,32,p),teal)
      +featureMatrix(lerp(inset.x,236,p),lerp(inset.y,180,p),10,10,teal,lerp(inset.cell,8,p),lerp(inset.gap,2,p));
    if(p>.9)out+=opacity((p-.9)/.1,math('\\times',374,228,54,55,35)+math('V=XW_V',491,157,202,51,29,blue)+featureMatrix(461,180,10,6,blue,8,2)
      +label('Mix source values',399,307,21,muted)+math('U=AV',399,341,330,54,32,teal)
      +arrow(399,367,399,377,teal)+math('\\Delta X=UW_O',399,409,370,55,32,teal)
      +label('Write back to the stream',399,455,23,muted));
    return out;
  }
  const zoomHead=t=>opacity(1-.88*ease(t),architecture(0,{showMotion:false}))+expandedHead(t)+opacity(ease(t),locationMarker(19,31));
  function attentionGrid({x=216,y=151,cell=31,row=-1,column=-1,labelOpacity=1}={}) {
    let out='';
    if(row>=0)out+=rect(x,y+row*cell,10*cell,cell,'#edf7f4','none',0);
    for(let i=0;i<=10;i++)out+=line(x,y+i*cell,x+10*cell,y+i*cell,rule,1)+line(x+i*cell,y,x+i*cell,y+10*cell,rule,1);
    if(row>=0)out+=rect(x,y+row*cell,10*cell,cell,'none',teal,0,'stroke-width="2.4"');
    if(column>=0&&row>=0)out+=rect(x+column*cell,y+row*cell,cell,cell,'#fff',rust,0,'stroke-width="3"');
    let names=label('Destination',x-18,y-21,21,blue,'end');
    WORD_TOKENS.forEach((word,i)=>{names+=label(word,x-17,y+(i+.5)*cell+7,21,i===row?teal:ink,'end')+label(i+1,x+(i+.5)*cell,y-20,20,i===column?rust:muted);});
    return out+opacity(labelOpacity,names);
  }
  function extractMatrix(t) {
    const p=ease(t),scale=lerp(10/31,1,p),tx=lerp(236-216*(10/31),0,p),ty=lerp(180-151*(10/31),0,p);
    return opacity(1-p,expandedHead(1))+`<g transform="translate(${tx},${ty}) scale(${scale})">${attentionGrid({labelOpacity:p})}</g>`
      +opacity(p,locationMarker(20,16)+label('Source position',371,53,27,teal)+label('in the same sentence',371,87,21,muted)
        +math('A\\in\\mathbb R^{10\\times10}',371,508,430,58,33,teal));
  }
  function readEntry(t) {
    return locationMarker(20,16)+label('Column 5: don’t',371,55,27,rust)+label('Row 1: Good',371,91,23,teal)
      +attentionGrid({row:0,column:4})+label('don’t',227,518,28,rust)+arrow(272,508,444,508,teal)
      +dot(lerp(280,435,ease(t)),508,teal,5)+label('Good',493,518,28,teal);
  }
  function readRow(t) {
    const p=ease(t);
    return locationMarker(20,16)+attentionGrid({x:lerp(216,253,p),y:lerp(151,75,p),cell:lerp(31,24,p),row:0})
      +opacity(p,label('The weights used by Good',345,362,26,teal)+math('a_{11},\\ a_{12},\\ldots,\\ a_{1,10}',345,407,535,65,31,teal)
        +math('a_{1j}\\geq0,\\qquad\\sum_{j=1}^{10}a_{1j}=1',345,485,550,67,30,teal));
  }
  function combineValues(t) {
    const total=clamp(t)*10,active=Math.min(9,Math.floor(total)),p=ease(total-Math.floor(total));
    let out=label('Source',115,47,25)+math('V',239,42,100,55,32,blue)+label('Weight',355,47,25,teal)
      +label('Good',543,214,26,teal)+box(486,234,120,82,teal,'#fff',2)+math('u_1',546,275,104,67,40,teal);
    WORD_TOKENS.forEach((word,j)=>{const y=100+j*38,color=j===active?teal:muted;
      out+=label(word,173,y+7,22,j===active?ink:muted,'end')+math(`v_{${j+1}}`,240,y,98,39,27,blue)+math(`a_{1,${j+1}}`,355,y,115,39,27,color)
        +opacity(j===active?1:.18,arrow(410,y,475,275,color,j===active?2.4:1));
      if(j===active)out+=box(190,y-18,217,36,teal,'none',1.7);
    });
    if(total<10)out+=dot(lerp(414,470,p),lerp(100+active*38,275,p),teal,6);
    return out+label('A weighted sum of all ten value vectors.',320,523,25,muted);
  }
  return [
    {id:'sentence-stream',part:'01 · From a sentence to a score',label:'',title:'Follow the residual stream.',
      caption:'Each token carries a row of features. Attention and an MLP add updates; a readout scores the final sentence.',
      equation:'r=\\frac1T\\sum_i(X_2)_i,\\qquad z=b+r\\beta',
      detail:'One encoder block; one head shown. Normalization and attention biases omitted. A is symbolic, with no measured weights. Readings are labels, not model outputs.',
      visual:'The attention matrix A sits on the branch that adds an update to the blue residual stream.',quotes:SENTENCE_EXAMPLES,draw:architecture},
    {id:'sentence-contrast',part:'01 · From a sentence to a score',label:'',title:'“Good” is not enough.',
      caption:'The first sentence rejects “good”; the second says “better than good.” Negation changes meaning through its surrounding phrase.',
      equation:'z=f(\\text{whole sentence})',detail:'The task is sentiment toward Infinite Jest. These qualitative readings are not outputs from a fitted model.',
      visual:'Shared words can occur in sentences with opposite sentiment.',draw:compareSentences,discretePhases:true},
    {id:'sentence-head',part:'02 · Inside attention',label:'',title:'Open one attention head.',
      caption:'X holds one feature row per token. The value projection reads features, A mixes them across positions, and the output projection writes them back.',
      equation:'\\Delta X=AXW_VW_O',
      detail:'The value map <span data-math="W_V"></span> forms <span data-math="V=XW_V"></span>. The output map <span data-math="W_O"></span> returns mixtures <span data-math="U=AV"></span> to stream width. V and U contain features, not outcome labels.',
      visual:'The highlighted branch remains attached to its place in the architecture.',draw:zoomHead},
    {id:'sentence-matrix',part:'02 · Inside attention',label:'',title:'Pull out the attention matrix.',
      caption:'Each row is a destination. Each column is a source. Both axes refer to positions in the first sentence.',equation:'A=(a_{ij})',
      detail:'For this schematic, the ten displayed words are tokens and punctuation is omitted. A model may split them further. No measured weights are shown.',
      visual:'The small architecture marker locates this matrix inside the selected attention head.',draw:extractMatrix},
    {id:'sentence-entry',part:'03 · Read the matrix',label:'',title:'One entry connects two positions.',
      caption:'The outlined entry weights information from “don’t” when computing the output at “Good.”',
      equation:'a_{15}:\\quad\\text{source 5}\\longrightarrow\\text{destination 1}',
      detail:'The outlined cell illustrates the indexing. It makes no claim that a trained model gives this pair a large weight.',
      visual:'Information flows from the source column to the destination row.',draw:readEntry},
    {id:'sentence-row',part:'03 · Read the matrix',label:'',title:'The whole row specifies a mixture.',
      caption:'The row for “Good” assigns one weight to each source position, including “Good” itself.',equation:'\\sum_{j=1}^{10}a_{1j}=1',
      detail:'Weights are nonnegative. Rows sum to one; columns generally do not. The matrix need not be symmetric.',
      visual:'The weights are symbolic; cell shading does not encode invented values.',draw:readRow},
    {id:'sentence-values',part:'03 · Read the matrix',label:'',title:'Mix the value vectors.',
      caption:'Each source supplies a value vector: its feature row transformed by the value projection. Multiply each value by its row weight, then add the vectors.',
      equation:'u_1=\\sum_{j=1}^{10}a_{1j}v_j,\\qquad U=AV',
      detail:'This produces one output vector at each destination. At later layers, source vectors can already contain context from other positions.',
      visual:'The matrix of weights acts on the matrix of source values.',draw:combineValues},
    {id:'sentence-return',part:'04 · Return to the prediction',label:'',title:'Write the update back.',
      caption:'The output projection maps the mixtures back to stream width; add them to X. After the MLP update, pooling and a readout produce the sentence score.',equation:'X_1=X_0+AX_0W_VW_O',
      detail:'The attention branch supplies an update, not the final score. In a multi-head block, all projected head updates are added.',
      visual:'Final token rows feed the sentence summary and classifier; no numerical sentiment score is claimed here.',draw:t=>architecture(t,{writeUpdate:true})},
  ];
}
