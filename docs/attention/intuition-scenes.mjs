// Word-sense intuition. Symbols and routes are schematic, not model measurements.
export const DESERT_EXAMPLES = Object.freeze([
  Object.freeze({
    text: 'The jury was concerned with moral desert and not the laws.',
    reading: 'What someone deserves',
  }),
  Object.freeze({
    text: 'The upper west side is a taco bell desert.',
    reading: 'A place where something is scarce',
  }),
]);

export function createIntuitionScenes({m, line, rect, C}) {
  const {blue, teal, rust, ink, muted, rule} = C;
  const clamp = t => Math.max(0, Math.min(1, t));
  const ease = t => { const p=clamp(t); return p*p*(3-2*p); };
  const lerp = (a,b,t) => a+(b-a)*t;
  const esc = value => String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
  const label = (s,x,y,size=25,color=ink,anchor='middle') => `<text x="${x}" y="${y}" text-anchor="${anchor}" style="font-size:${size}px;fill:${color}">${esc(s)}</text>`;
  const math = (s,x,y,w=250,h=55,size=30,color=blue) => m(s,x,y,w,h,size,color);
  const box = (x,y,w,h,color=rule,fill='#fff',width=1.5) => rect(x,y,w,h,fill,color,5,`stroke-width="${width}"`);
  const dot = (x,y,color=blue) => `<circle cx="${x}" cy="${y}" r="5" fill="${color}"/>`;
  const fade = (t,s) => `<g opacity="${clamp(t)}">${s}</g>`;
  const path = (points,color=blue,width=2,head=true) => {
    let out=`<path d="${points.map(([x,y],i)=>`${i?'L':'M'}${x} ${y}`).join('')}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linecap="round" stroke-linejoin="round"/>`;
    if(head){
      const [x,y]=points.at(-1),[px,py]=points.at(-2),d=Math.hypot(x-px,y-py),ux=(x-px)/d,uy=(y-py)/d;
      out+=`<path d="M${x-8*ux+4*uy} ${y-8*uy-4*ux}L${x} ${y}L${x-8*ux-4*uy} ${y-8*uy+4*ux}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linecap="round"/>`;
    }
    return out;
  };
  function pointOnPath(points,t) {
    const lengths=points.slice(1).map(([x,y],i)=>Math.hypot(x-points[i][0],y-points[i][1]));
    let distance=clamp(t)*lengths.reduce((a,b)=>a+b,0);
    for(let i=0;i<lengths.length;i++){
      if(distance<=lengths[i]||i===lengths.length-1){const p=distance/lengths[i];return [lerp(points[i][0],points[i+1][0],p),lerp(points[i][1],points[i+1][1],p)];}
      distance-=lengths[i];
    }
  }
  const sentenceLine = (before,after,x,y) => `<text x="${x}" y="${y}" style="font-size:30px;fill:${ink}">${esc(before)}<tspan style="fill:${blue};text-decoration:underline;text-underline-offset:5px">desert</tspan>${esc(after)}</text>`;
  function sameWord(t) {
    const active=clamp(t)<.5?0:1;
    return `<g aria-label="${esc(DESERT_EXAMPLES[0].text)}">`
      +box(28,57,584,182,active===0?rust:rule,'#fff',active===0?2.4:1.3)
      +label('The jury was concerned with',54,111,30,ink,'start')
      +sentenceLine('moral ',' and not the laws.',54,152)
      +label('What someone deserves',54,208,23,rust,'start')+'</g>'
      +`<g aria-label="${esc(DESERT_EXAMPLES[1].text)}">`
      +box(28,286,584,182,active===1?teal:rule,'#fff',active===1?2.4:1.3)
      +label('The upper west side is',54,340,30,ink,'start')
      +sentenceLine('a taco bell ','.',54,381)
      +label('A place where something is scarce',54,437,23,teal,'start')+'</g>'
      +label('The spelling alone does not choose the sense.',320,518,23,muted);
  }
  function sharedLookup(t) {
    const routes=[[[160,176],[160,218],[265,258]],[[480,176],[480,218],[375,258]]];
    let out=label('Two occurrences',320,43,27,ink)
      +label('moral desert',160,84,22,rust)+label('taco bell desert',480,84,22,teal)
      +box(85,108,150,68,blue,'#f5f8fb')+label('desert',160,151,29,blue)
      +box(405,108,150,68,blue,'#f5f8fb')+label('desert',480,151,29,blue);
    routes.forEach(route=>{out+=path(route,blue);const [x,y]=pointOnPath(route,clamp(t/.55));out+=dot(x,y,blue);});
    out+=box(218,258,204,95,blue,'#f5f8fb')
      +label('One lookup row',320,289,22,blue)+math('e_{\\mathrm{desert}}',320,326,190,52,34,blue)
      +fade(clamp((t-.35)/.4),path([[268,353],[160,403]],blue)+path([[372,353],[480,403]],blue))
      +fade(clamp((t-.55)/.4),math('e_{\\mathrm{desert}}',160,436,220,55,31,blue)+math('e_{\\mathrm{desert}}',480,436,220,55,31,blue))
      +label('Then add position information.',320,506,24,muted);
    // The shared row is copied into both occurrences as the scroll advances.
    if(t>.55&&t<.95){
      const q=ease((t-.55)/.4);
      out+=dot(lerp(268,160,q),lerp(353,403,q),blue)+dot(lerp(372,480,q),lerp(353,403,q),blue);
    }
    return out;
  }
  function contextUpdates(t) {
    const p=ease(t);
    let out=label('Different context can supply different updates.',320,36,24,ink);
    [
      {x:160,sub:'i',name:'Jury sentence',cue:'jury · moral',color:rust},
      {x:480,sub:'k',name:'Restaurant sentence',cue:'taco · bell',color:teal},
    ].forEach(({x,sub,name,cue,color})=>{
      out+=label(name,x,84,23,color)+label('desert',x,124,28,blue)
        +box(x-75,144,150,62,blue,'#f5f8fb')+math(`x_${sub}^{(0)}`,x,175,142,50,31,blue)
        +path([[x,206],[x,323]],blue)+path([[x,347],[x,395]],blue)
        +box(x-140,248,120,54,color,'#fff')+label(cue,x-80,283,20,color);
      const route=[[x-80,302],[x-80,335],[x-12,335]];
      out+=path(route,color)+dot(...pointOnPath(route,p),color)
        +`<circle cx="${x}" cy="335" r="12" fill="#fff" stroke="${color}" stroke-width="2"/>`
        +line(x-6,335,x+6,335,color,2)+line(x,329,x,341,color,2)
        +label('context update',x-80,365,17,color)
        +box(x-75,395,150,64,color,'#fff')+math(`x_${sub}^{(1)}`,x,427,142,50,32,color)
        +label('contextual state',x,493,22,muted);
    });
    return out;
  }

  function learnedMaps(t) {
    const learned=name=>`\\underbrace{\\color{${rust}}${name}}_{\\text{learned}}`;
    let out=label('Fixed learned maps; calculations depend on the input.',320,28,23,ink)
      +math(`Q=X${learned('W_Q')}`,167,103,296,112,30,blue)
      +math(`K=X${learned('W_K')}`,473,103,296,112,30,blue);
    const matching=[[[167,161],[272,188]],[[473,161],[368,188]]];
    matching.forEach(route=>{out+=path(route,blue);if(t<.5)out+=dot(...pointOnPath(route,ease(t*2)),blue);});
    out+=math('A(X)=\\operatorname{softmax}_{\\mathrm{rows}}\\!\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right)',320,233,620,110,29,teal)
      +label('Computed attention coefficients',320,294,23,teal)
      +line(25,316,615,316,rule,1.3)
      +math(`V=X${learned('W_V')}`,164,378,300,106,30,blue)
      +math('A(X)',476,358,220,60,32,teal)
      +label('computed above',476,402,20,muted);
    const writes=[[[164,432],[258,450]],[[476,432],[382,450]]];
    writes.forEach(route=>{out+=path(route,teal);if(t>=.5)out+=dot(...pointOnPath(route,ease((t-.5)*2)),teal);});
    return out+math(`\\Delta X=A(X)V${learned('W_O')}`,320,492,620,90,31,teal);
  }
  function repeatedUpdates(t) {
    const mlp=clamp(t)>=.5,local=mlp?(clamp(t)-.5)*2:clamp(t)*2;
    const color=mlp?rust:teal;
    let out=label(mlp?'MLP: transform each token row separately':'Attention: mix across token positions',320,32,26,color)
      +label('Incoming rows',128,72,22,muted)+label('Updated rows',512,72,22,muted);
    const ys=[115,185,255];
    for(let i=0;i<3;i++)for(let j=0;j<3;j++)if(!mlp||i===j){
      out+=path([[211,ys[j]],[429,ys[i]]],i===j?color:'#b8c4cb',i===j?1.8:1.2);
    }
    const destination=mlp?1:0,source=mlp?1:2;
    out+=dot(lerp(211,429,ease(local)),lerp(ys[source],ys[destination],ease(local)),color);
    ys.forEach((y,i)=>{
      out+=box(45,y-25,166,50,blue,'#f5f8fb')+math(mlp?`x_${i+1}'`:`x_${i+1}`,128,y,154,49,31,blue)
        +box(429,y-25,166,50,color,'#fff')+math(mlp?`x_${i+1}^{\\mathrm{next}}`:`x_${i+1}'`,512,y,154,49,31,color);
    });
    return out+math("X'=X+\\Delta_{\\mathrm{attn}}X",320,348,610,68,33,mlp?muted:teal)
      +label('Add the attention update',320,393,22,mlp?muted:teal)
      +math("X_{\\mathrm{next}}=X'+\\operatorname{MLP}(X')",320,454,620,75,31,mlp?rust:muted)
      +label('Add the row-wise MLP update',320,510,22,mlp?rust:muted);
  }

  return [
    {id:'desert-senses',part:'Before attention · Meaning in context',label:'',title:'The same word, used differently.',
      caption:'“Moral desert” concerns what someone deserves. A “taco bell desert” describes scarcity. The surrounding words tell us which use is intended.',
      equation:'',
      detail:'These are readings of the example sentences, not predictions from a model.',
      visual:'Both complete sentences contain desert, with a different intended sense in each.',quotes:DESERT_EXAMPLES,draw:sameWord,discretePhases:true},
    {id:'desert-lookup',part:'Before attention · Meaning in context',label:'',title:'The lookup starts the same way.',
      caption:'The same token selects the same learned embedding vector in either sentence. That lookup alone does not select the sense of this occurrence.',
      equation:'e_{\\mathrm{desert}}',
      detail:'Treat “desert” as one token here. The lookup vector is shared; full input rows can differ because position information is added. Models may also normalize or otherwise process those rows.',
      visual:'Both occurrences select one shared lookup row; position information is a separate input.',draw:sharedLookup},
    {id:'desert-context',part:'Before attention · Meaning in context',label:'',title:'Let the context change the representation.',
      caption:'A useful representation of “desert” should depend on its neighbors. Attention provides a way for other tokens to contribute information at that position.',
      equation:'x_i^{(1)}=x_i^{(0)}+\\Delta x_i',
      detail:'The arrows illustrate useful context clues, not measured attention weights or proof that a model has learned these senses. The update shown is schematic; normalization is omitted.',
      visual:'Context can contribute different updates even when the token lookup vector is shared.',draw:contextUpdates},
    {id:'learned-maps',part:'Before attention · Parameters and calculations',label:'',title:'Learn the maps. Compute the attention.',
      caption:'The W matrices are learned parameters and stay fixed during inference. The attention weights in A are computed from the current token vectors; they are not learned parameters.',
      equation:'\\Delta X=A(X)XW_VW_O',
      detail:'Within a head, each map is shared across positions; different heads and layers generally have different maps. Q and K determine A. The value map <span data-math="W_V"></span> supplies the content to mix; it does not directly determine this head’s A for a given X. Here <span data-math="d_k"></span> is the query/key width; row-wise softmax makes each row of A sum to one.',
      visual:'Every W is a learned parameter. Q, K, V, A and the update are computed for the current representations.',draw:learnedMaps},
    {id:'repeated-updates',part:'Before attention · Updating token vectors',label:'',title:'Keep updating the token vectors.',
      caption:'Each token starts with a lookup vector and position information. Attention mixes across positions; the MLP transforms each row separately. Both add updates to the running token representations.',
      equation:"X'=X+\\Delta_{\\mathrm{attn}}X",
      detail:'Blocks repeat. Useful features can be added, revised or canceled; each update need not improve the representation. The full model sums the attention heads’ projected updates. These equations omit normalization and biases.',
      visual:'Attention can connect different token rows. The MLP applies the same function to each row without mixing positions.',draw:repeatedUpdates,discretePhases:true},
  ];
}
