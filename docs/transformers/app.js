/* Architecture-only diagrams. Color cells are schematic, not model outputs. */
const tokens = ['voters','support','the','proposal'];
const steps = [...document.querySelectorAll('.step')];
let current = -1;
let selectedRow = 0;
let sceneProgress = 0;
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const diagram = document.getElementById('diagram');
const detail = document.getElementById('stage-detail');
const titles = ['The residual stream begins','Two parallel updates','One shared MLP','Trace each token row','Pool after the block','Only the prefix','Read every position','Use the final position'];
const descriptions = [
  'A bottom-to-top residual-stream diagram. Token embeddings plus positions form X zero. Two attention heads read X zero in parallel and add to it to form X one. The MLP reads X one and adds to it to form X two. An encoder readout follows.',
  'Two parallel attention heads both read X zero. Their outputs add to the unchanged X zero to form X one. Neither head reads the other head’s output.',
  'Four token rows pass horizontally through the same MLP. Each output row depends only on the corresponding input row. The MLP output is added back to X one.',
  'Scrolling traces each token row horizontally through one shared MLP. There are no paths between token rows.',
  'Four rows of X two merge through a mean-pooling operation into one row vector r. A scalar head maps r to a single prediction.',
  'Two four-by-four causal attention masks. Row i can use columns one through i, including its own position. All cells above the diagonal are blocked.',
  'Every row of X two passes through the same unembedding matrix W U to produce vocabulary logits. Rowwise softmax produces one next-token distribution for each position.',
  'The final token row passes through the unembedding and softmax. Its distribution is the one used to predict the token after the supplied sequence.'
];

function escapeHtml(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function math(tex){return `<span data-tex="${escapeHtml(tex)}"></span>`;}
window.mathErrors = [];
function renderMath(root=document){
  root.querySelectorAll('[data-tex]').forEach(el=>{
    if(el.dataset.rendered)return;
    try{katex.render(el.dataset.tex,el,{throwOnError:true,displayMode:el.hasAttribute('data-display'),output:'htmlAndMathml',strict:'error'});el.dataset.rendered='true';}
    catch(error){window.mathErrors.push({tex:el.dataset.tex,message:error.message});el.classList.add('katex-error');el.textContent=el.dataset.tex;}
  });
}
function fo(x,y,w,h,html,cls=''){return `<foreignObject x="${x}" y="${y}" width="${w}" height="${h}"><div xmlns="http://www.w3.org/1999/xhtml" class="svg-label ${cls}">${html}</div></foreignObject>`;}
function line(d,color='',arrow=true,width=''){return `<path d="${d}" class="svg-line ${color} ${width}" ${arrow?`marker-end="url(#arrow${color==='teal'?'T':color==='rust'?'R':''})"`:''}/>`;}
function rect(x,y,w,h,fill='var(--stage)',stroke='var(--line)',r=4){return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" stroke="${stroke}"/>`;}
function add(x,y,color='var(--ink)'){return `<circle cx="${x}" cy="${y}" r="11" fill="var(--stage)" stroke="${color}" stroke-width="1.5"/>${fo(x-10,y-11,20,21,math('+'))}`;}
function box(x,y,w,h,label,color='',sub=''){
 const fill=color==='teal'?'var(--teal-light)':color==='rust'?'var(--rust-light)':'var(--stage)';
 const stroke=color==='teal'?'var(--teal)':color==='rust'?'var(--rust)':'var(--line)';
 return rect(x,y,w,h,fill,stroke)+fo(x+4,y+2,w-8,h-4,label+(sub?`<span class="subtitle">${sub}</span>`:''),`${color} ${sub?'vertical':''}`);
}
function cells(x,y,rows=4,cols=6,size=15,gap=4,highlight=-1,color='teal'){
 let out='';
 for(let i=0;i<rows;i++)for(let j=0;j<cols;j++){
  const active=highlight<0||highlight===i;
  const intensity=.20+((i*7+j*3+2)%9)*.065;
  const base=color==='rust'?'165,79,50':'0,127,115';
  out+=rect(x+j*(size+gap),y+i*(size+gap),size,size,`rgba(${base},${active?intensity:.08})`,'none',2);
 }
 return out;
}
function vector(x,y,cols=6,size=15,gap=4,color='teal',active=true,offset=0){
 let out=''; for(let j=0;j<cols;j++){const base=color==='rust'?'165,79,50':'0,127,115';const alpha=.18+((j*3+offset*7+2)%9)*.065;out+=rect(x+j*(size+gap),y,size,size,`rgba(${base},${active?alpha:.09})`,'none',2);}return out;
}
function svgStart(){return `<svg class="desktop-diagram" viewBox="0 0 620 560" role="img" aria-label="${escapeHtml(descriptions[current])}" xmlns="http://www.w3.org/2000/svg"><defs><marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#234E70"/></marker><marker id="arrowT" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#007F73"/></marker><marker id="arrowR" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#A54F32"/></marker><pattern id="blocked" width="7" height="7" patternUnits="userSpaceOnUse"><rect width="7" height="7" fill="#f4f4f4"/><path d="M0 7 L7 0" stroke="#d9d9d9" stroke-width="1"/></pattern></defs>`;}
function routeSvg(active='mlp'){
 let s=''; const xs=[38,170,306,440],labels=[math('X_0'),'Attention','MLP',math('X_2')];
 xs.forEach((x,i)=>{s+=box(x,14,104,36,labels[i],(i===1&&active==='attention')?'teal':(i===2&&active==='mlp')?'rust':'');if(i<3)s+=line(`M${x+108} 32 H${xs[i+1]-8}`);});
 return s+fo(55,56,500,22,'ONE BLOCK · BOTH RESIDUAL ADDITIONS RETAINED','tiny');
}
function mainMap(){
 const a=current===1?'':'dim',m='dim';
 return svgStart()+
 `<g class="svg-group dim">${line('M460 189 V114')}${box(350,67,220,46,'Mean pool → scalar')}${fo(478,135,115,48,math('X_2')+'<span class="subtitle">'+math('4\\times6')+'</span>','vertical left')}</g>`+
 `<g class="svg-group ${m}">${line('M460 329 V201','',false)}${line('M460 290 H208 V264','rust')}${box(110,217,196,46,'Shared MLP','rust')}${line('M208 217 V190 H448','rust')}${add(460,190,'var(--rust)')}${fo(180,155,145,25,'mixes features','tiny rust')}${fo(482,267,110,50,math('X_1')+'<span class="subtitle">'+math('4\\times6')+'</span>','vertical left')}</g>`+
 `<g class="svg-group ${a}">${line('M460 465 V352','',false,'heavy')}${line('M460 454 H130 V414')}${line('M460 454 H305 V414')}${box(57,367,146,46,'Head 1','teal')}${box(232,367,146,46,'Head 2','teal')}${line('M130 367 V340 H448','teal')}${line('M305 367 V340 H448','teal')}${add(460,340,'var(--teal)')}${fo(75,296,345,32,math('X_1=X_0+h_1+h_2'))}${fo(48,467,292,24,'Both heads read the same input','tiny')}</g>`+
 `${box(350,476,220,47,math('X_0=SW_E+P'))}${line('M460 476 V454')}`+
 `${fo(24,90,230,100,`<span style="font-family:var(--serif);font-size:27px;line-height:1.25">Read from the stream.<br/>Add back to it.</span>`,'left')}`+
 `${fo(5,218,100,30,math('\\operatorname{MLP}(X_1)'),'small rust')}`+
 `${fo(205,530,410,24,tokens.join(' &nbsp; / &nbsp; '),'small')}`+
 `${fo(347,12,230,28,'INFORMATION FLOWS UPWARD ↑','tiny')}`+
 `${mainMotion()}</svg>`;
}
function mlpMap(){
 const focus=current===3?selectedRow:-1;
 let s=svgStart()+routeSvg('mlp');
 s+=fo(94,103,135,30,math('X_1'),'')+fo(264,103,125,30,'Same MLP','rust')+fo(438,103,144,30,math('\\operatorname{MLP}(X_1)'), 'rust');
 s+=fo(104,133,115,24,math('T\\times d'),'small')+fo(446,133,120,24,math('T\\times d'),'small');
 s+=rect(270,168,112,228,'var(--rust-light)','var(--rust)',6);
 for(let i=0;i<4;i++){
   const y=183+i*54,active=focus<0||i===focus;
   if(focus===i)s+=rect(0,y-8,610,38,'rgba(165,79,50,.06)','none',3);
   s+=fo(0,y-4,86,28,tokens[i],`small left`);
   s+=vector(96,y,6,16,4,'teal',active,i);
   s+=`<g opacity="${active?1:.2}">${line(`M219 ${y+8} H260`,'rust')}${fo(282,y-5,88,30,math('\\operatorname{MLP}'),'rust')}${line(`M388 ${y+8} H434`,'rust')}</g>`;
   s+=vector(444,y,6,16,4,'rust',active,i+1);
   if(i<3)s+=`<path d="M280 ${y+37} H372" stroke="#d7b6aa" stroke-dasharray="3 4"/>`;
 }
 s+=fo(262,403,125,22,'Shared weights','tiny rust');
 s+=fo(30,450,560,40,math('d=6\\ \\longrightarrow\\ m=10\\ \\longrightarrow\\ d=6'));
 s+=fo(40,489,540,22,'FEATURE WIDTH CHANGES. FOUR ROWS STAY FOUR ROWS.','tiny');
 s+=fo(30,526,560,31,math('X_2=X_1+\\operatorname{MLP}(X_1)'),'');
 return s+rowMotion('mlp')+'</svg>';
}
function poolMap(){
 let s=svgStart()+routeSvg('');
 s+=fo(54,111,210,32,math('X_2\\in\\mathbb R^{4\\times6}'));
 for(let i=0;i<4;i++){
  const y=181+i*48;s+=fo(0,y-6,85,29,tokens[i],'small left')+vector(100,y,6,16,4,'teal',true,i);
  s+=line(`M224 ${y+8} C290 ${y+8}, 285 262, 350 262`,'teal');
 }
 s+=vector(366,254,6,21,5,'teal',true,2);
 s+=fo(350,207,176,30,math('r\\in\\mathbb R^{1\\times6}'));
 s+=fo(240,355,335,24,'MEAN OVER TOKEN ROWS','tiny teal');
 s+=line('M441 282 V417');
 s+=box(360,419,164,51,math('rw+b'));
 s+=line('M441 471 V511');
 s+=fo(398,514,88,40,math('\\hat y'));
 s+=fo(10,415,250,75,'The MLP is finished.<br/>Pooling is a new operation.','small left');
 return s+poolMotion()+'</svg>';
}
function maskMap(){
 let s=svgStart()+routeSvg('attention');
 s+=fo(95,96,440,28,'A causal mask in each head','');
 for(let h=0;h<2;h++){
  const x=80+h*295,y=211,size=42,gap=5;
  s+=fo(x,130,182,28,math(`A^{(${h+1})}`),'teal');
  s+=fo(x-12,163,204,22,'SOURCE POSITION '+math('k'),'tiny');
  for(let j=0;j<4;j++)s+=fo(x+j*47,y-23,42,20,String(j+1),'small');
  for(let i=0;i<4;i++){
   s+=fo(x-28,y+i*47+6,23,28,String(i+1),'small');
   for(let j=0;j<4;j++){
    const allowed=j<=i,active=i===selectedRow;
    s+=rect(x+j*47,y+i*47,size,size,allowed?(active?'#80bfb9':'#e0efed'):'url(#blocked)','none',3);
    if(!allowed)s+=fo(x+j*47,y+i*47,42,42,math('0'),'small');
   }
   if(i===selectedRow)s+=rect(x-3,y+i*47-3,189,48,'none','var(--teal)',5);
  }
 }
 s+=fo(51,415,535,28,'RECEIVING POSITION '+math('i')+' IS THE MATRIX ROW','tiny');
 s+=rect(130,467,13,13,'#80bfb9','none',2)+fo(152,461,140,26,'Available source','small left');
 s+=rect(342,467,13,13,'url(#blocked)','none',2)+fo(364,461,135,26,'Future: blocked','small left');
 s+=fo(70,511,480,40,math(`i=${selectedRow+1}:\\quad k\\in\\{${Array.from({length:selectedRow+1},(_,i)=>i+1).join(',')}\\}`),'teal');
 return s+'</svg>';
}
function vocabMap(){
 const last=current===7;
 let s=svgStart()+routeSvg('');
 s+=fo(67,106,140,38,math('X_2'),'')+fo(234,106,135,38,math('W_U'),'')+fo(423,106,173,38,math('p_i'),'');
 s+=fo(57,144,155,26,math('T\\times d'),'small')+fo(234,144,135,26,math('d\\times V'),'small')+fo(420,144,180,26,'Vocabulary distributions','small');
 s+=rect(256,183,95,236,'var(--teal-light)','var(--teal)',5);
 for(let i=0;i<4;i++){
  const y=198+i*57,active=!last||i===3;
  if(last&&i===3)s+=rect(0,y-10,617,42,'rgba(0,127,115,.08)','none',3);
  s+=fo(0,y-5,78,27,tokens[i],'small left');
  s+=vector(86,y,6,14,3,'teal',active,i);
  s+=`<g opacity="${active?1:.18}">${line(`M192 ${y+7} H244`)}${fo(265,y-7,76,32,math('W_U'))}${line(`M357 ${y+7} H422`)}`;
  for(let j=0;j<8;j++)s+=rect(434+j*17,y,13,14,`rgba(0,127,115,${.13+((i*4+j*5)%8)*.065})`,'none',2);
  s+=fo(574,y-7,34,30,math('\\cdots'))+'</g>';
 }
 s+=fo(357,424,250,26,'LOGITS → ROW-WISE SOFTMAX','tiny');
 if(last){
  s+=line('M501 390 V489','teal');
  s+=fo(259,493,350,44,math('p_4=\\Pr(t_5=\\cdot\\mid t_{1:4})'),'teal');
  s+=fo(275,535,315,22,'THE NEXT TOKEN AFTER THE GIVEN SEQUENCE','tiny');
 }else{
  s+=fo(45,481,530,35,math('p_i=\\Pr(t_{i+1}=\\cdot\\mid t_{1:i})'));
  s+=fo(60,523,500,23,'FOUR POSITIONS → FOUR NEXT-TOKEN DISTRIBUTIONS','tiny');
 }
 return s+rowMotion('vocab')+'</svg>';
}

function detailHtml(){
 switch(current){
 case 0:return `<span class="detail-math">${math('X_0=SW_E+P\\in\\mathbb R^{T\\times d}')}</span><p>The blue grid is the whole ${math('X_0')} matrix: four token rows and six feature columns.</p>`;
 case 1:return `<span class="detail-math">${math('X_1=X_0+h_1+h_2')}</span><p>Blue copies of ${math('X_0')} enter both heads. Teal ${math('h_1')} and ${math('h_2')} add to the unchanged blue ${math('X_0')}.</p>`;
 case 2:return `<span class="detail-math">${math('X_2=X_1+\\operatorname{MLP}(X_1)')}</span><p>The same ordinary MLP processes every token row. It mixes features within a row, never between tokens.</p>`;
 case 3:return `<span class="detail-math">${math(`\\operatorname{MLP}(X_{1,${selectedRow+1},:})`)}</span><p>Only row ${selectedRow+1}, “${tokens[selectedRow]}”, enters this MLP evaluation. Attention has already supplied its context.</p>`;
 case 4:return `<span class="detail-math">${math('T\\times d\\ \\longrightarrow\\ 1\\times d\\ \\longrightarrow\\ 1')}</span><p>The rows now converge for the readout. Mean pooling is separate from the MLP.</p>`;
 case 5:return `<span class="detail-math">${math(`i=${selectedRow+1}:\\quad k\\in\\{${Array.from({length:selectedRow+1},(_,i)=>i+1).join(',')}\\}`)}</span><p>Receiving position ${selectedRow+1} can read ${selectedRow===0?'itself':`positions 1–${selectedRow+1}`}. Shading shows availability, not attention weights.</p>`;
 case 6:return `<span class="detail-math">${math('T\\times d\\ \\longrightarrow\\ T\\times V')}</span><p>Each row becomes vocabulary logits, then a distribution after a row-wise softmax. There is no pooling.</p>`;
 case 7:return `<span class="detail-math">${math('p_T=\\operatorname{softmax}(X_{2,T,:}W_U)')}</span><p>The final row supplies the distribution for the token after the given sequence.</p>`;
 }
}

/* Every moving payload has a fixed path. Its position is a pure function of scroll. */
function travel(path, id, color='teal', kind='plain', row=0){
 const hex=color==='rust'?'#A54F32':color==='blue'?'#234E70':'#007F73';
 return `<path d="${path}" data-motion-path="${id}" fill="none" stroke="${hex}" stroke-width="2.4" opacity=".8"/>`+
 `<g data-traveler="${id}" data-kind="${kind}" data-row="${row}" aria-hidden="true">${Array.from({length:10},(_,j)=>`<rect data-feature="${j}" x="${(j-4.5)*6}" y="-4" width="5" height="8" rx="1" fill="${hex}" stroke="white" stroke-width=".5"/>`).join('')}</g>`;
}
function matrixTravel(path,id,label,color='blue',kind='matrix',labelPosition='right'){
 const hex=color==='teal'?'#007F73':'#234E70';
 let glyph=rect(-21,-16,42,31,'white',hex,3);
 for(let i=0;i<4;i++)for(let j=0;j<6;j++)glyph+=rect(-17+j*6,-11+i*6,5,5,hex,'none',.7);
 glyph+=fo(labelPosition==='left'?-78:labelPosition==='below'?-27:24,labelPosition==='below'?18:-15,54,30,math(label),'matrix-label '+(color==='teal'?'teal':''));
 return `<path d="${path}" data-motion-path="${id}" fill="none" stroke="${hex}" stroke-width="2.4" opacity=".8"/>`+
 `<g data-traveler="${id}" data-kind="${kind}" data-row="0" aria-hidden="true">${glyph}</g>`;
}
function mainMotion(){
 if(current===0)return matrixTravel('M460 451 V393','embedding','X_0');
 return matrixTravel('M460 454 H130 V433','head-one-input','X_0','blue','matrix-input')+
 matrixTravel('M460 454 H305 V433','head-two-input','X_0','blue','matrix-input')+
 matrixTravel('M130 347 V340 H220','head-one-output','h_1','teal','matrix-output')+
 matrixTravel('M305 347 V340 H365','head-two-output','h_2','teal','matrix-output')+
 matrixTravel('M460 454 V386','residual','X_0');
}
function rowMotion(kind){
 let s='';
 for(let i=0;i<4;i++){
  const mlp=kind==='mlp',y=mlp?191+i*54:205+i*57;
  s+=travel(`M${mlp?154:137} ${y} H${mlp?502:502}`,`${kind}-${i}`,mlp?'rust':'teal',kind,i);
 }
 return s;
}
function poolMotion(){
 return Array.from({length:4},(_,i)=>travel(`M224 ${189+i*48} C290 ${189+i*48}, 285 262, 366 262`,`pool-${i}`,'teal','pool',i)).join('');
}
function miniLine(d,color='var(--blue)'){return `<path d="${d}" fill="none" stroke="${color}" stroke-width="1.7"/>`;}
function mobileDiagram(){
 let s=`<div class="mobile-route"><svg class="compact-diagram" viewBox="0 0 600 235" role="img" aria-label="${escapeHtml(descriptions[current])}" xmlns="http://www.w3.org/2000/svg">`;
 if(current===0){
  s+=fo(120,0,400,35,math('X_0=SW_E+P'));
  s+=box(22,66,154,142,tokens.join('<br/>'))+miniLine('M176 136 H490');
  s+=fo(211,66,333,30,math('X_0\\in\\mathbb R^{4\\times6}'));
  s+=matrixTravel('M221 136 H455','m-embedding','X_0');
  s+=fo(203,185,363,30,'Four token rows · six feature columns','small');
 }else if(current===1){
  s+=box(15,86,98,44,math('X_0'))+box(490,86,98,44,math('X_1'));
  s+=miniLine('M113 108 H154 V44 H231')+miniLine('M154 108 V166 H231');
  s+=miniLine('M364 44 H465 V97','var(--teal)')+miniLine('M364 166 H465 V119','var(--teal)');
  s+=miniLine('M113 108 V218 H465 V119')+miniLine('M476 108 H490')+add(465,108,'var(--teal)');
  s+=box(231,23,133,42,'Head 1','teal')+box(231,145,133,42,'Head 2','teal');
  s+=fo(195,85,235,35,math('X_1=X_0+h_1+h_2'));
  s+=matrixTravel('M113 108 H154 V44 H198','m-head-one-input','X_0','blue','matrix-input','below')+matrixTravel('M113 108 H154 V166 H198','m-head-two-input','X_0','blue','matrix-input','below');
  s+=matrixTravel('M389 44 H430','m-head-one-output','h_1','teal','matrix-output','below')+matrixTravel('M389 166 H430','m-head-two-output','h_2','teal','matrix-output','below')+matrixTravel('M113 108 V218 H340','m-residual','X_0','blue','matrix','left');
 }else if(current===2||current===3){
  s+=fo(95,0,135,30,math('X_1'))+fo(395,0,173,30,math('\\operatorname{MLP}(X_1)'),'rust');
  s+=rect(268,37,84,177,'var(--rust-light)','var(--rust)');
  for(let i=0;i<4;i++){
   const y=49+i*43,active=current===2||i===selectedRow;
   s+=`<g opacity="${active?1:.23}">${fo(4,y-5,100,28,tokens[i],'small left')}${vector(114,y,6,10,4,'teal',true,i)}${miniLine(`M200 ${y+5} H263`,'var(--rust)')}${fo(271,y-8,78,32,math('\\operatorname{MLP}'),'rust')}${miniLine(`M357 ${y+5} H417`,'var(--rust)')}${vector(425,y,6,10,4,'rust',true,i+1)}</g>`;
   s+=travel(`M153 ${y+5} H465`,`m-mlp-${i}`,'rust','mlp',i);
  }
  s+=fo(150,211,300,24,'Same MLP at all four positions','small');
 }else if(current===4){
  s+=fo(80,0,160,28,math('X_2'))+fo(351,0,160,28,math('r'));
  for(let i=0;i<4;i++){
   const y=43+i*43;s+=vector(95,y,6,12,4,'teal',true,i)+miniLine(`M190 ${y+6} C275 ${y+6}, 280 109, 350 109`,'var(--teal)')+travel(`M190 ${y+6} C275 ${y+6}, 280 109, 350 109`,`m-pool-${i}`,'teal','pool',i);
  }
  s+=vector(364,102,6,14,5,'teal',true,2)+fo(350,145,190,34,math('rw+b=\\hat y'))+fo(65,211,470,24,'Mean pooling combines the four rows','small');
 }else if(current===5){
  for(let h=0;h<2;h++){
   const x=90+h*285;
   s+=fo(x-8,0,132,32,math(`A^{(${h+1})}`),'teal');
   for(let i=0;i<4;i++){
    s+=fo(x-28,45+i*36,24,30,String(i+1),'small');
    for(let k=0;k<4;k++)s+=rect(x+k*36,46+i*36,31,31,k<=i?(i===selectedRow?'#80bfb9':'#e0efed'):'#f1f1f1','none',2)+(k>i?fo(x+k*36,46+i*36,31,31,math('0'),'small'):'');
    if(i===selectedRow)s+=rect(x-3,43+i*36,145,37,'none','var(--teal)',3);
   }
  }
  s+=fo(110,201,390,30,math(`i=${selectedRow+1}:\\quad k\\leq ${selectedRow+1}`),'teal');
 }else{
  s+=fo(104,0,120,30,math('X_2'))+fo(258,0,90,30,math('W_U'))+fo(419,0,160,30,math('p_i'));
  s+=rect(268,37,84,177,'var(--teal-light)','var(--teal)');
  for(let i=0;i<4;i++){
   const y=49+i*43,active=current===6||i===3;
   s+=`<g opacity="${active?1:.18}">${fo(4,y-5,100,28,tokens[i],'small left')}${vector(114,y,6,10,4,'teal',true,i)}${miniLine(`M200 ${y+5} H263`)}${fo(271,y-8,78,32,math('W_U'))}${miniLine(`M357 ${y+5} H417`)}${vector(425,y,8,10,4,'teal',true,i+1)}</g>`;
   s+=travel(`M153 ${y+5} H465`,`m-vocab-${i}`,'teal','vocab',i);
  }
  s+=fo(130,211,400,24,current===7?'Last row → next token':'Unembedding, then a row-wise softmax','small');
 }
 return s+'</svg></div>';
}

let motionItems=[];
function paintDiagram(){
 const draw=current<2?mainMap:current<4?mlpMap:current===4?poolMap:current===5?maskMap:vocabMap;
 diagram.innerHTML=draw()+mobileDiagram();
 detail.innerHTML=detailHtml();
 renderMath(diagram);renderMath(detail);
 motionItems=[...diagram.querySelectorAll('[data-motion-path]')].map(path=>{
  const traveler=diagram.querySelector(`[data-traveler="${path.dataset.motionPath}"]`);
  const length=path.getTotalLength();
  path.style.strokeDasharray=String(length);
  return {path,traveler,length,kind:traveler.dataset.kind,row:Number(traveler.dataset.row),features:[...traveler.querySelectorAll('[data-feature]')]};
 });
}
function setStep(index){
 if(index===current)return false;current=index;
 document.getElementById('stage-mode').textContent=current<5?'ENCODER':'DECODER';
 document.getElementById('stage-count').textContent=`${String(current+1).padStart(2,'0')} / 08`;
 document.getElementById('stage-title').textContent=titles[current];
 document.querySelectorAll('.site-header nav a').forEach(a=>a.classList.toggle('active',a.hash===(current<5?'#encoder':'#decoder')));
 document.getElementById('stage').dataset.step=String(current);
 paintDiagram();
 return true;
}
const clamp=(x,a=0,b=1)=>Math.max(a,Math.min(b,x));
function animateScene(progress){
 document.getElementById('stage').dataset.progress=progress.toFixed(3);
 document.getElementById('stage').dataset.motion=reducedMotion.matches?'reduced':'continuous';
 const p=clamp((progress-.08)/.82);
 for(const item of motionItems){
  let phase=p,visible=true;
  if(item.kind==='matrix-input'){
   phase=clamp(p/.44);
   visible=p<.48&&!reducedMotion.matches;
  }
  if(item.kind==='matrix-output'){
   phase=clamp((p-.56)/.44);
   visible=p>=.56||reducedMotion.matches;
  }
  if(current===3&&item.kind==='mlp'){
   visible=item.row===selectedRow;
   phase=clamp((progress*4-selectedRow-.08)/.84);
  }
  if(current===7&&item.kind==='vocab')visible=item.row===3;
  if(reducedMotion.matches)phase=1;
  const point=item.path.getPointAtLength(phase*item.length);
  item.traveler.setAttribute('transform',`translate(${point.x.toFixed(2)} ${point.y.toFixed(2)})`);
  item.traveler.style.opacity=visible?(item.kind==='pool'&&phase>.88?String((1-phase)/.12):'1'):'0';
  item.path.style.opacity=visible||item.kind==='matrix-input'?'.85':'0';
  item.path.style.strokeDashoffset=String(item.length*(1-phase));
  let count=6;
  if(item.kind==='mlp'&&phase>.32&&phase<.68)count=10;
  if(item.kind==='vocab'&&phase>.58)count=8;
  item.features.forEach((f,j)=>{f.style.display=j<count?'':'none';f.setAttribute('x',String((j-(count-1)/2)*6));});
 }
}
let ticking=false;
function updateScroll(){
 ticking=false;
 const mobile=matchMedia('(max-width: 800px)').matches;
 const readingLine=innerHeight*(mobile?.72:.52);
 const rects=steps.map(s=>s.getBoundingClientRect());
 const anchors=rects.map(r=>r.top+Math.min(r.height*.35,280));
 let nearest=0,distance=Infinity;
 anchors.forEach((anchor,i)=>{const d=Math.abs(anchor-readingLine);if(d<distance){distance=d;nearest=i;}});
 const start=nearest===0?anchors[0]-rects[0].height/2:(anchors[nearest-1]+anchors[nearest])/2;
 const end=nearest===steps.length-1?anchors[nearest]+rects[nearest].height/2:(anchors[nearest]+anchors[nearest+1])/2;
 sceneProgress=clamp((readingLine-start)/(end-start));
 const row=Math.min(3,Math.floor(sceneProgress*4));
 const rowChanged=row!==selectedRow;
 selectedRow=row;
 const changed=setStep(nearest);
 if(!changed&&rowChanged&&(current===3||current===5))paintDiagram();
 animateScene(sceneProgress);
 const max=document.documentElement.scrollHeight-innerHeight;
 document.getElementById('read-progress').style.width=`${clamp(scrollY/max)*100}%`;
}
addEventListener('scroll',()=>{if(!ticking){ticking=true;requestAnimationFrame(updateScroll);}},{passive:true});
addEventListener('resize',updateScroll);
reducedMotion.addEventListener('change',updateScroll);
renderMath();setStep(0);updateScroll();
document.fonts.ready.then(updateScroll);
