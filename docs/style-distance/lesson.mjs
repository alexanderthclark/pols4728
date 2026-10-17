import { passages, triplet, safePosition, stateAt, clamp, mix, projectVector } from './model.mjs';

const C = { ink: '#242424', blue: '#234E70', rust: '#A54F32', teal: '#007F73',
  muted: '#606060', rule: '#767676', paper: '#FFFFFF' };
const FONT = 'Georgia, "Times New Roman", serif';
const esc = value => String(value).replace(/[&<>"']/g, character =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
const n = value => Number(value).toFixed(2);
const smooth = value => { const t = clamp(value); return t * t * (3 - 2 * t); };
const text = (x, y, value, attributes = '') => `<text x="${n(x)}" y="${n(y)}" fill="${C.ink}" font-family='${FONT}' ${/font-size=/.test(attributes)?'':'font-size="22"'} ${attributes}>${esc(value)}</text>`;
const line = (x1, y1, x2, y2, attributes = '') => `<line x1="${n(x1)}" y1="${n(y1)}" x2="${n(x2)}" y2="${n(y2)}" stroke="${C.rule}" stroke-width="2" ${attributes}/>`;
const rect = (x, y, width, height, attributes = '') => `<rect x="${n(x)}" y="${n(y)}" width="${n(width)}" height="${n(height)}" fill="white" stroke="${C.rule}" stroke-width="1.4" rx="5" ${attributes}/>`;

function lines(x, y, values, attributes = '', leading = 28) {
  return values.map((value, index) => text(x, y + index * leading, value, attributes)).join('');
}

function wrap(value, maximum = 32) {
  const words = value.split(/\s+/), output = [];
  let current = '';
  for (const word of words) {
    if (current && `${current} ${word}`.length > maximum) {
      output.push(current); current = word;
    } else current = current ? `${current} ${word}` : word;
  }
  if (current) output.push(current);
  return output;
}

function point(passage, x, y, { opacity = 1, size = 11, stroke = C.ink, ring = false } = {}) {
  const fill = passage.style === 'biblical' ? C.blue : C.rust;
  const mark = passage.shape === 'circle'
    ? `<circle cx="${n(x)}" cy="${n(y)}" r="${size}" fill="${fill}" stroke="${stroke}" stroke-width="2"/>`
    : `<rect x="${n(x-size)}" y="${n(y-size)}" width="${size*2}" height="${size*2}" fill="${fill}" stroke="${stroke}" stroke-width="2"/>`;
  return `<g opacity="${n(opacity)}">${ring ? `<circle cx="${n(x)}" cy="${n(y)}" r="${size+8}" fill="none" stroke="${C.ink}" stroke-width="2"/>` : ''}${mark}</g>`;
}

function arrow(x1, y1, x2, y2, color = C.ink, attributes = '') {
  const angle = Math.atan2(y2-y1, x2-x1), length = 10;
  const x3 = x2 - length * Math.cos(angle - 0.45), y3 = y2 - length * Math.sin(angle - 0.45);
  const x4 = x2 - length * Math.cos(angle + 0.45), y4 = y2 - length * Math.sin(angle + 0.45);
  return line(x1, y1, x2, y2, `style="stroke:${color}" ${attributes}`)
    + `<path d="M ${n(x3)} ${n(y3)} L ${n(x2)} ${n(y2)} L ${n(x4)} ${n(y4)}" fill="none" stroke="${color}" stroke-width="2"/>`;
}

function plot(position, compact) {
  const state = stateAt(position), scene = Math.round(position);
  const box = compact ? { x: 55, y: 156, width: 390, height: 204 } : { x: 465, y: 135, width: 485, height: 300 };
  const project = vector => projectVector(vector, box);
  const xy = Object.fromEntries(passages.map(p => [p.id, project(state.vectors[p.id])]));
  const coordinates = box.x + box.width/2;
  let result = `<g class="sd-plot" data-training-progress="${n(state.progress)}">`;
  result += text(box.x + box.width/2, box.y - 25, 'Schematic coordinates · fixed axes', 'text-anchor="middle" font-size="20"');
  result += line(box.x, box.y+box.height, box.x+box.width, box.y+box.height);
  result += line(box.x, box.y, box.x, box.y+box.height);
  for (const tick of [-1,0,1]) {
    const [x] = project([tick,0]), [,y] = project([0,tick]);
    result += line(x,box.y+box.height,x,box.y+box.height+6);
    result += text(x,box.y+box.height+28,String(tick),'text-anchor="middle" font-size="20"');
    result += line(box.x-6,y,box.x,y);
    result += text(box.x-12,y+7,String(tick),'text-anchor="end" font-size="20"');
  }
  result += text(coordinates,box.y+box.height+57,'Coordinate 1','text-anchor="middle" font-size="20"');
  result += text(box.x,box.y-54,'Coordinate 2','font-size="20"');
  // Dashed outlines identify the original content pairs. Opacity fades as the
  // story moves to the triplet objective; position never changes the axes.
  const contentOutline = (1 - smooth((position-3)/1.3)) * smooth(position/1.2);
  for (const [left,right] of [['A','B'],['C','D']]) {
    const minimumY = Math.min(xy[left][1],xy[right][1]), height = Math.abs(xy[left][1]-xy[right][1]);
    result += `<rect x="${n(xy[left][0]-25)}" y="${n(minimumY-23)}" width="50" height="${n(height+46)}" fill="none" stroke="${C.rule}" stroke-width="1.5" stroke-dasharray="5 5" opacity="${n(contentOutline)}" rx="8"/>`;
  }
  const edgeOpacity = smooth((position - 3.2)/0.8) * (1 - smooth((position-9.2)/0.8));
  result += `<g opacity="${n(edgeOpacity)}">`;
  result += line(...xy.A,...xy.C,`data-edge="A-C" style="stroke:${C.blue};stroke-width:3"`);
  result += line(...xy.A,...xy.B,`data-edge="A-B" style="stroke:${C.rust};stroke-width:3" stroke-dasharray="7 5"`);
  if (scene >= 5 && scene <= 9) {
    const positiveMid = [(xy.A[0]+xy.C[0])/2, (xy.A[1]+xy.C[1])/2];
    const negativeMid = [(xy.A[0]+xy.B[0])/2, (xy.A[1]+xy.B[1])/2];
    result += text(positiveMid[0],positiveMid[1]+36,`d²(A,C) = ${n(state.positiveDistance)}`,'text-anchor="middle" font-size="20"');
    result += text(negativeMid[0]+22,negativeMid[1]-18,`d²(A,B) = ${n(state.negativeDistance)}`,'font-size="20"');
  }
  result += '</g>';
  for (const p of passages) {
    const [x,y] = xy[p.id];
    const dimmed = scene>=4 && scene<=8 && p.id==='D';
    result += point(p,x,y,{opacity:dimmed?.35:1,ring:p.id===triplet.anchor && scene>=4 && scene<=9});
    const labelX = x + (p.contentCode < 0 ? -21 : 21);
    result += text(labelX,y+7,p.id,`text-anchor="${p.contentCode<0?'end':'start'}" font-size="22" opacity="${dimmed?.45:1}"`);
  }
  if (!compact) {
    result += text(box.x,520,'Blue: biblical cadence · rust: academic','font-size="20"');
    result += text(box.x,548,'Circle: no guarantee · square: speed wins','font-size="20"');
  } else {
    result += text(30,445,'Blue: biblical · rust: academic','font-size="20"');
    result += text(30,473,'Circle: no guarantee · square: speed wins','font-size="20"');
  }
  return result + '</g>';
}

function passageCard(passage,x,y,width, role, compact=false) {
  const content = wrap(compact ? passage.short : passage.text,31);
  const height = 42 + content.length*27;
  const labels = {A:'Ecclesiastes · biblical cadence',B:'Orwell’s parody · excerpt',C:'Constructed counterexample',D:'New academic paraphrase'};
  const label = role ? `${role} · ${passage.id}` : `${passage.id} · ${labels[passage.id]}`;
  const result = rect(x,y,width,height)
    + text(x+17,y+29,label,'font-size="21"')
    + lines(x+17,y+57,content,'font-size="20"',27);
  return { result, height };
}

function mobileExamples() {
  let output=text(250,30,'Two claims, two writing styles','text-anchor="middle" font-size="23"');
  let y=50;
  for(const p of passages) {
    const body=wrap(p.text,43),height=40+body.length*27;
    output+=rect(20,y,460,height)+text(36,y+27,`${p.id} · ${{A:'Ecclesiastes',B:'Orwell’s parody · excerpt',C:'Constructed counterexample',D:'New teaching paraphrase'}[p.id]}`,'font-size="21"')
      +lines(36,y+54,body,'font-size="20"',27);
    y+=height+13;
  }
  return output;
}

function sharedEncoder(position,compact) {
  const rows = [passages[0],passages[2],passages[1]];
  const labels = ['Anchor','Positive','Negative'];
  if (compact) {
    let output = text(250,57,'Three inputs · one set of weights','text-anchor="middle" font-size="23"');
    rows.forEach((p,i) => {
      const y = 100+i*118;
      output += rect(20,y,170,66)+text(35,y+29,`${labels[i]} ${p.id}`,'font-size="21"')
        +text(35,y+55,p.style==='biblical'?'Biblical cadence':'Academic parody','font-size="20"');
      output += arrow(195,y+34,225,y+34);
      output += rect(228,y+3,125,60)+text(290,y+30,'Encoder','text-anchor="middle" font-size="21"')
        +text(290,y+54,'fθ','text-anchor="middle" font-size="22"');
      output += arrow(358,y+34,404,y+34);
      output += point(p,430,y+34,{size:12})+text(450,y+41,p.id,'font-size="21"');
    });
    output += line(290,163,290,221,'stroke-dasharray="5 4"');
    output += line(290,281,290,339,'stroke-dasharray="5 4"');
    return output + text(250,468,'Dashed link: the weights are shared.','text-anchor="middle" font-size="20"');
  }
  let output = text(500,55,'Three passages enter the same encoder','text-anchor="middle" font-size="26"');
  rows.forEach((p,i) => {
    const y=104+i*126;
    const card=passageCard(p,35,y,375,labels[i],true);
    output += card.result;
    output += arrow(422,y+45,490,y+45);
    output += rect(500,y+7,210,78)+text(605,y+42,'Encoder fθ','text-anchor="middle" font-size="26"')
      +text(605,y+69,'shared weights','text-anchor="middle" font-size="21"');
    output += arrow(722,y+45,827,y+45);
    output += point(p,865,y+45,{size:15})+text(898,y+53,`z${p.id}`,'font-size="24"');
  });
  output += line(605,189,605,237,'stroke-dasharray="5 4"');
  output += line(605,315,605,363,'stroke-dasharray="5 4"');
  return output + text(500,532,'The model is reused; three independent models are not trained.','text-anchor="middle" font-size="22"');
}

function updateDiagram(position,compact) {
  if(compact) {
    let output = text(250,58,'Loss changes the shared encoder','text-anchor="middle" font-size="23"');
    output += rect(90,90,320,72)+text(250,122,'Triplet loss L','text-anchor="middle" font-size="24"')
      +text(250,150,'from the three embeddings','text-anchor="middle" font-size="20"');
    output += arrow(250,174,250,214);
    output += rect(90,226,320,72)+text(250,259,'Gradient ∇θL','text-anchor="middle" font-size="24"')
      +text(250,286,'which parameter changes help?','text-anchor="middle" font-size="20"');
    output += arrow(250,310,250,350);
    output += rect(90,361,320,67)+text(250,401,'θ → updated θ','text-anchor="middle" font-size="25"');
    return output+text(250,467,'Next: recompute every embedding.','text-anchor="middle" font-size="21"');
  }
  let output = text(500,65,'The trainable parameters receive the update','text-anchor="middle" font-size="26"');
  const items=[{x:35,title:'Triplet loss',bottom:'L(a,p,n)'},{x:365,title:'Gradient',bottom:'∇θL'},{x:695,title:'Update',bottom:'θ → updated θ'}];
  for(const item of items) output += rect(item.x,170,270,130)
    +text(item.x+135,216,item.title,'text-anchor="middle" font-size="26"')
    +text(item.x+135,269,item.bottom,'text-anchor="middle" font-size="28"');
  output += arrow(315,235,355,235)+arrow(645,235,685,235);
  const bead=mix(320,680,smooth(position-6.5));
  output += `<circle cx="${n(bead)}" cy="235" r="6" fill="${C.rust}"/>`;
  output += text(500,365,'Published StyleDistance: pretrained RoBERTa + LoRA','text-anchor="middle" font-size="24"');
  output += text(500,406,'The base weights are frozen; the low-rank updates are trained.','text-anchor="middle" font-size="22"');
  return output+text(500,486,'New parameters change the function fθ(text), and therefore its outputs.','text-anchor="middle" font-size="22"');
}

function panel(position,compact) {
  const scene=Math.round(position), state=stateAt(position);
  if (compact) {
    const top=[
      ['What should be close?', 'Similarity is a modeling choice.'],
      ['Two claims × two writing styles', 'A–B: related ideas · A–C: similar style'],
      ['Meaning: A is near B', 'A–B: related claims · A–C: opposite claims'],
      [],
      ['Anchor A · positive C · negative B', 'C reverses A’s claim in a similar style.'],
      ['Compare squared Euclidean distances', 'Solid: positive · dashed: negative'],
      [`L = max(0, ${n(state.positiveDistance)} − ${n(state.negativeDistance)} + 0.1)`, `Illustrative loss = ${n(state.loss)}`],
      [],
      ['Re-encode after a parameter change', 'All four embeddings respond.'],
      ['Style: A is now near C', 'Close in style does not mean agreement.'],
      ['Evaluate on new, labeled passages', 'This map is not a test result.'],
      ['Style similarity serves one purpose', 'Content and context still matter.'],
    ][scene];
    return lines(250,40,top,'text-anchor="middle" font-size="21"',29);
  }
  let output='';
  if (scene===1) {
    let y=42;
    passages.forEach(p=>{
      const card=passageCard(p,30,y,380);
      output+=card.result;
      y+=card.height+16;
    });
    return output;
  }
  if(scene===4 || scene===5) {
    const rows=[[passages[0],'Anchor'],[passages[2],'Positive: similar style'],[passages[1],'Negative: related meaning']];
    rows.forEach(([p,role],i)=>{output+=passageCard(p,30,80+i*135,380,role,true).result;});
    return output;
  }
  const panels={
    0:{heading:['Similar meaning,','different style.'],body:['A: “the race is not to the swift”','','B: Orwell’s academic parody','expresses a related idea.','','C: “the swift shall win the race”','reverses A’s claim, yet sounds similar.','','Which pair should be close?']},
    2:{heading:['Meaning: A is near B.'],body:['A and B question whether ability','ensures success; C and D assert it.','','This starting map privileges','the claim each text expresses.','','d(A,B) < d(A,C)','','These are not measured vectors.']},
    6:{heading:['The positive should be closer.'],body:['Require its squared distance','to be smaller by a margin α.','','Here α = 0.1.','The loss is positive when that','requirement is not yet satisfied.']},
    8:{heading:['Recompute the embeddings.'],body:['A shared parameter change affects','all four passages, including D,','which was absent from this triplet.','','The points move because the','encoding function has changed.']},
    9:{heading:['Style: A is now near C.'],body:['A and C share a biblical cadence,','despite making opposite claims.','','d(A,C) < d(A,B)','','Repeat with varied topics and styles.','One triplet cannot establish that','meaning and style are disentangled.']},
    10:{heading:['Hold out new passages.'],body:['Example: school opening hours,','written plainly and bureaucratically.','','Encode them after training.','Measure whether labeled positives','rank closer than labeled negatives.','','Do not report this toy map as evidence.']},
    11:{heading:['Closeness is not agreement.'],body:['A and C can be stylistic neighbors','while contradicting one another.','','A and B remain semantically related','even when far apart in style space.','','The task determines which','relationships we want to preserve.']},
  };
  const item=panels[scene]||panels[0];
  output+=lines(30,84,item.heading,'font-size="27"',34);
  output+=lines(30,84+item.heading.length*34+34,item.body,'font-size="22"',29);
  if(scene===6) {
    output+=rect(30,392,375,105)+text(47,428,`d²(a,p) = ${n(state.positiveDistance)}`,'font-size="22"')
      +text(47,459,`d²(a,n) = ${n(state.negativeDistance)}`,'font-size="22"')
      +text(47,489,`Triplet loss = ${n(state.loss)}`,'font-size="22"');
  }
  return output;
}

const scenes = [
  { id:'purpose',title:'Meaning is not the only similarity',
    body:'“The race is not to the swift” and Orwell’s academic rendering of Ecclesiastes express related ideas in very different styles. “The swift shall win the race” reverses the claim while preserving much of the verse’s vocabulary and cadence. Which relationship should an embedding preserve?' },
  { id:'content-style',title:'Separate the claim from the style',
    body:'All four passages concern ability and success. A and B express related doubts about ability ensuring victory; C and D assert that speed brings success. A and C use a compressed biblical cadence; B and D use abstract academic language.',
    reading:'<p>A, from Ecclesiastes 9:11 (KJV): “the race is not to the swift”. B, an excerpt from Orwell’s academic parody: “' + esc(passages[1].text) + '”. C, a constructed counterexample: “the swift shall win the race”. D, a new academic paraphrase for this lesson: “' + esc(passages[3].text) + '”</p><p>A and B convey related ideas, rather than exactly equivalent meanings. C is not scripture, and D is not an Orwell quotation. Color identifies register; shape identifies the claim. Each text keeps its letter.</p>',
    note:'Orwell presents his rendering in Politics and the English Language and acknowledges that it loses detail. Only a short excerpt appears here. These four passages are not the StyleDistance training data.' },
  { id:'pretrained',title:'Begin with semantic neighbors',
    body:'Suppose a starting representation places A near B because their claims are semantically related, and C near D for the same reason. That geometry can serve meaning-based retrieval. For a style task, however, we want A nearer C, even though C reverses A’s claim.',
    note:'This is a conceptual starting geometry. No claim is made that an actual pretrained encoder yields these coordinates or treats negation this way.' },
  { id:'shared-encoder',title:'Use one encoder three times',
    body:'Use A as the anchor, C as the style-positive, and B as the style-negative. Each passage passes through the same encoder with shared parameters θ. The resulting three vectors enter one training objective.',
    equation:'<span>z<sub>a</sub> = f<sub>θ</sub>(A), &nbsp; z<sub>p</sub> = f<sub>θ</sub>(C), &nbsp; z<sub>n</sub> = f<sub>θ</sub>(B)</span>' },
  { id:'triplet',title:'Choose the desired neighbor',
    body:'The positive C resembles A in style while contradicting its claim. The negative B conveys a related idea in an academic parody. “Positive” and “negative” identify the desired stylistic ranking; they say nothing about truth, quality, or agreement.',
    note:'This illustrative triplet encodes a style preference. In the StyleDistance paper, a style-changing negative can paraphrase the anchor or the positive; a negative need not always share the anchor’s content.' },
  { id:'distances',title:'Compare the two distances',
    body:'The solid line connects A to C, the desired style neighbor. The dashed line connects A to B, the semantically related text in a different style. Compare their squared Euclidean distances using both coordinate differences.',
    equation:'<span>d²(z<sub>i</sub>, z<sub>j</sub>) = Σ<sub>k</sub>(z<sub>ik</sub> − z<sub>jk</sub>)²</span>' },
  { id:'loss',title:'A margin defines the target',
    body:'Triplet loss asks C to be closer to A than B is by at least a margin α. The loss reaches zero once that inequality is satisfied. It neither demands identical embeddings for A and C nor requires an unlimited increase in the A–B distance.',
    equation:'<span>L = max{0, d²(A,C) − d²(A,B) + α}, &nbsp; α = 0.1</span>',
    note:'This is the squared-distance triplet hinge objective described in the StyleDistance paper. The numerical distances here come from our schematic coordinates.' },
  { id:'update',title:'Change the encoder, not the points',
    body:'Differentiate the loss through the shared encoder and update its trainable parameters. The published StyleDistance model adapts pretrained RoBERTa with LoRA, so the small low-rank updates are trained while the base weights remain frozen.',
    equation:'<span>θ<sub>trainable</sub> ← θ<sub>trainable</sub> − η ∇<sub>θtrainable</sub>L</span>',
    note:'The published method freezes the base model and trains LoRA updates; the instructor’s linked notebook is a separate, unverified implementation.' },
  { id:'reencode',title:'Re-encode all four passages',
    body:'Scroll to recompute the embeddings after the shared parameters change: A and C draw together while A and B separate. Even D, absent from this triplet, can move because it passes through the same encoder. Scroll back to restore the earlier geometry.',
    note:'This reversible motion illustrates the intended training preference. It is not an optimizer trace, and D’s movement alone does not establish its correct stylistic ranking.' },
  { id:'repeat',title:'Repeat across varied examples',
    body:'The target geometry makes A closer to C in style despite their opposite meanings. A useful model needs many comparisons across topics and registers, including examples that defeat shortcuts based on shared words or length. The published work combines synthetic style data with natural authorship data.',
    note:'The published main checkpoint and its synthetic-only variant use different data mixtures. The displayed geometry is a teaching illustration for neither checkpoint.' },
  { id:'heldout',title:'Test on new passages',
    body:'Reserve new passages and labeled comparisons before fitting the model, then compare rankings under the pretrained and adapted encoders. Include semantically related negatives, style-matched positives on unfamiliar topics, and examples with little shared vocabulary.',
    equation:'<span>Held-out triplet accuracy: proportion with d(z<sub>a</sub>, z<sub>p</sub>) &lt; d(z<sub>a</sub>, z<sub>n</sub>)</span>',
    note:'No held-out scores are supplied here. Any claim that fine-tuning helped requires actual encoder outputs and evaluation data.' },
  { id:'limits',title:'Closeness does not imply agreement',
    body:'A and B remain semantically related even when separated in style space. A and C can become stylistic neighbors while contradicting one another. Fine-tuning rewards the relationship chosen for the task; success on this example would not establish complete separation of style from meaning.' },
];

function render(position, {compact=false}={}) {
  const p=safePosition(position,scenes.length-1);
  const scene=Math.round(p);
  const modeFor=index=>index===3?'encoder':index===7?'update':compact&&index===1?'examples':'geometry';
  const mode=modeFor(scene);
  // Mutually exclusive diagrams fade through white. Drawing two complete
  // diagrams at partial opacity made text unreadable between their scenes.
  // The embedding coordinates remain continuous whenever the map is active.
  const before=scene>0&&modeFor(scene-1)!==mode?smooth((p-(scene-.5))/.18):1;
  const after=scene<scenes.length-1&&modeFor(scene+1)!==mode?smooth(((scene+.5)-p)/.18):1;
  const opacity=Math.min(before,after);
  const content=mode==='encoder'?sharedEncoder(p,compact):mode==='update'?updateDiagram(p,compact)
    :mode==='examples'?mobileExamples():panel(p,compact)+plot(p,compact);
  const width=compact?500:1000, height=compact?480:560;
  return `<g role="img" aria-label="${esc(scenes[Math.round(p)].title)}" data-lesson="style-distance" data-position="${n(p)}">
    <title>${esc(scenes[Math.round(p)].title)}</title>
    <desc>Schematic embedding geometry with four fixed passage identities. Color indicates writing style and shape indicates the claim. This is not measured model output.</desc>
    <rect width="${width}" height="${height}" fill="${C.paper}"/>
    <g data-diagram-mode="${mode}" opacity="${n(opacity)}">${content}</g>
  </g>`;
}

export const lesson = {
  id:'style-distance',
  title:'Fine-tuning for writing style',
  subtitle:'Teach an embedding model what “similar” means',
  intro:'Ecclesiastes and Orwell’s academic parody can convey related ideas while sounding very different. A sentence that reverses the verse can sound much more like it. Follow those texts through a schematic fine-tuning example that teaches an encoder to measure style.',
  scenes,
  takeaways:[
    'Semantically related passages can differ in style; stylistically similar passages can make opposite claims.',
    'A shared encoder processes every member of a triplet, and its parameter updates can change every text embedding.',
    'Triplet loss compares an anchor–positive distance with an anchor–negative distance and a margin.',
    'A schematic geometric improvement is not evidence of model performance; evaluate held-out texts and styles.',
  ],
  dataNote:'A quotes Ecclesiastes 9:11 (KJV); B is a short excerpt from Orwell’s academic parody in Politics and the English Language. C is a constructed counterexample, not scripture; D is a new teaching paraphrase, not Orwell. Every displayed coordinate, distance, and loss is calculated from a labeled two-dimensional schematic, not from a trained language model. The published StyleDistance methodology supplies the learning idea; the instructor’s linked training notebook has not been inspected.',
  sources:[
    {label:'George Orwell, Politics and the English Language',url:'https://www.orwellfoundation.com/the-orwell-foundation/orwell/essays-and-other-works/politics-and-the-english-language/',note:'Primary source for the Ecclesiastes comparison and academic parody. The excerpt preserves the Orwell Foundation’s wording; Orwell notes that the parody loses some detail.'},
    {label:'Ecclesiastes 9:11, King James Version',url:'https://www.biblegateway.com/passage/?search=Ecclesiastes+9%3A11&version=KJV',note:'Source of A. C deliberately reverses its claim and is not a biblical quotation.'},
    {label:'StyleDistance: Stronger Content-Independent Style Embeddings with Synthetic Parallel Examples',url:'https://aclanthology.org/2025.naacl-long.436/',note:'Primary paper: triplet training for style embeddings, including pretrained RoBERTa with LoRA and synthetic/natural data.'},
    {label:'StyleDistance model card',url:'https://huggingface.co/StyleDistance/styledistance',note:'Published checkpoint; using this model is distinct from running a fine-tuning experiment.'},
    {label:'Instructor’s “Our Own StyleDistance Model” notebook',url:'https://colab.research.google.com/drive/1-hAD4LPOl_ahdaOvfH-X2SCKD86e-XJj?usp=sharing',note:'Linked in the lecture slides. The notebook’s implementation and outputs have not been inspected; this lesson does not attribute a particular training run to it.'},
  ],
  render,
};

// Concise captions retain the argument on short phone screens.
const shortCaptions = [
  ['Meaning and style', 'A and B convey related ideas in different styles. C reverses A’s claim while sounding similar.'],
  ['Two claims, two styles', 'A and B question ability ensuring success; C and D affirm it. A and C share a biblical cadence.'],
  ['Semantic neighbors', 'This starting schematic puts A near B. For a style task, we want A near C despite their opposite claims.'],
  ['One shared encoder', 'Anchor A, positive C, and negative B pass through the same encoder. Three embeddings enter one loss.'],
  ['Choose a style neighbor', 'C is the style-positive even though it contradicts A. B is the style-negative despite its related meaning.'],
  ['Compare the distances', 'Solid A–C should be shorter than dashed A–B. These distances come from schematic coordinates.'],
  ['Require a margin', 'Reward C being closer to A than B is by the margin. Zero loss does not require identical vectors.'],
  ['Change shared parameters', 'StyleDistance trains low-rank updates to a pretrained encoder. Changing that shared function can change every text embedding.'],
  ['Watch the ranking change', 'A and C draw together as A and B separate. Even D, absent from this triplet, can move.'],
  ['Style without agreement', 'A is now near C in style. Many varied comparisons are needed to learn beyond shared vocabulary or length.'],
  ['Test on new texts', 'Compare rankings on unfamiliar passages and topics. Include style matches with little shared vocabulary.'],
  ['Closeness is not agreement', 'A–B remain semantically related; A–C make opposite claims. The useful geometry depends on the task.'],
];
shortCaptions.forEach(([title, body], index) => { lesson.scenes[index].shortTitle=title; lesson.scenes[index].shortBody=body; });
