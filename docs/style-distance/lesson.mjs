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
  const fill = passage.style === 'plain' ? C.blue : C.rust;
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
  result += line(...xy.B,...xy.D,`style="stroke:${C.rust};stroke-width:3"`);
  result += line(...xy.B,...xy.A,`style="stroke:${C.blue};stroke-width:3" stroke-dasharray="7 5"`);
  if (scene >= 5 && scene <= 9) {
    const positiveMid = [(xy.B[0]+xy.D[0])/2, (xy.B[1]+xy.D[1])/2];
    const negativeMid = [(xy.B[0]+xy.A[0])/2, (xy.B[1]+xy.A[1])/2];
    result += text(positiveMid[0],positiveMid[1]-20,`d²(a,p) = ${n(state.positiveDistance)}`,'text-anchor="middle" font-size="20"');
    result += text(negativeMid[0]+22,negativeMid[1]+7,`d²(a,n) = ${n(state.negativeDistance)}`,'font-size="20"');
  }
  result += '</g>';
  for (const p of passages) {
    const [x,y] = xy[p.id];
    const dimmed = scene>=4 && scene<=8 && p.id==='C';
    result += point(p,x,y,{opacity:dimmed?.35:1,ring:p.id===triplet.anchor && scene>=4 && scene<=9});
    const labelX = x + (p.content === 'budget' ? -21 : 21);
    result += text(labelX,y+7,p.id,`text-anchor="${p.content==='budget'?'end':'start'}" font-size="22" opacity="${dimmed?.45:1}"`);
  }
  if (!compact) {
    result += text(box.x,520,'Blue: plain · rust: bureaucratic','font-size="20"');
    result += text(box.x,548,'Circle: budget · square: bridge','font-size="20"');
  } else {
    result += text(30,445,'Blue: plain · rust: bureaucratic','font-size="20"');
    result += text(30,473,'Circle: budget · square: bridge','font-size="20"');
  }
  return result + '</g>';
}

function passageCard(passage,x,y,width, role, compact=false) {
  const content = compact ? [passage.short] : wrap(passage.text,31);
  const height = compact ? 68 : 42 + content.length*27;
  const label = role ? `${role} · ${passage.id}` : `${passage.id} · ${passage.style}`;
  const result = rect(x,y,width,height)
    + text(x+17,y+29,label,'font-size="21"')
    + lines(x+17,y+57,content,'font-size="20"',27);
  return { result, height };
}

function mobileExamples() {
  let output=text(250,30,'Two topics, two writing styles','text-anchor="middle" font-size="23"');
  let y=50;
  for(const p of passages) {
    const body=wrap(p.text,43),height=40+body.length*27;
    output+=rect(20,y,460,height)+text(36,y+27,`${p.id} · ${p.style}`,'font-size="21"')
      +lines(36,y+54,body,'font-size="20"',27);
    y+=height+13;
  }
  return output;
}

function sharedEncoder(position,compact) {
  const rows = [passages[1],passages[3],passages[0]];
  const labels = ['Anchor','Positive','Negative'];
  if (compact) {
    let output = text(250,57,'Three inputs · one set of weights','text-anchor="middle" font-size="23"');
    rows.forEach((p,i) => {
      const y = 100+i*118;
      output += rect(20,y,170,66)+text(35,y+29,`${labels[i]} ${p.id}`,'font-size="21"')
        +text(35,y+55,p.style==='plain'?'Plain style':'Bureaucratic','font-size="20"');
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
    const card=passageCard(p,35,y,375,labels[i]);
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
      ['Two topics × two writing styles', 'The four texts are teaching examples.'],
      ['Starting schematic: content dominates', 'A–B share content; C–D share content.'],
      [],
      ['Anchor B · positive D · negative A', 'Same style versus the same content.'],
      ['Compare squared Euclidean distances', 'Solid: positive · dashed: negative'],
      [`L = max(0, ${n(state.positiveDistance)} − ${n(state.negativeDistance)} + 0.1)`, `Illustrative loss = ${n(state.loss)}`],
      [],
      ['Re-encode after a parameter change', 'All four embeddings respond.'],
      ['Desired geometry for this task', 'Same style can cross topic boundaries.'],
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
    const rows=[[passages[1],'Anchor'],[passages[3],'Positive: same style'],[passages[0],'Negative: same content']];
    rows.forEach(([p,role],i)=>{output+=passageCard(p,30,80+i*135,380,role).result;});
    return output;
  }
  const panels={
    0:{heading:['Representations encode','a notion of similarity.'],body:['Word2vec learns from word context.','A contextual encoder represents','a passage using its surroundings.','','We can fine-tune that encoder','for a different similarity task.']},
    2:{heading:['A possible starting point'],body:['A and B discuss the budget.','C and D discuss the bridge.','','Our schematic first places texts','with shared content near each other.','','These are not measured vectors.']},
    6:{heading:['The positive should be closer.'],body:['Require its squared distance','to be smaller by a margin α.','','Here α = 0.1.','The loss is positive when that','requirement is not yet satisfied.']},
    8:{heading:['Recompute the embeddings.'],body:['A shared parameter change affects','all four passages, including C,','which was absent from this triplet.','','The points move because the','encoding function has changed.']},
    9:{heading:['One triplet is insufficient.'],body:['Use many topics, writing styles,','and anchor–positive–negative sets.','','The diagram shows the intended','change in similarity, not an','observed training trajectory.']},
    10:{heading:['Hold out new passages.'],body:['Example: school opening hours,','written plainly and bureaucratically.','','Encode them after training.','Measure whether labeled positives','rank closer than labeled negatives.','','Do not report this toy map as evidence.']},
    11:{heading:['Different goals produce','different useful geometries.'],body:['Texts can resemble one another','in style and differ in meaning.','','Check new topics, languages,','and unfamiliar writing styles.','A good style score does not prove','that content has been removed.']},
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
  { id:'purpose',title:'A vector has a purpose',
    body:'Word2vec learns word vectors through a prediction task, and transformers produce contextual representations of text. An existing encoder therefore brings a useful starting representation, but the similarity it expresses may differ from the similarity we want to measure.' },
  { id:'content-style',title:'Keep content and style separate',
    body:'These four teaching passages vary two things: what they discuss and how they are written. A budget sentence and a bridge sentence differ in content; a plain sentence and a bureaucratic paraphrase differ in style.',
    note:'The neutral passages are new teaching examples, not the StyleDistance training data. Circles identify the budget topic, squares identify the bridge topic, and each point retains its letter.' },
  { id:'pretrained',title:'Start from a pretrained encoder',
    body:'Suppose the starting representation primarily brings texts with similar content together. That can be useful for semantic search, but our task is to recognize writing style across different subjects.' },
  { id:'shared-encoder',title:'Use one encoder three times',
    body:'The anchor, positive, and negative passages each pass through the same encoder with shared parameters θ. This produces three vectors whose distances can enter one training objective.',
    equation:'<span>z<sub>a</sub> = f<sub>θ</sub>(a), &nbsp; z<sub>p</sub> = f<sub>θ</sub>(p), &nbsp; z<sub>n</sub> = f<sub>θ</sub>(n)</span>' },
  { id:'triplet',title:'Choose a triplet',
    body:'Use B as the anchor and D as a positive example: both are bureaucratic, although they discuss different things. Use A as the negative example: it expresses the anchor’s content in a different style, so content alone cannot solve this comparison.',
    note:'This is one illustrative triplet construction. In the paper, a style-changing negative can paraphrase the anchor or the positive; a negative need not always share the anchor’s content.' },
  { id:'distances',title:'Compare the two distances',
    body:'The solid line connects the anchor to its positive and the dashed line connects it to its negative. For this demonstration we compare squared Euclidean distances, so both coordinate differences enter the calculation.',
    equation:'<span>d²(z<sub>i</sub>, z<sub>j</sub>) = Σ<sub>k</sub>(z<sub>ik</sub> − z<sub>jk</sub>)²</span>' },
  { id:'loss',title:'A margin defines the target',
    body:'Triplet loss asks the positive to be closer to the anchor than the negative by at least a margin α. The loss becomes zero once that inequality is satisfied; zero loss does not require identical embeddings for texts with the same style.',
    equation:'<span>L = max{0, d²(z<sub>a</sub>, z<sub>p</sub>) − d²(z<sub>a</sub>, z<sub>n</sub>) + α}, &nbsp; α = 0.1</span>',
    note:'This is the squared-distance triplet hinge objective described in the StyleDistance paper. The numerical distances here come from our schematic coordinates.' },
  { id:'update',title:'Change the encoder, not the points',
    body:'Differentiate the loss through the shared encoder and update its trainable parameters. The published StyleDistance model adapts pretrained RoBERTa with LoRA, so the small low-rank updates are trained while the base weights remain frozen.',
    equation:'<span>θ<sub>trainable</sub> ← θ<sub>trainable</sub> − η ∇<sub>θtrainable</sub>L</span>',
    note:'The published method freezes the base model and trains LoRA updates; the instructor’s linked notebook is a separate, unverified implementation.' },
  { id:'reencode',title:'Re-encode all four passages',
    body:'Apply the updated encoder again to obtain new vectors. Because the parameters are shared, even passage C, which did not appear in this triplet, can receive a different embedding.' },
  { id:'repeat',title:'Repeat across varied examples',
    body:'A useful style representation needs comparisons spanning many topics and styles, rather than memorizing this small set. The published work combines synthetic style data with natural authorship data; our four passages only expose the logic of one training signal.',
    note:'The published main checkpoint and its synthetic-only variant use different data mixtures. The displayed geometry is a teaching illustration for neither checkpoint.' },
  { id:'heldout',title:'Test on new passages',
    body:'Reserve new passages and labeled comparisons before fitting the model, then compare their rankings under the pretrained and adapted encoders. Include content-matched negatives and style-matched positives across unfamiliar topics, rather than evaluating on the triplets used for training.',
    equation:'<span>Held-out triplet accuracy: proportion with d(z<sub>a</sub>, z<sub>p</sub>) &lt; d(z<sub>a</sub>, z<sub>n</sub>)</span>',
    note:'No held-out scores are supplied here. Any claim that fine-tuning helped requires actual encoder outputs and evaluation data.' },
  { id:'limits',title:'Similarity depends on the task',
    body:'Fine-tuning changes the representation toward the relationships rewarded by its objective. A style-sensitive embedding may be useful for studying writing while being a poor choice for semantic retrieval, and success on familiar styles does not establish generalization or complete separation of style from content.' },
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
    <desc>Schematic embedding geometry with four fixed passage identities. Color indicates writing style and shape indicates topic. This is not measured model output.</desc>
    <rect width="${width}" height="${height}" fill="${C.paper}"/>
    <g data-diagram-mode="${mode}" opacity="${n(opacity)}">${content}</g>
  </g>`;
}

export const lesson = {
  id:'style-distance',
  title:'Fine-tuning for writing style',
  subtitle:'Teach an embedding model what “similar” means',
  intro:'A pretrained text encoder may bring passages together because they discuss similar things. This story uses the StyleDistance approach to show how fine-tuning can instead reward similar writing styles across different content.',
  scenes,
  takeaways:[
    'The training objective defines which relationships a representation should preserve.',
    'A shared encoder processes every member of a triplet, and its parameter updates can change every text embedding.',
    'Triplet loss compares an anchor–positive distance with an anchor–negative distance and a margin.',
    'A schematic geometric improvement is not evidence of model performance; evaluate held-out texts and styles.',
  ],
  dataNote:'All four passages are neutral teaching examples. Every displayed coordinate, distance, and loss is calculated from a labeled two-dimensional schematic, not from a trained language model. The published StyleDistance methodology supplies the learning idea; the instructor’s linked training notebook has not been inspected.',
  sources:[
    {label:'StyleDistance: Stronger Content-Independent Style Embeddings with Synthetic Parallel Examples',url:'https://aclanthology.org/2025.naacl-long.436/',note:'Primary paper: triplet training for style embeddings, including pretrained RoBERTa with LoRA and synthetic/natural data.'},
    {label:'StyleDistance model card',url:'https://huggingface.co/StyleDistance/styledistance',note:'Published checkpoint; using this model is distinct from running a fine-tuning experiment.'},
    {label:'Instructor’s “Our Own StyleDistance Model” notebook',url:'https://colab.research.google.com/drive/1-hAD4LPOl_ahdaOvfH-X2SCKD86e-XJj?usp=sharing',note:'Linked in the lecture slides. The notebook’s implementation and outputs have not been inspected; this lesson does not attribute a particular training run to it.'},
  ],
  render,
};

// Concise captions retain the argument on short phone screens.
const shortCaptions = [
  [
    "What should be similar?",
    "Pretraining supplies useful representations. Fine-tuning can reward the particular notion of similarity we want to measure."
  ],
  [
    "Content and style",
    "The four passages vary topic and writing style separately. Letters preserve each passage’s identity."
  ],
  [
    "Start with an encoder",
    "A representation that groups texts by content can be useful. Our task instead rewards style across different topics."
  ],
  [
    "One shared encoder",
    "Anchor, positive, and negative each pass through the same encoder. Three embeddings enter one loss."
  ],
  [
    "Choose a triplet",
    "B and D share bureaucratic style across topics. A shares B’s content but uses plain style."
  ],
  [
    "Compare the distances",
    "The positive should be closer than the negative. Distances are calculated on these schematic coordinates."
  ],
  [
    "Require a margin",
    "The loss reaches zero once the positive is closer by the required margin. Identical embeddings are unnecessary."
  ],
  [
    "Change shared parameters",
    "StyleDistance trains low-rank updates to a pretrained encoder. Changing that shared function can change every text embedding."
  ],
  [
    "Re-encode every passage",
    "Recompute the embeddings after changing parameters. Even C, absent from the selected triplet, can move."
  ],
  [
    "Vary the training examples",
    "Many topics and styles are needed. Four schematic points explain the objective; they are not empirical training results."
  ],
  [
    "Test on new texts",
    "Evaluate new style-matched positives and content-matched negatives. Compare their rankings before and after adaptation."
  ],
  [
    "Similarity has a purpose",
    "A style-sensitive representation serves a particular task. Check new topics and styles before drawing broader conclusions."
  ]
];
shortCaptions.forEach(([title, body], index) => { lesson.scenes[index].shortTitle=title; lesson.scenes[index].shortBody=body; });
