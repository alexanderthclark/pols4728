import {escapeText as esc,mathLabel} from './math.mjs?v=20261028-applications-6';

const clamp=value=>Math.max(0,Math.min(1,value));
const mix=(from,to,amount)=>from+(to-from)*amount;
const blue='#234e70',ink='#242424',muted='#606060',hairline='#deded9';
const text=(x,y,value,extra='')=>`<text x="${x}" y="${y}" font-size="32" fill="${ink}" ${extra}>${esc(value)}</text>`;
const svg=(body,label,kind)=>`<svg class="application-visual application-${kind}" viewBox="0 0 680 330" role="img" aria-label="${esc(label)}">${body}</svg>`;
const note=content=>`<p class="visual-note">${content}</p>`;

/** Constructed teaching documents, independent of the earlier four-document corpus. */
export const topicDocuments=[
  ['budget tax policy','budget spending policy'],
  ['health clinic policy','health care policy'],
];

/** BERTopic's default class-based weighting, with raw counts pooled per topic. */
export function cTfidfModel(documents=topicDocuments){
  const bags=documents.map(topic=>topic.flatMap(document=>document.toLowerCase().match(/[a-z]+/g)??[]));
  const terms=[...new Set(bags.flat())].sort();
  const counts=bags.map(bag=>terms.map(term=>bag.filter(token=>token===term).length));
  const lengths=bags.map(bag=>bag.length);
  const averageLength=Math.floor(lengths.reduce((sum,length)=>sum+length,0)/lengths.length);
  const termTotals=terms.map((_,column)=>counts.reduce((sum,row)=>sum+row[column],0));
  const weights=counts.map((row,topic)=>row.map((count,column)=>count/lengths[topic]*Math.log(1+averageLength/termTotals[column])));
  return {terms,counts,lengths,averageLength,termTotals,weights};
}

export const applicationModel=cTfidfModel();

function pooling(progress){
  const arrival=clamp((progress-.55)/.45);
  const slots=[25,205,350,540];
  const topicTerms=[['budget','tax','spending','policy'],['health','clinic','care','policy']];
  const body=topicDocuments.map((documents,topic)=>{
    const offset=topic*166,terms=topicTerms[topic],seen=new Set();
    const words=documents.flatMap((document,row)=>document.split(' ').map(word=>({word,row})));
    const labels=words.map(({word,row},index)=>{
      const column=terms.indexOf(word),duplicate=seen.has(word);
      seen.add(word);
      const y=mix(offset+70+row*47,offset+104,progress);
      const count=applicationModel.counts[topic][applicationModel.terms.indexOf(word)];
      return `<g data-topic="${topic}" data-token="${index}" data-word="${word}" opacity="${duplicate?1-arrival:1}"><text x="${slots[column]}" y="${y}" font-size="32" fill="${ink}">${word}${duplicate?'':`<tspan fill="${blue}" opacity="${arrival}" data-pooled-term="${word}" data-count="${count}"> ${count}</tspan>`}</text></g>`;
    }).join('');
    return text(25,offset+30,`Cluster ${topic===0?'A':'B'}`)+
      `<text x="655" y="${offset+30}" font-size="26" text-anchor="end" fill="${muted}">${progress<.55?'two documents':'pooled word counts'}</text>`+
      `<path d="M25 ${offset+140}H655" stroke="${hairline}" fill="none"/>`+labels;
  }).join('');
  return svg(body,'Constructed example: pool two documents in each cluster. A has budget 2, tax 1, spending 1, policy 2. B has health 2, clinic 1, care 1, policy 2.','pooling')+
    note('Six word occurrences in each cluster');
}

function ranking(progress){
  const shown=[['budget','tax','policy'],['health','clinic','policy']];
  const valueOpacity=clamp((progress-.88)/.12);
  const bars=shown.map((terms,topic)=>{
    const left=topic*350;
    return text(left+15,32,`Cluster ${topic===0?'A':'B'}`)+terms.map((term,rank)=>{
      const column=applicationModel.terms.indexOf(term);
      const share=applicationModel.counts[topic][column]/applicationModel.lengths[topic];
      const weight=applicationModel.weights[topic][column];
      // Fixed rows let readers compare changing weights without crossing labels.
      const y=78+rank*51;
      const value=mix(share,weight,progress),width=value/.5*108;
      return `<g data-ranking-topic="${topic}" data-term="${term}" data-final-weight="${weight}">`+
        text(left+15,y+10,term)+
        `<rect x="${left+125}" y="${y-14}" width="${width}" height="24" fill="${term==='policy'?muted:blue}" data-current-weight="${value}"/>`+
        `<g opacity="${valueOpacity}"><text x="${left+320}" y="${y+8}" font-size="28" text-anchor="end" fill="${ink}">${weight.toFixed(3)}</text></g></g>`;
    }).join('');
  }).join('');
  const heading=progress===0?'Within-cluster word shares':progress<1?'Reweighting and ranking terms':'c-TF–IDF weights';
  const worked=String.raw`\frac{2}{6}\ln\!\left(1+\frac{6}{2}\right)=0.462`;
  return `<h4 class="visual-title">${heading}</h4>`+
    svg(bars+`<path d="M15 218H665" stroke="${hairline}" fill="none"/>`+
      `<text x="15" y="276" font-size="28" fill="${blue}">budget, health</text>`+
      `<g opacity="${valueOpacity}">${mathLabel(205,235,455,70,worked,32)}</g>`,
    'Class-based TF-IDF ranks budget and health at 0.462, tax and clinic at 0.324, and shared policy at 0.305. Spending ties tax, and care ties clinic.','ranking')+
    note('Each cluster has six tokens. Spending ties tax; care ties clinic.');
}

function workflow(progress){
  const nodes=[
    {x:10,y:10,title:'Text embeddings',detail:'descriptions and reviews'},
    {x:370,y:10,title:'k-means groups',detail:'similar listings'},
    {x:370,y:125,title:'Top 3 clusters',detail:'highest rebooking rates'},
    {x:10,y:125,title:'TF–IDF keywords',detail:'within those clusters'},
    {x:10,y:240,title:'Suppress common',detail:'terms across all listings'},
    {x:370,y:240,title:'Listing score',detail:'weighted keyword presence'},
  ];
  const routes=[[[310,47],[370,47]],[[520,85],[520,125]],[[370,162],[310,162]],[[160,200],[160,240]],[[310,277],[370,277]]];
  const routesMarkup=routes.map(([start,end],index)=>{
    const phase=clamp(progress*5-index),x=mix(start[0],end[0],phase),y=mix(start[1],end[1],phase);
    const angle=Math.atan2(end[1]-start[1],end[0]-start[0]);
    const arrow=[-.55,.55].map(delta=>[end[0]-10*Math.cos(angle+delta),end[1]-10*Math.sin(angle+delta)]);
    return `<path d="M${start[0]} ${start[1]}L${end[0]} ${end[1]}" stroke="${hairline}" stroke-width="2" fill="none"/>`+
      `<path d="M${start[0]} ${start[1]}L${x} ${y}" stroke="${blue}" stroke-width="3" fill="none" data-route="${index}" data-route-progress="${phase}"/>`+
      `<path d="M${arrow[0][0]} ${arrow[0][1]}L${end[0]} ${end[1]}L${arrow[1][0]} ${arrow[1][1]}" stroke="${blue}" stroke-width="2" fill="none" opacity="${phase}"/>`+
      `<circle cx="${x}" cy="${y}" r="4" fill="${blue}" opacity="${phase>0&&phase<1?1:0}"/>`;
  }).join('');
  const boxes=nodes.map((node,index)=>{
    const reached=index===0?1:clamp(progress*5-index+1);
    return `<g data-workflow-step="${index}" opacity="${.3+.7*reached}">`+
      `<rect x="${node.x}" y="${node.y}" width="300" height="75" fill="white" stroke="${index===2?blue:hairline}" stroke-width="${index===2?2:1}"/>`+
      text(node.x+150,node.y+29,node.title,'text-anchor="middle"')+
      `<text x="${node.x+150}" y="${node.y+58}" font-size="25" text-anchor="middle" fill="${muted}">${esc(node.detail)}</text></g>`;
  }).join('');
  return svg(routesMarkup+boxes,'Airbnb workflow: embed descriptions and reviews, cluster listings with k-means, select the three clusters with highest rebooking, extract TF-IDF keywords, suppress terms common across listings, and construct a weighted listing score.','workflow')+
    note('TF–IDF supplies readable features for the groups.');
}

export function applicationVisual(stage,p,motion=true){
  const progress=motion?clamp((p-stage+.5)*2):1;
  if(stage===5)return pooling(progress);
  if(stage===6)return ranking(progress);
  if(stage===7)return workflow(progress);
  throw new RangeError(`Unknown text application stage: ${stage}`);
}
