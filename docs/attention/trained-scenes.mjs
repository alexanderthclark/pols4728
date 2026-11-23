// Render exported checkpoint measurements; never synthesize attention weights.
const escapeText = value => String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
export const formatProbability = value => `${(100*value).toFixed(2)}%`;
export const formatWeight = value => Number(value).toFixed(6);
export function attentionColor(value) {
  const p=Math.max(0,Math.min(1,value));
  return `rgb(${Math.round(248*(1-p))},${Math.round(251*(1-p)+127*p)},${Math.round(250*(1-p)+115*p)})`;
}
export function normalizeTrainedData(data) {
  const labels=data.labels||data.model?.labels||['NEGATIVE','POSITIVE'];
  const examples=(data.examples||data.sentences||[]).map(example=>{
    const probabilities=example.probabilities;
    const positive=Array.isArray(probabilities)?probabilities[labels.indexOf('POSITIVE')]:probabilities?.POSITIVE??probabilities?.positive;
    const negative=Array.isArray(probabilities)?probabilities[labels.indexOf('NEGATIVE')]:probabilities?.NEGATIVE??probabilities?.negative;
    const tokens=example.tokens,attentions=example.attentions;
    if(!Array.isArray(tokens)||!tokens.length||!Array.isArray(attentions)||!attentions.length||!Number.isFinite(positive)||!Number.isFinite(negative)) throw new Error('Checkpoint export is missing tokens, probabilities, or attention tensors.');
    if(attentions.some(layer=>!Array.isArray(layer)||!layer.length||layer.some(head=>head.length!==tokens.length||head.some(row=>row.length!==tokens.length||row.some(value=>!Number.isFinite(value)||value<0||value>1))))) throw new Error('Checkpoint attention dimensions do not match the tokenizer output.');
    const prediction=example.predicted_label||example.prediction?.label||(typeof example.prediction==='string'?example.prediction:null)||(positive>=negative?'POSITIVE':'NEGATIVE');
    return {...example,tokens,attentions,positive,negative,prediction};
  });
  if(examples.length<2) throw new Error('Both Infinite Jest model outputs are required.');
  return {examples,model:typeof data.model==='string'?data.model:data.model?.id||data.model_id||'DistilBERT SST-2',revision:data.revision||data.model?.revision||'',layers:examples[0].attentions.length,heads:examples[0].attentions[0].length};
}
function wrapWords(text,max=41) {
  const lines=[];let line='';
  for(const word of text.split(/\s+/)){if(line&&`${line} ${word}`.length>max){lines.push(line);line=word;}else line+=(line?' ':'')+word;}
  if(line)lines.push(line);return lines;
}
export function createTrainedScenes({m,text,line,rect,C},data) {
  const measured=normalizeTrainedData(data),{examples,layers,heads}=measured;
  const layer=layers-1,head=0;
  const label=(s,x,y,size=23,color=C.ink,anchor='middle',extra='')=>text(s,x,y,size,color,anchor,extra);
  function predictions(t) {
    let out=label(`${layers} encoder blocks · ${heads} heads in each`,320,30,25,C.blue);
    const gap=12,blockWidth=Math.min(66,(500-gap*(layers-1))/layers),total=layers*blockWidth+(layers-1)*gap,start=(640-total)/2;
    out+=line(start-22,76,start+total+22,76,C.blue,3);
    const current=Math.min(layers-1,Math.floor(Math.max(0,Math.min(1,t))*layers));
    for(let i=0;i<layers;i++)out+=rect(start+i*(blockWidth+gap),49,blockWidth,54,i===current?'#e5f0f6':'#fff',C.blue,4,'stroke-width="1.5"')+label(i+1,start+i*(blockWidth+gap)+blockWidth/2,84,23,C.blue);
    out+=label('Final [CLS] state',105,144,22,C.blue)+label('Linear + ReLU',333,144,22,C.blue)+label('Two logits',552,144,22,C.blue)
      +line(191,137,240,137,C.blue,1.8,'marker-end="url(#arrow)"')+line(417,137,487,137,C.blue,1.8,'marker-end="url(#arrow)"')
      +label('A classifier trained on SST-2',320,189,23,C.muted);
    examples.slice(0,2).forEach((example,i)=>{
      const y=219+i*146,color=example.prediction==='POSITIVE'?C.teal:C.rust;
      out+=rect(26,y,588,130,'#fff','#d3dadc',5,'stroke-width="1.3"');
      wrapWords(example.text,43).forEach((part,j)=>out+=label(part,44,y+30+j*29,24,C.ink,'start'));
      out+=label(example.prediction,44,y+111,22,color,'start')+label(`P(positive) = ${formatProbability(example.positive)}`,590,y+111,22,C.blue,'end');
    });
    return out+label('Computed outputs, including any model mistakes.',320,528,21,C.muted);
  }
  function attention(example,t) {
    const matrix=example.attentions[layer][head],n=example.tokens.length,x=80,y=112,size=300,cell=size/n;
    const destination=Math.max(0,example.tokens.indexOf('[CLS]'));
    const ranked=matrix[destination].map((weight,index)=>({weight,index})).sort((a,b)=>b.weight-a.weight).slice(0,3);
    const active=Math.min(ranked.length-1,Math.floor(Math.max(0,Math.min(1,t))*ranked.length));
    let out=label('Source position →',x+size/2,29,23,C.teal)+label('Weight',422,55,21,C.muted,'start');
    for(let i=0;i<100;i++)out+=rect(x+i*size/100,47,size/100+.2,12,attentionColor(i/99));
    for(let i=0;i<=4;i++)out+=label((i/4).toFixed(i===0||i===4?0:2),x+i*size/4,82,18,C.muted);
    out+=label('Destination',19,y+size/2,21,C.blue,'middle',`transform="rotate(-90 19 ${y+size/2})"`);
    for(let i=0;i<n;i++){
      out+=label(i+1,x+(i+.5)*cell,103,Math.min(18,cell*.76),C.muted)+label(i+1,x-12,y+(i+.5)*cell+5,Math.min(18,cell*.76),i===destination?C.rust:C.muted,'end');
      for(let j=0;j<n;j++)out+=`<g><title>${escapeText(`Destination ${i+1} ${example.tokens[i]}, source ${j+1} ${example.tokens[j]}: ${formatWeight(matrix[i][j])}`)}</title>${rect(x+j*cell,y+i*cell,cell,cell,attentionColor(matrix[i][j]),'#e0e6e3',0,'stroke-width=".65"')}</g>`;
    }
    out+=rect(x,y+destination*cell,size,cell,'none',C.rust,0,'stroke-width="2.5"');
    const chosen=ranked[active];out+=rect(x+chosen.index*cell,y+destination*cell,cell,cell,'none',C.ink,0,'stroke-width="2.5"');
    out+=label('Position · token',424,105,20,C.muted,'start');
    example.tokens.forEach((token,i)=>out+=label(`${i+1}  ${token}`,424,126+i*Math.min(21,286/n),19,i===destination?C.rust:C.ink,'start'));
    out+=label(`Largest coefficients in the [CLS] row · layer ${layer+1}, head ${head+1}`,27,445,20,C.muted,'start');
    ranked.forEach((entry,i)=>out+=label(`${entry.index+1} · ${example.tokens[entry.index]}`,42,472+i*27,22,i===active?C.teal:C.ink,'start')+label(formatWeight(entry.weight),590,472+i*27,22,i===active?C.teal:C.ink,'end'));
    return out;
  }
  const matrixScene=(example,index)=>({
    id:`trained-attention-${index+1}`,part:'05 · A trained sentiment model',label:'Measured attention',
    title:index===0?'Read an actual attention matrix.':'Compare the same head.',
    caption:index===0?'This is one head’s measured matrix for the first sentence. The outlined row updates [CLS]; the classifier reads its final representation.':'The second sentence has its own token sequence and attention matrix. The layer, head, and color scale stay fixed.',
    equation:`A^{(${layer+1},\\,${head+1})}\\in\\mathbb R^{${example.tokens.length}\\times${example.tokens.length}}`,
    detail:`Layer ${layer+1}, head ${head+1}, chosen by position—not by its pattern. All special tokens and subwords remain. Coefficients describe mixing of contextual value vectors; they are not word-importance scores.`,
    visual:`${example.text} Model: ${example.prediction.toLowerCase()}; P(positive) ${formatProbability(example.positive)}. Colors use the same linear 0–1 scale.`,
    draw:t=>attention(example,t),
  });
  return [{id:'trained-predictions',part:'05 · A trained sentiment model',label:'Measured predictions',title:'Now run the two sentences.',
    caption:'DistilBERT was fine-tuned for positive versus negative sentiment. These are its actual predictions for the unchanged sentences.',
    equation:'p=\\operatorname{softmax}(z)',
    detail:`This checkpoint has ${layers} encoder blocks and ${heads} heads per block. Its classifier uses the final [CLS] state and a nonlinear readout, rather than the earlier schematic’s mean pooling. It scores whole-input sentiment, not a separately specified target.`,
    visual:'Both scores come from the same pinned checkpoint in evaluation mode. Intended readings and model predictions are distinct.',draw:predictions},matrixScene(examples[0],0),matrixScene(examples[1],1)];
}
