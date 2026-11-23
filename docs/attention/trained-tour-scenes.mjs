// Scroll reveals recorded forward passes; it never interpolates model scores.
import {normalizeTrainedData, attentionColor, formatWeight} from './trained-scenes.mjs';

const esc = value => String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const percent = (value, digits = 2) => `${(100 * value).toFixed(digits)}%`;
const wrap = (value, width = 43) => {
  const lines = []; let row = '';
  for (const word of value.split(/\s+/)) {
    if (row && `${row} ${word}`.length > width) { lines.push(row); row = word; }
    else row += `${row ? ' ' : ''}${word}`;
  }
  if (row) lines.push(row);
  return lines;
};

export function createGuidedTrainedScenes(drawing, data, investigation) {
  const {text, line, rect, circle, m, C, ease, lerp, clamp} = drawing;
  const measured = normalizeTrainedData(data);
  if (!investigation || investigation.schema_version !== 1 || investigation.model.revision !== measured.revision) {
    throw new Error('The intervention export must match the original pinned checkpoint.');
  }
  const {behavior, phrase, brightness, context, validation} = investigation;
  const label = (value,x,y,size=26,color=C.ink,anchor='middle',extra='') =>
    text(value,x,y,size,color,anchor,extra).replace('style="',`style="font-size:${size}px;`);
  const group = (opacity,content,extra='') => `<g opacity="${opacity}" ${extra}>${content}</g>`;
  const phase = (t,start,end,reduced=false) => reduced ? Number(t>=end) : ease(clamp((t-start)/(end-start)));
  const arrow = (x1,y1,x2,y2,color=C.blue,extra='') => line(x1,y1,x2,y2,color,2,`marker-end="url(#arrow)" ${extra}`);
  const box = (value,x,y,width=124,color=C.blue) =>
    rect(x-width/2,y-25,width,50,'#fff',color,4,'stroke-width="1.6"') + label(value,x,y+9,28,color);
  const words = (value,x,y,width=43,size=27,color=C.ink,anchor='middle') =>
    wrap(value,width).map((row,i)=>label(row,x,y+i*(size+10),size,color,anchor)).join('');
  const score = (value,x,y,width=480,color=C.blue,digits=2) => {
    const p=value.positive_probability;
    return `<g data-positive-probability="${p}"><title>P(positive): ${percent(p,4)}</title>`+
      label(percent(p,digits),x+width,y,38,color,'end')+label('P(positive)',x,y,23,C.muted,'start')+
      rect(x,y+17,width,11,'#e8ecec')+rect(x,y+17,width*p,11,color)+'</g>';
  };
  const comparison = (before,after,y=359,afterLabel='Two messages removed',digits=2) =>
    label('Original',74,y,23,C.blue,'start')+score(before,74,y+37,492,C.blue,digits)+
    label(afterLabel,74,y+105,23,C.rust,'start')+score(after,74,y+142,492,C.rust,digits);
  const flowPath = (sourceX,destinationX,y1,y2,color,opacity=1) =>
    `<path d="M ${sourceX} ${y1} C ${sourceX} ${y1+52}, ${destinationX} ${y2-52}, ${destinationX} ${y2}" fill="none" stroke="${color}" stroke-width="3" opacity="${opacity}" marker-end="url(#arrow)"/>`;

  function predictions(t,{reducedMotion=false}={}) {
    let out=label('Same word. Different constructions.',320,32,26,C.muted);
    [behavior.negative,behavior.full].forEach((example,index)=>{
      const y=74+index*220,reveal=phase(t,index*.2,.28+index*.2,reducedMotion);
      out+=rect(28,y,584,194,'#fff','#cbd2d5',5,'stroke-width="1.2"');
      out+=words(example.text,58,y+40,39,28,C.ink,'start');
      out+=group(reveal,score(example.score,58,y+151,524,C.blue));
    });
    return out+label('6 blocks · 12 heads per block · final [CLS] readout',320,535,22,C.muted);
  }

  function phraseTest(t,{reducedMotion=false}={}) {
    const state=t<.34?0:t<.70?1:2;
    const examples=[behavior.full,behavior.short,behavior.without_just];
    const removeEnding=phase(t,.16,.34,reducedMotion),removeJust=phase(t,.52,.70,reducedMotion);
    let out=label(['Full review','Keep only the first sentence','Remove “just”'][state],320,36,25,C.muted);
    out+=label('Infinite Jest',320,109,35);
    out+=label('wasn’t',lerp(191,238,removeJust),170,35);
    out+=group(1-removeJust,label('just',321,170,35,C.teal));
    out+=label('good.',lerp(438,392,removeJust),170,35);
    out+=group(1-removeEnding,label('It was spectacular.',320,233+removeEnding*24,32,C.teal));
    out+=score(examples[state].score,74,309,492,C.blue);
    ['Full review','First sentence','Without “just”'].forEach((name,i)=>{
      const x=116+i*204,active=i===state;
      out+=circle(x,404,active?7:4,active?C.blue:'#bbc4c9');
      out+=label(name,x,441,21,active?C.blue:C.muted);
      out+=label(percent(examples[i].score.positive_probability),x,478,28,active?C.blue:C.muted);
    });
    return out;
  }

  function matrix() {
    const n=phrase.tokens.length,x=153,y=139,size=330,cell=size/n;
    let out=label('Layer 2 · head 9',320,31,27,C.blue);
    out+=label('Read from →',x+size/2,72,23,C.muted);
    out+=label('Update',36,y+size/2,22,C.muted,'middle',`transform="rotate(-90 36 ${y+size/2})"`);
    phrase.tokens.forEach((token,i)=>{
      const selected=['t','just'].includes(token),color=selected?C.blue:C.muted;
      out+=label(token,x-12,y+(i+.5)*cell+6,19,color,'end');
      out+=label(token,x+(i+.5)*cell,y-12,18,color,'start',`transform="rotate(-48 ${x+(i+.5)*cell} ${y-12})"`);
      phrase.attention[i].forEach((weight,j)=>{
        out+=`<g data-attention-cell="${i},${j}"><title>${esc(`Layer 2 head 9. Destination ${i+1} ${token}, source ${j+1} ${phrase.tokens[j]}: ${formatWeight(weight)}`)}</title>`+
          rect(x+j*cell,y+i*cell,cell,cell,attentionColor(weight),'#dce4df',0,'stroke-width=".5"')+'</g>';
      });
    });
    for(const edge of phrase.edges)out+=rect(x+edge.source_index*cell,y+edge.destination_index*cell,cell,cell,'none',C.rust,0,'stroke-width="3"');
    for(let i=0;i<80;i++)out+=rect(222+i*2.4,499,2.5,8,attentionColor(i/79));
    out+=label('0',208,511,19,C.muted,'end')+label('1',428,511,19,C.muted,'start');
    return out;
  }

  function messages(t,{cut=false,reducedMotion=false,compact=false}={}) {
    const top=compact?99:130,bottom=compact?259:367;
    const move=phase(t,.12,.84,reducedMotion),opacity=cut?0:1;
    let out=compact?'':label('wasn  ·  ’  ·  t',160,top-61,25,C.muted)+label('just',480,top-61,25,C.muted);
    out+=label('Incoming states',320,top-23,21,C.muted);
    out+=box('t',160,top+15)+box('just',480,top+15,124,C.teal);
    out+=box('t',160,bottom,124,C.blue)+box('just',480,bottom,124,C.teal);
    const eT=phrase.edges.find(edge=>edge.source_token==='t'),eJust=phrase.edges.find(edge=>edge.source_token==='just');
    out+=flowPath(160,480,top+43,bottom-30,C.blue,opacity)+flowPath(480,160,top+43,bottom-30,C.teal,opacity);
    // Route markers are schematic, not measured vector coordinates or norms.
    if(!cut){
      const bend=move*move*(3-2*move);
      out+=circle(lerp(160,480,bend),lerp(top+44,bottom-32,move),6,C.blue);
      out+=circle(lerp(480,160,bend),lerp(top+44,bottom-32,move),6,C.teal);
      out+=label(percent(eJust.weight,1),113,(top+bottom)/2+14,23,C.teal);
      out+=label(percent(eT.weight,1),527,(top+bottom)/2+14,23,C.blue);
    }else{
      out+=line(210,top+64,430,bottom-51,'#bbc4c9',2,'stroke-dasharray="5 7"');
      out+=line(430,top+64,210,bottom-51,'#bbc4c9',2,'stroke-dasharray="5 7"');
      out+=rect(226,(top+bottom)/2-18,188,39,'#fff');
      out+=label('Two messages off',320,(top+bottom)/2+9,23,C.rust);
    }
    out+=label('Update t',160,bottom+59,24,C.blue)+label('Update just',480,bottom+59,24,C.teal);
    return out;
  }

  function routing(t,{reducedMotion=false}={}) {
    const unfold=phase(t,.32,.57,reducedMotion);
    let out=group(1-unfold,matrix(),`transform="translate(0 ${-32*unfold})"`);
    out+=group(unfold,label('Layer 2 · head 9',320,31,27,C.blue)+messages(phase(t,.57,1,reducedMotion),{reducedMotion}));
    if(t>=.57)out+=label('Both reads use the incoming layer-1 states.',320,506,23,C.muted);
    return out;
  }

  function removal(t,{reducedMotion=false}={}) {
    let out=label(behavior.short.text,320,31,27,C.ink);
    out+=messages(t,{cut:t>=.36,reducedMotion,compact:true});
    out+=group(phase(t,.38,.53,reducedMotion),comparison(phrase.short.before,phrase.short.after,351));
    return out;
  }

  function restored(t,{reducedMotion=false}={}) {
    const example=t>=.45?phrase.full:phrase.short,arrive=phase(t,.17,.45,reducedMotion);
    let out=words(behavior.short.text,320,72,32,30);
    out+=group(arrive,label('It was spectacular.',320,184,34,C.teal));
    out+=label('The same two messages stay off.',320,258,25,C.rust);
    return out+comparison(example.before,example.after,329);
  }

  function bright(t,{reducedMotion=false}={}) {
    const [apostrophe,tPiece]=brightness.messages,showMagnitude=t>=.34,showEffect=t>=.70;
    let out=label('Negative review · layer 6, head 10 · [CLS] row',320,30,24,C.blue);
    out+=label('Source',57,94,23,C.muted,'start')+label(showMagnitude?'Message magnitude':'Attention weight',345,94,24,C.muted);
    [apostrophe,tPiece].forEach((message,i)=>{
      const y=139+i*88,v=showMagnitude?message.message_norm/.5:message.weight;
      out+=label(i===0?'apostrophe':'t',57,y+23,26,C.ink,'start');
      out+=rect(222,y,300,30,'#edf0f0')+rect(222,y,300*v,30,i?C.teal:C.blue);
      out+=label(showMagnitude?message.message_norm.toFixed(3):percent(message.weight,1),576,y+24,25,i?C.teal:C.blue,'end');
    });
    out+=label('0',222,327,21,C.muted)+label(showMagnitude?'0.5':'1',522,327,21,C.muted);
    out+=group(phase(t,.36,.55,reducedMotion),label('A larger weight can multiply a smaller vector.',320,379,25,C.muted));
    if(showEffect){
      out+=label('Remove only the apostrophe message',320,433,25,C.rust);
      out+=label(`${percent(brightness.baseline.positive_probability,4)}  →  ${percent(apostrophe.after.positive_probability,4)}`,320,479,34,C.blue);
      out+=label('P(positive) · the prediction stays negative',320,518,23,C.muted);
    }
    return out;
  }

  function stateSwap(t,{reducedMotion=false}={}) {
    const swap=phase(t,.22,.66,reducedMotion),done=t>=.66;
    let out=label('Same words; good and bad trade places.',320,30,25,C.muted);
    [context.negative,context.positive].forEach((example,i)=>{
      const x=162+i*316;
      out+=label('Infinite Jest was',x,94,27)+label(i?'good, not bad.':'bad, not good.',x,133,29);
      out+=label(percent(example.score.positive_probability),x,180,30,C.blue);
    });
    out+=label('Exchange only the state at not after block 4',320,237,25,C.muted);
    const arc=52*Math.sin(Math.PI*swap);
    out+=box('not',lerp(162,478,swap),301-arc,126,C.blue)+box('not',lerp(478,162,swap),301+arc,126,C.teal);
    out+=arrow(162,340,162,386,C.muted)+arrow(478,340,478,386,C.muted);
    out+=label('Rerun the remaining blocks',320,373,24,C.muted);
    if(done){
      context.directions.forEach((direction,i)=>{out+=label(percent(direction.state_after.positive_probability),162+i*316,431,38,C.rust);});
      out+=label('P(positive) after the swap',320,469,23,C.muted);
      out+=label(`${validation.context.flips} of ${validation.context.interventions} new swap directions also reversed.`,320,521,23,C.muted);
    }
    return out;
  }

  function valueSwap(t,{reducedMotion=false}={}) {
    const patched=t>=.61,move=phase(t,.20,.61,reducedMotion);
    const direction=context.directions.find(entry=>entry.recipient==='negative');
    let out=label('Recipient: “bad, not good.”',320,32,28,C.ink);
    out+=rect(50,87,238,137,'#fff',C.blue,5,'stroke-width="1.5"');
    out+=m('A^{(5,h)}',169,126,174,56,36,C.blue);
    out+=label('Weights unchanged',169,185,24,C.blue)+label('all 12 heads',169,213,21,C.muted);
    out+=label('Value at not',463,85,25,C.muted)+box(patched?'donor value':'own value',463,141,216,patched?C.teal:C.blue);
    out+=group(move,label('from “good, not bad”',463,207,22,C.teal));
    out+=arrow(169,230,267,275,C.blue)+arrow(463,230,379,275,patched?C.teal:C.blue);
    out+=circle(lerp(463,379,move),lerp(230,271,move),6,C.teal);
    out+=rect(147,280,346,54,'#fff','#9da9af',4,'stroke-width="1.3"')+label('Layer-5 token updates',320,316,26,C.ink);
    out+=arrow(320,336,320,367,C.muted)+label('Layer 6 → classifier',320,398,26,C.muted);
    return out+score(patched?direction.values_after:direction.before,74,472,492,patched?C.rust:C.blue);
  }

  const common={part:'DistilBERT · tested computations',label:'',equation:'',detail:'',discretePhases:true};
  return [
    {...common,id:'trained-predictions',title:'Two reviews, opposite predictions.',caption:'These are the model’s scores for the original reviews.',
      visual:'Both cards report P(positive) from the same pinned DistilBERT checkpoint.',
      reading:`<p>${esc(behavior.negative.text)}: P(positive) ${percent(behavior.negative.score.positive_probability,4)}. ${esc(behavior.full.text)}: P(positive) ${percent(behavior.full.score.positive_probability,4)}. This model has six encoder blocks, twelve heads per block, and a classifier reading the final [CLS] state.</p>`,
      footer:'Original inputs · measured predictions',draw:predictions},
    {...common,id:'trained-phrase-test',title:'“Just” changes the interpretation.',caption:'The short review stays positive until we remove just.',
      visual:'Scroll through the full review, its first sentence, and the sentence without just. Each score is a separate model run.',
      reading:`<p>The full review scores ${percent(behavior.full.score.positive_probability,4)} positive. Its first sentence alone scores ${percent(behavior.short.score.positive_probability,4)}; removing just gives ${percent(behavior.without_just.score.positive_probability,4)}. These edits rerun the complete model.</p>`,
      footer:'Text edits · three measured inputs',draw:phraseTest},
    {...common,id:'trained-phrase-routing',title:'One head connects the construction.',caption:'Layer 2, head 9 mixes information between t and just.',
      visual:'The full measured matrix unfolds into two source-to-destination message routes. The t is one piece of wasn’t.',
      reading:`<p>The destination t reads just with weight ${formatWeight(phrase.edges.find(edge=>edge.destination_token==='t').weight)}; just reads t with weight ${formatWeight(phrase.edges.find(edge=>edge.destination_token==='just').weight)}. Both rows read incoming layer-1 states simultaneously. They are not a within-layer feedback loop. Each arrow carries a weighted value vector, not a raw word.</p>`,
      footer:'Layer 2 · head 9 · arrows carry value messages',draw:routing},
    {...common,id:'trained-message-removal',title:'Keep the words. Remove two messages.',caption:'The short review’s prediction reverses.',
      visual:`The same sentence changes from ${percent(phrase.short.before.positive_probability)} to ${percent(phrase.short.after.positive_probability)} positive after the two selected messages are removed.`,
      reading:`<p>Set the two selected layer-2/head-9 coefficients to zero without renormalizing the remaining weights. Every other message, the residual paths, the shared output bias, and feed-forward networks remain; later computation reruns. The original margin ${phrase.short.before.margin.toFixed(3)} becomes ${phrase.short.after.margin.toFixed(3)}. The same intervention reverses all ${validation.phrase.short_flips} tested short title variants. These were exploratory, construction-specific checks, not a universal role for the head.</p>`,
      footer:`Same intervention reverses ${validation.phrase.short_flips}/${validation.phrase.cases.length} tested short reviews`,draw:removal},
    {...common,id:'trained-context-restored',title:'More context changes what is necessary.',caption:'With spectacular restored, the prediction survives the same intervention.',
      visual:'Restore the final sentence while the same two messages stay removed. The full review remains strongly positive.',
      reading:`<p>The full review changes from ${percent(phrase.full.before.positive_probability,4)} to ${percent(phrase.full.after.positive_probability,4)} positive under exactly the same two-edge intervention. Adding text recomputes contextual states and attention; the intervention definition stays fixed. This result does not identify a particular alternative pathway.</p>`,
      footer:'Same two messages removed · full review rerun',draw:restored},
    {...common,id:'trained-brightness-test',title:'A bright square needs a test.',caption:'The coefficient, the message, and the prediction are different quantities.',
      visual:'Compare the apostrophe and t in the same final-layer head. Their message magnitudes are similar despite very different attention weights.',
      reading:`<p>For the negative review, layer 6/head 10 gives the apostrophe weight ${formatWeight(brightness.messages[0].weight)} and t weight ${formatWeight(brightness.messages[1].weight)}. Their projected value norms are ${brightness.messages[0].projected_value_norm.toFixed(3)} and ${brightness.messages[1].projected_value_norm.toFixed(3)}. Multiplication gives message norms ${brightness.messages[0].message_norm.toFixed(3)} and ${brightness.messages[1].message_norm.toFixed(3)}. These norms measure vector magnitude, not signed sentiment contributions. Removing only the apostrophe message changes P(positive) from ${percent(brightness.baseline.positive_probability,4)} to ${percent(brightness.messages[0].after.positive_probability,4)}.</p>`,
      footer:'Layer 6 · head 10 · exact single-message removal',draw:bright},
    {...common,part:'DistilBERT · context across layers',id:'trained-state-swap',title:'The label stays “not”.',caption:'Exchange its contextual state; keep each sentence’s words fixed.',
      visual:'Swap only the not state after block 4 between two opposite word-order readings, then rerun later blocks.',
      reading:`<p>The two inputs contain the same words; good and bad exchange positions. Replacing only not’s state after the complete fourth block reverses both predictions. The initial not states are identical; swapping them has no effect. With the block-4 intervention fixed before validation, ${validation.context.flips} of ${validation.context.interventions} directions across ${validation.context.pairs} additional pairs reverse. The one nonflip also moves toward the donor reading. This is a distinct experiment from the earlier two-message removal.</p>`,
      footer:'After block 4 · both remaining blocks rerun',draw:stateSwap},
    {...common,part:'DistilBERT · context across layers',id:'trained-value-swap',title:'Same weights. Different contents.',caption:'Change only not’s layer-5 values; the prediction reverses.',
      visual:'Keep every layer-5 attention matrix fixed, replace the values at not across all twelve heads, and rerun later computation.',
      reading:`<p>The negative recipient receives only the donor’s layer-5 value projection at not, across all twelve heads. Its queries, keys, incoming residual states, and layer-5 attention coefficients are unchanged. P(positive) changes from ${percent(context.directions[0].before.positive_probability,4)} to ${percent(context.directions[0].values_after.positive_probability,4)}. The reverse direction changes from ${percent(context.directions[1].before.positive_probability,4)} to ${percent(context.directions[1].values_after.positive_probability,4)}. Layer 6, including its attention weights, is recomputed. This is not a single direct not-to-[CLS] route or a complete explanation of either original review.</p>`,
      footer:'Layer-5 weights fixed · layer 6 rerun',draw:valueSwap},
  ];
}
