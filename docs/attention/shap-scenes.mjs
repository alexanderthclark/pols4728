// Exact token-position Shapley values from rerunning the pinned classifier.
// Word labels below only sum those token attributions for display.
const esc = value => String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const signed = (value,digits=2) => `${value<0?'−':'+'}${Math.abs(value).toFixed(digits)}`;
const plain = (value,digits=2) => `${value<0?'−':''}${Math.abs(value).toFixed(digits)}`;

export function displayShapWords(example) {
  const words=[];
  for(const position of example.feature_positions){
    const [start,end]=example.token_offsets[position];
    const previous=words.at(-1);
    const fragment=example.text.slice(start,end);
    if(previous && !/\s/.test(example.text.slice(previous.end,start)) && /^[\p{L}\p{N}’']+$/u.test(fragment) && /^[\p{L}\p{N}’']+$/u.test(previous.text)){
      previous.end=end;
      previous.text=example.text.slice(previous.start,end);
      previous.token_positions.push(position);
      previous.shap_value+=example.shap_values[position];
    }else words.push({text:example.text.slice(start,end),start,end,token_positions:[position],shap_value:example.shap_values[position]});
  }
  return words;
}

export function shapWaterfall(example) {
  let total=example.baseline_margin;
  return [{label:'Baseline',start:0,end:total,value:total,kind:'baseline'},
    ...displayShapWords(example).map(word=>{
      const start=total; total+=word.shap_value;
      return {...word,label:word.text,start,end:total,value:word.shap_value,kind:'contribution'};
    }),
    {label:'Prediction',start:0,end:example.full_margin,value:example.full_margin,kind:'prediction'}];
}

function validate(data,originalData) {
  if(data?.schema_version!==1 || data.method?.name!=='exact-token-shapley') throw new Error('Expected the exact token Shapley export.');
  if(originalData && data.model?.revision!==originalData.model?.revision) throw new Error('SHAP and attention must use the same pinned checkpoint.');
  if(!Array.isArray(data.examples) || data.examples.length!==2) throw new Error('Both original reviews need SHAP results.');
  for(const example of data.examples){
    const {tokens,feature_positions,shap_values,token_offsets}=example;
    if(!Array.isArray(tokens) || tokens.length!==shap_values?.length || tokens.length!==token_offsets?.length || !shap_values.every(Number.isFinite)) throw new Error('Token attribution dimensions do not match.');
    if(!feature_positions?.length || new Set(feature_positions).size!==feature_positions.length || feature_positions.some((p,i)=>!Number.isInteger(p)||p<0||p>=tokens.length||(i&&p<=feature_positions[i-1]))) throw new Error('SHAP feature positions must be ordered, distinct token positions.');
    if(![example.baseline_margin,example.full_margin,example.positive_probability].every(Number.isFinite)) throw new Error('SHAP scores must be finite.');
    const sum=feature_positions.reduce((total,p)=>total+shap_values[p],example.baseline_margin);
    if(Math.abs(sum-example.full_margin)>2e-5) throw new Error('SHAP contributions do not reconstruct the prediction.');
    if(example.coalition_count!==2**feature_positions.length) throw new Error('Exact SHAP must evaluate every coalition.');
    const original=originalData?.examples.find(item=>item.text===example.text);
    if(originalData && (!original || Math.abs(original.logits[1]-original.logits[0]-example.full_margin)>2e-4)) throw new Error('SHAP does not reproduce the published original prediction.');
  }
  const positive=data.examples.find(example=>example.id==='positive');
  const comparison=positive?.illustrative_comparison;
  if(!comparison || ![comparison.without_margin,comparison.with_margin,comparison.marginal_gain].every(Number.isFinite) || Math.abs(comparison.with_margin-comparison.without_margin-comparison.marginal_gain)>2e-5) throw new Error('The measured masking comparison is missing or inconsistent.');
}

export function createShapScenes(drawing,data,originalData) {
  validate(data,originalData);
  const {text,line,rect,C,ease,clamp}=drawing;
  const label=(value,x,y,size=23,color=C.ink,anchor='middle',extra='')=>text(value,x,y,size,color,anchor,extra).replace('style="',`style="font-size:${size}px;`);
  const group=(opacity,content,extra='')=>`<g opacity="${opacity}" ${extra}>${content}</g>`;
  const positive=data.examples.find(example=>example.id==='positive');
  const negative=data.examples.find(example=>example.id==='negative');
  const phase=(t,start,end,reduced)=>reduced?Number(t>=end):ease(clamp((t-start)/(end-start)));
  const allRows=data.examples.flatMap(shapWaterfall);
  const rawMin=Math.min(0,...allRows.flatMap(row=>[row.start,row.end]));
  const rawMax=Math.max(0,...allRows.flatMap(row=>[row.start,row.end]));
  const axisMin=Math.floor(rawMin)-1,axisMax=Math.ceil(rawMax)+1;
  const chartX=value=>128+(value-axisMin)/(axisMax-axisMin)*424;
  const positiveColor=C.teal,negativeColor=C.rust;
  const valueColor=value=>value>=0?positiveColor:negativeColor;
  const maxMagnitude=Math.max(...data.examples.flatMap(example=>displayShapWords(example).map(word=>Math.abs(word.shap_value))));

  function coloredSentence(example) {
    let previousEnd=0;
    const spans=displayShapWords(example).map(word=>{
      const whitespace=esc(example.text.slice(previousEnd,word.start)); previousEnd=word.end;
      const color=valueColor(word.shap_value),alpha=.05+.17*Math.abs(word.shap_value)/maxMagnitude;
      const rgb=word.shap_value>=0?'0,127,115':'165,79,50';
      return whitespace+`<span title="${esc(`${word.text}: ${signed(word.shap_value,4)} log-odds; sum of token attributions`)}" style="color:${color};background:rgba(${rgb},${alpha});border-bottom:2px solid ${color};padding:0 1px">${esc(word.text)}</span>`;
    }).join('')+esc(example.text.slice(previousEnd));
    return `<foreignObject x="20" y="8" width="600" height="81"><div xmlns="http://www.w3.org/1999/xhtml" style="font-family:Arial,sans-serif;font-size:25px;line-height:1.5;text-align:center;color:${C.ink}" aria-label="${esc(example.text)}">${spans}</div></foreignObject>`;
  }

  function waterfall(example,t,{reducedMotion=false}={}) {
    const rows=shapWaterfall(example),count=rows.length-2;
    const y0=132,y1=433,gap=(y1-y0)/(rows.length-1),zero=chartX(0);
    let out=coloredSentence(example)+label('Word display = sum of its token contributions',320,101,21,C.muted);
    out+=line(zero,y0-17,zero,y1+16,'#aab4b9',1.2,'stroke-dasharray="3 4"');
    const tickStep=axisMax-axisMin>16?5:4;
    for(let value=Math.ceil(axisMin/tickStep)*tickStep;value<=axisMax;value+=tickStep){
      const x=chartX(value);
      out+=line(x,y1+26,x,y1+32,C.muted,1)+label(plain(value,0),x,y1+53,20,C.muted);
    }
    out+=line(128,y1+26,552,y1+26,C.rule,1)+label('Positive-minus-negative logit margin',340,511,21,C.muted);
    rows.forEach((row,i)=>{
      const y=y0+i*gap,progress=i===0?1:phase(t,(i-1)/(count+1),i/(count+1),reducedMotion);
      const color=row.kind==='contribution'?valueColor(row.value):C.blue;
      const rowLabel=row.label==='.'?`period ${rows.slice(0,i+1).filter(item=>item.label==='.').length}`:row.label;
      out+=label(rowLabel,113,y+6,21,i===0||progress>0?C.ink:'#9da5aa','end');
      if(i>0 && row.kind==='contribution')out+=line(chartX(row.start),y-gap+8,chartX(row.start),y-8,C.rule,1.3);
      const left=Math.min(chartX(row.start),chartX(row.end)),width=Math.abs(chartX(row.end)-chartX(row.start));
      const bar=rect(left,y-9*progress,width,18*progress,color,'none',0)+
        (width<1?line(chartX(row.end),y-8,chartX(row.end),y+8,color,1.2):'');
      out+=`<g data-shap-row="${i}" data-kind="${row.kind}" data-start="${row.start}" data-end="${row.end}" data-value="${row.value}"><title>${esc(`${row.label}: ${row.kind==='contribution'?signed(row.value,5):plain(row.value,5)} log-odds`)}</title>`+
        bar+group(progress,label(row.kind==='contribution'?signed(row.value):plain(row.value),625,y+6,21,color,'end'))+'</g>';
    });
    return out;
  }

  const methodReading=`<p>We explain the whole-input sentiment score of the pinned classifier: positive logit minus negative logit, equal to positive-class log odds. Each non-special token position is one player. An absent token is replaced by [MASK], with all positions, [CLS], [SEP], and the attention mask held fixed. Every coalition reruns the entire model in evaluation mode. This is a specified masked-input attribution problem; deleting tokens would define a different one.</p><p>For a coalition S, let v(S) be the measured logit margin after masking positions outside S. The baseline is v(∅), which need not be neutral. Exact ordinary Shapley values use <span data-math="\\phi_i=\\sum_{S\\subseteq N\\setminus\\{i\\}}\\frac{|S|!(n-|S|-1)!}{n!}\\bigl[v(S\\cup\\{i\\})-v(S)\\bigr]"></span>. Equivalently, average each token’s marginal contribution over all token-entry orders. We evaluate all ${positive.coalition_count.toLocaleString('en-US')} coalitions for each original input. This uses the ordinary Shapley game, without a Partition/Owen hierarchy.</p><p>See the <a href="https://shap.readthedocs.io/en/latest/generated/shap.maskers.Text.html">SHAP text masker</a> and <a href="https://shap.readthedocs.io/en/latest/generated/shap.PartitionExplainer.html">Partition/Owen documentation</a> for the distinction between masking and coalition choices.</p>`;
  const explanationReading=example=>`<blockquote><p>${esc(example.text)}</p></blockquote><p>Baseline ${plain(example.baseline_margin,5)} plus token contributions ${signed(example.full_margin-example.baseline_margin,5)} equals the full-input logit margin ${plain(example.full_margin,5)}. P(positive) is ${(100*example.positive_probability).toFixed(4)}%. The waterfall accumulates allocated contributions in text order; intermediate totals are not predictions for partly restored sentences. The displayed words sum constituent token values, rather than recomputing a separate word-group game. Interactions are allocated among tokens, so a word’s contribution can change with its context.</p><table><thead><tr><th>Token position</th><th>Model token</th><th>SHAP contribution (log-odds)</th></tr></thead><tbody>${example.feature_positions.map(position=>`<tr><td>${position+1}</td><td>${esc(example.tokens[position])}</td><td>${signed(example.shap_values[position],5)}</td></tr>`).join('')}</tbody></table><p>These contributions attribute the final score under the stated masking rule. Attention matrices describe internal routing; they are not the values in this decomposition.</p>`;
  const common={part:'DistilBERT · SHAP attributions',label:'',equation:'',detail:'',discretePhases:true};
  return [
    ...[negative,positive].map(example=>({...common,id:`trained-shap-${example.id}`,title:example.id==='negative'?'“Good” still pushes positive.':'Which words push this score up?',
      caption:'Baseline + signed token contributions = the prediction’s log-odds.',
      visual:`Exact token SHAP for ${example.text} The waterfall starts at ${plain(example.baseline_margin,3)} and ends at ${plain(example.full_margin,3)}. Teal, positive contributions raise the score; rust, negative contributions lower it.`,
      reading:explanationReading(example)+(example.id==='negative'?methodReading:''),footer:`${example.feature_positions.length} token players · ${example.coalition_count.toLocaleString('en-US')} coalition evaluations · exact Shapley sum`,draw:(t,options)=>waterfall(example,t,options)})),
  ];
}
