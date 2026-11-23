import {attentionColor,formatProbability,formatWeight,normalizeTrainedData} from './trained-scenes.mjs';

const esc=value=>String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');

export function buildExplorer(element,data) {
  if(!element)throw new Error('An attention-explorer container is required.');
  const measured=normalizeTrainedData(data),examples=measured.examples;
  const state={sentence:0,layer:measured.layers-1,head:0,destination:0};
  const prefix=element.id||'trained-explorer';
  const select=(name,title,options)=>`<label for="${prefix}-${name}">${title}<select id="${prefix}-${name}" name="${name}">${options}</select></label>`;
  const option=(value,label,selected=false)=>`<option value="${value}"${selected?' selected':''}>${esc(label)}</option>`;
  element.innerHTML=`<div class="trained-explorer-controls">
    ${select('sentence','Sentence',examples.map((example,i)=>option(i,`${i+1}. ${example.text}`)).join(''))}
    ${select('layer','Encoder layer',Array.from({length:measured.layers},(_,i)=>option(i,i+1,i===state.layer)).join(''))}
    ${select('head','Attention head',Array.from({length:measured.heads},(_,i)=>option(i,i+1)).join(''))}
    ${select('destination','Destination token','')}
    </div><p class="trained-explorer-summary"></p><p class="trained-explorer-status" role="status" aria-live="polite"></p>
    <div class="trained-explorer-grid"><div class="trained-heatmap-scroll" tabindex="0" aria-label="Attention matrix; scroll horizontally if needed"><div class="trained-heatmap"></div></div><div class="trained-weight-table"></div></div>
    <p class="trained-explorer-note">Rows are destinations; columns are sources. The selected row includes every source, including special tokens. Weights use a fixed linear 0–1 color scale. The table reports the stored coefficients to six decimal places; displayed rounding can affect their sum.</p>`;
  const fields=Object.fromEntries(['sentence','layer','head','destination'].map(name=>[name,element.querySelector(`[name="${name}"]`)]));
  const summary=element.querySelector('.trained-explorer-summary'),status=element.querySelector('.trained-explorer-status'),heatmap=element.querySelector('.trained-heatmap'),table=element.querySelector('.trained-weight-table');
  function fillDestinations() {
    const example=examples[state.sentence];state.destination=Math.min(state.destination,example.tokens.length-1);
    fields.destination.innerHTML=example.tokens.map((token,i)=>option(i,`${i+1}. ${token}`,i===state.destination)).join('');
  }
  function render() {
    const example=examples[state.sentence],matrix=example.attentions[state.layer][state.head],n=example.tokens.length,row=matrix[state.destination];
    summary.textContent=`${example.text} — ${example.prediction.toLowerCase()}; P(positive) = ${formatProbability(example.positive)}.`;
    status.textContent=`Layer ${state.layer+1}, head ${state.head+1}. Destination ${state.destination+1}: ${example.tokens[state.destination]}. Row sum: ${row.reduce((a,b)=>a+b,0).toFixed(6)}.`;
    const x=78,y=66,size=340,cell=size/n;
    let svg=`<svg viewBox="0 0 640 ${Math.max(476,100+n*23)}" role="img" aria-label="Measured attention for sentence ${state.sentence+1}, layer ${state.layer+1}, head ${state.head+1}. Selected destination ${esc(example.tokens[state.destination])}. Exact weights appear in the adjacent table."><text x="248" y="25" text-anchor="middle" fill="#007f73" font-size="20">Source position →</text><text x="20" y="236" text-anchor="middle" fill="#234e70" font-size="20" transform="rotate(-90 20 236)">Destination</text>`;
    for(let i=0;i<n;i++){
      svg+=`<text x="${x+(i+.5)*cell}" y="55" text-anchor="middle" font-size="16" fill="#606060">${i+1}</text><text x="${x-12}" y="${y+(i+.5)*cell+5}" text-anchor="end" font-size="16" fill="#606060">${i+1}</text>`;
      for(let j=0;j<n;j++)svg+=`<rect x="${x+j*cell}" y="${y+i*cell}" width="${cell}" height="${cell}" fill="${attentionColor(matrix[i][j])}" stroke="#e0e6e3" stroke-width=".65"><title>${esc(`${example.tokens[i]} reads ${example.tokens[j]}: ${formatWeight(matrix[i][j])}`)}</title></rect>`;
      svg+=`<text x="446" y="${y+14+i*23}" font-size="18" fill="${i===state.destination?'#a54f32':'#242424'}">${i+1}  ${esc(example.tokens[i])}</text>`;
    }
    svg+=`<rect x="${x}" y="${y+state.destination*cell}" width="${size}" height="${cell}" fill="none" stroke="#a54f32" stroke-width="2.5"/>`;
    for(let i=0;i<100;i++)svg+=`<rect x="${x+i*size/100}" y="${y+size+20}" width="${size/100+.2}" height="10" fill="${attentionColor(i/99)}"/>`;
    svg+=`<text x="${x}" y="${y+size+48}" font-size="17" fill="#606060">0</text><text x="${x+size}" y="${y+size+48}" text-anchor="end" font-size="17" fill="#606060">1</text></svg>`;
    heatmap.innerHTML=svg;
    table.innerHTML=`<table><caption>Sources read by ${esc(example.tokens[state.destination])}</caption><thead><tr><th scope="col">Position</th><th scope="col">Source token</th><th scope="col">Weight</th></tr></thead><tbody>${row.map((value,i)=>`<tr><td>${i+1}</td><th scope="row">${esc(example.tokens[i])}</th><td>${formatWeight(value)}</td></tr>`).join('')}</tbody></table>`;
  }
  for(const [name,field] of Object.entries(fields))field.addEventListener('change',()=>{
    state[name]=Number(field.value);
    if(name==='sentence'){
      const example=examples[state.sentence];state.layer=Math.min(state.layer,example.attentions.length-1);state.head=Math.min(state.head,example.attentions[state.layer].length-1);
      fields.layer.value=String(state.layer);fields.head.value=String(state.head);state.destination=Math.max(0,example.tokens.indexOf('[CLS]'));fillDestinations();
    }
    render();
  });
  state.destination=Math.max(0,examples[0].tokens.indexOf('[CLS]'));fillDestinations();render();
  return {render};
}
