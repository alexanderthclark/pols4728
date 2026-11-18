import {m, equation, svg, mathLabel, renderMath, COLORS} from '../shared.mjs';
import {question21} from './q21.mjs';
const c=COLORS;
const tx=(x,y,s,size=14,color=c.ink,anchor='middle')=>`<text x="${x}" y="${y}" fill="${color}" text-anchor="${anchor}" style="font-size:${size}px">${s}</text>`;
const natural=Array.from({length:20},(_,i)=>i+1);
const labels=[0,1,1,0,1,0,0,1,0,1,0,1,1,1,0,0,0,1,0,1];
// A fixed illustrative realization, not a simulation of a typical search.
// Omitted candidates have indices above 6 and at most 18 correct predictions.
const errors=[[1,2],[5,6,7],[9,10],[13,14,15],[19,20],[20]];
const candidates=errors.map((wrong,k)=>({k:k+1,predictions:labels.map((y,i)=>wrong.includes(i+1)?1-y:y)}));
const correct=(candidate,ids=natural)=>ids.reduce((n,id)=>n+Number(candidate.predictions[id-1]===labels[id-1]),0);
const heldIds=fold=>natural.filter(id=>Math.floor((id-1)/4)===fold);
const selectionIds=fold=>natural.filter(id=>Math.floor((id-1)/4)!==fold);
const choose=ids=>candidates.reduce((best,candidate)=>correct(candidate,ids)>correct(best,ids)?candidate:best,candidates[0]);
const winner=choose(natural);
const outerResults=Array.from({length:5},(_,fold)=>{
  const selected=choose(selectionIds(fold));
  return {fold,k:selected.k,selectionCorrect:correct(selected,selectionIds(fold)),evaluationCorrect:correct(selected,heldIds(fold))};
});
export const q22Example={labels,candidates,winner:winner.k,outerResults};
const reshuffleOrders=[natural,[7,12,20,3,16,2,18,5,9,15,1,11,6,14,4,19,13,8,17,10],[11,4,15,2,9,17,6,13,1,8,19,14,20,5,12,7,3,18,10,16]];
const frame=(title,content,note='')=>`<h3 class="viz-title">${title}</h3>${content}${note?`<p class="viz-note">${note}</p>`:''}`;
const center=(tex)=>`<div style="text-align:center">${equation(tex)}</div>`;

function matrix({scores=false,selected=0,outer=-1,hideHeld=false}={}) {
  const ids=outer<0?natural:selectionIds(outer);
  const bits=(values,fold,target=false)=>heldIds(fold).map(id=>{
    const masked=target&&hideHeld&&fold===outer;
    const tested=scores&&!target&&(outer<0||fold!==outer);
    const matches=values[id-1]===labels[id-1];
    const background=tested?(matches?'#e8f3ef':'#f4e4dd'):'transparent';
    return `<span style="display:inline-block;width:23%;border-radius:2px;background:${background};color:${tested&&!matches?c.orange:c.ink}">${masked?'?':values[id-1]}</span>`;
  }).join('');
  const cells=(values,target=false)=>Array.from({length:5},(_,fold)=>`<td style="padding:5px 1px;${fold===outer?'background:#fff3e8;':''}">${bits(values,fold,target)}</td>`).join('');
  return `<table class="mini-table" style="table-layout:fixed;margin:8px 0;font-size:12px;line-height:1.25" aria-label="Fixed prediction matrix, grouped into five folds of four observations">
    <thead><tr><th style="width:11%;padding:5px 0">Row</th>${Array.from({length:5},(_,j)=>`<th style="padding:5px 0;font-size:10px">${j*4+1}–${j*4+4}</th>`).join('')}<th style="width:17%;padding:5px 0;font-size:10px">${outer<0?'Correct':'On 16'}</th></tr></thead>
    <tbody><tr style="border-bottom:2px solid #abb6bd"><th style="padding:6px 0">${m('y')}</th>${cells(labels,true)}<td style="padding:5px 0">—</td></tr>
    ${candidates.map(candidate=>`<tr ${candidate.k===selected?`style="outline:2px solid ${c.green};outline-offset:-1px;background:#e8f3ef"`:''}><th style="padding:6px 0">${m(`X_${candidate.k}`)}</th>${cells(candidate.predictions)}<td style="padding:5px 0">${scores?m(`${correct(candidate,ids)}/${ids.length}`):'—'}</td></tr>`).join('')}</tbody></table>`;
}

function foldGrid(order=natural,revealed=5,x0=20,y0=27,width=360) {
  let content=tx(x0+9,y0-11,'Fold',11,c.muted)+tx(x0+97,y0-11,'Observation IDs',11,c.muted)+tx(x0+224,y0-11,'Correct',11,c.muted);
  for(let j=0;j<5;j++) {
    const visible=j<revealed;
    const ids=order.slice(j*4,j*4+4);
    content+=tx(x0+9,y0+j*42+20,`${j+1}`,12,c.muted);
    ids.forEach((id,i)=>{
      const right=winner.predictions[id-1]===labels[id-1];
      const x=x0+32+i*34,y=y0+j*42;
      content+=`<rect x="${x}" y="${y}" width="29" height="29" rx="3" fill="${visible?(right?'#e8f3ef':'#f4e4dd'):'#f4f6f7'}" stroke="${visible?(right?c.green:c.orange):c.line}"/>`;
      content+=tx(x+14.5,y+19,`${id}`,12,visible?(right?c.ink:c.orange):c.muted);
    });
    if(visible) content+=mathLabel(x0+224,y0+j*42+15,`${correct(winner,ids)}/4`,70,32);
    else content+=tx(x0+224,y0+j*42+20,'—',14,c.muted);
  }
  return svg(content,`0 0 ${width} 245`,'Five folds of four observation identities. Green means correct and rust means incorrect for the same fixed candidate.');
}

function ranking(outer=-1,selected=0) {
  const ids=outer<0?natural:selectionIds(outer);
  return `<table class="mini-table" style="margin:8px 0" aria-label="Candidate accuracy comparison"><thead><tr><th>Candidate</th><th>${outer<0?'On all 20':'On the other 16'}</th><th>Correct</th></tr></thead><tbody>${candidates.map(candidate=>{
    const n=correct(candidate,ids),active=candidate.k===selected;
    return `<tr ${active?'style="background:#e8f3ef"':''}><th>${m(`X_${candidate.k}`)}${active?' ✓':''}</th><td style="width:55%"><div style="height:10px;background:#e8edef"><div style="height:100%;width:${n/ids.length*100}%;background:${active?c.green:c.blue}"></div></div></td><td>${m(`${n}/${ids.length}`)}</td></tr>`;
  }).join('')}</tbody></table>`;
}

function heldoutCards(fold,reveal=false) {
  const result=outerResults[fold],selected=candidates[result.k-1];
  return `<div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:7px;margin:10px 0">${heldIds(fold).map(id=>{
    const matches=selected.predictions[id-1]===labels[id-1];
    return `<div style="text-align:center;padding:7px 3px;border:1px solid ${reveal?(matches?c.green:c.orange):c.line};background:${reveal?(matches?'#e8f3ef':'#f4e4dd'):'#fff3e8'};font-size:12px;line-height:1.6"><span style="font-size:10px;color:${c.muted}">Obs. ${id}</span><div>${m(`X_${result.k}`)}: ${selected.predictions[id-1]}</div><div>${m('y')}: ${reveal?labels[id-1]:'?'}</div><div>${reveal?(matches?'Correct':'Error'):'Hidden'}</div></div>`;
  }).join('')}</div>`;
}

function scoreTray(completed=5,active=-1) {
  return `<div style="display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:5px;margin:12px 0">${outerResults.map((result,j)=>`<div style="text-align:center;border-top:3px solid ${j===active?c.orange:c.line};padding:4px 1px;background:${j<completed?'#f4f6f7':'transparent'};font-size:12px;line-height:1.6"><span style="font-size:10px">Fold ${j+1}</span><div>${j<completed?m(`X_${result.k}`):'—'}</div><div>${j<completed?m(`${result.evaluationCorrect}/4`):'—'}</div></div>`).join('')}</div>`;
}

function outerSummary(fold) {
  const result=outerResults[fold];
  return `<div style="font-size:13px;line-height:1.5;padding:6px 0;border-top:2px solid #c8cecf">Fixed choice: ${m(`X_${result.k}`)} · ${m(`${result.selectionCorrect}/16`)} on selection data</div>${heldoutCards(fold,true)}${scoreTray(fold+1,fold)}`;
}

const phaseFor=(stage,progress)=>{
  const count=stage===2||stage===12?5:stage===7?3:1;
  return Math.min(count-1,Math.floor(Math.max(0,progress)*count/.8));
};

function renderCoins(stage,container,progress=1) {
  const phase=phaseFor(stage,progress);
  container.dataset.coinPhase=String(phase);
  container.dataset.coinStage=String(stage);
  if(stage===0) {
    container.innerHTML=frame('Compare the prediction with the true label',`<table class="mini-table"><thead><tr><th>True label</th><th>Fair-coin prediction</th><th>Chance of a match</th></tr></thead><tbody><tr><td>0</td><td>0 or 1, equally likely</td><td>${m('1/2')}</td></tr><tr><td>1</td><td>0 or 1, equally likely</td><td>${m('1/2')}</td></tr></tbody></table>${center(String.raw`\mathbb E[\mathrm{Accuracy}_{\mathrm{new}}]=50\%`)}`,'The expectation is 50%. A realized score can be higher or lower.');
  } else if(stage===1) {
    container.innerHTML=frame('Draw once. Keep every prediction fixed.',matrix(),'Columns are observation IDs, grouped four at a time. Six illustrative candidate rows are shown; rows 7–32,768 are omitted.');
  } else if(stage===2) {
    container.innerHTML=frame(`Inspect one fixed candidate: ${m('X_6')}`,foldGrid(natural,phase+1),'Scroll to count each fold. Cell numbers identify observations; green means correct, rust means incorrect. Nothing is fitted.');
  } else if(stage===3) {
    container.innerHTML=frame('Five equal folds recover the same total',foldGrid()+center(String.raw`\frac15\left(\frac44+\frac44+\frac44+\frac44+\frac34\right)=\frac{19}{20}`),'Every observation contributes once. Moving it to another fold cannot change its prediction.');
  } else if(stage===4) {
    container.innerHTML=frame('The labels determine every candidate’s score',matrix({scores:true}),'Green cells match the target above; rust cells do not. Omitted candidates score below 19/20 in this fixed illustration.');
  } else if(stage===5) {
    container.innerHTML=frame('Choose the largest score',ranking(-1,winner.k)+center(String.raw`y\ \longrightarrow\ \text{scores}\ \longrightarrow\ X_6`),'The same 20 outcomes both rank candidates and supply the reported 95% score. This illustrates the given winner, not a typical search maximum.');
  } else if(stage===6) {
    container.innerHTML=frame('Old-data selection cannot predict fresh coin flips',`<div class="metrics"><div class="metric">Selected on these 20<strong>${m('19/20')}</strong>95% observed</div><div class="metric">Untouched observations<strong>${m(String.raw`50\%`)}</strong>expected accuracy</div></div>${center(String.raw`\Pr(\text{new match}\mid\text{old data})=\tfrac12`)}`,'The new predictions remain independent of the new target labels. Expected accuracy is not a promise about a particular test sample.');
  } else if(stage===7) {
    const order=reshuffleOrders[phase];
    container.innerHTML=frame(phase===0?'Original partition: 19 correct':`New partition ${phase}: still 19 correct`,foldGrid(order)+center(String.raw`\frac15\sum_{j=1}^{5}a_j=\frac{19}{20}=95\%`),'Scroll to regroup the same observation IDs. The one error moves between folds, but the predictions and labels remain fixed.');
  } else if(stage===8) {
    container.innerHTML=frame('Outer fold 1: hide four target labels',matrix({outer:0,hideHeld:true}),'Orange columns are held out. Their predictions already exist, but their four target labels cannot help choose the predictor.');
  } else if(stage===9) {
    container.innerHTML=frame('Select using only the other 16 outcomes',ranking(0,outerResults[0].k),`${m('X_1')} is perfect on these 16 and wins this selection. For every fold, break ties by the lowest candidate index, fixed in advance.`);
  } else if(stage===10) {
    container.innerHTML=frame('Freeze the choice before revealing the labels',`<div class="metrics"><div class="metric">Choice is now fixed<strong>${m('k_1=1')}</strong>${m('X_1')}, selected on 16 outcomes</div></div>${heldoutCards(0)}`,'The four held-out predictions are fixed. Their target labels are still hidden; the choice cannot be changed after seeing them.');
  } else if(stage===11) {
    container.innerHTML=frame('Reveal the four labels and score that choice',heldoutCards(0,true)+center(String.raw`a_1=\frac24=50\%`),'Only now do the held-out outcomes enter. This score evaluates the selected predictor on observations excluded from its selection.');
  } else if(stage===12) {
    container.innerHTML=frame('Repeat the entire selection in every outer fold',`<div data-outer-diagram></div><div class="control"><label>Inspect fold <output data-outer-number>${phase+1}</output> of 5</label><input data-outer-fold type="range" min="1" max="5" step="1" value="${phase+1}" aria-label="Outer evaluation fold"></div>`,'Scroll through all five folds, collecting their held-out scores. The optional slider lets you inspect a fold.');
    const slider=container.querySelector('[data-outer-fold]');
    container._outerFoldManual=false;
    const draw=()=>{
      const fold=Number(slider.value)-1;
      container.querySelector('[data-outer-diagram]').innerHTML=outerSummary(fold);
      container.querySelector('[data-outer-number]').textContent=String(fold+1);
      renderMath(container);
    };
    container._outerFoldDraw=draw;
    const manual=()=>{container._outerFoldManual=true;};
    slider.addEventListener('pointerdown',manual);
    slider.addEventListener('keydown',manual);
    slider.addEventListener('input',()=>{manual();draw();});
    draw();
  } else if(stage===13) {
    container.innerHTML=frame('Average the five evaluation scores',scoreTray()+center(String.raw`\widehat a_{\mathrm{outer}}=\frac15\left(\frac24+\frac14+\frac24+\frac14+\frac24\right)`)+`<div class="metrics"><div class="metric">This illustration<strong>${m(String.raw`8/20=40\%`)}</strong>one realized estimate</div><div class="metric">Across repetitions<strong>${m(String.raw`50\%`)}</strong>expected outer accuracy</div></div>`,'All candidate predictions stayed fixed. The illustrative 40% is not the expectation; with only 20 observations, the estimate is noisy.');
  }
  renderMath(container);
}

function updateCoins(progress,container,stage) {
  const phase=phaseFor(stage,progress);
  if(stage===12) {
    if(container._outerFoldManual) return;
    const slider=container.querySelector('[data-outer-fold]');
    if(slider&&Number(slider.value)!==phase+1) {
      slider.value=String(phase+1);
      container._outerFoldDraw();
    }
  } else if((stage===2||stage===7)&&container.dataset.coinPhase!==String(phase)) renderCoins(stage,container,progress);
}

export const questions = [question21,
  {
    id:22,
    kicker:'Selection & evaluation',
    title:'Overfitting without fitting a single parameter.',
    dek:'A search over fixed random predictors can adapt to outcomes just as surely as a fitted coefficient can.',
    prompt:`<p>We have 20 total observations with a target ${m(String.raw`y\in\{0,1\}^{20}`)}. Suppose there are ${m('2^{15}')} random predictors that are random vectors ${m(String.raw`X\in\{0,1\}^{20}`)} with each element drawn iid from Bernoulli(0.5). We consider each predictor to be a “model” with no training parameters so that ${m(String.raw`\hat{y}=X_k`)} for the model that uses predictor ${m('k')}. The goal is to pick a single model, meaning we select a single predictor. The predictor values are drawn once and kept fixed throughout the analysis. The predictors are independent of ${m('y')} and remain independent fair coin flips on new observations.</p>`,
    parts:[
      'What is the expected test accuracy of each of these candidate predictors?',
      '<p>A researcher evaluates every candidate using five equally sized CV folds on all 20 observations, selects the predictor with the highest average CV accuracy, and reports its score of 95%. She argues:</p><blockquote>“Every score was cross-validated, and none of the models was trained, removing all paths toward overfitting, so the selected model should achieve about 95% accuracy on new data.”</blockquote><p>Assess this argument. Explain where the observed outcomes enter the procedure and whether selection can overfit even though there are no training parameters.</p>',
      'She keeps the selected predictor fixed, randomly divides the <em>same 20 observations</em> into five new equally sized folds, and repeats CV for that predictor alone. Will its CV accuracy change? Does this remove the optimism? Explain briefly.',
      'Describe an evaluation procedure using five outer folds that keeps each evaluation observation out of model selection. State where the predictor is selected and evaluated. Should any data be devoted to fitting the candidate models’ parameters?'
    ],
    sections:[
      {label:'(a) Chance predictions',step:0},
      {label:'(b) Scores & selection',step:2},
      {label:'(c) Reshuffling',step:7},
      {label:'(d) Outer evaluation',step:8}
    ],
    steps:[
      {title:'Accuracy starts with a match.',body:`<p>A prediction is correct when it equals the true label. For either true label, an independent fair coin matches with probability ${m('1/2')}.</p><p>Every candidate therefore has expected accuracy ${m(String.raw`50\%`)} on new observations. A particular sample can score above or below that expectation.</p>`,caption:'Part (a) · Either target label has the same chance of a correct prediction.'},
      {title:'Keep the predictions fixed.',body:`<p>Here is one schematic realization of the data. Each row is a candidate; each column is an observation. Compare a candidate’s bits with the target row ${m('y')}.</p><p>We show six candidates from the much larger search. These values stay fixed throughout every score, reshuffle, and outer fold. The illustration is constructed to match the question’s reported ${m('19/20')} winner.</p>`,caption:'Setup · One consistent matrix supports every subsequent scene.'},
      {title:'Count one candidate’s matches, fold by fold.',body:`<p>First inspect ${m('X_6')} without choosing among candidates. In the first four folds, it gets all four observations right. In the fifth, it gets three right.</p><p>Scroll to count the folds. There is no fitting step: excluding an observation cannot change this candidate’s prediction for it.</p>`,caption:'Part (b) · Five folds yield 4, 4, 4, 4, and 3 correct predictions.'},
      {title:'Average the five fold scores.',body:`<p>The five folds are equally sized. Averaging their accuracies gives exactly the fraction correct across all 20 observations: ${m(String.raw`19/20=95\%`)}.</p><p>Every observation appears once in the evaluation total. For any fixed candidate here, cross-validation only groups the same prediction–label comparisons.</p>`,caption:'Part (b) · Equal fold weights recover the full-sample accuracy.'},
      {title:'Now score every candidate against those labels.',body:`<p>Repeat the same calculation for each candidate. Green cells match the target; rust cells disagree. The scores come from all 20 observed outcomes.</p><p>In this illustration, ${m('X_6')} has 19 matches. The other displayed candidates have 17 or 18, and the omitted candidates have fewer than 19.</p>`,caption:'Part (b) · The observed outcomes become the inputs to a comparison.'},
      {title:'The largest score determines the choice.',body:`<p>Choose ${m('X_6')}, the candidate with the highest score. Although its predictions were never fitted, its <em>selection</em> depended on the observed outcomes.</p><p>Those same outcomes then supply its reported ${m(String.raw`95\%`)} accuracy. This reuse is why selection can overfit. The given winning score is not the expected maximum in every search.</p>`,caption:'Part (b) · Outcomes enter through scores and then through the selected identity.'},
      {title:'The lucky match does not carry to new observations.',body:`<p>Choosing a candidate that matched these labels cannot make its future fair-coin predictions informative. On untouched observations, the selected candidate’s expected accuracy is still ${m(String.raw`50\%`)}.</p><p>The distinction is between its observed score on outcomes used for selection and its expected performance on new outcomes.</p>`,caption:'Parts (a)–(b) · A 95% search score and 50% expected test accuracy can coexist.'},
      {title:'New folds do not create new information.',body:`<p>Keep ${m('X_6')} fixed and scroll to repartition the same observations. The one error changes folds, but it remains an error. There are still 19 correct predictions.</p><p>The repeated CV score stays exactly ${m(String.raw`95\%`)}. These outcomes already helped choose the winner; regrouping them cannot remove that reuse.</p>`,caption:'Part (c) · New fold membership leaves every prediction–label match unchanged.'},
      {title:'Start an outer fold by hiding four labels.',body:`<p>For outer fold 1, set aside observations 1–4. Their target labels are unavailable to the selection procedure.</p><p>All candidates keep their original predictions. Use only the other 16 outcomes to compare them. This creates an evaluation group whose outcomes cannot influence the choice.</p>`,caption:'Part (d) · Exclude evaluation outcomes before any comparison is made.'},
      {title:'Select again using the other 16.',body:`<p>Recalculate each candidate’s accuracy on these 16 observations. ${m('X_1')} gets all 16 right, so it is selected instead of the full-sample winner ${m('X_6')}.</p><p>Fix tie-breaking before looking at evaluation outcomes. Here the lowest candidate index wins a tie. The omitted, higher-index candidates cannot beat a perfect score.</p>`,caption:'Part (d) · Outer evaluation repeats selection; it does not reuse the old winner.'},
      {title:'Freeze the selected identity.',body:`<p>The first outer fold’s choice is now ${m('k_1=1')}. Record the four held-out predictions from ${m('X_1')} while their true labels remain hidden.</p><p>Nothing is fitted, and the choice cannot change after the evaluation outcomes are revealed.</p>`,caption:'Part (d) · Choose first, then evaluate that fixed choice.'},
      {title:'Reveal the labels and count the matches.',body:`<p>Now reveal the four held-out outcomes. ${m('X_1')} matches two and misses two, giving ${m(String.raw`a_1=2/4=50\%`)}.</p><p>This is one outer-fold evaluation score. Its four outcomes played no role in selecting ${m('X_1')}.</p>`,caption:'Part (d) · Held-out outcomes enter only after the choice is frozen.'},
      {title:'Repeat the whole procedure five times.',body:`<p>Scroll from the fold 1 recap through folds 2–5. Each excludes a different group of four, repeats selection on the remaining 16, then evaluates the chosen predictor on its excluded group.</p><p>In this fixed example, the choices are ${m(String.raw`X_1,\ldots,X_5`)}. In fold 5, ${m('X_5')} and ${m('X_6')} tie on the selection data; the predetermined rule chooses ${m('X_5')}.</p>`,caption:'Part (d) · Collect five evaluation scores from five separate selections.'},
      {title:'Average the evaluations, not the selection scores.',body:`<p>The five held-out counts are 2, 1, 2, 1, and 2 correct. Their average accuracy is ${m(String.raw`8/20=40\%`)} in this illustration.</p><p>Under the question’s fair-coin model, expected outer accuracy is ${m(String.raw`50\%`)}. An estimate based on only 20 observations can be noisy.</p><p>No data need to be reserved for fitting candidate parameters: there are none. Selection uses the 16 available outcomes directly, with no additional inner training split.</p>`,caption:'Part (d) · The illustrative 40% estimate is distinct from the 50% expectation.'}
    ],
    takeaway:`The reported ${m(String.raw`95\%`)} winning score reuses selection outcomes. Reshuffling those outcomes leaves it unchanged. Honest outer evaluation repeats selection on 16 observations, freezes that choice, and scores it on four excluded observations. Average the five held-out scores; expected new-data accuracy remains ${m(String.raw`50\%`)}.`,
    render:renderCoins,update:updateCoins
  }
];
