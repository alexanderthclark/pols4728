(() => {
  'use strict';
  const M = window.ImbalanceModel;
  const blue='#234e70', orange='#a54f32', teal='#007f73', grid='#e4e7e9';
  const pct=(x,d=1)=>(100*x).toFixed(d)+'%';
  const num=x=>Math.round(x).toLocaleString('en-US');
  const samplePi=.0247, balancedS=samplePi/(1-samplePi);
  function svg(content,label,height=300){return `<svg class="chart" viewBox="0 0 620 ${height}" role="img" aria-label="${label}">${content}</svg>`;}
  function linePath(points){return points.map((p,i)=>(i?'L':'M')+p[0].toFixed(2)+','+p[1].toFixed(2)).join(' ');}
  function chartAxes(yLabel,xLabel,xTicks,yTicks,fx,fy,height=300){let s='';yTicks.forEach(t=>{s+=`<line x1="62" y1="${fy(t.v)}" x2="585" y2="${fy(t.v)}" stroke="${grid}"/><text x="51" y="${fy(t.v)+5}" text-anchor="end">${t.label}</text>`;});xTicks.forEach(t=>{s+=`<text x="${fx(t.v)}" y="${height-27}" text-anchor="middle">${t.label}</text>`;});return s+`<text x="62" y="20" class="title">${yLabel}</text><text x="585" y="${height-2}" text-anchor="end">${xLabel}</text>`;}

  // Each figure owns its state. Scrolling never resets or replaces a figure.
  document.querySelectorAll('[data-visual]').forEach((panel,index)=>{
    const $=role=>panel.querySelector(`[data-role="${role}"]`);
    const kind=panel.dataset.visual, mode=panel.dataset.mode;
    const state={step:Number(mode),s:mode==='population'?1:.1,scale:mode==='population'?'prob':'log',batch:128,prior:mode==='11'?.5:.01};
    const ctx={panel,state,$,mode};
    const id=`figure-${index}`;
    if(kind==='counts'){drawCounts(ctx);return;}
    if(kind==='logistic'){
      const interactive=mode==='interactive';
      const corrected=mode==='correction'||interactive;
      panel.innerHTML=`<div class="panel-head"><h3>${mode==='population'?'Population probabilities':mode==='shift'?'The same slope, 2.30 higher':mode==='correction'?'Add log(s) to return to the population':'Sampling changes the intercept'}</h3>${interactive?`<div class="segmented"><button data-role="prob" type="button">Probability</button><button data-role="log" type="button">Log odds</button></div>`:''}</div><div class="legend"><span><i class="swatch"></i>Population</span>${mode!=='population'?'<span><i class="swatch orange"></i>Sampled</span>':''}${corrected?'<span><i class="swatch teal"></i>Corrected</span>':''}</div><div data-role="figure"></div>${interactive?`<div class="control"><div class="control-top"><label for="${id}-retention">Negative cases retained, s</label><output data-role="retained" for="${id}-retention"></output></div><input id="${id}-retention" data-role="retention" type="range" min="-3" max="0" step="0.001" value="-1"><div class="control-actions"><span>0.1%</span><button class="preset" data-role="balance" type="button">Make sample 50/50</button><button class="preset" data-role="keep-all" type="button">Keep all</button><span>100%</span></div></div>`:''}<div class="metrics" data-role="metrics"></div><div class="formula-chip" data-role="formula"></div><p class="panel-note" data-role="note" ${interactive?'aria-live="polite"':''}></p>`;
      if(interactive){
        $('retention').addEventListener('input',e=>{state.s=10**Number(e.target.value);refreshLogistic(ctx);});
        $('balance').addEventListener('click',()=>{state.s=balancedS;$('retention').value=Math.log10(state.s);refreshLogistic(ctx);});
        $('keep-all').addEventListener('click',()=>{state.s=1;$('retention').value=0;refreshLogistic(ctx);});
        ['prob','log'].forEach(v=>$(v).addEventListener('click',()=>{state.scale=v;refreshLogistic(ctx);}));
      }
      refreshLogistic(ctx);
    }else if(kind==='efficiency'){
      const showBatch=mode!=='6';
      panel.innerHTML=`<div class="panel-head"><h3>${mode==='6'?'One pass through the data':mode==='7'?'Positive examples per batch':'Sampling and influence'}</h3></div><div data-role="figure"></div>${showBatch?`<div class="control"><div class="control-top"><label for="${id}-batch">Batch size</label><output data-role="batch-value" for="${id}-batch"></output></div><input type="range" min="4" max="10" step="1" value="7" id="${id}-batch" data-role="batch"><div class="control-actions"><span>16 examples</span><span>1,024 examples</span></div></div>`:''}<div class="metrics" data-role="metrics"></div><div class="formula-chip" data-role="formula"></div><p class="panel-note" data-role="note" aria-live="polite"></p>`;
      if(showBatch)$('batch').addEventListener('input',e=>{state.batch=2**Number(e.target.value);refreshEfficiency(ctx);});
      refreshEfficiency(ctx);
    }else if(kind==='learning'){
      panel.innerHTML=`<div class="panel-head"><h3>Two fully optimized models</h3></div><table class="signal-table"><thead><tr><th scope="col">Signal</th><th scope="col">Fires among positives</th><th scope="col">Fires among negatives</th><th scope="col">Optimized loss</th></tr></thead><tbody data-role="signal-rows"></tbody></table><div data-role="figure"></div><div class="control"><div class="control-top"><label for="${id}-influence">Positive-class influence in the loss</label><output for="${id}-influence" data-role="influence-value"></output></div><input type="range" min="1" max="50" step="0.1" id="${id}-influence" data-role="influence" value="${state.prior*100}"><div class="control-actions"><button type="button" data-role="natural" class="preset">Natural: 1%</button><button type="button" data-role="equal" class="preset">Balanced: 50%</button></div></div><div class="winner"><span>Lower-loss model</span><strong data-role="winner"></strong></div><p class="panel-note" data-role="note" aria-live="polite"></p>`;
      $('influence').addEventListener('input',e=>{state.prior=Number(e.target.value)/100;refreshLearning(ctx);});
      $('natural').addEventListener('click',()=>{state.prior=.01;$('influence').value=1;refreshLearning(ctx);});
      $('equal').addEventListener('click',()=>{state.prior=.5;$('influence').value=50;refreshLearning(ctx);});
      refreshLearning(ctx);
    }else if(kind==='rank'){
      panel.innerHTML=`<div class="panel-head"><h3>A ranking change survives correction</h3></div><div class="legend"><span><i class="swatch"></i>A: narrow signal</span><span><i class="swatch orange"></i>B: broad signal</span></div><div data-role="figure"></div><table class="comparison"><thead><tr><th scope="col">Corrected population scores</th><th scope="col">Model A</th><th scope="col">Model B</th></tr></thead><tbody data-role="rank-rows"></tbody></table><p class="panel-note">A ranks the A-only case higher. B ranks the B-only case higher. An intercept or threshold change cannot cause that reversal.</p>`;
      refreshRank(ctx);
    }else if(kind==='trade'){
      panel.innerHTML=`<div class="panel-head"><h3>Evaluate at 1% prevalence</h3></div><table class="comparison"><thead><tr><th scope="col">Higher-score branch flagged</th><th scope="col">Model A</th><th scope="col">Model B</th></tr></thead><tbody data-role="trade-rows"></tbody></table><div data-role="figure"></div><p class="panel-note">B gains recall, but most of its flags are false alarms. Both models are evaluated in the same natural population.</p><p class="micro">Population log loss uses prior-corrected leaf probabilities. Thresholds between the two leaves flag signal = 1.</p>`;
      refreshTrade(ctx);
    }
  });

  function drawCounts({panel,mode}){
    const sample=mode==='sample';
    let c='<text x="48" y="30" class="title">At x = 0 · 10,000 original observations</text>';
    const row=(y,label,negative,showDiscard)=>`<text x="48" y="${y-16}">${label}</text><rect x="48" y="${y}" width="524" height="34" fill="#e7e9ea"/><rect x="48" y="${y}" width="5.24" height="34" fill="${orange}"/><rect x="53.24" y="${y}" width="${524*negative/10000}" height="34" fill="${blue}"/><text x="48" y="${y+61}">100 positives · ${num(negative)} negatives${showDiscard?' retained':''}</text>`;
    c+=row(80,'Population',9900,false);
    if(sample)c+=row(205,'Retained sample · same count scale',990,true);
    panel.innerHTML=`<div class="panel-head"><h3>${sample?'The positives stay. The denominator shrinks.':'Count first, then divide'}</h3></div><div class="legend"><span><i class="swatch orange"></i>Positive</span><span><i class="swatch"></i>Negative</span>${sample?'<span>Pale = discarded</span>':''}</div>${svg(c,sample?'At x equals zero: original 100 positives, 9900 negatives. Sample retains 100 positives and 990 negatives in expectation.':'At x equals zero: 100 positives and 9900 negatives.',sample?300:180)}<table class="count-table"><thead><tr><th scope="col">${sample?'At x = 0':'Population at x = 0'}</th><th scope="col">Positives</th><th scope="col">Negatives</th><th scope="col">Probability</th></tr></thead><tbody><tr><td>Population</td><td>100</td><td>9,900</td><td>1%</td></tr>${sample?'<tr><td>Retained sample</td><td>100</td><td>990</td><td>9.17%</td></tr>':''}</tbody></table><p class="panel-note">${sample?'The sample bar uses the original count scale: 8,910 negatives have been removed. Random retention gives these counts in expectation.':'This is a group conditional on x = 0, rather than a picture of how often each x occurs in the full population.'}</p>`;
  }

  function refreshLogistic({state,$,mode}){
    const s=state.s, log=state.scale==='log', showSample=mode!=='population', corrected=mode==='correction'||mode==='interactive';
    if(mode==='interactive'){['prob','log'].forEach(v=>$(v).setAttribute('aria-pressed',String(v===state.scale)));$('retained').textContent=pct(s,s<.01?2:1);}
    const fx=x=>62+x/2*523, fy=y=>log?248-(y+5)/17*214:248-y*214;
    let c=chartAxes(log?'Log odds':'Event probability','Predictor x',[0,1,2].map(v=>({v,label:v})),(log?[-4,0,4,8,12]:[0,.25,.5,.75,1]).map(v=>({v,label:log?v:pct(v,0)})),fx,fy);
    const p=x=>M.sigmoid(Math.log(99)*(x-1)), q=x=>M.sampledProbability(p(x),1,s);
    const path=f=>linePath(Array.from({length:101},(_,j)=>{const x=j/50;const y=f(x);return[fx(x),fy(log?M.logit(y):y)];}));
    c+=`<path d="${path(p)}" fill="none" stroke="${blue}" stroke-width="3"/>`;
    if(showSample)c+=`<path d="${path(q)}" fill="none" stroke="${orange}" stroke-width="3"/>`;
    if(corrected)c+=`<path d="${path(x=>M.correctedProbability(q(x),1,s))}" fill="none" stroke="${teal}" stroke-width="3" stroke-dasharray="7 5"/>`;
    [0,1,2].forEach(x=>{c+=`<circle cx="${fx(x)}" cy="${fy(log?M.logit(p(x)):p(x))}" r="5" fill="${corrected?teal:blue}"/>`;if(showSample)c+=`<circle cx="${fx(x)}" cy="${fy(log?M.logit(q(x)):q(x))}" r="5" fill="${orange}"/>`;});
    if(log&&s<1&&showSample){
      const x=fx(1)+18,top=fy(-Math.log(s)),bottom=fy(0);
      c+=`<path d="M${x-5},${top} H${x+5} M${x},${top} V${bottom} M${x-5},${bottom} H${x+5}" fill="none" stroke="${corrected?teal:orange}" stroke-width="2"/><text x="${fx(1)-12}" y="${top-14}" text-anchor="end" style="fill:${corrected?teal:orange}">${corrected?'correction −':'shift +'}${(-Math.log(s)).toFixed(2)}</text>`;
    }
    $('figure').innerHTML=svg(c,`${log?'Log odds':'Probabilities'} at x 0, 1, 2. ${showSample?`Sampled log odds increase by ${(-Math.log(s)).toFixed(2)}.`:''} ${corrected?'Corrected curve matches population.':''}`);
    const rho=M.sampledProbability(samplePi,1,s);
    $('metrics').innerHTML=mode==='population'?`<div><span>Probability at x = 0</span><strong>1%</strong></div><div><span>Probability at x = 1</span><strong>50%</strong></div><div><span>Probability at x = 2</span><strong>99%</strong></div>`:`<div><span>Negative retention</span><strong>${pct(s,s<.01?2:1)}</strong></div><div><span>Added log odds</span><strong>+${(-Math.log(s)).toFixed(2)}</strong></div><div><span>Sample event rate, overall</span><strong>${pct(rho)}</strong></div>`;
    $('formula').textContent=mode==='population'?'The overall event rate is 2.47%; these three x values are not equally common.':corrected?`Sampled intercept + log(s) = −log(99) ≈ −4.60`:'Sampled intercept = −log(99) + log(10) ≈ −2.29';
    $('note').textContent=mode==='population'?'A probability is conditional on x. A rare event overall can still be common at some values of x.':corrected?`At x = 0: ${pct(q(0),2)} sampled → ${pct(M.correctedProbability(q(0),1,s),0)} corrected. The correction preserves score order.`:'Every x gets the same addition. The orange line has the same slope as the blue line.';
  }
  function refreshEfficiency({state,$}){const e=M.efficiency(.01,1e6,state.batch);if($('batch-value'))$('batch-value').textContent=num(state.batch);const i=state.step;let c='';
    if(i===6){c=`<text x="62" y="28" class="title">Examples processed per pass</text>`;[['Full training set',1000000,blue,75],['Balanced subset',20000,orange,165]].forEach(([label,n,color,y])=>{c+=`<text x="62" y="${y-12}">${label}</text><rect x="62" y="${y}" width="${523*n/1e6}" height="24" fill="${color}"/><text x="62" y="${y+49}">${num(n)} examples</text>`;});c+=`<text x="62" y="270">The same 10,000 positive cases in both sets.</text>`;
    }else{const fx=x=>62+x*523; c=chartAxes('Expected batch composition','Share of the batch',[0,.25,.5,.75,1].map(v=>({v,label:pct(v,0)})),[],fx,x=>x);[['Natural sampling',.01,84],['Balanced sampling',.5,170]].forEach(([label,p,y])=>{c+=`<text x="62" y="${y-15}">${label}: ${(state.batch*p).toFixed(p===.01?2:0)} positive visits</text><rect x="62" y="${y}" width="523" height="32" fill="#dbe4ea"/><rect x="62" y="${y}" width="${523*p}" height="32" fill="${orange}"/>`;});c+=`<text x="62" y="242">Orange = positive · pale blue = negative</text>`;}
    $('figure').innerHTML=svg(c,i===6?'Full set: one million examples; balanced subset: twenty thousand; ten thousand positives in both.':`Batch of ${state.batch}: natural expected positives ${e.expectedMinorityInBatch}; balanced ${e.balancedExpectedMinorityInBatch}.`);
    $('metrics').innerHTML=i===6?`<div><span>Positives kept</span><strong>10,000</strong></div><div><span>Negatives retained</span><strong>10,000</strong></div><div><span>Fewer visits per pass</span><strong>50×</strong></div>`:`<div><span>Positive visits, natural</span><strong>${e.expectedMinorityInBatch.toFixed(2)}</strong></div><div><span>Positive visits, balanced</span><strong>${num(e.balancedExpectedMinorityInBatch)}</strong></div><div><span>Natural batch has no positives</span><strong>${pct(e.zeroMinorityProbability)}</strong></div>`;
    $('formula').textContent=i===6?'1,000,000 / 20,000 = 50 times fewer example visits':i===8?'Preserve natural loss: positive weight 0.02 · negative weight 1.98':`P(no positive in a natural batch) = 0.99^${state.batch} = ${pct(e.zeroMinorityProbability)}`;
    $('note').textContent=i===6?'50× fewer visits per pass is not a guarantee of equal statistical performance.':i===8?'Unweighted balanced sampling changes the objective. Importance weights can restore it in expectation.':'Independent draws from a 50/50 mixture almost never give an all-negative batch. Exact stratified batches can guarantee class counts.';
  }
  function refreshLearning({state,$}){const q=state.prior,a=M.stump('A',q),b=M.stump('B',q),win=a.logloss<b.logloss?'A':'B';$('influence-value').textContent=pct(q);$('signal-rows').innerHTML=[a,b].map((m,j)=>`<tr class="${win===(j?'B':'A')?'selected':''}"><td>${j?'B · broad':'A · narrow'}</td><td>${pct(m.tpr,0)}</td><td>${pct(m.fpr,j?0:1)}</td><td>${m.logloss.toFixed(4)}</td></tr>`).join('');
    const fx=x=>62+(x-.01)/.49*523,fy=y=>168-y/.52*134;let c=chartAxes('Optimized log loss (nats)','Positive-class influence',[.01,.1,.25,.5].map(v=>({v,label:pct(v,0)})),[0,.2,.4].map(v=>({v,label:v.toFixed(1)})),fx,fy,220);
    ['A','B'].forEach((sig,j)=>{const pts=Array.from({length:101},(_,k)=>{const p=.01+.49*k/100;return[fx(p),fy(M.stump(sig,p).logloss)];});c+=`<path d="${linePath(pts)}" fill="none" stroke="${j?orange:blue}" stroke-width="3"/><circle cx="${fx(q)}" cy="${fy(j?b.logloss:a.logloss)}" r="6" fill="${j?orange:blue}"/>`;});c+=`<line x1="${fx(q)}" x2="${fx(q)}" y1="31" y2="168" stroke="#87939a" stroke-dasharray="4 4"/><text x="500" y="26" style="fill:${blue}">A</text><text x="542" y="26" style="fill:${orange}">B</text>`;
    $('figure').innerHTML=svg(c,`At ${pct(q)} positive influence, optimized A loss ${a.logloss.toFixed(4)}, B loss ${b.logloss.toFixed(4)}; model ${win} wins.`,220);$('winner').textContent=`Use signal ${win}`;
    $('note').textContent=`At ${pct(q)} influence, ${win==='A'?'the narrow A':'the broad B'} signal earns the single available slot. Each leaf probability is optimized; more iterations cannot change this optimum.`;
  }
  function refreshRank({$}){const fx=x=>62+x*523,fy=y=>248-y*214;let c=chartAxes('True-positive rate','False-positive rate',[0,.1,.5,1].map(v=>({v,label:pct(v,0)})),[0,.5,1].map(v=>({v,label:pct(v,0)})),fx,fy);c+=`<path d="M62,248 L585,34" stroke="#cbd0d3" stroke-dasharray="4 5" fill="none"/>`;['A','B'].forEach((s,j)=>{const m=M.stump(s,.01);c+=`<path d="${linePath([[fx(0),fy(0)],[fx(m.fpr),fy(m.tpr)],[fx(1),fy(1)]])}" stroke="${j?orange:blue}" stroke-width="3" fill="none"/><circle cx="${fx(m.fpr)}" cy="${fy(m.tpr)}" r="5" fill="${j?orange:blue}"/>`;});$('figure').innerHTML=svg(c,'ROC curves cross; AUC A .7495, B .90. Prior correction leaves both curves unchanged.');const a=M.stump('A',.01),b=M.stump('B',.01);$('rank-rows').innerHTML=`<tr><td>A = 1, B = 0</td><td>${pct(a.highProb,2)} <b>↑</b></td><td>${pct(b.lowProb,2)}</td></tr><tr><td>A = 0, B = 1</td><td>${pct(a.lowProb,2)}</td><td>${pct(b.highProb,2)} <b>↑</b></td></tr><tr><td>AUC</td><td>${a.auc.toFixed(4)}</td><td>${b.auc.toFixed(2)}</td></tr>`;}
  function refreshTrade({$}){const a=M.stump('A',.01),b=M.stump('B',.01);$('trade-rows').innerHTML=[['Recall',pct(a.tpr,0),pct(b.tpr,0)],['False-positive rate',pct(a.fpr,1),pct(b.fpr,0)],['Precision',pct(a.precision),pct(b.precision)],['Best balanced accuracy',pct(a.auc),pct(b.auc)],['AUC',a.auc.toFixed(4),b.auc.toFixed(2)],['Population log loss ↓',a.populationLogloss.toFixed(5),b.populationLogloss.toFixed(5)]].map(r=>`<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td></tr>`).join('');let c='<text x="62" y="28" class="title">Expected flags per 10,000 observations</text>';[[a,'A',blue,78],[b,'B',orange,168]].forEach(([m,s,color,y])=>{const tp=100*m.tpr,fp=9900*m.fpr; c+=`<text x="62" y="${y-12}">Model ${s}: ${num(tp)} true positives, ${fp.toFixed(s==='A'?1:0)} false positives</text><rect x="62" y="${y}" width="${523*(tp+fp)/1080}" height="26" fill="#dbe4ea"/><rect x="62" y="${y}" width="${523*tp/1080}" height="26" fill="${color}"/>`;});c+='<text x="62" y="257">Color = true positives · pale = false positives</text>';$('figure').innerHTML=svg(c,'A flags 50 positives and 9.9 negatives. B flags 90 positives and 990 negatives per ten thousand.');}
})();
