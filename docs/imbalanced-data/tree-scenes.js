(() => {
  'use strict';
  const M = window.TreeBalance;
  const colors = window.TreeColors;
  const clamp = x => Math.max(0, Math.min(1, x));
  const ease = x => {x = clamp(x); return x*x*(3-2*x);};
  const pct = x => `${(100*x).toFixed(1)}%`;
  const text = (x,y,s,fill=colors.ink,anchor='start',size=18) => `<text x="${x}" y="${y}" fill="${fill}" text-anchor="${anchor}" style="font-size:${size}px">${s}</text>`;
  const line = (x,y,x2,y2,fill=colors.line,extra='') => `<line x1="${x}" y1="${y}" x2="${x2}" y2="${y2}" stroke="${fill}" ${extra}/>`;
  const svg = (body,label,width=800,height=320) => `<svg class="tree-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${label}">${body}</svg>`;

  function gainFigure(p, cutoff = null, narrow = false) {
    const width = narrow ? 360 : 800;
    const left = narrow ? 50 : 78, right = width - (narrow ? 24 : 45);
    // Same normalization for both candidates: their order cannot change.
    const max = cutoff === null ? 1 : .03;
    const x = v => left + v/max * (right-left);
    let body = text(left,25,cutoff === null ? 'Share of parent entropy removed' : 'Entropy gain at 1% positives (nats)',colors.muted,'start',narrow?15:19);
    (cutoff === null ? [0,.25,.5,.75,1] : [0,.01,.02,.03]).forEach(v => {
      body += line(x(v),60,x(v),232) + text(x(v),256,cutoff === null ? `${100*v}%` : v.toFixed(3),colors.muted,'middle',narrow?13:17);
    });
    ['A','B'].forEach((name,i) => {
      const r = M.evaluate(name,p), y = 89 + i*95;
      const v = cutoff === null ? r.relativeGain : r.gain;
      const winner = M.choose(p,cutoff??0) === name;
      body += `<rect x="${left}" y="${y}" width="${x(v)-left}" height="24" fill="${colors[name]}" data-split="${name}"/>`;
      body += text(left-17,y+20,name,colors[name],'end',22);
      body += text(left,y-12,`Δ${name} = ${r.gain.toFixed(6)}${winner?' · chosen':''}`,colors[name],'start',narrow?16:21);
    });
    if (cutoff !== null) {
      body += line(x(cutoff),48,x(cutoff),233,colors.ink,'stroke-width="2" stroke-dasharray="5 4"');
      const anchor = cutoff > .015 ? 'end' : 'start';
      body += text(x(cutoff)+(anchor==='end'?-5:5),48,`cutoff ${cutoff.toFixed(4)}`,colors.ink,anchor,narrow?13:17);
    }
    const winner = M.choose(p,cutoff??0);
    body += text(left,298,winner ? `Split ${winner} wins at p = ${pct(p)}` : 'Neither split clears the cutoff',colors.ink,'start',narrow?18:24);
    return svg(body,`Training positives ${pct(p)}. A gain ${M.evaluate('A',p).gain.toFixed(6)}. B gain ${M.evaluate('B',p).gain.toFixed(6)}. ${winner?`Choose ${winner}.`:'Do not split.'}`,width);
  }

  const story = [
    ['Two splits. Four leaf balances.', 'Both candidates split the same cases. The right leaf gets everyone not sent left.'],
    ['Remove negatives. Keep every positive.', 'Randomly thin negatives in every branch. A’s left leaf stays 100% positive; its right leaf becomes more positive as negatives disappear.'],
    ['Now see the same experiment in entropy space.', 'Return to the original 1% positives. Connect the two leaf proportions on H(p); the vertical gap at the parent is the split’s information gain.'],
    ['The same thinning moves the chords.', 'Repeat the removal of negatives. The routing rules stay fixed, but the leaf proportions move and B’s entropy reduction overtakes A’s.'],
    ['A cutoff does not reorder the candidates.', 'Return to 1% positives. A gain cutoff can exclude B while A survives; it cannot reorder the two candidates.'],
    ['Raise the cutoff further.', 'Once the cutoff exceeds A’s gain, neither split is accepted. The cutoff can choose A or stop; rebalancing can make B the best split.']
  ];
  window.ImbalanceScenes.trees = progress => {
    const z = Math.max(0,Math.min(5,progress));
    const stage = Math.round(z);
    const retention = stage === 1 || stage === 3 ? Math.pow(99,-ease((z-(stage-.5))/.5)) : 1;
    const p = 1/(1+99*retention);
    const cutoff = stage === 4 ? .02*ease((z-3.5)/.5) : stage === 5 ? .02+.01*ease((z-4.5)/.5) : null;
    const a=M.evaluate('A',p),b=M.evaluate('B',p);
    const remainingNegatives=(9900*retention).toLocaleString('en-US',{maximumFractionDigits:1});
    const aRightShare=(100*a.pRight).toFixed(3).replace(/(\.\d*?)0+$/,'$1').replace(/\.$/,'');
    return {stage,eyebrow:'04 / Decision trees',title:story[stage][0],description:story[stage][1],
      svg:stage===0?window.TreeLeaves.figure(.01,innerWidth<=650):stage===1?window.TreeRebalance.figure(retention,innerWidth<=650):stage<=3?window.TreeEntropyChords.figure(p,innerWidth<=650):gainFigure(p,cutoff,innerWidth<=650),
      equation:stage===0?'':stage===1?`A right: 50 / (50 + ${remainingNegatives}) = ${aRightShare}%`:stage<=3?'Vertical gap = information gain':'ΔA '+(a.gain>b.gain?'>':'<')+' ΔB · gains fixed; cutoff moves',
      note:stage===0?'Original sample: 100 positives and 9,900 negatives. Scroll to remove negatives.':stage===1?'Positive counts stay fixed. The changing denominator changes each mixed leaf’s positive share. Expected counts shown.':stage>=4?'Root-node example. Hold all other constraints fixed.':'Chord height is the average leaf entropy, weighted by the number of cases in each leaf.'};
  };

  const prior=document.getElementById('tree-prior'), cutoff=document.getElementById('tree-cutoff');
  function updatePrior() {
    const p=Number(prior.value)/100;
    document.getElementById('tree-leaf-chart').innerHTML=window.TreeLeaves.figure(p,innerWidth<=650);
    document.getElementById('tree-chord-chart').innerHTML=window.TreeEntropyChords.figure(p,innerWidth<=650);
    document.getElementById('tree-prior-value').textContent=pct(p);
    document.getElementById('tree-prior-chart').innerHTML=gainFigure(p,null,innerWidth<=650);
    document.getElementById('tree-detail-body').innerHTML=['A','B'].map(name=>{
      const r=M.evaluate(name,p);
      return `<tr><th scope="row" style="color:${colors[name]}">${name}</th><td>${pct(r.q)}</td><td>${pct(r.pLeft)}</td><td>${pct(r.pRight)}</td><td>${r.gain.toFixed(6)}</td></tr>`;
    }).join('');
    document.getElementById('tree-prior-status').textContent=`At ${pct(p)} positives, split ${M.choose(p)} has the larger gain. Parent entropy: ${M.entropy(p).toFixed(6)} nats.`;
  }
  function updateCutoff() {
    const c=Number(cutoff.value);
    document.getElementById('tree-cutoff-value').textContent=c.toFixed(4);
    document.getElementById('tree-cutoff-chart').innerHTML=gainFigure(.01,c,innerWidth<=650);
    const winner=M.choose(.01,c);
    document.getElementById('tree-cutoff-status').textContent=winner?'A remains the best admissible split. No cutoff selects B here.':'The cutoff exceeds both gains: the root remains a leaf.';
  }
  prior.addEventListener('input',updatePrior);cutoff.addEventListener('input',updateCutoff);
  document.querySelectorAll('[data-tree-prior]').forEach(b=>b.addEventListener('click',()=>{prior.value=b.dataset.treePrior;updatePrior();}));
  addEventListener('resize',()=>{updatePrior();updateCutoff();});
  updatePrior();updateCutoff();
})();
