(() => {
  'use strict';
  const M = window.TreeBalance;
  const colors = {A:'#234e70', B:'#a54f32', ink:'#242424', muted:'#606060', line:'#d5d8da'};
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
    const max = cutoff === null ? 1 : .0025;
    const x = v => left + v/max * (right-left);
    let body = text(left,25,cutoff === null ? 'Share of parent Gini removed' : 'Gini decrease at 1% positives',colors.muted,'start',narrow?15:19);
    (cutoff === null ? [0,.25,.5,.75,1] : [0,.001,.002]).forEach(v => {
      body += line(x(v),60,x(v),232) + text(x(v),256,cutoff === null ? `${100*v}%` : v.toFixed(3),colors.muted,'middle',narrow?13:17);
    });
    ['A','B'].forEach((name,i) => {
      const r = M.evaluate(name,p), y = 89 + i*95;
      const v = cutoff === null ? r.relativeGain : r.gain;
      const winner = M.choose(p,cutoff??0) === name;
      body += `<rect x="${left}" y="${y}" width="${x(v)-left}" height="24" fill="${colors[name]}" ${name==='B'?'fill-opacity=".72"':''}/>`;
      body += text(left-17,y+20,name,colors[name],'end',22);
      body += text(left,y-12,`Δ${name} = ${r.gain.toFixed(6)}${winner?' · chosen':''}`,colors[name],'start',narrow?16:21);
    });
    if (cutoff !== null) {
      body += line(x(cutoff),48,x(cutoff),233,colors.ink,'stroke-width="2" stroke-dasharray="5 4"');
      const anchor = cutoff > .0013 ? 'end' : 'start';
      body += text(x(cutoff)+(anchor==='end'?-5:5),48,`cutoff ${cutoff.toFixed(4)}`,colors.ink,anchor,narrow?13:17);
    }
    const winner = M.choose(p,cutoff??0);
    body += text(left,298,winner ? `Split ${winner} wins at p = ${pct(p)}` : 'Neither split clears the cutoff',colors.ink,'start',narrow?18:24);
    return svg(body,`Training positives ${pct(p)}. A gain ${M.evaluate('A',p).gain.toFixed(6)}. B gain ${M.evaluate('B',p).gain.toFixed(6)}. ${winner?`Choose ${winner}.`:'Do not split.'}`,width);
  }

  function candidates(narrow) {
    if(narrow) {
      let body = text(12,27,'Fraction of each class sent left',colors.muted,'start',17);
      [['A','10% of positives','0% of negatives'],['B','90% of positives','10% of negatives']].forEach(([name,pos,neg],i)=>{
        const y=72+i*113;
        body+=text(12,y,`Split ${name}`,colors[name],'start',23)+text(110,y,pos)+text(110,y+31,neg);
        body+=line(12,y+54,348,y+54);
      });
      return svg(body+text(12,313,'Same rules before and after sampling',colors.muted,'start',17),'A sends 10% of positives and no negatives left. B sends 90% of positives and 10% of negatives left.',360,335);
    }
    let body='';
    [['A','10% positives','0% negatives'],['B','90% positives','10% negatives']].forEach(([name,pos,neg],i)=>{
      const cx=200+i*400;
      body += text(cx,28,`Candidate ${name}`,colors[name],'middle',24);
      body += `<rect x="${cx-83}" y="48" width="166" height="45" rx="2" fill="white" stroke="${colors[name]}" stroke-width="2"/>`;
      body += text(cx,77,`Split on X${name}`,colors[name],'middle',21);
      body += line(cx,93,cx-94,161,colors[name])+line(cx,93,cx+94,161,colors[name]);
      body += text(cx-94,188,'Left',colors[name],'middle',21)+text(cx+94,188,'Right',colors[name],'middle',21);
      body += text(cx-94,223,pos,colors.ink,'middle',18)+text(cx-94,250,neg,colors.ink,'middle',18);
      body += text(cx+94,223,'Everyone else',colors.muted,'middle',18);
    });
    return svg(body+text(400,306,'These within-class routing rates stay fixed.',colors.muted,'middle',20),'Two candidate tree splits with fixed class-conditional routing rates.');
  }

  const story = [
    ['Two candidate splits. One root node.', 'Use a new example with Gini impurity. A isolates a small, pure positive group. B separates most positives from most negatives. Both splits are admissible; sampling is random within each class.'],
    ['At 1% positives, the tree chooses A.', 'The tree computes both Gini decreases and takes the larger one. A removes more impurity than B under the original class mixture, even though B captures many more positives.'],
    ['Balancing reverses the ranking.', 'Keep the same split rules and increase the positive share toward 50%. B overtakes A. The preferred split changes even with exact population calculations, so sampling noise is not needed for this result.'],
    ['A cutoff does not reorder the candidates.', 'Return to 1% positives and move min_impurity_decrease. Raising the cutoff can rule out B while A survives. It cannot select B over the larger-gain A when both are otherwise admissible.'],
    ['Raise the cutoff again: no split.', 'Once the cutoff exceeds A’s gain, neither split is accepted. A gain cutoff can choose A or stop here; rebalancing can make B the best split. Other constraints, such as minimum leaf size, can change which candidates are admissible.']
  ];
  window.ImbalanceScenes.trees = progress => {
    const z = Math.max(0,Math.min(4,progress));
    const stage = z < .8 ? 0 : z < 1.15 ? 1 : z < 2.65 ? 2 : z < 3.7 ? 3 : 4;
    const p = stage === 2 ? .01 * Math.pow(50,ease((z-1.15)/.85)) : .01;
    const cutoff = stage >= 3 ? .0022*ease((z-2.65)/1.35) : null;
    const a=M.evaluate('A',p),b=M.evaluate('B',p);
    return {stage,eyebrow:'05 / Decision trees',title:story[stage][0],description:story[stage][1],
      svg:stage===0?candidates(innerWidth<=650):gainFigure(p,cutoff,innerWidth<=650),
      equation:stage===0?'Δ = I(p) − q I(pL) − (1 − q) I(pR)':'ΔA '+(a.gain>b.gain?'>':'<')+' ΔB · '+(stage>=3?'gains fixed; cutoff moves':'same candidates; class balance changes'),
      note:stage===0?'I(p) = 2p(1 − p); q is the share sent left; pL and pR are child positive shares.':stage>=3?'Root-node example. Hold all other constraints fixed.':'Bars divide both gains by the same parent impurity. Values above bars are the raw gains.'};
  };

  const prior=document.getElementById('tree-prior'), cutoff=document.getElementById('tree-cutoff');
  function updatePrior() {
    const p=Number(prior.value)/100;
    document.getElementById('tree-prior-value').textContent=pct(p);
    document.getElementById('tree-prior-chart').innerHTML=gainFigure(p,null,innerWidth<=650);
    document.getElementById('tree-detail-body').innerHTML=['A','B'].map(name=>{
      const r=M.evaluate(name,p);
      return `<tr><th scope="row">${name}</th><td>${pct(r.q)}</td><td>${pct(r.pLeft)}</td><td>${pct(r.pRight)}</td><td>${r.gain.toFixed(6)}</td></tr>`;
    }).join('');
    document.getElementById('tree-prior-status').textContent=`At ${pct(p)} positives, split ${M.choose(p)} has the larger gain. Parent Gini: ${M.gini(p).toFixed(6)}.`;
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
