(() => {
  'use strict';
  const M = window.MisspecModel;
  const E = M.example();
  const C = {population:'#234e70',sample:'#a54f32',corrected:'#007f73',ink:'#242424',muted:'#606060',grid:'#e4e7e9'};
  const clamp = x => Math.max(0,Math.min(1,x));
  const smooth = x => {x=clamp(x);return x*x*(3-2*x);};
  const mix = (a,b,t) => a+(b-a)*t;
  const text = (x,y,s,color=C.ink,size=18,anchor='start',extra='') => `<text x="${x}" y="${y}" fill="${color}" text-anchor="${anchor}" style="font-size:${size}px" ${extra}>${s}</text>`;
  const path = points => points.map((p,i)=>`${i?'L':'M'}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(' ');
  const stories = [
    ['Now the true log odds are curved.','A straight-line model can only approximate this curve. The sampling identity still holds.'],
    ['Fit a line to the population.','Minimize expected logistic log loss. This is the best line within a model that has the wrong shape.'],
    ['Sampling changes the fitted slope.','The true curve shifts by a constant. Refitting the straight-line model changes its slope.'],
    ['Correction cannot restore the slope.','Moving the sampled fit down leaves its new slope unchanged. The original fit does not return.']
  ];
  function figure(z) {
    const narrow=innerWidth<=650,W=narrow?380:900,H=420;
    const L=narrow?45:78,R=narrow?355:850,top=43,bottom=278;
    const xs=E.rows.map(r=>r.x),xmin=Math.min(...xs),xmax=Math.max(...xs);
    const shift=-Math.log(E.spec.retention);
    const values=E.rows.flatMap(r=>[r.trueLogit,r.trueLogit+shift,M.logOdds(E.naturalFit,r.x),M.logOdds(E.sampledFit,r.x),M.logOdds(E.correctedFit,r.x)]);
    const lo=Math.floor(Math.min(...values))-1,hi=Math.ceil(Math.max(...values))+1;
    const fx=x=>L+(R-L)*(x-xmin)/(xmax-xmin),fy=y=>bottom-(bottom-top)*(y-lo)/(hi-lo);
    let c=text(L,21,'Log odds',C.muted,narrow?14:18);
    for(let y=Math.ceil(lo/2)*2;y<=hi;y+=2)c+=`<line x1="${L}" y1="${fy(y)}" x2="${R}" y2="${fy(y)}" stroke="${C.grid}"/>`+text(L-10,fy(y)+5,y,C.muted,narrow?12:16,'end');
    xs.forEach(x=>{c+=text(fx(x),bottom+24,x,C.muted,narrow?13:17,'middle');});
    c+=text(R,bottom+45,'Predictor x',C.muted,narrow?13:17,'end');
    const curve=(offset,color,opacity=1,dash='')=>{
      const points=Array.from({length:121},(_,i)=>{const x=xmin+(xmax-xmin)*i/120;return[fx(x),fy(M.trueLogit(x,E.spec)+offset)];});
      return `<path d="${path(points)}" fill="none" stroke="${color}" stroke-width="2.5" stroke-dasharray="${dash}" opacity="${opacity}"/>`;
    };
    const line=(f,color,opacity=1,dash='')=>`<path d="${path([[fx(xmin),fy(M.logOdds(f,xmin))],[fx(xmax),fy(M.logOdds(f,xmax))]])}" fill="none" stroke="${color}" stroke-width="3.7" stroke-dasharray="${dash}" opacity="${opacity}"/>`;
    const sampleT=smooth((z-1.5)*2),correctT=smooth((z-2.5)*2);
    c+=curve(0,C.population,.75,'4 4');
    E.rows.forEach(r=>c+=`<circle cx="${fx(r.x)}" cy="${fy(r.trueLogit)}" r="4" fill="${C.population}"/>`);
    if(z>.5)c+=line(E.naturalFit,C.population,smooth((z-.5)*2));
    if(z>1.5){
      c+=curve(shift*sampleT,C.sample,.65*sampleT,'4 4');
      const f={intercept:mix(E.naturalFit.intercept,E.sampledFit.intercept,sampleT),slope:mix(E.naturalFit.slope,E.sampledFit.slope,sampleT)};
      c+=line(f,C.sample,sampleT);
    }
    if(z>2.5){
      const f={intercept:mix(E.sampledFit.intercept,E.correctedFit.intercept,correctT),slope:E.sampledFit.slope};
      c+=line(f,C.corrected,correctT);
    }
    // Keep all fit values on the same scale, including after correction.
    const stage=Math.round(z),cols=narrow?[22,138,253]:[90,357,635];
    const labels=['Original fit','Sampled fit','Corrected fit'],fits=[E.naturalFit,E.sampledFit,E.correctedFit];
    labels.forEach((label,i)=>{
      const visible=i===0?stage>=1:i===1?stage>=2:stage>=3;
      if(!visible)return;
      const color=[C.population,C.sample,C.corrected][i];
      c+=text(cols[i],352,label,color,narrow?13:19);
      c+=text(cols[i],377,`slope ${fits[i].slope.toFixed(3)}`,color,narrow?14:23);
      c+=text(cols[i],401,`intercept ${fits[i].intercept.toFixed(3)}`,C.muted,narrow?11:17);
    });
    if(stage===0)c+=text(W/2,365,'Dashed curve = truth; points = exact group probabilities',C.muted,narrow?12:18,'middle');
    else c+=text(L,325,narrow?'Dashed = truth · Solid = fitted model':'Dashed = true relationship · Solid = fitted logistic model',C.muted,narrow?12:17);
    return `<svg class="scene-chart model-chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${stories[stage][0]} Original slope ${E.naturalFit.slope.toFixed(3)}; sampled and corrected slope ${E.sampledFit.slope.toFixed(3)}.">${c}</svg>`;
  }
  window.ImbalanceScenes.misspecification = progress => {
    const z=clamp(progress/3)*3,stage=Math.round(z);
    return {stage,eyebrow:'02 / When the model is wrong',title:stories[stage][0],description:stories[stage][1],svg:figure(z),
      equation:stage===0?'True sampled log odds = true population log odds − log(s)':stage===1?`Population fit: slope ${E.naturalFit.slope.toFixed(3)}`:stage===2?`Sampled fit: slope ${E.sampledFit.slope.toFixed(3)}`:`Corrected fit: slope ${E.correctedFit.slope.toFixed(3)} ≠ ${E.naturalFit.slope.toFixed(3)}`,
      note:stage===3?`Population log loss: ${M.logLoss(E.naturalRows,E.naturalFit).toFixed(5)} original fit; ${M.logLoss(E.naturalRows,E.correctedFit).toFixed(5)} corrected sample fit. Lower is better.`:'Exact expected-loss optimization; no sampling noise or regularization.'};
  };
  const target=document.getElementById('misspec-details');
  if(target){
    const pct=v=>(100*v).toFixed(2)+'%';
    target.innerHTML=`<p>True log odds: ${E.spec.intercept} + ${E.spec.linear}x + ${E.spec.quadratic}x². Overall positive share: ${pct(E.rows.reduce((s,r)=>s+r.mass*r.p,0))}. The sampled fit keeps all positives and ${pct(E.spec.retention)} of negatives.</p><div class="table-scroll"><table class="comparison"><caption>Exact population at each predictor value</caption><thead><tr><th scope="col">x</th><th scope="col">Population share</th><th scope="col">Positive probability</th><th scope="col">True log odds</th></tr></thead><tbody>${E.rows.map(r=>`<tr><th scope="row">${r.x}</th><td>${pct(r.mass)}</td><td>${pct(r.p)}</td><td>${r.trueLogit.toFixed(3)}</td></tr>`).join('')}</tbody></table></div><div class="table-scroll"><table class="comparison"><caption>Fits evaluated in the original population</caption><thead><tr><th scope="col">Fit</th><th scope="col">Intercept</th><th scope="col">Slope</th><th scope="col">Population log loss</th></tr></thead><tbody>${[['Original population',E.naturalFit],['Sample, uncorrected',E.sampledFit],['Sample, prior corrected',E.correctedFit],['Sample, inverse-selection weighted',E.weightedFit]].map(([name,f])=>`<tr><th scope="row">${name}</th><td>${f.intercept.toFixed(4)}</td><td>${f.slope.toFixed(4)}</td><td>${M.logLoss(E.naturalRows,f).toFixed(6)}</td></tr>`).join('')}</tbody></table></div>`;
  }
})();
