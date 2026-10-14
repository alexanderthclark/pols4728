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
    ['What if the log odds are curved?','The full-data fit is a straight-line approximation. The model cannot reproduce the true curve.'],
    ['A correction can leave the wrong slope.','The true curve still shifts by a constant. But refitting changes the slope, and an intercept correction cannot undo that.']
  ];
  function figure(z) {
    const narrow=innerWidth<=650,W=narrow?380:900,H=420;
    const L=narrow?45:78,R=narrow?355:850,top=43,bottom=278;
    const xs=E.rows.map(r=>r.x),xmin=Math.min(...xs),xmax=Math.max(...xs);
    const values=E.rows.flatMap(r=>[r.trueLogit,M.logOdds(E.naturalFit,r.x),M.logOdds(E.sampledFit,r.x),M.logOdds(E.correctedFit,r.x)]);
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
    const sampleT=smooth((z-.5)*4),correctT=smooth((z-.75)*4);
    const sampleFit={intercept:mix(E.naturalFit.intercept,E.sampledFit.intercept,sampleT),slope:mix(E.naturalFit.slope,E.sampledFit.slope,sampleT)};
    const correctedFit={intercept:mix(E.sampledFit.intercept,E.correctedFit.intercept,correctT),slope:E.sampledFit.slope};
    c+=curve(0,C.population,.75,'4 4');
    E.rows.forEach(r=>c+=`<circle cx="${fx(r.x)}" cy="${fy(r.trueLogit)}" r="4" fill="${C.population}"/>`);
    c+=line(E.naturalFit,C.population);
    if(z>.5){
      c+=line(sampleFit,C.sample,sampleT);
    }
    if(z>.75){
      c+=line(correctedFit,C.corrected,correctT);
    }
    // Keep all fit values on the same scale, including after correction.
    const stage=Math.round(z),cols=narrow?[22,138,253]:[90,357,635];
    const labels=['Full-data fit','Sampled fit','Corrected fit'],fits=[E.naturalFit,sampleFit,correctedFit];
    labels.forEach((label,i)=>{
      const opacity=i===0?1:i===1?sampleT:correctT;
      if(opacity===0)return;
      const color=[C.population,C.sample,C.corrected][i];
      c+=`<g opacity="${opacity}">`;
      c+=text(cols[i],352,label,color,narrow?13:19);
      c+=text(cols[i],377,`slope ${fits[i].slope.toFixed(3)}`,color,narrow?14:23);
      c+=text(cols[i],401,`intercept ${fits[i].intercept.toFixed(3)}`,C.muted,narrow?11:17);
      c+='</g>';
    });
    c+=text(L,325,narrow?'Dashed = truth · Solid = fitted model':'Dashed = true relationship · Solid = fitted logistic model',C.muted,narrow?12:17);
    const fitDescription=`Full-data slope ${E.naturalFit.slope.toFixed(3)}.${sampleT>0?` Sampled slope ${sampleFit.slope.toFixed(3)}.`:''}${correctT>0?` Corrected slope ${correctedFit.slope.toFixed(3)}.`:''}`;
    return `<svg class="scene-chart model-chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${stories[stage][0]} ${fitDescription}">${c}</svg>`;
  }
  window.ImbalanceScenes.misspecification = progress => {
    const z=clamp(progress),stage=Math.round(z);
    return {stage,eyebrow:'02 / When the model is wrong',title:stories[stage][0],description:stories[stage][1],svg:figure(z),
      equation:stage===0?`Full-data fit: slope ${E.naturalFit.slope.toFixed(3)}`:'Downsampling changes the slope; correction shifts the intercept.',
      note:stage===1?`Final population log loss: ${M.logLoss(E.naturalRows,E.naturalFit).toFixed(5)} full-data fit; ${M.logLoss(E.naturalRows,E.correctedFit).toFixed(5)} corrected sample fit. Lower is better.`:'Exact expected-loss optimization; no sampling noise or regularization.'};
  };
  const target=document.getElementById('misspec-details');
  if(target){
    const pct=v=>(100*v).toFixed(2)+'%';
    target.innerHTML=`<p>True log odds: ${E.spec.intercept} + ${E.spec.linear}x + ${E.spec.quadratic}x². Overall positive share: ${pct(E.rows.reduce((s,r)=>s+r.mass*r.p,0))}. The sampled fit keeps all positives and ${pct(E.spec.retention)} of negatives.</p><div class="table-scroll"><table class="comparison"><caption>Exact population at each predictor value</caption><thead><tr><th scope="col">x</th><th scope="col">Population share</th><th scope="col">Positive probability</th><th scope="col">True log odds</th></tr></thead><tbody>${E.rows.map(r=>`<tr><th scope="row">${r.x}</th><td>${pct(r.mass)}</td><td>${pct(r.p)}</td><td>${r.trueLogit.toFixed(3)}</td></tr>`).join('')}</tbody></table></div><div class="table-scroll"><table class="comparison"><caption>Fits evaluated in the original population</caption><thead><tr><th scope="col">Fit</th><th scope="col">Intercept</th><th scope="col">Slope</th><th scope="col">Population log loss</th></tr></thead><tbody>${[['Original population',E.naturalFit],['Sample, uncorrected',E.sampledFit],['Sample, prior corrected',E.correctedFit],['Sample, inverse-selection weighted',E.weightedFit]].map(([name,f])=>`<tr><th scope="row">${name}</th><td>${f.intercept.toFixed(4)}</td><td>${f.slope.toFixed(4)}</td><td>${M.logLoss(E.naturalRows,f).toFixed(6)}</td></tr>`).join('')}</tbody></table></div>`;
  }
})();
