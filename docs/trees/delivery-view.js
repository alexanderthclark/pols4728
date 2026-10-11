(function (root, factory) {
  'use strict';
  const api = factory(root.DeliveryExample || (typeof require === 'function' ? require('./delivery.js') : null));
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.DeliveryView = api;
})(typeof window !== 'undefined' ? window : globalThis, function (D) {
  'use strict';
  const C = { ink:'#242424', muted:'#606060', blue:'#234E70', rust:'#A54F32', teal:'#007F73', grid:'#e8e8e8' };
  const topics = ['Start with one prediction', 'Fit the first correction', 'Add it to the prediction', 'Fit the remaining mistakes', 'Add a second correction', 'Every prediction is a sum', 'Keep correcting the remaining errors'];
  const captions = [
    'Observed delivery times and the starting prediction appear above; their residuals appear below.',
    'Measure the errors, form the residuals, split them, and then calculate each leaf’s mean.',
    'The fitted correction stays fixed above as it moves the prediction below. Observed outcomes do not move.',
    'Subtract the first correction from the old residuals before fitting the next tree.',
    'The second tree’s right leaf moves both existing plateaus together. Some errors grow while total squared error falls.',
    'Move the guide across the three aligned plots. Each prediction adds the starting mean and both tree contributions.',
    'Continue the same delivery example through eight trees. Each orange stump fits fixed residual targets below and updates the prediction above.'
  ];
  const clamp=(n,lo=0,hi=1)=>Math.max(lo,Math.min(hi,n));
  const fmt=n=>Math.abs(n)<1e-9?'0':String(Number(n.toFixed(2))).replace('-', '−');
  const signed=n=>n>0?'+'+fmt(n):fmt(n);
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const text=(x,y,s,extra='')=>`<text x="${x}" y="${y}" ${extra}>${esc(s)}</text>`;
  const line=(x1,y1,x2,y2,color=C.grid,width=1,extra='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}" ${extra}/>`;
  const circle=(x,y,r,color,extra='')=>`<circle cx="${x}" cy="${y}" r="${r}" fill="${color}" ${extra}/>`;
  const mix=(a,b,t)=>a+(b-a)*t;
  const sub=n=>String(n).split('').map(d=>'₀₁₂₃₄₅₆₇₈₉'[Number(d)]).join('');
  const precise=n=>n===0?'0':String(Number(n.toPrecision(5))).replace('-', '−');
  const mseText=n=>n===0?'0':String(Number(n.toPrecision(3))).replace('-', '−');
  const preciseSigned=n=>n>0?'+'+precise(n):precise(n);
  const smoothstep=t=>t<=1e-9?0:t>=1-1e-9?1:t*t*(3-2*t);
  function continuationState(progress=1) {
    const p=clamp(Number.isFinite(progress)?progress:1);
    if(p===0)return {round:2,amount:1};
    if(p<=.15)return {round:3,amount:0};
    if(p<.4)return {round:3,amount:smoothstep((p-.15)/.25)};
    if(p<=.5)return {round:3,amount:1};
    const elapsed=(p-.5)*10;
    const round=3+clamp(Math.ceil(elapsed-1e-9),1,5);
    return {round,amount:smoothstep(clamp(elapsed-(round-4)))};
  }
  function continuationProgress(round) {
    if(!Number.isInteger(round) || round<2 || round>8)throw new RangeError('Continuation round must be between 2 and 8.');
    return [0,.45,.6,.7,.8,.9,1][round-2];
  }
  const phase=state=>Number.isFinite(state.process)?clamp(state.process,0,3):3;
  function updateAmount(state) {
    if(Number.isFinite(state.process))return clamp(state.process-1);
    return Number.isFinite(state.correction)?clamp(state.correction):1;
  }
  function processLabels(scene) {
    if(scene===1)return ['Measure the errors','Form the residuals','Split the residuals','Fit the leaf means'];
    if(scene===3)return ['Inspect the old residuals','Recompute the residuals','Split the residuals','Fit the leaf means'];
    return ['Inspect the correction','Attach the correction','Add to the prediction','Inspect remaining errors'];
  }
  function candidate(scene,state) {
    if((scene!==1 && scene!==3) || !Number.isInteger(state.candidateIndex) || !D.candidates)return null;
    return D.candidates(scene===1?1:2)[state.candidateIndex] || null;
  }
  function readout(scene,state={}) {
    const p=phase(state), trial=candidate(scene,state);
    if(scene===0)return 'F₀ = mean(y) = 12 min · MSE = 18.75';
    if(scene===1 || scene===3){
      const round=scene===1?1:2, r=trial||D.rounds[round-1];
      if(trial)return `Candidate cut ${fmt(r.threshold)} km · leaf means ${signed(r.left)}, ${signed(r.right)} · residual SSE = ${fmt(r.sse)}`;
      if(p<1)return scene===1?'Each error is y − 12; carry its signed value to the residual plot.':'New residual = old residual − h₁; the prediction F₁ stays fixed.';
      if(p<2)return `Residuals now use F${sub(round-1)}. Next, separate them at ${fmt(r.threshold)} km.`;
      if(p<3)return `Within each leaf, average the residuals to obtain h${sub(round)}.`;
      return `h${sub(round)}: ${signed(r.left)} if x ≤ ${fmt(r.threshold)}; ${signed(r.right)} if x > ${fmt(r.threshold)}`;
    }
    if(scene===5){
      const x=Number.isFinite(state.inspectX)?clamp(state.inspectX,0,13):6;
      const [a,b]=D.rounds.slice(0,2).map(r=>x<=r.threshold?r.left:r.right);
      return `At x = ${fmt(x)} km: 12 ${a<0?'−':'+'} ${fmt(Math.abs(a))} ${b<0?'−':'+'} ${fmt(Math.abs(b))} = ${fmt(D.predict(2,x))} min`;
    }
    if(scene===6){
      const {round,amount}=continuationState(state.continuation);
      const mse=D.mse(D.x.map(x=>D.predict(round,x,amount)));
      const equation=round===2 || amount===0?`F${sub(Math.max(3,round)-1)} · Inspect h${sub(Math.max(3,round))}, fitted to y − F${sub(Math.max(3,round)-1)}`:`F${sub(round)} = F${sub(round-1)} + ${precise(amount)} × h${sub(round)}`;
      return `${equation} · training MSE = ${mseText(mse)}`;
    }
    const round=scene===2?1:2, amount=updateAmount(state), mse=D.mse(D.x.map(x=>D.predict(round,x,amount)));
    return `F${sub(round)} = F${sub(round-1)} + ${fmt(amount)} × h${sub(round)} · MSE = ${fmt(mse)}${scene===4 && p>=3?' (was 3.75)':''}`;
  }
  function render(scene,state={},compact=false) {
    const W=compact?420:780,H=compact?378:550;
    const x=compact?48:76,w=compact?350:652;
    const top={y:compact?45:66,h:compact?95:148,min:-7,max:6};
    const bottom={y:compact?242:340,h:compact?100:164,min:0,max:20};
    const X=v=>x+v/13*w;
    const Y=(v,b)=>b.y+b.h-(v-b.min)/(b.max-b.min)*b.h;
    const round=scene<3?1:2, r=D.rounds[round-1], p=phase(state), amount=updateAmount(state), trial=candidate(scene,state);
    const fitting=scene===1 || scene===3, adding=scene===2 || scene===4;
    let out='';
    function axis(b,ticks,label,showXLabel=true,showTicks=true) {
      let s=`<g data-axis="${label}" data-x-domain="0,13" data-y-domain="${b.min},${b.max}" data-panel="${x},${b.y},${w},${b.h}">`;
      for(const v of ticks){s+=line(x,Y(v,b),x+w,Y(v,b),v===0?'#929292':C.grid,v===0?1.3:1);s+=text(x-9,Y(v,b)+4,fmt(v),'text-anchor="end" class="tick"');}
      s+=line(x,b.y,x,b.y+b.h,'#929292');
      if(showTicks)for(const v of [0,3,6,9,12]){s+=line(X(v),b.y+b.h,X(v),b.y+b.h+4,'#929292');s+=text(X(v),b.y+b.h+18,v,'text-anchor="middle" class="tick"');}
      const lx=compact?12:23,ly=b.y+b.h/2;
      s+=text(lx,ly,label,`text-anchor="middle" transform="rotate(-90 ${lx} ${ly})" class="axis-label"`);
      if(showXLabel)s+=text(x+w,H-3,'Distance (km)','text-anchor="end" class="axis-label"');
      return s+'</g>';
    }
    function step(values,cuts,b,color,extra='',kind='',reveal=1) {
      const bounds=[0,...cuts,13];
      let s=`<g data-curve="${kind}" data-values="${values.join(',')}" data-cuts="${cuts.join(',')}" opacity="${reveal}">`;
      values.forEach((v,i)=>{const mid=(bounds[i]+bounds[i+1])/2;s+=line(X(mix(mid,bounds[i],reveal)),Y(v,b),X(mix(mid,bounds[i+1],reveal)),Y(v,b),color,3,extra);if(i && reveal===1)s+=line(X(bounds[i]),Y(values[i-1],b),X(bounds[i]),Y(v,b),color,1.5,extra);});
      return s+'</g>';
    }
    function predictionPath(count,alpha,color,extra='',kind='') {
      const cuts=[...new Set(D.rounds.slice(0,count).map(q=>q.threshold))].sort((a,b)=>a-b);
      const bounds=[0,...cuts,13],values=bounds.slice(0,-1).map((lo,i)=>D.predict(count,(lo+bounds[i+1])/2,alpha));
      return step(values,cuts,bottom,color,extra,kind);
    }
    function labels(values,cuts,b,color,useSigns=true,negativeBelow=false) {
      const bounds=[0,...cuts,13];
      return values.map((v,i)=>text(X((bounds[i]+bounds[i+1])/2),Y(v,b)+(negativeBelow && v<0?(compact?17:24):-(compact?7:10)),useSigns?signed(v):fmt(v),`text-anchor="middle" class="value" fill="${color}"`)).join('');
    }
    function points(values,b,kind,extra='') {
      return `<g data-points="${kind}" data-values="${values.join(',')}" ${extra}>`+D.x.map((v,i)=>circle(X(v),Y(values[i],b),compact?3.8:4.5,C.ink,'stroke="white" stroke-width="1.2"')).join('')+'</g>';
    }
    function bracket(lo,hi,y,label,opacity=1) {
      return `<g opacity="${opacity}" data-group="${lo},${hi}">`+line(X(lo)+3,y,X(hi)-3,y,C.muted)+line(X(lo)+3,y-4,X(lo)+3,y,C.muted)+line(X(hi)-3,y-4,X(hi)-3,y,C.muted)+text(X((lo+hi)/2),y+(compact?15:20),label,'text-anchor="middle" class="arithmetic"')+'</g>';
    }
    function cutGuide(cut,opacity=1,boxes=[top,bottom]) {
      return `<g data-cut="${cut}" opacity="${opacity}">`+boxes.map(b=>line(X(cut),b.y,X(cut),b.y+b.h,'#999',1,'stroke-dasharray="3 5"')).join('')+text(X(cut)+5,(boxes.includes(top)?top:bottom).y+(boxes.includes(top)?top:bottom).h-7,`${fmt(cut)} km`,'class="cut-label"')+'</g>';
    }
    function arrow(at,from,to,opacity,kind='correction') {
      if(Math.abs(from-to)<1e-8)return '';
      const cx=X(at),sy=Y(from,bottom),ey=Y(to,bottom),direction=Math.sign(ey-sy),head=Math.min(compact?4:5,Math.abs(ey-sy)*.4);
      return `<g data-arrow="${kind}" data-x="${at}" data-from="${from}" data-to="${to}" opacity="${opacity}">`+line(cx,sy,cx,ey,C.rust,2)+`<path d="M${cx-head},${ey-direction*head} L${cx},${ey} L${cx+head},${ey-direction*head}" fill="none" stroke="${C.rust}" stroke-width="2"/>`+text(cx+8,(sy+ey)/2+4,signed(to-from),'class="arrow-label"')+'</g>';
    }
    function residualStems(predictions,emphasis=0) {
      return `<g data-errors="remaining" data-predictions="${predictions.join(',')}">`+D.x.map((v,i)=>line(X(v),Y(D.y[i],bottom),X(v),Y(predictions[i],bottom),scene===4 && i>=5 && emphasis>0?C.rust:'#b0b0b0',scene===4 && i>=5 && emphasis>0?2.8:1.6)).join('')+'</g>';
    }
    function wrap(body) {
      return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" fill="${C.ink}" role="img" aria-labelledby="delivery-title delivery-description" data-process="${p}"><title id="delivery-title">${esc(topics[scene])}</title><desc id="delivery-description">${esc(captions[scene]+' '+readout(scene,state))}</desc><style>text{font-family:Georgia,serif;font-size:${compact?14:18}px}.panel-title{font-size:${compact?16:21}px}.tick,.key,.cut-label,.axis-label,.arrow-label,.arithmetic{font-family:Arial,sans-serif;font-size:${compact?12:14}px}.tick,.axis-label,.key{fill:${C.muted}}.cut-label,.value,.arrow-label,.arithmetic{paint-order:stroke;stroke:white;stroke-width:5px;stroke-linejoin:round}.value{font-size:${compact?15:20}px}.arrow-label{fill:${C.rust}}.annotation{font-size:${compact?13:17}px}</style>${body}</svg>`;
    }
    if(scene===0){
      const observedPanel={...bottom,y:top.y},residualPanel={...top,y:bottom.y};
      out+=text(x,compact?18:27,'Observed times and starting prediction','class="panel-title"');
      out+=circle(x+3,compact?32:46,3.5,C.ink)+text(x+13,compact?36:50,'Observed y','class="key"');
      const keyX=x+(compact?145:220),keyY=compact?32:46;
      out+=line(keyX,keyY,keyX+22,keyY,C.blue,3)+text(keyX+30,keyY+4,'Mean F₀ = 12','class="key"');
      out+=text(x,compact?207:288,'Residuals from the starting prediction','class="panel-title"');
      out+=circle(x+3,compact?225:316,3.5,C.ink)+text(x+13,compact?229:320,'Residual y − 12','class="key"');
      out+=axis(observedPanel,[0,5,10,15,20],'Time (min)',false)+axis(residualPanel,[-6,-3,0,3,6],'Residual (min)');
      out+=`<g data-errors="remaining" data-predictions="${D.y.map(()=>D.baseline).join(',')}">`;
      D.x.forEach((v,i)=>{out+=line(X(v),Y(D.y[i],observedPanel),X(v),Y(D.baseline,observedPanel),'#b0b0b0',1.6);});
      out+='</g>';
      out+=step([D.baseline],[],observedPanel,C.blue,'','prediction');
      out+=points(D.y,observedPanel,'observations');
      out+=points(D.rounds[0].residual,residualPanel,'residuals');
      out+=text(X(11),Y(D.baseline,observedPanel)-10,fmt(D.baseline),`class="value" fill="${C.blue}" text-anchor="middle"`);
      return wrap(out);
    }
    if(scene===6){
      const continued=continuationState(state.continuation),count=continued.round,alpha=continued.amount;
      const shown=D.x.map(v=>D.predict(count,v,alpha)),mse=D.mse(shown);
      const observedPanel={...bottom,y:top.y},residualPanel={...top,y:bottom.y};
      const cuts=[...new Set(D.rounds.slice(0,count).map(q=>q.threshold))].sort((a,b)=>a-b);
      const bounds=[0,...cuts,13],values=bounds.slice(0,-1).map((lo,i)=>D.predict(count,(lo+bounds[i+1])/2,alpha));
      const q=D.rounds[Math.max(3,count)-1];
      const complete=alpha===1;
      out+=text(x,compact?18:27,complete?`Prediction after ${count} trees`:`Adding tree ${count}`,'class="panel-title"');
      out+=circle(x+3,compact?32:46,3.5,C.ink)+text(x+13,compact?36:50,'Observed y','class="key"');
      const oldX=x+(compact?105:160),keyY=compact?32:46,newX=x+(compact?235:350);
      out+=line(oldX,keyY,oldX+20,keyY,C.muted,2,'stroke-dasharray="6 5"')+text(oldX+27,keyY+4,'Earlier F₂','class="key"');
      out+=line(newX,keyY,newX+20,keyY,C.blue,3)+text(newX+27,keyY+4,`Current F${sub(count)}`,'class="key"');
      out+=text(x,compact?207:288,`Fit h${sub(q.number)} to y − F${sub(q.number-1)}`,'class="panel-title"');
      const lowerKeyY=compact?225:316,correctionKeyX=x+(compact?141:186);
      out+=circle(x+3,lowerKeyY,3.5,C.ink)+text(x+13,lowerKeyY+4,'Residual target','class="key"');
      out+=line(correctionKeyX,lowerKeyY,correctionKeyX+21,lowerKeyY,C.rust,3)+text(correctionKeyX+28,lowerKeyY+4,`Correction h${sub(q.number)}`,'class="key"');
      out+=axis(observedPanel,[0,5,10,15,20],'Time (min)',false)+axis(residualPanel,[-6,-3,0,3,6],'Residual (min)');
      for(const cut of cuts){
        for(const panel of [observedPanel,residualPanel])out+=line(X(cut),panel.y,X(cut),panel.y+panel.h,'#aaa',1,'stroke-dasharray="3 5"');
      }
      out+=step([6,10,18],[3.5,8],observedPanel,C.muted,'stroke-dasharray="6 5"','reference');
      out+=`<g data-errors="remaining" data-predictions="${shown.join(',')}">`;
      D.x.forEach((v,i)=>{out+=line(X(v),Y(D.y[i],observedPanel),X(v),Y(shown[i],observedPanel),'#aaa',1.6);});
      out+='</g>';
      out+=step(values,cuts,observedPanel,C.blue,'','prediction');
      values.forEach((v,i)=>{out+=text(X((bounds[i]+bounds[i+1])/2),Y(v,observedPanel)-10,precise(v),`class="value" fill="${C.blue}" text-anchor="middle"`);});
      out+=points(D.y,observedPanel,'observations');
      out+=text(x+w/2,compact?184:267,`h${sub(q.number)}: ${preciseSigned(q.left)} at x ≤ ${fmt(q.threshold)}; ${preciseSigned(q.right)} at x > ${fmt(q.threshold)}`,'text-anchor="middle" class="annotation"');
      // Keep the fitting target fixed while the accumulated prediction moves.
      // The selected tree and its target change only when the next round starts.
      out+=`<g data-selected-split="${q.threshold}" data-correction-round="${q.number}">`;
      out+=line(X(q.threshold),residualPanel.y,X(q.threshold),residualPanel.y+residualPanel.h,C.rust,1.8,'stroke-dasharray="5 4"');
      out+=text(X(q.threshold)+6,residualPanel.y+17,`Split at ${fmt(q.threshold)} km`,`class="cut-label" fill="${C.rust}"`);
      out+='</g>';
      out+=step([q.left,q.right],[q.threshold],residualPanel,C.rust,'',`h${q.number}`);
      out+=points(q.residual,residualPanel,'residuals');
      const leafBounds=[0,q.threshold,13];
      [q.left,q.right].forEach((v,i)=>{
        const labelX=i===0?leafBounds[i]+.35*(leafBounds[i+1]-leafBounds[i]):(leafBounds[i]+leafBounds[i+1])/2;
        const labelY=Y(v,residualPanel)+(v<0?(compact?17:24):-(compact?7:10));
        out+=text(X(labelX),labelY,preciseSigned(v),`class="value" fill="${C.rust}" text-anchor="middle"`);
      });
      out+=`<g data-continuation-round="${count}" data-continuation-amount="${alpha}" data-training-mse="${mse}"/>`;
      return wrap(out);
    }
    if(scene===5){
      const strips=[{y:compact?39:56,h:compact?66:104,min:-4,max:6},{y:compact?159:220,h:compact?66:104,min:-4,max:6},{y:compact?279:387,h:compact?62:118,min:0,max:20}];
      const inspectX=Number.isFinite(state.inspectX)?clamp(state.inspectX,0,13):6;
      const vals=D.rounds.slice(0,2).map(q=>inspectX<=q.threshold?q.left:q.right).concat(D.predict(2,inspectX));
      out+=text(x,compact?19:29,'First tree h₁','class="panel-title"')+text(x+w,compact?19:29,`x = ${fmt(inspectX)} km`,'text-anchor="end" class="key"');
      out+=text(x,compact?139:192,'Second tree h₂','class="panel-title"');
      out+=text(x,compact?259:359,'Prediction F₂ = 12 + h₁ + h₂','class="panel-title"');
      strips.forEach((b,i)=>{
        out+=axis(b,i<2?[-3,0,5]:[0,10,20],i<2?'Correction (min)':'Time (min)',i===2,i===2);
        for(const cut of [3.5,8])out+=line(X(cut),b.y,X(cut),b.y+b.h,'#999',1,'stroke-dasharray="3 5"');
      });
      out+=line(X(inspectX),strips[0].y,X(inspectX),strips[2].y+strips[2].h,C.blue,1.5,'stroke-dasharray="5 4" data-contribution-guide="true"');
      D.rounds.slice(0,2).forEach((q,i)=>{
        out+=step([q.left,q.right],[q.threshold],strips[i],i?C.teal:C.rust,i?'stroke-dasharray="7 4"':'',`h${i+1}`);
        out+=labels([q.left,q.right],[q.threshold],strips[i],i?C.teal:C.rust);
      });
      out+=step([6,10,18],[3.5,8],strips[2],C.blue,'','prediction')+labels([6,10,18],[3.5,8],strips[2],C.blue,false);
      for(const cut of [3.5,8])out+=text(X(cut)+4,strips[0].y+strips[0].h+17,`${fmt(cut)} km`,'class="cut-label"');
      strips.forEach((b,i)=>{
        out+=circle(X(inspectX),Y(vals[i],b),compact?5:6,C.blue,'stroke="white" stroke-width="2"');
        out+=`<rect x="${x}" y="${b.y}" width="${w}" height="${b.h}" fill="transparent" data-contribution-plot="1" data-box="${x},${b.y},${w},${b.h}" style="cursor:crosshair"/>`;
      });
      out+=`<g data-contributions="${vals.join(',')}" data-inspect-x="${inspectX}"/>`;
      return wrap(out);
    }
    const transitFade=scene===1 && p>0 && p<1 && !trial?1-Math.min(clamp(p*8),clamp((1-p)*8)):1;
    const topTitle=scene===0?'Residuals from the starting prediction':scene===3 && p<1 && !trial?'Subtract the previous correction':`Residuals before update ${round}`;
    out+=text(x,compact?18:27,topTitle,'class="panel-title"');
    out+=text(x,compact?207:288,scene===0?'Observed times and starting prediction':'Observed times and model predictions',`class="panel-title" opacity="${transitFade}"`);
    out+=circle(x+3,compact?32:46,3.5,C.ink)+text(x+13,compact?36:50,'Residual y − F','class="key"');
    const meanReveal=trial?1:fitting?clamp(p-2):scene===0?0:1;
    if(meanReveal>0)out+=`<g opacity="${meanReveal}">`+line(x+(compact?141:186),compact?32:46,x+(compact?162:207),compact?32:46,C.rust,3)+text(x+(compact?169:214),compact?36:50,trial?'Candidate stump':`Correction h${sub(round)}`,'class="key"')+'</g>';
    out+=`<g opacity="${transitFade}">`+circle(x+3,compact?225:316,3.5,C.ink)+text(x+13,compact?229:320,'Observed y','class="key"');
    if(scene!==0){out+=line(x+(compact?105:147),compact?225:316,x+(compact?123:170),compact?225:316,C.muted,2,'stroke-dasharray="5 4"')+text(x+(compact?130:179),compact?229:320,`Previous F${sub(round-1)}`,'class="key"');}
    if(!fitting){const sx=x+(compact?235:346);out+=line(sx,compact?225:316,sx+20,compact?225:316,C.blue,3)+text(sx+27,compact?229:320,scene===0?'Mean F₀':'Updated F','class="key"');}
    out+='</g>';
    out+=axis(top,[-6,-3,0,3,6],'Residual (min)',false)+axis(bottom,[0,5,10,15,20],'Time (min)');
    if(scene>0){
      if(round===2)out+=cutGuide(8,.55,[bottom]);
      if(!fitting || trial || p>1)out+=cutGuide((trial||r).threshold,trial || !fitting?1:clamp(p-1));
    }
    const shownRound=scene===0?0:fitting?round-1:round;
    const predictions=D.x.map(v=>D.predict(shownRound,v,adding?amount:1));
    out+=residualStems(predictions,adding?clamp(p-2):0);
    if(scene===0){
      out+=points(D.rounds[0].residual,top,'residuals');
    } else if(scene===1 && p<1 && !trial){
      // A residual is a signed distance, not a point translated on equal axes.
      // Interpolating both endpoints explicitly rescales minutes between panels.
      const order=[0,3,2,4,5,1,6,7];
      out+='<g data-residual-transfer="true">';
      D.x.forEach((v,i)=>{
        const t=clamp((p-order[i]*.045)/.67);
        if(t===0)return;
        const zero=mix(Y(D.baseline,bottom),Y(0,top),t),end=mix(Y(D.y[i],bottom),Y(r.residual[i],top),t);
        out+=line(X(v),zero,X(v),end,C.muted,1.5,`opacity="${.7*(1-t)}"`)+circle(X(v),end,compact?3.8:4.5,C.ink,'stroke="white" stroke-width="1.2"');
      });
      out+='</g>';
    } else if(scene===3 && p<1 && !trial){
      const first=D.rounds[0],t=clamp(p);
      out+=step([first.left,first.right],[first.threshold],top,C.rust,'','previous-correction',1-t);
      out+=points(first.residual,top,'old-residuals',`opacity="${.3*(1-t)}"`);
      out+=points(first.residual.map((v,i)=>mix(v,r.residual[i],t)),top,'residuals');
    } else {
      const q=trial||r;
      if(meanReveal>0)out+=step([q.left,q.right],[q.threshold],top,C.rust,'',trial?'candidate':`h${round}`,meanReveal);
      out+=points(r.residual,top,'residuals');
      if(meanReveal>0)out+=`<g opacity="${meanReveal}">`+labels([q.left,q.right],[q.threshold],top,C.rust,true,true)+'</g>';
    }
    if(fitting && (trial || p>1)){
      const q=trial||r,groupOpacity=trial?1:clamp(p-1),bounds=[0,q.threshold,13];
      for(let side=0;side<2;side++){
        const indices=D.x.map((v,i)=>i).filter(i=>(D.x[i]<=q.threshold)===(side===0));
        const sum=indices.reduce((s,i)=>s+r.residual[i],0),mean=side===0?q.left:q.right;
        const label=meanReveal===1?`${signed(sum)} ÷ ${indices.length} = ${signed(mean)}`:`${indices.length} residuals`;
        out+=bracket(bounds[side],bounds[side+1],compact?170:244,label,groupOpacity);
      }
    }
    if(scene!==0)out+=predictionPath(round-1,1,C.muted,'stroke-dasharray="6 5"','previous');
    if(!fitting)out+=predictionPath(shownRound,adding?amount:1,C.blue,'','prediction');
    if(adding){
      const arrowOpacity=Number.isFinite(state.process)?clamp(p)* (1-clamp(p-2)):1;
      const positions=round===1?[6.4,11.5]:[2.6,6.8,11.5];
      positions.forEach(v=>out+=arrow(v,D.predict(round-1,v),D.predict(round,v),arrowOpacity));
      if(round===2 && p>0){
        out+=bracket(3.5,13,compact?170:244,p<2?'One leaf moves both plateaus: +1':'17 → 18: three errors grow',clamp(p));
      } else if(round===1 && p>0)out+=text(x+w/2,compact?186:264,'Add the same leaf value throughout its interval','text-anchor="middle" class="annotation"');
      const cuts=round===1?[8]:[3.5,8],bounds=[0,...cuts,13];
      out+=labels(bounds.slice(0,-1).map((v,i)=>D.predict(round,(v+bounds[i+1])/2,amount)),cuts,bottom,C.blue,false);
    }
    out+=points(D.y,bottom,'observations');
    if(scene===0)out+=text(X(11),Y(12,bottom)-10,'12',`class="value" fill="${C.blue}" text-anchor="middle"`);
    if(scene===4 && p>2){
      out+=`<g data-worsened-errors="9,10,12" opacity="${clamp(p-2)}">`;
      for(const i of [5,6,7])out+=circle(X(D.x[i]),Y(D.y[i],bottom),compact?6:7,'none',`stroke="${C.rust}" stroke-width="1.5"`);
      out+='</g>';
    }
    return wrap(out);
  }
  return {render,readout,topics,captions,processLabels,continuationState,continuationProgress};
});
