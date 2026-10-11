(function (root, factory) {
  'use strict';
  const api = factory(root.DeliveryExample || (typeof require === 'function' ? require('./delivery.js') : null));
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.DeliveryView = api;
})(typeof window !== 'undefined' ? window : globalThis, function (D) {
  'use strict';
  const C = { ink:'#242424', muted:'#606060', blue:'#234E70', rust:'#A54F32', teal:'#007F73', grid:'#e8e8e8' };
  const topics = ['Start with one prediction', 'Fit the first correction', 'Add it to the prediction', 'Fit the remaining mistakes', 'Add a second correction', 'Two stumps, three plateaus'];
  const captions = [
    'The same eight deliveries appear in both panels. A residual is observed time minus predicted time.',
    'The stump predicts the mean residual on each side of 8 km: −3 minutes and +5 minutes.',
    'Move the slider: add the upper step function to the lower prediction at every distance.',
    'Recompute residuals from F₁. The next stump separates the two short deliveries from the other six.',
    'Add h₂ everywhere, including across the first stump’s boundary at 8 km.',
    'The two distinct cut points divide this one-dimensional input into three intervals.'
  ];
  const fmt = n => Math.abs(n)<1e-9?'0':String(Number(n.toFixed(2))).replace('-', '−');
  const signed = n => n>0?'+'+fmt(n):fmt(n);
  const esc = s => String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const text=(x,y,s,extra='')=>`<text x="${x}" y="${y}" ${extra}>${esc(s)}</text>`;
  const line=(x1,y1,x2,y2,color=C.grid,width=1,extra='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}" ${extra}/>`;
  const circle=(x,y,r,color,extra='')=>`<circle cx="${x}" cy="${y}" r="${r}" fill="${color}" ${extra}/>`;
  function progress(state) { return Math.max(0,Math.min(1,Number.isFinite(state.correction)?state.correction:1)); }
  function readout(scene,state={}) {
    const p=progress(state);
    if(scene===0)return 'F₀ = mean(y) = 12 min · MSE = 18.75';
    if(scene===1)return 'h₁: −3 if x ≤ 8; +5 if x > 8';
    if(scene===3)return 'h₂: −3 if x ≤ 3.5; +1 if x > 3.5';
    if(scene===5)return 'F₂ = 12 + h₁ + h₂ · predictions: 6, 10, 18';
    const round=scene===2?1:2;
    const mse=D.mse(D.x.map(x=>D.predict(round,x,p)));
    return `F${round===1?'₁':'₂'} = F${round===1?'₀':'₁'} + ${fmt(p)} × h${round===1?'₁':'₂'} · MSE = ${fmt(mse)}`;
  }
  function render(scene,state={},compact=false) {
    const W=compact?420:780,H=compact?355:540;
    const x=compact?48:76,w=compact?350:652;
    const top={y:compact?45:66,h:compact?100:157,min:-7,max:6};
    const bottom={y:compact?214:333,h:compact?100:166,min:0,max:20};
    const X=v=>x+v/13*w;
    const Y=(v,b)=>b.y+b.h-(v-b.min)/(b.max-b.min)*b.h;
    const round=scene<3?1:2, r=D.rounds[round-1], p=progress(state);
    let out='';
    function axis(b,ticks,label,showXLabel) {
      let s=`<g data-axis="${label}" data-x-domain="0,13" data-y-domain="${b.min},${b.max}">`;
      for(const v of ticks){s+=line(x,Y(v,b),x+w,Y(v,b),v===0?'#929292':C.grid,v===0?1.3:1);s+=text(x-9,Y(v,b)+4,fmt(v),'text-anchor="end" class="tick"');}
      s+=line(x,b.y,x,b.y+b.h,'#929292');
      for(const v of [0,3,6,9,12]){s+=line(X(v),b.y+b.h,X(v),b.y+b.h+4,'#929292');s+=text(X(v),b.y+b.h+18,v,'text-anchor="middle" class="tick"');}
      const lx=compact?12:23,ly=b.y+b.h/2;
      s+=text(lx,ly,label,`text-anchor="middle" transform="rotate(-90 ${lx} ${ly})" class="axis-label"`);
      if(showXLabel)s+=text(x+w, H-2,'Distance (km)','text-anchor="end" class="axis-label"');
      return s+'</g>';
    }
    function step(values,cuts,b,color,extra='',kind='') {
      const bounds=[0,...cuts,13];
      let s=`<g data-curve="${kind}" data-values="${values.join(',')}" data-cuts="${cuts.join(',')}">`;
      values.forEach((v,i)=>{s+=line(X(bounds[i]),Y(v,b),X(bounds[i+1]),Y(v,b),color,3,extra);if(i)s+=line(X(bounds[i]),Y(values[i-1],b),X(bounds[i]),Y(v,b),color,1.5,extra);});
      return s+'</g>';
    }
    function predictionPath(count,amount,color,extra='',kind='') {
      const cuts=[...new Set(D.rounds.slice(0,count).map(r=>r.threshold))].sort((a,b)=>a-b);
      const bounds=[0,...cuts,13],values=bounds.slice(0,-1).map((lo,i)=>D.predict(count,(lo+bounds[i+1])/2,amount));
      return step(values,cuts,bottom,color,extra,kind);
    }
    function labels(values,cuts,b,color) {
      const bounds=[0,...cuts,13];
      return values.map((v,i)=>text(X((bounds[i]+bounds[i+1])/2),Y(v,b)-(compact?7:10),signed(v),`text-anchor="middle" class="value" fill="${color}"`)).join('');
    }
    function points(values,b,kind) {
      return `<g data-points="${kind}" data-values="${values.join(',')}">`+D.x.map((v,i)=>circle(X(v),Y(values[i],b),compact?3.8:4.5,C.ink,'stroke="white" stroke-width="1.2"')).join('')+'</g>';
    }
    const topTitle=scene===0?'Residuals from the starting prediction':scene===5?'The two fitted corrections':`Residuals before update ${round}`;
    const lowerTitle=scene===0?'Observed times and starting prediction':scene===5?'Their sum, plus the starting mean':'Observed times and model predictions';
    out+=text(x,compact?18:27,topTitle,'class="panel-title"');
    out+=text(x,compact?183:282,lowerTitle,'class="panel-title"');
    // Small, stable keys keep the role of dots and lines explicit across rounds.
    if(scene!==5){out+=circle(x+3,compact?32:46,3.5,C.ink);out+=text(x+13,compact?36:50,'Residual y − F','class="key"');}
    if(scene>0){const sx=scene===5?x:x+(compact?141:186);out+=line(sx,compact?32:46,sx+21,compact?32:46,C.rust,3);out+=text(sx+28,compact?36:50,scene===5?'h₁':`Correction h${round===1?'₁':'₂'}`,'class="key"');}
    if(scene===5){out+=line(x+95,compact?32:46,x+116,compact?32:46,C.teal,3,'stroke-dasharray="6 4"');out+=text(x+123,compact?36:50,'h₂','class="key"');}
    out+=circle(x+3,compact?198:310,3.5,C.ink)+text(x+13,compact?202:314,'Observed y','class="key"');
    if(scene!==0 && scene!==5){out+=line(x+(compact?105:147),compact?198:310,x+(compact?123:170),compact?198:310,C.muted,2,'stroke-dasharray="5 4"');out+=text(x+(compact?130:179),compact?202:314,`Previous F${round===1?'₀':'₁'}`,'class="key"');}
    if(scene!==1 && scene!==3){const sx=x+(compact?235:346);out+=line(sx,compact?198:310,sx+20,compact?198:310,C.blue,3);out+=text(sx+27,compact?202:314,scene===0?'Mean F₀':scene===5?'Sum F₂':'Updated F','class="key"');}
    out+=axis(top,[-6,-3,0,3,6],'Residual (min)',false)+axis(bottom,[0,5,10,15,20],'Time (min)',true);
    const shownRound=scene===0?0:scene===1?0:scene===2?1:scene===3?1:2;
    const amount=scene===2 || scene===4?p:1;
    if(scene>0){
      const cuts=scene===5?[3.5,8]:[r.threshold];
      for(const cut of cuts){for(const box of [top,bottom])out+=line(X(cut),box.y,X(cut),box.y+box.h,'#aaa',1,'stroke-dasharray="3 5"');out+=text(X(cut)+5,top.y+top.h-7,`${fmt(cut)} km`,'class="cut-label"');}
    }
    // Lower stems show the error remaining after the displayed partial update.
    for(let i=0;i<D.x.length;i++)out+=line(X(D.x[i]),Y(D.y[i],bottom),X(D.x[i]),Y(D.predict(shownRound,D.x[i],amount),bottom),'#b9b9b9',1.6);
    if(scene===5){
      for(let i=0;i<2;i++){const q=D.rounds[i];out+=step([q.left,q.right],[q.threshold],top,i?C.teal:C.rust,i?'stroke-dasharray="6 4"':'',`h${i+1}`);}
      out+=text(X(11),Y(5,top)-8,'h₁: +5',`class="value" fill="${C.rust}" text-anchor="middle"`);
      out+=text(X(10),Y(1,top)-8,'h₂: +1',`class="value" fill="${C.teal}" text-anchor="middle"`);
      out+=text(X(1.6),Y(-3,top)-8,'−3',`class="value" text-anchor="middle"`);
    } else {
      if(scene>0){out+=step([r.left,r.right],[r.threshold],top,C.rust,'',`h${round}`);out+=labels([r.left,r.right],[r.threshold],top,C.rust);}
      out+=points(scene===0?D.rounds[0].residual:r.residual,top,'residuals');
    }
    if(scene!==0 && scene!==5)out+=predictionPath(round-1,1,C.muted,'stroke-dasharray="6 5"','previous');
    if(scene!==1 && scene!==3)out+=predictionPath(shownRound,amount,C.blue,'','prediction');
    out+=points(D.y,bottom,'observations');
    if(scene===0)out+=text(X(11),Y(12,bottom)-10,'12',`class="value" fill="${C.blue}" text-anchor="middle"`);
    if(scene===2 || scene===4 || scene===5){
      const cuts=shownRound===1?[8]:[3.5,8],bounds=[0,...cuts,13];
      bounds.slice(0,-1).forEach((v,i)=>{const mid=(v+bounds[i+1])/2,pred=D.predict(shownRound,mid,amount);out+=text(X(mid),Y(pred,bottom)-10,fmt(pred),`class="value" fill="${C.blue}" text-anchor="middle"`);});
    }
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" fill="${C.ink}" role="img" aria-labelledby="delivery-title delivery-description"><title id="delivery-title">${esc(topics[scene])}</title><desc id="delivery-description">${esc(captions[scene]+' '+readout(scene,state))}</desc><style>text{font-family:Georgia,serif;font-size:${compact?14:18}px}.panel-title{font-size:${compact?16:21}px}.tick,.key,.cut-label,.axis-label{font-family:Arial,sans-serif;font-size:${compact?12:14}px}.tick,.axis-label,.key{fill:${C.muted}}.cut-label,.value{paint-order:stroke;stroke:white;stroke-width:5px;stroke-linejoin:round}.value{font-size:${compact?15:20}px}</style>${out}</svg>`;
  }
  return {render,readout,topics,captions};
});
