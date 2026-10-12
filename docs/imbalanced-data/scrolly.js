(() => {
  'use strict';
  const M=window.ImbalanceModel;
  const blue='#234e70',orange='#a54f32',teal='#007f73',grid='#e4e7e9';
  const clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,v));
  const ease=v=>{v=clamp(v);return v*v*(3-2*v);};
  const lerp=(a,b,t)=>a+(b-a)*t;
  const number=n=>Math.round(n).toLocaleString('en-US');
  const titles=['Start with the population.','Hold x fixed. Count the classes.','Keep positives. Thin negatives.','Compare the odds.','The same multiplier at every x.','Logs turn multiplication into addition.','That is why the line shifts.','Undo the addition.'];
  const descriptions=[
    'At x = 0, 1, and 2, the event probabilities are 1%, 50%, and 99%. The overall event rate is 2.47% because most observations have x = 0.',
    'Focus on 10,000 observations at x = 0: 100 positives and 9,900 negatives. Probability divides positives by the total.',
    'Keep every positive and randomly retain 10% of negatives. The expected positive count stays at 100 while the negative count falls to 990.',
    'Odds divide positives by negatives. A tenfold smaller denominator makes the odds ten times larger.',
    'Use the same retention fraction at every x. Every value of x gets the same odds multiplier: ×10.',
    'The logit is the logarithm of the odds. Multiplying odds by ten adds log(10) ≈ 2.30 to every log-odds value.',
    'Scroll to lift the sampled line. Every point moves by the same amount, so the intercept changes and the slope stays fixed.',
    'Add log(s) back to the sampled log odds. With s = 0.10, the correction is about −2.30. Scroll to bring the line back down.'
  ];
  function wrap(content,label,width=720){return `<svg viewBox="0 0 ${width} 300" role="img" aria-label="${label}" class="scene-chart">${content}</svg>`;}
  function path(points){return points.map((p,i)=>`${i?'L':'M'}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(' ');}
  function axis(log){
    const narrow=innerWidth<=650,left=narrow?45:70,right=narrow?315:670,width=narrow?360:720;
    const fx=x=>left+x*(right-left)/2,fy=y=>log?251-(y+5)/13*210:251-y*210;
    let c='';
    (log?[-4,0,4,8]:[0,.5,1]).forEach(y=>{c+=`<line x1="${left}" y1="${fy(y)}" x2="${right}" y2="${fy(y)}" stroke="${grid}"/><text x="${left-15}" y="${fy(y)+6}" text-anchor="end">${log?y:y*100+'%'}</text>`;});
    [0,1,2].forEach(x=>c+=`<text x="${fx(x)}" y="279" text-anchor="middle">${x}</text>`);
    return {width,c:c+`<text x="${left}" y="25">${log?'Log odds':'Event probability'}</text><text x="${right}" y="299" text-anchor="end">Predictor x</text>`,fx,fy};
  }
  function logistic(p){
    const correcting=p>6.1,stage=correcting?7:Math.round(p),s=lerp(1,.1,ease(p-1)),q=100/(100+9900*s);
    let c='',equation='',width=720,note='Exact population relationships; sampling is random within each class.';
    if(p<.55){
      const a=axis(false),points=Array.from({length:121},(_,j)=>{const x=j/60;return[a.fx(x),a.fy(M.sigmoid(Math.log(99)*(x-1)))];});
      width=a.width;c=a.c+`<path d="${path(points)}" stroke="${blue}" stroke-width="4" fill="none"/>`;
      [0,1,2].forEach(x=>c+=`<circle cx="${a.fx(x)}" cy="${a.fy(M.sigmoid(Math.log(99)*(x-1)))}" r="6" fill="${blue}"/>`);
      equation='logit p(x) = log(99) × (x − 1)';
    }else if(p<2.55){
      const retained=99*s;
      c='<text x="70" y="26">x = 0 · each dot = 100 observations</text>';
      for(let i=0;i<100;i++){
        const x=72+(i%20)*30,y=66+Math.floor(i/20)*29;
        const weight=i===0?1:clamp(retained-(i-1));
        const fall=i===0?0:(1-weight)*24*ease(p-1);
        c+=`<circle cx="${x}" cy="${y+fall}" r="7" fill="${i===0?orange:blue}" opacity="${i===0?1:lerp(.07,1,weight)}"/>`;
      }
      c+=`<text x="70" y="251" fill="${orange}">100 positives kept</text><text x="380" y="251" fill="${blue}">${number(9900*s)} negatives kept</text>`;
      equation=`Probability = 100 / (100 + ${number(9900*s)}) = ${(100*q).toFixed(2)}%`;
      note='The thinning animation shows expected retained counts, with partial dots at the boundary.';
    }else if(p<3.55){
      const r=ease((p-2.55)/.8);
      c=`<text x="70" y="35">Population odds</text><text x="480" y="35">Sample odds</text><text x="170" y="102" text-anchor="middle" class="large">100</text><line x1="75" y1="122" x2="265" y2="122" stroke="${blue}" stroke-width="2"/><text x="170" y="172" text-anchor="middle" class="large">9,900</text><text x="575" y="102" text-anchor="middle" class="large">100</text><line x1="480" y1="122" x2="670" y2="122" stroke="${orange}" stroke-width="2"/><text x="575" y="172" text-anchor="middle" class="large">990</text><path d="M290,120 H${lerp(290,450,r)}" stroke="${orange}" stroke-width="3"/><text x="370" y="100" text-anchor="middle" opacity="${r}">×10</text><text x="170" y="244" text-anchor="middle">1 / 99</text><text x="575" y="244" text-anchor="middle">10 / 99</text>`;
      equation='Sample odds = population odds / s = population odds × 10';
    }else if(p<4.55){
      c='<text x="70" y="30">x</text><text x="175" y="30">Population odds</text><text x="500" y="30">Sample odds</text>';
      [[0,'1 / 99','10 / 99'],[1,'1','10'],[2,'99','990']].forEach(([x,a,b],i)=>{
        const r=ease((p-3.55)*2.4-i*.27),y=89+i*80;
        c+=`<text x="70" y="${y}">${x}</text><text x="200" y="${y}">${a}</text><path d="M325,${y-7} H${lerp(325,460,r)}" stroke="${orange}" stroke-width="2"/><text x="390" y="${y-21}" text-anchor="middle" opacity="${r}">×10</text><text x="520" y="${y}" opacity="${r}">${b}</text>`;
      });
      equation='q(x) / (1 − q(x)) = p(x) / [s(1 − p(x))]';
    }else if(p<5.25){
      c='<text x="70" y="28">x</text><text x="145" y="28">Population log odds</text><text x="500" y="28">Sample log odds</text>';
      [[0,-Math.log(99)],[1,0],[2,Math.log(99)]].forEach(([x,l],i)=>{
        const r=ease((p-4.55)*3-i*.25),y=90+i*78;
        c+=`<text x="70" y="${y}">${x}</text><text x="215" y="${y}">${l.toFixed(2)}</text><text x="360" y="${y}" opacity="${r}" fill="${orange}">+2.30</text><text x="560" y="${y}" opacity="${r}">${(l+Math.log(10)).toFixed(2)}</text>`;
      });
      equation='log(sample odds) = log(population odds) + log(10)';
      note='Natural logarithms; displayed values are rounded after calculation.';
    }else{
      const a=axis(true),shift=Math.log(10)*ease((p-5.25)/.75),correction=ease((p-6.1)/.8),current=shift*(1-correction);
      const draw=(offset,color,dashed=false)=>`<path d="${path([[a.fx(0),a.fy(-Math.log(99)+offset)],[a.fx(2),a.fy(Math.log(99)+offset)]])}" stroke="${color}" stroke-width="4" ${dashed?'stroke-dasharray="8 6"':''} fill="none"/>`;
      width=a.width;c=a.c+draw(0,blue)+draw(shift,orange);
      if(correcting)c+=draw(current,teal,true);
      [0,1,2].forEach(x=>{
        const v=Math.log(99)*(x-1);
        c+=`<circle cx="${a.fx(x)}" cy="${a.fy(v)}" r="5" fill="${blue}"/><circle cx="${a.fx(x)}" cy="${a.fy(v+shift)}" r="5" fill="${orange}"/>`;
        if(correcting)c+=`<circle cx="${a.fx(x)}" cy="${a.fy(v+current)}" r="6" fill="${teal}"/>`;
      });
      const gap=correcting?current:shift,midX=a.fx(1)+18;
      c+=`<path d="M${midX-5},${a.fy(gap)} H${midX+5} M${midX},${a.fy(gap)} V${a.fy(0)} M${midX-5},${a.fy(0)} H${midX+5}" stroke="${correcting?teal:orange}" stroke-width="2" fill="none"/><text x="${a.fx(1)-18}" y="${a.fy(shift)-19}" text-anchor="end" fill="${correcting?teal:orange}">${width===360?'':correcting?'Remaining shift ':'Added log odds '}+${gap.toFixed(2)}</text><text x="${a.fx(2)}" y="${a.fy(Math.log(99)+shift)-14}" text-anchor="end" fill="${orange}">Sampled</text><text x="${a.fx(2)}" y="${a.fy(Math.log(99))+35}" text-anchor="end" fill="${correcting?teal:blue}">${correcting?'Corrected':'Population'}</text>`;
      equation=correcting?`Corrected log odds = sampled log odds − ${(Math.log(10)*correction).toFixed(2)}`:`Sampled log odds = population log odds + ${shift.toFixed(2)}`;
      note=correcting?'At the end, adding log(0.10) ≈ −2.30 restores the original line and probabilities.':'A fixed addition moves the entire line. Its slope remains log(99) ≈ 4.60.';
    }
    return {stage,title:titles[stage],description:descriptions[stage],equation,note,svg:wrap(c,`${titles[stage]} ${equation}`,width),eyebrow:'01 / Logistic correction'};
  }

  const reducedQuery=matchMedia('(prefers-reduced-motion: reduce)');
  let motion=!reducedQuery.matches,queued=false;
  const chapters=[...document.querySelectorAll('.scroll-chapter')].map(el=>{
    const n=Number(el.dataset.stages),scene=el.querySelector('.scene'),refs=Object.fromEntries([...el.querySelectorAll('[data-scene]')].map(x=>[x.dataset.scene,x]));
    const dots=el.querySelector('.scene-dots');
    const chapter={el,n,scene,refs,last:-1,stage:-1,dots:[]};
    for(let i=0;i<n;i++){
      const b=document.createElement('button');b.type='button';b.title=`Stage ${i+1}`;b.setAttribute('aria-label',`Stage ${i+1}`);
      b.addEventListener('click',()=>{
        const top=el.getBoundingClientRect().top+scrollY;
        const span=el.offsetHeight-scene.offsetHeight;
        scrollTo({top:top+i/(n-1)*span-(innerWidth<=650?60:76),behavior:motion?'smooth':'instant'});
      });dots.append(b);chapter.dots.push(b);
    }
    el.querySelector('.motion-toggle').addEventListener('click',()=>{motion=!motion;chapters.forEach(c=>c.last=-1);update();});
    return chapter;
  });
  function resize(){
    chapters.forEach(c=>{
      const span=c.el.offsetHeight-c.scene.offsetHeight;
      c.el.querySelectorAll('.scroll-marker').forEach((marker,i)=>marker.style.top=`${i/(c.n-1)*span}px`);
      c.last=-1;
    });update();
  }
  function update(){
    queued=false;
    const header=innerWidth<=650?60:76;
    chapters.forEach(c=>{
      const rect=c.el.getBoundingClientRect();
      const progress=clamp((header-rect.top)/(c.el.offsetHeight-c.scene.offsetHeight))*(c.n-1);
      const p=motion?progress:Math.round(progress);
      if(Math.abs(p-c.last)<.0001)return;c.last=p;
      const data=c.el.dataset.chapter==='logistic'?logistic(p):window.ImbalanceScenes[c.el.dataset.chapter](p);
      const stage=data.stage??Math.round(p);
      c.refs.title.textContent=data.title;c.refs.description.textContent=data.description;c.refs.eyebrow.textContent=data.eyebrow;
      c.refs.canvas.innerHTML=data.svg;c.refs.equation.textContent=data.equation;c.refs.note.textContent=data.note||'Scroll back to retrace the argument.';
      c.refs.count.textContent=`${String(stage+1).padStart(2,'0')} / ${String(c.n).padStart(2,'0')}`;
      c.el.querySelector('.scene-progress span').style.width=`${progress/(c.n-1)*100}%`;
      const toggle=c.el.querySelector('.motion-toggle');toggle.textContent=motion?'Motion on':'Motion off';toggle.setAttribute('aria-pressed',String(motion));
      c.dots.forEach((b,i)=>{b.classList.toggle('active',i===stage);if(i===stage)b.setAttribute('aria-current','step');else b.removeAttribute('aria-current');});
      c.el.dataset.progress=p.toFixed(4);c.el.dataset.stage=stage;
    });
  }
  function schedule(){if(!queued){queued=true;requestAnimationFrame(update);}}
  addEventListener('scroll',schedule,{passive:true});addEventListener('resize',resize);
  reducedQuery.addEventListener('change',e=>{motion=!e.matches;chapters.forEach(c=>c.last=-1);update();});
  resize();
  // A direct stage URL needs its marker positioned before scrolling to it.
  const anchor=document.getElementById(location.hash.slice(1));
  if(anchor?.classList.contains('scroll-marker'))requestAnimationFrame(()=>anchor.scrollIntoView({behavior:'instant',block:'start'}));
})();
