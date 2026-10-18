(() => {
  'use strict';
  const C={blue:'#234e70',orange:'#a54f32',teal:'#007f73',ink:'#242424',muted:'#606060',grid:'#e4e7e9',removed:'#edf0f2'};
  const L=Math.log(99),SHIFT=Math.log(10),groups=[{x:0,positive:100,negative:9900},{x:1,positive:5000,negative:5000}];
  const clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,v)),ease=v=>{const t=clamp(v);return t*t*(3-2*t);};
  const safe=v=>String(v).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  const count=v=>v.toLocaleString('en-US',{maximumFractionDigits:1}),signed=v=>`${v<-.00001?'−':v>.00001?'+':''}${Math.abs(v).toFixed(2)}`;
  const text=(x,y,v,c=C.ink,z=18,a='start',extra='')=>`<text x="${x}" y="${y}" fill="${c}" text-anchor="${a}" style="font-size:${z}px;font-variant-numeric:tabular-nums" ${extra}>${safe(v)}</text>`;
  const formula=(key,x,y,width,height,size,opacity=1,color=C.ink)=>{
    if(opacity<=0)return '';
    const f=window.LogisticMath[key];
    return `<foreignObject class="logistic-formula" data-formula="${key}" x="${x}" y="${y}" width="${width}" height="${height}" opacity="${opacity}"><div xmlns="http://www.w3.org/1999/xhtml" style="font-size:${size}px;color:${color}">${f.html}</div></foreignObject>`;
  };
  const line=(x1,y1,x2,y2,c=C.grid,extra='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${c}" ${extra}/>`;
  const dot=(x,y,c,r=5.5,extra='')=>`<circle cx="${x}" cy="${y}" r="${r}" fill="${c}" stroke="white" stroke-width="1.6" ${extra}/>`;
  const story=[
    ['Logistic regression assumes linear log odds.','The model makes log odds a straight line in the predictor. Here, assume that relationship is also the truth.'],
    ['Write the probabilities as counts.','Within each group sharing a predictor value, divide the positive and negative counts by the group total.'],
    ['Cancel the shared total.','The group total drops out. Log odds is the logarithm of the positive-to-negative count ratio.'],
    ['Keep positives. Thin the negatives.','Keep every positive and randomly retain the same share s of negatives in each x group.'],
    ['One sampling share, one constant shift.','Dividing the odds by s subtracts log s from the log odds. Every group gets the same shift.'],
    ['Only the intercept changes.','With 10% of negatives retained, both points rise by 2.30. The sampled line is parallel to the population line.'],
    ['Prior correction shifts it back.','Shift the sampled intercept back to recover the population line: the prior correction in King and Zeng’s rare-events work.']
  ];
  function state(progress){
    const p=clamp(Number(progress)||0,0,6),stage=Math.round(p),action=ease((p-stage+.5)/.5);
    // Integer stages are completed snapshots, also used when motion is off.
    const negativeShare=stage===3||stage===5?Math.exp(-SHIFT*action):stage>=4?.1:1;
    return {p,stage,action,negativeShare,offset:-Math.log(negativeShare),corrected:stage===6?action:0};
  }
  function equations(s,narrow){
    const width=narrow?380:900,height=narrow?400:420,size=narrow?30:46;
    const main=(key,opacity=1)=>formula(key,20,45,width-40,230,size,opacity);
    let body='';
    if(s.stage===0||s.stage===1){
      const fade=s.stage===1?s.action:0;
      body+=main('linear',1-fade)+formula('coefficientKey',20,280,width-40,55,narrow?20:26,1-fade,C.muted);
      if(s.stage===1)body+=main('normalized',fade)+formula('total',20,292,width-40,55,narrow?24:32,fade)
        +text(width/2,narrow?370:390,'The total within this group',C.muted,narrow?16:21,'middle',`opacity="${fade}"`);
    }else if(s.stage===2){
      const mark=ease(s.action*2),finish=ease(s.action*2-1);
      body+=main('normalized',1-mark)+main('cancelled',mark*(1-finish))+main('ratio',finish);
      body+=text(width/2,narrow?335:355,'Only the positive-to-negative ratio remains.',C.muted,narrow?16:22,'middle');
    }else if(s.stage===4){
      if(narrow){
        body+=formula('sampledRatio',15,15,350,155,32)
          +formula('expansionRhs',10,170,360,125,28,s.action);
      }else{
        body+=formula('expansionStart',20,45,860,230,46,1-s.action)
          +formula('expansion',20,45,860,230,46,s.action);
      }
      body+=formula('shift',20,narrow?315:300,width-40,65,narrow?27:35,s.action,C.orange);
    }
    return {body,width,height};
  }
  function tiles(x,y,cell,gap,share){
    let body='';
    const tileHeight=7;
    for(let i=0;i<100;i++){
      // A fixed permutation makes removal scatter through the negative pool.
      const rank=(i*37)%100,retained=clamp(share*100-rank),rx=x+i%10*(cell+gap),ry=y+Math.floor(i/10)*(tileHeight+gap);
      body+=`<rect x="${rx}" y="${ry}" width="${cell}" height="${tileHeight}" rx="1" fill="${C.removed}"/><rect x="${rx}" y="${ry}" width="${cell}" height="${tileHeight}" rx="1" fill="${C.blue}" opacity="${retained}"/>`;
    }
    return body;
  }
  function thinning(s,narrow){
    const width=narrow?380:900,height=narrow?400:420;
    let body=formula('sampledRatio',20,0,width-40,narrow?145:175,narrow?32:46);
    body+=text(width/2,narrow?159:194,`s = ${(s.negativeShare*100).toFixed(1)}% of negatives retained`,C.blue,narrow?19:24,'middle');
    const cw=narrow?172:355,top=narrow?179:215;
    groups.forEach((g,i)=>{
      const x=narrow?10+i*188:80+i*385;
      body+=`<rect x="${x}" y="${top}" width="${cw}" height="${narrow?199:190}" rx="5" fill="white" stroke="#d8dfe3"/>`;
      body+=text(x+cw/2,top+26,`${count(g.positive)} positive`,C.orange,narrow?15:19,'middle')+text(x+cw/2,top+52,`${count(g.negative*s.negativeShare)} negative`,C.blue,narrow?15:19,'middle');
      body+=tiles(x+(cw-(narrow?128:228))/2,top+66,narrow?11:20,narrow?2:3,s.negativeShare);
      const q=g.positive/(g.positive+s.negativeShare*g.negative);
      body+=text(x+cw/2,top+(narrow?187:177),`Positive share ${(100*q).toFixed(1)}%`,C.muted,narrow?14:18,'middle');
    });
    return {body,width,height};
  }
  function plot(s,narrow){
    const width=narrow?380:900,height=narrow?335:440,left=narrow?42:91,right=narrow?347:837,top=narrow?80:82,bottom=narrow?190:269;
    const fx=x=>narrow?88+x*203:199+x*499,fy=v=>bottom-(v+5.5)/9*(bottom-top),corrected=s.stage===6,offset=s.offset*(1-s.corrected);
    let body=formula(corrected?'correctedLine':'sampleLine',10,0,width-20,48,narrow?21:32,1,corrected?C.teal:C.ink);
    const legend=(x,label,color,dash='',opacity=1)=>`<g opacity="${opacity}">${line(x,56,x+(narrow?16:26),56,color,`stroke-width="3" ${dash}`)}${text(x+(narrow?22:33),61,label,color,narrow?12:17)}</g>`;
    body+=legend(narrow?44:440,'Population',C.blue)+legend(narrow?161:585,'Sampled',C.orange,'',corrected?.5:1);
    if(corrected)body+=legend(narrow?268:720,'Corrected',C.teal,'stroke-dasharray="5 3"');
    body+=text(left,top-6,'Log odds',C.muted,narrow?13:17);
    [-4,-2,0,2].forEach(v=>{body+=line(left,fy(v),right,fy(v),v===0?'#ccd4d9':C.grid)+text(left-10,fy(v)+5,v<0?`−${-v}`:v,C.muted,narrow?13:17,'end');});
    body+=line(left,top,left,bottom,'#aeb9c1')+line(left,bottom,right,bottom,'#aeb9c1');
    const segment=(shift,color,extra='')=>line(fx(0),fy(-L+shift),fx(1),fy(shift),color,`stroke-width="3.5" ${extra}`);
    body+=segment(0,C.blue)+segment(s.offset,C.orange,corrected?'opacity=".35"':'');
    if(corrected)body+=segment(offset,C.teal,'stroke-dasharray="8 4"');
    groups.forEach(g=>{
      const x=fx(g.x),original=L*(g.x-1);
      body+=dot(x,fy(original),C.blue)+dot(x,fy(original+s.offset),C.orange,5.5,corrected?'opacity=".45"':'');
      if(corrected)body+=dot(x,fy(original+offset),C.teal,6);
      body+=text(x,bottom+21,g.x,C.muted,narrow?14:18,'middle');
      const bx=x+(narrow?12:19),y1=fy(original+s.offset),y2=fy(original+(corrected?offset:0));
      if(Math.abs(y2-y1)>.1){
        const color=corrected?C.teal:C.orange;
        body+=line(bx,y1,bx,y2,color,'stroke-width="1.8"')+line(bx-4,y1,bx+4,y1,color,'stroke-width="1.8"')+line(bx-4,y2,bx+4,y2,color,'stroke-width="1.8"');
        body+=text(bx+7,(y1+y2)/2+5,`${corrected?'−':'+'}${(corrected?s.offset*s.corrected:s.offset).toFixed(2)}`,color,narrow?13:17);
      }
    });
    body+=text(width/2,bottom+(narrow?44:48),'Predictor x',C.muted,narrow?14:18,'middle');
    const cardTop=narrow?243:337,cw=narrow?170:350;
    groups.forEach(g=>{
      const x=fx(g.x)-cw/2,neg=g.negative*s.negativeShare;
      body+=`<rect x="${x}" y="${cardTop}" width="${cw}" height="${narrow?88:92}" rx="5" fill="white" stroke="#d8dfe3"/>`;
      body+=text(x+cw/2,cardTop+23,`${count(g.positive)} positive`,C.orange,narrow?14:17,'middle')+text(x+cw/2,cardTop+48,`${count(neg)} negative`,C.blue,narrow?14:17,'middle')+text(x+cw/2,cardTop+72,`Sampled log odds ${signed(Math.log(g.positive/neg))}`,C.muted,narrow?12:16,'middle');
    });
    return {body,width,height};
  }
  function logistic(progress){
    const s=state(progress),narrow=window.innerWidth<=650,figure=s.stage===3?thinning(s,narrow):s.stage>=5?plot(s,narrow):equations(s,narrow);
    const notes=[
      'An assumption about the relationship at every x; two points below will illustrate it.',
      'The subscripts distinguish positive and negative counts within this group.',
      'The same total appears in the numerator and denominator.',
      'Expected retained counts · Blue tiles show the original negative pool; pale tiles are removed. Finite random samples fluctuate.',
      's is the negative retention share. Keeping 10% means s = 0.10.',
      'Expected group counts are shown. A constant shift preserves the slope and ranking.',
      '<a href="https://gking.harvard.edu/files/0s.pdf">King &amp; Zeng (2001), §4.1</a> · Prior correction changes the fitted intercept; sampling and finite-sample bias corrections are separate.'
    ];
    const equation=['','','','','',`Same slope: ${L.toFixed(2)} · Intercept shift: +${s.offset.toFixed(2)}`,s.corrected>=.999?'The corrected line returns to the population line.':`Shift the fitted line back by ${(s.offset*s.corrected).toFixed(2)}.`][s.stage];
    const label=s.stage>=5?`${story[s.stage][0]} ${groups.map(g=>`At x ${g.x}: ${count(g.positive)} positives and ${count(g.negative*s.negativeShare)} expected retained negatives. Sampled log odds ${signed(L*(g.x-1)+s.offset)}.`).join(' ')}${s.stage===6?` Prior correction subtracts ${(s.offset*s.corrected).toFixed(2)} from the sampled intercept.`:''}`
      :s.stage===1?'Log of N positive over N divided by N negative over N. N is the total within one x group.'
      :s.stage===2?'The shared totals N cancel, leaving log of N positive divided by N negative.'
      :s.stage===3?`Log of N positive divided by s times N negative. Retain ${(100*s.negativeShare).toFixed(1)} percent of negatives. Expected retained counts are illustrated.`
      :s.stage===4?'Log of N positive divided by s times N negative equals log of N positive divided by N negative minus log s. With s 0.10, the shift is plus 2.30.'
      :'Log of p divided by one minus p equals alpha plus beta x. Logistic regression assumes linear log odds.';
    const svg=`<svg class="scene-chart logistic-chart" viewBox="0 0 ${figure.width} ${figure.height}" role="img" aria-label="${safe(label)}" xmlns="http://www.w3.org/2000/svg"><g font-family="LM, Georgia, serif">${figure.body}</g></svg>`;
    return {stage:s.stage,eyebrow:'01 / Logistic regression',title:story[s.stage][0],description:story[s.stage][1],svg,equation,note:notes[s.stage]};
  }
  window.ImbalanceScenes=window.ImbalanceScenes||{};
  window.ImbalanceScenes.logistic=logistic;
})();
