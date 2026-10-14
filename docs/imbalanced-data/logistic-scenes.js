(() => {
  'use strict';
  const C={blue:'#234e70',orange:'#a54f32',teal:'#007f73',ink:'#242424',muted:'#606060',grid:'#e4e7e9',removed:'#edf0f2'};
  const L=Math.log(99),SHIFT=Math.log(10),groups=[{x:0,positive:100,negative:9900},{x:1,positive:5000,negative:5000}];
  const clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,v)),ease=v=>{const t=clamp(v);return t*t*(3-2*t);};
  const safe=v=>String(v).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  const count=v=>v.toLocaleString('en-US',{maximumFractionDigits:1}),signed=v=>`${v<-.00001?'−':v>.00001?'+':''}${Math.abs(v).toFixed(2)}`;
  const text=(x,y,v,c=C.ink,z=18,a='start',extra='')=>`<text x="${x}" y="${y}" fill="${c}" text-anchor="${a}" style="font-size:${z}px;font-variant-numeric:tabular-nums" ${extra}>${safe(v)}</text>`;
  const math=(x,y,v,z,a='middle',c=C.ink,extra='')=>`<text x="${x}" y="${y}" fill="${c}" text-anchor="${a}" style="font-size:${z}px" ${extra}>${v}</text>`;
  const sub=(v,z)=>`<tspan baseline-shift="sub" style="font-size:${z*.62}px">${v}</tspan>`,n=(v,z)=>`N${sub(v,z)}`;
  const line=(x1,y1,x2,y2,c=C.grid,extra='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${c}" ${extra}/>`;
  const dot=(x,y,c,r=5.5,extra='')=>`<circle cx="${x}" cy="${y}" r="${r}" fill="${c}" stroke="white" stroke-width="1.6" ${extra}/>`;
  const story=[
    ['Logistic regression assumes linear log odds.','The logarithm of p / (1 − p) is a straight line in x. For this example, assume that relationship is also the truth.'],
    ['Write the probabilities as counts.','Within a group with the same x, the positive share is N₊ / N and the negative share is N₋ / N.'],
    ['Cancel the shared total.','N drops out. Log odds is the logarithm of the positive-to-negative count ratio.'],
    ['Keep positives. Thin the negatives.','Keep every positive and randomly retain the same share s of negatives in each x group.'],
    ['One sampling share, one constant shift.','Dividing the odds by s subtracts log s from the log odds. Every group gets the same shift.'],
    ['Only the intercept changes.','With 10% of negatives retained, both points rise by 2.30. The sampled line is parallel to the population line.'],
    ['Prior correction shifts it back.','Add log s to the sampled intercept to recover the population line: the prior correction in King and Zeng’s rare-events work.']
  ];
  function state(progress){
    const p=clamp(Number(progress)||0,0,6),stage=Math.round(p),action=ease((p-stage+.5)/.5);
    // Integer stages are completed snapshots, also used when motion is off.
    const negativeShare=stage===3||stage===5?Math.exp(-SHIFT*action):stage>=4?.1:1;
    return {p,stage,action,negativeShare,offset:-Math.log(negativeShare),corrected:stage===6?action:0};
  }
  function brackets(a,b,top,bottom){return `<path d="M${a+7},${top}H${a}V${bottom}H${a+7} M${b-7},${top}H${b}V${bottom}H${b-7}" fill="none" stroke="${C.ink}" stroke-width="1.6"/>`;}
  // Live SVG text and fraction rules let the common Ns cancel as we scroll.
  function logRatio(left,y,z,{sampled=false,normalized=false,cancel=0,probability=false}={}){
    const w=z*(normalized?3.1:sampled?2.65:probability?2.3:2.05),lw=z*1.27,fx=left+lw+13,cx=fx+w/2;
    let body=math(left,y+z*.37,'log',z,'start');
    if(normalized){
      body+=math(cx,y-z*1.7+cancel*(z*1.7-12),n('+',z),z,'middle',C.orange);
      body+=math(cx,y+z+8,n('−',z),z,'middle',C.blue);
      body+=`<g opacity="${1-cancel}">`+math(cx,y-8,'N',z*.88,'middle',C.muted)+math(cx,y+z*2.4+8,'N',z*.88,'middle',C.muted);
      body+=line(cx-w*.36,y-z*.92,cx+w*.36,y-z*.92,C.ink,'stroke-width="1.5"')+line(cx-w*.36,y+z*1.75+8,cx+w*.36,y+z*1.75+8,C.ink,'stroke-width="1.5"');
      const opacity=Math.sin(Math.PI*cancel);
      body+=line(cx-15,y-34,cx+15,y-3,C.orange,`stroke-width="2.4" opacity="${opacity}"`)+line(cx-15,y+z*2.4-18,cx+15,y+z*2.4+13,C.orange,`stroke-width="2.4" opacity="${opacity}"`);
      body+='</g>'+brackets(fx-6,fx+w+6,y-z*2+cancel*(z*2-z-20),y+z*2.75+10-cancel*(z*1.75-5));
    }else{
      body+=math(cx,y-12,probability?'p':n('+',z),z,'middle',probability?C.ink:C.orange);
      body+=math(cx,y+z+8,probability?'1 − p':`${sampled?'s ':''}${n('−',z)}`,z,'middle',probability?C.ink:C.blue);
      body+=brackets(fx-6,fx+w+6,y-z-20,y+z+25);
    }
    body+=line(fx,y,fx+w,y,C.ink,'stroke-width="1.7"');
    return {body,width:lw+w+26};
  }
  function equations(s,narrow){
    const width=narrow?380:900,height=narrow?400:420,z=narrow?32:46,cx=width/2;
    let body='';
    if(s.stage===0||s.stage===1){
      const ratioWidth=z*(1.27+2.3)+26,rhsWidth=z*3.85,left=cx-(ratioWidth+rhsWidth)/2,ratio=logRatio(left,153,z,{probability:true});
      body+=`<g opacity="${s.stage===0?1:1-s.action}">${ratio.body}`+math(left+ratio.width+9,153+z*.37,'= α + βx',z,'start')+text(cx,289,'α = intercept     β = slope',C.muted,narrow?18:24,'middle')+'</g>';
      if(s.stage===1){
        const w=z*(1.27+3.1)+26;
        body+=`<g opacity="${s.action}">${logRatio(cx-w/2,149,z,{normalized:true}).body}`+math(cx,narrow?312:347,`N = ${n('+',z*.68)} + ${n('−',z*.68)}`,z*.68)+text(cx,narrow?347:382,'The total within this x group',C.muted,narrow?16:21,'middle')+'</g>';
      }
    }else if(s.stage===2){
      const w=z*(1.27+3.1)+26;
      body+=logRatio(cx-w/2,149,z,{normalized:true,cancel:s.action}).body;
      body+=text(cx,narrow?310:344,'Only the positive-to-negative ratio remains.',C.muted,narrow?16:22,'middle');
    }else if(s.stage===4){
      const w1=z*(1.27+2.65)+26,w2=z*(1.27+2.05)+26;
      if(narrow){
        body+=logRatio(cx-w1/2,89,z,{sampled:true}).body;
        const total=w2+z*4.35,left=cx-total/2;
        body+=`<g opacity="${s.action}">${math(left,224+z*.37,'=',z,'start')}`+logRatio(left+z*1.05,224,z).body+math(left+z*1.05+w2+10,224+z*.37,'− log s',z,'start')+'</g>';
        body+=math(cx,353,'− log(0.10) = +2.30',26,'middle',C.orange,`opacity="${s.action}"`);
      }else{
        const total=w1+w2+z*4.35,left=cx-total/2;
        body+=logRatio(left,159,z,{sampled:true}).body;
        body+=`<g opacity="${s.action}">${math(left+w1+9,159+z*.37,'=',z,'start')}`+logRatio(left+w1+z*1.05,159,z).body+math(left+w1+z*1.05+w2+10,159+z*.37,'− log s',z,'start')+'</g>';
        body+=math(cx,317,'− log(0.10) = +2.30',34,'middle',C.orange,`opacity="${s.action}"`);
      }
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
    const width=narrow?380:900,height=narrow?400:420,z=narrow?32:46,w=z*(1.27+2.65)+26;
    let body=logRatio(width/2-w/2,narrow?73:82,z,{sampled:true}).body;
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
    const fx=x=>narrow?88+x*203:199+x*499,fy=v=>bottom-(v+5.5)/9*(bottom-top),corrected=s.stage===6,offset=s.offset*(1-s.corrected),z=narrow?25:34;
    let body=corrected?math(width/2,31,`α${sub('sampled',z)} + log(0.10) = α${sub('population',z)}`,narrow?21:32,'middle',C.teal):math(width/2,31,'(α − log s) + βx',z);
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
      'N₊ counts positives, N₋ counts negatives, and N = N₊ + N₋ within this x group.',
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
