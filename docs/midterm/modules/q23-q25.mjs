import { m, equation, svg, mathLabel, renderMath, COLORS, format } from '../shared.mjs';

const T = String.raw;
const c = COLORS;
const tx = (x,y,s,fill=c.muted,size=15,anchor='middle') => `<text x="${x}" y="${y}" text-anchor="${anchor}" fill="${fill}" font-size="${size}" font-family="system-ui,sans-serif">${s}</text>`;
const ln = (x1,y1,x2,y2,color=c.line,width=1,dash='') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}" ${dash ? `stroke-dasharray="${dash}"` : ''}/>`;
const dot = (x,y,r,color) => `<circle cx="${x}" cy="${y}" r="${r}" fill="${color}"/>`;
const box = (x,y,w,h,color=c.paper,stroke=c.line) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="9" fill="${color}" stroke="${stroke}"/>`;
const metric = (label,value,color=c.ink) => `<div class="metric"><span>${label}</span><strong style="color:${color}">${value}</strong></div>`;
const clamp = x => Math.max(0,Math.min(1,x));


function setAverage(container,n) {
  container.querySelector('#q23-count').value=n;
  container.querySelector('#q23-readings').value=n;
  container.querySelector('[data-average-chart]').innerHTML=mseChart(n);
  container.querySelector('[data-average-metrics]').innerHTML=metric('Josh MSE (square inches)','16',c.orange)+metric('Second MSE (square inches)',format(36/n,2),c.green);
  container.querySelector('[data-average-note]').textContent=n>=3?'The second carpenter now has lower MSE.':'Josh still has lower MSE.';
  renderMath(container);
}

function ruler(second=false) {
  const x = v => 65+(v-44)/20*470;
  let out = ln(65,235,535,235,c.ink);
  for(let v=44;v<=64;v+=2) out += ln(x(v),235,x(v),243,c.ink)+tx(x(v),266,v);
  out += ln(x(54),68,x(54),235,c.blue,2,'5 5')+tx(x(54),45,'True length: 54',c.blue,17);
  if(second) {
    out += dot(x(48),172,15,c.green)+dot(x(60),172,15,c.green);
    out += tx(x(48),126,'48',c.green,24)+tx(x(60),126,'60',c.green,24);
    out += mathLabel(x(48),208,T`\Pr=\tfrac12`,100,34)+mathLabel(x(60),208,T`\Pr=\tfrac12`,100,34);
    out += tx(x(54),306,'Equally far from the target; mean = 54',c.ink,17);
  } else {
    for(let i=0;i<5;i++) out += dot(x(50),205-i*23,8,c.orange);
    out += tx(x(50)-25,100,'Every reading',c.orange,15,'end')+tx(x(50)-25,123,'is 50',c.orange,18,'end');
    out += ln(x(50),290,x(54),290,c.orange,2)+ln(x(50),283,x(50),297,c.orange,2)+ln(x(54),283,x(54),297,c.orange,2);
    out += mathLabel((x(50)+x(54))/2,323,T`\text{Bias}=-4`,160,36);
  }
  return svg(out,'0 0 600 360', second ? 'The second carpenter reports 48 or 60 with equal probability, centered on the fixed target 54.' : 'Josh always records 50, four inches below the true length 54.');
}

function mseChart(n) {
  const px=v=>72+(v-1)/11*454, py=v=>283-v/40*220;
  let s='';
  for(const v of [0,10,20,30,40]) s+=ln(72,py(v),526,py(v),c.line)+tx(55,py(v)+5,v,c.muted,13);
  s+=ln(72,63,72,283,c.ink)+ln(72,283,526,283,c.ink);
  for(const v of [1,3,6,9,12]) s+=tx(px(v),309,v,c.muted,13);
  s+=mathLabel(304,343,'n',70,30)+mathLabel(75,31,T`\mathrm{MSE}\;(\mathrm{in}^2)`,170,32);
  s+=ln(72,py(16),526,py(16),c.orange,3);
  const path=Array.from({length:111},(_,i)=>{const v=1+i/10;return `${i?'L':'M'}${px(v)},${py(36/v)}`;}).join(' ');
  s+=`<path d="${path}" fill="none" stroke="${c.green}" stroke-width="3"/>`;
  s+=ln(px(n),63,px(n),283,c.blue,1,'4 4')+dot(px(n),py(36/n),6,c.green)+dot(px(n),py(16),6,c.orange);
  s+=tx(526,py(16)-12,'Josh: 16',c.orange,15,'end')+mathLabel(421,py(36/8)-29,T`\text{Second: }36/n`,190,36);
  return svg(s,'0 0 600 365',`MSE versus number of independent readings. Josh stays at 16; the second carpenter has MSE 36 divided by n. At n = ${n}, the second carpenter has MSE ${format(36/n,2)} square inches, compared with Josh’s 16.`);
}


const triples=Array.from({length:8},(_,i)=>[0,1,2].map(j=>(i>>(2-j))&1?60:48))
  .sort((a,b)=>a.reduce((s,v)=>s+v,0)-b.reduce((s,v)=>s+v,0));
const meanValues=[48,52,56,60], meanCounts=[1,3,3,1];

function varianceTable(average=false) {
  const values=average?meanValues:[48,60], deviations=average?[-6,-2,2,6]:[-6,6];
  const weights=average?['1/8','3/8','3/8','1/8']:['1/2','1/2'];
  const pieces=average?['9/2','3/2','3/2','9/2']:['18','18'];
  const xs=average?[235,335,435,535]:[305,475];
  const ys=[57,113,169,225,281];
  let s='';
  [average?'Mean of three':'Reading','Minus mean 54','Square the gap','Probability','Weighted square'].forEach((label,i)=>s+=tx(26,ys[i]+5,label,c.muted,16,'start'));
  values.forEach((v,i)=>{
    const entries=[String(v),String(deviations[i]),String(deviations[i]**2),weights[i],pieces[i]];
    entries.forEach((tex,j)=>s+=mathLabel(xs[i],ys[j],tex,average?90:130,36));
  });
  [86,142,198,254].forEach(y=>s+=ln(22,y,578,y,c.line));
  s+=box(195,310,370,49,'#e7f3ef',c.green)+mathLabel(380,334,average?T`\operatorname{Var}(\bar Z)=12\;\mathrm{in}^2`:T`\operatorname{Var}(Z)=36\;\mathrm{in}^2`,365,40);
  return svg(s,'0 0 600 375',average?'The average of three has deviations minus six, minus two, two, and six. Their squared deviations weighted by probabilities sum to twelve square inches.':'The two readings are six inches below or above the mean. Squaring each deviation gives thirty-six; weighting each by one half gives eighteen plus eighteen, or variance thirty-six square inches.');
}

function triplesTable(showMeans=false) {
  let s=tx(168,28,'Three independent readings',c.ink,17);
  s+=showMeans?mathLabel(454,28,T`\bar Z=(Z_1+Z_2+Z_3)/3`,265,36):mathLabel(450,28,T`\Pr(\text{each triple})=1/8`,270,36);
  triples.forEach((row,i)=>{
    const y=67+i*37,mean=row.reduce((a,b)=>a+b,0)/3;
    s+=tx(36,y+5,i+1,c.muted,13);
    row.forEach((v,j)=>s+=box(70+j*69,y-15,57,30,v===48?'#e8eef4':'#e7f3ef',v===48?c.blue:c.green)+tx(98+j*69,y+6,v,v===48?c.blue:c.green,18));
    if(showMeans) s+=ln(286,y,351,y,c.line,2)+`<path d="M343 ${y-5}l8 5-8 5" fill="none" stroke="${c.line}" stroke-width="2"/>`+box(385,y-15,135,30,'#e7f3ef',c.green)+tx(453,y+6,mean,c.green,20);
    else s+=mathLabel(449,y,T`(1/2)^3=1/8`,220,33);
  });
  s+=tx(300,386,showMeans?'Different triples can produce the same mean.':'Order matters: 48, 48, 60 and 60, 48, 48 are distinct triples.',c.muted,15);
  return svg(s,'0 0 600 405',showMeans?'The eight triples map to means 48, 52, 52, 52, 56, 56, 56, 60.':'Enumerate all eight ordered triples formed from independent readings of 48 or 60. Every triple has probability one eighth.');
}

function meanBins(probabilities=false) {
  const xs=[120,240,360,480],base=273,scale=53;
  let s=tx(300,30,probabilities?'Divide each count by the eight equally likely triples':'Gather identical means into four bins',c.ink,17);
  s+=ln(65,base,535,base,c.ink);
  for(let k=0;k<=3;k++)s+=ln(65,base-k*scale,535,base-k*scale,c.line)+mathLabel(40,base-k*scale,probabilities?`${k}/8`:String(k),65,32);
  meanCounts.forEach((count,i)=>{
    for(let j=0;j<count;j++)s+=box(xs[i]-33,base-(j+1)*scale+4,66,scale-8,'#e7f3ef',c.green)+dot(xs[i],base-(j+.5)*scale,7,c.green);
    s+=mathLabel(xs[i],base-count*scale-22,probabilities?`${count}/8`:String(count),80,34)+tx(xs[i],304,meanValues[i],c.ink,19);
  });
  s+=mathLabel(300,343,probabilities?T`\sum_k\Pr(\bar Z=z_k)=\frac{1+3+3+1}{8}=1`:T`1+3+3+1=8\text{ triples}`,530,40);
  return svg(s,'0 0 600 370',probabilities?'The mean has probabilities one eighth, three eighths, three eighths, one eighth at 48, 52, 56, 60. The probabilities sum to one.':'There is one triple with mean 48, three with mean 52, three with mean 56, and one with mean 60. Each block represents one triple.');
}

function averagingVariance() {
  let s=tx(300,25,'Independent readings: add their variances',c.ink,19);
  [112,300,488].forEach((x,i)=>{
    s+=box(x-78,51,156,75,'#e7f3ef',c.green)+mathLabel(x,76,T`\operatorname{Var}(Z_${i+1})`,150,34)+mathLabel(x,105,'36',100,34);
    if(i<2)s+=mathLabel(x+94,88,'+',35,35);
  });
  s+=mathLabel(300,163,T`\operatorname{Var}(Z_1+Z_2+Z_3)=108`,540,42);
  s+=tx(300,211,'Dividing a variable by 3 divides its variance by 9.',c.ink,17);
  s+=box(112,231,376,65,'#e7f3ef',c.green)+mathLabel(300,263,T`\operatorname{Var}(\bar Z)=\frac{108}{3^2}=12`,370,49);
  s+=mathLabel(300,335,T`\operatorname{Var}\!\left(\frac1n\sum_{i=1}^{n}Z_i\right)=\frac{n\cdot36}{n^2}=\frac{36}{n}`,565,44);
  return svg(s,'0 0 600 365','Three independent variances of thirty-six add to one hundred eight. Scaling the sum by one third scales its variance by one ninth, giving twelve. In general the variance of the average is thirty-six divided by n.');
}

const q23 = {
  id:23,
  kicker:'Bias, variance & averaging',
  title:'Averaging fixes variation. Does it fix error?',
  dek:'Two carpenters, one fixed board, and the difference between being consistent and being accurate.',
  prompt:`<p>Josh is a carpenter who records lengths in inches but always rounds to the nearest ten inches. Apart from rounding, his measurements are exact. Consider repeated measurements of the same board, whose true length is fixed at 54 inches. Assess errors relative to this fixed true length.</p>`,
  parts:[
    `Josh's mean squared error for this board is ${m(T`\mathrm{MSE}(54)=16`)}. Decompose this into squared bias, variance, and irreducible error. State the bias itself as well, and briefly justify each component.`,
    `A second carpenter's measurement is either 48 or 60 inches, each with probability ${m(T`1/2`)}. Calculate this method's bias, variance, and MSE for the same board.`,
    `Each carpenter now averages ${m('n')} independent measurements from their own method. Does averaging improve Josh's method? What is the smallest ${m('n')} for which the second carpenter has lower MSE than Josh?`
  ],
  sections:[{label:'Josh',step:0},{label:'Second carpenter',step:2},{label:'Eight triples',step:4},{label:'Why variance falls',step:8},{label:'Compare MSE',step:10}],
  steps:[
    {title:'Repeated does not mean accurate.',body:`<p>The board stays at 54 inches. Josh records 50 every time, so repeated measurements pile up at the wrong location.</p><p>His bias is ${m('50-54=-4')} inches. Squaring that error gives ${m('16')} square inches.</p>`,caption:'Part (a) · Compare the mean reading with the fixed truth.'},
    {title:'Put the error in the right component.',body:`<p>Josh’s readings never vary, so variance is zero. The target is a fixed, noiseless length, so irreducible error is also zero.</p><p>All his MSE comes from squared bias: ${m('16+0+0=16')}. Bias is measured in inches; squared bias, variance, and MSE are measured in square inches.</p>`,caption:'Part (a) · Systematic rounding accounts for the entire error.'},
    {title:'First find the second carpenter’s center.',body:`<p>The second carpenter records 48 or 60, each with probability ${m('1/2')}. Their mean is ${m('(48+60)/2=54')}.</p><p>The mean equals the true length, so bias is zero. Individual readings still fluctuate six inches to either side; being centered correctly does not make every reading accurate.</p>`,caption:'Part (b) · The mean is 54 inches, matching the fixed board.'},
    {title:'Square the deviations, then average them.',body:`<p>Subtract the mean from each possible reading: ${m('48-54=-6')} and ${m('60-54=6')}. Both squared deviations are ${m('36')}.</p><p>Weight each by its probability ${m('1/2')}: ${m('18+18=36')}. This is the variance. With zero bias and zero irreducible error, MSE is also ${m('36')} square inches.</p>`,caption:'Part (b) · Variance is an average of squared deviations, not an average of signed errors.'},
    {title:'List every ordered triple of readings.',body:`<p>Each of three independent readings can be 48 or 60. Keeping their order gives ${m('2^3=8')} possible triples.</p><p>Independence makes every row equally likely: ${m('(1/2)^3=1/8')}. For example, 48, 48, 60 and 60, 48, 48 are different outcomes even though they will have the same average.</p>`,caption:'Part (c) · Begin with the equally likely outcomes, before drawing a distribution.'},
    {title:'Turn each triple into its average.',body:`<p>Now add the three readings in each row and divide by three. For example, ${m('(48+48+60)/3=52')}.</p><p>Three different triples produce 52; three produce 56. Only the all-48 triple produces 48, and only the all-60 triple produces 60. The table supplies every bar in the next graph.</p>`,caption:'Part (c) · Eight outcomes map to four distinct means.'},
    {title:'Count how many triples land in each bin.',body:`<p>Collect equal averages. The four bins at 48, 52, 56, and 60 contain ${m('1,3,3,1')} triples.</p><p>Every block represents one row of the table. The eight blocks are still the same eight equally likely outcomes; grouping them changes their presentation, not their probabilities.</p>`,caption:'Part (c) · Bar height currently counts triples.'},
    {title:'Convert the counts into probabilities.',body:`<p>Divide each bin’s count by eight. The average is 48, 52, 56, or 60 with probabilities ${m(T`1/8,\;3/8,\;3/8,\;1/8`)}.</p><p>The probabilities sum to one. Six of the eight triples now yield 52 or 56, close to 54; the distribution remains centered at 54.</p>`,caption:'Part (c) · The same blocks now show probability mass, with one block equal to one eighth.'},
    {title:'Check the variance of this new distribution.',body:`<p>The four averages differ from 54 by ${m('-6,-2,2,6')}. Square those gaps, then weight them by the probabilities just constructed.</p><p>The weighted squares are ${m(T`9/2,\;3/2,\;3/2,\;9/2`)}. They sum to ${m('12')} square inches: one third of the single-reading variance ${m('36')}.</p>`,caption:'Part (c) · The distribution itself verifies the variance reduction.'},
    {title:'Why does the rule become thirty-six divided by n?',body:`<p>Independent readings have zero covariance, so the variance of their sum is ${m('36+36+36=108')}. An average divides that sum by three.</p><p>Scaling a variable by ${m('1/3')} scales its variance by ${m('1/9')}, giving ${m('108/9=12')}. With ${m('n')} independent readings, the same reasoning gives ${m('36/n')}.</p>`,caption:'Part (c) · Add independent variances, then square the averaging factor.'},
    {title:'Find the first sample size that wins.',body:`<p>Josh’s average remains 50, so his MSE stays ${m('16')}. The second carpenter’s average is unbiased and has MSE ${m('36/n')}.</p>${equation(T`36/n<16\iff n>2.25`)}<p>At ${m('n=2')}, ${m('18>16')}; at ${m('n=3')}, ${m('12<16')}. Three is the smallest integer that works. Scroll through one, two, and three readings; the slider explores other integer sample sizes.</p>`,caption:'Part (c) · Averaging reduces variance; Josh’s fixed rounding bias remains.'}
  ],
  takeaway:`Josh: bias ${m('-4')}, variance ${m('0')}, MSE ${m('16')}. Second carpenter: bias ${m('0')}, variance and MSE ${m('36')}. Averaging gives ${m(T`36/n`)} for the second method, which first beats Josh at ${m('n=3')}.`,
  render(stage,container) {
    if(stage===0) container.innerHTML=`<h3 class="viz-title">One board, the same reading every time</h3>${ruler()}<p class="viz-note">Each dot is one measurement. The fixed true length is 54 inches.</p>`;
    else if(stage===1) {
      let s=ln(65,250,535,250,c.ink)+tx(300,35,'Error components in square inches',c.muted,17);
      [16,0,0].forEach((v,i)=>{const x=130+i*170;s+=`<rect x="${x-36}" y="${250-v*10}" width="72" height="${Math.max(v*10,2)}" fill="${i===0?c.orange:c.line}" rx="3"/>`+tx(x,250-v*10-18,v,c.ink,26)+tx(x,280,['Squared bias','Variance','Irreducible error'][i],c.muted,15);});
      container.innerHTML=`<h3 class="viz-title">Josh’s MSE: all squared bias</h3>${svg(s,'0 0 600 325','Squared bias sixteen, variance zero, irreducible error zero.')}<div class="metrics">${metric('Bias (inches)',m('-4'),c.orange)}${metric('MSE (square inches)','16')}</div>`;
    } else if(stage===2) container.innerHTML=`<h3 class="viz-title">Balanced around the true length</h3>${ruler(true)}<p class="viz-note">Equal probability on either side gives mean 54 and bias zero.</p>`;
    else if(stage===3) container.innerHTML=`<h3 class="viz-title">From readings to squared deviations</h3>${varianceTable()}<p class="viz-note">The two weighted squares add to the variance.</p>`;
    else if(stage===4||stage===5) container.innerHTML=`<h3 class="viz-title">${stage===4?'All eight ordered triples':'Calculate the mean of every triple'}</h3>${triplesTable(stage===5)}`;
    else if(stage===6||stage===7) container.innerHTML=`<h3 class="viz-title">${stage===6?'Table rows become bin counts':'Counts divided by eight become probabilities'}</h3>${meanBins(stage===7)}<p class="viz-note">${stage===6?'Each block is one equally likely triple.':'The horizontal positions are mean readings, in inches.'}</p>`;
    else if(stage===8) container.innerHTML=`<h3 class="viz-title">The average has variance twelve</h3>${varianceTable(true)}<p class="viz-note">Weight each squared gap by its probability, then add.</p>`;
    else if(stage===9) container.innerHTML=`<h3 class="viz-title">The same result from independence</h3>${averagingVariance()}<p class="viz-note">All variance values use square inches.</p>`;
    else {
      container.dataset.averageManual='false';
      container.innerHTML=`<h3 class="viz-title">Three is the first integer that wins</h3><div data-average-chart>${mseChart(3)}</div><label class="control" for="q23-readings">Number of readings ${m('n')}: <output id="q23-count">3</output><input id="q23-readings" type="range" min="1" max="12" step="1" value="3"></label><div class="metrics" data-average-metrics>${metric('Josh MSE (square inches)','16',c.orange)}${metric('Second MSE (square inches)','12',c.green)}</div><p class="viz-note" data-average-note>At three readings, the second carpenter has lower MSE.</p>`;
      container.querySelector('#q23-readings').addEventListener('input',event=>{container.dataset.averageManual='true';setAverage(container,Number(event.target.value));});
    }
  },
  update(progress,container,stage) {
    if(stage!==10||container.dataset.averageManual==='true')return;
    const n=1+Math.min(2,Math.floor(clamp(progress)*5));
    if(Number(container.querySelector('#q23-readings').value)!==n)setAverage(container,n);
  }
};

const probabilities=[1/3,4/9,8/9], fitted=[.2705,.5702,.8260];
const logisticRisk=t=>probabilities.reduce((sum,p,i)=>sum+(fitted[i]>=t?1-p:p),0)/3;
const fittedLogLoss=-probabilities.reduce((sum,p,i)=>sum+p*Math.log(fitted[i])+(1-p)*Math.log(1-fitted[i]),0)/3;

function decisionRows() {
  let s=mathLabel(48,37,'x',50,36)+tx(213,30,'OLS fitted score',c.blue,18)+tx(445,30,'Logistic fitted probability',c.orange,18);
  s+=mathLabel(213,62,T`\text{pass if }\widehat S\geq80`,215,35)+mathLabel(445,62,T`\text{pass if }\widehat p\geq0.5`,235,35);
  [-1,0,1].forEach((x,i)=>{
    const y=112+i*89;
    s+=box(15,y-25,570,76,i===1?'#fcf0e9':c.paper,i===1?c.orange:c.line);
    s+=mathLabel(48,y+11,String(x),50,35);
    s+=mathLabel(213,y,`${77+2*i}${i===2?'\\geq':'<'}80`,200,34)+tx(213,y+34,i===2?'Pass':'Fail',c.blue,20);
    s+=mathLabel(445,y,`${fitted[i].toFixed(4)}${i===0?'<':'\\geq'}0.5`,240,34)+tx(445,y+34,i===0?'Fail':'Pass',c.orange,20);
  });
  return svg(s,'0 0 600 355','At x minus one: OLS 77 below 80, logistic 0.2705 below 0.5, both fail. At x zero: OLS 79 below 80 predicts fail; logistic 0.5702 above 0.5 predicts pass. At x one: OLS 81 and logistic 0.8260 both predict pass.');
}

function probabilityPlot(threshold=.5) {
  const py=p=>284-220*p, px=i=>135+165*i;
  let s='';
  for(const p of [0,.25,.5,.75,1]) s+=ln(65,py(p),540,py(p),c.line)+tx(47,py(p)+4,p,c.muted,12);
  s+=ln(65,py(threshold),540,py(threshold),c.ink,1.5,'5 5')+tx(540,py(threshold)-10,`Threshold ${format(threshold,2)}`,c.ink,13,'end');
  for(let i=0;i<3;i++) {
    s+=ln(px(i),py(probabilities[i]),px(i),py(fitted[i]),c.line,6);
    s+=dot(px(i)-9,py(probabilities[i]),7,c.blue)+dot(px(i)+9,py(fitted[i]),7,c.orange);
    s+=mathLabel(px(i),310,`x=${i-1}`,80,32);
    s+=tx(px(i),345,fitted[i]>=threshold?'Predict pass':'Predict fail',c.ink,13);
  }
  s+=dot(112,25,6,c.blue)+tx(127,30,'True probability',c.blue,14,'start')+dot(318,25,6,c.orange)+tx(333,30,'Logistic fit',c.orange,14,'start');
  return svg(s,'0 0 600 365',`True pass probabilities and fixed fitted logistic probabilities. At threshold ${format(threshold,2)}, logistic predictions for x = minus one, zero, one are ${fitted.map(p=>p>=threshold?'pass':'fail').join(', ')}. At x zero, the true probability 4/9 is ${probabilities[1]>=threshold?'at or above':'below'} the threshold; the fitted probability 0.5702 is ${fitted[1]>=threshold?'at or above':'below'} it.`);
}
function middleErrors() {
  let out=mathLabel(300,29,T`x=0:\quad \Pr(Y=1\mid X=0)=4/9`,540,40);
  const draw=(y,method,predictPass)=>{
    let row=tx(42,y-24,method,c.ink,17,'start');
    for(let i=0;i<9;i++){
      const x=68+i*48,actualPass=i<4,wrong=actualPass!==predictPass;
      row+=actualPass?dot(x,y+10,12,c.blue):`<rect x="${x-11}" y="${y-1}" width="22" height="22" rx="2" fill="${c.line}"/>`;
      if(wrong)row+=`<circle cx="${x}" cy="${y+10}" r="19" fill="none" stroke="${c.orange}" stroke-width="2.4"/>`;
    }
    row+=mathLabel(535,y+9,predictPass?'5/9':'4/9',85,40);
    return row;
  };
  out+=draw(103,'OLS predicts fail',false)+draw(221,'Logistic predicts pass',true);
  out+=tx(535,69,'Error',c.muted,15);
  out+=dot(62,299,8,c.blue)+tx(78,305,'Actually passes',c.ink,14,'start');
  out+=`<rect x="245" y="291" width="16" height="16" fill="${c.line}"/>`+tx(272,305,'Actually fails',c.ink,14,'start');
  out+=`<circle cx="460" cy="299" r="12" fill="none" stroke="${c.orange}" stroke-width="2"/>`+tx(481,305,'Error',c.orange,14,'start');
  return svg(out,'0 0 600 340','Represent the middle group using nine cases: four pass, five fail. Predicting fail makes four errors; predicting pass makes five. Outlines mark the errors.');
}


function allGroups() {
  let s=tx(300,25,'Three equally sized groups: 9 + 9 + 9 = 27 cases',c.ink,18);
  const centers=[125,300,475],passCounts=[3,4,8];
  centers.forEach((x,i)=>s+=mathLabel(x,60,`x=${i-1}`,130,35));
  [false,true].forEach((logit,j)=>{
    const y=118+j*178;
    s+=tx(28,y-25,logit?'Logistic: same 27 cases':'OLS: 27 cases',c.ink,18,'start');
    centers.forEach((cx,i)=>{
      const predictPass=logit?i>0:i===2;
      for(let k=0;k<9;k++){
        const x=cx-34+(k%3)*34,cy=y+Math.floor(k/3)*29,pass=k<passCounts[i],wrong=pass!==predictPass;
        s+=pass?dot(x,cy,8,c.blue):`<rect x="${x-7}" y="${cy-7}" width="14" height="14" rx="1" fill="${c.line}"/>`;
        if(wrong)s+=`<circle cx="${x}" cy="${cy}" r="12" fill="none" stroke="${c.orange}" stroke-width="2"/>`;
      }
      const mistakes=predictPass?9-passCounts[i]:passCounts[i];
      s+=mathLabel(cx,y+91,T`${mistakes}\text{ ${mistakes===1?'error':'errors'}}`,150,34);
    });
  });
  s+=tx(300,434,'Blue circles pass; gray squares fail; orange outlines mark mistakes.',c.muted,15);
  return svg(s,'0 0 600 455','Each model classifies the same illustrative twenty-seven cases in three equally weighted groups of nine. OLS makes three, four, and one errors; logistic makes three, five, and one.');
}

function errorPlot(groups=3) {
  const colors=[c.blue,c.orange,c.green],numbers=[[3,4,1],[3,5,1]];
  let s=tx(300,25,'Each square is one mistake among 27 cases',c.ink,18);
  numbers.forEach((parts,j)=>{
    const y=93+j*144,total=parts.slice(0,groups).reduce((a,b)=>a+b,0);
    s+=tx(65,y-20,j?'Logistic regression':'OLS + threshold',c.ink,18,'start');
    let n=0;
    parts.slice(0,groups).forEach((count,i)=>{
      for(let k=0;k<count;k++,n++)s+=box(65+n*45,y,39,43,colors[i]+'24',colors[i]);
    });
    s+=mathLabel(536,y+22,`${total}/27`,100,42);
    s+=mathLabel(300,y+76,parts.slice(0,groups).map(n=>`\\frac{${n}}{27}`).join('+')+`=\\frac{${total}}{27}`,520,43);
  });
  for(let i=0;i<3;i++)s+=`<rect x="${84+i*170}" y="352" width="14" height="14" fill="${colors[i]}" opacity="${i<groups?1:.25}"/>`+mathLabel(143+i*170,359,`x=${i-1}`,90,32);
  return svg(s,'0 0 600 385',`After including ${groups} of three groups, counted OLS error is ${numbers[0].slice(0,groups).reduce((a,b)=>a+b,0)} out of 27 and logistic error is ${numbers[1].slice(0,groups).reduce((a,b)=>a+b,0)} out of 27. Only complete contributions are included.`);
}

function lossPipeline() {
  let out='';
  const labels=[['Fit probabilities','Minimize log loss'],['Make a decision','Apply threshold 0.5'],['Evaluate decisions','Count incorrect labels']];
  labels.forEach((a,i)=>{const y=42+i*105;out+=box(120,y,360,65,i===2?'#e7f3ef':c.paper,i===2?c.green:c.blue)+tx(300,y+25,a[0],c.ink,19)+tx(300,y+49,a[1],c.muted,16);if(i<2)out+=`<path d="M300 ${y+70}v26m-7-9 7 9 7-9" stroke="${c.ink}" fill="none" stroke-width="2"/>`;});
  return svg(out,'0 0 600 350','Logistic regression fits by log loss, makes decisions using a threshold, and is evaluated here by misclassification error. These are different steps.');
}


function setThreshold(container,t) {
  container.querySelector('#q24-threshold').value=t;
  container.querySelector('#q24-threshold-value').value=format(t,2);
  container.querySelector('[data-threshold-plot]').innerHTML=probabilityPlot(t);
  container.querySelector('[data-threshold-risk]').innerHTML=metric('Explored logistic error',`${format(logisticRisk(t)*100,2)}%`,c.orange);
  container.querySelector('[data-threshold-decision]').textContent=`Left to right: ${fitted.map(p=>p>=t?'pass':'fail').join(', ')}. Fitted probabilities stay fixed.`;
  renderMath(container);
}

const q24={
  id:24,kicker:'Loss functions & decisions',title:'A binary outcome does not choose the winner.',
  dek:'Track the fitted quantity, the decision threshold, and the metric used to judge the decision.',
  prompt:`<p>A student passes an exam if the numerical score ${m('S')} is at least 80. Let ${m(T`Y=\mathbf{1}\{S\geq80\}`)}, and suppose ${m(T`X\in\{-1,0,1\}`)} takes each value with probability ${m(T`1/3`)}. We compare unpenalized OLS regression of ${m('S')} on ${m('X')} with unpenalized logistic regression of ${m('Y')} on ${m('X')}. Both models include an intercept and a linear term in ${m('X')}.</p><p>In the infinite-data limit, the OLS fitted score is ${m(T`\widehat S_{\mathrm{OLS}}=79+2X`)}. The table gives the true population quantities and the logistic regression's fitted probabilities in this limit.</p><table class="mini-table"><thead><tr><th>${m('x')}</th><th>${m(T`\mathbb{E}[S\mid X=x]`)}</th><th>${m(T`\Pr(Y=1\mid X=x)`)}</th><th>${m(T`\widehat p_{\mathrm{logit}}(x)`)}</th></tr></thead><tbody><tr><td>−1</td><td>77</td><td>${m(T`1/3`)}</td><td>0.2705</td></tr><tr><td>0</td><td>79</td><td>${m(T`4/9`)}</td><td>0.5702</td></tr><tr><td>1</td><td>81</td><td>${m(T`8/9`)}</td><td>0.8260</td></tr></tbody></table><p>The OLS procedure predicts a pass when ${m(T`\widehat S_{\mathrm{OLS}}\geq80`)}. The logistic procedure predicts a pass when ${m(T`\widehat p_{\mathrm{logit}}(X)\geq0.5`)}.</p>`,
  parts:[`Give each procedure's pass/fail prediction at each value of ${m('X')}. Calculate each procedure's population misclassification rate. Which has the lower error?`,`A colleague says, “Because this is a classification problem, you should always choose logistic regression over OLS.” Respond using your result. Explain why fitting the binary outcome does not guarantee lower misclassification error.`],
  sections:[{label:'Decisions',step:0},{label:'Count mistakes',step:2},{label:'Population error',step:4},{label:'Why it happens',step:7},{label:'Explore threshold',step:8}],
  steps:[
    {title:'Apply each threshold to its fitted quantity.',body:`<p>OLS predicts a numerical score, compared with 80. Logistic regression predicts a probability, compared with 0.5.</p><p>Read across each row: the procedures agree at ${m('x=-1')} and ${m('x=1')}. At ${m('x=0')}, score ${m('79<80')} means fail, while probability ${m('0.5702>0.5')} means pass.</p>`,caption:'Part (a) · Both fitted quantities and their threshold decisions are visible.'},
    {title:'Locate the middle-group disagreement.',body:`<p>At ${m('x=0')}, the true pass probability is ${m('4/9<1/2')}. The fitted logistic probability is ${m('0.5702>0.5')}.</p><p>The true and fitted probabilities lie on opposite sides of the threshold. More students in this group fail than pass, yet the logistic procedure predicts pass for the whole group.</p>`,caption:'Part (a) · The fitted probability crosses the cutoff in the wrong direction.'},
    {title:'Count the middle group’s mistakes.',body:`<p>Represent the middle group’s proportions with nine cases: four pass and five fail. These are illustrative counts representing the population probabilities.</p><p>Predicting fail misses four pass cases, giving ${m('4/9')}. Predicting pass misses five fail cases, giving ${m('5/9')}. The orange outlines identify each mistake.</p>`,caption:'Part (a) · Nine cases make the conditional errors visible.'},
    {title:'Expand to all three equally likely groups.',body:`<p>Give each group nine cases, making a representative population of 27. Their equal sizes encode ${m(T`\Pr(X=x)=1/3`)}.</p><p>The outer groups produce three and one mistakes for both procedures. The middle produces four for OLS and five for logistic. Each model is evaluated on the same 27 cases.</p>`,caption:'Part (a) · A case now represents one twenty-seventh of the whole population.'},
    {title:'Start the total with the first group.',body:`<p>At ${m('x=-1')}, both procedures predict fail. Three of the nine cases actually pass, so both make three mistakes.</p>${equation(T`\frac13\times\frac39=\frac3{27}`)}<p>The running total is ${m('3/27')} for each procedure. The other two groups have not yet been added.</p>`,caption:'Part (a) · Add a complete group contribution, measured against all 27 cases.'},
    {title:'Add the group where the decisions differ.',body:`<p>The middle group adds four mistakes for OLS and five for logistic. Because it contains one third of the population, its contributions are ${m('4/27')} and ${m('5/27')}.</p><p>The running totals become ${m('7/27')} and ${m('8/27')}. The one-case difference enters here; the final group still remains.</p>`,caption:'Part (a) · Each new square is an additional mistake among the same 27 cases.'},
    {title:'Add the final group and compare.',body:`<p>At ${m('x=1')}, both predict pass. One of the nine cases fails, so both add ${m('1/27')}.</p>${equation(T`R_{\mathrm{OLS}}=\frac{3+4+1}{27}=\frac8{27}`)}${equation(T`R_{\mathrm{logit}}=\frac{3+5+1}{27}=\frac9{27}`)}<p>OLS has lower misclassification error by ${m('1/27')}. Every group is now included.</p>`,caption:'Part (a) · Final population errors: eight versus nine mistakes out of 27.'},
    {title:'Separate fitting, deciding, and evaluating.',body:`<p>Logistic regression minimizes log loss within its specified probability model. Here that model is misspecified: even its infinite-data fit gives the wrong middle-group decision.</p><p>A binary target does not guarantee the lowest misclassification error. The fitted loss, model restrictions, and threshold matter. This example also does not show that OLS always wins.</p>`,caption:'Part (b) · Fitting probabilities and judging thresholded decisions use different losses.'},
    {title:'Move only the threshold.',body:`<p>The exam uses 0.5, giving logistic error ${m('9/27')}. Scrolling moves the threshold from 0.5 to 0.6: only the middle decision flips, reducing error to ${m('8/27')}.</p><p>The slider explores other cutoffs, where other groups can flip too. Every fitted probability stays fixed, so the model’s log loss remains unchanged.</p>`,caption:'Optional exploration · The exam answer remains based on the stated 0.5 threshold.'}
  ],
  takeaway:`With the stated thresholds: OLS decisions are fail, fail, pass and error is ${m(T`8/27`)}; logistic decisions are fail, pass, pass and error is ${m(T`1/3`)}. A model's fitting loss, restrictions, and decision threshold jointly determine classification performance.`,
  render(stage,container) {
    if(stage===0)container.innerHTML=`<h3 class="viz-title">Fitted values → thresholds → decisions</h3>${decisionRows()}<p class="viz-note">The middle row is the only disagreement.</p>`;
    else if(stage===1)container.innerHTML=`<h3 class="viz-title">The middle probability crosses the cutoff</h3>${probabilityPlot()}<p class="viz-note">In the middle group, the blue probability is below 0.5; the orange fitted probability is above it.</p>`;
    else if(stage===2)container.innerHTML=`<h3 class="viz-title">Which cases become mistakes?</h3>${middleErrors()}<p class="viz-note">Illustrative cases reproduce the exact population proportions.</p>`;
    else if(stage===3)container.innerHTML=`<h3 class="viz-title">Same 27 cases, different middle decisions</h3>${allGroups()}`;
    else if(stage>=4&&stage<=6)container.innerHTML=`<h3 class="viz-title">${stage===6?'All three groups: final error totals':`Running total after ${stage-3} of 3 groups`}</h3>${errorPlot(stage-3)}<p class="viz-note">${stage===6?'OLS: 8 mistakes. Logistic: 9. Both denominators are 27.':'Only the colored groups have been counted so far.'}</p>`;
    else if(stage===7)container.innerHTML=`<h3 class="viz-title">Three separate operations</h3>${lossPipeline()}<p class="viz-note">A lower fitting loss need not produce fewer thresholded mistakes.</p>`;
    else {
      container.dataset.thresholdManual='false';
      container.innerHTML=`<h3 class="viz-title">Optional: fixed model, moving threshold</h3><div data-threshold-plot>${probabilityPlot(.6)}</div><label class="control" for="q24-threshold">Explored threshold: <output id="q24-threshold-value">0.60</output><input id="q24-threshold" type="range" min="0.1" max="0.95" step="0.01" value="0.6"></label><div class="metrics"><div data-threshold-risk>${metric('Explored logistic error','29.63%',c.orange)}</div>${metric('Exam error at 0.50','33.33%',c.ink)}</div><p class="viz-note"><span data-threshold-decision>Left to right: fail, fail, pass. Fitted probabilities stay fixed.</span> Fixed log loss: ${format(fittedLogLoss,3)}.</p>`;
      container.querySelector('#q24-threshold').addEventListener('input',event=>{container.dataset.thresholdManual='true';setThreshold(container,Number(event.target.value));});
    }
  },
  update(progress,container,stage) {
    if(stage!==8||container.dataset.thresholdManual==='true')return;
    const t=progress<.45?.5:.6;
    if(Number(container.querySelector('#q24-threshold').value)!==t)setThreshold(container,t);
  }
};

function fixedInterestModel() {
  let s=tx(300,26,'Illustration: two perfectly redundant interest measures',c.ink,17);
  s+=mathLabel(300,64,T`A=B\in\{0,1,2\}`,400,39);
  s+=box(45,105,190,65,'#e8eef4',c.blue)+mathLabel(140,138,T`A=2`,170,40);
  s+=box(45,207,190,65,c.paper,c.line)+mathLabel(140,240,T`B=2`,170,40);
  s+=`<path d="M235 138H265V166H298m-8-6 8 6-8 6" fill="none" stroke="${c.blue}" stroke-width="2"/>`;
  s+=box(298,115,265,105,'#e8eef4',c.blue)+mathLabel(430,149,T`\hat f(A,B)=0.2+0.3A`,255,40)+mathLabel(430,192,T`\hat f(2,2)=0.8`,245,40);
  s+=tx(140,301,'Unused by this fitted function',c.muted,15)+tx(430,256,'Participation probability',c.blue,16);
  s+=tx(300,345,'Explaining this model holds its fitted function fixed.',c.ink,17);
  return svg(s,'0 0 600 370','Illustrative perfectly redundant interest measures A and B. The existing model uses A to predict probability 0.2 plus 0.3 A and does not use B. At A and B equal two, its probability is 0.8.');
}

function interestAttributions() {
  const ys=[128,202,276],values=['-0.3','0','0.3'];
  let s=mathLabel(300,28,T`\text{Reference mean}=\frac{0.2+0.5+0.8}{3}=0.5`,565,43);
  s+=mathLabel(45,77,T`A=B`,85,36);
  const headers=[['Prediction',157],['Baseline',280],['Attribution to A',425],['To B',552]];
  headers.forEach(([v,x])=>s+=tx(x,77,v,c.muted,16));
  [0,1,2].forEach((v,i)=>{
    const y=ys[i];s+=ln(20,y-34,580,y-34,c.line)+tx(45,y+5,v,c.ink,18);
    s+=mathLabel(157,y,(.2+.3*v).toFixed(1),100,38)+mathLabel(280,y,'0.5',100,38)+mathLabel(425,y,values[i],130,38)+mathLabel(552,y,'0',60,38);
  });
  s+=mathLabel(300,342,T`\phi_A=\hat f(A,B)-0.5,\qquad\phi_B=0`,550,42);
  return svg(s,'0 0 600 370','Under the stated interventional SHAP reference, cases with A and B both equal to 0, 1, 2 have model predictions 0.2, 0.5, 0.8 and baseline 0.5 and attributions to A of minus 0.3, zero, plus 0.3. B receives zero because this function does not use B.');
}

function absoluteAggregation() {
  let s=tx(300,25,'Same three attributions, summarized by magnitude',c.ink,18);
  s+=tx(140,68,'Signed attribution',c.muted,18)+tx(442,68,'Absolute attribution',c.muted,18);
  ['-0.3','0','0.3'].forEach((value,i)=>{
    const y=119+i*62;
    s+=box(60,y-24,160,48,i===0?'#fcf0e9':'#e8eef4',i===0?c.orange:c.blue)+mathLabel(140,y,value,140,38);
    s+=ln(235,y,345,y,c.line,2)+`<path d="M337 ${y-6}l8 6-8 6" fill="none" stroke="${c.line}" stroke-width="2"/>`;
    s+=box(360,y-24,165,48,'#e7f3ef',c.green)+mathLabel(442,y,i===1?'0':'0.3',145,38);
  });
  s+=mathLabel(300,306,T`\operatorname{mean}|\phi_A|=\frac{0.3+0+0.3}{3}=0.2`,560,48);
  s+=mathLabel(300,354,T`\operatorname{mean}(\phi_A)=0\quad\ne\quad\operatorname{mean}|\phi_A|`,560,40);
  return svg(s,'0 0 600 385','Take the absolute value of each attribution before averaging. Minus 0.3 becomes 0.3, zero remains zero, plus 0.3 remains 0.3. The mean absolute attribution is 0.2 even though the signed mean is zero.');
}

function retrainingFlow() {
  let s=tx(300,25,'A different question requires a different fitted model',c.ink,18);
  const ys=[53,157,261];
  const titles=['1. Remove measure A','2. Retrain using measure B','3. Evaluate on the same held-out cases'];
  const formulas=[T`(A,B)\longrightarrow B`,T`\hat g(B)=0.2+0.3B`,T`\hat f(A_i,B_i),\;\hat g(B_i)\quad\text{versus }y_i`];
  ys.forEach((y,i)=>{
    s+=box(42,y,516,76,i===1?'#e7f3ef':c.paper,i===1?c.green:c.line)+tx(300,y+25,titles[i],c.ink,18)+mathLabel(300,y+53,formulas[i],490,39);
    if(i<2)s+=`<path d="M300 ${y+77}v23m-6-7 6 7 6-7" fill="none" stroke="${c.ink}" stroke-width="2"/>`;
  });
  return svg(s,'0 0 600 355','Retrained ablation removes A, fits a new model using B, then compares both models against the same outcomes on the same held-out cases. The new fitted model is 0.2 plus 0.3 B.');
}

function heldoutComparison() {
  let s=tx(300,25,'Same three held-out cases and outcomes for both models',c.ink,17);
  const xs=[62,180,300,422,543];
  s+=mathLabel(xs[0],63,T`A_i=B_i`,115,37)+mathLabel(xs[1],63,T`y_i`,100,37);
  ['Original','Retrained','Squared error'].forEach((v,i)=>s+=tx(xs[i+2],63,v,c.muted,16));
  [0,1,2].forEach((v,i)=>{
    const y=121+i*66,pred=(.2+.3*v).toFixed(1),loss=i===1?'0.25':'0.04';
    s+=ln(20,y-32,580,y-32,c.line)+tx(xs[0],y+5,v,c.ink,18);
    [i===2?'1':'0',pred,pred,loss].forEach((val,j)=>s+=mathLabel(xs[j+1],y,val,j===3?100:110,38));
  });
  s+=mathLabel(300,302,T`\operatorname{Brier}(f)=\operatorname{Brier}(g)=\frac{0.04+0.25+0.04}{3}=0.11`,575,48);
  s+=tx(300,354,'Identical predictions → identical loss on these same cases',c.ink,17);
  return svg(s,'0 0 600 375','On the same three held-out cases, original and retrained models both predict 0.2, 0.5, 0.8. Outcomes are zero, zero, one. Squared errors are 0.04, 0.25, 0.04, giving identical Brier score 0.11.');
}

function referencePredictions(compare=false) {
  let s=tx(300,25,'Numerical example: equally weighted inputs',c.ink,18);
  s+=mathLabel(90,68,'X',100,37)+mathLabel(280,68,T`\hat f(X)=X^2`,230,37);
  if(compare)s+=mathLabel(475,68,T`\hat g(X)=X`,200,37);
  [-1,0,1].forEach((v,i)=>{
    const y=119+i*62;s+=box(45,y-22,90,44,c.paper,c.line)+mathLabel(90,y,String(v),80,36);
    s+=ln(146,y,203,y,c.line,2)+`<path d="M195 ${y-5}l8 5-8 5" fill="none" stroke="${c.line}" stroke-width="2"/>`+box(216,y-22,128,44,'#e8eef4',c.blue)+mathLabel(280,y,String(v*v),110,36);
    if(compare)s+=box(410,y-22,130,44,'#fcf0e9',c.orange)+mathLabel(475,y,String(v),110,36);
    else s+=mathLabel(468,y,T`\text{weight }1/3`,210,36);
  });
  s+=ln(25,282,575,282,c.line)+tx(78,320,'Average',c.ink,17);
  s+=mathLabel(280,320,T`\frac{1+0+1}{3}=\frac23`,260,49);
  if(compare)s+=mathLabel(475,320,T`\frac{-1+0+1}{3}=0`,230,49);
  s+=tx(300,375,'The baseline averages model predictions over the reference.',c.muted,16);
  return svg(s,'0 0 600 400',compare?'Using the same equally weighted reference inputs minus one, zero, one, squared-model predictions one, zero, one average to two thirds. Linear-model predictions minus one, zero, one average to zero.':'Pass the three equally weighted reference inputs minus one, zero, one through the squared model. Their predictions one, zero, one average to two thirds. This expected model prediction is the baseline.');
}

function baselineComparison(compare=false) {
  const px=v=>90+440*v;
  let out=mathLabel(300,26,T`x_i=\tfrac12,\qquad\Pr(X=-1)=\Pr(X=0)=\Pr(X=1)=\tfrac13`,580,42);
  const rows=compare?[{y:108,model:T`\hat f(x)=x^2`,base:2/3,pred:1/4,baseTex:'2/3',predTex:'1/4',phiTex:'-5/12',color:c.blue},{y:245,model:T`\hat g(x)=x`,base:0,pred:1/2,baseTex:'0',predTex:'1/2',phiTex:'1/2',color:c.orange}]:[{y:172,model:T`\hat f(x)=x^2`,base:2/3,pred:1/4,baseTex:'2/3',predTex:'1/4',phiTex:'-5/12',color:c.blue}];
  rows.forEach(r=>{
    out+=mathLabel(100,r.y-41,r.model,170,34)+ln(px(0),r.y,px(1),r.y,c.line,1);
    [0,.25,.5,.75,1].forEach(v=>out+=ln(px(v),r.y-4,px(v),r.y+4,c.line));
    out+=`<path d="M${px(r.base)} ${r.y}H${px(r.pred)}" fill="none" stroke="${r.color}" stroke-width="4"/>`;
    const sign=r.pred>r.base?1:-1;
    out+=`<path d="M${px(r.pred)-sign*8} ${r.y-6}l${sign*8} 6 ${-sign*8} 6" fill="none" stroke="${r.color}" stroke-width="2"/>`;
    out+=`<circle cx="${px(r.base)}" cy="${r.y}" r="7" fill="white" stroke="${c.ink}" stroke-width="2"/>`+dot(px(r.pred),r.y,6,r.color);
    out+=mathLabel(390,r.y-38,T`\phi_x=${r.phiTex}`,210,38);
    const baseLabel=r.pred<r.base?405:100,predLabel=r.pred<r.base?150:365;
    out+=ln(px(r.base),r.y+10,baseLabel,r.y+24,c.muted,1)+ln(px(r.pred),r.y+10,predLabel,r.y+24,r.color,1);
    out+=mathLabel(baseLabel,r.y+43,T`\text{Baseline }${r.baseTex}`,180,36)+mathLabel(predLabel,r.y+43,T`\text{Prediction }${r.predTex}`,200,36);
  });
  out+=`<circle cx="114" cy="337" r="6" fill="white" stroke="${c.ink}" stroke-width="2"/>`+tx(129,343,'Baseline',c.ink,16,'start');
  out+=dot(361,337,6,c.blue)+tx(377,343,'Fitted prediction',c.ink,16,'start');
  return svg(out,'0 0 600 370',compare?'Same focal input one half and same finite reference. For the squared model, baseline two thirds and prediction one quarter give attribution minus five twelfths. For the linear model, baseline zero and prediction one half give attribution one half.':'The focal input one half gives prediction one quarter under the squared model. Subtract the reference baseline two thirds, giving attribution minus five twelfths. The signed arrow runs from the higher baseline to the lower prediction.');
}

function shapInputs() {
  let out='';
  const xs=[30,220,410],labels=['Predictor values','Fitted model','Background'];
  const formulas=['x_i',T`\hat f`,T`P_X`];
  xs.forEach((x,i)=>{out+=box(x,35,160,76,c.paper,c.blue)+tx(x+80,60,labels[i],c.ink,17)+mathLabel(x+80,88,formulas[i],130,34);out+=`<path d="M${x+80} 111V145L300 178" fill="none" stroke="${c.blue}" stroke-width="2"/>`;});
  out+=box(165,178,270,70,'#e7f3ef',c.green)+tx(300,205,'Explain the prediction',c.ink,19)+mathLabel(300,230,T`\phi_1,\ldots,\phi_p`,230,35);
  out+=box(150,289,300,48,'#f4f4f4',c.line)+tx(300,319,'New outcome: not required',c.muted,18);
  return svg(out,'0 0 600 365','Predictor values, the fitted model, and a chosen background enter SHAP. The new observation’s outcome is not an input.');
}

const q25={
  id:25,kicker:'Interpreting fitted predictions',title:'What, exactly, does SHAP explain?',
  dek:'Separate credit within a fitted model from information that survives retraining.',
  prompt:'<p>Consider the following questions about SHAP.</p>',
  parts:[
    `Alice predicts political participation using two strongly correlated measures of political interest and several other predictors. One interest measure has a large mean absolute SHAP value. Removing that measure and retraining barely changes test performance. Explain how both results can be correct. Distinguish what the SHAP summary and the retrained ablation measure.`,
    `Suppose that ${m(T`\hat{f}`)} is a nonlinear model and ${m(T`\hat{g}`)} is a linear, non-constant model trained on the same data with only a single predictor variable. For a single observation, can the SHAP value for the predictor depend on the choice of ${m(T`\hat{f}`)} or ${m(T`\hat{g}`)}? Explain briefly.`,
    `For a fitted model, do you need the observed outcomes of new observations to calculate their SHAP values? Explain briefly.`
  ],
  sections:[{label:'Fixed-model SHAP',step:0},{label:'Mean absolute SHAP',step:2},{label:'Retrain & evaluate',step:3},{label:'Build the baseline',step:5},{label:'Change the model',step:7},{label:'New observations',step:9}],
  steps:[
    {title:'Start with a fitted model and two redundant measures.',body:`<p>Consider an illustration where interest measures ${m('A')} and ${m('B')} are identical. This is an extreme case of strongly correlated predictors.</p><p>The original model uses ${m('A')} to predict participation probability: ${m(T`\hat f(A,B)=0.2+0.3A`)}. It does not use ${m('B')}. SHAP explains this existing fitted function while holding it fixed.</p>`,caption:'Part (a) · Illustrative values show the mechanism; exact redundancy is a limiting case.'},
    {title:'Calculate signed credit in that fixed model.',body:`<p>Use reference profiles ${m('(A,B)=(0,0),(1,1),(2,2)')}, equally weighted. Their predictions ${m('0.2,0.5,0.8')} average to ${m('0.5')}.</p><p>Under interventional SHAP with this reference, ${m('A')} receives ${m('-0.3,0,0.3')}; unused ${m('B')} receives zero. This reference specification matters: other ways of treating correlated features can allocate credit differently.</p>`,caption:'Part (a) · The model, reference, SHAP specification, and probability scale are fixed.'},
    {title:'Take absolute values before averaging.',body:`<p>Mean absolute SHAP summarizes attribution magnitudes. Turn ${m('-0.3,0,0.3')} into ${m('0.3,0,0.3')}, then average: ${m('(0.3+0+0.3)/3=0.2')}.</p><p>The signed mean is zero because positive and negative contributions cancel. The mean absolute value is positive because both directions count as substantial movement from the baseline. It measures attribution in the existing model.</p>`,caption:'Part (a) · A summary of magnitude does not measure the cost of removing a feature.'},
    {title:'Remove the feature and fit another model.',body:`<p>Retrained ablation changes the procedure: remove ${m('A')}, then fit a new model using the remaining predictors.</p><p>In this illustration, ${m('B')} contains exactly the same information. The new fitted model ${m(T`\hat g(B)=0.2+0.3B`)} therefore reproduces the original predictions. We now evaluate a different fitted function, not a different SHAP allocation.</p>`,caption:'Part (a) · The remaining correlated measure can replace the removed information.'},
    {title:'Evaluate both models on the same held-out cases.',body:`<p>Keep the held-out cases and their observed outcomes fixed. Both models predict ${m('0.2,0.5,0.8')} for outcomes ${m('0,0,1')}.</p><p>Each produces squared errors ${m('0.04,0.25,0.04')}, giving the same mean squared probability error, or Brier score: ${m('0.11')}. Large original attributions and unchanged retrained performance are compatible. Imperfectly correlated measures can give a small performance change instead.</p>`,caption:'Part (a) · SHAP asks about this model’s credit; ablation tests predictive information after refitting.'},
    {title:'Build a one-feature baseline from reference predictions.',body:`<p>For part (b), switch to a separate one-feature example with numerical outputs. Use ${m(T`\hat f(x)=x^2`)} and equally weighted reference inputs ${m('-1,0,1')}.</p><p>First pass those inputs through the model. Predictions ${m('1,0,1')} average to ${m('2/3')}. This expected prediction is the baseline. No focal observation or observed outcome is needed to form it.</p>`,caption:'Part (b) · Reference inputs → model predictions → average prediction.'},
    {title:'Subtract that baseline from the focal prediction.',body:`<p>Now introduce the focal input ${m('x_i=1/2')}. The model predicts ${m(T`\hat f(x_i)=(1/2)^2=1/4`)}.</p><p>With one feature, that feature receives the entire difference from the baseline:</p>${equation(T`\phi_x^{(f)}=\hat f(x_i)-\mathbb E[\hat f(X)]=\frac14-\frac23=-\frac5{12}`)}<p>The arrow points left, from the larger baseline to the smaller prediction.</p>`,caption:'Part (b) · A SHAP value is a signed departure from the model’s reference mean.'},
    {title:'Change the model and rebuild its baseline.',body:`<p>Keep the same reference inputs, but consider the linear, nonconstant model ${m(T`\hat g(x)=x`)}. Its reference predictions are ${m('-1,0,1')}, whose mean is zero.</p><p>The reference distribution has not changed. The baseline changes because it averages predictions from the chosen model. The nonlinear model’s baseline remains ${m('2/3')}.</p>`,caption:'Part (b) · Same reference, different fitted functions and different expected predictions.'},
    {title:'Compare the two baseline-to-prediction departures.',body:`<p>At the same ${m('x_i=1/2')}, the nonlinear model gives ${m('1/4-2/3=-5/12')}. The linear model gives ${m('1/2-0=1/2')}.</p><p>SHAP can therefore depend on the fitted model even with one predictor. It attributes centered model predictions, which can change when the model changes. The attributions need not differ at every observation.</p>`,caption:'Part (b) · Same focal input and reference; different fitted models can give different SHAP values.'},
    {title:'Explain the prediction before its outcome arrives.',body:`<p>For a new observation, SHAP needs its predictor values, the fitted model, and the chosen background or reference specification. Its observed outcome is not an input.</p><p>Observed outcomes are needed to evaluate prediction accuracy, as in the held-out comparison. Explaining what a model predicts and measuring whether it predicts well are separate tasks.</p>`,caption:'Part (c) · New outcomes are required for evaluation, not for SHAP calculation.'}
  ],
  takeaway:`SHAP attributes predictions in an existing model; retrained ablation evaluates a new model without a feature. With one predictor, SHAP is prediction minus that model's baseline. New observed outcomes are not required.`,
  render(stage,container) {
    if(stage===0)container.innerHTML=`<h3 class="viz-title">One fitted model uses measure A</h3>${fixedInterestModel()}<p class="viz-note">All values in this example are illustrative.</p>`;
    else if(stage===1)container.innerHTML=`<h3 class="viz-title">Signed attributions around a baseline of 0.5</h3>${interestAttributions()}<p class="viz-note">Interventional SHAP; the three reference profiles are equally weighted.</p>`;
    else if(stage===2)container.innerHTML=`<h3 class="viz-title">Signed values → magnitudes → mean</h3>${absoluteAggregation()}<p class="viz-note">Mean absolute SHAP is 0.2 for A and zero for B in this illustration.</p>`;
    else if(stage===3)container.innerHTML=`<h3 class="viz-title">Remove → retrain → evaluate</h3>${retrainingFlow()}<p class="viz-note">Refitting allows the model to use the remaining information.</p>`;
    else if(stage===4)container.innerHTML=`<h3 class="viz-title">Same outcomes, same predictions, same error</h3>${heldoutComparison()}<p class="viz-note">The last column is ${m(T`(y_i-\hat p_i)^2`)} for either model.</p>`;
    else if(stage===5)container.innerHTML=`<h3 class="viz-title">Construct the nonlinear model’s baseline</h3>${referencePredictions()}<p class="viz-note">This separate one-feature example uses a finite reference distribution and a numerical model output.</p>`;
    else if(stage===6)container.innerHTML=`<h3 class="viz-title">Only now add the focal observation</h3>${baselineComparison()}<p class="viz-note">The arrow runs from baseline ${m('2/3')} to prediction ${m('1/4')}.</p>`;
    else if(stage===7)container.innerHTML=`<h3 class="viz-title">Run the same reference through both models</h3>${referencePredictions(true)}<p class="viz-note">Each baseline averages its own model’s reference predictions.</p>`;
    else if(stage===8)container.innerHTML=`<h3 class="viz-title">Different centered predictions</h3>${baselineComparison(true)}<p class="viz-note">The input and reference stay fixed. The fitted model changes.</p>`;
    else container.innerHTML=`<h3 class="viz-title">What goes into a SHAP calculation?</h3>${shapInputs()}<p class="viz-note">The new outcome can still be unknown.</p>`;
  }
};

export const questions=[q23,q24,q25];
