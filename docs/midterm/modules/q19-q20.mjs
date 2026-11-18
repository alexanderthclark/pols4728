import {m, equation, svg, mathLabel, renderMath, COLORS} from '../shared.mjs';

const C=COLORS;
const txt=(x,y,s,attrs='')=>`<text x="${x}" y="${y}" fill="${C.ink}" font-size="20" ${attrs}>${s}</text>`;
const line=(x1,y1,x2,y2,color=C.line,extra='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" ${extra}/>`;
const box=(x,y,w,h,label,color=C.blue)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="7" fill="${color}" fill-opacity=".09" stroke="${color}"/>${label?txt(x+w/2,y+h/2+6,label,'text-anchor="middle"'):''}`;
const M=(x,y,tex,width=150,height=44,size=22)=>mathLabel(x,y,tex,width,height).replace('class="svg-math"',`class="svg-math" style="font-size:${size}px"`);
const title=s=>`<div class="viz-title">${s}</div>`;
const note=s=>`<p class="viz-note">${s}</p>`;
const dots=[{id:'A',x:42,d:0},{id:'B',x:48,d:1},{id:'C',x:54,d:0},{id:'D',x:58,d:1},{id:'E',x:62,d:0},{id:'F',x:67,d:1},{id:'G',x:72,d:0},{id:'H',x:78,d:1}];
const baseline=x=>50+40*(x>=60);
const outcome=d=>baseline(d.x)+10*d.d;
const studentColor=d=>d.d?C.blue:C.ink;
const xp=x=>70+(x-40)*11.5;
const yp=y=>285-(y-40)*3.6;
const q19Key=()=>`<div class="chips"><span class="chip">A, C, E, G: ${m('D=0')}</span><span class="chip" style="color:${C.blue}">B, D, F, H: ${m('D=1')}</span></div>`;
function q19Plot(memorize=false){
  let p=line(60,285,555,285)+line(60,45,60,285);
  [40,50,60,70,80].forEach(x=>p+=txt(xp(x),308,x,'text-anchor="middle"'));
  [40,60,80,100].forEach(y=>p+=txt(47,yp(y)+5,y,'text-anchor="end"'));
  p+=M(305,340,'X_i',90,36)+M(25,25,'Y_i',65,36);
  if(!memorize){
    p+=`<path d="M${xp(40)},${yp(50)} H${xp(60)} M${xp(60)},${yp(90)} H${xp(80)}" stroke="${C.ink}" stroke-width="3" fill="none"/>`;
    p+=`<path d="M${xp(40)},${yp(60)} H${xp(60)} M${xp(60)},${yp(100)} H${xp(80)}" stroke="${C.blue}" stroke-width="3" fill="none"/>`;
    p+=line(xp(73),yp(90),xp(73),yp(100),C.orange,'stroke-width="3"')+M(xp(73)+37,yp(95),'10',50,32);
  }else{
    let path='';
    dots.forEach((d,i)=>{const a=i===0?40:(dots[i-1].x+d.x)/2,b=i===dots.length-1?80:(d.x+dots[i+1].x)/2;path+=`M${xp(a)},${yp(outcome(d))}H${xp(b)}`;if(i<dots.length-1)path+=`V${yp(outcome(dots[i+1]))}`;p+=line(xp(b),45,xp(b),285,C.line,'stroke-dasharray="3 5"');});
    p+=`<path d="${path}" stroke="${C.orange}" stroke-width="3" fill="none"/>`;
  }
  dots.forEach(d=>{const x=xp(d.x),y=yp(outcome(d));p+=`<circle cx="${x}" cy="${y}" r="6" fill="${studentColor(d)}" stroke="${C.paper}" stroke-width="2"/>`+txt(x,y+(d.d?-16:25),d.id,`text-anchor="middle" style="fill:${studentColor(d)}"`);});
  return svg(p,'0 0 600 365',memorize?'The same students A through H each occupy one leaf. The tree predicts each observed outcome exactly.':'Eight illustrative students A through H. Tutoring adds ten at every pre-tutoring score. Black students are untutored and blue students are tutored.');
}
function render19(stage,container){
  if(stage===0){
    container.innerHTML=title('Follow these eight students')+q19Plot()+q19Key()+note('Illustrative sample; the true vertical treatment gap is 10 throughout.');
  }else if(stage===1){
    container.innerHTML=title('One student in each leaf')+q19Plot(true)+q19Key()+equation('\\widehat y(X_i)=Y_i')+note('Keep the letters and colors: the tree memorizes each complete outcome.');
  }else if(stage===2){
    let p=txt(60,29,'Student','text-anchor="middle"')+M(200,24,'Y_i')+M(340,24,'\\widehat y(X_i)')+M(500,24,'R_i',130)+line(20,47,580,47);
    dots.forEach((d,i)=>{const y=73+i*33,v=outcome(d),c=studentColor(d);p+=`<rect x="20" y="${y-15}" width="560" height="30" rx="3" fill="${c}" fill-opacity=".045"/>`+txt(60,y+6,d.id,`text-anchor="middle" style="fill:${c}"`)+M(200,y,String(v),100,32)+M(340,y,String(v),100,32)+M(500,y,`${v}-${v}=0`,155,32);});
    container.innerHTML=title('Subtract each student’s own outcome')+svg(p,'0 0 600 330','Rows retain all eight students and their treatment colors. Each observed outcome minus its exact prediction is zero.')+q19Key()+note('A through H all have zero residuals; treatment status does not change that arithmetic.');
  }else if(stage===3){
    let p='';
    [0,1].forEach(d=>{const x=d?315:15,c=d?C.blue:C.ink;p+=box(x,18,270,245,'',c)+M(x+135,48,`D=${d}`,130);dots.filter(s=>s.d===d).forEach((s,i)=>{const xx=x+55+(i%2)*150,yy=95+Math.floor(i/2)*59;p+=txt(xx-22,yy+5,s.id,`text-anchor="middle" style="fill:${c}"`)+M(xx+27,yy,'0',48,35);});p+=M(x+135,218,'\\frac{0+0+0+0}{4}=0',255,60);});
    p+=M(300,306,'\\widehat\\tau=\\overline R_{D=1}-\\overline R_{D=0}=0',570,52);
    container.innerHTML=title('Now compare the two group means')+svg(p,'0 0 600 345','Untutored students A C E G each have residual zero, as do tutored students B D F H. Both group means and their difference are zero.')+note('The true effect is 10. This in-sample residual comparison reports 0.');
  }else if(stage===4){
    let p=box(25,30,220,65,'Training students',C.blue)+box(355,30,220,65,'Evaluation students',C.green)+line(135,95,135,152,C.blue,'stroke-width="3"')+box(25,152,220,65,'Fit and freeze tree',C.blue)+line(245,184,355,184,C.blue,'stroke-width="3"')+txt(300,168,'predict','text-anchor="middle"')+box(355,152,220,65,'Find residuals',C.green)+line(465,95,465,152,C.green,'stroke-width="3"')+line(465,217,465,274,C.green,'stroke-width="3"')+box(355,274,220,55,'Compare groups',C.green)+txt(135,273,'Evaluation outcomes','text-anchor="middle"')+txt(135,299,'never enter this fit.','text-anchor="middle"');
    container.innerHTML=title('Separate fitting from the effect comparison')+svg(p,'0 0 600 355','Training outcomes determine a frozen tree. Only the independent evaluation students supply the residual comparison.')+note('For the next example, A–H are training students; student I is held out.');
  }else if(stage===5){
    const x=v=>80+(v-44)*53,y=v=>265-(v-40)*7;
    let p=`<rect x="${x(45)}" y="50" width="${x(51)-x(45)}" height="215" fill="${C.orange}" fill-opacity=".09"/>`+line(65,265,555,265)+line(65,45,65,265)+line(x(45),y(60),x(51),y(60),C.orange,'stroke-width="4"');
    [45,48,49,51].forEach(v=>p+=txt(x(v),290,v,'text-anchor="middle"'));
    [40,50,60,70].forEach(v=>p+=txt(50,y(v)+6,v,'text-anchor="end"'));
    p+=`<circle cx="${x(48)}" cy="${y(60)}" r="8" fill="${C.blue}"/><circle cx="${x(49)}" cy="${y(50)}" r="8" fill="${C.ink}"/>`+txt(x(48)-12,y(60)-21,'B: training','text-anchor="middle"')+txt(x(49)+83,y(50)+7,'I: held out','text-anchor="middle"')+line(x(49),y(50),x(49),y(60),C.green,'stroke-width="3" stroke-dasharray="4 4"')+M(x(49)+35,y(55),'-10',65,36)+M(300,329,'X_i',100)+M(27,25,'Y_i',60);
    container.innerHTML=title('The frozen leaf need not fit a new student')+svg(p,'0 0 600 355','Training student B has score 48 and outcome 60. Held-out student I has score 49 and outcome 50. The same leaf predicts 60 for I, leaving residual minus ten.')+equation('R_I=50-60=-10')+note(`Illustrative student I: ${m('X_I=49,\\ D_I=0')}. The frozen leaf is ${m('45\\le X<51')}. This is one residual, not the treatment estimate.`);
  }else if(stage===6){
    let p='';
    [0,1].forEach(d=>{const y=82+d*142,c=d?C.blue:C.ink;p+=M(70,y-16,`D=${d}`,100)+line(150,y+42,545,y+42)+`<rect x="170" y="${y-34}" width="350" height="76" fill="${c}" fill-opacity=".12" stroke="${c}" stroke-width="2"/>`+txt(170,y+68,'40','text-anchor="middle"')+txt(520,y+68,'80','text-anchor="middle"')+txt(345,y+10,'Same distribution of scores','text-anchor="middle"');});
    container.innerHTML=title('Random assignment balances the input distribution')+svg(p,'0 0 600 335','Both evaluation treatment groups have the same distribution of scores. An illustrative uniform distribution from 40 to 80 is identical in both rows.')+equation('X\\mid D=0\\ \\overset{d}{=}\\ X\\mid D=1')+note(`The rectangles illustrate ${m('X\\sim\\mathrm{Uniform}(40,80)')}. This example is not an additional exam assumption; random assignment gives equality for any score distribution.`);
  }else if(stage===7){
    let p=box(20,35,150,85,'',C.ink)+M(95,59,'D=0',100)+txt(95,99,'Same inputs','text-anchor="middle"')+box(20,210,150,85,'',C.blue)+M(95,234,'D=1',100)+txt(95,274,'Same inputs','text-anchor="middle"')+box(225,100,170,130,'',C.orange)+txt(310,133,'One frozen','text-anchor="middle"')+txt(310,161,'prediction rule','text-anchor="middle"')+M(310,201,'\\widehat y(X)',160)+line(170,77,225,130,C.ink,'stroke-width="3"')+line(170,252,225,200,C.blue,'stroke-width="3"')+line(395,130,460,77,C.ink,'stroke-width="3"')+line(395,200,460,252,C.blue,'stroke-width="3"')+M(515,77,'c=75',140,55,26)+M(515,252,'c=75',140,55,26);
    container.innerHTML=title('The same inputs through the same rule give the same mean')+svg(p,'0 0 600 325','Identically distributed inputs pass through one frozen prediction function. Both groups therefore have the same expected prediction, illustrated by 75.')+equation('\\mathbb E[\\widehat y(X)\\mid D=0,\\widehat y]=\\mathbb E[\\widehat y(X)\\mid D=1,\\widehat y]=c')+note('For the uniform-score example and the frozen A–H tree, both expected predictions are 75. Equality follows from randomization, not from the tree being accurate.');
  }else if(stage===8){
    let p=txt(300,28,'Adjustment applied: 75 of the common 75','text-anchor="middle" data-common-subtraction=""');
    const x=v=>95+(v+10)*4.5;
    [-10,0,20,40,60,80].forEach(v=>p+=line(x(v),57,x(v),249,C.line,'stroke-dasharray="3 5"')+txt(x(v),276,v,'text-anchor="middle"'));
    p+=M(44,86,'D=0',80)+M(44,194,'D=1',80);
    [[70,110,C.ink],[80,218,C.blue]].forEach(([v,y,c])=>{p+=`<circle cx="${x(v)}" cy="${y}" r="8" fill="white" stroke="${c}" stroke-width="2" opacity=".3"/><g data-expected-point="" data-original="${v}" data-y="${y}" transform="translate(${x(v-75)} ${y})"><circle r="9" fill="${c}"/><text data-expected-value="" x="0" y="-20" text-anchor="middle" fill="${c}" font-size="22">${v-75}</text></g>`;});
    p+=`<line data-expected-connector="" x1="${x(-5)}" y1="110" x2="${x(5)}" y2="218" stroke="${C.orange}" stroke-width="2" stroke-dasharray="4 4"/>`+M(300,322,'(80-c)-(70-c)=10',510,46);
    container.innerHTML=title('Subtracting equal amounts preserves the gap')+svg(p,'0 0 600 350','Expected group means 70 and 80 both move left by 75, to minus five and five. Their difference remains ten.')+note(`The frozen expected prediction stays ${m('c=75')}. Scroll to apply it equally to both means: outcomes 70 and 80 become residuals −5 and 5.`);
  }else{
    let p=line(75,210,545,210,C.line,'stroke-width="2"')+line(310,55,310,252,C.orange,'stroke-width="3" stroke-dasharray="5 5"');
    [-190,-125,-80,-37,22,67,111,165].forEach((dx,i)=>p+=`<circle cx="${310+dx}" cy="${155-(i%3)*24}" r="8" fill="${C.blue}" opacity=".75"/>`);
    p+=txt(300,29,'Repeated evaluation samples','text-anchor="middle"')+M(310,280,'\\mathbb E[\\widehat\\tau\\mid\\widehat y]=10',450,52)+txt(300,328,'Individual estimates can lie on either side.','text-anchor="middle"');
    container.innerHTML=title('An expected effect of 10 can still be estimated noisily')+svg(p,'0 0 600 355','A schematic collection of effect estimates varies around an expected effect of ten. The question does not specify the variance.')+note('Schematic spread, not a simulated sampling distribution. Both evaluation groups must be present for their mean difference to be defined.');
  }
  renderMath(container);
}

const q19={
  id:19,kicker:'Required · Partialing out',title:'When prediction erases the effect',
  dek:'Follow the same students into the tree, through the subtraction, and into the group comparison. Then change who is evaluated.',
  prompt:`<p>A researcher studies tutoring's effect on final exam scores ${m('Y_i')}. Let ${m('X_i')} be a continuous pre-tutoring score and ${m('D_i\\in\\{0,1\\}')} indicate tutoring. Suppose the causal relationship is</p>${equation('Y_i=50+40\\,\\mathbf{1}\\{X_i\\ge60\\}+10D_i.')}<p>The researcher predicts ${m('Y')} from ${m('X')} alone and proposes estimating the tutoring effect using adjusted outcomes after partialing out ${m('X')}:</p>${equation('R_i=Y_i-\\widehat y(X_i),\\qquad\\widehat\\tau=\\overline R_{D=1}-\\overline R_{D=0}.')}<p>Tutoring is randomly assigned with probability ${m('1/2')}, independently of ${m('X')}. All observed ${m('X_i')} values are distinct.</p>`,
  parts:[`Imagine fitting a tree to the full sample with one student per leaf. Find ${m('\\widehat\\tau')}. Does this recover tutoring's effect? Why does excluding ${m('D')} from the predictors not prevent the problem?`,`Imagine randomly split students into training and evaluation groups, independently of treatment assignment, fitting the tree on the training group, and computing ${m('\\widehat\\tau')} on the evaluation group. Why does perfect training prediction no longer force the result in part (a)? Why does random assignment preserve the expected treatment-group difference after adjustment? Does this guarantee a precise estimate?`],
  sections:[{label:'Memorization',step:0},{label:'Held-out students',step:4},{label:'Equal adjustment',step:6},{label:'Precision',step:9}],
  steps:[
    {title:'The treatment effect is 10.',body:`<p>The baseline jumps from 50 to 90 at ${m('X_i=60')}. Tutoring adds 10 on either side.</p><p>Follow the eight labeled students. A, C, E and G are untutored; B, D, F and H are tutored. These identities and colors stay fixed throughout the in-sample comparison.</p>`,caption:'Setup · The vertical gap is the treatment effect.'},
    {title:'Each leaf remembers one complete outcome.',body:`<p>A tree with one student per leaf predicts that student's observed outcome exactly: ${m('\\widehat y(X_i)=Y_i')}.</p><p>The distinct scores identify individual students. Their outcomes already contain the treatment contribution, so leaving ${m('D')} out of the predictors does not prevent memorization.</p>`,caption:'Part (a) · The same eight students now occupy eight leaves.'},
    {title:'Carry every student through the subtraction.',body:`<p>Read across any row: observed outcome, memorized prediction, then their difference. A has ${m('50-50=0')}; B has ${m('60-60=0')}. The same arithmetic holds for all eight students.</p><p>Adjustment removes each complete outcome, including its treatment contribution.</p>`,caption:'Part (a) · Every identity remains visible as its residual becomes zero.'},
    {title:'Four zeros versus four zeros gives zero.',body:`<p>Now gather the residuals by treatment status. A, C, E and G average to zero. B, D, F and H also average to zero.</p><p>The proposed estimate is ${m('\\widehat\\tau=0-0=0')}. It fails to recover the true ten-point effect.</p>`,caption:'Part (a) · Individual residuals become the two group means.'},
    {title:'Use other students to evaluate the fitted tree.',body:`<p>Split students independently of treatment assignment. Fit the tree using training students and freeze it. Compute residuals and compare groups using evaluation students.</p><p>Evaluation outcomes never enter the fit. Perfect prediction on training students no longer forces evaluation residuals to be zero.</p>`,caption:'Part (b) · Separate the source of the fit from the source of the comparison.'},
    {title:'One held-out student makes the difference concrete.',body:`<p>Training student B has ${m('X_B=48')}, is tutored, and scores 60. B's leaf predicts 60 for scores from 45 up to 51.</p><p>New student I has ${m('X_I=49')}, is untutored, and scores 50. The frozen prediction is 60, so I's residual is −10. Perfect training fit does not force a zero here.</p>`,caption:'Part (b) · A nearby score enters the same leaf but need not share its outcome.'},
    {title:'First compare the distributions entering the tree.',body:`<p>In the evaluation population, random assignment makes the score distribution the same for tutored and untutored students, conditional on the independently fitted tree.</p><p>The identical rectangles illustrate a uniform distribution from 40 to 80. A particular evaluation sample can still have chance imbalance.</p>`,caption:'Part (b) · Equal distributions in expectation do not require identical realized samples.'},
    {title:'Then pass both distributions through one frozen rule.',body:`<p>The tree uses only ${m('X')} and stays fixed. Identically distributed inputs therefore produce identically distributed predictions. Both groups have the same expected prediction, ${m('c')}.</p><p>For the illustrative uniform distribution and the displayed eight-leaf tree, ${m('c=75')}. Predictive accuracy is not needed for this equality.</p>`,caption:'Part (b) · Equal input distributions imply equal expected adjustments.'},
    {title:'Now subtract that common expected prediction.',body:`<p>In the example, expected outcomes are 70 without tutoring and 80 with tutoring. Subtracting 75 produces expected residuals of −5 and 5.</p><p>Their difference remains 10. More generally, the common ${m('c')} cancels: ${m('(\\mathbb E[Y\\mid D=1]-c)-(\\mathbb E[Y\\mid D=0]-c)=10')}.</p>`,caption:'Part (b) · Both levels shift by the same amount; the expected gap stays fixed.'},
    {title:'A correct expectation can still be noisy.',body:`<p>Provided both evaluation groups are present, the expected adjusted difference is 10. One realized estimate need not equal 10.</p><p>A small evaluation sample, chance score imbalance, or an unstable tree can produce substantial variation. Independent evaluation removes the forced-zero result; it does not guarantee precision.</p>`,caption:'Part (b) · Separate the expected answer from its sampling variability.'}
  ],takeaway:`In-sample memorization gives ${m('\\widehat\\tau=0')}. Independent evaluation and random assignment preserve an expected difference of 10, without guaranteeing a precise realized estimate.`,render:render19,
  update(progress,container,stage){
    if(stage!==8)return;
    const a=Math.max(0,Math.min(1,progress/.65)),adjustment=75*a,x=v=>95+(v+10)*4.5;
    container.querySelectorAll('[data-expected-point]').forEach(point=>{const value=Number(point.dataset.original)-adjustment;point.setAttribute('transform',`translate(${x(value)} ${point.dataset.y})`);const label=point.querySelector('[data-expected-value]');if(label)label.textContent=value.toFixed(a===1?0:1);});
    const connector=container.querySelector('[data-expected-connector]');if(connector){connector.setAttribute('x1',x(70-adjustment));connector.setAttribute('x2',x(80-adjustment));}
    const label=container.querySelector('[data-common-subtraction]');if(label)label.textContent=`Adjustment applied: ${adjustment.toFixed(a===1?0:1)} of the common 75`;
  }
};

function rateChart(){
  let p='';const x=v=>150+v*17;
  [0,5,10,15,20].forEach(v=>p+=line(x(v),62,x(v),260,C.line)+txt(x(v),289,`${v}%`,'text-anchor="middle"'));
  [['One',2,112],['Six or more',20,215]].forEach(([label,rate,y])=>p+=txt(137,y+6,label,'text-anchor="end"')+`<rect x="150" y="${y-17}" width="${rate*17}" height="34" rx="3" fill="${C.green}" fill-opacity=".65"/>`+txt(x(rate)+13,y+6,`${rate}%`));
  p+=line(x(8),62,x(8),260,C.orange,'stroke-width="3" stroke-dasharray="7 5"')+txt(x(8),34,'Both predicted 8%','text-anchor="middle"');
  return svg(p,'0 0 600 315','The observed donation rates are two percent and twenty percent, but both groups receive an eight percent prediction.');
}
function eventHistory20(future=false){
  const events=[{x:70,date:'May 10',amount:20},{x:175,date:'Jul 20',amount:35},{x:280,date:'Aug 15',amount:25},{x:425,date:'Oct 12',amount:15},{x:530,date:'Dec 4',amount:10}];
  let p=line(35,146,570,146,C.line,'stroke-width="3"')+line(350,45,350,290,C.orange,'stroke-width="3" stroke-dasharray="5 5"')+txt(350,30,'Sep 30','text-anchor="middle"');
  events.forEach((e,i)=>{const c=i<3?C.blue:(future?C.red:C.muted),opacity=i<3||future?1:.35;p+=`<g opacity="${opacity}"><circle cx="${e.x}" cy="146" r="9" fill="${c}"/>`+txt(e.x,113,e.date,'text-anchor="middle"')+M(e.x,184,`\\$${e.amount}`,90,42)+`</g>`;});
  p+=box(35,242,285,58,'No September donation',C.blue)+txt(457,270,'Future records','text-anchor="middle"');
  return svg(p,'0 0 600 320',future?'Three donations of twenty, thirty-five, and twenty-five dollars occur before September 30. October and December donations lie after the prediction cutoff.':'An illustrative donor has three earlier donations and no September donation. The September 30 forecast row still exists; future events are unavailable.');
}
function sameUsers20(){
  let p='';
  ['Earlier','Earlier','Earlier','Later','Later'].forEach((s,i)=>p+=txt(153+i*87,40,s,'text-anchor="middle"'));
  ['User A','User B','User C'].forEach((s,r)=>{p+=txt(91,100+r*74,s,'text-anchor="end"');for(let i=0;i<5;i++){const c=i<3?C.blue:C.green;p+=box(122+i*87,73+r*74,62,44,'',c);if((r+i)%3===0)p+=txt(153+i*87,101+r*74,'–','text-anchor="middle"');else p+=`<circle cx="${153+i*87}" cy="${95+r*74}" r="7" fill="${c}"/>`;}});
  p+=line(369,21,369,290,C.orange,'stroke-width="3" stroke-dasharray="5 5"')+txt(239,317,'TRAIN','text-anchor="middle"')+txt(474,317,'TEST','text-anchor="middle"');
  return svg(p,'0 0 600 345','Each user has earlier training rows and later test rows. Dashes indicate inactive prediction months, whose rows are retained.');
}
function cvTimeline20(){
  let p='';
  ['Fold 1','Fold 2','Fold 3'].forEach((s,r)=>{const y=46+r*80,w=100+r*75;p+=txt(76,y+27,s,'text-anchor="end"')+`<g data-cv-row="${r}">`+box(95,y,w,47,'Train',C.blue)+box(100+w,y,70,47,'Valid.',C.green)+`</g>`;});
  p+=line(455,22,455,292,C.orange,'stroke-width="3" stroke-dasharray="5 5"')+box(475,46,110,207,'',C.orange)+txt(530,105,'TEST','text-anchor="middle"')+txt(530,145,'Untouched','text-anchor="middle"')+txt(530,180,'12 months','text-anchor="middle"')+txt(280,321,'Later prediction months →','text-anchor="middle"');
  return svg(p,'0 0 600 345','Expanding training windows predict three nonoverlapping later validation blocks. Each validation user-month is scored once. The final twelve prediction months are never used in these folds.');
}
function predictionRows20(loss=false){
  let p=txt(104,31,'Validation row','text-anchor="middle"')+M(240,28,'y',70)+M(367,28,loss?'\\ell_A':'\\widehat p_A',130)+M(513,28,loss?'\\ell_B':'\\widehat p_B',130)+line(20,54,585,54);
  [{label:'Frequent donor',y:1,a:loss?'0.8464':'0.08',b:loss?'0.64':'0.20'},{label:'Once-only donor',y:0,a:loss?'0.0064':'0.08',b:loss?'0.0004':'0.02'}].forEach((r,i)=>{const y=110+i*99;p+=box(20,y-32,565,65,'',i?C.blue:C.green)+txt(104,y+6,r.label,'text-anchor="middle"')+M(240,y,loss?String(r.y):'?',70)+M(367,y,r.a,130)+M(513,y,r.b,130);});
  p+=M(300,298,loss?'\\ell=(y-\\widehat p)^2':'A:\\ d_{\\max}=1\\qquad B:\\ d_{\\max}=3',570,54);
  return svg(p,'0 0 600 335',loss?'The two held-out outcomes are compared with both forecasts using squared error. For outcome one, losses are .8464 and .64. For outcome zero, losses are .0064 and .0004.':'Both candidate models predict the same two illustrative validation rows, with outcomes still hidden. Model A predicts .08 for each, while model B predicts .20 and .02.');
}
function brierCalculation20(prob){
  const calibrated=prob===.05;
  let p=M(300,27,`\\widehat p=${calibrated?'0.05':'0'}`,250,48,27)+box(20,73,270,171,'',C.green)+box(310,73,270,171,'',C.blue)+txt(155,106,'5 donation months','text-anchor="middle"')+txt(445,106,'95 other months','text-anchor="middle"');
  p+=M(155,153,calibrated?'(1-0.05)^2=0.9025':'(1-0)^2=1',265,45)+M(445,153,calibrated?'(0-0.05)^2=0.0025':'(0-0)^2=0',265,45)+M(155,211,calibrated?'5(0.9025)=4.5125':'5(1)=5',265,45)+M(445,211,calibrated?'95(0.0025)=0.2375':'95(0)=0',265,45);
  p+=line(155,244,265,273,C.green,'stroke-width="2"')+line(445,244,335,273,C.blue,'stroke-width="2"')+M(300,307,calibrated?'\\mathrm{Brier}=\\frac{4.5125+0.2375}{100}=0.0475':'\\mathrm{Brier}=\\frac{5+0}{100}=0.05',580,62,24);
  return svg(p,'0 0 600 350',calibrated?'A constant probability .05 yields squared error .9025 for five positive outcomes and .0025 for ninety-five negative outcomes. Average Brier loss is .0475.':'A constant zero probability yields squared error one for five positive outcomes and zero for ninety-five negative outcomes. Average Brier loss is .05.');
}
function render20(stage,container){
  if(stage===0){
    container.innerHTML=title('A stable pattern remains inside one leaf')+rateChart()+note('Frequency groups: one versus six or more donations. The same observed rates recur in training and later validation months.');
  }else if(stage===1){
    let p=box(185,15,230,49,'Recent-donor leaf',C.blue)+line(300,64,300,96,C.blue,'stroke-width="2"')+box(185,96,230,49,'One donation?',C.blue)+line(185,121,100,183,C.blue,'stroke-width="2"')+line(415,121,455,183,C.blue,'stroke-width="2"')+txt(126,153,'yes','text-anchor="middle"')+txt(450,153,'no','text-anchor="middle"')+box(20,183,160,54,'One',C.green)+box(350,183,230,54,'Six or more?',C.blue)+line(407,237,345,285,C.blue,'stroke-width="2"')+line(522,237,520,285,C.blue,'stroke-width="2"')+txt(359,264,'no','text-anchor="middle"')+txt(531,264,'yes','text-anchor="middle"')+box(264,285,160,46,'Two to five',C.green)+box(440,285,145,46,'Six or more',C.green);
    container.innerHTML=title('Additional splits can separate those groups')+svg(p,'0 0 600 355','A deeper binary tree can split the recent-donor leaf into one donation, two to five donations, and six or more donations.')+note('Illustrative possible splits. Training data determine the splits; validation chooses depth and minimum leaf size.');
  }else if(stage===2){
    container.innerHTML=title('Start with a donor’s event history')+eventHistory20()+note('Illustrative complete history. This user previously donated, so an eligible September 30 row exists even though September is inactive.');
  }else if(stage===3){
    let p=txt(300,27,'Build one row as of September 30','text-anchor="middle"')+box(30,63,540,55,'',C.blue)+M(300,91,'20+35+25=80',490)+line(300,118,300,153,C.blue,'stroke-width="3"');
    [['Prior 12-month count','3'],['Prior 12-month amount','\\$80'],['Days since last donation','46']].forEach(([label,value],i)=>{const y=184+i*48;p+=txt(45,y,label)+M(492,y-5,value,140,40)+line(35,y+18,565,y+18);});
    p+=txt(300,342,'Joining surveys are already available.','text-anchor="middle"');
    container.innerHTML=title('Turn available history into predictor values')+svg(p,'0 0 600 365','The September 30 row has three donations, eighty dollars donated, and forty-six days since August 15. Joining survey measures can also be used.')+note('Three past events produce these features. Nothing from October or December enters the row.');
  }else if(stage===4){
    let p=box(25,25,245,65,'September 30 features',C.blue)+box(330,25,245,65,'October outcome',C.green)+line(270,57,330,57,C.line,'stroke-width="3"')+M(145,124,'\\mathrm{count}=3',240)+M(455,124,'y_{\\mathrm{Oct}}=1',240)+line(300,100,300,315,C.orange,'stroke-width="3" stroke-dasharray="5 5"')+box(25,192,245,75,'',C.blue)+txt(147,221,'January–September','text-anchor="middle"')+M(147,248,'3',100)+box(330,192,245,75,'',C.red)+txt(452,221,'January–December','text-anchor="middle"')+M(452,248,'5',100)+txt(147,311,'Available at prediction','text-anchor="middle"')+txt(452,311,'Includes the future','text-anchor="middle"');
    container.innerHTML=title('Keep the future outcome on the other side')+svg(p,'0 0 600 340','Three donations are known by September 30. The October donation makes the next-month outcome one. The full-year count of five incorrectly includes October and December events.')+note('The target is October donation status. The invalid full-year feature includes that same future event.');
  }else if(stage===5){
    container.innerHTML=title('The same users can cross the time boundary')+sameUsers20()+note('Each cell is an eligible user-month. A dash marks an inactive prediction month; its row remains in the data.');
  }else if(stage===6){
    let p=box(25,49,245,88,'Development',C.blue)+box(310,49,265,88,'Final test',C.orange)+txt(147,26,'Earlier months','text-anchor="middle"')+txt(442,26,'Latest 12 months','text-anchor="middle"');
    for(let i=0;i<12;i++){const x=318+(i%6)*42,y=179+Math.floor(i/6)*47;p+=box(x,y,34,34,'',C.orange)+txt(x+17,y+24,i+1,'text-anchor="middle"');}
    p+=txt(147,199,'Select settings here.','text-anchor="middle"')+txt(147,237,'Then refit here.','text-anchor="middle"')+line(291,35,291,285,C.line,'stroke-width="2" stroke-dasharray="4 5"')+txt(300,323,'Next-month outcomes must be fully observed.','text-anchor="middle"');
    container.innerHTML=title('Reserve every eligible row in the final twelve months')+svg(p,'0 0 600 350','Earlier months form development data. Every eligible user-month in the latest twelve prediction months with known following-month outcomes is reserved for final testing.')+note('Numbers 1–12 index reserved prediction months. No test row helps choose the model.');
  }else if(stage===7){
    container.innerHTML=title('Repeat the earlier-to-later exercise within development data')+cvTimeline20()+note('Validation blocks do not overlap: each validation row is scored once. Training uses only outcomes already available at its cutoff.');
  }else if(stage===8){
    let p=box(175,15,250,50,'Earlier training rows',C.blue)+line(300,65,300,102,C.blue,'stroke-width="3"')+box(105,102,390,64,'Fit preprocessing here',C.blue)+txt(300,192,'Save this fitted transformation.','text-anchor="middle"')+line(220,166,150,226,C.blue,'stroke-width="3"')+line(380,166,450,226,C.blue,'stroke-width="3"')+box(20,226,260,91,'',C.blue)+box(320,226,260,91,'',C.green)+txt(150,254,'Fit candidate A','text-anchor="middle"')+txt(450,254,'Fit candidate B','text-anchor="middle"')+M(150,290,'d_{\\max}=1',210)+M(450,290,'d_{\\max}=3',210);
    container.innerHTML=title('Open one fold: fit both candidates using its training data')+svg(p,'0 0 600 340','Preprocessing is fitted only on the fold training data. The transformed training data fit candidate A with maximum depth one and candidate B with maximum depth three.')+note('Illustrative depth choices. Any imputation is learned from training rows; validation rows cannot influence it.');
  }else if(stage===9){
    container.innerHTML=title('Apply both fitted candidates to the same later rows')+predictionRows20()+note('Two illustrative validation rows, after the saved training transformation. The outcomes are withheld until scoring.');
  }else if(stage===10){
    container.innerHTML=title('Replace each forecast with its squared probability error')+predictionRows20(true)+equation('\\text{First row: }(1-0.08)^2=0.8464,\\quad(1-0.20)^2=0.64')+note('These two rows explain the calculation. Score every validation row before comparing candidates.');
  }else if(stage===11){
    let p=txt(80,30,'Fold','text-anchor="middle"')+txt(210,30,'Rows','text-anchor="middle"')+txt(360,30,'A: mean loss','text-anchor="middle"')+txt(520,30,'B: mean loss','text-anchor="middle"')+line(20,49,585,49);
    [[1,100,'0.06','0.04'],[2,200,'0.04','0.045'],[3,100,'0.05','0.04']].forEach(([fold,n,a,b],i)=>{const y=82+i*50;p+=txt(80,y+6,fold,'text-anchor="middle"')+M(210,y,String(n),100)+M(360,y,a,130)+M(520,y,b,130)+line(20,y+23,585,y+23);});
    p+=M(300,250,'A:\\ \\frac{100(0.06)+200(0.04)+100(0.05)}{400}=0.0475',580,64)+M(300,320,'B:\\ \\frac{100(0.04)+200(0.045)+100(0.04)}{400}=0.0425',580,64);
    container.innerHTML=title('Pool validation loss by the number of rows')+svg(p,'0 0 600 360','Illustrative full-fold summaries contain one hundred, two hundred, and one hundred validation rows. Weighted mean losses are .0475 for A and .0425 for B, so B is selected.')+note('Illustrative complete-fold summaries, not totals of the two preview rows. Fold 2 receives twice the weight of either other fold because it has twice as many user-months. Lower loss selects B.');
  }else if(stage===12){
    let p=txt(300,29,'100 user-months: 5 donations, 95 without','text-anchor="middle"');
    for(let i=0;i<100;i++){const x=60+(i%20)*25,y=58+Math.floor(i/20)*27;p+=i<5?`<circle cx="${x+8}" cy="${y+8}" r="8" fill="${C.green}"/>`:`<rect x="${x}" y="${y}" width="16" height="16" rx="2" fill="${C.line}" opacity=".55"/>`;}
    p+=M(150,242,'\\widehat p=0',250,48,26)+M(450,242,'\\widehat p=0.05',250,48,26)+txt(150,289,'95% accuracy','text-anchor="middle"')+txt(450,289,'95% accuracy','text-anchor="middle"');
    container.innerHTML=title('Accuracy gives these different probability forecasts a tie')+svg(p,'0 0 600 320','In a hundred cases with five donations, both constant zero and constant .05 probability forecasts classify every case negative at threshold .5 and are ninety-five percent accurate.')+note(`At a ${m('0.5')} classification threshold both predict no donation. Circles are donations; squares are non-donations.`);
  }else if(stage===13){
    container.innerHTML=title('Score the zero-probability forecast')+brierCalculation20(0)+note('Each missed donation contributes 1. Each non-donation contributes 0. Sum over all 100 rows, then divide by 100.');
  }else if(stage===14){
    container.innerHTML=title('Score the 5% forecast using the same outcomes')+brierCalculation20(.05)+note('Brier loss falls from 0.05 to 0.0475 even though accuracy remains 95%. The score recognizes the better probability forecasts.');
  }else{
    let p=box(15,45,170,80,'Choose settings',C.blue)+box(215,45,170,80,'Refit on all',C.blue)+box(415,45,170,80,'Test once',C.orange)+line(185,85,215,85,C.ink,'stroke-width="3"')+line(385,85,415,85,C.ink,'stroke-width="3"')+txt(100,155,'Validation loss','text-anchor="middle"')+txt(300,155,'Development data','text-anchor="middle"')+txt(500,155,'Latest 12 months','text-anchor="middle"')+line(30,198,570,198)+txt(300,237,'Complete records and available outcomes','text-anchor="middle"')+txt(300,278,'Historical patterns remain informative','text-anchor="middle"')+txt(300,319,'Users are comparable to those forecast','text-anchor="middle"');
    container.innerHTML=title('Freeze the selected procedure, refit, and test once')+svg(p,'0 0 600 345','Choose settings using validation loss, refit preprocessing and the model on all development data, and evaluate once on the final twelve months. The approach needs complete records, informative historical patterns, and comparable users.')+note('Keep natural prevalence in validation and test. Any training balancing belongs inside the training folds, with probability calibration checked on untouched validation rows.');
  }
  renderMath(container);
}
const q20={
  id:20,kicker:'Required · Prediction design',title:'Forecast the next donation',
  dek:'Build a forecast row from its history, rehearse the future in validation, and follow each probability all the way to its score.',
  prompt:`<p>A researcher has several years of users' donation dates and amounts from an online political fundraising platform, along with survey measures of political interest and ideological commitment collected when users joined.</p><p>At the end of each month, she estimates the probability that each user who has previously donated will donate during the following month. Users remain eligible during inactive months. Only 5% of eligible user-months have a donation in the following month. Her goal is to predict future months for users already observed on the platform.</p><p>Use one observation per eligible user-month.</p><p>Her initial model is a decision tree restricted to one split. Its candidate predictors include donation recency, donation count, and total amount donated during the preceding twelve months. The fitted tree splits on whether the last donation occurred within three months.</p><p>Both groups below belong to the same recent-donor leaf. The groups are large, and the observed pattern persists across several later-month validation samples.</p><div class="table-wrap"><table class="mini-table"><thead><tr><th>Donations in preceding twelve months</th><th>Tree's predicted probability</th><th>Observed training rate</th><th>Observed validation rate</th></tr></thead><tbody><tr><td>One</td><td>8%</td><td>2%</td><td>2%</td></tr><tr><td>Six or more</td><td>8%</td><td>20%</td><td>20%</td></tr></tbody></table></div><p>Answer briefly. Concise bullet points are fine; no calculations are required.</p>`,
  parts:[`Explain why these results provide evidence of underfitting. What restriction prevents the initial model from capturing this pattern?`,`Can observations from the same user appear in both the training and test sets for this prediction task? Explain briefly.`,`To predict October donations on September 30, a colleague uses each user's total number of donations from January 1 through December 31 of that year as a predictor. Explain the problem and define a valid replacement.`,`Propose one alternative model and explain why it is worth trying. Precisely define which user-month observations you would reserve for final testing. Describe how you would construct cross-validation folds on the remaining data and use them to select hyperparameters. Specify the predictors and hyperparameters you would consider, and justify the performance measure you would use for model selection and final evaluation. State any additional assumptions needed to justify your approach.`],
  sections:[{label:'Missed pattern',step:0},{label:'Rows and time',step:2},{label:'Validation',step:6},{label:'Probability scores',step:12}],
  steps:[
    {title:'One prediction misses two stable rates.',body:`<p>Both groups receive an 8% prediction. Their observed rates are 2% and 20%, repeatedly in training and later validation months.</p><p>The only allowed split is already used for recency. The tree cannot further split that leaf by frequency. This stable missed pattern is evidence of underfitting.</p>`,caption:'Part (a) · Compare each observed rate with the shared prediction.'},
    {title:'Let the recent-donor leaf split again.',body:`<p>Try a deeper tree, tuning maximum depth and minimum leaf size. It can distinguish frequency groups within the recent-donor leaf.</p><p>Consider recency, trailing-twelve-month donation count and amount, and joining surveys of political interest and ideological commitment. Validation decides whether extra flexibility helps.</p>`,caption:'Part (d) · Possible additional splits address the specific missed pattern.'},
    {title:'An inactive month still needs a forecast row.',body:`<p>Consider a user with three earlier donations and none during September. The user has previously donated, so they remain eligible at September 30.</p><p>Create that user-month row even without a September event. The target will indicate whether a donation occurs in October.</p>`,caption:'Setup · Event records become one row per eligible user-month.'},
    {title:'Summarize only the history available that day.',body:`<p>At September 30, the illustrative user has three donations in the preceding twelve months, totaling $80. The last occurred August 15, 46 days earlier.</p><p>Those become count, amount, and recency features. Joining survey measures are also available. The October and December events are still in the future.</p>`,caption:'Parts (c) and (d) · Every predictor is constructed as of its prediction date.'},
    {title:'The October donation is the target, not a feature.',body:`<p>The October donation makes this row's target ${m('y=1')}. A January–December count would be 5, but it includes that target event and the December event.</p><p>Use the count through September 30 instead: 3 here. A trailing-twelve-month count ending September 30 is also valid.</p>`,caption:'Part (c) · The full-year count crosses the information boundary.'},
    {title:'A user can supply earlier training rows and later test rows.',body:`<p>The task is to predict future months for users already observed on the platform. Earlier rows for a user can therefore train a model evaluated on that same user's later rows.</p><p>Split along time and retain eligible inactive months. Holding out entirely different people would answer a different prediction question.</p>`,caption:'Part (b) · Follow a user horizontally across the time boundary.'},
    {title:'Define the final test before tuning anything.',body:`<p>Order the prediction months whose complete following-month outcomes are observed. Reserve every eligible user-month in the latest twelve of those prediction months for final testing.</p><p>Use only earlier months for development. The final test never contributes to preprocessing choices, hyperparameter selection, or probability calibration.</p>`,caption:'Part (d) · Reserve a precise set of future observations.'},
    {title:'Make each validation fold rehearse a future forecast.',body:`<p>Within development data, expand the training window and validate on the next, nonoverlapping block of prediction months. Each validation user-month is scored once. At each training cutoff, use only labels already available.</p><p>Fit preprocessing within that training window. Apply the fitted transformation and model to the later validation rows, then advance the window.</p>`,caption:'Part (d) · Scroll through the earlier-to-later folds.'},
    {title:'Open one fold and fit two candidate settings.',body:`<p>Start with the fold's earlier training rows. Learn any preprocessing, such as missing-value imputation, from those rows alone.</p><p>Use the transformed training rows to fit each candidate. The illustration compares maximum depths 1 and 3; the actual search also considers minimum leaf size. Validation outcomes remain outside fitting.</p>`,caption:'Part (d) · Training-only preprocessing comes before each candidate fit.'},
    {title:'Give both candidates the same later observations.',body:`<p>Apply the saved training transformation to the validation block. Each fitted candidate predicts the same rows.</p><p>Two illustrative rows show the comparison: one frequent donor and one once-only donor, both recent. Candidate A predicts 8% for each; candidate B predicts 20% and 2%. Observe their outcomes only to score the forecasts.</p>`,caption:'Part (d) · Match candidate predictions row by row.'},
    {title:'Turn each prediction into a probability loss.',body:`<p>Brier loss for one row is ${m('(y-\\widehat p)^2')}. For the donation row, predictions of 8% and 20% produce losses of 0.8464 and 0.64.</p><p>Repeat the calculation for the non-donation row, then for every other validation row. Lower loss means better probability forecasts.</p>`,caption:'Part (d) · The outcomes convert predictions into comparable scores.'},
    {title:'Let every validation row have the same weight.',body:`<p>Average loss over all validation user-months. Equivalently, weight each fold's mean by its number of rows.</p><p>In these illustrative complete-fold summaries, the 200-row fold counts twice as much as either 100-row fold. Candidate B has the smaller pooled loss, so this comparison selects B.</p>`,caption:'Part (d) · Fold averages become one observation-weighted selection score.'},
    {title:'Why use a probability score instead of accuracy?',body:`<p>Take 100 user-months with five donations, matching the stated 5% prevalence. Compare constant probability forecasts of 0 and 0.05.</p><p>At a 0.5 classification threshold, both predict “no donation” for everyone. Both are 95% accurate. Accuracy cannot distinguish their probability quality.</p>`,caption:'Part (d) · Fix the outcomes and change only the forecast probabilities.'},
    {title:'A zero forecast has Brier loss 0.05.',body:`<p>For each of the five donation months, squared error is ${m('(1-0)^2=1')}. Each of the other 95 months contributes zero.</p><p>The total loss is 5 across 100 rows, giving mean Brier loss ${m('5/100=0.05')}. Follow the category contributions into that average.</p>`,caption:'Part (d) · Individual squared errors become a mean over all eligible rows.'},
    {title:'A 5% forecast improves the probability score.',body:`<p>Now predict 0.05 for every row. Each donation contributes 0.9025; each non-donation contributes 0.0025. The weighted total is 4.75, so Brier loss is 0.0475.</p><p>This beats 0.05 despite identical 95% accuracy. Use Brier or log loss consistently for selection and final evaluation.</p>`,caption:'Part (d) · Same outcomes and accuracy; different probability quality.'},
    {title:'Choose, refit, and test once.',body:`<p>Choose settings by validation loss, refit preprocessing and the model on all development data, then evaluate once on the reserved months. Preserve natural prevalence in validation and test.</p><p>The plan assumes complete records, available outcomes, informative historical patterns, and comparable users. Any training balancing stays inside folds; check calibration on untouched validation rows.</p>`,caption:'Part (d) · A temporal test measures the observed future period, without guaranteeing permanent stability.'}
  ],takeaway:`Allow the model to represent the missed frequency pattern, construct every feature as of its prediction date, tune on earlier-to-later folds, and reserve the latest twelve eligible prediction months for one probability-loss evaluation.`,render:render20,
  update(progress,container,stage){
    if(stage!==7)return;
    const active=progress>=.99?-1:Math.min(2,Math.floor(Math.max(0,progress)*3));
    container.querySelectorAll('[data-cv-row]').forEach(row=>{const selected=active<0||Number(row.dataset.cvRow)===active;row.querySelectorAll('rect').forEach(rect=>{rect.setAttribute('stroke-width',selected?'3':'1');rect.setAttribute('fill-opacity',selected?'.17':'.045');});});
  }
};

export const questions=[q19,q20];
