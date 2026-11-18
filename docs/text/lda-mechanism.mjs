import {escapeText as esc, mathLabel} from './math.mjs?v=20261108-peloton-1';
import {pelotonExample, topicLabels, topicColors, theta, beta, conditionalTopics, interpolatedTopics, clamp, interpolate} from './lda-model.mjs?v=20261108-peloton-1';
export {pelotonExample} from './lda-model.mjs?v=20261108-peloton-1';

const f = n => Number(n.toFixed(4));
const text = (key,x,y,label,size=30,extra='') => `<text data-key="${key}" x="${f(x)}" y="${f(y)}" font-size="${size}" fill="#242424" ${extra}>${esc(label)}</text>`;
const coloredText = (key,x,y,label,color,size=30,extra='') => `<text data-key="${key}" x="${f(x)}" y="${f(y)}" font-size="${size}" fill="${color}" ${extra}>${esc(label)}</text>`;
const math = (key,x,y,w,h,tex,size=30) => `<g data-key="${key}">${mathLabel(x,y,w,h,tex,size)}</g>`;
const group = (key,opacity,body,extra='') => `<g data-key="${key}" opacity="${f(opacity)}" aria-hidden="${opacity<.001}" ${extra}>${body}</g>`;
const line = (key,x1,y1,x2,y2,progress=1,color='#767676') => `<path data-key="${key}" d="M${f(x1)} ${f(y1)} L${f(x2)} ${f(y2)}" pathLength="1" stroke="${color}" stroke-width="2" fill="none" stroke-dasharray="1" stroke-dashoffset="${f(1-progress)}"/>`;
const arrowhead = (key,x,y,opacity=1,color='#767676',up=false) => `<path data-key="${key}" d="M${f(x-7)} ${f(y+(up?10:-10))} L${f(x)} ${f(y)} L${f(x+7)} ${f(y+(up?10:-10))}" stroke="${color}" stroke-width="2" fill="none" opacity="${f(opacity)}"/>`;

// Numeric states interpolate the very same objects; backwards scrolling retraces them.
const states = [
  {matrix:1,matrixY:170,quote:0,bar:0,barY:320,choice:0,word:0,infer:0,stm:0,content:0,posterior:0,selection:0},
  {matrix:0,matrixY:345,quote:1,bar:1,barY:325,choice:0,word:0,infer:0,stm:0,content:0,posterior:0,selection:0},
  {matrix:1,matrixY:360,quote:0,bar:1,barY:85,choice:1,word:0,infer:0,stm:0,content:0,posterior:0,selection:1},
  {matrix:1,matrixY:285,quote:0,bar:1,barY:70,choice:1,word:1,infer:0,stm:0,content:0,posterior:0,selection:1},
  {matrix:0,matrixY:285,quote:0,bar:1,barY:350,choice:0,word:1,infer:1,stm:0,content:0,posterior:1,selection:0},
  {matrix:0,matrixY:225,quote:0,bar:1,barY:430,choice:0,word:0,infer:0,stm:1,content:0,posterior:0,selection:0},
  {matrix:1,matrixY:210,quote:0,bar:1,barY:475,choice:0,word:0,infer:0,stm:0,content:1,posterior:0,selection:1},
];
export function mechanismState(stage,p=stage,motion=true) {
  if(!Number.isInteger(stage)||stage<0||stage>6)throw new RangeError(`Unknown LDA mechanism stage: ${stage}`);
  const q=motion?clamp((p-stage+.5)*2):1, previous=states[Math.max(0,stage-1)], current=states[stage];
  const s=Object.fromEntries(Object.keys(current).map(key=>[key,interpolate(previous[key],current[key],q)]));
  s.position=stage===0?0:stage-1+q;
  s.phaseOpacity=states.map((_,i)=>Math.max(0,1-Math.abs(s.position-i)));
  return s;
}
export function ldaMechanismVisual(stage,p=stage,motion=true) {
  const s=mechanismState(stage,p,motion), ph=s.phaseOpacity, posterior=conditionalTopics(2);
  const mixture=theta.map((v,k)=>interpolate(v,posterior[k],s.posterior));
  const topics=interpolatedTopics(s.content), barX=28, barWidth=624;
  const matrixColumns=[254,362,470,596];
  const description=`${pelotonExample.instructor}, ${pelotonExample.classTitle}. ${pelotonExample.excerpts[0]} Illustrative three-topic model: Movement, Breath, Reflection. The class mixture is 50, 20, and 30 percent. In the earlier baseline example, the probability of precious is 0.072; conditional on that word and the baseline parameters, Reflection has topic probability about 0.833. STM prevalence enters the topic-mixture model; optional content covariates enter the topic-word distributions.`;
  const matrix=group('topic-word-matrix',s.matrix,
    text('matrix-label',28,-42,'Topic–word probabilities',31)+
    ['cradle','inhale','precious','other'].map((word,j)=>text(`word-heading-${j}`,matrixColumns[j],0,word,28,'text-anchor="middle"')).join('')+
    topicLabels.map((label,k)=>{
      const rowY=49+k*55, emphasis=k===2?s.selection:0;
      return `<g data-key="topic-row-${k}" data-topic="${k}">
        <rect data-key="topic-outline-${k}" x="18" y="${rowY-36}" width="642" height="49" fill="none" stroke="${topicColors[k]}" stroke-width="2" opacity="${f(emphasis)}"/>
        ${coloredText(`topic-label-${k}`,28,rowY,label,topicColors[k],29)}
        ${topics[k].map((value,j)=>`<g data-key="word-probability-${k}-${j}" data-probability="${f(value)}"><rect x="${matrixColumns[j]-46}" y="${rowY-31}" width="${f(92*value)}" height="37" fill="${topicColors[k]}" opacity="0.12"/>${text(`probability-${k}-${j}`,matrixColumns[j],rowY,value.toFixed(2),29,'text-anchor="middle"')}</g>`).join('')}
      </g>`;
    }).join('')+text('matrix-other-note',340,214,'“Other” pools the remaining vocabulary.',24,'text-anchor="middle" fill-opacity="0.75"'),
    `transform="translate(0 ${f(s.matrixY)})"`);
  const quote=group('class-quotation',s.quote,
    text('quote-author',28,30,pelotonExample.instructor,31)+
    text('quote-citation',28,63,'Seated Baby Cradle · May 13, 2021',25,'fill-opacity="0.75"')+
    pelotonExample.excerptLines.map((part,i)=>text(`quote-line-${i}`,28,102+i*34,part,30)).join(''));
  const mixtureBar=group('document-mixture',s.bar,
    `<rect data-key="mixture-outline" x="28" y="0" width="624" height="45" fill="#fff" stroke="#767676" stroke-width="1.5"/>`+
    mixture.map((value,k)=>{
      const previous=mixture.slice(0,k).reduce((a,b)=>a+b,0), center=barX+barWidth*(previous+value/2);
      return `<g data-key="mixture-topic-${k}" data-share="${f(value)}"><rect data-key="mixture-segment-${k}" x="${f(barX+barWidth*previous)}" y="0" width="${f(barWidth*value)}" height="45" fill="${topicColors[k]}"/>${coloredText(`mixture-percent-${k}`,center,32,`${Math.round(value*100)}%`,'#fff',29,`text-anchor="middle" opacity="${f(1-s.posterior)}"`)}</g>`;
    }).join('')+
    topicLabels.map((label,k)=>coloredText(`mixture-name-${k}`,28+k*218,82,`${label}${s.posterior>.001?' '+Math.round(mixture[k]*100)+'%':''}`,topicColors[k],28)).join(''),
    `transform="translate(0 ${f(s.barY)})"`);
  const choiceY=interpolate(215,190,s.word*(1-s.infer));
  const wordY=interpolate(510,38,s.infer);
  return `<div class="lda-mechanism-flow" data-key="lda-workspace" data-visual-family="lda-mechanism" data-position="${f(s.position)}" data-content-progress="${f(s.content)}" data-posterior-progress="${f(s.posterior)}">
  <svg class="lda-mechanism-workspace" viewBox="0 0 680 610" role="img" aria-label="${esc(description)}">
    ${quote}${matrix}${mixtureBar}
    ${group('topic-prior',ph[0],math('beta-prior',28,0,624,55,String.raw`\boldsymbol\beta_k\sim\operatorname{Dirichlet}(\boldsymbol\lambda)`,32)+text('beta-level',340,70,'One vocabulary distribution for each topic.',28,'text-anchor="middle"')+math('beta-sum',28,430,624,55,String.raw`\beta_{k,w}\geq0,\qquad\sum_{w=1}^{V}\beta_{k,w}=1`,31)+text('beta-once',340,520,'Draw once per topic; share across documents.',28,'text-anchor="middle"'))}
    ${group('document-prior',ph[1],math('theta-prior',28,250,624,54,String.raw`\boldsymbol\theta_d\sim\operatorname{Dirichlet}(\boldsymbol\alpha)`,32)+math('theta-example',28,435,624,55,String.raw`\boldsymbol\theta_d=(0.50,\ 0.20,\ 0.30)`,32)+text('one-class',340,528,'One mixture for this whole class transcript.',28,'text-anchor="middle"'))}
    ${group('token-topic-title',ph[2]+ph[3],math('theta-label',28,8,624,50,String.raw`\boldsymbol\theta_d=(0.50,\ 0.20,\ 0.30)`,30))}
    ${group('latent-topic-choice',s.choice,
      line('theta-to-topic',558,s.barY+90,558,choiceY-12,clamp(s.choice*1.5),topicColors[2])+arrowhead('theta-topic-tip',558,choiceY-12,s.choice,topicColors[2])+
      math('draw-topic',28,choiceY-12,410,52,String.raw`z_{dn}\sim\operatorname{Categorical}(\boldsymbol\theta_d)`,29)+
      coloredText('selected-topic',470,choiceY+23,'Reflection',topicColors[2],30)+
      text('token-topic-note',28,choiceY+66,'A separate topic draw for every token.',27,`opacity="${f(1-s.word)}"`)+
      line('topic-to-row',632,choiceY+72,632,s.matrixY+134,s.word,topicColors[2])+arrowhead('topic-row-tip',632,s.matrixY+134,s.word,topicColors[2]))}
    ${group('generated-word',s.word,
      group('word-draw-formula',1-s.infer,math('word-formula',28,wordY,390,50,String.raw`w_{dn}\sim\operatorname{Categorical}(\boldsymbol\beta_{z_{dn}})`,28))+
      `<rect data-key="word-box" x="443" y="${f(wordY-2)}" width="209" height="47" fill="#fff" stroke="#a54f32" stroke-width="2"/>`+
      coloredText('observed-word',548,wordY+31,'precious','#a54f32',33,'text-anchor="middle"'))}
    ${group('word-marginal',ph[3],math('marginal-example',28,551,624,35,String.raw`P(\text{precious}\mid d)=.50(.02)+.20(.01)+.30(.20)=.072`,26))}
    ${group('infer-posterior',s.infer,
      text('observed-label',28,28,'Observe the word; infer its possible topic.',29)+
      math('posterior-rule',28,113,624,57,String.raw`P(z_{dn}=k\mid w_{dn},\boldsymbol\theta_d,\boldsymbol\beta)\propto\theta_{dk}\beta_{k,w_{dn}}`,29)+
      text('contribution-label',28,194,'Topic weights for “precious”',29)+
      topicLabels.map((label,k)=>coloredText(`contribution-${k}`,28+k*218,238,[ '.010', '.002', '.060'][k],topicColors[k],35)).join('')+
      math('normalize-contributions',28,250,624,50,String.raw`(.010,\ .002,\ .060)\,/\,.072`,31)+
      text('posterior-bar-label',340,338,'Topic probabilities for this token',27,'text-anchor="middle"')+
      math('posterior-result',28,454,624,50,String.raw`P(z_{dn}=\mathrm{Reflection}\mid\text{precious},\boldsymbol\theta_d,\boldsymbol\beta)\approx.833`,27)+
      text('conditional-note',340,535,'This local calculation holds the parameters fixed.',27,'text-anchor="middle"'))}
    ${group('stm-prevalence',s.stm,
      text('metadata-title',340,30,'Document metadata',31,'text-anchor="middle"')+
      text('instructor',28,82,'Instructor: Chelsea Jackson Roberts',29)+text('activity',28,122,'Class type: Yoga',29)+
      math('x-label',470,84,182,50,String.raw`\mathbf{x}_d`,32)+line('metadata-to-eta',340,152,340,195,s.stm)+arrowhead('metadata-eta-tip',340,195,s.stm)+
      math('eta-normal',28,210,624,57,String.raw`\boldsymbol\eta_d\sim\mathcal{N}(\boldsymbol\Gamma^{\mathsf T}\mathbf{x}_d,\boldsymbol\Sigma)`,32)+
      text('log-ratio-dimension',340,297,'Latent log-ratios to the last topic',27,'text-anchor="middle"')+
      math('stm-softmax',28,324,624,56,String.raw`\boldsymbol\theta_d=\operatorname{softmax}(\boldsymbol\eta_d,0)`,32)+
      line('eta-to-mixture',340,385,340,418,s.stm)+arrowhead('eta-mixture-tip',340,418,s.stm)+
      text('stm-fixed-note',340,566,'Mixture held fixed here; no fitted metadata effect.',26,'text-anchor="middle"'))}
    ${group('stm-content',s.content,
      text('content-title',340,28,'Optional content covariate',31,'text-anchor="middle"')+
      math('content-covariate',28,53,624,52,String.raw`y_d=\text{instructor}`,31)+
      line('content-to-beta',340,113,340,145,s.content,topicColors[2])+arrowhead('content-beta-tip',340,145,s.content,topicColors[2])+
      math('theta-fixed',28,436,624,32,String.raw`\boldsymbol\theta_d\text{ stays fixed}`,28))}
    ${text('illustrative-note',340,606,'Illustrative probabilities · not fitted to these classes',23,'text-anchor="middle" fill-opacity="0.75"')}
  </svg></div>`;
}
