import {escapeText as esc,mathLabel} from './math.mjs?v=20261108-peloton-1';
import {simplexVertices,exampleMixture,topicNames,topicColors,simplexPoint,simplexMesh,dirichletLogDensity,concentrationAt} from './dirichlet-model.mjs?v=20261108-peloton-1';

const clamp=x=>Math.max(0,Math.min(1,x));
const f=x=>Number(x.toFixed(4));
const text=(key,x,y,label,size=30,extra='')=>`<text data-key="${key}" x="${x}" y="${y}" font-size="${size}" ${extra}>${esc(label)}</text>`;
const math=(key,x,y,w,h,tex,size=30)=>`<g data-key="${key}">${mathLabel(x,y,w,h,tex,size)}</g>`;
const mesh=simplexMesh(20);
const meshPaths=mesh.map(cell=>cell.vertices.map(theta=>simplexPoint(theta).map(f).join(',')).join(' '));
const point=simplexPoint(exampleMixture),center=simplexPoint([1/3,1/3,1/3]);
const possibilities=[[.8,.1,.1],[.1,.75,.15],[.1,.1,.8],[.35,.5,.15]].map(simplexPoint);

export function dirichletVisual(stage,p=stage,motion=true) {
  if(![0,1,2].includes(stage))throw new RangeError(`Unknown Dirichlet scene: ${stage}`);
  const q=motion?clamp((p-stage+.5)*2):1;
  const densityReveal=stage===0?0:stage===1?q:1;
  const concentration=stage===2?q:0;
  const meanReveal=clamp(concentration/.2);
  const a=concentrationAt(concentration),alpha=[a,a,a];
  const logs=mesh.map(cell=>dirichletLogDensity(cell.centroid,alpha)),maximum=Math.max(...logs);
  const sparse=a<1,uniform=Math.abs(a-1)<1e-9;
  const shape=uniform?'Uniform over the triangle':sparse?'More density near the corners':'More density near the center';
  const concentrationLabel=Number(a.toFixed(2)).toString();
  const description=densityReveal===0
    ? 'The simplex contains all probability vectors for Movement, Breath, and Reflection. The marked example assigns probabilities 0.5, 0.2, and 0.3.'
    : concentration===0
      ? 'A Dirichlet distribution puts density over possible probability vectors. Choosing a vector is different from choosing one topic from a categorical distribution.'
      : `A symmetric Dirichlet distribution has concentration ${concentrationLabel} in every coordinate. Its mean stays one third for each topic. ${shape}.`;
  const diagramOpacity=1-clamp(concentration/.1);
  const concentrationText=clamp((concentration-.1)/.1);
  return `<div class="dirichlet-flow" data-key="dirichlet-flow" data-visual-family="dirichlet" data-density-progress="${f(densityReveal)}" data-concentration-progress="${f(concentration)}" data-alpha="${f(a)}">
    <svg class="dirichlet-workspace" viewBox="0 0 680 600" role="img" aria-label="${esc(description)}">
      <g data-key="density" opacity="${f(densityReveal)}" aria-hidden="${densityReveal===0}">${mesh.map((cell,i)=>{
        const shade=.12+.78*Math.pow(Math.exp(logs[i]-maximum),.25);
        return `<polygon data-key="density-${i}" points="${meshPaths[i]}" fill="#234e70" fill-opacity="${f(shade)}" stroke="#234e70" stroke-opacity="${f(shade)}" stroke-width=".35"/>`;
      }).join('')}</g>
      <polygon data-key="simplex-boundary" points="${simplexVertices.map(v=>v.join(',')).join(' ')}" fill="none" stroke="#767676" stroke-width="2"/>
      ${text('topic-movement',20,357,topicNames[0],31,`fill="${topicColors[0]}"`)}
      ${text('topic-breath',340,35,topicNames[1],31,`fill="${topicColors[1]}" text-anchor="middle"`)}
      ${text('topic-reflection',660,357,topicNames[2],31,`fill="${topicColors[2]}" text-anchor="end"`)}
      <g data-key="possible-vectors" opacity="${f(densityReveal*.65*diagramOpacity)}" aria-hidden="${densityReveal===0||diagramOpacity===0}">${possibilities.map(([x,y],i)=>`<circle data-key="possible-${i}" cx="${f(x)}" cy="${f(y)}" r="5" fill="#fff" stroke="#606060" stroke-width="2"/>`).join('')}</g>
      <g data-key="selected-vector" opacity="${f(diagramOpacity)}" aria-hidden="${diagramOpacity===0}">
        <circle data-key="selected-dot" cx="${f(point[0])}" cy="${f(point[1])}" r="8" fill="#242424" stroke="#fff" stroke-width="3"/>
        <rect data-key="selected-label-paper" x="${f(point[0]+14)}" y="${f(point[1]-35)}" width="49" height="45" fill="#fff"/>
        ${math('selected-theta',point[0]+12,point[1]-37,53,49,String.raw`\boldsymbol\theta`,31)}
      </g>
      <g data-key="mean-vector" opacity="${f(meanReveal)}" aria-hidden="${meanReveal===0}">
        <circle data-key="mean-dot" cx="${f(center[0])}" cy="${f(center[1])}" r="9" fill="#fff" stroke="#242424" stroke-width="3"/>
        <path data-key="mean-cross" d="M${f(center[0]-5)} ${f(center[1])}H${f(center[0]+5)}M${f(center[0])} ${f(center[1]-5)}V${f(center[1]+5)}" stroke="#242424" stroke-width="2"/>
      </g>
      <g data-key="example-mixture" opacity="${f(diagramOpacity)}" aria-hidden="${diagramOpacity===0}">
        ${exampleMixture.map((share,i)=>{
          const prior=exampleMixture.slice(0,i).reduce((sum,v)=>sum+v,0),x=52+576*prior;
          return `<g data-key="mixture-${i}"><rect data-key="mixture-rect-${i}" x="${f(x)}" y="384" width="${f(576*share)}" height="34" fill="${topicColors[i]}"/>${text(`mixture-number-${i}`,f(x+288*share),411,String(share),28,'fill="#fff" text-anchor="middle"')}</g>`;
        }).join('')}
        ${math('example-theta',110,427,460,48,String.raw`\boldsymbol\theta=(0.5,\,0.2,\,0.3)`,30)}
      </g>
      <g data-key="simplex-definition" opacity="${f(1-densityReveal)}" aria-hidden="${densityReveal===1}">
        ${math('probability-constraints',65,495,550,48,String.raw`\theta_k\geq0,\qquad\sum_{k=1}^{3}\theta_k=1`,30)}
        ${text('space-note',340,576,'Each point is one categorical distribution.',28,'text-anchor="middle"')}
      </g>
      <g data-key="two-draws" opacity="${f(densityReveal*diagramOpacity)}" aria-hidden="${densityReveal===0||diagramOpacity===0}">
        ${math('dirichlet-draw',12,493,328,48,String.raw`\boldsymbol\theta\sim\operatorname{Dir}(\boldsymbol\alpha)`,29)}
        ${math('categorical-draw',342,493,326,48,String.raw`z\sim\operatorname{Cat}(\boldsymbol\theta)`,29)}
        ${text('dirichlet-output',176,568,'a probability vector',27,'text-anchor="middle"')}
        ${text('categorical-output',505,568,'one topic',27,'text-anchor="middle"')}
      </g>
      <g data-key="concentration-comparison" opacity="${f(concentrationText)}" aria-hidden="${concentrationText===0}">
        ${math('concentration-alpha',60,382,560,52,String.raw`\boldsymbol\alpha=${concentrationLabel}(1,1,1)`,32)}
        ${math('fixed-mean',40,446,600,52,String.raw`\mathbb E[\boldsymbol\theta]=(\tfrac13,\tfrac13,\tfrac13)`,31)}
        ${text('density-shape',340,537,shape,29,'text-anchor="middle"')}
        ${text('density-key',340,580,'Darker = greater relative density.',25,'text-anchor="middle"')}
      </g>
    </svg>
  </div>`;
}
