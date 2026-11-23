import {survivalProbability} from './bayes.mjs';

// This tree states the simulation's true conditional distribution. Scroll
// reveals branches and fixed leaf values; it never interpolates probabilities.
const C={ink:'#242424',blue:'#234E70',teal:'#007F73',rust:'#A54F32',muted:'#606060'};
const clamp=x=>Math.max(0,Math.min(1,Number.isFinite(x)?x:1));
const text=(x,y,label,size=24,color=C.ink)=>`<text x="${x}" y="${y}" text-anchor="middle" font-size="${size}" fill="${color}">${label}</text>`;
const line=(x1,y1,x2,y2,p)=>`<line x1="${x1}" y1="${y1}" x2="${x1+(x2-x1)*p}" y2="${y1+(y2-y1)*p}" stroke="${C.blue}" stroke-width="2" stroke-linecap="round"/>`;

/** SVG contents: 900×460 desktop, 400×350 mobile. */
export function renderDgp(progress=1,mobile=false){
  const p=clamp(progress), first=clamp(p/.42), second=clamp((p-.36)/.64);
  const center=mobile?200:450, rootY=mobile?28:42;
  const nodes=mobile?[105,295]:[235,665];
  const leaves=mobile?[52,148,252,348]:[115,345,555,785];
  const bearY=mobile?118:157, strategyY=mobile?149:198;
  const leafY=mobile?222:303, valueY=mobile?269:365;
  let s=text(center,rootY,'Bear type',mobile?24:31,C.blue);
  [1,0].forEach((brown,j)=>{
    const x=nodes[j];
    s+=line(center,rootY+15,x,bearY-28,first);
    s+=`<g opacity="${first}">${text(x,bearY,brown?'Brown':'Black',mobile?24:31,C.blue)}${text(x,strategyY,'Strategy',mobile?18:23,C.muted)}</g>`;
    [0,1].forEach((lieDown,k)=>{
      const xx=leaves[j*2+k], probability=survivalProbability('A',brown,lieDown), color=probability>.5?C.teal:C.rust;
      s+=line(x,strategyY+13,xx,leafY-24,second);
      s+=`<g opacity="${second}" data-brown="${brown}" data-lie-down="${lieDown}" data-survival="${probability}">
        ${text(xx,leafY,lieDown?'Lie down':'Fight',mobile?17:25)}
        ${text(xx,valueY,`${(probability*100).toFixed(1)}%`,mobile?28:39,color)}
      </g>`;
    });
  });
  s+=`<g opacity="${second}">${text(center,mobile?322:425,'Probability of survival',mobile?20:25,C.muted)}</g>`;
  return `<g data-dgp="true" data-dgp-progress="${p}" data-dgp-mobile="${mobile}">${s}</g>`;
}
