import {C,canvas,set} from './sgd-detail.mjs';
import {mathText,richText} from './math.mjs';

const number=value=>(value<0?'-':'')+Math.abs(value).toFixed(4);
const label=(x,y,value,color,size,anchor='middle')=>`<text x="${x}" y="${y}" text-anchor="${anchor}" style="fill:${color};font-size:${size}px">${value}</text>`;
const mathLabel=(x,y,value,color,size,anchor='middle',width)=>mathText(x,y,value,'',anchor,color,{size,width});
const circle=(x,y,r,color)=>`<circle cx="${x}" cy="${y}" r="${r}" fill="white" stroke="${color}" stroke-width="1.5"/>`;
function outputLabel(right,y,size){
 return mathLabel(right,y+size,String.raw`\hat f(x;\theta,\phi)`,C.blue,size,'end',size*7);
}
function arrow(x1,y1,x2,y2,color,dashed=false){
 const angle=Math.atan2(y2-y1,x2-x1),size=5;
 return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="1.5" ${dashed?'stroke-dasharray="4 3"':''}/><path d="M${x2-size*Math.cos(angle-.55)},${y2-size*Math.sin(angle-.55)}L${x2},${y2}L${x2-size*Math.cos(angle+.55)},${y2-size*Math.sin(angle+.55)}" fill="none" stroke="${color}" stroke-width="1.5"/>`;
}
function connection(from,to,fromRadius,toRadius,color){
 const angle=Math.atan2(to.y-from.y,to.x-from.x);
 return arrow(from.x+fromRadius*Math.cos(angle),from.y+fromRadius*Math.sin(angle),to.x-(toRadius+3)*Math.cos(angle),to.y-(toRadius+3)*Math.sin(angle),color);
}

// Draw the same seven-parameter model used by the prediction curve.
// Constant-1 nodes make the three additive biases explicit without a derivation.
export function renderInitialNetwork(svg,model){
 const {w,h}=canvas(svg),compact=w<330||h<240,wide=globalThis.innerWidth>=1600;
 const font=compact?10:wide?22:18,nodeRadius=compact?16:wide?30:26,biasRadius=compact?8:wide?15:13;
 const input={x:compact?15:wide?27:23,y:h/2},output={x:w-(compact?20:wide?36:30),y:h/2},hiddenX=w*.49;
 const hiddenYs=[h*.31,h*.69],biasYs=[compact?16:wide?30:24,h-(compact?18:wide?46:38)];
 const displayedParameters=model.units.flatMap((unit,i)=>{
  const sub=i===0?'₁':'₂';
  return [
   {parameter:`theta${i+1}1`,label:`θ${sub}₁`,tex:`\\theta_{${i+1}1}`,value:unit.w},
   {parameter:`theta${i+1}0`,label:`θ${sub}₀`,tex:`\\theta_{${i+1}0}`,value:unit.b},
   {parameter:`phi${i+1}`,label:`φ${sub}`,tex:`\\phi_{${i+1}}`,value:unit.a},
  ];
 }).concat({parameter:'phi0',label:'φ₀',tex:'\\phi_0',value:model.c});
 const valueLabel=(parameter,x,y,color,anchor='middle',stacked=false)=>{
  const entry=displayedParameters.find(item=>item.parameter===parameter);
  const content=stacked
   ?mathLabel(x,y-(wide?11:9),entry.tex,color,font,anchor)+mathLabel(x,y+(wide?17:13),number(entry.value),color,font,anchor)
   :mathLabel(x,y,`${entry.tex}=${number(entry.value)}`,color,font,anchor);
  return `<g class="network-math-label" data-parameter="${entry.parameter}" data-value="${entry.value}">${content}</g>`;
 };
 let body='';
 for(let i=0;i<2;i++){
  const node={x:hiddenX,y:hiddenYs[i]},color=i===0?C.blue:C.rust,biasY=biasYs[i];
  body+=connection(input,node,compact?10:13,nodeRadius,color)+connection(node,output,nodeRadius,nodeRadius,color);
  body+=arrow(hiddenX,biasY+(i===0?biasRadius:-biasRadius),hiddenX,node.y+(i===0?-nodeRadius-3:nodeRadius+3),color,true);
  body+='<g class="network-bias-node">'+circle(hiddenX,biasY,biasRadius,color)+label(hiddenX,biasY+(compact?3.5:wide?7:5),'1',color,compact?10:wide?20:16)+'</g>';
  body+=circle(node.x,node.y,nodeRadius,color)+mathLabel(node.x,node.y-(compact?2:3),`h_{${i+1}}`,color,compact?12:wide?24:20)+mathLabel(node.x,node.y+(compact?10:wide?18:15),'\\operatorname{ReLU}',color,compact?8:wide?16:13);
  const weightY=(input.y+node.y)/2+(i===0?-(compact?9:wide?27:21):(compact?20:wide?30:24));
  body+=valueLabel(`theta${i+1}1`,(input.x+hiddenX)/2-4,weightY,color,'middle',!compact);
  body+=valueLabel(`phi${i+1}`,(hiddenX+output.x)/2,weightY,color,'middle',!compact);
  body+=valueLabel(`theta${i+1}0`,hiddenX-(compact?12:wide?24:20),biasY+(i===0?(wide&&!compact?20:16):-(compact?14:wide?12:8)),color,'end');
 }
 const outputBiasY=compact?16:wide?30:24;
 body+=arrow(output.x,outputBiasY+biasRadius,output.x,output.y-nodeRadius-3,C.teal,true);
 body+='<g class="network-bias-node">'+circle(output.x,outputBiasY,biasRadius,C.teal)+label(output.x,outputBiasY+(compact?3.5:wide?7:5),'1',C.teal,compact?10:wide?20:16)+'</g>';
 body+=valueLabel('phi0',output.x-(compact?12:wide?24:20),outputBiasY+(wide&&!compact?20:16),C.teal,'end');
 body+=circle(input.x,input.y,compact?10:wide?17:14,C.muted)+mathLabel(input.x,input.y+(compact?4:wide?8:6),'x',C.ink,compact?12:wide?24:20);
 body+=circle(output.x,output.y,nodeRadius,C.blue)+mathLabel(output.x,output.y+(compact?5:wide?9:7),'\\sum',C.blue,compact?18:wide?34:28);
 body+=outputLabel(w-3,output.y+nodeRadius+(compact?3:wide?8:6),compact?11:wide?22:18);
 const keySize=compact?9:wide?20:16,keyY=h-(compact?18:wide?31:24);
 body+=richText(3,keyY,String.raw`\(\theta\): hidden layer`,'','start',C.muted,{size:keySize})+richText(w-3,keyY,String.raw`\(\phi\): output layer`,'','end',C.muted,{size:keySize});
 body+=label(3,h-3,'Dashed: bias',C.muted,keySize,'start')+label(w-3,h-3,'1: constant',C.muted,keySize,'end');
 set(svg,body,`Initial network: input x connects to two hidden ReLU neurons. ${displayedParameters.map(item=>`${item.label} = ${item.value}`).join('; ')}. Hidden biases θ₁₀ and θ₂₀ enter the hidden neurons from constant 1; output bias φ₀ enters the sum from constant 1. Network output f-hat(x; θ, φ) is φ₁ times h₁ plus φ₂ times h₂ plus φ₀. Hidden-layer parameters form θ; output-layer parameters form φ.`);
 return displayedParameters;
}
