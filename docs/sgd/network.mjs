import {C,canvas,set} from './sgd-detail.mjs';

const number=value=>(value<0?'−':'')+Math.abs(value).toFixed(4);
const label=(x,y,value,color,size,anchor='middle')=>`<text x="${x}" y="${y}" text-anchor="${anchor}" style="fill:${color};font-size:${size}px">${value}</text>`;
const circle=(x,y,r,color)=>`<circle cx="${x}" cy="${y}" r="${r}" fill="white" stroke="${color}" stroke-width="1.5"/>`;
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
 const {w,h}=canvas(svg),compact=w<330||h<240;
 const font=compact?10:13,nodeRadius=compact?16:22,biasRadius=compact?8:11;
 const input={x:compact?15:20,y:h/2},output={x:w-(compact?20:25),y:h/2},hiddenX=w*.49;
 const hiddenYs=[h*.31,h*.69],biasYs=[compact?16:30,h-(compact?18:34)];
 const displayedParameters=model.units.flatMap((unit,i)=>['w','b','a'].map(key=>({parameter:`${key}${i+1}`,label:`${key}${i===0?'₁':'₂'}`,value:unit[key]}))).concat({parameter:'c',label:'c',value:model.c});
 const valueLabel=(parameter,x,y,color,anchor='middle')=>{
  const entry=displayedParameters.find(item=>item.parameter===parameter);
  return `<g data-parameter="${entry.parameter}" data-value="${entry.value}" style="paint-order:stroke;stroke:white;stroke-width:3;stroke-linejoin:round">${label(x,y,`${entry.label} = ${number(entry.value)}`,color,font,anchor)}</g>`;
 };
 let body='';
 for(let i=0;i<2;i++){
  const node={x:hiddenX,y:hiddenYs[i]},color=i===0?C.blue:C.rust,biasY=biasYs[i],sub=i===0?'₁':'₂';
  body+=connection(input,node,compact?10:13,nodeRadius,color)+connection(node,output,nodeRadius,nodeRadius,color);
  body+=arrow(hiddenX,biasY+(i===0?biasRadius:-biasRadius),hiddenX,node.y+(i===0?-nodeRadius-3:nodeRadius+3),color,true);
  body+=circle(hiddenX,biasY,biasRadius,color)+label(hiddenX,biasY+3.5,'1',color,compact?10:12);
  body+=circle(node.x,node.y,nodeRadius,color)+label(node.x,node.y-(compact?2:3),`h${sub}`,color,compact?12:16)+label(node.x,node.y+(compact?10:12),'ReLU',color,compact?8:10);
  const weightY=(input.y+node.y)/2+(i===0?-(compact?9:14):(compact?20:25));
  body+=valueLabel(`w${i+1}`,(input.x+hiddenX)/2-4,weightY,color);
  body+=valueLabel(`a${i+1}`,(hiddenX+output.x)/2,weightY,color);
  body+=valueLabel(`b${i+1}`,hiddenX-(compact?12:17),biasY+(i===0?16:-6),color,'end');
 }
 const outputBiasY=compact?16:30;
 body+=arrow(output.x,outputBiasY+biasRadius,output.x,output.y-nodeRadius-3,C.teal,true);
 body+=circle(output.x,outputBiasY,biasRadius,C.teal)+label(output.x,outputBiasY+3.5,'1',C.teal,compact?10:12);
 body+=valueLabel('c',output.x-(compact?12:17),outputBiasY+16,C.teal,'end');
 body+=circle(input.x,input.y,compact?10:13,C.muted)+label(input.x,input.y+4,'x',C.ink,compact?12:16);
 body+=circle(output.x,output.y,nodeRadius,C.teal)+label(output.x,output.y+5,'Σ',C.teal,compact?18:24);
 body+=label(w-3,output.y+nodeRadius+(compact?15:22),'fθ(x)',C.teal,compact?11:14,'end');
 body+=label(3,h-3,'Dashed: bias',C.muted,compact?9:11,'start')+label(w-3,h-3,'1: constant',C.muted,compact?9:11,'end');
 set(svg,body,`Initial network: input x connects to two hidden ReLU neurons. ${displayedParameters.map(item=>`${item.parameter} = ${item.value}`).join('; ')}. Biases b1 and b2 enter the hidden neurons from constant 1; output bias c enters the sum from constant 1. Output f theta of x is a1 times h1 plus a2 times h2 plus c.`);
 return displayedParameters;
}
