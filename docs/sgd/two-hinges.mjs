import {ramp} from './model.mjs';
import {C,canvas,line,text,set} from './sgd-detail.mjs';

const label=(x,y,value,size,anchor='middle',color=C.ink)=>`<text x="${x}" y="${y}" text-anchor="${anchor}" style="font-size:${size}px;fill:${color}">${value}</text>`;

// This fixed example makes the geometry visible before introducing parameters.
export function renderTwoHinges(plot,detail,scene,p){
 const {w,h}=canvas(plot),large=w>=600,left=large?52:34,right=w-(large?28:18);
 const top=large?78:58,bottom=h-(large?45:32),height=Math.max(30,bottom-top);
 const X=x=>left+(x+1)*(right-left)/2,Y=y=>bottom-height*(.12+.76*y);
 const bracketY=large?43:29,regionFont=large?24:14,hingeFont=large?18:11;
 const model={c:0,units:[{w:2,b:0,a:1},{w:2,b:-1,a:-1}]};
 const hinges=[0,.5],regionSlopes=[0,2,0],regions=[[-1,0,'Flat'],[0,.5,'Rising'],[.5,1,'Flat']];
 let body='';
 regions.forEach(([start,end,name],i)=>{
  const x1=X(start)+3,x2=X(end)-3;
  body+=`<g data-region="${i+1}" data-slope="${regionSlopes[i]}">`;
  body+=label((x1+x2)/2,large?27:18,name,regionFont);
  body+=`<path d="M${x1},${bracketY+6}V${bracketY}H${x2}V${bracketY+6}" fill="none" stroke="#a5a5a5" stroke-width="1.2"/></g>`;
 });
 for(const x of hinges)body+=line(X(x),bracketY+9,X(x),bottom,'#c9c9c9','3 5');
 for(const y of [0,1])body+=line(left,Y(y),right,Y(y),'#e7e7e7')+text(left-9,Y(y)+5,String(y),'small','end');
 body+=line(left,bottom,right,bottom,'#767676');
 for(const x of [-1,0,.5,1])body+=line(X(x),bottom,X(x),bottom+4,'#767676')+text(X(x),bottom+(large?27:20),String(x).replace('-','−'),'small','middle');
 body+=text(right+9,bottom+(large?27:20),'<tspan style="font-family:Georgia,serif;font-style:italic">x</tspan>','small');
 body+=`<path data-target="step" d="M${X(-1)},${Y(0)}H${X(0)}M${X(0)},${Y(1)}H${X(1)}" fill="none" stroke="${C.muted}" stroke-width="2" stroke-dasharray="6 5"/>`;
 body+=`<circle cx="${X(0)}" cy="${Y(1)}" r="${large?5:3.5}" fill="white" stroke="${C.muted}" stroke-width="1.5"/>`;
 const vertices=[-1,0,.5,1];
 body+=`<path data-network="two-hinges" d="${vertices.map((x,i)=>`${i?'L':'M'}${X(x)},${Y(ramp(x,2))}`).join('')}" fill="none" stroke="${C.teal}" stroke-width="${large?3.5:2.6}" stroke-linejoin="round"/>`;
 hinges.forEach((x,i)=>{
  body+=`<circle data-hinge="${x}" cx="${X(x)}" cy="${Y(ramp(x,2))}" r="${large?7:4.5}" fill="${C.rust}" stroke="white" stroke-width="1.5"/>`;
  body+=label(X(x)+(i===0?-10:10),Y(ramp(x,2))+(i===0?-(large?17:13):(large?29:20)),`Hinge ${i+1}`,hingeFont,i===0?'end':'start',C.rust);
 });
 set(plot,body,'This width-two shallow ReLU network has two hinges, at x = 0 and x = 0.5, dividing the input axis into three linear regions. The network approximation f-hat is zero to the left of zero, rises with slope two from zero to one between x = 0 and x = 0.5, and remains one to the right. Its slopes are 0, 2, and 0. The teal network curve is continuous at both hinges. The dashed gray true step f jumps at zero, so this ramp approximates the step but does not equal it.');
 set(detail,'','');
 return {kind:scene.kind,progress:p,width:2,hinges,regionSlopes,model,plotTitle:'Dashed gray: true step · teal: network approximation',detailTitle:''};
}
