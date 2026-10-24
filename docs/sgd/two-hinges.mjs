import {ramp} from './model.mjs';
import {C,clamp,canvas,line,text,set} from './sgd-detail.mjs';

const mathCache=new Map();
function mathLabel(x,y,tex,color,size){
 if(!mathCache.has(tex))mathCache.set(tex,globalThis.katex.renderToString(tex,{displayMode:false,output:'htmlAndMathml',throwOnError:true,strict:'error'}));
 return `<foreignObject x="${x}" y="${y}" width="${size*4}" height="${size*1.7}" class="probe-math"><div xmlns="http://www.w3.org/1999/xhtml" style="font-size:${size}px;color:${color};line-height:1.2">${mathCache.get(tex)}</div></foreignObject>`;
}

// Narrow a two-ReLU ramp while keeping the target and the axes fixed.
export function renderTwoHinges(plot,detail,scene,p){
 const {w,h}=canvas(plot),large=w>=600,left=large?52:34,right=w-(large?28:18);
 const top=large?66:48,bottom=h-(large?45:32),height=Math.max(30,bottom-top);
 const X=x=>left+(x+1)*(right-left)/2,Y=y=>bottom-height*(.12+.76*y);
 const k=2*12**clamp((p-.12)/.7),model={c:0,units:[{w:k,b:0,a:1},{w:k,b:-1,a:-1}]};
 const hinges=[0,1/k],regionSlopes=[0,k,0],labelSize=large?28:18;
 let body='';
 const legendWidth=large?270:200,legendX=(left+right-legendWidth)/2,swatch=large?42:26,legendY=large?24:18;
 body+=line(legendX,legendY,legendX+swatch,legendY,C.muted,'6 5',large?2.5:2);
 body+=mathLabel(legendX+swatch+10,legendY-labelSize*.65,'f(x)',C.muted,labelSize);
 const second=legendX+legendWidth*.55;
 body+=line(second,legendY,second+swatch,legendY,C.teal,'',large?3.5:2.6);
 body+=mathLabel(second+swatch+10,legendY-labelSize*.65,String.raw`\hat f(x)`,C.teal,labelSize);
 for(const y of [0,1])body+=line(left,Y(y),right,Y(y),'#e7e7e7')+text(left-9,Y(y)+5,String(y),'small','end');
 body+=line(left,bottom,right,bottom,'#767676');
 for(const x of [-1,0,1])body+=line(X(x),bottom,X(x),bottom+4,'#767676')+text(X(x),bottom+(large?27:20),String(x).replace('-','−'),'small','middle');
 body+=text(right+9,bottom+(large?27:20),'<tspan style="font-family:Georgia,serif;font-style:italic">x</tspan>','small');
 const vertices=[-1,0,1/k,1];
 body+=`<path data-network="two-hinges" d="${vertices.map((x,i)=>`${i?'L':'M'}${X(x)},${Y(ramp(x,k))}`).join('')}" fill="none" stroke="${C.teal}" stroke-width="${large?3.5:2.6}" stroke-linejoin="round"/>`;
 // Draw the dashed target on top so both curves remain visible where they agree.
 body+=`<path data-target="step" d="M${X(-1)},${Y(0)}H${X(0)}M${X(0)},${Y(1)}H${X(1)}" fill="none" stroke="${C.muted}" stroke-width="${large?2.5:2}" stroke-dasharray="6 5"/>`;
 body+=`<circle cx="${X(0)}" cy="${Y(0)}" r="${large?4:3}" fill="${C.muted}"/>`;
 body+=`<circle cx="${X(0)}" cy="${Y(1)}" r="${large?5:3.5}" fill="white" stroke="${C.muted}" stroke-width="1.5"/>`;
 set(plot,body,`The dashed gray target f(x) is the fixed step function. The teal approximation f-hat(x) is a width-two shallow ReLU network. As you scroll, its ramp becomes steeper and narrower, with slope increasing from 2 to 24. The current slope is ${k.toFixed(2)}, and the ramp reaches one at x = ${(1/k).toFixed(4)}. The approximation remains continuous and approaches the target outside an increasingly small interval to the right of zero.`);
 set(detail,'','');
 return {kind:scene.kind,progress:p,width:2,steepness:k,hinges,regionSlopes,model,plotTitle:'',detailTitle:''};
}
