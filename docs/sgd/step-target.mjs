import {data} from './model.mjs';
import {C,canvas,line,text,set} from './sgd-detail.mjs';

// Show actual observations, leaving the boundary dots at zero unobscured.
const negativeIndices=Array.from({length:12},(_,i)=>Math.round(32+i*(928-32)/11));
const sampleIndices=[...negativeIndices,...negativeIndices.map(i=>data.length-1-i).reverse()];
const math=s=>`<tspan style="font-family:Georgia,serif;font-style:italic">${s}</tspan>`;

export function renderStepTarget(plot,detail,scene,p){
 const {w,h}=canvas(plot),large=globalThis.innerWidth>=1000;
 const left=large?64:40,right=w-(large?24:16),top=large?44:26,bottom=h-(large?62:43);
 const height=Math.max(20,bottom-top),X=x=>left+(x+1)*(right-left)/2;
 const Y=y=>bottom-height*(.12+.76*y),radius=large?6:4;
 let body='';
 for(const value of [0,1])body+=line(left,Y(value),right,Y(value),'#e7e7e7')+text(left-10,Y(value)+5,String(value),'small','end');
 body+=line(left,bottom,right,bottom,'#767676');
 for(const x of [-1,0,1])body+=line(X(x),bottom,X(x),bottom+5,'#767676')+text(X(x),bottom+(large?25:17),String(x).replace('-','−'),'small','middle');
 body+=text((left+right)/2,h-8,`Input ${math('x')}`,'target-label','middle');
 body+=`<g transform="translate(${large?19:13},${(Y(0)+Y(1))/2}) rotate(-90)">`+text(0,0,`Outcome ${math('y')}`,'target-label','middle')+'</g>';
 body+=`<path data-target="step" d="M${X(-1)},${Y(0)}H${X(0)}M${X(0)},${Y(1)}H${X(1)}" fill="none" stroke="${C.muted}" stroke-width="${large?3:2}" stroke-dasharray="6 5"/>`;
 body+=text(X(.5),Y(1)-(large?21:15),`Target ${math('H')}(${math('x')})`,'target-label','middle',C.muted);
 body+=text(X(-.5),Y(0)-(large?21:15),'Example observations','target-label','middle',C.rust);
 const shownPoints=sampleIndices.map(index=>({index,...data[index]}));
 for(const point of shownPoints)body+=`<circle data-observation="${point.index}" data-x="${point.x}" data-y="${point.y}" cx="${X(point.x)}" cy="${Y(point.y)}" r="${large?4.5:2.6}" fill="${C.rust}"/>`;
 body+=`<circle data-target-boundary="closed-zero" cx="${X(0)}" cy="${Y(0)}" r="${radius}" fill="${C.muted}"/>`;
 body+=`<circle data-target-boundary="open-one" cx="${X(0)}" cy="${Y(1)}" r="${radius}" fill="white" stroke="${C.muted}" stroke-width="2"/>`;
 set(plot,body,'The target step function H(x) is zero for x ≤ 0 and one for x > 0. The closed dot at (0, 0) includes zero; the open dot at (0, 1) excludes zero. The dataset contains 2,048 evenly spaced inputs in [−1, 1], with outcomes y_i = H(x_i) and no noise. Rust points show 24 representative actual observations, each lying exactly on the dashed gray target.');
 set(detail,'','');
 return {kind:scene.kind,progress:p,plotTitle:'Noiseless observations on the target step',detailTitle:'',dataCount:data.length,sampleIndices:[...sampleIndices],shownPoints};
}
