import {C,canvas,clamp,line,text,set} from './sgd-detail.mjs';

// Establish the goal before introducing either neurons or training data.
export function renderStepTarget(plot,detail,scene,p){
 const {w,h}=canvas(plot),large=globalThis.innerWidth>=1000;
 const left=large?64:32,right=w-(large?32:16),top=large?72:54,bottom=h-(large?52:43);
 const height=Math.max(20,bottom-top),X=x=>left+(x+1)*(right-left)/2;
 const Y=y=>bottom-height*(.08+.84*y);
 const progress=clamp((p-.1)/.8),probeX=-.8+1.6*progress,probeY=probeX>0?1:0;
 const xLabel=probeX.toFixed(2).replace('-','−');
 // Do not round a positive input to “0.00” beside the target value one.
 const inputLabel=probeX!==0&&Math.abs(probeX)<.005?`Input x ${probeX>0?'&gt;':'&lt;'} 0`:`Input x = ${xLabel}`;
 const radius=large?7:4.5,px=X(probeX),py=Y(probeY);
 let body=text((left+right)/2,large?25:19,`${inputLabel} → target y = ${probeY}`,'target-readout','middle',C.rust);
 for(const value of [0,1]){
  body+=line(left,Y(value),right,Y(value),'#e7e7e7')+text(left-10,Y(value)+5,String(value),'small','end');
 }
 body+=line(left,bottom,right,bottom,'#767676');
 for(const x of [-1,0,1])body+=line(X(x),bottom,X(x),bottom+5,'#767676')+text(X(x),bottom+(large?25:17),String(x).replace('-','−'),'small','middle');
 body+=text((left+right)/2,h-8,'Input x','target-label','middle');
 body+=`<path data-target="step" d="M${X(-1)},${Y(0)}H${X(0)}M${X(0)},${Y(1)}H${X(1)}" fill="none" stroke="${C.muted}" stroke-width="${large?3:2}" stroke-dasharray="6 5"/>`;
 body+=text(X(.5),Y(1)-(large?17:12),'Target H(x)','target-label','middle',C.muted);
 body+=`<circle data-target-boundary="closed-zero" cx="${X(0)}" cy="${Y(0)}" r="${radius}" fill="${C.muted}"/>`;
 body+=`<circle data-target-boundary="open-one" cx="${X(0)}" cy="${Y(1)}" r="${radius}" fill="white" stroke="${C.muted}" stroke-width="2"/>`;
 body+=line(px,bottom,px,py,C.rust,'3 4',1.5);
 body+=`<circle data-probe-x="${probeX}" data-probe-y="${probeY}" cx="${px}" cy="${py}" r="${radius+1}" fill="${C.rust}" stroke="white" stroke-width="2"/>`;
 set(plot,body,`The target step function H(x) is zero for x ≤ 0 and one for x > 0. The closed dot at (0, 0) includes zero; the open dot at (0, 1) excludes zero. Scrolling moves the input across the jump. Current input x = ${Number(probeX.toPrecision(6))}, target y = ${probeY}. The dashed gray curve is the target to approximate; no network curve is shown yet.`);
 set(detail,'','');
 return {kind:scene.kind,progress:p,plotTitle:'The curve we want to approximate',detailTitle:'',probeX,probeY};
}
