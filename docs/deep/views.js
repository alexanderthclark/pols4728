import { curveGeometry } from './geometry.mjs';
import { evaluateNetwork, surfaceValue, INPUT_DOMAIN, FIRST_LAYER_KNOTS, OUTPUT_WEIGHTS, networkParameters } from './model.mjs';

const NS='http://www.w3.org/2000/svg';
const INK='#242424',BLUE='#234E70',RUST='#A54F32',RULE='#767676';
const COLORS=[BLUE,RUST,'#606060'];
const number=value=>Math.abs(value)<.005?'0':Number(value.toFixed(2)).toString().replace('-','−');
export const modeLabels={h11:'h₁,₁',h12:'h₁,₂',h13:'h₁,₃',shallow:'q₁',preactivation:'q₁',unit1:'h₂,₁',unit2:'h₂,₂',unit3:'h₂,₃',combined:'y'};
function node(tag,attributes={},value) {
  const element=document.createElementNS(NS,tag);
  for(const [name,value] of Object.entries(attributes)) element.setAttribute(name,value);
  if(value!==undefined) element.textContent=value;
  return element;
}
function text(svg,x,y,value,{anchor='middle',size=12,color=INK,...other}={}) {
  svg.append(node('text',{x,y,'text-anchor':anchor,'font-size':size,fill:color,...other},value));
}
function path(svg,points,attributes={},close=false) {
  const d=points.map((p,i)=>(i?'L':'M')+p.map(v=>v.toFixed(3)).join(',')).join(' ')+(close?' Z':'');
  svg.append(node('path',{d,fill:'none',...attributes}));
}
function reset(svg,width,height,title,description) {
  svg.replaceChildren(node('title',{},title),node('desc',{},description));
  svg.setAttribute('viewBox','0 0 '+width+' '+height);
  svg.setAttribute('height',height);
  svg.setAttribute('aria-label',title+'. '+description);
}
function cross(svg,p,r=4,color=RUST) {
  path(svg,[[p[0]-r,p[1]-r],[p[0]+r,p[1]+r]],{stroke:color,'stroke-width':2});
  path(svg,[[p[0]-r,p[1]+r],[p[0]+r,p[1]-r]],{stroke:color,'stroke-width':2});
}
const near=(a,b)=>Math.abs(a-b)<1e-7;

export class LinkedFigures {
  constructor(surface,features,onPoint) {
    this.surface=surface;this.features=features;this.onPoint=onPoint;
    this.options=null;this.animation=0;this.mix=1;this.previousMode=null;
    this.motion=matchMedia('(prefers-reduced-motion: reduce)');
    surface.addEventListener('pointerdown',event=>{
      if(!this.scale || !this.options.showProbe) return;
      const bounds=surface.getBoundingClientRect();
      const sx=(event.clientX-bounds.left)*this.scale.width/bounds.width;
      if(sx<this.scale.left || sx>this.scale.right) return;
      onPoint(INPUT_DOMAIN[0]+(sx-this.scale.left)/(this.scale.right-this.scale.left)*(INPUT_DOMAIN[1]-INPUT_DOMAIN[0]));
    });
    new ResizeObserver(()=>this.draw()).observe(surface.closest('.stage-figure'));
  }
  update(options,{animate=false}={}) {
    cancelAnimationFrame(this.animation);
    const previous=this.options;
    this.options={...options};
    this.geometry=curveGeometry(options.mode,{threshold:options.threshold});
    this.mix=1;this.previousMode=null;
    if(animate && previous && !this.motion.matches && previous.threshold===options.threshold && previous.scene!==1 && options.scene!==1 && previous.mode!==options.mode) {
      this.previousMode=previous.mode;this.mix=0;
      const start=performance.now();
      const tick=now=>{
        const p=Math.min(1,(now-start)/400);
        this.mix=p*p*(3-2*p);this.draw();
        if(p<1) this.animation=requestAnimationFrame(tick);
      };
      this.animation=requestAnimationFrame(tick);
    } else this.draw();
  }
  value(x) {
    const target=surfaceValue(this.options.mode,x,{threshold:this.options.threshold});
    if(this.mix===1 || !this.previousMode) return target;
    const from=surfaceValue(this.previousMode,x,{threshold:this.options.threshold});
    return from+this.mix*(target-from);
  }
  draw() {
    if(!this.options) return;
    this.drawCurve();this.drawFeatures();
  }
  drawCurve() {
    const {scene,mode,threshold,x:input,showProbe,cover}=this.options;
    const svg=this.surface;
    const width=Math.max(200,svg.parentElement.getBoundingClientRect().width);
    const height=Math.max(105,svg.closest('.stage-figure').getBoundingClientRect().height||350);
    const narrow=width<500;
    const left=narrow?35:48,right=width-(narrow?12:22),top=28,bottom=height-34;
    const first=scene===1;
    const limits=first?[-.2,3.5]:[-.75,1.65];
    const px=x=>left+(x-INPUT_DOMAIN[0])/(INPUT_DOMAIN[1]-INPUT_DOMAIN[0])*(right-left);
    const py=y=>bottom-(y-limits[0])/(limits[1]-limits[0])*(bottom-top);
    this.scale={width,left,right};
    const marks=scene===3?'Open rust circles mark zero crossings.':mode==='shallow'||mode==='preactivation'?'Blue dots mark the original bends.':'Rust dots mark new bends inside those intervals.';
    const description=cover?'One input x maps to one response y. This is the output curve the story will build.':first?'The three fixed ReLU ramps have hinges at zero, one, and two.':
      'Gray guides mark the fixed first-layer hinges at zero, one, and two. '+marks+' Select an input by clicking the curve; a keyboard input control is available in the last frame.';
    reset(svg,width,height,first?'Three first-layer activation curves':'Response curve '+modeLabels[mode],description);
    if(!cover) {
      for(const [a,b] of [[0,1],[2,INPUT_DOMAIN[1]]]) svg.append(node('rect',{x:px(a),y:top,width:px(b)-px(a),height:bottom-top,fill:'#F5F7F9'}));
      for(const knot of FIRST_LAYER_KNOTS) path(svg,[[px(knot),top],[px(knot),bottom]],{stroke:'#C8CDD1','stroke-width':.9,'stroke-dasharray':'3 4'});
    }
    path(svg,[[left,top],[left,bottom],[right,bottom]],{stroke:RULE,'stroke-width':.9});
    path(svg,[[left,py(0)],[right,py(0)]],{stroke:'#B8B8B8','stroke-width':.8});
    for(const tick of first?[0,1,2,3]:[-.5,0,.5,1,1.5]) {
      if(narrow && height<180 && tick!==0 && tick!==1) continue;
      path(svg,[[left-3,py(tick)],[left,py(tick)]],{stroke:RULE,'stroke-width':.8});
      text(svg,left-7,py(tick)+4,number(tick),{anchor:'end',size:12});
    }
    for(const tick of [0,1,2,3]) {
      path(svg,[[px(tick),bottom],[px(tick),bottom+3]],{stroke:RULE,'stroke-width':.8});
      text(svg,px(tick),bottom+18,number(tick),{size:12});
    }
    text(svg,left,15,first?'Activation':'Response '+modeLabels[mode],{anchor:'start',size:12});
    text(svg,right,height-3,'Input x',{anchor:'end',size:12});
    if(first) {
      ['h11','h12','h13'].forEach((name,i)=>{
        const geometry=curveGeometry(name,{threshold});
        path(svg,geometry.points.map(p=>[px(p[0]),py(p[1])]),{stroke:COLORS[i],'stroke-width':2.2});
        svg.append(node('circle',{cx:px(i),cy:py(0),r:3,fill:COLORS[i]}));
        text(svg,px(3.12),py(surfaceValue(name,3.12,{threshold}))-7,modeLabels[name],{anchor:'end',size:12,color:COLORS[i]});
        if(showProbe) cross(svg,[px(input),py(surfaceValue(name,input,{threshold}))],3,COLORS[i]);
      });
    } else {
      if(scene===4) {
        const before=curveGeometry('preactivation',{threshold});
        path(svg,before.points.map(p=>[px(p[0]),py(p[1])]),{stroke:'#9DA8AF','stroke-width':1.3,'stroke-dasharray':'5 4'});
      }
      let xs=this.geometry.points.map(p=>p[0]);
      if(this.previousMode && this.mix<1) xs=[...new Set([...xs,...curveGeometry(this.previousMode,{threshold}).points.map(p=>p[0])])].sort((a,b)=>a-b);
      path(svg,xs.map(x=>[px(x),py(this.value(x))]),{stroke:BLUE,'stroke-width':2.5,'stroke-linejoin':'round'});
      if(this.mix===1 && !cover) {
        for(const bend of this.geometry.bends) {
          const old=FIRST_LAYER_KNOTS.some(k=>near(k,bend.x));
          svg.append(node('circle',{cx:px(bend.x),cy:py(bend.y),r:narrow?3:3.5,fill:old?BLUE:RUST,stroke:'#FFFFFF','stroke-width':.6}));
        }
        const roots=scene===3?this.geometry.secondPreactivationRoots[0]:scene===4?this.geometry.secondPreactivationRoots[0]:scene===5?this.geometry.secondPreactivationRoots[1]:[];
        for(const root of roots) {
          if(scene===3) svg.append(node('circle',{cx:px(root),cy:py(0),r:4,fill:'#FFFFFF',stroke:RUST,'stroke-width':1.6}));
          text(svg,px(root),py(0)+18,number(root),{size:12,color:RUST});
        }
      }
      if(showProbe) cross(svg,[px(input),py(this.value(input))],narrow?4:5);
    }
    if(showProbe) {
      path(svg,[[px(input),top],[px(input),bottom]],{stroke:RUST,'stroke-width':.8,'stroke-dasharray':'2 4',opacity:.55});
      text(svg,right,15,'x = '+number(input),{anchor:'end',size:12,color:RUST});
    }
  }
  drawFeatures() {
    const svg=this.features;
    if(svg.parentElement.hidden || svg.parentElement.getBoundingClientRect().width<1) return;
    const {threshold,x:input}=this.options;
    const width=svg.parentElement.getBoundingClientRect().width;
    const height=svg.parentElement.getBoundingClientRect().height;
    const gap=14,panel=(width-2*gap)/3;
    const first=this.options.mode==='shallow';
    reset(svg,width,height,first?'Three first-layer features':'Three second-layer features','Each miniature curve is a separate '+(first?'first':'second')+'-layer activation over the same input range. The vertical mark follows the selected input.');
    (first?['h11','h12','h13']:['unit1','unit2','unit3']).forEach((mode,i)=>{
      const start=i*(panel+gap),left=start+3,right=start+panel-3,top=19,bottom=height-8;
      const px=x=>left+(x-INPUT_DOMAIN[0])/(INPUT_DOMAIN[1]-INPUT_DOMAIN[0])*(right-left);
      const py=y=>bottom-y/(first?3.5:1.3)*(bottom-top);
      text(svg,start+panel/2,12,modeLabels[mode],{size:12,color:COLORS[i]});
      path(svg,[[left,bottom],[right,bottom]],{stroke:'#C5C5C5','stroke-width':.8});
      path(svg,curveGeometry(mode,{threshold}).points.map(p=>[px(p[0]),py(p[1])]),{stroke:COLORS[i],'stroke-width':1.5});
      path(svg,[[px(input),top],[px(input),bottom]],{stroke:RUST,'stroke-width':.7,'stroke-dasharray':'2 3'});
      svg.append(node('circle',{cx:px(input),cy:py(surfaceValue(mode,input,{threshold})),r:2.5,fill:COLORS[i]}));
    });
  }
}

export function renderNetwork(svg,{scene,mode,threshold,x}) {
  const width=Math.max(260,svg.parentElement.getBoundingClientRect().width),narrow=width<500;
  const height=narrow?64:122;
  const hasSecond=scene>=3 && !(scene===7 && mode==='shallow');
  const value=evaluateNetwork(x,{threshold}),parameters=networkParameters({threshold});
  const second=mode.startsWith('unit')?Number(mode.at(-1))-1:mode==='preactivation'?0:-1;
  const levels=hasSecond?[.07,.35,.65,.94]:[.1,.5,.92];
  const positions=levels.map(fraction=>fraction*width);
  const top=narrow?17:25,low=height-(narrow?7:15),middle=(top+low)/2;
  const ys=[top,middle,low];
  const activations=scene===3?'First-layer fills show positive activations; the outlined new units are being inspected before ReLU.':'Filled hidden units have positive activations.';
  const connections=hasSecond?' Each second-layer unit receives all three first-layer activations.':' The affine output combines the first-layer features.';
  reset(svg,width,height,hasSecond?'Two hidden layers, three units per layer':'One hidden layer, three units','The selected input is '+number(x)+'. '+activations+connections);
  const input={x:positions[0],y:middle},h1=ys.map(y=>({x:positions[1],y})),h2=hasSecond?ys.map(y=>({x:positions[2],y})):[];
  const output={x:positions.at(-1),y:middle};
  function edge(a,b,{focus=false,label}={}) {
    path(svg,[[a.x,a.y],[b.x,b.y]],{stroke:focus?RUST:'#C5C5C5','stroke-width':focus?1.5:.8});
    if(label!==undefined && !narrow) text(svg,a.x*.42+b.x*.58,a.y*.42+b.y*.58-4,label,{size:11,color:focus?RUST:INK});
  }
  h1.forEach(p=>edge(input,p,{focus:scene===1}));
  if(hasSecond) {
    h1.forEach((a,i)=>h2.forEach((b,j)=>edge(a,b,{focus:second===j,label:second===j?number(parameters.omega1[j][i]):undefined})));
    h2.forEach((a,i)=>edge(a,output,{focus:mode==='combined',label:mode==='combined'?number(OUTPUT_WEIGHTS[i]):undefined}));
  } else h1.forEach((a,i)=>edge(a,output,{focus:scene===2||mode==='shallow',label:scene===2||mode==='shallow'?number([1,-2,2][i]):undefined}));
  function unit(p,label,active,focused=false) {
    svg.append(node('circle',{cx:p.x,cy:p.y,r:narrow?4:6,fill:active?'#D9E5EE':'#FFFFFF',stroke:focused?RUST:BLUE,'stroke-width':focused?1.6:1}));
    text(svg,p.x+(narrow?6:9),p.y+3,label,{anchor:'start',size:narrow?11:12});
  }
  unit(input,'x',false);
  h1.forEach((p,i)=>unit(p,'h₁,'+(i+1),value.h1[i]>0,scene===1));
  h2.forEach((p,i)=>unit(p,'h₂,'+(i+1),scene===3?false:value.h2[i]>0,second===i));
  unit(output,hasSecond?'y':'q₁',false,mode==='combined'||mode==='shallow');
  if(!narrow) {
    text(svg,positions[0],11,'Input',{size:11});text(svg,positions[1],11,'Hidden layer 1',{size:11});
    if(hasSecond) text(svg,positions[2],11,'Hidden layer 2',{size:11});
    text(svg,positions.at(-1),11,'Output',{size:11});
  }
}
