import { curveGeometry, foldedCurveGeometry } from './geometry.mjs';
import { evaluateNetwork, surfaceValue, foldedValue, foldedPreimages, INPUT_DOMAIN, FIRST_LAYER_KNOTS, OUTPUT_WEIGHTS, networkParameters } from './model.mjs';

const NS='http://www.w3.org/2000/svg';
const INK='#242424',BLUE='#234E70',RUST='#A54F32',RULE='#767676';
const COLORS=[BLUE,RUST,'#606060'];
const number=value=>Math.abs(value)<.005?'0':Number(value.toFixed(2)).toString().replace('-','−');
export const modeLabels={h11:'h₁,₁',h12:'h₁,₂',h13:'h₁,₃',fold:'q(x)',shallow:'q(x)',preactivation:'q − τ',unit1:'h₂,₁',unit2:'h₂,₂',unit3:'h₂,₃',combined:'y',downstream:'g(q)'};
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
const foldScene=scene=>scene===3||scene===5;

export class LinkedFigures {
  constructor(surface,features,onPoint) {
    this.surface=surface;this.features=features;this.onPoint=onPoint;
    this.options=null;this.animation=0;this.foldMix=1;
    this.motion=matchMedia('(prefers-reduced-motion: reduce)');
    surface.addEventListener('pointerdown',event=>{
      if(!this.scale || !this.options.showProbe || this.foldMix<1) return;
      const transform=surface.getScreenCTM();
      if(!transform) return;
      const pointer=surface.createSVGPoint();pointer.x=event.clientX;pointer.y=event.clientY;
      const local=pointer.matrixTransform(transform.inverse());
      const sx=local.x,sy=local.y;
      if(sx<this.scale.left || sx>this.scale.right) return;
      const coordinate=this.scale.domain[0]+(sx-this.scale.left)/(this.scale.right-this.scale.left)*(this.scale.domain[1]-this.scale.domain[0]);
      if(this.scale.type==='fold') {
        const row=this.scale.rows.map((y,index)=>({index,distance:Math.abs(sy-y)})).sort((a,b)=>a.distance-b.distance)[0].index;
        onPoint(foldedPreimages(Math.max(0,Math.min(1,coordinate)))[row]);
      } else if(this.scale.type==='downstream') {
        const row=this.options.x<=1?0:this.options.x<=2?1:2;
        onPoint(foldedPreimages(Math.max(0,Math.min(1,coordinate)))[row]);
      } else onPoint(coordinate);
    });
    new ResizeObserver(()=>this.draw()).observe(surface.closest('.visual-stage'));
  }
  update(options,{animate=false}={}) {
    const previous=this.options;
    cancelAnimationFrame(this.animation);
    this.options={...options};this.foldMix=1;
    this.geometry=options.mode==='downstream'?foldedCurveGeometry({threshold:options.threshold}):curveGeometry(options.mode,{threshold:options.threshold});
    if(animate && options.scene===3 && previous?.scene!==3) this.replayFold();
    else this.draw();
  }
  replayFold() {
    if(!this.options || !foldScene(this.options.scene)) return;
    cancelAnimationFrame(this.animation);
    if(this.motion.matches) {this.foldMix=1;this.draw();return;}
    this.foldMix=0;this.draw();
    const start=performance.now();
    const tick=now=>{
      const p=Math.min(1,(now-start)/1000);
      this.foldMix=p*p*(3-2*p);this.draw();
      if(p<1) this.animation=requestAnimationFrame(tick);
    };
    this.animation=requestAnimationFrame(tick);
  }
  draw() {
    if(!this.options) return;
    if(foldScene(this.options.scene)) this.drawFold();
    else this.drawCurve();
    this.drawPattern();
  }
  dimensions() {
    return {width:Math.max(200,this.surface.parentElement.getBoundingClientRect().width),height:Math.max(110,this.surface.parentElement.getBoundingClientRect().height||350)};
  }
  drawFold() {
    const svg=this.surface,{scene,threshold,x:input}=this.options;
    const {width,height}=this.dimensions(),narrow=width<500;
    const left=narrow?14:30,right=width-(narrow?16:30);
    const top=60,bottom=height-24;
    const rows=[top,(top+bottom)/2,bottom];
    const px=q=>left+q*(right-left);
    const middle=(top+bottom)/2,p=this.foldMix;
    this.scale={type:'fold',width,height,left,right,domain:[0,1],rows};
    const showThresholds=scene===5;
    reset(svg,width,height,'The input line folds into three passes through q',
      'The first pass is x from zero to one, the second is x from two back to one, and the third is x from two to three. Horizontal position is the folded coordinate q, not x. Vertical row spacing separates the overlapping intervals. '+(showThresholds?'Each of three thresholds intersects all three passes.':'Select a point on a row to see three inputs with the same q.'));
    // Separate the coincident pieces vertically, retaining the two connected turns.
    // The starting state is a straight input line; each vertex then moves to q(x).
    const vertices=[[0,0,0],[1,1,0],[1,1,1],[2,0,1],[2,0,2],[3,1,2]];
    const points=vertices.map(([x,q,row])=>[left+(right-left)*((1-p)*x/3+p*q),middle+(rows[row]-middle)*p]);
    path(svg,points,{stroke:BLUE,'stroke-width':2.5,'stroke-linejoin':'round'});
    if(p<.98) {
      text(svg,left,18,'Input x',{anchor:'start',size:12});
      text(svg,left,middle+22,'0',{anchor:'start',size:12});text(svg,right,middle+22,'3',{anchor:'end',size:12});
      return;
    }
    text(svg,left,14,'Folded coordinate q',{anchor:'start',size:12});
    for(const q of [0,1]) text(svg,px(q),30,number(q),{size:12});
    const labels=['x: 0 → 1','x: 2 ← 1','x: 2 → 3'];
    rows.forEach((y,i)=>text(svg,left+5,y-12,labels[i],{anchor:'start',size:12,color:BLUE}));
    const thresholds=[.2,threshold,.8];
    if(showThresholds) {
      thresholds.forEach((q,j)=>{
        path(svg,[[px(q),33],[px(q),bottom+5]],{stroke:RUST,'stroke-width':.9,'stroke-dasharray':j===1?'4 3':'1 3'});
        text(svg,px(q),30,number(q),{size:12,color:RUST});
        foldedPreimages(q).forEach((x,i)=>{
          svg.append(node('circle',{cx:px(q),cy:rows[i],r:3.5,fill:RUST,stroke:'#FFFFFF','stroke-width':.7}));
          text(svg,px(q),rows[i]+17,number(x),{size:12,color:RUST});
        });
      });
    }
    const q=evaluateNetwork(input,{threshold}).fold;
    if(q>=0 && q<=1) {
      path(svg,[[px(q),35],[px(q),bottom+5]],{stroke:BLUE,'stroke-width':.8,'stroke-dasharray':'2 4',opacity:.65});
      foldedPreimages(q).forEach((x,i)=>{
        const selected=near(input,x);
        svg.append(node('circle',{cx:px(q),cy:rows[i],r:selected?5:4,fill:selected?BLUE:'#FFFFFF',stroke:BLUE,'stroke-width':1.4}));
        if(!showThresholds) text(svg,px(q),rows[i]+17,'x = '+number(x),{size:12,color:BLUE});
      });
      if(!showThresholds) text(svg,px(q),30,number(q),{size:12,color:BLUE});
    }
  }
  drawCurve() {
    const {scene,mode,threshold,x:input,showProbe}=this.options;
    const svg=this.surface,{width,height}=this.dimensions();
    const narrow=width<500,left=narrow?35:48,right=width-(narrow?12:22),top=27,bottom=height-32;
    const first=scene===1,downstream=mode==='downstream';
    const domain=downstream?[0,1]:INPUT_DOMAIN;
    const coverValues=this.geometry.points.map(point=>point[1]);
    const coverLimits=[Math.min(0,...coverValues)-.05,Math.max(.4,...coverValues)+.05];
    const limits=first?[-.2,3.5]:scene===0?coverLimits:mode==='fold'?[-.2,1.35]:mode.startsWith('unit')?[-.2,1.1]:[-.4,.85];
    const px=x=>left+(x-domain[0])/(domain[1]-domain[0])*(right-left);
    const py=y=>bottom-(y-limits[0])/(limits[1]-limits[0])*(bottom-top);
    const selected=downstream?evaluateNetwork(input,{threshold}).fold:input;
    const value=x=>downstream?foldedValue(x,{threshold}):surfaceValue(mode,x,{threshold});
    this.scale={type:downstream?'downstream':'curve',width,height,left,right,domain};
    reset(svg,width,height,first?'Three first-layer activation curves':'Response curve '+modeLabels[mode],
      first?'The three ReLU ramps have hinges at zero, one, and two.':
      (downstream?'Horizontal position is q. Three downstream joints occur at the thresholds.':'Horizontal position is original input x. Rust dots are copied downstream joints; blue dots are surviving first-layer turns.')+' Height is '+modeLabels[mode]+'. Select an input by clicking the curve; a keyboard input control is available in the last frame.');
    if(!downstream && !first && scene!==0) {
      for(const knot of [1,2]) path(svg,[[px(knot),top],[px(knot),bottom]],{stroke:'#B7BDC1','stroke-width':.8,'stroke-dasharray':'3 4'});
    }
    path(svg,[[left,top],[left,bottom],[right,bottom]],{stroke:RULE,'stroke-width':.9});
    path(svg,[[left,py(0)],[right,py(0)]],{stroke:'#B8B8B8','stroke-width':.8});
    const ticks=first?[0,1,2,3]:scene===0?[-.3,-.2,-.1,0,.1,.2,.3,.4,.5,.6,.7,.8].filter(tick=>tick>=limits[0]&&tick<=limits[1]):mode==='fold'?[0,.5,1]:mode.startsWith('unit')?[0,.5,1]:[-.25,0,.25,.5,.75];
    for(const tick of ticks) {
      if(height<170 && tick!==0 && tick!==.5 && tick!==1 && tick!==3) continue;
      path(svg,[[left-3,py(tick)],[left,py(tick)]],{stroke:RULE,'stroke-width':.8});
      text(svg,left-7,py(tick)+4,number(tick),{anchor:'end',size:12});
    }
    for(const tick of downstream?[0,.5,1]:[0,1,2,3]) {
      path(svg,[[px(tick),bottom],[px(tick),bottom+3]],{stroke:RULE,'stroke-width':.8});
      text(svg,px(tick),bottom+17,number(tick),{size:12});
    }
    text(svg,left,14,first?'Activation':'Response '+modeLabels[mode],{anchor:'start',size:12});
    text(svg,right,height-1,downstream?'Folded q':'Input x',{anchor:'end',size:12});
    if(first) {
      ['h11','h12','h13'].forEach((name,i)=>{
        path(svg,curveGeometry(name,{threshold}).points.map(p=>[px(p[0]),py(p[1])]),{stroke:COLORS[i],'stroke-width':2.2});
        svg.append(node('circle',{cx:px(i),cy:py(0),r:3,fill:COLORS[i]}));
        text(svg,px(3.12),py(surfaceValue(name,3.12,{threshold}))-7,modeLabels[name],{anchor:'end',size:12,color:COLORS[i]});
        if(showProbe) cross(svg,[px(input),py(surfaceValue(name,input,{threshold}))],3,COLORS[i]);
      });
    } else {
      path(svg,this.geometry.points.map(p=>[px(p[0]),py(p[1])]),{stroke:BLUE,'stroke-width':2.5,'stroke-linejoin':'round'});
      for(const bend of this.geometry.bends) {
        const old=!downstream && FIRST_LAYER_KNOTS.some(k=>near(k,bend.x));
        svg.append(node('circle',{cx:px(bend.x),cy:py(bend.y),r:narrow?3:3.5,fill:old?BLUE:RUST,stroke:'#FFFFFF','stroke-width':.6}));
      }
      if(scene===4) {
        for(const root of foldedPreimages(threshold)) {
          path(svg,[[px(root),py(0)],[px(root),bottom]],{stroke:RUST,'stroke-width':.7,'stroke-dasharray':'2 4'});
          text(svg,px(root),py(0)+18,number(root),{size:12,color:RUST});
        }
      }
      if(showProbe && selected>=domain[0] && selected<=domain[1]) cross(svg,[px(selected),py(value(selected))],narrow?4:5);
    }
    if(showProbe && selected>=domain[0] && selected<=domain[1]) {
      path(svg,[[px(selected),top],[px(selected),bottom]],{stroke:RUST,'stroke-width':.8,'stroke-dasharray':'2 4',opacity:.55});
      text(svg,right,14,(downstream?'q':'x')+' = '+number(selected),{anchor:'end',size:12,color:RUST});
    } else if(showProbe && downstream) {
      text(svg,right,14,'q = '+number(selected)+' (outside [0, 1])',{anchor:'end',size:12,color:RUST});
    }
  }
  drawPattern() {
    const svg=this.features;
    if(svg.parentElement.hidden || svg.parentElement.getBoundingClientRect().width<1) return;
    const {threshold,x:input}=this.options;
    const width=svg.parentElement.getBoundingClientRect().width,height=svg.parentElement.getBoundingClientRect().height;
    const folded=foldScene(this.options.scene),narrow=width<500;
    const left=folded?(narrow?14:30):35,right=width-(folded?(narrow?16:30):14),top=20,bottom=height-19;
    const geometry=foldedCurveGeometry({threshold});
    const ys=geometry.points.map(point=>point[1]);
    const low=Math.min(0,...ys)-.05,high=Math.max(.1,...ys)+.05;
    const px=q=>left+q*(right-left),py=y=>bottom-(y-low)/(high-low)*(bottom-top);
    reset(svg,width,height,'The reused downstream pattern g(q)','Horizontal position is q from zero to one. The three joints of g occur at q=0.2, the adjustable middle threshold, and q=0.8. This same pattern is applied to every folded input interval.');
    text(svg,left,12,'One pattern g(q)',{anchor:'start',size:12,color:BLUE});
    path(svg,[[left,py(0)],[right,py(0)]],{stroke:'#B8B8B8','stroke-width':.8});
    text(svg,left-6,py(0)+3,'0',{anchor:'end',size:11,color:RULE});
    path(svg,geometry.points.map(p=>[px(p[0]),py(p[1])]),{stroke:BLUE,'stroke-width':1.8});
    for(const q of [.2,threshold,.8]) {
      svg.append(node('circle',{cx:px(q),cy:py(foldedValue(q,{threshold})),r:3,fill:RUST}));
      text(svg,px(q),height-3,number(q),{size:12,color:RUST});
    }
    const q=evaluateNetwork(input,{threshold}).fold;
    if(q>=0 && q<=1) cross(svg,[px(q),py(foldedValue(q,{threshold}))],3);
    text(svg,right,12,q>1?'Selected q = '+number(q)+' (outside [0, 1])':'q ∈ [0, 1]',{anchor:'end',size:12});
  }
}

export function renderNetwork(svg,{scene,mode,threshold,x}) {
  const width=Math.max(260,svg.parentElement.getBoundingClientRect().width),narrow=width<500;
  const height=narrow?64:122,hasSecond=scene>=4;
  const value=evaluateNetwork(x,{threshold}),parameters=networkParameters({threshold});
  const second=mode.startsWith('unit')?Number(mode.at(-1))-1:-1;
  const positions=(hasSecond?[.07,.35,.65,.94]:[.1,.5,.92]).map(fraction=>fraction*width);
  const top=narrow?17:25,low=height-(narrow?7:15),middle=(top+low)/2,ys=[top,middle,low];
  reset(svg,width,height,hasSecond?'Two hidden layers, three units per layer':'Three first-layer units and their weighted combination',
    'The selected input is '+number(x)+'. Filled hidden units have positive activations. '+(hasSecond?'Each second-layer unit receives all three first-layer activations; q labels their common weighted input and is not an extra neuron.':'The combination q weights the first-layer features by one, minus two, and two.'));
  const input={x:positions[0],y:middle},h1=ys.map(y=>({x:positions[1],y})),h2=hasSecond?ys.map(y=>({x:positions[2],y})):[],output={x:positions.at(-1),y:middle};
  function edge(a,b,{focus=false,label}={}) {
    path(svg,[[a.x,a.y],[b.x,b.y]],{stroke:focus?RUST:'#B3B3B3','stroke-width':focus?1.5:.8});
    if(label!==undefined && !narrow) text(svg,a.x*.42+b.x*.58,a.y*.42+b.y*.58-4,label,{size:11,color:focus?RUST:INK});
  }
  h1.forEach(p=>edge(input,p,{focus:scene===1}));
  if(hasSecond) {
    h1.forEach((a,i)=>h2.forEach((b,j)=>edge(a,b,{focus:second===j,label:second===j?number(parameters.omega1[j][i]):undefined})));
    h2.forEach((a,i)=>edge(a,output,{focus:mode==='combined'||mode==='downstream',label:mode==='combined'||mode==='downstream'?number(OUTPUT_WEIGHTS[i]):undefined}));
  } else h1.forEach((a,i)=>edge(a,output,{focus:scene>=2,label:scene>=2?number(OUTPUT_WEIGHTS[i]):undefined}));
  function unit(p,label,active,focused=false) {
    svg.append(node('circle',{cx:p.x,cy:p.y,r:narrow?4:6,fill:active?'#D9E5EE':'#FFFFFF',stroke:focused?RUST:BLUE,'stroke-width':focused?1.6:1}));
    text(svg,p.x+(narrow?6:9),p.y+3,label,{anchor:'start',size:narrow?11:12});
  }
  unit(input,'x',false);
  h1.forEach((p,i)=>unit(p,'h₁,'+(i+1),value.h1[i]>0,scene===1));
  h2.forEach((p,i)=>unit(p,'h₂,'+(i+1),value.h2[i]>0,second===i));
  unit(output,hasSecond?'y':'q',false,mode==='combined'||mode==='fold');
  if(!narrow) {
    text(svg,positions[0],11,'Input',{size:11});text(svg,positions[1],11,'Hidden layer 1',{size:11});
    if(hasSecond) text(svg,positions[2],11,'Hidden layer 2',{size:11});
    text(svg,positions.at(-1),11,hasSecond?'Output':'Combination',{size:11});
  }
}
