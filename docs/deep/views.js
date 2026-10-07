import { surfaceMesh } from './geometry.mjs';
import { evaluateNetwork, surfaceValue, INPUT_EXTENT, OUTPUT_WEIGHTS } from './model.mjs';

const NS = 'http://www.w3.org/2000/svg';
const INK = '#242424', BLUE = '#234E70', RUST = '#A54F32', RULE = '#767676';
const key = (a, b) => [Math.min(a, b), Math.max(a, b)].join(':');
const number = value => Math.abs(value) < .005 ? '0' : Number(value.toFixed(2)).toString().replace('-', '−');
export const modeLabels = {
  h11:'h₁,₁', h12:'h₁,₂', h13:'h₁,₃', shallow:'y shallow', preactivation:'Before ReLU',
  unit1:'h₂,₁', unit2:'h₂,₂', unit3:'h₂,₃', combined:'y',
};

function node(tag, attributes = {}, text) {
  const element = document.createElementNS(NS, tag);
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
  if (text !== undefined) element.textContent = text;
  return element;
}
function text(svg, x, y, value, {anchor='middle', size=12, color=INK, ...other}={}) {
  svg.append(node('text', {x, y, 'text-anchor':anchor, 'font-size':size, fill:color, ...other}, value));
}
function path(svg, points, attributes = {}, close = false) {
  const d = points.map((p, index) => `${index ? 'L' : 'M'}${p.map(v=>v.toFixed(3)).join(',')}`).join(' ') + (close ? ' Z' : '');
  const element = node('path', {d, fill:'none', ...attributes});
  svg.append(element);
  return element;
}
function cross(svg, point, radius=4) {
  path(svg, [[point[0]-radius, point[1]-radius],[point[0]+radius, point[1]+radius]], {stroke:RUST,'stroke-width':2.2});
  path(svg, [[point[0]-radius, point[1]+radius],[point[0]+radius, point[1]-radius]], {stroke:RUST,'stroke-width':2.2});
}
function reset(svg, width, height, title, description) {
  svg.replaceChildren(node('title', {}, title), node('desc', {}, description));
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('height', height);
  svg.setAttribute('aria-label', `${title}. ${description}`);
}
function gradient(vertices, indices) {
  const a=vertices[indices[0]];
  for (let index=1;index<indices.length-1;index++) {
    const b=vertices[indices[index]],c=vertices[indices[index+1]];
    const dx=b[0]-a[0],dy=b[1]-a[1],ex=c[0]-a[0],ey=c[1]-a[1],det=dx*ey-dy*ex;
    if (Math.abs(det)>1e-9) return [((b[2]-a[2])*ey-(c[2]-a[2])*dy)/det,(dx*(c[2]-a[2])-ex*(b[2]-a[2]))/det];
  }
  return [0,0];
}
function faceColor(slope) {
  const length=Math.hypot(slope[0],slope[1],1);
  const light=(-.35*slope[0]-.6*slope[1]+.72)/length;
  const amount=.13+.14*(1-Math.max(-1,Math.min(1,light)))/2;
  const blue=[35,78,112];
  return `rgb(${blue.map(channel=>Math.round(255+(channel-255)*amount)).join(' ')})`;
}

export class LinkedFigures {
  constructor(surface, plane, onPoint) {
    this.surface=surface;
    this.plane=plane;
    this.onPoint=onPoint;
    this.options=null;
    this.animation=0;
    this.motion=matchMedia('(prefers-reduced-motion: reduce)');
    this.plane.addEventListener('pointerdown', event=> {
      if(!this.planeScale || !this.options.showProbe) return;
      const bounds=this.plane.getBoundingClientRect();
      const x=(event.clientX-bounds.left)*this.planeScale.width/bounds.width;
      const y=(event.clientY-bounds.top)*this.planeScale.height/bounds.height;
      const {left,top,side}=this.planeScale;
      if(x<left || x>left+side || y<top || y>top+side) return;
      this.onPoint([2*INPUT_EXTENT*(x-left)/side-INPUT_EXTENT, INPUT_EXTENT-2*INPUT_EXTENT*(y-top)/side]);
    });
    new ResizeObserver(()=>this.draw()).observe(surface.closest('.stage-figure'));
  }
  update(options, {animate=false}={}) {
    cancelAnimationFrame(this.animation);
    const previous=this.options;
    this.options={...options,point:[...options.point]};
    this.mesh=surfaceMesh(options.mode,{pinch:options.pinch});
    const target=this.mesh.vertices.map(vertex=>vertex[2]);
    if(!animate || !previous || this.motion.matches || previous.pinch!==options.pinch) {
      this.heights=target;this.draw();return;
    }
    // Every mode shares this exact partition at a fixed set of weights, so the
    // intermediate animation is a piecewise affine interpolation, not a grid.
    const from=this.mesh.vertices.map(vertex=>surfaceValue(previous.mode,vertex[0],vertex[1],{pinch:options.pinch}));
    const start=performance.now();
    const tick=now=> {
      const fraction=Math.min(1,(now-start)/480),ease=fraction*fraction*(3-2*fraction);
      this.heights=target.map((value,index)=>from[index]+ease*(value-from[index]));
      this.draw();
      if(fraction<1) this.animation=requestAnimationFrame(tick);
    };
    this.animation=requestAnimationFrame(tick);
  }
  draw() {
    if(!this.options || !this.heights) return;
    this.drawSurface();this.drawPlane();
  }
  drawSurface() {
    const {mode,pinch,point,showProbe,cover}=this.options;
    const svg=this.surface;
    const width=Math.max(110,svg.parentElement.getBoundingClientRect().width);
    const available=svg.closest('.stage-figure').getBoundingClientRect().height;
    const height=Math.max(105,Math.min(cover?540:480,available || 350));
    const narrow=width<280;
    const left=narrow?25:44,right=narrow?9:22,top=22,bottom=narrow?24:38;
    const vertical=cover?[-1.25,2.0]:mode==='h13'?[-3.85,3.25]:[-1.5,3.25];
    const scale=Math.min((width-left-right)/4.75,(height-top-bottom)/(vertical[1]-vertical[0]));
    const origin=[left+(width-left-right)/2,top-scale*vertical[0]];
    const project=vertex=>[origin[0]+scale*.72*(vertex[0]-vertex[1]),origin[1]+scale*(.55*(vertex[0]+vertex[1])-.6*(vertex[2]||0))];
    reset(svg,width,height,`Response surface: ${modeLabels[mode]}`,`Inputs x one and x two each range from minus ${INPUT_EXTENT} to ${INPUT_EXTENT}. Solid lines show changes in slope. Dashed lines mark the zero contour. The input domain remains fixed through the explanation.`);
    const vertices=this.mesh.vertices.map((v,index)=>[v[0],v[1],this.heights[index]]);
    const d=INPUT_EXTENT;
    const square=[[-d,-d,0],[d,-d,0],[d,d,0],[-d,d,0]];
    path(svg,square.map(project),{stroke:'#B8B8B8','stroke-width':.8},true);
    const edgeFaces=new Map();
    const slopes=this.mesh.faces.map(face=>gradient(vertices,face.indices));
    this.mesh.faces.forEach((face,faceIndex)=>face.indices.forEach((a,index)=>{
      const b=face.indices[(index+1)%face.indices.length],k=key(a,b);
      if(!edgeFaces.has(k)) edgeFaces.set(k,[]);
      edgeFaces.get(k).push(faceIndex);
    }));
    const trueEdges=new Set();
    for(const [k,faces] of edgeFaces) {
      if(faces.length===1) trueEdges.add(k);
      else if(Math.hypot(slopes[faces[0]][0]-slopes[faces[1]][0],slopes[faces[0]][1]-slopes[faces[1]][1])>1e-7) trueEdges.add(k);
    }
    const ordered=this.mesh.faces.map((face,index)=>({face,index,depth:face.indices.reduce((sum,id)=>sum+vertices[id][0]+vertices[id][1],0)/face.indices.length})).sort((a,b)=>a.depth-b.depth);
    for(const {face,index} of ordered) {
      if(!face.indices.some(id=>Math.abs(vertices[id][2])>1e-8)) continue;
      path(svg,face.indices.map(id=>project(vertices[id])),{fill:faceColor(slopes[index])},true);
      face.indices.forEach((a,i)=>{
        const b=face.indices[(i+1)%face.indices.length];
        if(trueEdges.has(key(a,b))) path(svg,[project(vertices[a]),project(vertices[b])],{stroke:BLUE,'stroke-width':1.1});
      });
    }
    for(const [a,b] of this.mesh.zeroContours) path(svg,[project([vertices[a][0],vertices[a][1],0]),project([vertices[b][0],vertices[b][1],0])],{stroke:RUST,'stroke-width':1.5,'stroke-dasharray':'4 3'});
    // Axes follow the edges of the z=0 input square; height ticks remain on a
    // separate response axis rather than being mistaken for a third input.
    const axisPoint=[-d-.12,d,0];
    const high=mode==='h13'?3.2:1.7;
    const zTicks=cover?[0,1]:narrow?[0,1]:mode==='h13'?[-2,0,1,3]:[-2,0,1];
    path(svg,[project([...axisPoint.slice(0,2),cover?0:-2.2]),project([...axisPoint.slice(0,2),high])],{stroke:RULE,'stroke-width':.8});
    for(const z of zTicks) {
      const p=project([...axisPoint.slice(0,2),z]);
      path(svg,[[p[0]-2,p[1]],[p[0]+2,p[1]]],{stroke:RULE,'stroke-width':.8});
      text(svg,p[0]-5,p[1]+4,number(z),{anchor:'end',size:narrow?11:12});
    }
    const label=project([...axisPoint.slice(0,2),high]);
    text(svg,label[0],Math.max(13,label[1]-10),'Response',{anchor:'start',size:narrow?11:12});
    const ticks=narrow?[-1,1]:[-1,0,1];
    for(const value of ticks) {
      const px=project([value,d,0]),py=project([d,value,0]);
      text(svg,px[0]-2,px[1]+14,number(value),{size:narrow?11:12});
      text(svg,py[0]+2,py[1]+14,number(value),{size:narrow?11:12});
    }
    const px=project([0,d,0]),py=project([d,0,0]);
    text(svg,px[0],px[1]+(narrow?28:33),'x₁',{size:narrow?11:13});
    text(svg,py[0],py[1]+(narrow?28:33),'x₂',{size:narrow?11:13});
    if(showProbe) {
      const z=surfaceValue(mode,...point,{pinch});
      const base=project([...point,0]),tip=project([...point,z]);
      if(Math.abs(z)>.01) path(svg,[base,tip],{stroke:RUST,'stroke-width':1,'stroke-dasharray':'2 3'});
      cross(svg,tip,narrow?3:4);
    }
  }
  drawPlane() {
    const {mode,point,showProbe}=this.options;
    const svg=this.plane;
    if(svg.parentElement.getBoundingClientRect().width<1) return;
    const width=svg.parentElement.getBoundingClientRect().width;
    const available=svg.closest('.stage-figure').getBoundingClientRect().height;
    const height=Math.max(100,Math.min(width+30,available||240));
    const narrow=width<170;
    const left=narrow?22:30,right=8,top=22,bottom=32;
    const side=Math.max(40,Math.min(width-left-right,height-top-bottom));
    const x=value=>left+side*(value+INPUT_EXTENT)/(2*INPUT_EXTENT);
    const y=value=>top+side*(INPUT_EXTENT-value)/(2*INPUT_EXTENT);
    this.planeScale={width,height,left,top,side};
    const contourLabel=mode==='combined'?'boundary of the positive response':mode==='shallow'||mode==='preactivation'?'zero crossing':'activation boundary';
    reset(svg,width,height,'Input plane',`Solid lines are crease locations in the original input plane. Dashed lines mark the ${contourLabel}. Select a point to inspect its response; keyboard input controls are available in the final frame.`);
    svg.append(node('rect',{x:left,y:top,width:side,height:side,fill:'#FFFFFF',stroke:RULE,'stroke-width':.7}));
    for(const face of this.mesh.faces) {
      const mean=face.indices.reduce((sum,id)=>sum+this.mesh.vertices[id][2],0)/face.indices.length;
      if(mean>.00001) path(svg,face.indices.map(id=>[x(this.mesh.vertices[id][0]),y(this.mesh.vertices[id][1])]),{fill:'#ECF1F5'},true);
    }
    for(const [a,b] of this.mesh.creases) path(svg,[a,b].map(id=>[x(this.mesh.vertices[id][0]),y(this.mesh.vertices[id][1])]),{stroke:BLUE,'stroke-width':.85});
    for(const [a,b] of this.mesh.zeroContours) path(svg,[a,b].map(id=>[x(this.mesh.vertices[id][0]),y(this.mesh.vertices[id][1])]),{stroke:RUST,'stroke-width':1.3,'stroke-dasharray':'3 2'});
    const ticks=narrow?[-1,1]:[-1,0,1];
    for(const value of ticks) {
      text(svg,x(value),top+side+14,number(value),{size:11});
      text(svg,left-5,y(value)+4,number(value),{anchor:'end',size:11});
    }
    text(svg,left+side/2,13,'Input plane',{size:narrow?11:12});
    text(svg,left+side/2,top+side+29,'x₁',{size:12});
    text(svg,8,top+side/2,'x₂',{size:12,transform:`rotate(-90 8 ${top+side/2})`});
    if(showProbe) cross(svg,[x(point[0]),y(point[1])],narrow?3:4);
  }
}

export function renderNetwork(svg,{scene,mode,pinch,point}) {
  const width=Math.max(260,svg.parentElement.getBoundingClientRect().width),narrow=width<500;
  const height=narrow?64:122;
  const hasSecond=scene>=3 && !(scene===7 && mode==='shallow');
  const value=evaluateNetwork(...point,{pinch});
  const first=mode.startsWith('h1')?Number(mode.at(-1))-1:-1;
  const second=mode.startsWith('unit')?Number(mode.at(-1))-1:mode==='preactivation'?0:-1;
  const levels=hasSecond?[.07,.35,.65,.94]:[.1,.5,.92];
  const positions=levels.map(fraction=>fraction*width);
  const top=narrow?17:25,low=height-(narrow?7:15),middle=(top+low)/2;
  const ys=[top,middle,low];
  reset(svg,width,height,hasSecond?'Two hidden layers, three units per layer':'One hidden layer, three units',`The selected input is (${number(point[0])}, ${number(point[1])}). Filled hidden units have positive activations. The highlighted paths identify the unit or output shown above.`);
  const inputs=[{x:positions[0],y:top+5},{x:positions[0],y:low-5}];
  const h1=ys.map(y=>({x:positions[1],y}));
  const h2=hasSecond?ys.map(y=>({x:positions[2],y})):[];
  const output={x:positions.at(-1),y:middle};
  function edge(a,b,{focus=false,label}={}) {
    path(svg,[[a.x,a.y],[b.x,b.y]],{stroke:focus?RUST:'#C5C5C5','stroke-width':focus?1.5:.8});
    if(label!==undefined && !narrow) text(svg,a.x*.42+b.x*.58,a.y*.42+b.y*.58-4,label,{size:11});
  }
  edge(inputs[0],h1[0],{focus:first===0});edge(inputs[1],h1[1],{focus:first===1});
  edge(inputs[0],h1[2],{focus:first===2});edge(inputs[1],h1[2],{focus:first===2});
  if(hasSecond) {
    const weights=[[-1,-1,-1],[-pinch,-1,-1],[-1,-2,-1]];
    h1.forEach((a,i)=>h2.forEach((b,j)=>edge(a,b,{focus:second===j,label:second===j?number(weights[j][i]):undefined})));
    h2.forEach((a,i)=>edge(a,output,{focus:mode==='combined',label:mode==='combined'?number(OUTPUT_WEIGHTS[i]):undefined}));
  } else h1.forEach((a,i)=>edge(a,output,{focus:scene===2 || mode==='shallow',label:scene===2 || mode==='shallow'?'−1':undefined}));
  function unit(p,label,active,focused=false) {
    svg.append(node('circle',{cx:p.x,cy:p.y,r:narrow?4:6,fill:active?'#D9E5EE':'#FFFFFF',stroke:focused?RUST:BLUE,'stroke-width':focused?1.6:1}));
    text(svg,p.x+(narrow?6:9),p.y+3,label,{anchor:'start',size:narrow?11:12});
  }
  inputs.forEach((p,i)=>unit(p,`x${i?'₂':'₁'}`,false));
  h1.forEach((p,i)=>unit(p,`h₁,${i+1}`,value.h1[i]>0,first===i));
  h2.forEach((p,i)=>unit(p,`h₂,${i+1}`,scene===3?false:value.h2[i]>0,second===i));
  unit(output,'y',false,mode==='combined' || mode==='shallow');
  if(!narrow) {
    text(svg,positions[0],11,'Inputs',{size:11});text(svg,positions[1],11,'Hidden layer 1',{size:11});
    if(hasSecond) text(svg,positions[2],11,'Hidden layer 2',{size:11});
    text(svg,positions.at(-1),11,'Output',{size:11});
  }
}
