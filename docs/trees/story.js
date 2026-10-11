(function (root, factory) {
  'use strict';
  const api = factory(root.TreePartitions || (typeof require === 'function' ? require('./model.js') : null));
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root.document) api.init(root.document, root);
})(typeof window !== 'undefined' ? window : globalThis, function (M) {
  'use strict';
  const E = M.examples;
  const C = { ink: '#242424', muted: '#606060', rule: '#b9b9b9', blue: '#234E70', rust: '#A54F32', teal: '#007F73' };
  const topics = ['A split is a rule', 'Refine one leaf', 'Add functions', 'Split the input plane', 'A split stays inside its parent', 'Overlay the partitions', 'Where interactions enter', 'From inputs to predictions'];
  const captions = [
    'The highlighted leaf and horizontal segment give the same prediction.',
    'The second rule applies only after x > −0.5. The left leaf stays intact.',
    'At each x, add the heights. At full strength the sum is 0, 1, then 3.',
    'Every point in a leaf’s rectangle receives the same prediction.',
    'The horizontal cut stops at x₁ = 0: only the right-hand region is split.',
    'Each outlined cell is the intersection of a leaf from each tree. Read its sum.',
    'Compare the x₁ jump below and above x₂ = 0. Does that jump depend on x₂?',
    'The input plane locates observations; the vector records their predictions.'
  ];
  const fmt = n => Math.abs(n) < 1e-9 ? '0' : String(Number(n.toFixed(2))).replace('-', '−');
  const escape = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const text = (x, y, value, options = '') => `<text x="${x}" y="${y}" ${options}>${escape(value)}</text>`;
  const line = (x1, y1, x2, y2, color = C.rule, width = 1.2, extra = '') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}" ${extra}/>`;
  const rect = (x, y, w, h, fill, extra = '') => `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${fill}" ${extra}/>`;
  const circle = (x, y, r, fill, extra = '') => `<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" ${extra}/>`;
  const leaf = n => Object.prototype.hasOwnProperty.call(n, 'value');
  const scaled = (tree, weight) => leaf(tree) ? {...tree, value: tree.value * weight} : {...tree, left: scaled(tree.left, weight), right: scaled(tree.right, weight)};
  const color = v => {
    const a = Math.max(0, Math.min(1, v / 5));
    return `rgb(${[255,255,255].map((start,i) => Math.round(start + ([35,78,112][i] - start)*a)).join(',')})`;
  };

  function treeDiagram(tree, point, box, oneD = false, accent = C.blue) {
    const { x, y, w, h } = box;
    const leaves = [];
    function collect(node) { if (leaf(node)) leaves.push(node); else { collect(node.left); collect(node.right); } }
    collect(tree);
    const maxDepth = node => leaf(node) ? 0 : 1 + Math.max(maxDepth(node.left), maxDepth(node.right));
    const depth = maxDepth(tree);
    const positions = new Map();
    const step = w / leaves.length;
    let ordinal = 0;
    function place(node, level) {
      const px = leaf(node) ? x + step * (ordinal++ + .5) : (place(node.left, level + 1) + place(node.right, level + 1))/2;
      positions.set(node.id, [px, y + level * h / Math.max(1, depth)]);
      return px;
    }
    place(tree, 0);
    const path = new Set(M.pathNodes(tree, point).map(n => n.id));
    const boxes = M.regions(tree, oneD ? [[-2,2]] : [[-2,2],[-2,2]]);
    let out = '';
    function draw(node) {
      const [px,py] = positions.get(node.id);
      if (!leaf(node)) {
        for (const [child, branch] of [[node.left,'yes'],[node.right,'no']]) {
          const [cx,cy] = positions.get(child.id);
          const active = path.has(child.id);
          out += line(px,py + 11,cx,cy - 17,active ? accent : C.rule,active ? 2.6 : 1.2);
          out += text((px+cx)/2 + (cx<px?-10:10),(py+cy)/2,branch,`text-anchor="middle" class="branch-label" fill="${active?accent:C.muted}"`);
        }
        const label = `${oneD?'x':node.feature === 0?'x₁':'x₂'} ≤ ${fmt(node.threshold)}?`;
        out += rect(px-49,py-15,98,28,'white');
        out += text(px,py+5,label,`text-anchor="middle" fill="${path.has(node.id)?accent:C.ink}"`);
        draw(node.left); draw(node.right);
      } else {
        const b = boxes.find(b => b.leafId === node.id);
        const center = b.bounds.map(([lo,hi]) => Math.round((lo+hi)/2*100)/100);
        if (oneD) center.push(point[1] || 0);
        const region = b.bounds.map(([lo,hi],i)=>`${oneD?'x':i===0?'x₁':'x₂'} between ${fmt(lo)} and ${fmt(hi)}`).join(', ');
        out += `<g class="leaf-button" role="button" tabindex="0" data-probe="${center.join(',')}" aria-label="Inspect leaf predicting ${node.value}; interior region ${region}">`;
        out += rect(px-22,py-16,44,32,path.has(node.id)?accent:'white',`rx="2" stroke="${path.has(node.id)?accent:C.rule}" stroke-width="1.5"`);
        out += text(px,py+6,fmt(node.value),`text-anchor="middle" fill="${path.has(node.id)?'white':C.ink}"`);
        out += '</g>';
      }
    }
    draw(tree);
    return out;
  }

  function stepPlot(trees, point, box, options = {}) {
    const {x,y,w,h} = box;
    const segments = M.sumSegments1D(trees);
    const X = value => x + (value+2)/4*w;
    const Y = value => y+h - value/3.5*h;
    const accent = options.accent || C.blue;
    let out = '';
    if (options.label) out += text(x-32,y+h/2+5,options.label,'text-anchor="end" class="plot-heading"');
    for (const v of [0,1,2,3]) {
      out += line(x,Y(v),x+w,Y(v),v===0?C.muted:'#ededed',v===0?1.2:1);
      out += text(x-9,Y(v)+5,fmt(v),'text-anchor="end" class="tick"');
    }
    const selected = trees.map(t => M.leafAt(t,point).id).join('|');
    for (const [i,s] of segments.entries()) {
      if (i) out += line(X(s.lo),y,X(s.lo),y+h,C.rule,1,'stroke-dasharray="3 4"');
      const active = selected === s.leafId;
      out += line(X(s.lo),Y(s.value),X(s.hi),Y(s.value),accent,active?4.5:2.5);
      if (i > 0 && segments[i-1].value !== s.value) out += circle(X(s.lo),Y(s.value),3.6,'white',`stroke="${accent}" stroke-width="1.5"`);
      out += circle(X(s.hi),Y(s.value),3,accent);
    }
    for (const v of [-2,-.5,.75,2]) {
      const isCut = v === -.5 || v === .75;
      if (isCut && !segments.some(s => s.lo === v)) continue;
      out += line(X(v),y+h,X(v),y+h+5,C.muted,1);
      out += text(X(v),y+h+21,fmt(v),'text-anchor="middle" class="tick"');
    }
    const value = trees.reduce((sum,t) => sum + M.evaluate(t,point),0);
    out += line(X(point[0]),y-4,X(point[0]),y+h,C.muted,1.3,'stroke-dasharray="4 4"');
    out += circle(X(point[0]),Y(value),5,accent,'stroke="white" stroke-width="1.5"');
    out += text(x+w+9,y+h+5,'x','class="axis-label"');
    out += rect(x,y,w,h,'transparent',`data-plot="1" data-box="${x},${y},${w},${h}" class="plot-hit"`);
    return out;
  }

  function cuts(tree, bounds = [[-2,2],[-2,2]], depth = 0) {
    if (leaf(tree)) return [];
    const [lo,hi] = bounds[tree.feature];
    const result = [];
    if (tree.threshold > lo && tree.threshold < hi) result.push({feature:tree.feature,threshold:tree.threshold,bounds,depth});
    const l = bounds.map(a=>a.slice()), r = bounds.map(a=>a.slice());
    l[tree.feature][1] = Math.min(hi,tree.threshold);
    r[tree.feature][0] = Math.max(lo,tree.threshold);
    if (l[tree.feature][0] < l[tree.feature][1]) result.push(...cuts(tree.left,l,depth+1));
    if (r[tree.feature][0] < r[tree.feature][1]) result.push(...cuts(tree.right,r,depth+1));
    return result;
  }

  function partitionMap(trees, point, box, options = {}) {
    const {x,y,w,h} = box;
    const X = v => x+(v+2)/4*w, Y = v => y+h-(v+2)/4*h;
    const cells = M.refinement(trees);
    const selected = trees.map(t=>M.leafAt(t,point).id).join('|');
    let out = '';
    if (options.label) out += text(x+w/2,y-22,options.label,'text-anchor="middle" class="plot-heading"');
    for (const cell of cells) {
      const [[x0,x1],[y0,y1]] = cell.bounds;
      out += rect(X(x0),Y(y1),X(x1)-X(x0),Y(y0)-Y(y1),color(cell.value));
      let cx=(X(x0)+X(x1))/2, cy=(Y(y0)+Y(y1))/2;
      if(!options.noProbe && Math.hypot(cx-X(point[0]),cy-Y(point[1]))<23) {
        const candidates=[.28,.72].flatMap(u=>[.28,.72].map(v=>[X(x0)+(X(x1)-X(x0))*u,Y(y1)+(Y(y0)-Y(y1))*v]));
        [cx,cy]=candidates.sort((a,b)=>Math.hypot(b[0]-X(point[0]),b[1]-Y(point[1]))-Math.hypot(a[0]-X(point[0]),a[1]-Y(point[1])))[0];
      }
      out += text(cx,cy+6,fmt(cell.value),`text-anchor="middle" class="cell-value" fill="${cell.value>=3.5?'white':C.ink}"`);
    }
    for (const [i,tree] of trees.entries()) {
      for (const cut of cuts(tree)) {
        const attrs = `class="partition-cut cut-${cut.depth}" ${i===1?'stroke-dasharray="6 4"':''}`;
        out += cut.feature === 0
          ? line(X(cut.threshold),Y(cut.bounds[1][0]),X(cut.threshold),Y(cut.bounds[1][1]),i===1?C.rust:C.blue,2,attrs)
          : line(X(cut.bounds[0][0]),Y(cut.threshold),X(cut.bounds[0][1]),Y(cut.threshold),i===1?C.rust:C.blue,2,attrs);
      }
    }
    out += rect(x,y,w,h,'none',`stroke="${C.muted}" stroke-width="1"`);
    if (!options.noProbe) {
      const active = cells.find(cell=>cell.leafId===selected);
      if (active) {
        const [[x0,x1],[y0,y1]] = active.bounds;
        out += rect(X(x0)+3,Y(y1)+3,X(x1)-X(x0)-6,Y(y0)-Y(y1)-6,'none',`stroke="${C.ink}" stroke-width="2" stroke-dasharray="3 3"`);
      }
      out += circle(X(point[0]),Y(point[1]),5,'white',`stroke="${C.ink}" stroke-width="2"`);
      out += line(X(point[0])-8,Y(point[1]),X(point[0])+8,Y(point[1]),C.ink,1);
      out += line(X(point[0]),Y(point[1])-8,X(point[0]),Y(point[1])+8,C.ink,1);
    }
    for (const v of [-2,0,2]) {
      out += line(X(v),y+h,X(v),y+h+4,C.muted,1);
      out += text(X(v),y+h+19,fmt(v),'text-anchor="middle" class="tick"');
      if (!options.minimalTicks) out += text(x-8,Y(v)+4,fmt(v),'text-anchor="end" class="tick"');
    }
    out += w<120 ? text(x+w-2,y+h+37,'x₁','text-anchor="end" class="axis-label"') : text(x+w+7,y+h+5,'x₁','class="axis-label"');
    out += text(x-3,y-8,'x₂','class="axis-label"');
    if (!options.noProbe) out += rect(x,y,w,h,'transparent',`data-plot="2" data-box="${x},${y},${w},${h}" class="plot-hit"`);
    return out;
  }

  function legend(x,y,w) {
    let out = text(x,y-12,'Prediction · shared scale','class="legend-label"');
    for(let v=0;v<=5;v++) {
      out += rect(x+v*w/6,y,w/6,10,color(v),'stroke="#aaa" stroke-width=".5"');
      out += text(x+(v+.5)*w/6,y+27,v,'text-anchor="middle" class="tick"');
    }
    return out;
  }

  function readout(scene,state) {
    const p = [state.x,state.y], a = state.addition;
    if (scene === 0 || scene === 1) return `x = ${fmt(p[0])}  →  prediction = ${fmt(M.evaluate(scene===0?E.stump1D:E.grown1D,p))}`;
    if (scene === 2) {
      const u=M.evaluate(E.stump1D,p),v=M.evaluate(E.second1D,p);
      return `${fmt(u)} + ${fmt(a)} × ${fmt(v)} = ${fmt(u+a*v)}`;
    }
    if (scene === 3 || scene === 4) return `(x₁, x₂) = (${fmt(p[0])}, ${fmt(p[1])})  →  ${scene===3?'T₁':'U'}(x) = ${fmt(M.evaluate(scene===3?E.stumpX:E.conditional,p))}`;
    if (scene === 5) {
      const u=M.evaluate(E.stumpX,p),v=M.evaluate(E.stumpY,p);
      return `T₁ + λT₂ = ${fmt(u)} + ${fmt(a)} × ${fmt(v)} = ${fmt(u+a*v)}`;
    }
    if (scene === 6) {
      const values = [E.stumpX,E.stumpY].map(t=>M.evaluate(t,p));
      if(state.interaction) values.push(M.evaluate(E.conditional,p));
      return `${values.map(fmt).join(' + ')} = ${fmt(values.reduce((a,b)=>a+b,0))} · x₁ jump: ${state.interaction?'1 below, 3 above':'1 below, 1 above'}`;
    }
    return 'Three fixed observations → three prediction coordinates.';
  }

  function renderScene(scene, provided = {}, compact = false) {
    const state = {x:.8,y:.8,addition:1,interaction:false,...provided};
    const p = [state.x,state.y], W=compact?420:780, H=compact?280:510;
    let out = '';
    if (scene === 0 || scene === 1) {
      const tree = scene===0?E.stump1D:E.grown1D;
      out += text(compact?90:165,compact?22:55,'Decision tree','text-anchor="middle" class="plot-heading"');
      out += text(compact?310:572,compact?22:55,'Prediction function','text-anchor="middle" class="plot-heading"');
      out += treeDiagram(tree,p,compact?{x:8,y:72,w:168,h:120}:{x:20,y:140,w:290,h:200},true);
      out += stepPlot([tree],p,compact?{x:216,y:66,w:175,h:160}:{x:397,y:116,w:326,h:255});
      out += text(compact?210:390,compact?266:440,'Follow a branch; read the corresponding height.','text-anchor="middle" class="figure-note"');
    } else if (scene === 2) {
      const second = scaled(E.second1D,state.addition);
      const labels = ['T₁','λT₂','F'];
      for(const [i,trees] of [[E.stump1D],[second],[E.stump1D,second]].entries()) {
        out += stepPlot(trees,p,compact?{x:73,y:10+i*83,w:310,h:49}:{x:124,y:25+i*150,w:585,h:105},{label:labels[i],accent:[C.blue,C.rust,C.teal][i]});
      }
      out += text(W/2,H-4,'Same x, same vertical scale. F = T₁ + λT₂.','text-anchor="middle" class="figure-note"');
    } else if (scene === 3 || scene === 4) {
      const tree = scene===3?E.stumpX:E.conditional;
      out += text(compact?85:165,compact?22:52,scene===3?'One rule':'Two nested rules','text-anchor="middle" class="plot-heading"');
      out += treeDiagram(tree,p,compact?{x:2,y:64,w:169,h:113}:{x:20,y:142,w:290,h:200});
      out += partitionMap([tree],p,compact?{x:222,y:43,w:155,h:155}:{x:398,y:91,w:303,h:303},{label:'Input space'});
      out += legend(compact?217:443,compact?251:455,compact?172:220);
    } else if (scene === 5) {
      const second=scaled(E.stumpY,state.addition);
      const maps=[[E.stumpX],[second],[E.stumpX,second]];
      const labels=['T₁','λT₂','F = T₁ + λT₂'];
      for(let i=0;i<3;i++) out += partitionMap(maps[i],p,compact?{x:26+i*138,y:57,w:96,h:96}:{x:38+i*254,y:111,w:196,h:196},{label:labels[i],minimalTicks:i>0});
      out += text(compact?136:261,compact?107:214,'+','text-anchor="middle" class="operator"');
      out += text(compact?277:515,compact?107:214,'=','text-anchor="middle" class="operator"');
      out += text(W/2,compact?211:369,'One location. Two leaf values. Their sum.','text-anchor="middle" class="figure-note"');
      out += legend(compact?110:250,compact?247:432,compact?200:280);
    } else if (scene === 6) {
      const trees=[E.stumpX,E.stumpY];
      if(state.interaction) trees.push(E.conditional);
      out += partitionMap([E.conditional],p,compact?{x:28,y:42,w:137,h:137}:{x:73,y:92,w:253,h:253},{label:'Conditional tree U'});
      out += partitionMap(trees,p,compact?{x:247,y:42,w:137,h:137}:{x:451,y:92,w:253,h:253},{label:state.interaction?'F = T₁ + T₂ + U':'F = T₁ + T₂'});
      out += text(compact?211:390,compact?103:220,state.interaction?'+ U':'U off','text-anchor="middle" class="figure-note"');
      out += text(W/2,compact?215:393,`x₁ jump: 1 below; ${state.interaction?'3':'1'} above.`, 'text-anchor="middle" class="figure-note"');
      out += legend(compact?115:250,compact?248:446,compact?190:280);
    } else {
      const box=compact?{x:29,y:47,w:162,h:162}:{x:92,y:108,w:275,h:275};
      out += partitionMap([E.stumpX,E.stumpY],p,box,{label:'Input space',noProbe:true});
      const points=[[-1.25,-1.3],[1.2,-1.35],[1.2,1.35]], vals=points.map(p=>M.evaluate(E.stumpX,p)+M.evaluate(E.stumpY,p));
      points.forEach((p,i)=>{
        const px=box.x+(p[0]+2)/4*box.w,py=box.y+box.h-(p[1]+2)/4*box.h;
        out += circle(px,py,11,'white',`stroke="${C.ink}" stroke-width="1.5"`);
        out += text(px,py+4,String.fromCharCode(65+i),'text-anchor="middle" class="tick"');
      });
      out += line(compact?219:403,compact?125:242,compact?245:448,compact?125:242,C.muted,1.5);
      out += text(compact?244:448,compact?131:248,'›','text-anchor="end" class="operator"');
      out += text(compact?328:598,compact?23:76,'Prediction vector','text-anchor="middle" class="plot-heading"');
      vals.forEach((v,i)=>{
        out += text(compact?278:508,compact?81+i*53:161+i*76,`F(${String.fromCharCode(65+i)})`, 'class="axis-label"');
        out += text(compact?363:665,compact?81+i*53:161+i*76,fmt(v),'text-anchor="middle" class="vector-value"');
      });
      out += text(W/2,compact?259:459,'Next: choose a tree by the correction it makes.','text-anchor="middle" class="figure-note"');
    }
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" fill="${C.ink}" role="group" aria-labelledby="scene-title scene-description"><title id="scene-title">${escape(topics[scene])}</title><desc id="scene-description">${escape(captions[scene]+' '+readout(scene,state))}</desc><style>text{font-family:Georgia,serif;font-size:${compact?14:18}px}.tick,.branch-label{font-family:Arial,sans-serif;font-size:${compact?12:13}px}.plot-heading{font-size:${compact?15:20}px}.figure-note,.legend-label{font-size:${compact?13:16}px}.legend-label{font-family:Arial,sans-serif}.cell-value{font-size:${compact?18:27}px}.operator{font-size:${compact?20:28}px}.vector-value{font-size:${compact?25:35}px}.axis-label{font-style:italic}.leaf-button,.plot-hit{cursor:crosshair}.leaf-button:focus{outline:none}.leaf-button:focus rect{stroke:#242424;stroke-width:4}.partition-cut{vector-effect:non-scaling-stroke}</style>${out}</svg>`;
  }

  function init(doc,win) {
    const get=id=>doc.getElementById(id);
    const state={x:.8,y:.8,addition:1,interaction:false};
    const steps=Array.from(doc.querySelectorAll('.step'));
    const diagram=get('diagram'), shell=diagram.closest('.stage-shell');
    const reduced=win.matchMedia('(prefers-reduced-motion: reduce)');
    const compact=win.matchMedia('(max-width: 800px)');
    let scene=0, frame=0, animation=0, targetScene=null, targetTimeout=0;
    function render(transition=false) {
      shell.dataset.scene=String(scene);
      get('stage-topic').textContent=topics[scene];
      get('stage-count').textContent=`${String(scene+1).padStart(2,'0')} / 08`;
      get('stage-caption').textContent=captions[scene];
      get('probe-controls').hidden=scene===7;
      get('probe-y-control').hidden=scene<3;
      get('addition-controls').hidden=scene!==2 && scene!==5;
      get('interaction-controls').hidden=scene!==6;
      get('probe-x-value').textContent=fmt(state.x);
      get('probe-y-value').textContent=fmt(state.y);
      get('addition-value').textContent=fmt(state.addition);
      get('probe-x').value=state.x;
      get('probe-y').value=state.y;
      get('addition').value=state.addition;
      get('calculation').textContent=readout(scene,state);
      get('previous').disabled=scene===0;
      get('next').disabled=scene===7;
      steps.forEach((s,i)=>s.classList.toggle('active',i===scene));
      diagram.innerHTML=renderScene(scene,state,compact.matches);
      if(transition && !reduced.matches && diagram.animate) diagram.animate([{opacity:.25},{opacity:1}],{duration:280,easing:'ease-out'});
    }
    function stopAnimation(){win.cancelAnimationFrame(animation);animation=0;get('replay-addition').textContent='Replay addition';}
    function activate(index){
      if(index===scene) return;
      stopAnimation(); scene=index; state.addition=1; render(true);
    }
    function readingLine(){
      return compact.matches && win.innerHeight>560 ? shell.getBoundingClientRect().bottom+35 : win.innerHeight*.46;
    }
    function onScroll(){
      if(frame) return;
      frame=win.requestAnimationFrame(()=>{
        frame=0;
        get('progress').style.width=`${100*Math.min(1,Math.max(0,win.scrollY/(doc.documentElement.scrollHeight-win.innerHeight||1)))}%`;
        if(targetScene!==null) return;
        const line=readingLine();
        let nearest=0;
        steps.forEach((s,i)=>{if(s.getBoundingClientRect().top<=line) nearest=i;});
        activate(nearest);
      });
    }
    function go(index){
      index=Math.max(0,Math.min(steps.length-1,index));
      targetScene=index;win.clearTimeout(targetTimeout);activate(index);
      const offset=compact.matches && win.innerHeight>560?shell.getBoundingClientRect().height+86:win.innerHeight*.18;
      const top=win.scrollY+steps[index].getBoundingClientRect().top-offset;
      win.scrollTo({top,behavior:reduced.matches?'instant':'smooth'});
      // Prevent an in-between section from replacing a button's target while scrolling.
      targetTimeout=win.setTimeout(()=>{targetScene=null;onScroll();},reduced.matches?40:900);
    }
    get('previous').addEventListener('click',()=>go(scene-1));
    get('next').addEventListener('click',()=>go(scene+1));
    for(const [id,key] of [['probe-x','x'],['probe-y','y'],['addition','addition']]) get(id).addEventListener('input',e=>{if(key==='addition')stopAnimation();state[key]=Number(e.target.value);render();});
    get('include-interaction').addEventListener('change',e=>{state.interaction=e.target.checked;render(true);});
    get('replay-addition').addEventListener('click',()=>{
      if(animation){stopAnimation();return;}
      if(reduced.matches){state.addition=1;render();return;}
      const start=win.performance.now();state.addition=0;get('replay-addition').textContent='Pause';
      function tick(now){state.addition=Math.min(1,Math.round((now-start)/1800*100)/100);render();if(state.addition<1)animation=win.requestAnimationFrame(tick);else stopAnimation();}
      animation=win.requestAnimationFrame(tick);
    });
    function probe(event){
      const plot=event.target.closest('[data-plot]');
      if(!plot || (event.type==='pointermove' && event.pointerType!=='mouse'))return;
      const svg=diagram.querySelector('svg');
      const point=svg.createSVGPoint();point.x=event.clientX;point.y=event.clientY;
      const local=point.matrixTransform(svg.getScreenCTM().inverse());
      const [x,y,w,h]=plot.dataset.box.split(',').map(Number);
      state.x=Math.max(-2,Math.min(2,Math.round(((local.x-x)/w*4-2)*100)/100));
      if(plot.dataset.plot==='2')state.y=Math.max(-2,Math.min(2,Math.round((2-(local.y-y)/h*4)*100)/100));
      render();
    }
    diagram.addEventListener('pointermove',probe);
    diagram.addEventListener('pointerdown',probe);
    function chooseLeaf(event){
      const el=event.target.closest('[data-probe]');
      if(!el)return;
      if(event.type==='keydown' && event.key!=='Enter' && event.key!==' ')return;
      event.preventDefault();
      const values=el.dataset.probe.split(',').map(Number), key=el.dataset.probe;
      [state.x,state.y]=values;render();
      if(event.type==='keydown')diagram.querySelector(`[data-probe="${key}"]`)?.focus();
    }
    diagram.addEventListener('click',chooseLeaf);
    diagram.addEventListener('keydown',chooseLeaf);
    doc.addEventListener('keydown',event=>{
      if(event.target.closest('input,select,button,a,[role="button"]'))return;
      if(event.key==='ArrowRight'){event.preventDefault();go(scene+1);}
      if(event.key==='ArrowLeft'){event.preventDefault();go(scene-1);}
    });
    win.addEventListener('scroll',onScroll,{passive:true});
    win.addEventListener('resize',()=>{render();onScroll();},{passive:true});
    reduced.addEventListener('change',()=>{stopAnimation();render();});
    render();onScroll();
  }
  return {renderScene,readout,cuts,scaled,init};
});
