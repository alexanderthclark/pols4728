import {atusResults} from './atus-results.mjs';

// SVG contents for a 900 × 460 viewBox, or 400 × 350 on mobile.
// Scroll transitions interpolate geometry, not intermediate fitted results.
const C = {ink:'#242424', muted:'#606060', rule:'#767676', grid:'#e0e4e6'};
const clamp = p => Math.max(0, Math.min(1, Number.isFinite(p) ? p : 0));
const smooth = p => {const t=clamp(p); return t*t*(3-2*t);};
const mix = (a,b,p) => a+(b-a)*p;
const f = n => Number(n.toFixed(4));
const escape = value => String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const text = (x,y,value,size=18,color=C.ink,anchor='start',attrs='') => `<text x="${f(x)}" y="${f(y)}" font-size="${size}" fill="${color}" text-anchor="${anchor}" ${attrs}>${escape(value)}</text>`;
const line = (x1,y1,x2,y2,color=C.rule,width=1,attrs='') => `<line x1="${f(x1)}" y1="${f(y1)}" x2="${f(x2)}" y2="${f(y2)}" stroke="${color}" stroke-width="${width}" ${attrs}/>`;
const group = (s,opacity=1,attrs='') => `<g opacity="${f(opacity)}" ${attrs}>${s}</g>`;
const score = value => value.toFixed(5);

function layout(mobile) {
  return mobile
    ? {W:400,H:350,labelX:8,left:183,right:332,valueX:394,firstY:85,rowGap:44,
      top:65,bottom:282,axisY:288,tickY:308,axisLabelY:328,footY:346,
      titleY:23,subtitleY:47,nameSize:17,variantSize:17,valueSize:17}
    : {W:900,H:460,labelX:32,left:335,right:795,valueX:882,firstY:110,rowGap:62,
      top:82,bottom:382,axisY:400,tickY:425,axisLabelY:453,footY:453,
      titleY:32,subtitleY:59,nameSize:23,variantSize:18,valueSize:24};
}

function regimeHeading(t,mobile,L) {
  const moving=t>0&&t<1, isSmall=t===1;
  const title=moving?(mobile?'Changing to 200 rows':'Changing to 200 balanced training rows')
    :isSmall?'Same 200 training rows':'Larger training set';
  const subtitle=moving||isSmall?'100 positive + 100 negative':mobile
    ? 'AG 136,146 · TFM context 6,468'
    : 'AutoGluon fits 136,146 rows · Transformers receive 6,468 context rows';
  return group(text(L.labelX,L.titleY,title,mobile?22:27)
    +text(L.labelX,L.subtitleY,subtitle,mobile?16:18,C.muted));
}

function axes(L,mobile) {
  const scale=value=>L.left+(value/atusResults.axis.max)*(L.right-L.left);
  let s='';
  for(const value of (mobile?[0,.025,.05]:[0,.02,.04,.055])) {
    const x=scale(value), label=value===0?'0':value.toFixed(3).replace(/^0/,'');
    s+=line(x,L.top,x,L.bottom,C.grid,1)
      +line(x,L.axisY,x,L.axisY+5,C.rule,1)
      +text(x,L.tickY,label,mobile?17:18,C.muted,value===0?'start':value===.055?'end':'middle');
  }
  const baseline=scale(atusResults.test.prevalence);
  s+=line(L.left,L.axisY,L.right,L.axisY,C.rule,1.2)
    +line(baseline,L.top,baseline,L.bottom,C.rule,1.3,'stroke-dasharray="4 5" data-mark="prevalence-baseline"')
    +text((L.left+L.right)/2,L.axisLabelY,'Average precision (AP)',mobile?17:18,C.ink,'middle');
  if(mobile) {
    // Keep the footnote at the bottom of the viewBox, outside the five rows.
    s+=text(200,L.footY,'Dashed: prevalence .0117',16,C.muted,'middle');
  } else {
    s+=text(L.labelX,L.footY,'Dashed: prevalence .0117',16,C.muted);
  }
  return s;
}

function rows(stage,p,t,mobile,L) {
  const scale=value=>L.left+(value/atusResults.axis.max)*(L.right-L.left);
  let s='';
  atusResults.models.forEach((model,i)=>{
    const y=L.firstY+i*L.rowGap;
    // AutoGluon appears first, followed by the four frozen TFM variants.
    const reveal=stage===0?smooth((p-(i===0?0:.23+(i-1)*.15))/(i===0?.22:.19)):1;
    const currentAP=mix(model.larger,model.small,t);
    const x=scale(currentAP), value=t<.5?model.larger:model.small;
    const valueOpacity=stage===0?reveal:(t>0&&t<1?0:1);
    s+=`<g data-model="${model.id}" data-ap-larger="${model.larger}" data-ap-small="${model.small}">`
      +`<title>${escape(`${model.family}, ${model.variant}: larger regime AP ${model.larger}; 200-row regime AP ${model.small}.`)}</title>`
      +text(L.labelX,y-(mobile?5:7),model.family,L.nameSize,model.color)
      +text(L.labelX,y+(mobile?14:18),model.variant,L.variantSize,C.muted)
      +group(`<rect x="${L.left}" y="${f(y-6)}" width="${f(x-L.left)}" height="12" fill="${model.color}" fill-opacity=".24"/>`
        +line(L.left,y,x,y,model.color,mobile?3:4)
        +`<circle cx="${f(x)}" cy="${y}" r="${mobile?4.5:6}" fill="${model.color}" data-mark="ap"/>`,reveal)
      +group(text(L.valueX,y+(mobile?6:8),score(value),L.valueSize,model.color,'end',
        'font-variant-numeric="tabular-nums" data-mark="ap-label"'),valueOpacity)
      +'</g>';
  });
  return s;
}

export function renderAtus(stage,p,mobile=false) {
  const scene=Number(stage)===1?1:0, progress=clamp(p), L=layout(mobile);
  const t=scene===1?smooth((progress-.06)/.86):0;
  const regime=scene===0?'larger':t===0?'larger':t===1?'small':'transition';
  const title=scene===0
    ? 'Official-test average precision in the larger training regime'
    : 'Official-test average precision: a new 200-row balanced training regime';
  const desc='Five prespecified procedures share 44,885 official test observations, including 526 positives. '
    +'The dashed line is test prevalence, 0.0117. AutoGluon selects ensembles using training out-of-fold predictions. '
    +'The larger regime gives AutoGluon 136,146 fit rows and transformers 6,468 context rows. '
    +'The second regime gives every learner the same 200 rows, 100 positive and 100 negative. '
    +'Both sample size and balance change; this is not a controlled learning curve. '
    +'The 200-row TabDPT 100-feature minus AutoGluon AP difference is 0.002087, with paired 95% interval [-0.0020475, 0.0086369]. '
    +'Animation between endpoint bars is a visual transition, not another measured result.';
  return `<g font-family="inherit" data-atus-stage="${scene}" data-regime="${regime}" data-viewbox="0 0 ${L.W} ${L.H}" data-scale-min="0" data-scale-max="0.055">`
    +`<title>${title}</title><desc>${escape(desc)}</desc>`
    +regimeHeading(t,mobile,L)+axes(L,mobile)+rows(scene,progress,t,mobile,L)+'</g>';
}
