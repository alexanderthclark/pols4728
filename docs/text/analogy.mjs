import {latex} from './math.mjs';

const clamp = value => Math.max(0, Math.min(1, value));

const examples = {
  royalty: {words:['man','king','woman','queen'], relation:'Royalty'},
  capitals: {words:['France','Paris','Germany','Berlin'], relation:'Capital city'}
};

const vector = word => String.raw`v_{\mathrm{${word}}}`;

/** Constructed coordinates make a familiar vector relationship visible. */
export function wordAnalogy(progress, {reducedMotion=false, stage=0, example='royalty'}={}) {
  const name = Object.hasOwn(examples,example) ? example : 'royalty';
  const {words:[a,b,c,d],relation} = examples[name];
  const p = reducedMotion ? Number(stage>=1) : clamp(progress);
  const offset = 280*p;
  const arithmetic = reducedMotion ? Number(stage>=1) : clamp((p-.62)/.38);
  const copyOpacity = reducedMotion ? Number(stage>=1) : clamp(p/.16);
  const point = (x,y,label,color) => `<circle cx="${x}" cy="${y}" r="7" fill="white" stroke="${color}" stroke-width="3"/><text x="${x}" y="${y<200?y-27:y+41}" text-anchor="middle" fill="#222" font-size="32">${label}</text>`;
  // Both fixed arrows have displacement (175, -170); tips stop at the word dots.
  const arrow = (x,color,id,extra='') => `<line x1="${x+7}" y1="268" x2="${x+167}" y2="113" stroke="${color}" stroke-width="4" marker-end="url(#${id})" ${extra}/>`;
  const id = `word-analogy-${name}`;
  return `<div class="analogy-scene analogy-${name}" data-example="${name}" data-progress="${p.toFixed(3)}" data-translation="${offset.toFixed(3)}" data-stage="${stage}">
    <svg class="analogy-graph" viewBox="0 0 680 390" role="img" aria-labelledby="${id}-title ${id}-desc">
      <title id="${id}-title">${a} to ${b} is parallel to ${c} to ${d}</title>
      <desc id="${id}-desc">Two parallel arrows show the ${relation.toLowerCase()} relationship. A copy of the first arrow moves to start at ${c} and ends at ${d}. The coordinates are illustrative.</desc>
      <defs>
        <marker id="${id}-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#234e70"/></marker>
        <marker id="${id}-rust" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#a54f32"/></marker>
        <marker id="${id}-teal" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#007f73"/></marker>
      </defs>
      <g fill="none" stroke="#b6b6b6" stroke-width="1.5" stroke-dasharray="4 7" aria-hidden="true"><path d="M110 275H390M285 105H565"/></g>
      <g class="analogy-fixed-arrows">
        ${arrow(110,'#234e70',`${id}-blue`)}
        ${arrow(390,'#a54f32',`${id}-rust`)}
      </g>
      <g class="analogy-moving-arrow" transform="translate(${offset.toFixed(3)} 0)" opacity="${copyOpacity.toFixed(3)}" aria-hidden="true">
        ${arrow(110,'#007f73',`${id}-teal`,'stroke-dasharray="9 6"')}
        <circle cx="110" cy="275" r="11" fill="white" stroke="#007f73" stroke-width="2"/>
      </g>
      ${point(110,275,a,'#234e70')}
      ${point(285,105,b,'#234e70')}
      ${point(390,275,c,'#a54f32')}
      ${point(565,105,d,'#a54f32')}
      <text x="340" y="370" text-anchor="middle" fill="#007f73" font-size="21" opacity="${copyOpacity.toFixed(3)}" aria-hidden="true">Translate the same step</text>
    </svg>
    <div class="analogy-equations">
      <p class="analogy-relation">${latex(String.raw`${vector(b)} - ${vector(a)} \approx ${vector(d)} - ${vector(c)}`)}</p>
      <p class="analogy-arithmetic" style="opacity:${arithmetic.toFixed(3)}" aria-hidden="${arithmetic<.9}">${latex(String.raw`${vector(b)} - ${vector(a)} + ${vector(c)} \approx ${vector(d)}`)}</p>
    </div>
    <p class="visual-note analogy-note">Schematic geometry · illustrative coordinates</p>
  </div>`;
}
