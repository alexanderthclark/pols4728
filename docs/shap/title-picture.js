(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.SHAPTitlePicture = api;
})(typeof window !== 'undefined' ? window : globalThis, function () {
  'use strict';

  // Walking uses its own repeating clock. The hatch is a pure function of the
  // scroll progress, so reversing a scroll never restarts or reverses the walk.
  const loopDuration = 10400;
  const viewBox = '0 0 1000 500';
  const hinge = { x: 866, y: 325 };
  const lid = {
    left: { x: 564, y: 325 }, right: { x: 866, y: 325 },
    upper: [{ x: 576, y: 230 }, { x: 854, y: 230 }],
    lower: [{ x: 840, y: 355 }, { x: 590, y: 355 }]
  };
  const clamp = value => Math.max(0, Math.min(1, value));
  const finite = value => Number.isFinite(value) ? value : 0;
  const smooth = value => { const p = clamp(value); return p * p * (3 - 2 * p); };
  const mix = (a, b, p) => a + (b - a) * p;
  const num = value => String(Number(value.toFixed(3)));
  const point = p => `${num(p.x)} ${num(p.y)}`;

  function rotate(p, angle) {
    const radians = angle * Math.PI / 180, c = Math.cos(radians), s = Math.sin(radians);
    const x = p.x - hinge.x, y = p.y - hinge.y;
    return { x: hinge.x + x * c - y * s, y: hinge.y + x * s + y * c };
  }

  function cubic(a, b, c, d, t) {
    const u = 1 - t;
    return {
      x: u ** 3 * a.x + 3 * u * u * t * b.x + 3 * u * t * t * c.x + t ** 3 * d.x,
      y: u ** 3 * a.y + 3 * u * u * t * b.y + 3 * u * t * t * c.y + t ** 3 * d.y
    };
  }

  function walkAt(time) {
    if (time < 1200) return { phase: 'inspect-back', progress: 0, from: 0, to: 0 };
    if (time < 3600) return { phase: 'approach', progress: (time - 1200) / 2400, from: 0, to: 48 };
    if (time < 5200) return { phase: 'inspect-near', progress: 1, from: 48, to: 48 };
    if (time < 7600) return { phase: 'retreat', progress: (time - 5200) / 2400, from: 48, to: 0 };
    return { phase: 'inspect-back', progress: 0, from: 0, to: 0 };
  }

  // Exactly one foot swings in each half of a walk. Outside its own swing,
  // that foot's world position is constant, including throughout each pause.
  function walkingFoot(index, home, walk) {
    const progress = clamp((walk.progress - index / 2) * 2);
    const swinging = walk.from !== walk.to && progress > 0 && progress < 1;
    const lift = swinging ? 13 * Math.sin(Math.PI * progress) : 0;
    const groundY = index === 0 ? 440 : 446;
    return {
      x: home + mix(walk.from, walk.to, smooth(progress)), y: groundY - lift,
      groundY, lift, planted: !swinging,
      rock: swinging ? 7 * Math.sin(2 * Math.PI * progress) : 0
    };
  }

  function jointBetween(a, b, length, bend) {
    const dx = b.x - a.x, dy = b.y - a.y, distance = Math.hypot(dx, dy) || 1;
    const height = Math.sqrt(Math.max(0, length * length - distance * distance / 4));
    return {
      x: (a.x + b.x) / 2 - dy / distance * height * bend,
      y: (a.y + b.y) / 2 + dx / distance * height * bend
    };
  }

  function figure(name, loopTime, reducedMotion) {
    const near = name === 'magnifier', home = near ? 345 : 166;
    const time = (loopTime + (near ? 0 : loopDuration - 260)) % loopDuration;
    const walk = reducedMotion ? { phase: 'inspect-near', progress: 1, from: 48, to: 48 } : walkAt(time);
    const feet = [-17, 17].map((offset, index) => walkingFoot(index, home + offset, walk));
    // Follow the midpoint of the feet instead of sliding past a planted foot.
    const x = (feet[0].x + feet[1].x) / 2;
    const y = (near ? 316 : 307) - Math.max(feet[0].lift, feet[1].lift) * .18;
    const radius = near ? 60 : 57;
    const legLength = near ? 39 : 44.5;
    const legs = feet.map((foot, index) => {
      // Hip depth follows the six-unit shoe depth, keeping both resting knees
      // gently bent instead of pushing one knee far out to the side.
      const hip = { x: x + (index === 0 ? -20 : 20), y: y + radius - 13 + index * 6 };
      const ankle = { x: foot.x, y: foot.y - 5 };
      return { hip, knee: jointBetween(hip, ankle, legLength, index === 0 ? 1 : -1), ankle };
    });
    const arms = near ? [
      { shoulder: { x: x - 48, y: y + 13 }, elbow: { x: x - 79, y: y + 34 }, hand: { x: x - 66, y: y + 57 }, kind: 'relaxed' },
      { shoulder: { x: x + 48, y: y + 13 }, elbow: { x: x + 64, y: y + 44 }, hand: { x: x + 88, y: y + 26 }, kind: 'grip' }
    ] : [
      { shoulder: { x: x - 46, y: y + 8 }, elbow: { x: x - 80, y: y - 3 }, hand: { x: x - 36, y: y - 48 }, kind: 'scratch' },
      { shoulder: { x: x + 46, y: y + 9 }, elbow: { x: x + 77, y: y + 31 }, hand: { x: x + 44, y: y + 43 }, kind: 'hip' }
    ];
    const target = { x: 684, y: 257 };
    const gazeLength = Math.hypot(target.x - x, target.y - y);
    return {
      name, x, y, radius, phase: walk.phase, walking: walk.from !== walk.to,
      walkProgress: walk.progress, legLength, feet, legs, arms,
      gaze: { x: 3.6 * (target.x - x) / gazeLength, y: 3.6 * (target.y - y) / gazeLength },
      blink: !reducedMotion && time > 8780 && time < 8910,
      magnifier: near ? { x: x + 118, y: y - 29, rx: 31, ry: 40, angle: 18 } : null
    };
  }

  function stateAt(elapsedMs = 0, openProgress = 0, reducedMotion = false) {
    const loopTime = reducedMotion ? 4300 : Math.max(0, finite(elapsedMs)) % loopDuration;
    const progress = reducedMotion ? 1 : clamp(finite(openProgress));
    const opening = smooth(progress), angle = 64 * opening;
    const outline = Array.from({ length: 64 }, (_, index) => rotate(index < 32
      ? cubic(lid.left, ...lid.upper, lid.right, index / 32)
      : cubic(lid.right, ...lid.lower, lid.left, (index - 32) / 32), angle));
    // The reveal starts at the cockpit lip, not at the top of the glyph. The
    // opaque lid then occludes the portion still physically behind the lid.
    const revealTop = mix(343, 189, opening);
    return {
      loopTime, reducedMotion: Boolean(reducedMotion), openProgress: progress,
      hatch: { hinge: { ...hinge }, angle, outline, revealTop, revealBottom: 343 },
      figures: [figure('puzzled', loopTime, reducedMotion), figure('magnifier', loopTime, reducedMotion)]
    };
  }

  function path(d, width = 4, extra = '') {
    return `<path d="${d}" fill="none" stroke="#000" stroke-width="${width}" stroke-linecap="round" stroke-linejoin="round" ${extra}/>`;
  }

  function tube(d, width = 12) {
    return path(d, width) + path(d, width - 6, 'data-limb-interior="true" style="stroke:#fff"');
  }

  function hand(arm) {
    const { x, y } = arm.hand;
    if (arm.kind === 'grip') {
      return `<g transform="translate(${num(x)} ${num(y)})"><path d="M-7-8Q1-15 9-9L13-5Q17 0 11 4Q16 8 10 12Q2 18-7 12L-11 6Q-15-1-7-8Z" fill="#fff" stroke="#000" stroke-width="3.5" stroke-linejoin="round"/>${path('M1-5 10-2M2 2 11 5M1 9 8 10', 2.5)}</g>`;
    }
    if (arm.kind === 'scratch') {
      return `<g transform="translate(${num(x)} ${num(y)}) rotate(-12)"><path d="M-8 10Q-17 0-12-10Q-8-20-1-15Q3-25 10-19Q16-16 10-9Q23-12 23-5Q23 0 12 3Q22 3 20 10Q17 17 5 18L-3 17Z" fill="#fff" stroke="#000" stroke-width="3.5" stroke-linejoin="round"/>${path('M-1-8 8-14M1 0 12-4M2 8 12 6', 2.5)}</g>`;
    }
    return `<g transform="translate(${num(x)} ${num(y)})"><path d="M-10-8Q0-17 10-8Q18-1 11 10Q6 16 2 9Q-1 18-6 10Q-14 14-14 4Q-18-2-10-8Z" fill="#fff" stroke="#000" stroke-width="3.5" stroke-linejoin="round"/></g>`;
  }

  function shoe(foot, index) {
    return `<g data-foot="${index}" data-planted="${foot.planted}" transform="translate(${num(foot.x)} ${num(foot.y)}) rotate(${num(foot.rock)})"><path d="M-6-8C-17-9-29 1-27 8C-25 15 11 15 18 10C25 4 12-5 3-7Z" fill="#fff" stroke="#000" stroke-width="3.5" stroke-linejoin="round"/></g>`;
  }

  function renderFigure(person) {
    const { x, y, radius, gaze } = person;
    const legs = person.legs.map(leg => tube(`M${point(leg.hip)}L${point(leg.knee)}L${point(leg.ankle)}`, 13)).join('');
    const arms = person.arms.map(arm => tube(`M${point(arm.shoulder)}Q${point(arm.elbow)} ${point(arm.hand)}`, 12)).join('');
    const eyes = [7, 31].map(offset => person.blink
      ? path(`M${num(x + offset - 9)} ${num(y - 9)}Q${num(x + offset)} ${num(y - 5)} ${num(x + offset + 9)} ${num(y - 9)}`, 3)
      : `<ellipse cx="${num(x + offset)}" cy="${num(y - 9)}" rx="11" ry="15" fill="#fff" stroke="#000" stroke-width="3"/><ellipse cx="${num(x + offset + gaze.x)}" cy="${num(y - 9 + gaze.y)}" rx="5" ry="8" fill="#000"/>`).join('');
    const brows = path(`M${num(x - 3)} ${num(y - 31)}Q${num(x + 7)} ${num(y - 38)} ${num(x + 17)} ${num(y - 32)}M${num(x + 22)} ${num(y - 33)}Q${num(x + 31)} ${num(y - 41)} ${num(x + 42)} ${num(y - 36)}`, 3.5);
    const mouth = person.magnifier
      ? path(`M${num(x + 9)} ${num(y + 22)}Q${num(x + 23)} ${num(y + 36)} ${num(x + 35)} ${num(y + 20)}`, 3.5)
      : path(`M${num(x + 13)} ${num(y + 27)}Q${num(x + 24)} ${num(y + 17)} ${num(x + 34)} ${num(y + 24)}`, 3.5);
    let magnifier = '';
    if (person.magnifier) {
      const lens = person.magnifier;
      magnifier = tube(`M${num(x + 106)} ${num(y + 6)}L${num(x + 86)} ${num(y + 45)}`, 13)
        + `<g transform="rotate(${lens.angle} ${num(lens.x)} ${num(lens.y)})"><ellipse cx="${num(lens.x)}" cy="${num(lens.y)}" rx="${lens.rx}" ry="${lens.ry}" fill="#fff" stroke="#000" stroke-width="4"/><ellipse cx="${num(lens.x)}" cy="${num(lens.y)}" rx="24" ry="33" fill="#fff" stroke="#000" stroke-width="2.5"/>${path(`M${num(lens.x - 12)} ${num(lens.y - 2)}L${num(lens.x + 7)} ${num(lens.y - 23)}M${num(lens.x - 7)} ${num(lens.y + 6)}L${num(lens.x + 14)} ${num(lens.y - 17)}`, 2.5)}</g>`;
    }
    return `<g data-figure="${person.name}" data-phase="${person.phase}" data-x="${num(x)}">${legs}${arms}<circle data-round-body="true" cx="${num(x)}" cy="${num(y)}" r="${radius}" fill="#fff" stroke="#000" stroke-width="4.5"/>${eyes}${brows}${mouth}${person.feet.map(shoe).join('')}${magnifier}${person.arms.map(hand).join('')}</g>`;
  }

  function renderSaucer(state) {
    const { hatch } = state;
    const revealHeight = hatch.revealBottom - hatch.revealTop;
    const cavity = '<ellipse cx="715" cy="328" rx="147" ry="37" fill="#fff" stroke="#000" stroke-width="4"/><ellipse cx="715" cy="328" rx="132" ry="28" fill="#fff" stroke="#000" stroke-width="3"/>';
    const prediction = `<defs><clipPath id="shap-title-symbol-reveal"><rect x="566" y="${num(hatch.revealTop)}" width="291" height="${num(revealHeight)}"/></clipPath></defs><g data-prediction-symbol="y-hat" clip-path="url(#shap-title-symbol-reveal)"><text x="683" y="316" text-anchor="middle" font-family="'Course Math','Cambria Math',serif" font-style="italic" font-size="151" fill="#000" stroke="none">y</text>${path('M658 231 678 219 699 231', 4)}</g>`;
    const lidPath = `M${point(lid.left)}C${point(lid.upper[0])} ${point(lid.upper[1])} ${point(lid.right)}C${point(lid.lower[0])} ${point(lid.lower[1])} ${point(lid.left)}Z`;
    const openLid = `<g data-hinged-lid="true" transform="rotate(${num(hatch.angle)} ${hinge.x} ${hinge.y})"><path d="${lidPath}" fill="#fff" stroke="#000" stroke-width="4.5"/><path d="M578 325C589 249 840 249 852 325C823 345 607 345 578 325Z" fill="none" stroke="#000" stroke-width="2.5"/>${path('M617 302Q637 289 656 290M630 315Q643 305 656 304M782 288Q811 298 827 314', 2.5)}</g>`;
    const landingLegs = tube('M594 392 574 438M828 394 850 440', 15)
      + '<ellipse cx="571" cy="445" rx="26" ry="10" fill="#fff" stroke="#000" stroke-width="4"/><ellipse cx="854" cy="447" rx="26" ry="10" fill="#fff" stroke="#000" stroke-width="4"/>';
    const hull = '<path d="M536 359Q549 333 578 325Q715 371 850 325Q885 337 903 359Q894 412 720 418Q555 414 536 359Z" fill="#fff" stroke="#000" stroke-width="4.5" stroke-linejoin="round"/>'
      + path('M536 359Q714 414 903 359M549 376Q719 425 890 376', 3)
      + [572, 639, 715, 791, 866].map((cx, index) => `<ellipse cx="${cx}" cy="${index === 0 || index === 4 ? 356 : index === 2 ? 377 : 371}" rx="10" ry="13" fill="#fff" stroke="#000" stroke-width="3"/>`).join('')
      + path('M568 343 578 350M584 338 592 345M840 343 850 337M847 349 858 343', 2.5);
    // Back rim, symbol, lid, then front hull: both the moving lid and the
    // cockpit lip are opaque foreground surfaces, never transparent overlays.
    return `<g data-saucer="true">${landingLegs}${cavity}${prediction}${openLid}${hull}<ellipse cx="865" cy="325" rx="11" ry="8" fill="#fff" stroke="#000" stroke-width="3.5"/></g>`;
  }

  function renderState(state) {
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${viewBox}" aria-hidden="true" focusable="false" data-shap-title-picture="true" data-open-progress="${num(state.openProgress)}"><g fill="none" stroke="#000" stroke-linecap="round" stroke-linejoin="round">${path('M93 460H239M281 460H440M549 465H888', 2)}${renderSaucer(state)}${state.figures.map(renderFigure).join('')}</g></svg>`;
  }

  return { loopDuration, viewBox, stateAt, renderState, render: (elapsedMs = 0, openProgress = 0, reducedMotion = false) => renderState(stateAt(elapsedMs, openProgress, reducedMotion)) };
});
