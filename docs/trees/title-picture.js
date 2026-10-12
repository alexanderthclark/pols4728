(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.TitlePicture = api;
})(typeof window !== 'undefined' ? window : globalThis, function () {
  'use strict';

  // Millisecond timeline, with a terminal inspection rather than a loop.
  // stateAt is pure; renderState accepts that state, and render(ms) combines both.
  // During adjustment, figures[active].hand equals the rotating contact corner.
  const duration = 24000;
  const angles = [-23, 13, -7.5, 4, -2.2, 1.2];
  const initialHold = 900, cycleDuration = 4200;
  const frame = { cx: 420, cy: 151, width: 238, height: 148 };
  const clamp = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));
  const smooth = t => { const p = clamp(t); return p * p * (3 - 2 * p); };
  const mix = (a, b, p) => a + (b - a) * p;
  const pointMix = (a, b, p) => ({ x: mix(a.x, b.x, p), y: mix(a.y, b.y, p) });
  const num = value => String(Number(value.toFixed(3)));
  const esc = value => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  function rotatePoint(x, y, angle) {
    const radians = angle * Math.PI / 180, c = Math.cos(radians), s = Math.sin(radians);
    return { x: frame.cx + x * c - y * s, y: frame.cy + x * s + y * c };
  }

  function picture(angle) {
    const x = frame.width / 2, y = frame.height / 2;
    return {
      ...frame, angle,
      corners: {
        upperLeft: rotatePoint(-x, -y, angle), upperRight: rotatePoint(x, -y, angle),
        lowerLeft: rotatePoint(-x, y, angle), lowerRight: rotatePoint(x, y, angle)
      }
    };
  }

  // A foot stays planted while the other advances. Each walk contains four
  // alternating steps; its two feet end at the exact standing positions.
  function walkingFoot(side, start, end, p) {
    const intervals = side === 0 ? [[0, .25], [.5, .75]] : [[.25, .5], [.75, 1]];
    let advance = 0, lift = 0, rock = 0;
    for (const [lo, hi] of intervals) {
      const u = clamp((p - lo) / (hi - lo));
      advance += .5 * smooth(u);
      if (p > lo && p < hi) {
        lift = 11 * Math.sin(Math.PI * u);
        rock = 8 * Math.sin(2 * Math.PI * u);
      }
    }
    return { x: mix(start, end, advance), y: 415 - lift, lift, rock };
  }

  function figure(name, phase, phaseProgress, isActive, elapsedMs, pic) {
    const direction = name === 'left' ? 1 : -1;
    const back = name === 'left' ? 144 : 696, front = name === 'left' ? 270 : 570;
    const walking = isActive && (phase === 'approach' || phase === 'retreat');
    let x = back, y = 306, walkProgress = 0, walkFrom = back, walkTo = back;
    if (walking) {
      walkProgress = smooth(phaseProgress);
      walkFrom = phase === 'approach' ? back : front;
      walkTo = phase === 'approach' ? front : back;
      x = mix(walkFrom, walkTo, walkProgress);
      y -= 3 * Math.sin(4 * Math.PI * walkProgress) ** 2;
    } else if (isActive && ['touch', 'adjust', 'release'].includes(phase)) x = front;
    const feet = [-20, 20].map((offset, i) => walking
      ? walkingFoot(i, walkFrom + offset, walkTo + offset, walkProgress)
      : { x: x + offset, y: 415, lift: 0, rock: 0 });
    const chin = { x: x + direction * 14, y: y + 43 };
    const relaxed = { x: x + direction * 49, y: y + 41 };
    const cornerName = name === 'left' ? 'lowerLeft' : 'lowerRight';
    const corner = pic.corners[cornerName];
    let hand = chin, contact = false, outerSwing = 0;
    const reachWeight = !isActive ? 0 : phase === 'touch' ? smooth(phaseProgress) : phase === 'adjust' ? 1 : phase === 'release' ? 1 - smooth(phaseProgress) : 0;
    if (walking) {
      const swing = Math.sin(4 * Math.PI * walkProgress) * Math.sin(Math.PI * walkProgress);
      const movingHand = { x: relaxed.x + 14 * swing * direction, y: relaxed.y - 6 * swing };
      const blend = phase === 'approach' ? smooth(phaseProgress / .22) : 1 - smooth((phaseProgress - .78) / .22);
      hand = pointMix(chin, movingHand, blend);
      outerSwing = -swing * 16;
    } else if (isActive && phase === 'touch') hand = pointMix(relaxed, corner, smooth(phaseProgress));
    else if (isActive && phase === 'adjust') { hand = { ...corner }; contact = true; }
    else if (isActive && phase === 'release') hand = pointMix(corner, relaxed, smooth(phaseProgress));
    const outerHand = { x: x - direction * (46 + outerSwing), y: y + 38 + outerSwing * .35 };
    const gazeTarget = isActive && ['touch', 'adjust', 'release'].includes(phase) ? corner : { x: pic.cx, y: pic.cy };
    const gazeLength = Math.hypot(gazeTarget.x - x, gazeTarget.y - y) || 1;
    const blinkAt = (elapsedMs + (name === 'left' ? 0 : 1700)) % 5600;
    return {
      name, direction, x, y, walking, walkProgress, feet, hand, outerHand, reachWeight,
      contact, corner: cornerName, looking: { x: 2.4 * (gazeTarget.x - x) / gazeLength, y: 2.4 * (gazeTarget.y - y) / gazeLength },
      blink: blinkAt > 4520 && blinkAt < 4660,
      expression: isActive && phase === 'adjust' ? 'concentrating' : phase === 'complete' ? 'pleased' : 'inspecting'
    };
  }

  function stateAt(elapsedMs = 0) {
    const elapsed = clamp(Number.isFinite(elapsedMs) ? elapsedMs : 0, 0, duration);
    let phase = 'inspect', phaseProgress = 0, active = null, iteration = 0, angle = angles[0];
    const cycleTime = elapsed - initialHold;
    if (cycleTime >= 0 && cycleTime < cycleDuration * (angles.length - 1)) {
      const cycle = Math.floor(cycleTime / cycleDuration), local = cycleTime - cycle * cycleDuration;
      iteration = cycle + 1;
      const actor = cycle % 2 === 0 ? 'left' : 'right';
      const intervals = [['approach', 0, 850], ['touch', 850, 1100], ['adjust', 1100, 2500], ['release', 2500, 2700], ['retreat', 2700, 3650], ['inspect', 3650, 4200]];
      const interval = intervals.find(([, start, end]) => local >= start && local < end);
      phase = interval[0];
      phaseProgress = (local - interval[1]) / (interval[2] - interval[1]);
      active = phase === 'inspect' ? null : actor;
      angle = local < 1100 ? angles[cycle] : local >= 2500 ? angles[cycle + 1] : mix(angles[cycle], angles[cycle + 1], smooth((local - 1100) / 1400));
    } else if (cycleTime >= cycleDuration * (angles.length - 1)) {
      iteration = angles.length - 1;
      angle = angles[angles.length - 1];
      phase = elapsed >= duration ? 'complete' : 'inspect';
      phaseProgress = clamp((elapsed - initialHold - cycleDuration * (angles.length - 1)) / (duration - initialHold - cycleDuration * (angles.length - 1)));
    }
    const pic = picture(angle);
    return {
      elapsedMs: elapsed, progress: elapsed / duration, done: elapsed === duration,
      phase, phaseProgress, active, iteration, angle, picture: pic,
      figures: {
        left: figure('left', phase, phaseProgress, active === 'left', elapsed, pic),
        right: figure('right', phase, phaseProgress, active === 'right', elapsed, pic)
      }
    };
  }

  function elbow(a, b, bend, length = 62) {
    const dx = b.x - a.x, dy = b.y - a.y, distance = Math.hypot(dx, dy) || 1;
    const height = Math.sqrt(Math.max(0, length * length - distance * distance / 4));
    return { x: (a.x + b.x) / 2 - dy / distance * height * bend, y: (a.y + b.y) / 2 + dx / distance * height * bend };
  }

  function renderState(state) {
    const pic = state.picture;
    const motionDescription = state.done
      ? 'Both figures stand back and inspect the nearly level picture.'
      : state.active ? `The ${state.active} figure is ${state.phase === 'adjust' ? 'rotating the picture past level while holding its lower corner' : state.phase === 'approach' ? 'walking toward the picture' : state.phase === 'retreat' ? 'walking back to inspect the picture' : state.phase === 'touch' ? 'reaching for a lower corner of the picture' : 'letting go of the picture'}. The other figure watches.`
        : 'Both figures stand back and inspect the picture.';
    const path = (d, width = 5, attrs = '') => `<path d="${d}" fill="none" stroke="#242424" stroke-width="${width}" stroke-linecap="round" stroke-linejoin="round" ${attrs}/>`;
    const cordLeft = rotatePoint(-63, -74, pic.angle), cordRight = rotatePoint(63, -74, pic.angle);
    let out = `<path d="M${num(cordLeft.x)} ${num(cordLeft.y)} L420 49 L${num(cordRight.x)} ${num(cordRight.y)}" fill="none" stroke="#606060" stroke-width="1.4" stroke-linejoin="round"/>`;
    out += '<circle cx="420" cy="49" r="4" fill="white" stroke="#242424" stroke-width="1.8"/><circle cx="420" cy="49" r="1.2" fill="#242424"/>';
    out += `<g data-picture="true" data-angle="${num(pic.angle)}" transform="translate(${pic.cx} ${pic.cy}) rotate(${num(pic.angle)})">`;
    out += '<rect x="-119" y="-74" width="238" height="148" rx="1.5" fill="white" stroke="#242424" stroke-width="4"/><rect x="-108" y="-63" width="216" height="126" fill="white" stroke="#242424" stroke-width="1.4"/>';
    out += path('M-96 43 L-42 -14 L7 38 M-12 20 L20 -10 L96 43 M-96 47 Q-45 39 -4 48 T96 47', 2);
    out += path('M-54 0 L-43 5 L-32 -2 M10 1 L22 9 L32 3', 1.4);
    out += '<circle cx="65" cy="-33" r="12" fill="none" stroke="#242424" stroke-width="1.8"/>';
    out += path('M65 -52 L65 -56 M84 -33 L88 -33 M78 -46 L81 -49 M52 -46 L49 -49', 1.5) + '</g>';
    out += path('M78 424 H337 M503 424 H762', 1, 'stroke-opacity=".25"');

    function glove(hand, rotation, contact) {
      return `<g transform="translate(${num(hand.x)} ${num(hand.y)}) rotate(${num(rotation)})" data-hand="${contact ? 'contact' : 'free'}" data-hand-x="${num(hand.x)}" data-hand-y="${num(hand.y)}"><path d="M-7 8 Q-13 5 -10 -1 L-9 -6 Q-8 -10 -5 -6 Q-3 -12 0 -7 Q4 -11 6 -6 Q10 -9 11 -3 L11 3 Q11 9 4 10 L-2 11 Z" fill="white" stroke="#242424" stroke-width="2" stroke-linejoin="round"/><path d="M-4 0 L-3 -4 M1 0 L1 -4 M6 1 L6 -3 M-7 6 Q-3 3 0 5" fill="none" stroke="#242424" stroke-width="1" stroke-linecap="round"/></g>`;
    }

    function person(f) {
      const { x, y, direction } = f;
      let body = `<g data-figure="${f.name}" data-walking="${f.walking}" data-contact="${f.contact}" data-x="${num(x)}" data-y="${num(y)}">`;
      for (let i = 0; i < f.feet.length; i++) {
        const foot = f.feet[i], hip = { x: x + (i === 0 ? -16 : 16), y: y + 30 };
        const knee = elbow(hip, foot, i === 0 ? -1 : 1, 45);
        body += path(`M${num(hip.x)} ${num(hip.y)} Q${num(knee.x)} ${num(knee.y)} ${num(foot.x)} ${num(foot.y)}`, 5.2);
      }
      const outerShoulder = { x: x - direction * 31, y: y + 10 };
      body += path(`M${num(outerShoulder.x)} ${num(outerShoulder.y)} Q${num(x - direction * 65)} ${num(y + 43)} ${num(f.outerHand.x)} ${num(f.outerHand.y)}`, 4.8);
      const innerShoulder = { x: x + direction * 31, y: y + 10 };
      const freeElbow = { x: x + direction * 62, y: y + 51 };
      const innerElbow = pointMix(freeElbow, elbow(innerShoulder, f.hand, direction), f.reachWeight);
      body += path(`M${num(innerShoulder.x)} ${num(innerShoulder.y)} Q${num(innerElbow.x)} ${num(innerElbow.y)} ${num(f.hand.x)} ${num(f.hand.y)}`, 4.8);
      body += `<circle cx="${num(x)}" cy="${num(y)}" r="44" fill="white" stroke="#242424" stroke-width="3.5"/>`;
      body += `<g transform="translate(${num(x)} ${num(y)}) scale(1.2) translate(${num(-x)} ${num(-y)})">`;
      for (const offset of [-10, 10]) {
        const ex = x + offset, ey = y - 7;
        if (f.blink) body += path(`M${num(ex - 4)} ${num(ey)} Q${num(ex)} ${num(ey + 2)} ${num(ex + 4)} ${num(ey)}`, 1.7);
        else body += `<ellipse cx="${num(ex)}" cy="${num(ey)}" rx="4.8" ry="6.7" fill="white" stroke="#242424" stroke-width="1.5"/><circle cx="${num(ex + f.looking.x)}" cy="${num(ey + f.looking.y)}" r="2.3" fill="#242424"/>`;
      }
      body += path(`M${num(x - 15)} ${num(y - 18)} Q${num(x - 10)} ${num(y - 21)} ${num(x - 5)} ${num(y - 18)} M${num(x + 5)} ${num(y - 18)} Q${num(x + 10)} ${num(y - 21)} ${num(x + 15)} ${num(y - 18)}`, 1.5);
      body += f.expression === 'concentrating'
        ? path(`M${num(x - 6)} ${num(y + 13)} Q${num(x)} ${num(y + 10)} ${num(x + 6)} ${num(y + 12)}`, 1.6)
        : path(`M${num(x - 8)} ${num(y + 11)} Q${num(x)} ${num(y + (f.expression === 'pleased' ? 21 : 17))} ${num(x + 8)} ${num(y + 11)}`, 1.6);
      body += '</g>';
      for (const foot of f.feet) {
        body += `<g transform="translate(${num(foot.x)} ${num(foot.y)}) rotate(${num(foot.rock * direction)}) scale(${direction} 1)"><path d="M-10 -5 Q-13 0 -10 7 Q5 10 23 6 Q27 4 23 0 Q15 -7 4 -6 Z" fill="white" stroke="#242424" stroke-width="2.3" stroke-linejoin="round"/><path d="M-9 4 Q6 7 23 4 M2 -4 L6 -1 M7 -4 L11 -1" fill="none" stroke="#242424" stroke-width="1.1" stroke-linecap="round"/></g>`;
      }
      body += glove(f.outerHand, -direction * 20, false);
      body += glove(f.hand, direction * mix(18, -20, f.reachWeight), f.contact);
      return body + '</g>';
    }
    out += person(state.figures.left) + person(state.figures.right);
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 840 440" fill="none" role="img" aria-labelledby="title-picture-title title-picture-description" data-title-picture="true" data-phase="${state.phase}" data-iteration="${state.iteration}" data-complete="${state.done}"><title id="title-picture-title">Two figures straighten a picture with smaller corrections</title><desc id="title-picture-description">Two round, torso-less stick figures take turns walking up to a wall-mounted landscape, turning it past level by smaller amounts, and stepping back to inspect it. ${esc(motionDescription)}</desc>${out}</svg>`;
  }

  return { duration, stateAt, renderState, render: (elapsedMs = 0) => renderState(stateAt(elapsedMs)) };
});
