/**
 * Exact two-hypothesis teaching example, not a TabPFN prediction.
 * B=1 means Brown; L=1 means Lie down. In world A, survival probability
 * is logistic(1) when B=L (Black + Fight or Brown + Lie down), and logistic(-1)
 * otherwise. World B reverses these probabilities. All observed cases are
 * Brown + Lie down survivors. Outcomes are independent conditional on the
 * fixed features and one shared world. These are hypothetical probabilities.
 */
export const MATCH_SURVIVAL = 1 / (1 + Math.exp(-1));
export const MISMATCH_SURVIVAL = 1 - MATCH_SURVIVAL;

export function survivalProbability(world, brown, lieDown) {
  if (!['A', 'B'].includes(world) || ![0, 1, false, true].includes(brown) || ![0, 1, false, true].includes(lieDown)) {
    throw new RangeError('Use world A or B, with binary brown and lieDown indicators.');
  }
  const matches = Number(brown) === Number(lieDown);
  return matches === (world === 'A') ? MATCH_SURVIVAL : MISMATCH_SURVIVAL;
}

export function toyState(n, prior = [.5, .5]) {
  if (!Number.isSafeInteger(n) || n < 0) throw new RangeError('n must be a nonnegative safe integer.');
  if (!Array.isArray(prior) || prior.length !== 2 || prior.some(p => !Number.isFinite(p) || p < 0) || !(prior[0] + prior[1] > 0) || !Number.isFinite(prior[0] + prior[1])) {
    throw new RangeError('prior must contain two finite nonnegative weights with a finite positive sum.');
  }
  const sum = prior[0] + prior[1];
  const normalized = prior.map(p => p / sum);
  const logWeights = normalized.map((p, i) => p === 0 ? -Infinity : Math.log(p) + n * Math.log(i === 0 ? MATCH_SURVIVAL : MISMATCH_SURVIVAL));
  const maxLog = Math.max(...logWeights);
  const weights = logWeights.map(p => Math.exp(p - maxLog));
  const total = weights[0] + weights[1];
  const posterior = weights.map(p => p / total);
  const matchPrediction = posterior[0] * MATCH_SURVIVAL + posterior[1] * MISMATCH_SURVIVAL;
  const mismatchPrediction = posterior[0] * MISMATCH_SURVIVAL + posterior[1] * MATCH_SURVIVAL;
  return {
    n,
    prior: normalized,
    likelihood: [MATCH_SURVIVAL ** n, MISMATCH_SURVIVAL ** n],
    likelihoodRatio: Math.exp(n),
    logLikelihoodRatio: n,
    posterior,
    prediction: matchPrediction,
    matchPrediction,
    mismatchPrediction,
  };
}

const C = { ink: '#242424', blue: '#234E70', rust: '#A54F32', teal: '#007F73', muted: '#606060', light: '#e5e8e9' };
const counts = [0, 1, 2, 4, 4];
const clamp = n => Math.max(0, Math.min(1, n));
const lerp = (a, b, t) => a + (b - a) * t;
const smooth = t => t * t * (3 - 2 * t);
const f = n => Number(n.toFixed(3));
const pct = n => `${(n * 100).toFixed(n === .5 ? 0 : 1)}%`;
const txt = (x, y, text, attrs = '') => `<text x="${x}" y="${y}" fill="${C.ink}" ${attrs.includes('font-size=') ? '' : 'font-size="21"'} ${attrs}>${text}</text>`;

/**
 * Return SVG contents for viewBox="0 0 900 460".
 * Stages: 0 prior; 1 one survivor; 2 two survivors; 3 four survivors;
 * 4 posterior-predictive mixture. Progress in [0,1] interpolates the update.
 * The in-between bars are animation states, not fractional-data posteriors.
 */
export function renderBayes(stage = 0, progress = 1) {
  stage = Math.max(0, Math.min(4, Math.round(Number(stage) || 0)));
  const raw = clamp(Number.isFinite(progress) ? progress : 1);
  const t = smooth(raw);
  const target = toyState(counts[stage]);
  const previous = toyState(counts[Math.max(0, stage - 1)]);
  const pa = lerp(previous.posterior[0], target.posterior[0], t);
  const pb = 1 - pa;
  const prediction = pa * MATCH_SURVIVAL + pb * MISMATCH_SURVIVAL;
  const updating = stage > 0 && stage < 4 && raw < .995;
  const count = counts[stage];
  const prevCount = counts[Math.max(0, stage - 1)];
  const status = updating ? `Adding ${count - prevCount} survivor${count - prevCount === 1 ? '' : 's'}` : count === 0 ? 'Before observing any outcomes' : `${count} survivor${count === 1 ? '' : 's'}: Brown + Lie down`;

  const bars = [[pa, C.blue, 'A', 86], [pb, C.rust, 'B', 162]].map(([p, color, name, y]) => `
    ${txt(500, y - 12, `World ${name}`, `style="fill:${color}"`)}
    <rect x="500" y="${y}" width="290" height="23" fill="#f0f2f3" rx="2"/>
    <rect x="500" y="${y}" width="${f(290 * p)}" height="23" fill="${color}" rx="2"/>
    ${txt(853, y + 20, pct(p), 'text-anchor="end" font-variant-numeric="tabular-nums"')}
  `).join('');

  const observations = Array.from({length: 4}, (_, i) => {
    const existed = i < prevCount;
    const isNew = i >= prevCount && i < count;
    const local = existed ? 1 : isNew ? clamp((raw - (i - prevCount) * .14) / .7) : 0;
    const x = 70 + i * 64;
    const y = lerp(355, 319, smooth(local));
    return `<circle cx="${x}" cy="319" r="20" fill="none" stroke="#c8cdd0" stroke-dasharray="3 4"/>
      ${local > 0 ? `<g opacity="${f(local)}" transform="translate(${x} ${f(y)})"><circle r="20" fill="${C.teal}"/><path d="M-8 0 L-2 6 L9 -7" fill="none" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></g>` : ''}`;
  }).join('');

  // A moving evidence marker joins the new observations to the changed weights.
  const flightT = clamp((raw - .12) / .72);
  const ex = lerp(317, 490, flightT);
  const ey = lerp(318, 214, flightT) - 22 * Math.sin(flightT * Math.PI);
  const evidence = stage > 0 && stage < 4 ? `
    <path d="M317 318 Q370 225 490 214" fill="none" stroke="#c6cbce" stroke-width="1.5" stroke-dasharray="4 6"/>
    ${txt(405, 288, 'Evidence', 'text-anchor="middle" style="fill:#606060"')}
    <circle cx="${f(ex)}" cy="${f(ey)}" r="7" fill="${C.teal}" opacity="${f(Math.sin(Math.PI * flightT))}"/>
  ` : '';

  const mixOpacity = stage === 4 ? lerp(.35, 1, t) : 1;
  const mixture = stage === 4
    ? `${(pa * 100).toFixed(1)}% × ${pct(MATCH_SURVIVAL)} + ${(pb * 100).toFixed(1)}% × ${pct(MISMATCH_SURVIVAL)} ≈ ${(prediction * 100).toFixed(1)}%`
    : `P(survive | match, D) = P(A | D) × p + P(B | D) × (1 − p)`;
  const highY = 215 - 145 * MATCH_SURVIVAL;
  const lowY = 215 - 145 * MISMATCH_SURVIVAL;

  return `<g font-family="inherit" data-bayes-stage="${stage}" data-bayes-count="${count}" data-posterior-a="${pa}" data-prediction="${prediction}">
    <title>Hypothetical bear survival rules and an exact Bayesian update</title>
    ${txt(40, 28, 'Two hypothetical survival rules')}
    ${txt(500, 28, count === 0 ? 'Prior probability' : 'Posterior probability')}
    ${txt(40, 61, 'P(survived)', 'font-size="20" style="fill:#606060"')}
    <path d="M115 70 V215 H330" fill="none" stroke="#8c9398" stroke-width="1.2"/>
    ${txt(99, f(highY + 7), pct(MATCH_SURVIVAL), 'text-anchor="end" font-size="20" style="fill:#606060"')}
    ${txt(99, f(lowY + 7), pct(MISMATCH_SURVIVAL), 'text-anchor="end" font-size="20" style="fill:#606060"')}
    <path d="M130 ${f(lowY)} L310 ${f(highY)}" fill="none" stroke="${C.blue}" stroke-width="3"/>
    <path d="M130 ${f(highY)} L310 ${f(lowY)}" fill="none" stroke="${C.rust}" stroke-width="3" stroke-dasharray="8 6"/>
    <circle cx="130" cy="${f(lowY)}" r="5" fill="${C.blue}"/><circle cx="310" cy="${f(highY)}" r="5" fill="${C.blue}"/>
    <rect x="125" y="${f(highY - 5)}" width="10" height="10" fill="${C.rust}"/><rect x="305" y="${f(lowY - 5)}" width="10" height="10" fill="${C.rust}"/>
    ${txt(329, f(highY + 7), 'A', `style="fill:${C.blue}"`)}
    ${txt(329, f(lowY + 7), 'B', `style="fill:${C.rust}"`)}
    ${txt(130, 239, 'Mismatch', 'text-anchor="middle"')}
    ${txt(310, 239, 'Match', 'text-anchor="middle"')}
    ${txt(40, 267, 'Match: Black + Fight,', 'font-size="20" style="fill:#606060"')}
    ${txt(40, 290, 'or Brown + Lie down.', 'font-size="20" style="fill:#606060"')}
    ${bars}
    ${txt(500, 224, status, 'font-size="20" style="fill:#606060"')}
    ${observations}${evidence}
    ${txt(40, 359, 'Observed survivors', 'font-size="20" style="fill:#606060"')}
    ${txt(500, 289, 'Next Brown + Lie down case')}
    ${txt(500, 346, pct(prediction), `font-size="52" style="fill:${C.teal}" font-variant-numeric="tabular-nums"`)}
    ${txt(702, 344, 'survival probability', 'font-size="20" style="fill:#606060"')}
    <line x1="40" y1="386" x2="860" y2="386" stroke="#d8dcde"/>
    <g opacity="${f(mixOpacity)}">${txt(450, 427, mixture, 'text-anchor="middle" font-size="22"')}</g>
  </g>`;
}

/** Compact equivalent for viewBox="0 0 400 330". */
export function renderBayesMobile(stage = 0, progress = 1) {
  stage = Math.max(0, Math.min(4, Math.round(Number(stage) || 0)));
  const raw = clamp(Number.isFinite(progress) ? progress : 1);
  const t = smooth(raw);
  const count = counts[stage];
  const prevCount = counts[Math.max(0, stage - 1)];
  const target = toyState(count);
  const previous = toyState(prevCount);
  const pa = lerp(previous.posterior[0], target.posterior[0], t);
  const pb = 1 - pa;
  const prediction = pa * MATCH_SURVIVAL + pb * MISMATCH_SURVIVAL;
  const updating = stage > 0 && stage < 4 && raw < .995;
  const status = updating
    ? `Adding ${count - prevCount} Brown + Lie down survivor${count - prevCount === 1 ? '' : 's'}`
    : count === 0 ? 'No observations yet' : `${count} survivor${count === 1 ? '' : 's'}: Brown + Lie down`;
  const mtext = (x, y, text, attrs = '') => txt(x, y, text, `${attrs.includes('font-size=') ? '' : 'font-size="16"'} ${attrs}`);
  const bars = [[pa, C.blue, 'A', 137], [pb, C.rust, 'B', 173]].map(([p, color, name, y]) => `
    ${mtext(20, y + 6, name, `style="fill:${color}"`)}
    <rect x="45" y="${y - 9}" width="250" height="19" fill="#f0f2f3" rx="2"/>
    <rect x="45" y="${y - 9}" width="${f(250 * p)}" height="19" fill="${color}" rx="2"/>
    ${mtext(380, y + 6, pct(p), 'text-anchor="end" font-variant-numeric="tabular-nums"')}
  `).join('');
  const observations = Array.from({length: 4}, (_, i) => {
    const existed = i < prevCount;
    const isNew = i >= prevCount && i < count;
    const local = existed ? 1 : isNew ? clamp((raw - (i - prevCount) * .14) / .7) : 0;
    const x = 31 + i * 36;
    const y = lerp(254, 237, smooth(local));
    return `<circle cx="${x}" cy="237" r="11" fill="none" stroke="#c8cdd0" stroke-dasharray="3 3"/>
      ${local > 0 ? `<g opacity="${f(local)}" transform="translate(${x} ${f(y)})"><circle r="11" fill="${C.teal}"/><path d="M-5 0 L-1 4 L6 -5" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></g>` : ''}`;
  }).join('');
  const flightT = clamp((raw - .12) / .72);
  const ex = lerp(178, 260, flightT);
  const ey = lerp(237, 190, flightT) - 12 * Math.sin(flightT * Math.PI);
  const evidence = stage > 0 && stage < 4 ? `
    <path d="M178 237 Q222 194 260 190" fill="none" stroke="#c6cbce" stroke-width="1.2" stroke-dasharray="3 4"/>
    <circle cx="${f(ex)}" cy="${f(ey)}" r="5" fill="${C.teal}" opacity="${f(Math.sin(Math.PI * flightT))}"/>
  ` : '';
  const mixture = stage === 4
    ? `${(pa * 100).toFixed(1)}% × ${pct(MATCH_SURVIVAL)} + ${(pb * 100).toFixed(1)}% × ${pct(MISMATCH_SURVIVAL)} ≈ ${(prediction * 100).toFixed(1)}%`
    : 'P(A | D) × p + P(B | D) × (1 − p)';
  return `<g font-family="inherit" data-bayes-mobile="true" data-bayes-stage="${stage}" data-bayes-count="${count}" data-posterior-a="${pa}" data-prediction="${prediction}">
    <title>Hypothetical bear survival rules and an exact Bayesian update</title>
    ${mtext(20, 18, 'P(survive)')}
    ${mtext(173, 18, 'Mismatch', 'text-anchor="middle"')}
    ${mtext(323, 18, 'Match', 'text-anchor="middle"')}
    ${mtext(20, 46, 'A', `style="fill:${C.blue}"`)}
    ${mtext(173, 46, pct(MISMATCH_SURVIVAL), `text-anchor="middle" style="fill:${C.blue}"`)}
    ${mtext(323, 46, pct(MATCH_SURVIVAL), `text-anchor="middle" style="fill:${C.blue}"`)}
    ${mtext(20, 74, 'B', `style="fill:${C.rust}"`)}
    ${mtext(173, 74, pct(MATCH_SURVIVAL), `text-anchor="middle" style="fill:${C.rust}"`)}
    ${mtext(323, 74, pct(MISMATCH_SURVIVAL), `text-anchor="middle" style="fill:${C.rust}"`)}
    <line x1="20" y1="87" x2="380" y2="87" stroke="#d8dcde"/>
    ${mtext(20, 111, count === 0 ? 'Prior probability' : 'Posterior probability')}
    ${bars}
    ${mtext(20, 211, status, 'font-size="15" style="fill:#606060"')}
    ${observations}${evidence}
    ${mtext(20, 279, 'Next Brown + Lie down', 'font-size="17"')}
    ${mtext(380, 280, pct(prediction), `text-anchor="end" font-size="29" style="fill:${C.teal}" font-variant-numeric="tabular-nums"`)}
    ${mtext(200, 314, mixture, 'text-anchor="middle" font-size="15"')}
  </g>`;
}
