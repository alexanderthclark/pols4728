/**
 * Exact two-hypothesis teaching example, not a TabPFN prediction.
 * Given fixed high-interest features, labels are iid Bernoulli conditional
 * on a single shared world: A has p=.8 and B has p=.2.
 */
export function toyState(n, prior = [.5, .5]) {
  if (!Number.isSafeInteger(n) || n < 0) throw new RangeError('n must be a nonnegative safe integer.');
  if (!Array.isArray(prior) || prior.length !== 2 || prior.some(p => !Number.isFinite(p) || p < 0) || !(prior[0] + prior[1] > 0) || !Number.isFinite(prior[0] + prior[1])) {
    throw new RangeError('prior must contain two finite nonnegative weights with a finite positive sum.');
  }
  const sum = prior[0] + prior[1];
  const normalized = prior.map(p => p / sum);
  const logWeights = normalized.map((p, i) => p === 0 ? -Infinity : Math.log(p) + n * Math.log(i === 0 ? .8 : .2));
  const maxLog = Math.max(...logWeights);
  const weights = logWeights.map(p => Math.exp(p - maxLog));
  const total = weights[0] + weights[1];
  const posterior = weights.map(p => p / total);
  const highPrediction = posterior[0] * .8 + posterior[1] * .2;
  const lowPrediction = posterior[0] * .2 + posterior[1] * .8;
  return {
    n,
    prior: normalized,
    likelihood: [.8 ** n, .2 ** n],
    likelihoodRatio: 4 ** n,
    posterior,
    prediction: highPrediction,
    highPrediction,
    lowPrediction,
  };
}

const C = { ink: '#242424', blue: '#234E70', rust: '#A54F32', teal: '#007F73', muted: '#606060', light: '#e5e8e9' };
const counts = [0, 1, 2, 4, 4];
const clamp = n => Math.max(0, Math.min(1, n));
const lerp = (a, b, t) => a + (b - a) * t;
const smooth = t => t * t * (3 - 2 * t);
const f = n => Number(n.toFixed(3));
const pct = n => `${(n * 100).toFixed(n === .5 || n === .8 || n === .2 ? 0 : 1)}%`;
const txt = (x, y, text, attrs = '') => `<text x="${x}" y="${y}" fill="${C.ink}" ${attrs.includes('font-size=') ? '' : 'font-size="21"'} ${attrs}>${text}</text>`;

/**
 * Return SVG contents for viewBox="0 0 900 460".
 * Stages: 0 prior; 1 one voter; 2 two voters; 3 four voters;
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
  const prediction = pa * .8 + pb * .2;
  const updating = stage > 0 && stage < 4 && raw < .995;
  const count = counts[stage];
  const prevCount = counts[Math.max(0, stage - 1)];
  const status = updating ? `Updating with ${count - prevCount} new voter${count - prevCount === 1 ? '' : 's'}` : count === 0 ? 'Before seeing any voters' : `${count} high-interest voter${count === 1 ? '' : 's'}; all voted`;

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
    ? `${(pa * 100).toFixed(1)}% × 80% + ${(pb * 100).toFixed(1)}% × 20% = ${(prediction * 100).toFixed(1)}%`
    : `P(vote | high interest) = P(A | data) × 0.8 + P(B | data) × 0.2`;

  return `<g font-family="inherit" data-bayes-stage="${stage}" data-bayes-count="${count}" data-posterior-a="${pa}" data-prediction="${prediction}">
    ${txt(40, 28, 'Two possible turnout rules')}
    ${txt(500, 28, count === 0 ? 'Prior probability' : 'Posterior probability')}
    ${txt(40, 61, 'P(voted)', 'font-size="20" style="fill:#606060"')}
    <path d="M115 72 V213 H330" fill="none" stroke="#8c9398" stroke-width="1.2"/>
    ${txt(99, 92, '80%', 'text-anchor="end" font-size="20" style="fill:#606060"')}
    ${txt(99, 197, '20%', 'text-anchor="end" font-size="20" style="fill:#606060"')}
    <path d="M130 190 L310 85" fill="none" stroke="${C.blue}" stroke-width="3"/>
    <path d="M130 85 L310 190" fill="none" stroke="${C.rust}" stroke-width="3" stroke-dasharray="8 6"/>
    <circle cx="130" cy="190" r="5" fill="${C.blue}"/><circle cx="310" cy="85" r="5" fill="${C.blue}"/>
    <rect x="125" y="80" width="10" height="10" fill="${C.rust}"/><rect x="305" y="185" width="10" height="10" fill="${C.rust}"/>
    ${txt(329, 91, 'A', `style="fill:${C.blue}"`)}
    ${txt(329, 197, 'B', `style="fill:${C.rust}"`)}
    ${txt(130, 239, 'Low', 'text-anchor="middle"')}
    ${txt(310, 239, 'High', 'text-anchor="middle"')}
    ${txt(220, 267, 'Political interest', 'text-anchor="middle" font-size="20" style="fill:#606060"')}
    ${bars}
    ${txt(500, 224, status, 'font-size="20" style="fill:#606060"')}
    ${observations}${evidence}
    ${txt(40, 359, 'Observed voters', 'font-size="20" style="fill:#606060"')}
    ${txt(500, 289, 'Next high-interest voter')}
    ${txt(500, 346, pct(prediction), `font-size="52" style="fill:${C.teal}" font-variant-numeric="tabular-nums"`)}
    ${txt(702, 344, 'chance of voting', 'font-size="20" style="fill:#606060"')}
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
  const prediction = pa * .8 + pb * .2;
  const updating = stage > 0 && stage < 4 && raw < .995;
  const status = updating
    ? `Adding ${count - prevCount} high-interest voter${count - prevCount === 1 ? '' : 's'}`
    : count === 0 ? 'No observations yet' : `${count} high-interest voter${count === 1 ? '' : 's'}; all voted`;
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
    ? `${(pa * 100).toFixed(1)}% × 80% + ${(pb * 100).toFixed(1)}% × 20% = ${(prediction * 100).toFixed(1)}%`
    : 'P(A | data) × 80% + P(B | data) × 20%';
  return `<g font-family="inherit" data-bayes-mobile="true" data-bayes-stage="${stage}" data-bayes-count="${count}" data-posterior-a="${pa}" data-prediction="${prediction}">
    ${mtext(20, 18, 'P(voted)')}
    ${mtext(173, 18, 'Low interest', 'text-anchor="middle"')}
    ${mtext(323, 18, 'High interest', 'text-anchor="middle"')}
    ${mtext(20, 46, 'A', `style="fill:${C.blue}"`)}
    ${mtext(173, 46, '20%', `text-anchor="middle" style="fill:${C.blue}"`)}
    ${mtext(323, 46, '80%', `text-anchor="middle" style="fill:${C.blue}"`)}
    ${mtext(20, 74, 'B', `style="fill:${C.rust}"`)}
    ${mtext(173, 74, '80%', `text-anchor="middle" style="fill:${C.rust}"`)}
    ${mtext(323, 74, '20%', `text-anchor="middle" style="fill:${C.rust}"`)}
    <line x1="20" y1="87" x2="380" y2="87" stroke="#d8dcde"/>
    ${mtext(20, 111, count === 0 ? 'Prior probability' : 'Posterior probability')}
    ${bars}
    ${mtext(20, 211, status, 'font-size="15" style="fill:#606060"')}
    ${observations}${evidence}
    ${mtext(20, 279, 'Next high-interest voter', 'font-size="17"')}
    ${mtext(380, 280, pct(prediction), `text-anchor="end" font-size="29" style="fill:${C.teal}" font-variant-numeric="tabular-nums"`)}
    ${mtext(200, 314, mixture, 'text-anchor="middle" font-size="15"')}
  </g>`;
}
