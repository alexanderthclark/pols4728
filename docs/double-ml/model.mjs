/** The constructed tutoring example, with one stable record per student. */
const scoreLevels = [40, 50, 60, 70];

export const clamp = (value, minimum = 0, maximum = 1) =>
  Math.min(maximum, Math.max(minimum, value));

export const lerp = (from, to, progress) => from + (to - from) * clamp(progress);

export function smoothstep(progress) {
  const p = clamp(progress);
  return p * p * (3 - 2 * p);
}

export const rows = Object.freeze(scoreLevels.flatMap(x => {
  const high = x >= 60;
  const m = high ? 97.5 : 52.5;
  const e = high ? 0.75 : 0.25;
  return [0, 1].flatMap(d => {
    const count = (high ? d === 1 : d === 0) ? 6 : 2;
    return Array.from({ length: count }, (_, index) => {
      const y = 50 + 40 * Number(high) + 10 * d;
      return Object.freeze({
        id: `x${x}-d${d}-${index + 1}`,
        x, d, y, m, e,
        r: y - m,
        t: d - e,
        tier: high ? 'high' : 'low'
      });
    });
  });
}));

export const groups = Object.freeze(['low', 'high'].flatMap(tier =>
  [0, 1].map(d => {
    const members = rows.filter(row => row.tier === tier && row.d === d);
    const { y, m, e, r, t } = members[0];
    return Object.freeze({ id: `${tier}-d${d}`, tier, d, count: members.length, y, m, e, r, t });
  })
));

const mean = values => values.reduce((total, value) => total + value, 0) / values.length;

/** Means and the through-origin residual slope, using every student once. */
export function calculateSummary(students = rows) {
  const treated = students.filter(row => row.d === 1);
  const control = students.filter(row => row.d === 0);
  const treatedMean = mean(treated.map(row => row.y));
  const controlMean = mean(control.map(row => row.y));
  const treatedResidualMean = mean(treated.map(row => row.r));
  const controlResidualMean = mean(control.map(row => row.r));
  const sumTR = students.reduce((total, row) => total + row.t * row.r, 0);
  const sumT2 = students.reduce((total, row) => total + row.t ** 2, 0);
  return {
    n: students.length,
    treatedCount: treated.length,
    controlCount: control.length,
    treatedMean,
    controlMean,
    rawGap: treatedMean - controlMean,
    treatedResidualMean,
    controlResidualMean,
    outcomeOnlyGap: treatedResidualMean - controlResidualMean,
    sumTR,
    sumT2,
    slope: sumTR / sumT2
  };
}

export const summary = Object.freeze(calculateSummary());

/** Intermediate coordinates for moving the same students, never resampling. */
export function residualizedRows(progressY = 1, progressD = 1) {
  const py = clamp(progressY);
  const pd = clamp(progressD);
  return rows.map(row => ({ ...row, r: row.y - py * row.m, t: row.d - pd * row.e }));
}

/** Reduced motion replaces continuous movement with two meaningful poses. */
export function phase(progress, reducedMotion = false) {
  const p = clamp(progress);
  return reducedMotion ? Number(p >= 0.5) : p;
}

/**
 * Find the section at a viewport reading line, without timers or direction state.
 * Steps use document coordinates: { top, height, id }. A section's progress is
 * zero when its top reaches the reading line and one after its full height has
 * passed. Gaps retain the previous completed pose. Scrolling back is symmetric.
 */
export function scrollState(scrollY, steps, viewportHeight, options = {}) {
  const { anchor = 0.55, reducedMotion = false } = options;
  if (!Number.isFinite(scrollY) || !Number.isFinite(viewportHeight) || viewportHeight <= 0) {
    throw new RangeError('Scroll position must be finite and viewport height positive.');
  }
  if (!steps.length) {
    return { index: -1, id: null, progress: 0, motionProgress: 0, overall: 0 };
  }
  const readingLine = scrollY + viewportHeight * clamp(anchor);
  let index = 0;
  for (let i = 0; i < steps.length; i += 1) {
    const step = steps[i];
    if (!Number.isFinite(step.top) || !Number.isFinite(step.height) || step.height <= 0) {
      throw new RangeError('Each step needs a finite top and positive height.');
    }
    if (step.top <= readingLine) index = i;
  }
  const step = steps[index];
  const progress = clamp((readingLine - step.top) / step.height);
  return {
    index,
    id: step.id ?? String(index),
    progress,
    motionProgress: phase(progress, reducedMotion),
    overall: (index + progress) / steps.length
  };
}
