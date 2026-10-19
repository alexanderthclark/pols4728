/** Pure scroll geometry shared by the controller and its regression tests. */
export const clamp01 = value => Math.min(1, Math.max(0, value));
export const lerp = (start, end, progress) => start + (end - start) * progress;
export function easeInOut(progress) {
  const p = clamp01(progress);
  return p * p * (3 - 2 * p);
}
export function phase(progress, start = 0, end = 1) {
  if (end < start) throw new RangeError('phase end must be at least its start.');
  return end === start ? Number(progress >= end) : clamp01((progress - start) / (end - start));
}
/** First 72% constructs the figure; the rest holds its completed calculation. */
export function sceneProgress(top, height, anchor, reducedMotion = false) {
  const value = clamp01((anchor - top) / Math.max(1, height * .72));
  // Logical, reversible steps preserve the mechanism without continuous motion.
  return reducedMotion ? Math.floor(value * 5 + 1e-8) / 5 : value;
}
export function activeScene(tops, anchor) {
  let index = 0;
  for (let i = 0; i < tops.length; i++) {
    if (tops[i] > anchor) break;
    index = i;
  }
  return index;
}
