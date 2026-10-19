const ENTER_AT = .12;
const FINISH_AT = .82;
const clamp = (value, lower, upper) => Math.max(lower, Math.min(upper, value));

/** Find the local scroll position that preserves an eased animation progress. */
export function scrollLocalAtProgress(progress) {
  if (!Number.isFinite(progress)) throw new RangeError('Animation progress must be finite.');
  if (progress <= 0) return ENTER_AT;
  if (progress >= 1) return FINISH_AT;

  // Smoothstep is monotone, so its inverse can be found without the loss of
  // precision that a trigonometric inverse introduces near the endpoints.
  let lower = 0;
  let upper = 1;
  for (let iteration = 0; iteration < 60; iteration++) {
    const phase = (lower + upper) / 2;
    const eased = phase * phase * (3 - 2 * phase);
    if (eased < progress) lower = phase;
    else upper = phase;
  }
  return ENTER_AT + (lower + upper) / 2 * (FINISH_AT - ENTER_AT);
}

/** Map scroll distance within equally spaced scenes to a reversible animation. */
export function scrollSceneAt(scrollOffset, sceneHeight, count) {
  if (!Number.isFinite(scrollOffset)) throw new RangeError('Scroll offset must be finite.');
  if (!Number.isFinite(sceneHeight) || sceneHeight <= 0) {
    throw new RangeError('Scene height must be finite and positive.');
  }
  if (!Number.isSafeInteger(count) || count <= 0) {
    throw new RangeError('Scene count must be a positive safe integer.');
  }

  const position = clamp(scrollOffset / sceneHeight, 0, count);
  const index = Math.min(count - 1, Math.floor(position));
  const local = clamp(position - index, 0, 1);
  const phase = clamp((local - ENTER_AT) / (FINISH_AT - ENTER_AT), 0, 1);
  const progress = phase * phase * (3 - 2 * phase);
  return { index, progress, local };
}
