const ENTER_AT = .12;
const FINISH_AT = .82;
const clamp = (value, lower, upper) => Math.max(lower, Math.min(upper, value));

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
