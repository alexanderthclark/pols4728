/* Pure selection and scroll fades for the stationary narrative panel. */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.NNStoryNarrative = api;
})(typeof window !== 'undefined' ? window : globalThis, function () {
  'use strict';

  const FADE = 0.08;
  const BIAS_CHANGE = 0.65;
  const clamp = value => Math.max(0, Math.min(1, value));
  const smooth = value => {
    const t = clamp(value);
    return t * t * (3 - 2 * t);
  };

  function cue(frame, options) {
    if (!frame || typeof frame.scene !== 'string' || !frame.scene) {
      throw new TypeError('A narrative cue requires a frame with a scene name.');
    }
    const progress = Number.isFinite(frame.progress) ? clamp(frame.progress) : 0;
    const affine = frame.scene === 'affine';
    // The rendered model determines the explanation, including reduced motion.
    const source = affine ? (frame.showBias ? 'bias-visible' : 'bias-omitted') : 'scene';
    const key = frame.scene + ':' + source;
    if (options && options.reduced) return { key, source, opacity: 1 };

    const start = affine && !frame.showBias ? BIAS_CHANGE : 0;
    const end = affine && frame.showBias ? BIAS_CHANGE : 1;
    // Keep the first explanation readable before the reader begins scrolling.
    const entering = affine && frame.showBias ? 1 : smooth((progress - start) / FADE);
    const leaving = smooth((end - progress) / FADE);
    const opacity = Math.round(Math.min(entering, leaving) * 1000000) / 1000000;
    return { key, source, opacity };
  }

  return Object.freeze({ cue });
});
