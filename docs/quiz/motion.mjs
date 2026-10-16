/** Small, dependency-free timing helpers shared by the lesson's SVG scenes. */
export const clamp01 = value => Math.min(1, Math.max(0, value));
export const lerp = (start, end, progress) => start + (end - start) * progress;

/** Cubic smoothstep: a symmetric easing with zero slope at either endpoint. */
export function easeInOut(progress) {
  const p = clamp01(progress);
  return p * p * (3 - 2 * p);
}

/** Progress within one interval; an empty interval is an instantaneous step. */
export function phase(progress, start = 0, end = 1) {
  if (end < start) throw new RangeError('phase end must be at least its start.');
  return end === start ? Number(progress >= end) : clamp01((progress - start) / (end - start));
}

/**
 * A one-shot timeline. All times are milliseconds on the injected now() clock.
 * onState receives a fresh { status, progress } snapshot whenever the timeline
 * renders or changes status. Visibility and reduced-motion policy belong to the
 * caller. No frame API is needed to construct, cancel, or finish a timeline.
 */
export function createTimeline({
  duration = 5000,
  render,
  onState = () => {},
  requestFrame = globalThis.requestAnimationFrame,
  cancelFrame = globalThis.cancelAnimationFrame,
  now = () => performance.now(),
} = {}) {
  if (!Number.isFinite(duration) || duration < 0) {
    throw new RangeError('duration must be finite and nonnegative.');
  }
  if (typeof render !== 'function') throw new TypeError('render must be a function.');
  if (typeof onState !== 'function') throw new TypeError('onState must be a function.');
  if (typeof now !== 'function') throw new TypeError('now must be a function.');

  let status = 'idle';
  let progress = 0;
  let originProgress = 0;
  let startedAt = 0;
  let frameId = null;
  let generation = 0;

  const getState = () => ({ status, progress });
  const notify = () => onState(getState());
  const draw = () => { render(progress); notify(); };

  function invalidate() {
    generation += 1;
    if (frameId !== null && typeof cancelFrame === 'function') cancelFrame(frameId);
    frameId = null;
  }

  function currentProgress() {
    // An adjustment to the clock cannot send an already rendered scene back.
    return Math.max(progress, clamp01(originProgress + (now() - startedAt) / duration));
  }

  function requireFrames() {
    if (typeof requestFrame !== 'function') {
      throw new TypeError('requestFrame must be a function to play the timeline.');
    }
  }

  function schedule(token) {
    frameId = requestFrame(() => {
      if (token !== generation || status !== 'playing') return;
      frameId = null;
      progress = currentProgress();
      if (progress === 1) status = 'complete';
      draw();
      if (token === generation && status === 'playing') schedule(token);
    });
  }

  function play({ from = 0 } = {}) {
    if (!Number.isFinite(from)) throw new RangeError('from must be finite.');
    const initial = clamp01(from);
    const complete = duration === 0 || initial === 1;
    if (!complete) requireFrames();
    invalidate();
    progress = complete ? 1 : initial;
    originProgress = progress;
    if (!complete) startedAt = now();
    status = complete ? 'complete' : 'playing';
    const token = generation;
    draw();
    if (token === generation && status === 'playing') schedule(token);
    return getState();
  }

  function pause() {
    if (status !== 'playing') return getState();
    invalidate();
    progress = currentProgress();
    status = progress === 1 ? 'complete' : 'paused';
    draw();
    return getState();
  }

  function resume() {
    if (status !== 'paused') return getState();
    requireFrames();
    invalidate();
    originProgress = progress;
    startedAt = now();
    status = 'playing';
    const token = generation;
    draw();
    if (token === generation && status === 'playing') schedule(token);
    return getState();
  }

  function finish() {
    if (status === 'complete') return getState();
    invalidate();
    progress = 1;
    status = 'complete';
    draw();
    return getState();
  }

  function cancel() {
    if (status === 'cancelled') return getState();
    invalidate();
    status = 'cancelled';
    notify();
    return getState();
  }

  return { play, pause, resume, finish, cancel, getState };
}
