(function () {
  'use strict';

  const art = document.getElementById('shap-title-picture');
  const cover = document.getElementById('beginning');
  const fallback = document.getElementById('shap-title-fallback');
  const picture = window.SHAPTitlePicture;
  if (!art || !cover || typeof picture?.render !== 'function') return;
  window.SHAPTitleMotion?.destroy();

  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  const masthead = document.querySelector('.masthead');
  const content = cover.querySelector?.('.title-page-content');
  const listeners = [];
  let elapsed = 0;
  let progress = reduced.matches ? 1 : 0;
  let visible = false;
  let frame = null;
  let lastTime = null;
  let observer = null;
  let pagePaused = false;
  let destroyed = false;
  let failed = false;
  let activated = false;
  let painted = null;

  function listen(target, type, callback, options) {
    target.addEventListener(type, callback, options);
    listeners.push(function () { target.removeEventListener(type, callback, options); });
  }
  function geometry() {
    const headerHeight = masthead?.getBoundingClientRect().height || 0;
    const viewportHeight = Math.max(1, window.innerHeight || document.documentElement?.clientHeight || 1);
    return { headerHeight, viewportHeight };
  }
  function updateProgress() {
    if (reduced.matches) { progress = 1; return; }
    const { headerHeight, viewportHeight } = geometry();
    const top = cover.getBoundingClientRect().top;
    // CSS uses the small viewport height for the sticky content. Measuring it
    // keeps hatch travel aligned when mobile browser chrome changes innerHeight.
    const contentHeight = content?.getBoundingClientRect().height;
    const availableHeight = contentHeight > 0 ? contentHeight : viewportHeight - headerHeight;
    const travel = cover.offsetHeight - Math.max(1, availableHeight);
    // A missing scroll track must still yield a finite closed/open state.
    progress = travel > 0
      ? Math.max(0, Math.min(1, (headerHeight - top) / travel))
      : top < headerHeight ? 1 : 0;
  }
  function updateFallbackVisibility() {
    if (observer) return;
    const rect = art.getBoundingClientRect();
    const { headerHeight, viewportHeight } = geometry();
    visible = rect.bottom > headerHeight && rect.top < viewportHeight;
  }
  function stop() {
    if (frame !== null) window.cancelAnimationFrame(frame);
    frame = null;
    lastTime = null;
  }
  function fail() {
    failed = true;
    stop();
    cover.classList.remove('has-title-animation');
    if (fallback) { fallback.hidden = false; art.hidden = true; }
  }
  function paint() {
    if (destroyed || failed) return false;
    const drawElapsed = reduced.matches ? 0 : elapsed;
    const key = `${drawElapsed}:${progress}:${reduced.matches}`;
    if (painted === key) return true;
    try {
      const markup = picture.render(drawElapsed, progress, reduced.matches);
      if (typeof markup !== 'string' || !markup.trim()) { fail(); return false; }
      art.innerHTML = markup;
      painted = key;
      if (!activated) {
        activated = true;
        art.hidden = false;
        if (fallback) fallback.hidden = true;
        cover.classList.add('has-title-animation');
        // Activating the sticky cover changes its travel distance. Read that
        // geometry before the first visible frame, including restored scrolls.
        const before = progress;
        updateProgress();
        if (progress !== before) return paint();
      }
      return true;
    } catch (_) { fail(); return false; }
  }
  function canPlay() {
    return !destroyed && !failed && !pagePaused && visible && !document.hidden && !reduced.matches;
  }
  function tick(time) {
    frame = null;
    if (!canPlay()) { lastTime = null; return; }
    if (lastTime !== null) elapsed += Math.max(0, time - lastTime);
    lastTime = time;
    updateProgress();
    if (paint() && canPlay()) frame = window.requestAnimationFrame(tick);
  }
  function sync() {
    if (destroyed || failed) return;
    updateProgress();
    updateFallbackVisibility();
    if (!canPlay()) {
      stop();
      // Reduced motion has one complete, open pose. Hidden tabs and offscreen
      // art retain their last DOM until they become visible again.
      if (reduced.matches && !document.hidden && !pagePaused) paint();
    } else if (frame === null) frame = window.requestAnimationFrame(tick);
  }
  function destroy() {
    if (destroyed) return;
    destroyed = true;
    stop();
    observer?.disconnect();
    for (const remove of listeners) remove();
  }

  window.SHAPTitleMotion = {
    destroy,
    refresh: sync,
    state: function () {
      return { elapsedMs: elapsed, openProgress: progress, visible,
        reducedMotion: reduced.matches, running: frame !== null, destroyed, failed };
    },
  };

  updateProgress();
  if (!paint()) return;
  if ('IntersectionObserver' in window) {
    observer = new window.IntersectionObserver(function (entries) {
      visible = entries.some(function (entry) { return entry.isIntersecting; });
      sync();
    }, { threshold: 0 });
    observer.observe(art);
  }
  listen(window, 'scroll', sync, { passive: true });
  listen(window, 'resize', sync);
  listen(window, 'hashchange', sync);
  listen(document, 'visibilitychange', sync);
  listen(reduced, 'change', function () { stop(); sync(); });
  listen(window, 'pagehide', function () { pagePaused = true; stop(); });
  listen(window, 'pageshow', function () { pagePaused = false; sync(); });
  sync();
})();
