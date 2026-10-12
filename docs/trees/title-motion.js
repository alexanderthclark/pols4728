(function () {
  'use strict';
  const art = document.getElementById('title-picture');
  const controls = document.getElementById('title-motion-controls');
  const toggle = document.getElementById('title-motion-toggle');
  const replay = document.getElementById('title-motion-replay');
  const picture = window.TitlePicture;
  if (!art || !controls || !toggle || !replay || !picture) return;

  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  let elapsed = reduced.matches ? picture.duration : 0;
  let manuallyPaused = false;
  let visible = false;
  let frame = null;
  let lastTime = null;

  function paint() {
    art.innerHTML = picture.render(elapsed);
  }
  function updateControls() {
    const finished = elapsed >= picture.duration;
    controls.hidden = reduced.matches;
    toggle.hidden = finished;
    toggle.textContent = manuallyPaused ? 'Resume animation' : 'Pause animation';
    replay.hidden = !finished && !manuallyPaused;
  }
  function stop() {
    if (frame !== null) window.cancelAnimationFrame(frame);
    frame = null;
    lastTime = null;
  }
  function canPlay() {
    return visible && !document.hidden && !manuallyPaused && !reduced.matches && elapsed < picture.duration;
  }
  function tick(time) {
    frame = null;
    if (!canPlay()) { lastTime = null; return; }
    if (lastTime !== null) elapsed = Math.min(picture.duration, elapsed + Math.max(0, time - lastTime));
    lastTime = time;
    paint();
    if (elapsed >= picture.duration) { stop(); updateControls(); }
    else frame = window.requestAnimationFrame(tick);
  }
  function sync() {
    if (!canPlay()) stop();
    else if (frame === null) frame = window.requestAnimationFrame(tick);
    updateControls();
  }
  toggle.addEventListener('click', function () {
    manuallyPaused = !manuallyPaused;
    sync();
  });
  replay.addEventListener('click', function () {
    elapsed = 0;
    manuallyPaused = false;
    stop();
    paint();
    sync();
    toggle.focus();
  });
  document.addEventListener('visibilitychange', sync);
  reduced.addEventListener('change', function () {
    if (reduced.matches) { elapsed = picture.duration; stop(); paint(); }
    sync();
  });
  paint();
  updateControls();
  if ('IntersectionObserver' in window) {
    const observer = new window.IntersectionObserver(function (entries) {
      visible = entries.some(function (entry) { return entry.isIntersecting; });
      sync();
    }, { threshold: 0.15 });
    observer.observe(art);
  } else {
    visible = true;
    sync();
  }
})();
