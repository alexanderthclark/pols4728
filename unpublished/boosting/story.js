(function () {
  'use strict';
  const E = window.WorkedBoostingExample;
  const W = window.WorkedBoostingView;
  const $ = id => document.getElementById(id);
  const stage = document.querySelector('.stage-shell');
  const steps = [...document.querySelectorAll('.step')];
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  const mobile = () => window.matchMedia('(max-width: 800px)').matches;
  const exampleState = { rate: 1, process: 1, candidateIndex: null, round: 8 };
  let scene = -1;
  let animation = 0;
  let scrollPending = false;
  let navigationTarget = null;

  function readout(items) {
    $('vector-readout').replaceChildren(...items.map(([name, value, color]) => {
      const group = document.createElement('div');
      const title = document.createElement('span');
      const number = document.createElement('span');
      title.className = 'readout-label'; title.textContent = name;
      number.className = 'value'; number.textContent = value;
      if (color) number.style.color = color;
      group.append(title, number);
      return group;
    }));
  }

  function render(announce = true) {
    if (scene < 0) return;
    const bounds = $('visual-area').getBoundingClientRect();
    const state = { ...exampleState, width: bounds.width, height: bounds.height };
    $('worked-diagram').innerHTML = W.render(scene, state, mobile());
    readout(W.readout(scene, state));
    if (announce) $('stage-caption').textContent = W.presentation(scene, state).caption;
    const fitting = scene === 2 || scene === 4;
    const updating = scene === 3 || scene === 5;
    $('worked-controls').hidden = scene === 7;
    $('vector-readout').hidden = scene === 7;
    $('worked-cut-label').hidden = !fitting;
    $('worked-progress-label').hidden = !updating;
    $('worked-progress').hidden = !updating;
    $('worked-round-label').hidden = scene !== 6;
    $('worked-round').hidden = scene !== 6;
    $('worked-rate').value = String(exampleState.rate);
    $('worked-progress').value = String(exampleState.process);
    $('worked-progress-value').textContent = `${Math.round(100 * exampleState.process)}%`;
    $('worked-round').value = String(exampleState.round);
    $('worked-round-value').textContent = String(exampleState.round);
    if (fitting) {
      const fitted = E.build(exampleState.rate, 2)[scene === 2 ? 1 : 2];
      const winner = fitted.candidates.findIndex(row => row.threshold === fitted.tree.threshold);
      $('worked-cut').value = String(exampleState.candidateIndex === null ? winner : exampleState.candidateIndex);
    }
    stage.dataset.rate = exampleState.rate;
    stage.dataset.updateProgress = exampleState.process;
    stage.dataset.round = exampleState.round;
  }

  function cancelUpdate() {
    cancelAnimationFrame(animation);
    animation = 0;
  }

  function animateUpdate() {
    const start = performance.now();
    exampleState.process = reduced.matches ? 1 : 0;
    render();
    if (reduced.matches) return;
    const tick = now => {
      const t = Math.min(1, (now - start) / 850);
      exampleState.process = t * t * (3 - 2 * t);
      render(false);
      if (t < 1) animation = requestAnimationFrame(tick);
      else { animation = 0; render(); }
    };
    animation = requestAnimationFrame(tick);
  }

  function changeScene(index, animate = true) {
    index = Math.max(0, Math.min(steps.length - 1, index));
    if (index === scene) return;
    cancelUpdate();
    scene = index;
    stage.dataset.scene = scene;
    stage.dataset.example = 'worked';
    $('stage-topic').textContent = W.topics[scene];
    $('stage-count').textContent = `${String(scene + 1).padStart(2, '0')} / ${steps.length}`;
    $('previous').disabled = scene === 0;
    $('next').textContent = scene === steps.length - 1 ? 'Beyond squared error' : 'Next';
    steps.forEach((step, i) => step.toggleAttribute('data-active', i === scene));
    exampleState.candidateIndex = null;
    exampleState.process = 1;
    if (animate && (scene === 3 || scene === 5)) animateUpdate();
    else render();
  }

  function scrollUpdate() {
    scrollPending = false;
    // Keep an explicitly requested scene selected until its scroll arrives.
    if (navigationTarget && Math.abs(scrollY - navigationTarget.top) <= 2) navigationTarget = null;
    const threshold = mobile() ? stage.getBoundingClientRect().bottom + 65 : window.innerHeight * 0.52;
    let index = 0;
    steps.forEach((step, i) => { if (step.getBoundingClientRect().top <= threshold) index = i; });
    if (!navigationTarget) changeScene(index);
    const range = document.documentElement.scrollHeight - innerHeight;
    $('progress').style.width = `${range > 0 ? Math.min(100, Math.max(0, scrollY / range * 100)) : 0}%`;
  }

  function scrollToStep(index) {
    if (index >= steps.length) {
      navigationTarget = null;
      $('other-losses').scrollIntoView({ behavior: reduced.matches ? 'instant' : 'smooth' });
      return;
    }
    index = Math.max(0, index);
    changeScene(index);
    const offset = mobile() ? stage.getBoundingClientRect().height + 64 : 90;
    const requested = steps[index].getBoundingClientRect().top + scrollY - offset;
    // Browsers clamp scroll destinations to the available document range.
    const top = Math.max(0, Math.min(document.documentElement.scrollHeight - innerHeight, requested));
    navigationTarget = { top };
    window.scrollTo({ top, behavior: reduced.matches ? 'instant' : 'smooth' });
  }

  function stepForId(id) {
    return id === 'story' || id === 'reading' ? 0 : steps.findIndex(step => step.id === id);
  }

  $('previous').addEventListener('click', () => scrollToStep(scene - 1));
  $('next').addEventListener('click', () => scrollToStep(scene + 1));
  document.addEventListener('click', event => {
    if (event.defaultPrevented || event.button > 0 || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
    const link = event.target.closest?.('a[href^="#"]');
    if (!link) return;
    const href = link.getAttribute('href');
    const index = stepForId(href.slice(1));
    if (index < 0) { navigationTarget = null; return; }
    event.preventDefault();
    window.history?.pushState(null, '', href);
    scrollToStep(index);
  });
  document.addEventListener('keydown', event => {
    if (event.altKey || event.ctrlKey || event.metaKey || /^(INPUT|SELECT|TEXTAREA|BUTTON|A|SUMMARY)$/.test(event.target.tagName)) return;
    if (/^(PageUp|PageDown|Home|End|ArrowUp|ArrowDown|Escape| )$/.test(event.key)) navigationTarget = null;
    const rect = document.querySelector('.story-layout').getBoundingClientRect();
    if (rect.top > 90 || rect.bottom < innerHeight / 2) return;
    if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
      event.preventDefault(); scrollToStep(scene + (event.key === 'ArrowRight' ? 1 : -1));
    }
  });

  $('worked-rate').addEventListener('change', () => {
    cancelUpdate();
    exampleState.rate = Number($('worked-rate').value);
    exampleState.candidateIndex = null;
    exampleState.process = 1;
    render();
  });
  $('worked-cut').addEventListener('change', () => {
    cancelUpdate();
    exampleState.candidateIndex = Number($('worked-cut').value);
    render();
  });
  $('worked-progress').addEventListener('input', () => {
    cancelUpdate();
    exampleState.process = Number($('worked-progress').value);
    render();
  });
  $('worked-round').addEventListener('input', () => {
    cancelUpdate();
    exampleState.round = Math.max(0, Math.min(24, Math.round(Number($('worked-round').value))));
    render();
  });

  window.addEventListener('scroll', () => {
    if (!scrollPending) { scrollPending = true; requestAnimationFrame(scrollUpdate); }
  }, { passive: true });
  ['wheel', 'touchstart'].forEach(type => window.addEventListener(type, () => { navigationTarget = null; }, { passive: true }));
  window.addEventListener('scrollend', () => {
    // A cancelled earlier scroll may report completion during a new one.
    if (navigationTarget && Math.abs(scrollY - navigationTarget.top) <= 2) {
      navigationTarget = null;
      scrollUpdate();
    }
  });
  new ResizeObserver(() => render()).observe($('visual-area'));
  window.addEventListener('resize', () => { navigationTarget = null; scrollUpdate(); });
  reduced.addEventListener('change', () => {
    cancelUpdate();
    exampleState.process = 1;
    render();
  });
  window.addEventListener('popstate', () => {
    const index = stepForId(window.location?.hash?.slice(1) || 'story');
    if (index >= 0) scrollToStep(index);
  });

  changeScene(0, false);
  scrollUpdate();
  const initialId = window.location?.hash?.slice(1);
  if (initialId) {
    const initialStep = stepForId(initialId);
    if (initialStep >= 0) requestAnimationFrame(() => scrollToStep(initialStep));
  }
})();
