(function () {
  'use strict';
  const E = window.WorkedBoostingExample;
  const W = window.WorkedBoostingView;
  const R = window.RestrictedBoostingView;
  const L = window.LossBoostingView;
  const $ = id => document.getElementById(id);
  const stage = document.querySelector('.stage-shell');
  const steps = [...document.querySelectorAll('.step')];
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  const mobile = () => window.matchMedia('(max-width: 800px)').matches;
  const exampleState = { rate: 1, process: 1, candidateIndex: null, round: 8 };
  const restrictedState = { prediction: 0.5, loss: 'squared', tau: 0.8, delta: 0.25, progress: 1 };
  const openingDurations = { 1: 1200, 2: 1400, 3: 1700, 4: 2600, 5: 3600, 6: 5000, 7: 1800, 8: 2600, 9: 18000 };
  const lossState = { loss: 'huber', dataset: 'outlier', delta: 1, tau: 0.8, rate: 0.5, round: 0 };
  let scene = -1;
  let animation = 0;
  let restrictedDestination = null;
  let scrollPending = false;
  let navigationTarget = null;
  let mobileLayout = mobile();

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

  function localScene() {
    const step = steps[scene];
    if (step.dataset.transitionScene !== undefined) return { transition: true, index: Number(step.dataset.transitionScene) };
    if (step.dataset.lossScene !== undefined) return { loss: true, index: Number(step.dataset.lossScene) };
    if (step.dataset.restrictedScene !== undefined) return { restricted: true, index: Number(step.dataset.restrictedScene) };
    return { worked: true, index: Number(step.dataset.workedScene) };
  }

  function render(announce = true) {
    if (scene < 0) return;
    const current = localScene();
    const index = current.index;
    stage.dataset.opening = String(Boolean(current.restricted || current.transition));
    $('stage-tools').hidden = Boolean(current.restricted || current.transition);
    $('worked-transition-diagram').hidden = !current.transition;
    $('opening-annotation').hidden = !current.restricted;
    $('opening-replay').hidden = !current.restricted || !openingDurations[index];
    // Controls change the available plot space, especially on a narrow screen.
    // Set their visibility before measuring the figure.
    $('restricted-controls').hidden = !current.restricted || index !== 11;
    $('restricted-diagram').hidden = !current.restricted;
    $('worked-diagram').hidden = !current.worked;
    $('worked-controls').hidden = !current.worked || index === 7;
    $('loss-diagram').hidden = !current.loss;
    $('loss-controls').hidden = !current.loss;
    const hideNumbers = Boolean(current.restricted || current.transition || (current.worked && index === 7));
    $('vector-readout').hidden = hideNumbers;
    $('numbers').hidden = hideNumbers;
    delete stage.dataset.transitionScene;
    if (current.transition) {
      stage.dataset.transitionScene = index;
      for (const key of ['lossScene', 'restrictedScene', 'sharedPrediction', 'descentPhase', 'workedScene', 'updateProgress', 'loss', 'rate', 'round']) delete stage.dataset[key];
      $('stage-caption').textContent = '';
      if (announce) $('opening-status').textContent = 'Scroll to continue';
      return;
    }
    if (current.loss) {
      $('loss-choice').value = lossState.loss;
      $('loss-dataset').value = lossState.dataset;
      $('loss-delta').value = String(lossState.delta);
      $('loss-tau').value = String(lossState.tau);
      $('loss-rate').value = String(lossState.rate);
      $('loss-round').value = String(lossState.round);
      $('loss-round-value').textContent = String(lossState.round);
      $('loss-delta-label').hidden = lossState.loss !== 'huber';
      $('loss-tau-label').hidden = lossState.loss !== 'expectile';
      const bounds = $('visual-area').getBoundingClientRect();
      const state = { ...lossState, width: bounds.width, height: bounds.height };
      $('loss-diagram').innerHTML = L.render(index, state, mobile());
      readout(L.readout(index, state));
      if (announce) $('stage-caption').textContent = L.presentation(index, state).caption;
      stage.dataset.lossScene = index;
      stage.dataset.loss = lossState.loss;
      stage.dataset.rate = lossState.rate;
      stage.dataset.round = lossState.round;
      delete stage.dataset.restrictedScene;
      delete stage.dataset.sharedPrediction;
      delete stage.dataset.descentPhase;
      delete stage.dataset.workedScene;
      delete stage.dataset.updateProgress;
      return;
    }
    delete stage.dataset.lossScene;
    delete stage.dataset.loss;
    if (current.restricted) {
      document.querySelectorAll('[data-restricted-loss]').forEach(button => {
        button.setAttribute('aria-pressed', String(button.dataset.restrictedLoss === restrictedState.loss));
      });
      const bounds = $('visual-area').getBoundingClientRect();
      const state = { ...restrictedState, width: bounds.width, height: bounds.height };
      $('restricted-diagram').innerHTML = R.render(index, state, mobile());
      $('stage-caption').textContent = R.presentation(index, state).caption;
      const resolved = R.resolve(index, state);
      const annotation = R.annotation(index, state);
      const note = $('opening-annotation');
      note.dataset.lossEquation = String(Boolean(annotation.equation));
      const escape = value => String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
      const roundLabel = index === 9 ? `<span class="opening-round">${resolved.phase === 'complete' ? '6 updates complete' : `Round ${resolved.round + 1} of 6`}</span>` : '';
      note.innerHTML = `${roundLabel}${annotation.caption ? `<p>${escape(annotation.caption)}</p>` : ''}${annotation.equation ? `<div class="math">${annotation.equation}</div>` : ''}`;
      const p = restrictedState.progress;
      const opacity = index === 6 ? Math.max(0, Math.min(1, p / .12, (1 - p) / .18)) : 1;
      note.style.opacity = String(opacity);
      note.style.transform = `translateY(${(1 - opacity) * 14}px)`;
      note.hidden = index === 6 && p === 1;
      if (announce) $('opening-status').textContent = annotation.status;
      stage.dataset.rate = resolved.learningRate ?? 1;
      stage.dataset.sharedPrediction = resolved.prediction[0];
      stage.dataset.loss = resolved.lossSpec.key;
      stage.dataset.restrictedScene = index;
      delete stage.dataset.workedScene;
      if (resolved.round !== undefined) stage.dataset.round = resolved.round;
      else delete stage.dataset.round;
      if (resolved.phase) stage.dataset.descentPhase = resolved.phase;
      else delete stage.dataset.descentPhase;
      stage.dataset.updateProgress = restrictedState.progress;
      return;
    }
    delete stage.dataset.restrictedScene;
    delete stage.dataset.sharedPrediction;
    delete stage.dataset.descentPhase;
    const fitting = index === 2 || index === 4;
    const updating = index === 3 || index === 5;
    $('worked-cut-label').hidden = !fitting;
    $('worked-progress-label').hidden = !updating;
    $('worked-progress').hidden = !updating;
    $('worked-round-label').hidden = index !== 6;
    $('worked-round').hidden = index !== 6;
    $('worked-rate').value = String(exampleState.rate);
    $('worked-progress').value = String(exampleState.process);
    $('worked-progress-value').textContent = `${Math.round(100 * exampleState.process)}%`;
    $('worked-round').value = String(exampleState.round);
    $('worked-round-value').textContent = String(exampleState.round);
    if (fitting) {
      const fitted = E.build(exampleState.rate, 2)[index === 2 ? 1 : 2];
      const winner = fitted.candidates.findIndex(row => row.threshold === fitted.tree.threshold);
      $('worked-cut').value = String(exampleState.candidateIndex === null ? winner : exampleState.candidateIndex);
    }
    const bounds = $('visual-area').getBoundingClientRect();
    const state = { ...exampleState, width: bounds.width, height: bounds.height };
    $('worked-diagram').innerHTML = W.render(index, state, mobile());
    readout(W.readout(index, state));
    if (announce) $('stage-caption').textContent = W.presentation(index, state).caption;
    stage.dataset.rate = exampleState.rate;
    stage.dataset.updateProgress = exampleState.process;
    stage.dataset.round = exampleState.round;
    stage.dataset.workedScene = index;
  }

  function cancelUpdate() {
    $('stage-caption').setAttribute('aria-live', 'polite');
    cancelAnimationFrame(animation);
    animation = 0;
    restrictedDestination = null;
  }

  function animateUpdate() {
    const start = performance.now();
    exampleState.process = reduced.matches ? 1 : 0;
    render();
    if (reduced.matches) return;
    const tick = now => {
      const t = Math.max(0, Math.min(1, (now - start) / 850));
      exampleState.process = t * t * (3 - 2 * t);
      render(false);
      if (t < 1) animation = requestAnimationFrame(tick);
      else { animation = 0; render(); }
    };
    animation = requestAnimationFrame(tick);
  }

  function animateRestrictedUpdate() {
    cancelUpdate();
    const current = localScene();
    const duration = openingDurations[current.index];
    if (!current.restricted || !duration) return;
    const start = performance.now();
    restrictedState.progress = reduced.matches ? 1 : 0;
    $('stage-caption').setAttribute('aria-live', reduced.matches ? 'polite' : 'off');
    render();
    if (reduced.matches) return;
    const initial = R.resolve(current.index, restrictedState);
    let previousPhase = `${initial.round ?? ''}:${initial.phase ?? ''}`;
    const tick = now => {
      const t = Math.max(0, Math.min(1, (now - start) / duration));
      // Keep repeated rounds and their update phases evenly timed.
      restrictedState.progress = current.index === 9 ? t : t * t * (3 - 2 * t);
      const resolved = R.resolve(current.index, restrictedState);
      const phase = `${resolved.round ?? ''}:${resolved.phase ?? ''}`;
      const announce = current.index === 9 && phase !== previousPhase;
      previousPhase = phase;
      if (t < 1) { render(announce); animation = requestAnimationFrame(tick); }
      else { animation = 0; $('stage-caption').setAttribute('aria-live', 'polite'); render(); }
    };
    animation = requestAnimationFrame(tick);
  }

  function animateRestrictedPrediction(destination) {
    cancelUpdate();
    const from = restrictedState.prediction, start = performance.now();
    if (reduced.matches) { restrictedState.prediction = destination; render(); return; }
    restrictedDestination = destination;
    const tick = now => {
      const t = Math.max(0, Math.min(1, (now - start) / 1600)), eased = t * t * (3 - 2 * t);
      restrictedState.prediction = from + (destination - from) * eased;
      render(t === 1);
      if (t < 1) animation = requestAnimationFrame(tick);
      else { animation = 0; restrictedDestination = null; }
    };
    animation = requestAnimationFrame(tick);
  }

  function changeScene(index, animate = true) {
    index = Math.max(0, Math.min(steps.length - 1, index));
    if (index === scene) return;
    cancelUpdate();
    scene = index;
    stage.dataset.scene = scene;
    const current = localScene();
    stage.dataset.example = current.transition ? 'transition' : current.loss ? 'loss' : current.restricted ? 'restricted' : 'worked';
    $('stage-topic').textContent = current.transition ? 'From two observations to four' : (current.loss ? L : current.restricted ? R : W).topics[current.index];
    $('stage-count').textContent = `${String(scene + 1).padStart(2, '0')} / ${steps.length}`;
    $('previous').disabled = scene === 0;
    $('next').textContent = scene === steps.length - 1 ? 'Beyond squared error' : 'Next';
    steps.forEach((step, i) => step.toggleAttribute('data-active', i === scene));
    if (current.worked) {
      exampleState.candidateIndex = null;
      exampleState.process = 1;
      if (animate && (current.index === 3 || current.index === 5)) animateUpdate();
      else render();
    } else if (current.restricted) {
      restrictedState.progress = 1;
      if (animate && openingDurations[current.index]) animateRestrictedUpdate();
      else render();
    } else render();
  }

  function scrollUpdate() {
    scrollPending = false;
    // Keep an explicitly requested scene selected until its scroll arrives.
    if (navigationTarget && Math.abs(scrollY - navigationTarget.top) <= 2) navigationTarget = null;
    const stageBounds = stage.getBoundingClientRect();
    // The figure stays pinned at every size; entering a card starts its reveal.
    if (!navigationTarget) {
      const threshold = stageBounds.bottom + 32;
      let index = 0;
      steps.forEach((step, i) => { if (step.getBoundingClientRect().top <= threshold) index = i; });
      changeScene(index);
    }
    const range = document.documentElement.scrollHeight - innerHeight;
    $('progress').style.width = `${range > 0 ? Math.min(100, Math.max(0, scrollY / range * 100)) : 0}%`;
  }

  function scrollToStep(index, instant = false) {
    if (index >= steps.length) {
      navigationTarget = null;
      $('other-losses').scrollIntoView({ behavior: reduced.matches ? 'instant' : 'smooth' });
      return;
    }
    index = Math.max(0, index);
    changeScene(index);
    const bounds = stage.getBoundingClientRect();
    const requested = steps[index].getBoundingClientRect().top + scrollY - bounds.height - 74;
    // Browsers clamp scroll destinations to the available document range.
    const top = Math.max(0, Math.min(document.documentElement.scrollHeight - innerHeight, requested));
    navigationTarget = { top };
    window.scrollTo({ top, behavior: instant || reduced.matches ? 'instant' : 'smooth' });
  }

  function stepForId(id) {
    if (id === 'vector-radius') id = 'identical-inputs';
    if (id === 'vector-geometry' || id === 'vector-correction') id = 'worked-transition';
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

  ['loss-choice', 'loss-dataset', 'loss-delta', 'loss-tau', 'loss-rate'].forEach(id => {
    $(id).addEventListener('change', () => {
      const key = { 'loss-choice': 'loss', 'loss-dataset': 'dataset', 'loss-delta': 'delta', 'loss-tau': 'tau', 'loss-rate': 'rate' }[id];
      lossState[key] = key === 'loss' || key === 'dataset' ? $(id).value : Number($(id).value);
      render();
    });
  });
  $('loss-round').addEventListener('input', () => {
    lossState.round = Math.max(0, Math.min(24, Math.round(Number($('loss-round').value))));
    render();
  });

  $('opening-replay').addEventListener('click', animateRestrictedUpdate);
  $('restricted-fit').addEventListener('click', () => {
    animateRestrictedPrediction(R.resolve(11, restrictedState).nextPrediction[0]);
  });
  $('restricted-reset').addEventListener('click', () => {
    animateRestrictedPrediction(0);
  });
  $('restricted-mean').addEventListener('click', () => {
    animateRestrictedPrediction(0.5);
  });
  document.querySelectorAll('[data-restricted-loss]').forEach(button => {
    button.addEventListener('click', () => {
      cancelUpdate();
      restrictedState.loss = button.dataset.restrictedLoss;
      render();
    });
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
  $('numbers').addEventListener('toggle', () => render());
  new ResizeObserver(() => render()).observe($('visual-area'));
  window.addEventListener('resize', () => {
    const nextLayout = mobile();
    if (nextLayout !== mobileLayout) {
      mobileLayout = nextLayout;
      render();
      if (stage.getBoundingClientRect().top < innerHeight - 80) scrollToStep(scene, true);
      else { navigationTarget = null; scrollUpdate(); }
    } else {
      navigationTarget = null;
      scrollUpdate();
    }
  });
  reduced.addEventListener('change', () => {
    const wasAnimating = animation !== 0;
    const destination = restrictedDestination;
    cancelUpdate();
    exampleState.process = 1;
    if (wasAnimating && localScene().restricted) restrictedState.progress = 1;
    if (destination !== null) restrictedState.prediction = destination;
    render();
  });
  window.addEventListener('popstate', () => {
    const index = stepForId(window.location?.hash?.slice(1) || 'story');
    if (index >= 0) scrollToStep(index);
  });
  window.addEventListener('hashchange', () => {
    const id = window.location?.hash?.slice(1) || 'story';
    const index = stepForId(id);
    if (index < 0) return;
    changeScene(index);
    // Same-document navigation applies a native fragment scroll after history
    // handling. Correct that position once it has settled.
    requestAnimationFrame(() => {
      if (scene === index && (window.location?.hash?.slice(1) || 'story') === id) {
        scrollToStep(index, true);
      }
    });
  });

  changeScene(0, false);
  scrollUpdate();
  const initialId = window.location?.hash?.slice(1);
  if (initialId) {
    const initialStep = stepForId(initialId);
    if (initialStep >= 0) {
      requestAnimationFrame(() => scrollToStep(initialStep, true));
      // Native fragment restoration can run after the deferred controller.
      // Restore the plot-aware position once loading finishes, unless the
      // reader has already navigated to another scene or another hash.
      if (document.readyState !== 'complete') window.addEventListener('load', () => {
        requestAnimationFrame(() => {
          if (scene === initialStep && window.location?.hash?.slice(1) === initialId) {
            scrollToStep(initialStep, true);
          }
        });
      }, { once: true });
    }
  }
})();
