(function () {
  'use strict';
  const E = window.WorkedBoostingExample;
  const W = window.WorkedBoostingView;
  const G = window.PredictionBoostingView;
  const R = window.RestrictedBoostingView;
  const L = window.LossBoostingView;
  const $ = id => document.getElementById(id);
  const stage = document.querySelector('.stage-shell');
  const steps = [...document.querySelectorAll('.step')];
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  const mobile = () => window.matchMedia('(max-width: 800px)').matches;
  const exampleState = { rate: 1, process: 1, candidateIndex: null, round: 8 };
  const geometryState = { round: 0, rate: 0.5, progress: 1 };
  const restrictedState = { prediction: 0.5, loss: 'squared', tau: 0.8, delta: 0.25, progress: 0 };
  const lossState = { loss: 'huber', dataset: 'outlier', delta: 1, tau: 0.8, rate: 0.5, round: 0 };
  let scene = -1;
  let animation = 0;
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
    if (step.dataset.lossScene !== undefined) return { loss: true, geometry: false, restricted: false, index: Number(step.dataset.lossScene) };
    if (step.dataset.geometryScene !== undefined) return { loss: false, geometry: true, restricted: false, index: Number(step.dataset.geometryScene) };
    if (step.dataset.restrictedScene !== undefined) return { loss: false, geometry: false, restricted: true, index: Number(step.dataset.restrictedScene) };
    return { loss: false, geometry: false, restricted: false, index: Number(step.dataset.workedScene) };
  }

  function render(announce = true) {
    if (scene < 0) return;
    const current = localScene();
    const index = current.index;
    // Controls change the available plot space, especially on a narrow screen.
    // Set their visibility before measuring the figure.
    $('geometry-controls').hidden = !current.geometry;
    $('geometry-tree-controls').hidden = !current.geometry || index !== 1;
    $('prediction-diagram').hidden = !current.geometry;
    $('restricted-controls').hidden = !current.restricted;
    $('restricted-diagram').hidden = !current.restricted;
    $('worked-diagram').hidden = current.geometry || current.restricted || current.loss;
    $('worked-controls').hidden = current.geometry || current.restricted || current.loss || index === 7;
    $('loss-diagram').hidden = !current.loss;
    $('loss-controls').hidden = !current.loss;
    const hideNumbers = !current.geometry && !current.restricted && !current.loss && index === 7;
    $('vector-readout').hidden = hideNumbers;
    $('numbers').hidden = hideNumbers;
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
      delete stage.dataset.geometryScene;
      delete stage.dataset.restrictedScene;
      delete stage.dataset.sharedPrediction;
      delete stage.dataset.workedScene;
      delete stage.dataset.updateProgress;
      return;
    }
    delete stage.dataset.lossScene;
    delete stage.dataset.loss;
    if (current.restricted) {
      $('restricted-loss').value = restrictedState.loss;
      $('restricted-loss-equation').innerHTML = R.equation(restrictedState, index);
      $('restricted-progression-controls').hidden = ![1, 3, 4].includes(index);
      $('restricted-sandbox-controls').hidden = index !== 5;
      $('restricted-progress').value = String(restrictedState.progress);
      $('restricted-progress-value').textContent = `${Math.round(100 * restrictedState.progress)}%`;
      $('restricted-play').textContent = restrictedState.progress === 1 ? 'Replay update' : 'Play update';
      $('restricted-tau').value = String(restrictedState.tau);
      $('restricted-tau-value').textContent = String(Number(restrictedState.tau.toFixed(2)));
      $('restricted-weights').textContent = `Underpredict ×${Number(restrictedState.tau.toFixed(2))} · Overpredict ×${Number((1 - restrictedState.tau).toFixed(2))}`;
      $('restricted-delta').value = String(restrictedState.delta);
      $('restricted-tau-controls').hidden = restrictedState.loss !== 'expectile';
      $('restricted-delta-label').hidden = restrictedState.loss !== 'huber';
      $('restricted-prediction').value = String(restrictedState.prediction);
      $('restricted-prediction-value').textContent = String(Number(restrictedState.prediction.toFixed(2)));
      const bounds = $('visual-area').getBoundingClientRect();
      const state = { ...restrictedState, width: bounds.width, height: bounds.height };
      $('restricted-diagram').innerHTML = R.render(index, state, mobile());
      readout(R.readout(index, state));
      $('stage-caption').textContent = R.presentation(index, state).caption;
      stage.dataset.rate = 1;
      stage.dataset.sharedPrediction = R.resolve(index, state).prediction[0];
      stage.dataset.loss = index < 3 ? 'squared' : index < 5 ? 'expectile' : restrictedState.loss;
      stage.dataset.restrictedScene = index;
      delete stage.dataset.geometryScene;
      delete stage.dataset.workedScene;
      delete stage.dataset.round;
      stage.dataset.updateProgress = restrictedState.progress;
      return;
    }
    delete stage.dataset.restrictedScene;
    delete stage.dataset.sharedPrediction;
    if (current.geometry) {
      $('geometry-round-label').textContent = index === 0 ? 'Trees added' : 'Starting round';
      $('geometry-tree-controls').hidden = index !== 1;
      $('geometry-update-controls').hidden = index !== 1;
      $('geometry-previous-tree').disabled = geometryState.round === 0;
      $('geometry-next-tree').disabled = geometryState.round === 24;
      $('geometry-round').value = String(geometryState.round);
      $('geometry-round-value').textContent = String(geometryState.round);
      $('geometry-rate').value = String(geometryState.rate);
      $('geometry-progress').value = String(geometryState.progress);
      $('geometry-progress-value').textContent = `${Math.round(100 * geometryState.progress)}%`;
      const bounds = $('visual-area').getBoundingClientRect();
      const state = { ...geometryState, width: bounds.width, height: bounds.height };
      $('prediction-diagram').innerHTML = G.render(index, state, mobile());
      readout(G.readout(index, state));
      if (announce) $('stage-caption').textContent = G.presentation(index, state).caption;
      stage.dataset.rate = geometryState.rate;
      stage.dataset.updateProgress = geometryState.progress;
      stage.dataset.round = geometryState.round;
      stage.dataset.geometryScene = index;
      delete stage.dataset.workedScene;
      return;
    }
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
    delete stage.dataset.geometryScene;
  }

  function cancelUpdate() {
    $('stage-caption').setAttribute('aria-live', 'polite');
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

  function animateRestrictedUpdate() {
    cancelUpdate();
    const start = performance.now();
    restrictedState.progress = reduced.matches ? 1 : 0;
    $('stage-caption').setAttribute('aria-live', reduced.matches ? 'polite' : 'off');
    render();
    if (reduced.matches) return;
    const tick = now => {
      const t = Math.min(1, (now - start) / 1800);
      restrictedState.progress = t * t * (3 - 2 * t);
      if (t < 1) { render(false); animation = requestAnimationFrame(tick); }
      else { animation = 0; $('stage-caption').setAttribute('aria-live', 'polite'); render(); }
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
    stage.dataset.example = current.loss ? 'loss' : current.geometry ? 'geometry' : current.restricted ? 'restricted' : 'worked';
    $('stage-topic').textContent = (current.loss ? L : current.geometry ? G : current.restricted ? R : W).topics[current.index];
    $('stage-count').textContent = `${String(scene + 1).padStart(2, '0')} / ${steps.length}`;
    $('previous').disabled = scene === 0;
    $('next').textContent = scene === steps.length - 1 ? 'Beyond squared error' : 'Next';
    steps.forEach((step, i) => step.toggleAttribute('data-active', i === scene));
    if (!current.geometry && !current.restricted && !current.loss) {
      exampleState.candidateIndex = null;
      exampleState.process = 1;
      if (animate && (current.index === 3 || current.index === 5)) animateUpdate();
      else render();
    } else if (current.restricted) {
      if ([1, 3, 4].includes(current.index)) restrictedState.progress = 0;
      render();
    } else render();
  }

  function scrollUpdate() {
    scrollPending = false;
    // Keep an explicitly requested scene selected until its scroll arrives.
    if (navigationTarget && Math.abs(scrollY - navigationTarget.top) <= 2) navigationTarget = null;
    const stageBounds = stage.getBoundingClientRect();
    // On a phone, navigation brings the static figure into view. Keep that
    // chosen scene while it is visible; reading below it can select new scenes.
    if (!navigationTarget && (!mobile() || stageBounds.bottom <= 64)) {
      const threshold = mobile() ? window.innerHeight * 0.35 : stageBounds.bottom + 32;
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
    const requested = mobile()
      ? bounds.top + scrollY - 64
      : steps[index].getBoundingClientRect().top + scrollY - bounds.height - 74;
    // Browsers clamp scroll destinations to the available document range.
    const top = Math.max(0, Math.min(document.documentElement.scrollHeight - innerHeight, requested));
    navigationTarget = { top };
    window.scrollTo({ top, behavior: instant || reduced.matches ? 'instant' : 'smooth' });
  }

  function stepForId(id) {
    if (id === 'vector-radius') id = 'identical-inputs';
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

  $('restricted-play').addEventListener('click', animateRestrictedUpdate);
  $('restricted-progress').addEventListener('input', () => {
    cancelUpdate();
    restrictedState.progress = Math.max(0, Math.min(1, Number($('restricted-progress').value)));
    render();
  });
  $('restricted-prediction').addEventListener('input', () => {
    restrictedState.prediction = Math.max(0, Math.min(1, Number($('restricted-prediction').value)));
    render();
  });
  $('restricted-fit').addEventListener('click', () => {
    restrictedState.prediction = R.resolve(5, restrictedState).nextPrediction[0];
    render();
  });
  $('restricted-reset').addEventListener('click', () => {
    restrictedState.prediction = 0;
    render();
  });
  $('restricted-mean').addEventListener('click', () => {
    restrictedState.prediction = 0.5;
    render();
  });
  $('restricted-loss').addEventListener('change', () => {
    restrictedState.loss = $('restricted-loss').value;
    render();
  });
  $('restricted-tau').addEventListener('input', () => {
    restrictedState.tau = Math.max(0.05, Math.min(0.95, Number($('restricted-tau').value)));
    render();
  });
  $('restricted-delta').addEventListener('change', () => {
    restrictedState.delta = Number($('restricted-delta').value);
    render();
  });

  $('geometry-round').addEventListener('input', () => {
    geometryState.round = Math.max(0, Math.min(24, Math.round(Number($('geometry-round').value))));
    geometryState.progress = 1;
    render();
  });
  function moveTree(direction) {
    geometryState.round = Math.max(0, Math.min(24, geometryState.round + direction));
    geometryState.progress = 1;
    render();
  }
  $('geometry-previous-tree').addEventListener('click', () => moveTree(-1));
  $('geometry-next-tree').addEventListener('click', () => moveTree(1));
  $('geometry-rate').addEventListener('change', () => {
    geometryState.rate = Number($('geometry-rate').value);
    render();
  });
  $('geometry-progress').addEventListener('input', () => {
    geometryState.progress = Math.max(0, Math.min(1, Number($('geometry-progress').value)));
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
    cancelUpdate();
    exampleState.process = 1;
    if (wasAnimating && localScene().restricted) restrictedState.progress = 1;
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
