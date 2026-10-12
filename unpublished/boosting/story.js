(function () {
  'use strict';
  const M = window.BoostingMath;
  const $ = id => document.getElementById(id);
  const svg = $('space');
  const stage = document.querySelector('.stage-shell');
  const steps = [...document.querySelectorAll('.step')];
  const target = [1, -1, 1];
  const zero = [0, 0, 0];
  const worked = M.buildPath();
  const colors = { ink: '#242424', blue: '#234e70', rust: '#a54f32', teal: '#007f73', rule: '#767676', faint: '#dedede' };
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  const mobile = () => window.matchMedia('(max-width: 800px)').matches;
  const topics = ['Three observations', 'Prediction coordinates', 'The target stays fixed', 'The first weak learner', 'Scale the correction', 'The remaining error', 'A local origin', 'A second weak learner', 'An additive model', 'The boosting path', 'Choose h or −h', 'Descent in function space'];
  const captions = [
    'Each row contributes one coordinate to the prediction vector.',
    'Each axis measures a prediction for one observation.',
    'The rust diamond is the fixed target y = (1, −1, 1).',
    'The teal arrow is h₁: predictions from a stump split at x = 1.5.',
    'The solid arrow uses half of h₁. The dotted extension shows the full correction.',
    'The blue point is the ensemble. The dashed rust arrow is its remaining error.',
    'The local axes move to F₁. The target and the current prediction stay fixed.',
    'The second stump splits at x = 2.5. Half of its correction gives the next step.',
    'Two scaled trees give F₂ = (0.375, −0.125, 0.5).',
    '', '',
    'The residual arrow is the negative gradient. A fitted tree approximates this direction.'
  ];
  let scene = -1;
  let activePath = M.buildPath();
  let animation = 0;
  let displayed = { pred: zero, local: zero };
  let scrollPending = false;

  const fmt = (n, places = 2) => Math.abs(n) < 0.00001 ? '0' : Number(n.toFixed(places)).toString().replace('-', '−');
  const vector = v => '(' + v.map(n => fmt(n, 3)).join(', ') + ')';
  const lossText = n => n > 0 && n < 0.0001 ? n.toExponential(2).replace('e-', ' × 10^−') : fmt(n, 4);
  function node(tag, attributes = {}, text, parent = svg) {
    const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.entries(attributes).forEach(([key, value]) => el.setAttribute(key, value));
    if (text !== undefined) el.textContent = text;
    parent.appendChild(el);
    return el;
  }
  function line(a, b, color, width = 1.4, options = {}) {
    return node('line', { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: color, 'stroke-width': width, ...options });
  }
  function arrow(a, b, color, options = {}) {
    if (Math.hypot(b[0] - a[0], b[1] - a[1]) < 0.5) return;
    line(a, b, color, 2.5, { 'marker-end': `url(#arrow-${color.slice(1)})`, ...options });
  }
  function label(point, text, options = {}) {
    return node('text', { x: point[0], y: point[1], class: 'point-label', ...options }, text);
  }
  function dot(point, color, radius = 4.5) {
    return node('circle', { cx: point[0], cy: point[1], r: radius, fill: color, stroke: '#fff', 'stroke-width': 1.5 });
  }
  function diamond(point) {
    node('path', { d: `M${point[0]},${point[1] - 6}l6,6l-6,6l-6,-6Z`, fill: colors.rust, stroke: '#fff', 'stroke-width': 1.5 });
  }
  function resetPlot(width, height, description) {
    svg.replaceChildren();
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    node('title', { id: 'plot-title' }, topics[scene]);
    node('desc', { id: 'plot-description' }, description);
    const defs = node('defs');
    [colors.blue, colors.rust, colors.teal, colors.rule].forEach(color => {
      const marker = node('marker', { id: `arrow-${color.slice(1)}`, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 5, markerHeight: 5, orient: 'auto-start-reverse' }, undefined, defs);
      node('path', { d: 'M0 0L10 5L0 10Z', fill: color }, undefined, marker);
    });
  }
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
  function destination() {
    let pred = zero;
    if (scene >= 4 && scene <= 7) pred = worked[1].pred;
    if (scene === 8 || scene === 11) pred = worked[2].pred;
    if (scene === 9) pred = activePath[Number($('round').value)].pred;
    return { pred, local: scene === 6 || scene === 7 ? worked[1].pred : zero };
  }
  function updateNumbers(pred = destination().pred, announce = true) {
    if (scene === 0) readout([['Inputs', 'x = (1, 2, 3)'], ['Observed outcomes', 'y = (1, −1, 1)', colors.rust]]);
    else if (scene === 1) readout([['Prediction vector', 'F = (F(x₁), F(x₂), F(x₃))']]);
    else if (scene === 2) readout([['Target y', vector(target), colors.rust], ['Initial loss ½‖y − F₀‖²', '1.5']]);
    else {
      let middle = ['Residual y − F', vector(M.sub(target, pred)), colors.rust];
      if (scene === 3 || scene === 4) middle = ['Tree h₁', vector(worked[1].correction), colors.teal];
      if (scene === 7) middle = ['Tree h₂', vector(worked[2].correction), colors.teal];
      readout([['Ensemble F', vector(pred), colors.blue], middle, ['Loss', lossText(M.loss(target, pred))]]);
    }
    if (announce) $('stage-caption').textContent = scene === 9 ? `${$('round').value} trees; learning rate ${Number($('rate').value).toFixed(2)}. Every tree is refitted to the current residuals.` : captions[scene];
    svg.dataset.loss = M.loss(target, pred);
    svg.dataset.prediction = JSON.stringify(pred);
  }
  function drawSpace(pred, local) {
    const { width, height } = $('visual-area').getBoundingClientRect();
    if (!width || !height) return;
    resetPlot(width, height, `Three prediction axes, one per observation. Target y is ${vector(target)}. Current ensemble F is ${vector(destination().pred)}. ${captions[scene]}`);
    const padding = mobile() ? 28 : 44;
    const scale = Math.min((width - 2 * padding) / 3.25, (height - 2 * padding) / 2.55);
    const cx = width / 2 - 0.27 * scale;
    const cy = height / 2 + 0.37 * scale;
    // Orthographic camera: these two row vectors are orthonormal.
    const phi = 35 * Math.PI / 180, elevation = 24 * Math.PI / 180;
    const u = [Math.cos(phi), -Math.sin(phi), 0];
    const v = [Math.sin(phi) * Math.sin(elevation), Math.cos(phi) * Math.sin(elevation), Math.cos(elevation)];
    const project = p => [cx + scale * M.dot(u, p), cy - scale * M.dot(v, p)];
    const origin = project(zero), end = project(target), current = project(pred);
    function axes(center, faint = false) {
      const starts = [[-0.8, 0, 0], [0, -0.8, 0], [0, 0, -0.6]];
      const ends = [[1.4, 0, 0], [0, 1.25, 0], [0, 0, 1.4]];
      const offsets = [[9, 12], [-8, -3], [0, -10]];
      starts.forEach((start, index) => {
        const a = project(M.add(center, start));
        const b = project(M.add(center, ends[index]));
        line(a, b, faint ? '#e2e2e2' : '#999', 1.2);
        if (faint) return;
        [-0.5, 0.5, 1].forEach(t => {
          const tick = [0, 0, 0]; tick[index] = t;
          const p = project(M.add(center, tick));
          line([p[0] - 2.5, p[1] - 3], [p[0] + 2.5, p[1] + 3], '#999', 1);
          if (t === 1) label([p[0] + 5, p[1] + 14], '1', { class: 'small-label point-label' });
        });
        label([b[0] + offsets[index][0], b[1] + offsets[index][1]], `obs. ${index + 1}`, { class: 'axis-label point-label', 'text-anchor': index === 1 ? 'end' : index === 2 ? 'middle' : 'start' });
      });
    }
    if (scene >= 6 && scene <= 7) axes(zero, true);
    axes(local);
    if (scene === 1) {
      dot(origin, colors.ink, 3);
      label([origin[0] - 8, origin[1] + 22], '0', { 'text-anchor': 'end' });
      return;
    }
    // Three independent coordinate rails make the meaning of y visible.
    if (scene === 2) {
      const rails = [[zero, [1, 0, 0]], [[1, 0, 0], [1, -1, 0]], [[1, -1, 0], target]];
      rails.forEach(([a, b]) => line(project(a), project(b), '#b8b8b8', 1.5, { 'stroke-dasharray': '4 5' }));
      rails.forEach(([, b], index) => {
        const p = project(b);
        if (index < 2) label([p[0] + 6, p[1] + 18], index === 0 ? '1' : '−1', { class: 'small-label point-label' });
      });
    }
    let path = [worked[0]];
    if (scene >= 4 && scene <= 7) path = [worked[0], { pred }];
    if (scene === 8 || scene === 11) path = [worked[0], worked[1], { pred }];
    if (scene === 9) path = activePath.slice(0, Number($('round').value) + 1);
    if (path.length > 1) {
      node('polyline', { points: path.map(s => project(s.pred).join(',')).join(' '), fill: 'none', stroke: colors.blue, 'stroke-width': 2.5, 'stroke-linejoin': 'round' });
      path.slice(0, -1).forEach(s => dot(project(s.pred), colors.blue, 2.7));
    }
    if (scene >= 5) {
      arrow(current, end, colors.rust, { 'stroke-dasharray': '5 5', opacity: 0.85, 'stroke-width': 2 });
      if (scene === 5 || scene === 6 || scene === 11) {
        const mid = [(current[0] + end[0]) / 2, (current[1] + end[1]) / 2];
        label([mid[0] + 10, mid[1] + 4], scene === 11 ? '−∇L(F)' : 'y − F₁', { class: 'math-label point-label', style: `fill:${colors.rust}` });
      }
    }
    if (scene === 3 || scene === 4) {
      const full = project(worked[1].correction);
      if (scene === 4) {
        arrow(origin, full, colors.teal, { 'stroke-dasharray': '2 5', opacity: 0.4 });
        arrow(origin, current, colors.teal, { 'stroke-width': 3 });
        const middle = [(origin[0] + current[0]) / 2, (origin[1] + current[1]) / 2];
        label([middle[0], middle[1] - 14], '½h₁', { class: 'math-label point-label', style: `fill:${colors.teal}`, 'text-anchor': 'middle' });
      } else {
        arrow(origin, full, colors.teal);
        label([full[0] + 10, full[1] - 12], 'h₁', { class: 'math-label point-label', style: `fill:${colors.teal}` });
      }
    }
    if (scene === 7) {
      const base = worked[1].pred, correction = worked[2].correction;
      const full = project(M.add(base, correction)), next = project(worked[2].pred);
      arrow(project(base), full, colors.teal, { 'stroke-dasharray': '2 5', opacity: 0.4 });
      arrow(project(base), next, colors.teal, { 'stroke-width': 3 });
      dot(next, colors.teal, 3.5);
      label([next[0] - 10, next[1] - 2], '½h₂', { class: 'math-label point-label', 'text-anchor': 'end', style: `fill:${colors.teal}` });
    }
    diamond(end);
    label([end[0] + 9, end[1] - 9], 'y', { class: 'math-label point-label', style: `fill:${colors.rust}` });
    dot(current, colors.blue);
    const name = scene <= 3 ? 'F₀ = 0' : scene <= 7 ? 'F₁' : scene === 9 ? `F${$('round').value.replace(/\d/g, d => '₀₁₂₃₄₅₆₇₈₉'[d])}` : 'F₂';
    label([current[0] - 9, current[1] + 23], name, { class: 'math-label point-label', 'text-anchor': 'end', style: `fill:${colors.blue}` });
    if (scene === 6) label([12, height - 10], 'Local origin at F₁', { class: 'small-label' });
  }
  function drawAngle() {
    const angle = Number($('direction-angle').value), length = Number($('length').value);
    const result = M.orientedStepGeometry(angle, length);
    const unit = M.orientedStepGeometry(angle, 1).step;
    const orthogonal = angle === 90;
    const chosen = result.sign === -1 ? '−h' : 'h';
    const opposite = result.sign === -1 ? 'h' : '−h';
    const stepColor = orthogonal ? colors.rule : colors.teal;
    const { width, height } = $('visual-area').getBoundingClientRect();
    const outcome = result.improvement > 1e-10 ? 'This step reduces loss.' : result.improvement < -1e-10 ? 'This step increases loss.' : 'This step leaves loss unchanged.';
    const orientation = orthogonal ? 'At 90°, neither sign offers descent.' : `Use ${chosen}: its angle to the residual is ${result.effectiveAngle}°.`;
    resetPlot(width, height, `The candidate h has angle ${angle} degrees to the residual. ${orientation} Both orientations are shown. Relative step length ${length}. ${outcome} Loss changes from 0.5 to ${fmt(result.afterLoss, 4)}.`);
    // Both orientations lie on the same line. Negating an obtuse candidate
    // flips BOTH coordinates, so its selected step appears below the residual.
    const extent = Math.max(0.65, length);
    const positive = M.scale(unit, extent), negative = M.scale(positive, -1);
    const minX = Math.min(-0.2, negative[0] - 0.2), maxX = Math.max(2.15, positive[0] + 0.2);
    const maxY = Math.max(1.15, Math.abs(positive[1]) + 0.2), minY = -maxY;
    const margin = mobile() ? 32 : 48;
    const s = Math.max(1, Math.min((width - margin * 2) / (maxX - minX), (height - margin * 2) / (maxY - minY)));
    const ox = (width - (maxX - minX) * s) / 2 - minX * s;
    const oy = (height - (maxY - minY) * s) / 2 + maxY * s;
    const project = p => [ox + p[0] * s, oy - p[1] * s];
    const a = project([0, 0]), b = project([1, 0]), c = project(result.step);
    const d = project(result.oppositeStep);
    node('circle', { cx: b[0], cy: b[1], r: s, fill: '#f8fafb', stroke: '#999', 'stroke-width': 1.2 });
    line(project([minX, 0]), project([maxX, 0]), '#ddd', 1);
    line(project(negative), project(positive), '#b8b8b8', 1.2);
    arrow(a, b, colors.rust, { 'stroke-dasharray': '4 5', 'stroke-width': 2 });
    line(c, b, colors.rule, 1.5, { 'stroke-dasharray': '3 4' });
    arrow(a, d, colors.rule, { 'stroke-dasharray': '3 5', 'stroke-width': 1.8 });
    arrow(a, c, stepColor, { 'stroke-width': 3 });
    if (result.effectiveAngle > 0 && length > 0) {
      const radius = Math.min(s * 0.28, 32);
      const radians = Math.atan2(unit[1], unit[0]);
      const arcEnd = [a[0] + radius * Math.cos(radians), a[1] - radius * Math.sin(radians)];
      node('path', { d: `M${a[0] + radius},${a[1]}A${radius},${radius},0,0,${radians < 0 ? 1 : 0},${arcEnd.join(',')}`, fill: 'none', stroke: stepColor, 'stroke-width': 1.2 });
      if (!mobile()) label([a[0] + (radius + 16) * Math.cos(radians / 2), a[1] - (radius + 16) * Math.sin(radians / 2)], `${result.effectiveAngle}°`, { class: 'small-label point-label', 'text-anchor': 'middle' });
    }
    dot(a, colors.blue); diamond(b);
    if (length > 0) dot(c, stepColor);
    label([a[0] - 9, a[1] + 21], 'current F', { 'text-anchor': 'end' });
    label([b[0] + 9, b[1] + 21], 'target y');
    const forwardTip = project(positive), reverseTip = project(negative);
    label([forwardTip[0] + 8, forwardTip[1] + (unit[1] < 0 ? 18 : -10)], chosen, { class: 'math-label point-label', style: `fill:${stepColor}` });
    label([reverseTip[0] - 8, reverseTip[1] + (unit[1] < 0 ? -10 : 18)], opposite, { class: 'math-label point-label', 'text-anchor': 'end', style: `fill:${colors.rule}` });
    readout([
      ['Descent orientation', orthogonal ? 'Neither' : chosen, stepColor],
      ['Selected angle', `${result.effectiveAngle}°`],
      ['Loss: before → after', `0.5 → ${fmt(result.afterLoss, 4)}`, result.improvement >= 0 ? colors.teal : colors.rust]
    ]);
    $('angle-value').textContent = `${angle}°`;
    $('direction-angle').setAttribute('aria-valuetext', `${angle} degrees. ${orientation}`);
    $('length-value').textContent = length.toFixed(2);
    $('stage-caption').textContent = `${orthogonal ? 'At 90°, neither sign offers descent. ' : ''}${outcome}${orthogonal ? '' : ` Improving lengths: 0 < s < ${fmt(result.maxRelativeStep, 3)}.`}`;
    svg.dataset.loss = result.afterLoss;
    svg.dataset.angle = angle;
    svg.dataset.effectiveAngle = result.effectiveAngle;
    svg.dataset.orientation = result.sign;
    svg.dataset.step = JSON.stringify(result.step);
    svg.dataset.stepLength = length;
  }
  function render() {
    if (scene === 0) return;
    if (scene === 10) drawAngle();
    else drawSpace(displayed.pred, displayed.local);
  }
  function changeScene(index, animate = true) {
    if (index === scene) return;
    const previous = scene;
    scene = Math.max(0, Math.min(steps.length - 1, index));
    stage.dataset.scene = scene;
    $('stage-topic').textContent = topics[scene];
    $('stage-count').textContent = `${String(scene + 1).padStart(2, '0')} / 12`;
    $('data-intro').hidden = scene !== 0;
    svg.toggleAttribute('hidden', scene === 0);
    $('path-controls').hidden = scene !== 9;
    $('angle-controls').hidden = scene !== 10;
    $('previous').disabled = scene === 0;
    $('next').textContent = scene === 11 ? 'XGBoost' : 'Next';
    steps.forEach((step, i) => step.toggleAttribute('data-active', i === scene));
    cancelAnimationFrame(animation);
    const from = displayed, to = destination();
    updateNumbers();
    // Only neighboring narrative steps interpolate. Slider rounds remain exact.
    if (animate && !reduced.matches && Math.abs(scene - previous) === 1 && scene >= 4 && scene <= 8) {
      const start = performance.now();
      const tick = now => {
        const t = Math.min(1, (now - start) / 650), ease = t * t * (3 - 2 * t);
        displayed = { pred: M.add(from.pred, M.scale(M.sub(to.pred, from.pred), ease)), local: M.add(from.local, M.scale(M.sub(to.local, from.local), ease)) };
        updateNumbers(displayed.pred, false);
        render();
        if (t < 1) animation = requestAnimationFrame(tick);
      };
      animation = requestAnimationFrame(tick);
    } else { displayed = to; render(); }
  }
  function scrollUpdate() {
    scrollPending = false;
    const threshold = mobile() ? stage.getBoundingClientRect().bottom + 65 : window.innerHeight * 0.52;
    let index = 0;
    steps.forEach((step, i) => { if (step.getBoundingClientRect().top <= threshold) index = i; });
    changeScene(index);
    const range = document.documentElement.scrollHeight - innerHeight;
    $('progress').style.width = `${Math.min(100, Math.max(0, scrollY / range * 100))}%`;
  }
  function scrollToStep(index) {
    if (index >= steps.length) { $('xgboost').scrollIntoView({ behavior: reduced.matches ? 'instant' : 'smooth' }); return; }
    index = Math.max(0, index);
    // Keep the next heading below the pinned figure on narrow screens.
    const offset = mobile() ? stage.getBoundingClientRect().height + 64 : 90;
    const y = steps[index].getBoundingClientRect().top + scrollY - offset;
    window.scrollTo({ top: y, behavior: reduced.matches ? 'instant' : 'smooth' });
  }
  $('previous').addEventListener('click', () => scrollToStep(scene - 1));
  $('next').addEventListener('click', () => scrollToStep(scene + 1));
  document.addEventListener('keydown', event => {
    if (event.altKey || event.ctrlKey || event.metaKey || /^(INPUT|SELECT|TEXTAREA|BUTTON|A|SUMMARY)$/.test(event.target.tagName)) return;
    const rect = document.querySelector('.story-layout').getBoundingClientRect();
    if (rect.top > 90 || rect.bottom < innerHeight / 2) return;
    if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
      event.preventDefault(); scrollToStep(scene + (event.key === 'ArrowRight' ? 1 : -1));
    }
  });
  $('round').addEventListener('input', () => {
    $('round-value').textContent = $('round').value;
    displayed = destination(); updateNumbers(); render();
  });
  $('rate').addEventListener('change', () => {
    activePath = M.buildPath({ rate: Number($('rate').value) });
    displayed = destination(); updateNumbers(); render();
  });
  ['direction-angle', 'length'].forEach(id => $(id).addEventListener('input', drawAngle));
  window.addEventListener('scroll', () => {
    if (!scrollPending) { scrollPending = true; requestAnimationFrame(scrollUpdate); }
  }, { passive: true });
  new ResizeObserver(() => render()).observe($('visual-area'));
  window.addEventListener('resize', scrollUpdate);
  changeScene(0, false);
  scrollUpdate();
})();
