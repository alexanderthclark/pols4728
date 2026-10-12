(function (root, factory) {
  'use strict';
  const math = root && root.BoostingMath || (typeof require === 'function' ? require('../shared/boosting-math.js') : null);
  const api = factory(math);
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.PredictionBoostingView = api;
})(typeof window !== 'undefined' ? window : globalThis, function (M) {
  'use strict';
  if (!M) throw new Error('Load BoostingMath before PredictionBoostingView.');

  const target = [1, -1, 1], zero = [0, 0, 0];
  const C = { ink: '#242424', blue: '#234e70', rust: '#a54f32', teal: '#007f73', muted: '#606060', rule: '#767676', faint: '#dedede' };
  const topics = ['The observed vector and its gradient', 'Fit and scale a correction', 'The residual norm is a radius'];
  const esc = value => String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
  const fmt = value => {
    if (value === 0) return '0';
    if (Math.abs(value) < 1e-4) return value.toExponential(2).replace(/e([+-]?\d+)/, (_, exponent) => ' × 10^' + String(Number(exponent)).replace('-', '−'));
    return String(Number(value.toPrecision(4))).replace('-', '−');
  };
  const vector = values => '(' + values.map(fmt).join(', ') + ')';
  const clamp = (value, minimum, maximum) => Math.max(minimum, Math.min(maximum, value));
  const number = (value, fallback) => Number.isFinite(value) ? value : fallback;
  const sub = value => String(value).replace(/\d/g, digit => '₀₁₂₃₄₅₆₇₈₉'[digit]);

  // A deterministic plane through F and y. Its orthonormal basis preserves
  // lengths when a two-dimensional trial step is lifted into prediction space.
  function basisForResidual(residual) {
    const radius = M.norm(residual);
    if (radius === 0) return { radius, e1: [1, 0, 0], e2: [0, 1, 0] };
    const e1 = residual.map(value => value / radius);
    let axisIndex = 0;
    for (let i = 1; i < e1.length; i++) if (Math.abs(e1[i]) < Math.abs(e1[axisIndex])) axisIndex = i;
    const axis = e1.map((_, i) => i === axisIndex ? 1 : 0);
    const perpendicular = M.sub(axis, M.scale(e1, M.dot(axis, e1)));
    const e2 = M.scale(perpendicular, 1 / M.norm(perpendicular));
    return { radius, e1, e2 };
  }

  function resolve(scene, state = {}) {
    if (!Number.isInteger(scene) || scene < 0 || scene > 2) throw new RangeError('Prediction scene must be 0, 1, or 2.');
    const round = clamp(Math.round(number(state.round, 0)), 0, 24);
    const rate = clamp(number(state.rate, .5), 0, 1);
    const progress = clamp(number(state.progress, 1), 0, 1);
    const angle = clamp(number(state.angle, 120), 0, 180);
    const length = Math.max(0, number(state.length, .5));
    const path = M.buildPath({ rate, rounds: 24, target, x: [1, 2, 3] });
    const base = path[round].pred.slice(), baseResidual = M.sub(target, base);
    const basis = basisForResidual(baseResidual), radius = basis.radius;
    const tree = M.fitStump([1, 2, 3], baseResidual);
    let step = zero.slice(), planeStep = [0, 0], orientation = 1, effectiveAngle = 0, maxRelativeStep = 0;
    if (scene === 1) step = M.scale(tree.values, rate * progress);
    if (scene === 2 && radius > 0) {
      const geometry = M.orientedStepGeometry(angle, length, radius);
      planeStep = geometry.step;
      orientation = geometry.sign;
      effectiveAngle = geometry.effectiveAngle;
      maxRelativeStep = geometry.maxRelativeStep;
      step = M.add(M.scale(basis.e1, planeStep[0]), M.scale(basis.e2, planeStep[1]));
    } else if (scene === 2) {
      orientation = angle > 90 ? -1 : 1;
      effectiveAngle = Math.min(angle, 180 - angle);
    }
    const prediction = M.add(base, step), residual = M.sub(target, prediction);
    const gradient = M.scale(residual, -1), norm = M.norm(residual), loss = M.loss(target, prediction);
    const beforeLoss = M.loss(target, base);
    return { scene, round, rate, progress, angle, length, target: target.slice(), path, base, baseResidual, baseGradient: M.scale(baseResidual, -1), basis, radius, tree, step, planeStep, stepNorm: M.norm(step), orientation, effectiveAngle, maxRelativeStep, prediction, residual, gradient, norm, loss, beforeLoss };
  }

  function readout(scene, state = {}) {
    const q = resolve(scene, state);
    const compact = Number.isFinite(state.width) && state.width < 400;
    const displayedVector = values => compact ? '(' + values.map(value => {
      if (value !== 0 && Math.abs(value) < .01) return value.toExponential(0).replace(/-/g, '−');
      return String(Number(value.toFixed(2))).replace(/-/g, '−');
    }).join(', ') + ')' : vector(values);
    return [
      ['Observed vector y', displayedVector(q.target), C.rust],
      [scene === 0 ? `Prediction F${sub(q.round)}` : scene === 1 ? 'Updated prediction F′' : 'Trial prediction F′', displayedVector(q.prediction), C.blue],
      [scene === 0 ? 'Gradient ∇L(F) = F − y' : 'Gradient at F′', displayedVector(q.gradient), C.rust],
      [scene === 2 ? 'Trial norm; trial loss' : 'Residual norm; loss ½‖r‖²', `${fmt(q.norm)}; ${fmt(q.loss)}`, C.ink]
    ];
  }

  // L(F + Δ) − L(F) = ½R²s(s − 2|cos θ|). Classify the
  // dimensionless geometry so small positive losses retain their true sign.
  // A relative floating-point tolerance handles exact circle intersections.
  function trialLossChange(q) {
    if (q.radius === 0 || q.length === 0) return 0;
    const difference = q.length - q.maxRelativeStep;
    const tolerance = 8 * Number.EPSILON * Math.max(q.length, q.maxRelativeStep);
    return Math.abs(difference) <= tolerance ? 0 : difference < 0 ? -1 : 1;
  }

  function presentation(scene, state = {}) {
    const q = resolve(scene, state);
    let caption;
    if (scene === 0) caption = `${q.round} ${q.round === 1 ? 'tree' : 'trees'} at α = ${fmt(q.rate)}. The target y stays fixed; each tree is refitted to y − F. Arrow lengths here are projected; the displayed norm is computed in three dimensions.`;
    if (scene === 1) caption = `Fit h = ${vector(q.tree.values)} to the current negative gradient, then add ${fmt(q.progress)} × αh. The moving residual and gradient are evaluated at F′.`;
    if (scene === 2) {
      const change = trialLossChange(q);
      const outcome = change < 0 ? 'The trial step reduces loss.' : change > 0 ? 'The trial step increases loss.' : 'The trial step leaves loss unchanged.';
      const orientation = q.radius === 0 ? 'The residual is zero, so there is no descent direction.' : q.angle === 90 ? 'At 90°, neither h nor −h is a descent direction.' : `Choose ${q.orientation === -1 ? '−h' : 'h'}; its angle to the residual is ${fmt(q.effectiveAngle)}°.`;
      caption = `${orientation} ${outcome} The circle is an exact equal-loss slice: R = ‖y − F‖ = ${fmt(q.radius)}, L(F) = ${fmt(q.beforeLoss)}.`;
    }
    if (Number.isFinite(state.width) && state.width < 400) {
      if (scene === 0) caption = `${q.round} trees; α = ${fmt(q.rate)}. y stays fixed. Gradient and norm use the displayed prediction.`;
      if (scene === 1) caption = `h = ${vector(q.tree.values)}; add ${fmt(q.progress)} × αh. The gradient is evaluated at F′.`;
      if (scene === 2) {
        const change = trialLossChange(q);
        const outcome = change < 0 ? 'Loss decreases' : change > 0 ? 'Loss increases' : 'Loss is unchanged';
        caption = `${q.radius === 0 ? 'The residual is zero.' : q.angle === 90 ? 'Neither orientation offers descent.' : `Use ${q.orientation === -1 ? '−h' : 'h'} (${fmt(q.effectiveAngle)}°).`} ${outcome}: ${fmt(q.beforeLoss)} → ${fmt(q.loss)}. Starting R = ${fmt(q.radius)}.`;
      }
    }
    return { topic: topics[scene], caption, readout: readout(scene, state) };
  }

  const text = (p, value, attributes = '') => `<text x="${p[0]}" y="${p[1]}" ${attributes}>${esc(value)}</text>`;
  const line = (a, b, color, width = 1.3, attributes = '') => `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="${color}" ${attributes.includes('stroke-width=') ? '' : `stroke-width="${width}"`} ${attributes}/>`;
  const dot = (p, color, radius = 4.5, attributes = '') => `<circle cx="${p[0]}" cy="${p[1]}" r="${radius}" fill="${color}" stroke="white" stroke-width="1.5" ${attributes}/>`;
  const diamond = p => `<path class="prediction-target" data-x="${p[0]}" data-y="${p[1]}" d="M${p[0]},${p[1] - 6}l6,6l-6,6l-6,-6Z" fill="${C.rust}" stroke="white" stroke-width="1.5"/>`;
  const arrow = (a, b, color, attributes = '') => Math.hypot(b[0] - a[0], b[1] - a[1]) < .25 ? '' : line(a, b, color, 2.3, `marker-end="url(#pv-arrow-${color.slice(1)})" ${attributes}`);
  const defs = `<defs>${[C.blue, C.rust, C.teal, C.rule].map(color => `<marker id="pv-arrow-${color.slice(1)}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="${color}"/></marker>`).join('')}</defs>`;
  const styles = `<style>.prediction-view{width:100%;height:100%;min-width:0;color:${C.ink};font-family:var(--sans,Arial,sans-serif);font-variant-numeric:tabular-nums}.prediction-view svg{display:block;width:100%;height:100%;overflow:visible}.prediction-view text{font-size:13px;fill:${C.ink}}.prediction-view .pv-small{font-size:12px;fill:${C.muted}}.prediction-view .pv-math{font-family:var(--math,'Cambria Math',serif);font-size:16px}.prediction-view.compact text{font-size:12px}.prediction-view.compact .pv-small{font-size:11px}.prediction-view.compact .pv-math{font-size:14px}</style>`;

  function dimensions(state, compact) {
    return { W: Math.max(200, number(state.width, compact ? 346 : 620)), H: Math.max(150, number(state.height, compact ? 250 : 430)) };
  }

  function space(q, state, compact) {
    const { W, H } = dimensions(state, compact), margin = compact ? 25 : 38;
    // Retain the original orthographic camera. It maps three observation
    // coordinates to the screen; projected distances are never used as norms.
    const phi = 35 * Math.PI / 180, elevation = 24 * Math.PI / 180;
    const u = [Math.cos(phi), -Math.sin(phi), 0];
    const v = [Math.sin(phi) * Math.sin(elevation), Math.cos(phi) * Math.sin(elevation), Math.cos(elevation)];
    const scale = Math.max(1, Math.min((W - 2 * margin) / 3.68, (H - 2 * margin - 19) / 2.95));
    const cx = W / 2 - .06 * scale, cy = (H - 19) / 2 + .075 * scale;
    const project = p => [cx + scale * M.dot(u, p), cy - scale * M.dot(v, p)];
    const origin = project(zero), observed = project(target), current = project(q.prediction), base = project(q.base);
    let body = '';
    function axes(center, local, faint) {
      const starts = local ? [[-.35, 0, 0], [0, -.35, 0], [0, 0, -.35]] : [[-.8, 0, 0], [0, -.8, 0], [0, 0, -.6]];
      const ends = local ? [[.6, 0, 0], [0, .6, 0], [0, 0, .6]] : [[1.4, 0, 0], [0, 1.25, 0], [0, 0, 1.4]];
      const offsets = [[7, 14], [-7, -3], [0, -9]];
      starts.forEach((start, index) => {
        const a = project(M.add(center, start)), b = project(M.add(center, ends[index]));
        body += line(a, b, faint ? C.faint : '#999', 1.1);
        if (faint) return;
        if (!local) [-.5, .5, 1].forEach(t => {
          const tick = [0, 0, 0]; tick[index] = t;
          const p = project(M.add(center, tick));
          body += line([p[0] - 2.5, p[1] - 3], [p[0] + 2.5, p[1] + 3], '#999', 1);
          if (t === 1) body += text([p[0] + 5, p[1] + 14], '1', 'class="pv-small"');
        });
        body += text([b[0] + offsets[index][0], b[1] + offsets[index][1]], local ? `Δ${sub(index + 1)}` : `obs. ${index + 1}`, `class="pv-small" text-anchor="${index === 1 ? 'end' : index === 2 ? 'middle' : 'start'}"`);
      });
    }
    axes(zero, false, q.scene === 1);
    if (q.scene === 1) axes(q.base, true, false);
    if (q.round > 0) {
      body += `<polyline class="prediction-path" points="${q.path.slice(0, q.round + 1).map(s => project(s.pred).join(',')).join(' ')}" fill="none" stroke="${C.blue}" stroke-width="2.5" stroke-linejoin="round"/>`;
      q.path.slice(0, q.round).forEach(s => { body += dot(project(s.pred), C.blue, 2.7); });
    }
    if (q.scene === 0 && q.round === 0) {
      const rails = [[zero, [1, 0, 0]], [[1, 0, 0], [1, -1, 0]], [[1, -1, 0], target]];
      rails.forEach(([a, b]) => { body += line(project(a), project(b), '#b8b8b8', 1.2, 'stroke-dasharray="3 5"'); });
    }
    if (q.scene === 1) {
      const full = project(M.add(q.base, q.tree.values));
      body += arrow(base, full, C.teal, 'stroke-dasharray="2 5" opacity=".45" class="prediction-full-tree"');
      body += arrow(base, current, C.teal, 'stroke-width="3" class="prediction-scaled-tree"');
      body += dot(base, C.blue, 3, 'class="prediction-base"');
      body += text([full[0] + 8, full[1] - 8], 'h', `class="pv-math" style="fill:${C.teal}"`);
      body += text([base[0] - 8, base[1] + 18], `F${sub(q.round)}`, `text-anchor="end" class="pv-math" style="fill:${C.blue}"`);
    }
    const gradientEnd = project(M.add(q.prediction, q.gradient));
    body += arrow(current, observed, C.teal, 'stroke-dasharray="5 4" class="prediction-residual-arrow"');
    body += arrow(current, gradientEnd, C.rust, 'stroke-dasharray="2 4" class="prediction-gradient-arrow"');
    if (q.norm > .04) {
      const midpoint = [(current[0] + observed[0]) / 2, (current[1] + observed[1]) / 2];
      body += text([midpoint[0] + 4, midpoint[1] - 10], 'r = −∇L', `class="pv-math" style="fill:${C.teal}"`);
      body += text([gradientEnd[0] - 5, gradientEnd[1] + 17], '∇L', `class="pv-math" text-anchor="end" style="fill:${C.rust}"`);
    }
    body += diamond(observed) + text([observed[0] + 9, observed[1] - 10], 'y = (1, −1, 1)', `class="pv-math" style="fill:${C.rust}" text-anchor="${observed[0] > W - 120 ? 'end' : 'start'}"`);
    body += dot(current, C.blue, 4.5, 'class="prediction-current"');
    const label = q.scene === 0 ? `F${sub(q.round)}` : 'F′';
    body += text([current[0] + 8, current[1] + 20], label, `class="pv-math" style="fill:${C.blue}"`);
    if (q.scene === 0 && q.round === 0) body += text([origin[0] - 8, origin[1] + 19], '= 0', 'class="pv-small" text-anchor="end"');
    body += text([W / 2, H - 4], `‖r‖ = ${fmt(q.norm)}     L = ½‖r‖² = ${fmt(q.loss)}`, 'class="pv-small" text-anchor="middle"');
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="prediction-title prediction-description" data-geometry="orthographic-3d" data-scale="${scale}"><title id="prediction-title">${esc(topics[q.scene])}</title><desc id="prediction-description">${esc(presentation(q.scene, state).caption)} Observed vector y is ${esc(vector(target))}. Prediction is ${esc(vector(q.prediction))}. Gradient is ${esc(vector(q.gradient))}. Residual norm is ${fmt(q.norm)}.</desc>${defs}${body}</svg>`;
  }

  function plane(q, state, compact) {
    const { W, H } = dimensions(state, compact), margin = compact ? 24 : 34;
    const initialRadius = Math.sqrt(3);
    // The physical domain is independent of the selected boosting round. This
    // makes the equal-loss circle shrink visibly as the residual norm falls.
    const extentY = initialRadius * 2.65;
    const minimumX = -2.25 * initialRadius;
    const maximumX = initialRadius * 1.7;
    const scale = Math.max(.01, Math.min((W - 2 * margin) / (maximumX - minimumX), (H - 2 * margin - 22) / (2 * extentY)));
    const cx = (W - (maximumX - minimumX) * scale) / 2 - minimumX * scale;
    const cy = (H - 22) / 2;
    const project = p => [cx + p[0] * scale, cy - p[1] * scale];
    const observed = project([0, 0]), base = project([-q.radius, 0]);
    const trial = project([-q.radius + q.planeStep[0], q.planeStep[1]]);
    const gradientEnd = project([-2 * q.radius, 0]);
    let body = `<circle class="equal-loss-circle" data-radius="${q.radius}" cx="${observed[0]}" cy="${observed[1]}" r="${q.radius * scale}" fill="#f8fafb" stroke="${C.rule}" stroke-width="1.3"/>`;
    if (compact) body += text([W / 2, 14], 'g = ∇L(F); r = −g', 'class="pv-small" text-anchor="middle"');
    body += line(project([minimumX, 0]), project([maximumX, 0]), C.faint, 1);
    body += arrow(base, observed, C.teal, 'stroke-dasharray="5 4" class="prediction-residual-arrow"');
    body += arrow(base, gradientEnd, C.rust, 'stroke-dasharray="2 4" class="prediction-gradient-arrow"');
    body += text([(base[0] + observed[0]) / 2, base[1] - 10], compact ? 'r' : '−∇L(F) = r', `class="pv-math" text-anchor="middle" style="fill:${C.teal}"`);
    body += text([(base[0] + gradientEnd[0]) / 2, base[1] + 17], compact ? 'g' : '∇L(F)', `class="pv-math" text-anchor="middle" style="fill:${C.rust}"`);
    if (q.radius > 0) {
      const unit = M.orientedStepGeometry(q.angle, 1, q.radius).step;
      const guide = M.scale(unit, .6), selectedGuide = project([-q.radius + guide[0], guide[1]]), oppositeGuide = project([-q.radius - guide[0], -guide[1]]);
      body += line(oppositeGuide, selectedGuide, '#aaa', 1.2, 'stroke-dasharray="3 4"');
      body += arrow(base, trial, q.angle === 90 ? C.rule : C.teal, 'stroke-width="3" class="prediction-trial-step"');
      body += line(trial, observed, C.rule, 1.2, 'stroke-dasharray="3 4" class="prediction-trial-residual"');
      if (q.length > 0) body += dot(trial, C.blue, 4.5, 'class="prediction-current"');
      const selectedName = q.orientation === -1 ? '−h' : 'h';
      body += text([selectedGuide[0] + 6, selectedGuide[1] + (q.planeStep[1] < 0 ? compact ? 30 : 17 : compact ? -20 : -9)], selectedName, `class="pv-math" style="fill:${q.angle === 90 ? C.rule : C.teal}"`);
      body += text([oppositeGuide[0] - 6, oppositeGuide[1] + (q.planeStep[1] < 0 ? -8 : 17)], q.orientation === -1 ? 'h' : '−h', 'class="pv-math" text-anchor="end"');
      if (q.length > 0) body += text([clamp(trial[0] + 8, 55, W - 55), clamp(trial[1] + (q.planeStep[1] < 0 ? 17 : -11), 15, H - 29)], compact ? 'F′' : 'trial F′', `class="pv-small" style="fill:${C.blue}"`);
    }
    body += dot(base, C.blue, 4.5, 'class="prediction-base"') + diamond(observed);
    body += text([base[0] - 6, base[1] + 34], `F${sub(q.round)}`, `class="pv-math" text-anchor="end" style="fill:${C.blue}"`);
    body += text([observed[0] + 8, observed[1] + 22], compact ? 'y' : 'target y', `class="pv-math" style="fill:${C.rust}"`);
    const radiusEnd = project([q.radius * Math.SQRT1_2, q.radius * Math.SQRT1_2]);
    body += line(observed, radiusEnd, C.rule, 1.2, 'class="prediction-radius"');
    body += text([radiusEnd[0] + 7, radiusEnd[1] - 5], `R = ${fmt(q.radius)}`, 'class="pv-small"');
    body += text([W / 2, H - 4], `Equal-loss circle: L = ½R² = ${fmt(q.beforeLoss)}`, 'class="pv-small" text-anchor="middle"');
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="prediction-title prediction-description" data-geometry="equal-loss-plane" data-scale="${scale}" data-units-per-pixel="${1 / scale}"><title id="prediction-title">${esc(topics[2])}</title><desc id="prediction-description">${esc(presentation(2, state).caption)} This is an equal-scale two-dimensional plane through the actual prediction F and target y. The trial step has norm ${fmt(q.stepNorm)}; its three-dimensional prediction is ${esc(vector(q.prediction))} and loss is ${fmt(q.loss)}.</desc>${defs}${body}</svg>`;
  }

  function render(scene, state = {}, compact = false) {
    const q = resolve(scene, state);
    const attributes = {
      scene, 'geometry-scene': scene, y: JSON.stringify(q.target), target: JSON.stringify(q.target), f: JSON.stringify(q.prediction), prediction: JSON.stringify(q.prediction),
      residual: JSON.stringify(q.residual), gradient: JSON.stringify(q.gradient), 'base-gradient': JSON.stringify(q.baseGradient),
      norm: q.norm, loss: q.loss, radius: q.radius, 'before-loss': q.beforeLoss, step: JSON.stringify(q.step),
      'step-norm': q.stepNorm, orientation: q.orientation, 'effective-angle': q.effectiveAngle, round: q.round, rate: q.rate, progress: q.progress
    };
    return `${styles}<div class="prediction-view${compact ? ' compact' : ''}" ${Object.entries(attributes).map(([key, value]) => `data-${key}="${esc(value)}"`).join(' ')}>${scene === 2 ? plane(q, state, compact) : space(q, state, compact)}</div>`;
  }

  return { render, readout, presentation, topics, resolve, basisForResidual };
});
