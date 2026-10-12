(function (root, factory) {
  'use strict';
  const math = root && root.BoostingMath || (typeof require === 'function' ? require('../shared/boosting-math.js') : null);
  const api = factory(math);
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.RestrictedBoostingView = api;
})(typeof window !== 'undefined' ? window : globalThis, function (M) {
  'use strict';
  if (!M) throw new Error('Load BoostingMath before RestrictedBoostingView.');

  const x = [1, 1], target = [0, 1];
  const C = { ink: '#242424', blue: '#234e70', rust: '#a54f32', teal: '#007f73', muted: '#606060', rule: '#767676', faint: '#dedede' };
  const topics = ['The loss changes the best shared prediction'];
  const esc = value => String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
  const number = (value, fallback) => Number.isFinite(value) ? value : fallback;
  const clamp = (value, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, value));
  const fmt = value => String(Number(value.toPrecision(4))).replace('-', '−');
  const vector = values => '(' + values.map(fmt).join(', ') + ')';

  function specification(state) {
    const key = ['squared', 'expectile', 'huber', 'quartic'].includes(state.loss) ? state.loss : 'expectile';
    const tau = clamp(number(state.tau, .8), .05, .95), delta = clamp(number(state.delta, .25), .05, 2);
    const name = key === 'squared' ? 'Half-squared error' : key === 'expectile' ? `Asymmetric squared (τ = ${fmt(tau)})` : key === 'huber' ? `Huber (δ = ${fmt(delta)})` : 'Fourth-power error';
    return { key, tau, delta, name };
  }

  function pointLoss(e, spec) {
    if (spec.key === 'expectile') return (e >= 0 ? spec.tau : 1 - spec.tau) * e * e;
    if (spec.key === 'huber') return Math.abs(e) <= spec.delta ? e * e / 2 : spec.delta * (Math.abs(e) - spec.delta / 2);
    if (spec.key === 'quartic') return e ** 4 / 4;
    return e * e / 2;
  }

  function negativeDerivative(e, spec) {
    if (spec.key === 'expectile') return 2 * (e >= 0 ? spec.tau : 1 - spec.tau) * e;
    if (spec.key === 'huber') return clamp(e, -spec.delta, spec.delta);
    if (spec.key === 'quartic') return e ** 3;
    return e;
  }

  const totalLoss = (prediction, spec) => target.reduce((sum, y, i) => sum + pointLoss(y - prediction[i], spec), 0);

  function resolve(scene = 0, state = {}) {
    if (scene !== 0) throw new RangeError('Restricted scene must be 0.');
    const c = clamp(number(state.prediction, 0)), lossSpec = specification(state);
    const prediction = [c, c], residual = M.sub(target, prediction);
    const negativeGradient = residual.map(e => negativeDerivative(e, lossSpec));
    const tree = M.fitStump(x, negativeGradient), nextTree = tree.values.slice();
    const optimalC = lossSpec.key === 'expectile' ? lossSpec.tau : .5;
    const optimumInterval = lossSpec.key === 'huber' && lossSpec.delta < .5 ? [lossSpec.delta, 1 - lossSpec.delta] : [optimalC, optimalC];
    // Use the closest optimum when the original loss has a flat minimum. A
    // prediction already in that interval receives a zero leaf correction.
    const nextC = clamp(c, ...optimumInterval), nextPrediction = [nextC, nextC];
    const optimalPrediction = [optimalC, optimalC];
    const gradientNextPrediction = M.add(prediction, nextTree), rawNextPrediction = [.5, .5];
    const norm = M.norm(residual), loss = totalLoss(prediction, lossSpec), minLoss = totalLoss(optimalPrediction, lossSpec);
    return { scene, c, x: x.slice(), target: target.slice(), lossSpec, prediction, residual, negativeGradient, gradient: M.scale(negativeGradient, -1), tree, nextTree,
      correction: M.sub(nextPrediction, prediction), nextPrediction, gradientNextPrediction, rawNextPrediction, optimalC, optimalPrediction, optimumInterval,
      optimalResidual: M.sub(target, optimalPrediction), norm, loss, minLoss, nextLoss: totalLoss(nextPrediction, lossSpec), rawNextLoss: totalLoss(rawNextPrediction, lossSpec) };
  }

  function readout(scene = 0, state = {}) {
    const q = resolve(scene, state), interval = q.optimumInterval;
    const best = interval[0] === interval[1] ? fmt(q.optimalC) : `[${fmt(interval[0])}, ${fmt(interval[1])}]`;
    return [
      ['Raw residual y − f', vector(q.residual), C.muted],
      ['Negative gradient −∇L', vector(q.negativeGradient), C.rust],
      ['Fitted gradient target h', vector(q.nextTree), C.teal],
      ['Best c; loss / minimum', `${best}; ${fmt(q.loss)} / ${fmt(q.minLoss)}`, C.ink]
    ];
  }

  function presentation(scene = 0, state = {}) {
    const q = resolve(scene, state), atBest = Math.abs(q.loss - q.minLoss) < 1e-12;
    let caption;
    if (q.lossSpec.key === 'expectile') caption = atBest
      ? `At c = τ = ${fmt(q.optimalC)}, the gradient is perpendicular to the diagonal. The fitted gradient is zero; the raw residual generally is not perpendicular.`
      : `Raw residual fitting goes to c = 0.5. The gradient fit goes to c = τ = ${fmt(q.optimalC)}. Under this loss, those endpoints have losses ${fmt(q.rawNextLoss)} and ${fmt(q.minLoss)}.`;
    else if (q.lossSpec.key === 'huber' && q.lossSpec.delta < .5) caption = `Every c in [${fmt(q.optimumInterval[0])}, ${fmt(q.optimumInterval[1])}] minimizes this Huber loss. The teal correction reaches the closest point of this flat minimum; a fitted gradient mean need not equal that correction.`;
    else if (q.lossSpec.key === 'huber') caption = `Huber clips gradient components at ±${fmt(q.lossSpec.delta)}. The fitted mean is ${fmt(q.nextTree[0])}; the original-loss leaf correction is ${fmt(q.correction[0])}. The best shared prediction is 0.5, with minimum loss ${fmt(q.minLoss)}.`;
    else if (q.lossSpec.key === 'quartic') caption = `Fourth-power loss has negative gradients e³, rather than e. The fitted mean is ${fmt(q.nextTree[0])}; the original-loss leaf correction is ${fmt(q.correction[0])}. The best shared prediction remains 0.5.`;
    else caption = atBest
      ? 'At (0.5, 0.5), the residual and gradient are perpendicular to the diagonal. The next fitted tree is zero; minimum loss is 0.25.'
      : `The residual and negative gradient coincide. Adding the mean residual reaches (0.5, 0.5), reducing half-squared loss from ${fmt(q.loss)} to 0.25.`;
    if (Number.isFinite(state.width) && state.width < 400) {
      if (q.lossSpec.key === 'expectile') caption = atBest
        ? `Gradient fitting stops at c = τ = ${fmt(q.optimalC)}. The raw residual mean need not be zero.`
        : `Residual fit: c = 0.5, loss ${fmt(q.rawNextLoss)}. Gradient fit: c = ${fmt(q.optimalC)}, loss ${fmt(q.minLoss)}.`;
      else if (q.lossSpec.key === 'huber') caption = q.optimumInterval[0] !== q.optimumInterval[1]
        ? `Best c: [${fmt(q.optimumInterval[0])}, ${fmt(q.optimumInterval[1])}]. The teal step reaches the nearest optimum; gradient fitting and leaf correction can differ.`
        : `Huber clips at ±${fmt(q.lossSpec.delta)}. Gradient mean ${fmt(q.nextTree[0])}; leaf correction ${fmt(q.correction[0])}. Best c = 0.5.`;
      else if (q.lossSpec.key === 'quartic') caption = `Targets are e³. Gradient mean ${fmt(q.nextTree[0])}; leaf correction ${fmt(q.correction[0])}. Best c = 0.5.`;
    }
    return { topic: topics[0], caption, readout: readout(scene, state) };
  }

  // A radial parameterization of a separable convex loss contour about y.
  // The asymmetric quadratic pieces use their actual quadrant weights.
  function contourPoint(level, angle, spec) {
    const a = Math.cos(angle), b = Math.sin(angle);
    let radius;
    if (spec.key === 'squared') radius = Math.sqrt(2 * level);
    else if (spec.key === 'expectile') radius = Math.sqrt(level / ((a > 0 ? 1 - spec.tau : spec.tau) * a * a + (b > 0 ? 1 - spec.tau : spec.tau) * b * b));
    else if (spec.key === 'quartic') radius = Math.pow(4 * level / (a ** 4 + b ** 4), .25);
    else {
      let lo = 0, hi = Math.sqrt(2 * level) + level / spec.delta + spec.delta;
      for (let i = 0; i < 50; i++) {
        const middle = (lo + hi) / 2;
        if (pointLoss(-middle * a, spec) + pointLoss(-middle * b, spec) > level) hi = middle;
        else lo = middle;
      }
      radius = (lo + hi) / 2;
    }
    return [target[0] + radius * a, target[1] + radius * b];
  }

  const text = (p, value, attributes = '') => `<text x="${p[0]}" y="${p[1]}" ${attributes}>${esc(value)}</text>`;
  const line = (a, b, color, width = 1.3, attributes = '') => `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="${color}" stroke-width="${width}" ${attributes}/>`;
  const arrow = (a, b, color, attributes = '') => Math.hypot(b[0] - a[0], b[1] - a[1]) < .25 ? '' : line(a, b, color, 2.3, `marker-end="url(#restricted-arrow-${color.slice(1)})" ${attributes}`);
  const styles = `<style>.restricted-view{width:100%;height:100%;min-width:0;color:${C.ink};font-family:var(--sans,Arial,sans-serif);font-variant-numeric:tabular-nums}.restricted-view svg{display:block;width:100%;height:100%;overflow:visible}.restricted-view text{font-size:13px;fill:${C.ink}}.restricted-view .rv-small{font-size:12px;fill:${C.muted}}.restricted-view .rv-math{font-family:var(--math,'Cambria Math',serif);font-size:16px}.restricted-view .rv-label{paint-order:stroke;stroke:white;stroke-width:4px;stroke-linejoin:round}.restricted-view.compact text{font-size:12px}.restricted-view.compact .rv-small{font-size:11px}.restricted-view.compact .rv-math{font-size:14px}</style>`;

  function plot(q, state, compact) {
    const W = Math.max(200, number(state.width, compact ? 320 : 620)), H = Math.max(180, number(state.height, compact ? 250 : 430));
    const domain = [-.3, 1.3], span = domain[1] - domain[0], top = 33, bottom = compact ? 27 : 30, margin = compact ? 28 : 40;
    const scale = Math.min((W - 2 * margin) / span, (H - top - bottom) / span);
    const side = span * scale, left = (W - side) / 2, upper = top + (H - top - bottom - side) / 2;
    const project = p => [left + (p[0] - domain[0]) * scale, upper + (domain[1] - p[1]) * scale];
    const f = project(q.prediction), y = project(q.target), best = project(q.nextPrediction), mean = project([.5, .5]);
    const defs = `<defs><clipPath id="restricted-plot-clip"><rect x="${left}" y="${upper}" width="${side}" height="${side}"/></clipPath>${[C.rust, C.teal, C.muted].map(color => `<marker id="restricted-arrow-${color.slice(1)}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="${color}"/></marker>`).join('')}</defs>`;
    let body = text([W / 2 - 9, 16], '½(y − f)', `class="rv-small" text-anchor="end" style="fill:${C.muted}"`) + text([W / 2 + 9, 16], '−½∇L', `class="rv-small" style="fill:${C.rust}"`);
    function contour(level, color, className, dashed = false) {
      const points = Array.from({ length: 193 }, (_, i) => project(contourPoint(level, i * Math.PI / 96, q.lossSpec)).join(',')).join(' ');
      return `<polyline class="${className}" data-loss="${level}" points="${points}" fill="none" stroke="${color}" stroke-width="1.5" ${dashed ? 'stroke-dasharray="3 4"' : ''}/>`;
    }
    body += '<g clip-path="url(#restricted-plot-clip)">';
    if (q.loss > q.minLoss + 1e-12) body += contour((q.loss + q.minLoss) / 2, C.faint, 'restricted-intermediate-contour');
    body += contour(q.loss, C.rust, 'restricted-current-contour', true);
    body += contour(q.minLoss, C.teal, 'restricted-optimal-contour');
    body += '</g>';
    body += line(project([domain[0], 0]), project([domain[1], 0]), C.faint, 1.1) + line(project([0, domain[0]]), project([0, domain[1]]), C.faint, 1.1);
    [0, 1].forEach(value => {
      const px = project([value, 0]), py = project([0, value]);
      body += line([px[0], px[1] - 3], [px[0], px[1] + 3], C.rule, 1) + line([py[0] - 3, py[1]], [py[0] + 3, py[1]], C.rule, 1);
      if (value === 1) body += text([px[0], px[1] + 17], '1', 'class="rv-small" text-anchor="middle"') + text([py[0] - 8, py[1] + 4], '1', 'class="rv-small" text-anchor="end"');
    });
    const halfX = project([.5, 0]), halfY = project([0, .5]);
    body += line([halfX[0], halfX[1] - 2], [halfX[0], halfX[1] + 2], C.rule, 1) + line([halfY[0] - 2, halfY[1]], [halfY[0] + 2, halfY[1]], C.rule, 1);
    body += text([halfX[0], halfX[1] + 17], '½', 'class="rv-small" text-anchor="middle"') + text([halfY[0] - 8, halfY[1] + 4], '½', 'class="rv-small" text-anchor="end"');
    body += text([project([domain[1], 0])[0], project([domain[1], 0])[1] - 10], 'f₁', 'class="rv-math" text-anchor="end"') + text([project([0, domain[1]])[0] - 8, project([0, domain[1]])[1] + 5], 'f₂', 'class="rv-math" text-anchor="end"');
    body += line(project([domain[0], domain[0]]), project([domain[1], domain[1]]), C.blue, 2, 'class="restricted-available-line"');
    body += text([project([1.18, 1.18])[0] - 8, project([1.18, 1.18])[1]], 'f₁ = f₂', `class="rv-small rv-label" text-anchor="end" style="fill:${C.blue}"`);
    if (q.optimumInterval[0] !== q.optimumInterval[1]) body += line(project(q.optimumInterval.map(() => q.optimumInterval[0])), project(q.optimumInterval.map(() => q.optimumInterval[1])), C.teal, 7, 'class="restricted-optimal-segment"');
    const residualEnd = project(M.add(q.prediction, M.scale(q.residual, .5))), gradientEnd = project(M.add(q.prediction, M.scale(q.negativeGradient, .5)));
    body += arrow(f, residualEnd, C.muted, 'stroke-dasharray="3 4" class="restricted-residual-arrow"');
    body += arrow(f, gradientEnd, C.rust, 'class="restricted-gradient-arrow"');
    body += arrow(f, best, C.teal, 'class="restricted-tree-arrow"');
    if (q.lossSpec.key === 'expectile' && q.lossSpec.tau !== .5) {
      body += `<circle class="restricted-mean" cx="${mean[0]}" cy="${mean[1]}" r="4" fill="white" stroke="${C.muted}" stroke-width="1.5"/>`;
      body += text([mean[0] + 8, mean[1] + 20], compact ? 'mean' : 'residual fit', 'class="rv-small rv-label"');
    }
    body += `<rect class="restricted-optimum" data-values="${q.nextPrediction.join(',')}" data-pixel-x="${best[0]}" data-pixel-y="${best[1]}" x="${best[0] - 4}" y="${best[1] - 4}" width="8" height="8" fill="white" stroke="${C.teal}" stroke-width="1.6"/>`;
    body += `<circle class="restricted-current" data-values="${q.prediction.join(',')}" cx="${f[0]}" cy="${f[1]}" r="4.5" fill="${C.blue}" stroke="white" stroke-width="1.5"/>`;
    body += `<path class="restricted-target" data-values="${q.target.join(',')}" data-pixel-x="${y[0]}" data-pixel-y="${y[1]}" d="M${y[0]},${y[1] - 6}l6,6l-6,6l-6,-6Z" fill="${C.rust}" stroke="white" stroke-width="1.5"/>`;
    body += text([y[0] + 9, y[1] - 10], compact ? 'y' : 'y = (0, 1)', `class="rv-math rv-label" style="fill:${C.rust}"`);
    body += text([best[0] - 9, best[1] - 11], Math.abs(q.loss - q.minLoss) < 1e-12 ? 'f = f*' : 'f*', `class="rv-math rv-label" text-anchor="end" style="fill:${C.teal}"`);
    if (Math.abs(q.loss - q.minLoss) >= 1e-12) body += text([f[0] - 8, f[1] + 19], 'f', `class="rv-math rv-label" text-anchor="end" style="fill:${C.blue}"`);
    body += text([W / 2, H - 5], compact ? 'Dashed: current loss · Teal: minimum on diagonal' : 'Dashed contour: current loss · Teal contour: best attainable loss', 'class="rv-small" text-anchor="middle"');
    const description = `${q.lossSpec.name}. ${presentation(0, state).caption} Two observations (x,y)=(1,0) and (1,1) constrain every prediction to the diagonal. Current prediction ${vector(q.prediction)}, raw residual ${vector(q.residual)}, negative gradient ${vector(q.negativeGradient)}, fitted gradient target ${vector(q.nextTree)}, and loss-optimal leaf correction ${vector(q.correction)}. Gradient and raw residual arrows are drawn at half their numerical lengths. Contours show total selected loss; only their portion in the displayed window is visible. Both axes have the same physical scale.`;
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="restricted-title restricted-description" data-geometry="two-observation-prediction-space" data-x-domain="${domain.join(',')}" data-y-domain="${domain.join(',')}" data-scale="${scale}" data-units-per-pixel="${1 / scale}" data-arrow-scale="0.5"><title id="restricted-title">${esc(topics[0])}</title><desc id="restricted-description">${esc(description)}</desc>${defs}${body}</svg>`;
  }

  function render(scene = 0, state = {}, compact = false) {
    const q = resolve(scene, state);
    const attributes = { 'restricted-scene': scene, c: q.c, loss: q.loss, 'loss-key': q.lossSpec.key, tau: q.lossSpec.tau, delta: q.lossSpec.delta,
      x: JSON.stringify(q.x), target: JSON.stringify(q.target), prediction: JSON.stringify(q.prediction), residual: JSON.stringify(q.residual), gradient: JSON.stringify(q.gradient),
      'negative-gradient': JSON.stringify(q.negativeGradient), 'next-tree': JSON.stringify(q.nextTree), correction: JSON.stringify(q.correction),
      'next-prediction': JSON.stringify(q.nextPrediction), 'optimal-prediction': JSON.stringify(q.optimalPrediction), 'optimum-interval': JSON.stringify(q.optimumInterval),
      norm: q.norm, 'min-loss': q.minLoss, 'next-loss': q.nextLoss };
    return `${styles}<div class="restricted-view${compact ? ' compact' : ''}" ${Object.entries(attributes).map(([key, value]) => `data-${key}="${esc(value)}"`).join(' ')}>${plot(q, state, compact)}</div>`;
  }

  return { render, readout, presentation, topics, resolve, specification, pointLoss, negativeDerivative, totalLoss, contourPoint };
});
