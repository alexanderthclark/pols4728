(function (root, factory) {
  'use strict';
  const math = root && root.BoostingMath || (typeof require === 'function' ? require('../shared/boosting-math.js') : null);
  const api = factory(math);
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.RestrictedBoostingView = api;
})(typeof window !== 'undefined' ? window : globalThis, function (M) {
  'use strict';
  if (!M) throw new Error('Load BoostingMath before RestrictedBoostingView.');

  const x = [1, 1], target = [1, 0], optimalPrediction = [.5, .5];
  const C = { ink: '#242424', blue: '#234e70', rust: '#a54f32', teal: '#007f73', muted: '#606060', rule: '#767676', faint: '#dedede' };
  const topics = ['Identical inputs restrict the available directions'];
  const esc = value => String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
  const number = (value, fallback) => Number.isFinite(value) ? value : fallback;
  const clamp = value => Math.max(0, Math.min(1, value));
  const fmt = value => {
    if (value === 0) return '0';
    if (Math.abs(value) < 1e-4) return value.toExponential(2).replace(/e([+-]?\d+)/, (_, exponent) => ' × 10^' + String(Number(exponent)).replace('-', '−'));
    return String(Number(value.toPrecision(4))).replace('-', '−');
  };
  const vector = values => '(' + values.map(fmt).join(', ') + ')';

  function resolve(scene = 0, state = {}) {
    if (scene !== 0) throw new RangeError('Restricted scene must be 0.');
    const c = clamp(number(state.prediction, 0));
    const prediction = [c, c], residual = M.sub(target, prediction);
    // Identical predictor values leave no available split. The shared fitter
    // therefore returns one leaf with the mean of the two residuals.
    const tree = M.fitStump(x, residual), nextTree = tree.values.slice();
    const nextPrediction = M.add(prediction, nextTree);
    const optimalResidual = M.sub(target, optimalPrediction);
    const norm = M.norm(residual), loss = M.loss(target, prediction);
    const minLoss = M.loss(target, optimalPrediction), nextLoss = M.loss(target, nextPrediction);
    return { scene, c, x: x.slice(), target: target.slice(), prediction, residual, gradient: M.scale(residual, -1), tree, nextTree, nextPrediction, optimalPrediction: optimalPrediction.slice(), optimalResidual, norm, loss, minLoss, nextLoss };
  }

  function readout(scene = 0, state = {}) {
    const q = resolve(scene, state);
    return [
      ['Current prediction f = (c, c)', vector(q.prediction), C.blue],
      ['Negative gradient r = y − f', vector(q.residual), C.rust],
      ['Next fitted tree h', vector(q.nextTree), C.teal],
      ['Residual norm; loss ½‖r‖²', `${fmt(q.norm)}; ${fmt(q.loss)}`, C.ink]
    ];
  }

  function presentation(scene = 0, state = {}) {
    const q = resolve(scene, state), compact = Number.isFinite(state.width) && state.width < 400;
    const caption = q.c === .5
      ? compact
        ? 'At f* = (0.5, 0.5), r is perpendicular to the line. The next tree is zero; loss stays at 0.25.'
        : 'At f* = (0.5, 0.5), the residual (0.5, −0.5) is perpendicular to every available tree direction. The next tree is zero. The equal-loss circle is tangent to the line, and the minimum attainable loss is 0.25.'
      : compact
        ? `Fit h = ${vector(q.nextTree)}. Adding it with α = 1 reaches f* = (0.5, 0.5); loss falls from ${fmt(q.loss)} to 0.25.`
        : `The fitted tree is h = ${vector(q.nextTree)}, the projection of the current residual onto the line. Adding it with α = 1 reaches f* = (0.5, 0.5), reducing loss from ${fmt(q.loss)} to 0.25. The circle has radius ‖y − f‖.`;
    return { topic: topics[0], caption, readout: readout(scene, state) };
  }

  const text = (p, value, attributes = '') => `<text x="${p[0]}" y="${p[1]}" ${attributes}>${esc(value)}</text>`;
  const line = (a, b, color, width = 1.3, attributes = '') => `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="${color}" stroke-width="${width}" ${attributes}/>`;
  const arrow = (a, b, color, attributes = '') => line(a, b, color, 2.3, `marker-end="url(#restricted-arrow-${color.slice(1)})" ${attributes}`);
  const styles = `<style>
.restricted-view{width:100%;height:100%;min-width:0;color:${C.ink};font-family:var(--sans,Arial,sans-serif);font-variant-numeric:tabular-nums}
.restricted-view svg{display:block;width:100%;height:100%;overflow:visible}
.restricted-view text{font-size:13px;fill:${C.ink}}
.restricted-view .rv-small{font-size:12px;fill:${C.muted}}
.restricted-view .rv-math{font-family:var(--math,'Cambria Math',serif);font-size:16px}
.restricted-view .rv-label{paint-order:stroke;stroke:white;stroke-width:4px;stroke-linejoin:round}
.restricted-view.compact text{font-size:12px}
.restricted-view.compact .rv-small{font-size:11px}
.restricted-view.compact .rv-math{font-size:14px}
</style>`;

  function plot(q, state, compact) {
    const W = Math.max(200, number(state.width, compact ? 320 : 620));
    const H = Math.max(180, number(state.height, compact ? 250 : 430));
    const domain = [-1.2, 2.2], span = domain[1] - domain[0];
    const margin = compact ? 24 : 38, top = compact ? 24 : 28, bottom = compact ? 29 : 31;
    // Both prediction coordinates have the same domain and physical scale.
    // Thus this circle's pixel radius is the actual residual norm times scale.
    const scale = Math.min((W - 2 * margin) / span, (H - top - bottom) / span);
    const side = span * scale, left = (W - side) / 2, upper = top + (H - top - bottom - side) / 2;
    const project = p => [left + (p[0] - domain[0]) * scale, upper + (domain[1] - p[1]) * scale];
    const f = project(q.prediction), y = project(q.target), best = project(q.optimalPrediction);
    const lo = domain[0], hi = domain[1];
    const defs = `<defs>${[C.rust, C.teal].map(color => `<marker id="restricted-arrow-${color.slice(1)}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="${color}"/></marker>`).join('')}</defs>`;
    let body = text([W / 2, 13], 'r = negative gradient; h = fitted tree', 'class="rv-small" text-anchor="middle"');
    body += `<circle class="restricted-loss-circle" data-radius="${q.norm}" cx="${y[0]}" cy="${y[1]}" r="${q.norm * scale}" fill="none" stroke="${C.rule}" stroke-width="1.3"/>`;
    body += line(project([lo, 0]), project([hi, 0]), C.faint, 1.1);
    body += line(project([0, lo]), project([0, hi]), C.faint, 1.1);
    [0, 1].forEach(value => {
      const px = project([value, 0]), py = project([0, value]);
      body += line([px[0], px[1] - 3], [px[0], px[1] + 3], C.rule, 1);
      body += line([py[0] - 3, py[1]], [py[0] + 3, py[1]], C.rule, 1);
      if (value === 1) body += text([px[0], px[1] + 17], '1', 'class="rv-small" text-anchor="middle"') + text([py[0] - 8, py[1] + 4], '1', 'class="rv-small" text-anchor="end"');
    });
    body += text([project([hi, 0])[0] - 2, project([hi, 0])[1] - 12], 'f(x₁)', 'class="rv-math" text-anchor="end"');
    body += text([project([0, hi])[0] - 7, project([0, hi])[1] + 5], 'f(x₂)', 'class="rv-math" text-anchor="end"');
    body += line(project([lo, lo]), project([hi, hi]), C.blue, 2, 'class="restricted-available-line"');
    body += text([project([1.35, 1.35])[0] - 8, project([1.35, 1.35])[1] - 9], '(c, c)', `class="rv-math rv-label" text-anchor="end" style="fill:${C.blue}"`);
    // The fixed projection residual shows the irreducible component. It is
    // perpendicular to the allowed line even when the current f is elsewhere.
    body += line(best, y, C.rule, 1.3, 'stroke-dasharray="2 4" class="restricted-projection-residual"');
    const mark = compact ? 7 : 9, unit = Math.SQRT1_2;
    const along = [unit * mark, -unit * mark], normal = [unit * mark, unit * mark];
    body += `<path class="restricted-right-angle" d="M${best[0] + along[0]},${best[1] + along[1]}l${normal[0]},${normal[1]}l${-along[0]},${-along[1]}" fill="none" stroke="${C.rule}" stroke-width="1.1"/>`;
    body += arrow(f, y, C.rust, 'stroke-dasharray="5 4" class="restricted-residual-arrow"');
    body += text([(f[0] + y[0]) / 2 + 8, (f[1] + y[1]) / 2 + (q.c < .1 ? -10 : 3)], 'r', `class="rv-math rv-label" style="fill:${C.rust}"`);
    if (q.c !== .5) {
      body += arrow(f, best, C.teal, 'class="restricted-tree-arrow"');
      body += text([(f[0] + best[0]) / 2 - 9, (f[1] + best[1]) / 2 - 1], 'h', `class="rv-math rv-label" text-anchor="end" style="fill:${C.teal}"`);
    }
    body += `<rect class="restricted-projection restricted-optimum" data-values="${q.optimalPrediction.join(',')}" data-pixel-x="${best[0]}" data-pixel-y="${best[1]}" x="${best[0] - 4}" y="${best[1] - 4}" width="8" height="8" fill="white" stroke="${C.blue}" stroke-width="1.6"/>`;
    body += `<circle class="restricted-current" data-values="${q.prediction.join(',')}" cx="${f[0]}" cy="${f[1]}" r="4.5" fill="${C.blue}" stroke="white" stroke-width="1.5"/>`;
    body += `<path class="restricted-target" data-values="${q.target.join(',')}" data-pixel-x="${y[0]}" data-pixel-y="${y[1]}" d="M${y[0]},${y[1] - 6}l6,6l-6,6l-6,-6Z" fill="${C.rust}" stroke="white" stroke-width="1.5"/>`;
    body += text([y[0] + 10, y[1] + 20], compact ? 'y' : 'y = (1, 0)', `class="rv-math rv-label" style="fill:${C.rust}"`);
    if (q.c === .5) {
      body += text([best[0] - 10, best[1] - 13], 'f = f*', `class="rv-math rv-label" text-anchor="end" style="fill:${C.blue}"`);
      body += text([best[0] - 10, best[1] + 19], 'h = 0', `class="rv-small rv-label" text-anchor="end" style="fill:${C.teal}"`);
    } else {
      body += text([best[0] - 10, best[1] - 13], 'f*', `class="rv-math rv-label" text-anchor="end" style="fill:${C.blue}"`);
      body += text([f[0] - 8, f[1] + 19], 'f', `class="rv-math rv-label" text-anchor="end" style="fill:${C.blue}"`);
    }
    body += text([W / 2, H - 5], 'Equal-loss circle centred at y; radius ‖r‖', 'class="rv-small" text-anchor="middle"');
    const description = `${presentation(0, state).caption} The horizontal and vertical axes are the predictions for observations 1 and 2. Both observations have x = 1, so the available prediction vectors satisfy f(x₁) = f(x₂). Target y is (1, 0). Current f is ${vector(q.prediction)}, negative gradient is ${vector(q.residual)}, and next fitted tree is ${vector(q.nextTree)}. The optimal prediction is (0.5, 0.5). Both axes use the same scale.`;
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="restricted-title restricted-description" data-geometry="two-observation-prediction-space" data-x-domain="${domain.join(',')}" data-y-domain="${domain.join(',')}" data-scale="${scale}" data-units-per-pixel="${1 / scale}"><title id="restricted-title">${esc(topics[0])}</title><desc id="restricted-description">${esc(description)}</desc>${defs}${body}</svg>`;
  }

  function render(scene = 0, state = {}, compact = false) {
    const q = resolve(scene, state);
    const attributes = {
      'restricted-scene': scene, c: q.c, x: JSON.stringify(q.x), target: JSON.stringify(q.target), prediction: JSON.stringify(q.prediction),
      residual: JSON.stringify(q.residual), gradient: JSON.stringify(q.gradient), 'next-tree': JSON.stringify(q.nextTree),
      'next-prediction': JSON.stringify(q.nextPrediction), 'optimal-prediction': JSON.stringify(q.optimalPrediction),
      norm: q.norm, loss: q.loss, 'min-loss': q.minLoss, 'next-loss': q.nextLoss
    };
    return `${styles}<div class="restricted-view${compact ? ' compact' : ''}" ${Object.entries(attributes).map(([key, value]) => `data-${key}="${esc(value)}"`).join(' ')}>${plot(q, state, compact)}</div>`;
  }

  return { render, readout, presentation, topics, resolve };
});
