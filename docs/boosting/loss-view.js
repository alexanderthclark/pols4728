(function (root, factory) {
  'use strict';
  const example = root && root.LossBoostingExample || (typeof require === 'function' ? require('./loss-example.js') : null);
  const api = factory(example);
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.LossBoostingView = api;
})(typeof window !== 'undefined' ? window : globalThis, function (E) {
  'use strict';
  if (!E) throw new Error('Load LossBoostingExample before LossBoostingView.');

  const C = { ink: '#242424', muted: '#606060', teal: '#007f73', rust: '#a54f32', blue: '#234e70', grid: '#e6e6e6' };
  const topics = ['Change the loss, change the targets', 'The targets can change the split', 'Choose corrections using the original loss', 'Follow the two boosting paths'];
  const esc = value => String(value).replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
  const number = (value, fallback) => Number.isFinite(value) ? value : fallback;
  const fmt = value => value === 0 ? '0' : String(Number(value.toPrecision(4))).replace(/-/g, '−');
  const full = value => String(Number(value.toPrecision(12))).replace(/-/g, '−');
  const vector = values => '(' + values.map(fmt).join(', ') + ')';
  const fullVector = values => '(' + values.map(full).join(', ') + ')';
  const sub = value => String(value).replace(/\d/g, digit => '₀₁₂₃₄₅₆₇₈₉'[digit]);
  const signed = value => value > 0 ? '+' + fmt(value) : fmt(value);
  const lossName = options => options.loss === 'squared' ? 'Half-squared loss' : options.loss === 'expectile' ? `Expectile loss (τ = ${fmt(options.tau)})` : `Huber loss (δ = ${fmt(options.delta)})`;
  let cachedKey, cachedModel;

  function validateScene(scene) {
    if (!Number.isInteger(scene) || scene < 0 || scene > 3) throw new RangeError('Loss scene must be an integer from 0 to 3.');
  }

  function resolve(state = {}) {
    const options = { loss: state.loss || 'huber', dataset: state.dataset || 'outlier', delta: number(state.delta, 1), tau: number(state.tau, .8), rate: number(state.rate, .5), rounds: 25 };
    const key = JSON.stringify(options);
    if (key !== cachedKey) {
      cachedModel = E.build(options);
      cachedKey = key;
    }
    const model = cachedModel;
    const round = Math.max(0, Math.min(24, Math.round(number(state.round, 0))));
    const before = model.gradient[round], after = model.gradient[round + 1];
    const rawCandidates = after.residualCandidates || E.candidates(before.residual, model.x);
    const rawTree = after.residualTree || rawCandidates.reduce((best, candidate) => candidate.sse < best.sse - 1e-10 ? candidate : best, rawCandidates[0]);
    return { model, options: model.options || options, round, before, after, tree: after.tree, rawTree, rawCandidates, shared: after.tree.threshold === null, gradient: model.gradient[round], residual: model.residual[round] };
  }

  function readout(scene, state = {}) {
    validateScene(scene);
    const q = resolve(state);
    if (scene === 0) return [
      ['Raw residual y − F', vector(q.before.residual), C.rust],
      ['Negative gradient −∇ℓ', vector(q.before.targets), C.teal]
    ];
    if (scene === 1) return [
      ['Gradient-selected cut', q.shared ? 'One shared region' : `x = ${fmt(q.tree.threshold)}`, C.teal],
      ['Residual-selected cut', q.shared ? 'One shared region' : `x = ${fmt(q.rawTree.threshold)}`, C.rust],
      [q.shared ? 'Calibrated correction' : 'Calibrated correction: left; right', q.shared ? signed(q.tree.left) : `${signed(q.tree.left)}; ${signed(q.tree.right)}`, C.blue]
    ];
    if (scene === 2) return [
      [q.shared ? 'Mean gradient target' : 'Mean gradient targets: left; right', q.shared ? signed(q.tree.targetLeft) : `${signed(q.tree.targetLeft)}; ${signed(q.tree.targetRight)}`, C.teal],
      [q.shared ? 'Loss-optimal correction' : 'Loss-optimal corrections: left; right', q.shared ? signed(q.tree.left) : `${signed(q.tree.left)}; ${signed(q.tree.right)}`, C.blue]
    ];
    return [
      ['Gradient path: chosen-loss total', fmt(q.gradient.loss), C.teal],
      ['Residual path: chosen-loss total', fmt(q.residual.loss), C.rust]
    ];
  }

  function presentation(scene, state = {}) {
    validateScene(scene);
    const q = resolve(state);
    let caption;
    if (scene === 0) {
      const mechanism = q.options.loss === 'squared' ? 'With half-squared loss the two targets coincide.' : q.options.loss === 'huber' ? `Huber clips each residual at ±${fmt(q.options.delta)} before fitting the tree.` : `Expectile negative gradients multiply positive residuals by ${fmt(2 * q.options.tau)} and negative residuals by ${fmt(2 * (1 - q.options.tau))}.`;
      caption = `At F${sub(q.round)}, compare each raw residual with its negative gradient. ${mechanism}`;
    } else if (scene === 1) {
      const same = q.tree.threshold === q.rawTree.threshold;
      caption = q.shared ? `All inputs are identical, so a tree has one shared region. At F${sub(q.round)}, its mean gradient target is ${fmt(q.tree.targetLeft)}; its mean raw residual is ${fmt(q.rawTree.targetLeft)}. The original-loss-optimal correction is ${fmt(q.tree.left)}.` : `From the same F${sub(q.round)}, ${same ? `both target vectors choose cut ${fmt(q.tree.threshold)}` : `gradient targets choose cut ${fmt(q.tree.threshold)}; raw residuals choose ${fmt(q.rawTree.threshold)}`}. Each column minimizes its own target-fitting SSE. Leaf corrections then minimize the chosen original loss.`;
    } else if (scene === 2) {
      caption = `${q.shared ? 'Keep the single shared region.' : `Fix the gradient-selected cut at ${fmt(q.tree.threshold)}.`} Minimize Σℓ(yᵢ, F${sub(q.round)}(xᵢ) + c) over c ${q.shared ? 'in that region' : 'in each leaf'}. ${q.shared && q.options.loss === 'huber' ? `The gradient targets cancel, and c = ${fmt(q.tree.left)} minimizes Huber loss. A residual-mean correction ${fmt(q.rawTree.left)} points away from this optimum.` : q.options.loss === 'squared' ? 'For half-squared loss, the optimum equals the mean gradient target.' : 'A mean gradient target identifies a direction; the loss-optimal correction can have a different magnitude.'} Add αc with α = ${fmt(q.options.rate)}.`;
    } else {
      caption = `After ${q.round} ${q.round === 1 ? 'tree' : 'trees'}, both paths use ${lossName(q.options).toLowerCase()} to measure performance and share its optimal initial constant. The gradient path fits −∇ℓ and calibrates leaves; the residual path fits y − F and uses residual means.`;
    }
    return { topic: topics[scene], caption, readout: readout(scene, state) };
  }

  const styles = `<style>
.loss-view{width:100%;height:100%;min-width:0;display:flex;flex-direction:column;justify-content:center;gap:8px;color:${C.ink};font-family:var(--sans,Arial,sans-serif);font-variant-numeric:tabular-nums}
.loss-view .loss-plot{display:block;width:100%;height:auto;flex:0 0 auto;overflow:visible;min-width:0}
.loss-view .loss-note{margin:0;color:${C.muted};font-size:12px;line-height:1.35;text-align:center}
.loss-view .loss-equation{margin:0;font:18px/1.35 var(--math,'Course Math','Cambria Math',serif);text-align:center;overflow-wrap:anywhere}
.loss-view .loss-table{border-collapse:collapse;width:100%;margin:0;font-size:13px;line-height:1.2}
.loss-view .loss-table caption{text-align:left;padding:0 0 6px;font-size:12px;line-height:1.3;color:${C.muted}}
.loss-view .loss-table th,.loss-view .loss-table td{padding:6px 8px;border-bottom:1px solid ${C.grid};text-align:right;font-weight:400}
.loss-view .loss-table th:first-child{text-align:left}
.loss-view .loss-table thead th{font-size:11px;color:${C.muted}}
.loss-view .loss-table .loss-gradient-best{color:${C.teal};font-weight:600;background:#f0f7f6}
.loss-view .loss-table .loss-residual-best{color:${C.rust};font-weight:600;background:#fbf3ef}
.loss-view .loss-leaf-corrections td:last-child{color:${C.blue}}
.loss-view .loss-sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
.loss-view.compact{gap:5px}
.loss-view.compact .loss-note{font-size:10px;line-height:1.25}
.loss-view.compact .loss-equation{font-size:15px;line-height:1.25}
.loss-view.compact .loss-table{font-size:11px;line-height:1.1}
.loss-view.compact .loss-table th,.loss-view.compact .loss-table td{padding:3px 5px}
.loss-view.compact .loss-table caption{font-size:10px;padding-bottom:3px}
.loss-view.compact .loss-table thead th{font-size:10px}
</style>`;

  const text = (x, y, value, attributes = '') => `<text x="${x}" y="${y}" ${attributes}>${esc(value)}</text>`;
  const line = (x1, y1, x2, y2, color, width = 1, attributes = '') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}" ${attributes}/>`;
  const dot = (x, y, color, radius = 3.5, attributes = '') => `<circle cx="${x}" cy="${y}" r="${radius}" fill="${color}" stroke="white" stroke-width="1.2" ${attributes}/>`;
  const diamond = (x, y, color, radius = 4) => `<path d="M${x},${y - radius}l${radius},${radius}l${-radius},${radius}l${-radius},${-radius}Z" fill="white" stroke="${color}" stroke-width="1.5"/>`;
  const path = (points, color, width = 2.3, attributes = '') => `<polyline points="${points.map(point => point.join(',')).join(' ')}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linejoin="round" ${attributes}/>`;

  function dimensions(state, compact, minimum = 205) {
    const W = Math.max(200, number(state.width, compact ? 346 : 620));
    const H = Math.max(minimum, number(state.height, compact ? 245 : 400) - (compact ? 30 : 26));
    return { W, H };
  }

  function svg(W, H, scene, description, body, attributes = '') {
    return `<svg class="loss-plot" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="loss-plot-title loss-plot-description" ${attributes}><title id="loss-plot-title">${esc(topics[scene])}</title><desc id="loss-plot-description">${esc(description)}</desc><style>.loss-plot text{font-family:var(--sans,Arial,sans-serif);font-size:12px;fill:${C.ink}}.loss-plot .lv-small{font-size:10px;fill:${C.muted}}.loss-plot .lv-title{font-size:13px;font-weight:600}.loss-plot .lv-gradient{fill:${C.teal}}.loss-plot .lv-residual{fill:${C.rust}}.loss-plot .lv-correction{fill:${C.blue}}.loss-plot .lv-label{paint-order:stroke;stroke:white;stroke-width:3px;stroke-linejoin:round}.loss-view.compact .loss-plot text{font-size:11px}.loss-view.compact .loss-plot .lv-small{font-size:10px}.loss-view.compact .loss-plot .lv-title{font-size:11px}</style>${body}</svg>`;
  }

  function ticks(minimum, maximum, count = 4) {
    const span = maximum - minimum || 1, rough = span / count, power = Math.pow(10, Math.floor(Math.log10(rough)));
    const step = [1, 2, 5, 10].map(factor => factor * power).find(value => value >= rough);
    const values = [];
    for (let value = Math.ceil(minimum / step) * step; value <= maximum + step * 1e-8; value += step) values.push(Math.abs(value) < step * 1e-8 ? 0 : value);
    return values;
  }

  function targetPlot(q, state, compact) {
    const { W, H } = dimensions(state, compact, 170);
    const residual = q.before.residual, targets = q.before.targets;
    let minimum = Math.min(0, ...residual, ...targets), maximum = Math.max(0, ...residual, ...targets);
    if (maximum === minimum) { minimum -= 1; maximum += 1; }
    const padding = .05 * (maximum - minimum);
    minimum -= padding; maximum += padding;
    const box = { x: 30, y: 28, w: W - 112, h: H - 58 };
    const X = value => box.x + (value - minimum) / (maximum - minimum) * box.w;
    const row = box.h / q.model.x.length;
    let body = text(4, 15, `F${sub(q.round)}${q.shared ? '; all x = 1' : ''}`, 'class="lv-title"') + text(W - 62, 15, 'raw', 'text-anchor="middle" class="lv-residual"') + text(W - 22, 15, '−∇ℓ', 'text-anchor="middle" class="lv-gradient"');
    body += text(4, 30, q.shared ? 'obs.' : 'x', 'class="lv-small"');
    ticks(minimum, maximum, compact ? 3 : 5).forEach(value => {
      body += line(X(value), box.y - 3, X(value), H - 27, value === 0 ? '#999' : C.grid, value === 0 ? 1.2 : 1);
      body += text(X(value), H - 12, fmt(value), 'class="lv-small" text-anchor="middle"');
    });
    q.model.x.forEach((input, index) => {
      const cy = box.y + (index + .5) * row;
      body += text(4, cy + 4, q.shared ? index + 1 : fmt(input), 'class="lv-small"');
      body += line(X(0), cy - 3, X(residual[index]), cy - 3, C.rust, 2, 'stroke-dasharray="3 2"') + diamond(X(residual[index]), cy - 3, C.rust, 3);
      body += line(X(0), cy + 3, X(targets[index]), cy + 3, C.teal, 2.5) + dot(X(targets[index]), cy + 3, C.teal, 3);
      body += text(W - 62, cy + 4, fmt(residual[index]), 'text-anchor="middle" class="lv-residual"');
      body += text(W - 22, cy + 4, fmt(targets[index]), 'text-anchor="middle" class="lv-gradient"');
    });
    const description = `${lossName(q.options)}. Predictions at round ${q.round}: ${fullVector(q.before.predictions)}. Observed responses: ${fullVector(q.model.y)}. Raw residuals: ${fullVector(residual)}. Negative gradients: ${fullVector(targets)}. Dashed rust bars with hollow diamonds are raw residuals. Solid teal bars with filled circles are negative gradients. All bars share one numerical scale.`;
    return svg(W, H, 0, description, body, `data-round="${q.round}" data-residuals="${residual.join(',')}" data-targets="${targets.join(',')}"`);
  }

  function splitTables(q, compact, state) {
    if (q.shared) return `<table class="loss-table loss-leaf-corrections"><caption>All x = 1: no split is available; one shared region</caption><thead><tr><th scope="col">Fit targets</th><th scope="col">Target mean</th><th scope="col">Added leaf c</th></tr></thead><tbody><tr><th scope="row">Negative gradient</th><td>${fmt(q.tree.targetLeft)}</td><td>${fmt(q.tree.left)}</td></tr><tr><th scope="row">Raw residual</th><td>${fmt(q.rawTree.targetLeft)}</td><td>${fmt(q.rawTree.left)}</td></tr></tbody></table><p class="loss-equation">h(x) = c for every observation</p><p class="loss-note">Gradient leaves use the original-loss optimum. The residual shortcut uses the raw residual mean.</p>`;
    const shortened = number(state.height, 400) < 245;
    const rows = q.after.candidates.map((candidate, index) => {
      const raw = q.rawCandidates[index], gradientBest = candidate.threshold === q.tree.threshold, rawBest = raw.threshold === q.rawTree.threshold;
      return `<tr data-cut="${candidate.threshold}"><th scope="row">${fmt(candidate.threshold)}</th><td class="${gradientBest ? 'loss-gradient-best' : ''}" data-gradient-sse="${candidate.sse}" aria-label="${esc(`Gradient target SSE ${full(candidate.sse)}${gradientBest ? ', selected' : ''}`)}">${fmt(candidate.sse)}${gradientBest ? ' ●' : ''}</td><td class="${rawBest ? 'loss-residual-best' : ''}" data-residual-sse="${raw.sse}" aria-label="${esc(`Residual SSE ${full(raw.sse)}${rawBest ? ', selected' : ''}`)}">${fmt(raw.sse)}${rawBest ? ' ◇' : ''}</td></tr>`;
    }).join('');
    const candidates = `<table class="loss-table loss-candidates"><caption>Fit at F${sub(q.round)}; ● / ◇ = selected</caption><thead><tr><th scope="col">Cut x</th><th scope="col">${compact ? 'Gradient SSE' : 'Gradient-target SSE'}</th><th scope="col">Residual SSE</th></tr></thead><tbody>${rows}</tbody></table>`;
    const leaves = `<table class="loss-table loss-leaf-corrections"><caption>Gradient-selected leaves, cut ${fmt(q.tree.threshold)}</caption><thead><tr><th scope="col">Leaf</th><th scope="col">Mean −∇ℓ</th><th scope="col">Optimal c</th></tr></thead><tbody><tr><th scope="row">Left</th><td>${fmt(q.tree.targetLeft)}</td><td>${fmt(q.tree.left)}</td></tr><tr><th scope="row">Right</th><td>${fmt(q.tree.targetRight)}</td><td>${fmt(q.tree.right)}</td></tr></tbody></table>`;
    const description = q.after.candidates.map((candidate, index) => `Cut ${full(candidate.threshold)}: gradient target SSE ${full(candidate.sse)}, residual SSE ${full(q.rawCandidates[index].sse)}.`).join(' ');
    return candidates + (shortened ? '' : leaves) + `<span class="loss-sr-only">All candidate target-fitting errors at the same current predictions: ${esc(description)}</span>`;
  }

  function correctionPlot(q, state, compact) {
    const stacked = !q.shared && (compact || number(state.width, 620) < 440);
    const { W, H } = dimensions(state, compact, stacked ? 198 : 155);
    const gap = stacked ? 7 : 20;
    const panelW = q.shared || stacked ? W : (W - gap) / 2, panelH = stacked ? (H - gap) / 2 : H;
    let body = '';
    const descriptions = [];
    (q.shared ? [true] : [true, false]).forEach((left, panel) => {
      const ox = stacked ? 0 : panel * (panelW + gap), oy = stacked ? panel * (panelH + gap) : 0;
      const subset = q.before.residual.filter((_, index) => q.shared || (q.model.x[index] <= q.tree.threshold) === left);
      const mean = left ? q.tree.targetLeft : q.tree.targetRight, optimum = left ? q.tree.left : q.tree.right;
      const width = Math.max(.75, Math.abs(mean - optimum) * 1.7, Math.min(4, Math.max(...subset) - Math.min(...subset)) * .65, q.options.loss === 'huber' ? q.options.delta : 0);
      const considered = [mean, optimum, 0, ...(q.shared ? [q.rawTree.left] : [])];
      const minimum = Math.min(...considered) - width, maximum = Math.max(...considered) + width;
      const valueAt = c => subset.reduce((sum, residual) => sum + E.pointLoss(residual, c, q.options), 0);
      const samples = Array.from({ length: 101 }, (_, index) => { const c = minimum + index * (maximum - minimum) / 100; return [c, valueAt(c)]; });
      const minLoss = valueAt(optimum), maxLoss = Math.max(...samples.map(sample => sample[1]));
      const lossPadding = .1 * (maxLoss - minLoss || 1);
      const loY = Math.max(0, minLoss - lossPadding), hiY = maxLoss + lossPadding;
      const box = { x: ox + 35, y: oy + (stacked ? 31 : 48), w: panelW - 47, h: panelH - (stacked ? 52 : 78) };
      const X = c => box.x + (c - minimum) / (maximum - minimum) * box.w;
      const Y = value => box.y + box.h - (value - loY) / (hiY - loY) * box.h;
      body += text(ox + 4, oy + 13, `${q.shared ? 'Shared region' : left ? 'Left leaf' : 'Right leaf'}: ${subset.length} ${subset.length === 1 ? 'point' : 'points'}`, 'class="lv-title"');
      body += text(ox + 4, oy + (stacked ? 26 : 30), `mean ${fmt(mean)}   →   optimal c ${fmt(optimum)}`, 'class="lv-small"');
      ticks(loY, hiY, stacked ? 2 : 3).forEach(value => { body += line(box.x, Y(value), box.x + box.w, Y(value), C.grid) + text(box.x - 5, Y(value) + 3, fmt(value), 'class="lv-small" text-anchor="end"'); });
      body += line(box.x, box.y + box.h, box.x + box.w, box.y + box.h, '#999');
      const cTicks = [minimum, 0, maximum].filter((value, index, values) => values.findIndex(other => Math.abs(other - value) < 1e-10) === index);
      cTicks.forEach(value => { body += text(X(value), oy + panelH - 5, fmt(value), 'class="lv-small" text-anchor="middle"'); });
      body += path(samples.map(([c, loss]) => [X(c), Y(loss)]), C.ink, 2);
      body += line(X(mean), box.y, X(mean), box.y + box.h, C.teal, 1.2, 'stroke-dasharray="3 3"');
      body += line(X(optimum), box.y, X(optimum), box.y + box.h, C.blue, 1.2);
      body += diamond(X(mean), Y(valueAt(mean)), C.teal, 4) + dot(X(optimum), Y(minLoss), C.blue, 4);
      if (q.shared) body += line(X(q.rawTree.left), box.y, X(q.rawTree.left), box.y + box.h, C.rust, 1.2, 'stroke-dasharray="5 3"') + diamond(X(q.rawTree.left), Y(valueAt(q.rawTree.left)), C.rust, 4);
      body += text(box.x + box.w - 2, box.y + box.h - 5, 'c', 'class="lv-small" text-anchor="end"');
      descriptions.push(`${q.shared ? 'Shared region' : left ? 'Left leaf' : 'Right leaf'} original residuals ${fullVector(subset)}. Mean negative gradient ${full(mean)}, original-loss total there ${full(valueAt(mean))}. Loss-optimal correction ${full(optimum)}, minimized leaf loss ${full(minLoss)}. Scaled correction actually added ${full(q.options.rate * optimum)}.${q.shared ? ` Raw residual mean ${full(q.rawTree.left)}; original loss at that correction ${full(valueAt(q.rawTree.left))}.` : ''}`);
    });
    return svg(W, H, 2, `Original loss summed ${q.shared ? 'in the shared region' : `separately in each gradient-selected leaf, cut ${full(q.tree.threshold)}`}, as a function of unscaled correction c. Each leaf has its own vertical loss scale. Hollow teal diamonds mark mean gradient targets; filled blue circles mark loss-optimal corrections. ${descriptions.join(' ')}`, body, `data-cut="${q.tree.threshold}" data-left-correction="${q.tree.left}" data-right-correction="${q.tree.right}"`);
  }

  function trajectories(q, state, compact) {
    const { W, H } = dimensions(state, compact, 180);
    const margin = compact ? 26 : 38, top = 32, upperHeight = Math.max(52, (H - 80) * .53);
    const panelW = W / 2, inner = compact ? 25 : 33;
    const allValues = [...q.model.y, ...q.model.gradient.flatMap(s => s.predictions), ...q.model.residual.flatMap(s => s.predictions)];
    const low = Math.min(...allValues), high = Math.max(...allValues), padding = (high - low || 1) * .08;
    const minimum = low - padding, maximum = high + padding;
    const inputs = q.shared ? q.model.x.map((_, index) => index + 1) : q.model.x;
    const firstX = inputs[0] - .5, lastX = inputs[inputs.length - 1] + .5;
    let body = text(4, 14, `Predictions after ${q.round} ${q.round === 1 ? 'tree' : 'trees'}`, 'class="lv-title"');
    [q.gradient, q.residual].forEach((current, panel) => {
      const box = { x: panel * panelW + inner, y: top, w: panelW - inner - 7, h: upperHeight };
      const X = x => box.x + (x - firstX) / (lastX - firstX) * box.w;
      const Y = y => box.y + box.h - (y - minimum) / (maximum - minimum) * box.h;
      body += text(panel * panelW + inner, 28, panel === 0 ? 'Gradient path' : 'Residual path', `class="lv-small ${panel === 0 ? 'lv-gradient' : 'lv-residual'}"`);
      ticks(minimum, maximum, compact ? 2 : 4).forEach(value => { body += line(box.x, Y(value), box.x + box.w, Y(value), C.grid) + text(box.x - 4, Y(value) + 3, fmt(value), 'class="lv-small" text-anchor="end"'); });
      const points = [[X(firstX), Y(current.predictions[0])]];
      current.predictions.forEach((value, index) => {
        const boundary = index === current.predictions.length - 1 ? lastX : (inputs[index] + inputs[index + 1]) / 2;
        points.push([X(boundary), Y(value)]);
        if (index < current.predictions.length - 1) points.push([X(boundary), Y(current.predictions[index + 1])]);
      });
      body += path(points, panel === 0 ? C.teal : C.rust, 2.5, panel === 0 ? 'data-curve="gradient-prediction"' : 'stroke-dasharray="5 4" data-curve="residual-prediction"');
      inputs.forEach((input, index) => {
        body += dot(X(input), Y(q.model.y[index]), C.ink, compact ? 2.8 : 4);
        body += text(X(input), box.y + box.h + 12, fmt(input), 'class="lv-small" text-anchor="middle"');
      });
    });
    if (q.shared) body += text(W / 2, top + upperHeight + 23, 'Observation index; every x = 1', 'class="lv-small" text-anchor="middle"');
    else body += text(W - 1, top + upperHeight + 12, 'x', 'class="lv-small" text-anchor="end"');
    const historyY = top + upperHeight + (q.shared ? 42 : 32);
    const history = { x: margin, y: historyY, w: W - margin - 12, h: Math.max(30, H - historyY - 16) };
    const maxLoss = Math.max(...q.model.gradient.slice(0, 25).map(s => s.loss), ...q.model.residual.slice(0, 25).map(s => s.loss), 1e-9);
    const HX = round => history.x + round / 24 * history.w, HY = loss => history.y + history.h * (1 - loss / (maxLoss * 1.08));
    body += text(4, history.y - 8, 'Chosen-loss total by round', 'class="lv-title"');
    ticks(0, maxLoss, compact ? 2 : 3).forEach(value => { body += line(history.x, HY(value), history.x + history.w, HY(value), C.grid) + text(history.x - 7, HY(value) + 3, fmt(value), 'class="lv-small" text-anchor="end"'); });
    body += line(HX(q.round), history.y, HX(q.round), history.y + history.h, '#999', 1, 'stroke-dasharray="2 3"');
    body += path(q.model.residual.slice(0, 25).map(s => [HX(s.round), HY(s.loss)]), C.rust, 2, 'stroke-dasharray="5 4" data-curve="residual-loss"');
    body += path(q.model.gradient.slice(0, 25).map(s => [HX(s.round), HY(s.loss)]), C.teal, 2.3, 'data-curve="gradient-loss"');
    body += diamond(HX(q.round), HY(q.residual.loss), C.rust, 4) + dot(HX(q.round), HY(q.gradient.loss), C.teal, 4);
    [0, 12, 24].forEach(round => { body += text(HX(round), H - 4, round, 'class="lv-small" text-anchor="middle"'); });
    const description = `${lossName(q.options)}; alpha ${full(q.options.rate)}. Both paths begin at the same constant ${full(q.model.initial)}. At round ${q.round}, gradient-path predictions ${fullVector(q.gradient.predictions)}, total chosen loss ${full(q.gradient.loss)}. Residual-path predictions ${fullVector(q.residual.predictions)}, total chosen loss ${full(q.residual.loss)}. Black dots are observed responses ${fullVector(q.model.y)}.${q.shared ? ' Horizontal positions are observation indices; all predictor values equal 1, so each fitted model is constant.' : ''} Gradient-path losses at rounds zero through 24: ${fullVector(q.model.gradient.slice(0, 25).map(s => s.loss))}. Residual-path losses at rounds zero through 24: ${fullVector(q.model.residual.slice(0, 25).map(s => s.loss))}. Both prediction panels share the same fixed response scale. Prediction and loss axes remain fixed across rounds.`;
    return svg(W, H, 3, description, body, `data-round="${q.round}" data-gradient-predictions="${q.gradient.predictions.join(',')}" data-residual-predictions="${q.residual.predictions.join(',')}" data-gradient-loss="${q.gradient.loss}" data-residual-loss="${q.residual.loss}"`);
  }

  function render(scene, state = {}, mobile = false) {
    validateScene(scene);
    const compact = mobile || number(state.width, 620) < 440;
    const q = resolve(state);
    let body;
    if (scene === 0) body = targetPlot(q, state, compact) + '<p class="loss-note">Hollow diamonds: raw residuals. Filled circles: negative gradients. Same scale.</p>';
    if (scene === 1) body = splitTables(q, compact, state) + (q.shared ? '' : '<p class="loss-note">Compare cuts within each column; its targets define its SSE scale.</p>');
    if (scene === 2) body = correctionPlot(q, state, compact) + `<p class="loss-note">◇ Mean gradient target &nbsp; ● Loss optimum.${q.shared ? ' Rust ◇: residual mean.' : ' Separate vertical scales.'}</p>`;
    if (scene === 3) body = trajectories(q, state, compact) + '<p class="loss-note">Black dots: fixed y. Same starting constant, observations, tree depth, and learning rate.</p>';
    return styles + `<div class="loss-view${compact ? ' compact' : ''}" data-loss-scene="${scene}" data-loss="${esc(q.options.loss)}" data-round="${q.round}">${body}</div>`;
  }

  return { topics, render, presentation, readout };
});
