(function (root, factory) {
  'use strict';
  const example = root && root.WorkedBoostingExample || (typeof require === 'function' ? require('./worked-example.js') : null);
  const api = factory(example);
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.WorkedBoostingView = api;
})(typeof window !== 'undefined' ? window : globalThis, function (E) {
  'use strict';
  if (!E) throw new Error('Load WorkedBoostingExample before WorkedBoostingView.');

  const C = { ink: '#242424', muted: '#606060', blue: '#234e70', rust: '#a54f32', teal: '#007f73', grid: '#e6e6e6' };
  const topics = ['Four points, one mean', 'Differentiate the predictions', 'Fit the first stump', 'Add the first stump', 'Fit the next residuals', 'Add the second stump', 'Repeat with fresh gradients', 'The gradient boosting loop'];
  const captions = [
    'Each black dot is a fixed observed response. The blue line starts at their mean, 2.',
    'Differentiate with respect to predictions zᵢ. The negative gradient is the residual yᵢ − zᵢ.',
    'Compare all three midpoint cuts by residual SSE. Each leaf predicts its residual mean.',
    'Keep y fixed and add the scaled stump to every prediction in its leaf.',
    'Recompute y − F₁ before comparing the same three possible cuts.',
    'The stump corrects its whole leaf. A single error can grow while total loss falls.',
    'Every additional tree is fitted to the current negative gradients. The observed responses and plot axes stay fixed.',
    'Choose a differentiable loss, fit the current negative gradients, and add a scaled tree. Recompute before the next round.'
  ];
  const esc = value => String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
  const sub = value => String(value).replace(/\d/g, digit => '₀₁₂₃₄₅₆₇₈₉'[digit]);
  const clamp = value => Math.max(0, Math.min(1, value));
  const fmt = value => E.formatFraction(value).replace(/-/g, '−');
  const signed = value => value > 1e-12 ? '+' + fmt(value) : fmt(value);
  const vector = (values, formatter = fmt) => '(' + values.map(formatter).join(', ') + ')';
  const intermediate = value => Math.abs(value) < 1e-10 ? '0' : String(Number(value.toFixed(3))).replace('-', '−');
  const concise = value => value === 0 ? '0' : String(Number(value.toPrecision(4))).replace('-', '−');
  const predictionFormat = q => q.repeat && q.round > 2 ? concise : q.updating && q.progress > 0 && q.progress < 1 ? intermediate : fmt;
  const text = (x, y, value, attributes = '') => `<text x="${x}" y="${y}" ${attributes}>${esc(value)}</text>`;
  const line = (x1, y1, x2, y2, color, width = 1, attributes = '') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}" ${attributes}/>`;
  const circle = (x, y, color, attributes = '', radius = 4) => `<circle cx="${x}" cy="${y}" r="${radius}" fill="${color}" stroke="white" stroke-width="1.2" ${attributes}/>`;
  const cutText = value => String(value).replace('-', '−');
  const rateOf = state => state.rate === .5 ? .5 : 1;
  const progressOf = state => Number.isFinite(state.process) ? clamp(state.process) : 1;
  const roundOf = state => Number.isFinite(state.round) ? Math.max(0, Math.min(24, Math.round(state.round))) : 8;

  function situation(scene, state) {
    const repeat = scene === 6;
    const rate = rateOf(state), states = E.build(rate, repeat ? 24 : 2);
    const round = repeat ? roundOf(state) : scene >= 4 ? 2 : 1;
    const current = states[round], before = states[Math.max(0, round - 1)];
    const fitting = scene === 2 || scene === 4;
    const updating = scene === 3 || scene === 5;
    const index = Number.isInteger(state.candidateIndex) ? state.candidateIndex : null;
    const candidate = fitting && index !== null ? current.candidates[index] || null : null;
    const chosen = candidate || current.tree;
    const progress = updating ? progressOf(state) : 1;
    const prediction = repeat ? current.predictions : scene <= 2 ? states[0].predictions : scene === 4 ? states[1].predictions : current.previous.map((value, i) => value + progress * current.step[i]);
    return { rate, states, round, current, before, fitting, updating, repeat, candidate, candidateIndex: index, chosen, progress, prediction };
  }

  function readout(scene, state = {}) {
    if (scene === 7) return [];
    const q = situation(scene, state);
    if (scene === 0) return [['Initial prediction F₀', vector(q.states[0].predictions), C.blue], ['Total loss ½Σ(yᵢ − zᵢ)²', '7']];
    if (scene === 1) return [['Gradient ∇L(z)', vector(q.states[0].gradient), C.rust], ['Negative gradient = residual', vector(q.states[0].residual), C.teal]];
    if (scene === 6) return [
      [`Prediction F${sub(q.round)}${q.round > 2 ? ' (rounded)' : ''}`, vector(q.prediction, predictionFormat(q)), C.blue],
      ['Total training loss', q.round <= 2 ? fmt(q.current.loss) : concise(q.current.loss)]
    ];
    if (q.fitting) return [
      [q.candidate ? 'Candidate cut' : 'Best cut', `x = ${cutText(q.chosen.threshold)}`, C.teal],
      ['Residual SSE for this stump', fmt(q.chosen.sse)],
      [`Current model loss at F${sub(q.round - 1)}`, fmt(q.before.loss), C.blue]
    ];
    return [
      [`Prediction F${sub(q.round)}`, vector(q.prediction, predictionFormat(q)), C.blue],
      ['Total loss: before → now', `${fmt(q.current.beforeLoss)} → ${predictionFormat(q)(E.loss(q.prediction))}`],
      ['Learning rate α', fmt(q.rate), C.teal]
    ];
  }

  function presentation(scene, state = {}) {
    if (scene === 7) return { topic: topics[scene], caption: captions[scene], readout: [] };
    const q = situation(scene, state);
    let caption = captions[scene];
    if (scene === 3 && q.rate === 1 && q.progress === 1) caption = 'The error at x = 2 grows from 0 to 1, while total half-squared loss falls from 7 to 1.';
    if (scene === 5 && q.rate === 1 && q.progress === 1) caption = 'At x = 4, the residual changes from 0 to −1/3. Total half-squared loss still falls from 1 to 1/3.';
    if (q.fitting && q.candidate) caption = 'Inspect this candidate’s leaves and residual SSE. The ensemble update still uses the best stump.';
    if (scene === 4 && q.rate === .5 && !q.candidate) caption = 'With α = 1/2, cuts 1.5 and 3.5 tie. The first cut, 1.5, is selected.';
    if (scene === 6) caption = `${q.round} ${q.round === 1 ? 'tree' : 'trees'} at α = ${fmt(q.rate)}. Each new tree is refitted to the current negative gradient y − F.`;
    if (q.rate === .5 && scene >= 2 && scene <= 5) caption += ' The prose numbers use α = 1.';
    return { topic: topics[scene], caption, readout: readout(scene, state) };
  }

  const scopedStyle = `<style>
.worked-view{width:100%;height:100%;min-width:0;display:flex;flex-direction:column;justify-content:center;gap:12px;font-family:var(--sans,Arial,sans-serif);font-variant-numeric:tabular-nums;color:${C.ink}}
.worked-view .worked-plot{width:100%;height:auto;flex:0 0 auto;min-height:0;max-height:none;overflow:visible}
.worked-view .worked-equation{margin:0;text-align:center;font:20px/1.4 var(--math,'Course Math','Cambria Math',serif);color:${C.ink}}
.worked-view .worked-note{margin:0;text-align:center;font-size:12px;line-height:1.4;color:${C.muted}}
.worked-view .worked-table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.2}
.worked-view .worked-table caption{text-align:left;font-size:13px;line-height:1.3;color:${C.muted};padding-bottom:6px}
.worked-view .worked-table th,.worked-view .worked-table td{text-align:right;padding:6px 8px;border-bottom:1px solid ${C.grid};font-weight:400}
.worked-view .worked-table th:first-child{text-align:left}
.worked-view .worked-table thead th{color:${C.muted};font-size:12px;border-bottom:1px solid #aaa;white-space:nowrap}
.worked-view .worked-table .worked-best th,.worked-view .worked-table .worked-best td{color:${C.teal};font-weight:600}
.worked-view .worked-table .worked-selected{background:#f0f6f5}
.worked-view .worked-gradient th,.worked-view .worked-gradient td{padding:10px 9px;font-size:17px}
.worked-view .worked-gradient td:nth-child(4){color:${C.rust}}
.worked-view .worked-gradient td:nth-child(5){color:${C.teal}}
.worked-view .worked-values th,.worked-view .worked-values td{padding:6px 8px}
.worked-view .worked-values .worked-predictions td{color:${C.blue}}
.worked-view .worked-leaves{display:flex;justify-content:space-around;gap:12px;margin:0;font:18px/1.35 var(--math,'Course Math','Cambria Math',serif);color:${C.teal}}
.worked-view .worked-leaf{display:flex;flex-direction:column;align-items:center;gap:1px}
.worked-view .worked-leaf small{font:11px/1.3 var(--sans,Arial,sans-serif);color:${C.muted}}
.worked-view .worked-loop{list-style:none;counter-reset:loop;padding:0;margin:0;display:flex;flex-direction:column;gap:13px}
.worked-view .worked-loop li{counter-increment:loop;padding-left:30px;position:relative;font-size:15px;line-height:1.3}
.worked-view .worked-loop li::before{content:counter(loop);position:absolute;left:0;top:0;color:${C.teal};font-weight:600}
.worked-view .worked-loop strong{display:block;font-weight:600}
.worked-view .worked-loop .loop-math{display:block;font:17px/1.4 var(--math,'Course Math','Cambria Math',serif);overflow-wrap:anywhere}
.worked-view .worked-loop .loop-detail{display:block;color:${C.muted};font-size:13px;line-height:1.35}
.worked-view.compact{gap:5px}
.worked-view.compact .worked-equation{font-size:16px;line-height:1.25}
.worked-view.compact .worked-table{font-size:12px;line-height:1.15}
.worked-view.compact .worked-table th,.worked-view.compact .worked-table td{padding:3px 5px}
.worked-view.compact .worked-table thead th{font-size:11px}
.worked-view.compact .worked-table caption{font-size:11px;padding-bottom:3px}
.worked-view.compact .worked-gradient th,.worked-view.compact .worked-gradient td{padding:7px 5px;font-size:14px}
.worked-view.compact .worked-leaves{gap:6px;font-size:14px;line-height:1.2}
.worked-view.compact .worked-leaf small{font-size:11px}
.worked-view.compact .worked-note{font-size:11px;line-height:1.25}
.worked-view.compact .worked-loop{gap:7px}
.worked-view.compact .worked-loop li{padding-left:22px;font-size:13px;line-height:1.2}
.worked-view.compact .worked-loop .loop-math{font-size:14px;line-height:1.25}
.worked-view.compact .worked-loop .loop-detail{font-size:11px;line-height:1.25}
</style>`;

  function plot(scene, q, compact, state) {
    const residual = q.fitting;
    const W = Number.isFinite(state.width) && state.width >= 200 ? state.width : compact ? 400 : 620;
    const available = Number.isFinite(state.height) ? state.height : compact ? 240 : 480;
    const extraHeight = scene === 0 || scene === 6 ? (compact ? 28 : 40) : residual ? (compact ? 146 : 184) : (compact ? 97 : 160);
    const H = compact ? Math.max(residual ? 100 : 118, Math.min(residual ? 138 : 184, available - extraHeight)) : Math.max(170, Math.min(288, available - extraHeight));
    const box = { x: compact ? 29 : 43, y: compact ? 21 : 36, w: W - (compact ? 46 : 72), h: H - (compact ? 44 : 67) };
    const minimum = residual ? -3.5 : -.5, maximum = residual ? 3.5 : 6;
    const X = value => box.x + (value - .5) / 4 * box.w;
    const Y = value => box.y + box.h - (value - minimum) / (maximum - minimum) * box.h;
    const round = q.round, shownFormat = predictionFormat(q);
    const title = residual ? `Residuals before tree ${round}` : scene === 0 ? 'Observed responses and F₀' : 'Responses and predictions';
    let body = compact ? '' : text(box.x, 20, title, 'class="wv-title"');
    const legendX = compact ? box.x : W - 190, legendY = compact ? 9 : 15;
    body += circle(legendX, legendY, C.ink) + text(legendX + 9, legendY + 4, residual ? 'residual' : 'y', 'class="wv-key"');
    body += line(legendX + (compact ? 64 : 79), legendY, legendX + (compact ? 79 : 99), legendY, residual ? C.teal : C.blue, 3) + text(legendX + (compact ? 85 : 105), legendY + 4, residual ? `h${sub(round)}` : `F${sub(scene === 0 ? 0 : round)}`, 'class="wv-key"');
    const ticks = compact ? (residual ? [-3, 0, 3] : [0, 2, 4, 6]) : residual ? [-3, -2, -1, 0, 1, 2, 3] : [0, 1, 2, 3, 4, 5, 6];
    ticks.forEach(value => { body += line(box.x, Y(value), box.x + box.w, Y(value), value === 0 ? '#999' : C.grid, value === 0 ? 1.1 : 1) + text(box.x - 8, Y(value) + 4, fmt(value), 'text-anchor="end" class="wv-tick"'); });
    body += line(box.x, box.y, box.x, box.y + box.h, '#999');
    E.x.forEach(value => { body += line(X(value), box.y + box.h, X(value), box.y + box.h + 3, '#999') + text(X(value), H - 5, fmt(value), 'class="wv-tick" text-anchor="middle"'); });
    body += text(W - 4, H - 5, 'x', 'class="wv-tick" text-anchor="end"');
    function curve(values, cuts, color, kind, dashed = false) {
      const bounds = [.5, ...cuts, 4.5];
      let result = `<g data-curve="${kind}" data-values="${values.join(',')}" data-cuts="${cuts.join(',')}">`;
      values.forEach((value, index) => {
        result += line(X(bounds[index]), Y(value), X(bounds[index + 1]), Y(value), color, 3, dashed ? 'stroke-dasharray="5 4"' : '');
        if (index > 0) result += line(X(bounds[index]), Y(values[index - 1]), X(bounds[index]), Y(value), color, 1.3, dashed ? 'stroke-dasharray="3 3"' : '');
      });
      return result + '</g>';
    }
    function accumulated(count, progress = 1) {
      const cuts = [...new Set(q.states.slice(1, count + 1).map(state => state.tree.threshold))].sort((a, b) => a - b);
      const bounds = [.5, ...cuts, 4.5];
      const values = bounds.slice(0, -1).map((lo, index) => {
        const input = (lo + bounds[index + 1]) / 2;
        return E.baseline + q.states.slice(1, count + 1).reduce((sum, state, j) => sum + q.rate * (j === count - 1 ? progress : 1) * (input <= state.tree.threshold ? state.tree.left : state.tree.right), 0);
      });
      return { values, cuts };
    }
    if (residual) {
      const cut = q.chosen.threshold;
      body += line(X(cut), box.y, X(cut), box.y + box.h, '#999', 1, 'stroke-dasharray="3 4"');
      body += text(X(cut) + 5, box.y + box.h - 5, `cut ${cutText(cut)}`, 'class="wv-cut"');
      body += curve([q.chosen.left, q.chosen.right], [cut], C.teal, q.candidate ? 'candidate' : 'selected-stump');
      body += `<g data-points="residuals" data-values="${q.before.residual.join(',')}">`;
      E.x.forEach((input, index) => { body += circle(X(input), Y(q.before.residual[index]), C.ink) + text(X(input) + 7, Y(q.before.residual[index]) + 13, fmt(q.before.residual[index]), 'class="wv-point-value"'); });
      body += '</g>';
    } else {
      const prediction = q.prediction;
      if (q.updating) {
        const old = accumulated(round - 1);
        body += curve(old.values, old.cuts, '#888', 'previous-prediction', true);
        body += `<g data-errors="remaining" data-predictions="${prediction.join(',')}">`;
        E.x.forEach((input, index) => { body += line(X(input), Y(E.y[index]), X(input), Y(prediction[index]), '#b0b0b0', 1.4); });
        body += '</g>';
      }
      const current = accumulated(scene === 0 ? 0 : round, q.progress);
      body += curve(current.values, current.cuts, C.blue, 'prediction');
      const bounds = [.5, ...current.cuts, 4.5];
      current.values.forEach((value, index) => {
        const center = (bounds[index] + bounds[index + 1]) / 2;
        body += text(X(center), Y(value) - 10, shownFormat(value), `class="wv-prediction-value" fill="${C.blue}" text-anchor="middle"`);
      });
      body += `<g data-points="observations" data-values="${E.y.join(',')}">`;
      E.x.forEach((input, index) => { body += circle(X(input), Y(E.y[index]), C.ink) + text(X(input) + 7, Y(E.y[index]) + 14, fmt(E.y[index]), 'class="wv-point-value"'); });
      body += '</g>';
      if (q.updating && q.progress === 1 && q.rate === 1) {
        const i = round === 1 ? 1 : 3;
        body += `<g data-worsened-error="${E.x[i]}">` + circle(X(E.x[i]), Y(E.y[i]), 'none', `style="stroke:${C.rust};stroke-width:1.6"`, 6) + '</g>';
      }
    }
    const description = presentation(scene, { rate: q.rate, process: q.progress, candidateIndex: q.candidateIndex, round: q.round }).caption + (q.repeat ? ` Current predictions: ${vector(q.prediction, predictionFormat(q))}. Current negative gradients: ${vector(q.current.residual, predictionFormat(q))}. Total training loss: ${concise(q.current.loss)}.` : '');
    return `<svg class="worked-plot" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="worked-plot-title worked-plot-desc" data-axis="${residual ? 'residual' : 'response'}" data-x-domain="0.5,4.5" data-y-domain="${minimum},${maximum}" data-rate="${q.rate}" data-progress="${q.progress}" data-predictions="${q.prediction.join(',')}" data-negative-gradient="${E.y.map((value, i) => value - q.prediction[i]).join(',')}" data-training-loss="${E.loss(q.prediction)}"><title id="worked-plot-title">${esc(title)}</title><desc id="worked-plot-desc">${esc(description)}</desc><style>.worked-plot text{font-family:Arial,sans-serif;fill:${C.ink};font-size:${compact ? 11 : 14}px}.worked-plot .wv-title{font-family:Georgia,serif;font-size:${compact ? 12 : 19}px}.worked-plot .wv-key,.worked-plot .wv-tick{font-size:${compact ? 11 : 12}px;fill:${C.muted}}.worked-plot .wv-prediction-value{fill:${C.blue};font-family:'Course Math',serif;font-size:${compact ? 14 : 20}px}.worked-plot .wv-point-value,.worked-plot .wv-cut,.worked-plot .wv-prediction-value{paint-order:stroke;stroke:white;stroke-width:4px;stroke-linejoin:round}.worked-plot .wv-cut{font-size:${compact ? 11 : 12}px;fill:${C.muted}}</style>${body}</svg>`;
  }

  function gradientTable(q, compact) {
    const header = compact ? '<th scope="col">i</th><th scope="col">zᵢ</th><th scope="col">yᵢ</th><th scope="col">gᵢ</th><th scope="col">rᵢ</th>' : '<th scope="col">i</th><th scope="col">zᵢ = F₀(xᵢ)</th><th scope="col">yᵢ</th><th scope="col">gᵢ = zᵢ − yᵢ</th><th scope="col">rᵢ = −gᵢ</th>';
    return '<table class="worked-table worked-gradient"><caption>One derivative for each prediction coordinate</caption><thead><tr>' + header + '</tr></thead><tbody>' + E.x.map((_, i) => `<tr><th scope="row">${i + 1}</th><td>${fmt(E.baseline)}</td><td>${fmt(E.y[i])}</td><td>${fmt(q.states[0].gradient[i])}</td><td>${fmt(q.states[0].residual[i])}</td></tr>`).join('') + '</tbody></table>';
  }

  function candidateTable(q, compact) {
    const headers = compact ? ['Cut x','Left','Right','SSE'] : ['Cut x','Left mean','Right mean','Residual SSE'];
    return '<table class="worked-table worked-candidates"><caption>Residual SSE = Σ(rᵢ − h(xᵢ))²; * = minimum</caption><thead><tr>' + headers.map(value => `<th scope="col">${value}</th>`).join('') + '</tr></thead><tbody>' + q.current.candidates.map((candidate, index) => {
      const best = Math.abs(candidate.sse - q.current.tree.sse) < 1e-9;
      const selected = q.chosen === candidate || (!q.candidate && candidate.threshold === q.current.tree.threshold);
      return `<tr class="${best ? 'worked-best' : ''}${selected ? ' worked-selected' : ''}" data-candidate="${index}" data-candidate-sse="${candidate.sse}"${selected ? ' aria-current="true"' : ''}><th scope="row">${cutText(candidate.threshold)}${best ? ' *' : ''}</th><td>${fmt(candidate.left)}</td><td>${fmt(candidate.right)}</td><td>${fmt(candidate.sse)}</td></tr>`;
    }).join('') + '</tbody></table>';
  }

  function leafCalculation(q) {
    return '<div class="worked-leaves" aria-label="Leaf means">' + [true, false].map(left => {
      const values = q.before.residual.filter((_, i) => (E.x[i] <= q.chosen.threshold) === left);
      const numerator = values.map(fmt).join(' + ').replace(/\+ −/g, '− ');
      const mean = left ? q.chosen.left : q.chosen.right;
      const equation = values.length === 1 ? fmt(mean) : `(${numerator}) / ${values.length} = ${fmt(mean)}`;
      return `<span class="worked-leaf"><small>${left ? 'Left' : 'Right'} leaf: ${values.length} ${values.length === 1 ? 'point' : 'points'}</small><span>${esc(equation)}</span></span>`;
    }).join('') + '</div>';
  }

  function valueTable(q, compact) {
    const shownFormat = predictionFormat(q);
    const residual = E.y.map((value, i) => value - q.prediction[i]);
    return `<table class="worked-table worked-values"><thead><tr><th scope="col">Input x</th>${E.x.map(value => `<th scope="col">${fmt(value)}</th>`).join('')}</tr></thead><tbody>${compact ? '' : `<tr><th scope="row">Fixed y</th>${E.y.map(value => `<td>${fmt(value)}</td>`).join('')}</tr>`}<tr class="worked-predictions"><th scope="row">F${sub(q.round)}</th>${q.prediction.map(value => `<td>${shownFormat(value)}</td>`).join('')}</tr><tr><th scope="row">y − F${sub(q.round)}</th>${residual.map(value => `<td>${shownFormat(value)}</td>`).join('')}</tr></tbody></table>`;
  }

  function render(scene, state = {}, compact = false) {
    if (!Number.isInteger(scene) || scene < 0 || scene > 7) throw new RangeError('Worked scene must be an integer from 0 to 7.');
    if (scene === 7) return scopedStyle + `<div class="worked-view${compact ? ' compact' : ''}" data-worked-scene="7"><ol class="worked-loop" aria-label="Gradient boosting at round m"><li><strong>Predict</strong><span class="loop-math">zᵢ = Fₘ₋₁(xᵢ)</span></li><li><strong>Differentiate the loss</strong><span class="loop-math">rₘᵢ = −∂ℓ(yᵢ, z) / ∂z | z = zᵢ</span></li><li><strong>Fit a tree to these targets</strong><span class="loop-detail">A squared-error fit to rₘᵢ defines the leaves Rₘⱼ.</span></li><li><strong>Choose leaf corrections and add</strong><span class="loop-math">cₘⱼ = arg min<sub>c</sub> Σ<sub>xᵢ∈Rₘⱼ</sub> ℓ(yᵢ, zᵢ + c)</span><span class="loop-math">hₘ(x) = Σⱼ cₘⱼ 1{x∈Rₘⱼ}</span><span class="loop-math">Fₘ = Fₘ₋₁ + α hₘ, &nbsp; 0 &lt; α ≤ 1</span></li><li><strong>Recompute and repeat</strong><span class="loop-detail">Use Fₘ to calculate the next negative gradients.</span></li></ol><p class="worked-note">For squared error, cₘⱼ is the leaf’s mean residual. Other losses can require different corrections. When no finite minimizer exists, constrain the correction to a finite closed interval or use a suitable penalty.</p></div>`;
    const q = situation(scene, state);
    let body;
    if (scene === 1) {
      body = '<p class="worked-equation">L(z) = ½ Σ(yᵢ − zᵢ)²</p><p class="worked-equation">gᵢ = ∂L / ∂zᵢ = zᵢ − yᵢ</p>' + gradientTable(q, compact) + '<p class="worked-note">zᵢ = F₀(xᵢ); rᵢ = −gᵢ. Differentiate the predictions z, with x fixed.</p>';
    } else {
      body = plot(scene, q, compact, state);
      if (scene === 0) body += '<p class="worked-equation">F₀(x) = (0 + 2 + 1 + 5) / 4 = 2</p>';
      else if (scene === 6) body += `<p class="worked-equation">F${sub(q.round)}(x) = 2 + α Σ<sub>m=1</sub><sup>${q.round}</sup> hₘ(x)</p>`;
      else if (q.fitting) body += leafCalculation(q) + candidateTable(q, compact);
      else body += `<p class="worked-equation">F${sub(q.round)} = F${sub(q.round - 1)} + ${predictionFormat(q)(q.rate * q.progress)} h${sub(q.round)}</p>` + valueTable(q, compact);
    }
    return scopedStyle + `<div class="worked-view${compact ? ' compact' : ''}" data-worked-scene="${scene}" data-rate="${q.rate}">${body}</div>`;
  }

  return { render, readout, presentation, topics, captions };
});
