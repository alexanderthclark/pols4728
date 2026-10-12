(function (root, factory) {
  'use strict';
  const api = factory(
    root.TreePartitions || (typeof require === 'function' ? require('./model.js') : null),
    root.GrowthExample || (typeof require === 'function' ? require('./growth.js') : null)
  );
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.GrowthView = api;
})(typeof window !== 'undefined' ? window : globalThis, function (P, G) {
  'use strict';
  const C = { ink: '#242424', muted: '#606060', blue: '#234E70', rust: '#A54F32', teal: '#007F73', rule: '#767676' };
  const clamp = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmt = (v, digits = 1) => String(Number(v.toFixed(digits))).replace('-', '−');
  const signed = v => `${v > 0 ? '+' : ''}${fmt(v, 2)}`;
  const sub = n => String(n).split('').map(d => '₀₁₂₃₄₅₆₇₈₉'[Number(d)]).join('');
  const text = (x, y, label, attrs = '') => `<text x="${x}" y="${y}" ${attrs}>${esc(label)}</text>`;
  const line = (x1, y1, x2, y2, color = C.rule, width = 1, attrs = '') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}" ${attrs}/>`;
  const stages = scene => scene === 0 ? G.treeStages : G.boostStages;
  const max = scene => scene === 2 ? 0 : stages(scene).length - 1;

  // One quantitative scale is shared by both fitted surfaces.
  // It is fixed across all stages so a color never changes its numerical meaning.
  const timeDomain = Object.freeze([8, 24]);
  function timeRGB(value) {
    const t = clamp((value - timeDomain[0]) / (timeDomain[1] - timeDomain[0]));
    return [237, 244, 250].map((v, i) => Math.round(v + ([35, 78, 112][i] - v) * t));
  }
  const timeColor = value => `rgb(${timeRGB(value).join(',')})`;
  function valueInk(value) {
    const linear = timeRGB(value).map(v => v / 255).map(v => v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4);
    const luminance = linear[0] * .2126 + linear[1] * .7152 + linear[2] * .0722;
    return luminance < .179 ? 'white' : '#000000';
  }
  function colorLegend(x, y, width, label, compact, quantity = 'predicted-F') {
    const h = compact ? 10 : 12;
    let out = `<g data-color-scale="8,24" data-color-quantity="${quantity}">`;
    out += text(x + width / 2, y - (compact ? 7 : 9), label, 'class="legend" text-anchor="middle"');
    // Solid adjacent strips also render consistently in exported SVG/PNG figures.
    for (let i = 0; i < 64; i++) out += `<rect x="${x + width * i / 64}" y="${y}" width="${width / 64 + .2}" height="${h}" fill="${timeColor(8 + 16 * (i + .5) / 64)}"/>`;
    for (const value of [8, 12, 16, 20, 24]) {
      const px = x + width * (value - 8) / 16;
      out += line(px, y + h, px, y + h + 3);
      out += text(px, y + h + (compact ? 15 : 19), value, 'class="legend" text-anchor="middle"');
    }
    return out + '</g>';
  }

  function current(scene, state) {
    const limit = max(scene);
    const progress = Number.isFinite(state.growth) ? clamp(state.growth, 0, limit) : limit;
    const index = Math.floor(progress + 1e-9);
    const amount = clamp(progress - index);
    const all = stages(scene);
    return { progress, index, amount, model: all[index], next: amount > 0 ? all[index + 1] : null };
  }

  function splitLabel(split) {
    return `x${sub(split.feature + 1)} ≤ ${fmt(split.threshold)}`;
  }

  function presentation(scene, state = {}) {
    if (scene === 2) return {
      heading: 'Compare the same split budget',
      equation: 'One tree: 4 splits · Additive stumps: 4 splits',
      takeaway: 'Same split budget, similar training error. Local cuts versus crossing cuts.'
    };
    const s = current(scene, state), n = s.index;
    if (scene === 0) return {
      heading: 'Grow one decision tree',
      equation: n === 0 ? `Predict the mean: ${fmt(G.baseline, 2)} minutes` : `${n} ${n === 1 ? 'split' : 'splits'} · ${s.model.regionCount} regions`,
      takeaway: s.next
        ? 'The new cut stops at the boundary of the leaf being split.'
        : n === 0 ? 'Now use two predictors: distance and the number of stops.'
          : n === max(0) ? 'Four split nodes create five leaves. Each leaf predicts its own mean.'
            : 'Choose the next split within one existing leaf.'
    };
    return {
      heading: 'Add one stump at a time',
      equation: n === 0 ? `F₀ = ${fmt(G.baseline, 2)}` : `F${sub(n)} = F${sub(n - 1)} + h${sub(n)}`,
      takeaway: s.next ? 'Fit a stump to the current residuals. Its cut crosses the whole input space.'
        : n === max(1) ? 'The four stump cuts cross, creating nine regions in the sum.'
          : 'Each stump adjusts both sides of a new cut, including previously formed regions.'
    };
  }

  function readout(scene, state = {}) {
    if (scene === 2) {
      const t = G.treeStages[max(0)], b = G.boostStages[max(1)];
      return `Training MSE: tree ${t.mse.toFixed(3)}, additive stumps ${b.mse.toFixed(3)} · ${t.splitCount} split nodes each`;
    }
    const s = current(scene, state);
    return `${s.next ? `Drawing split ${s.index + 1}; current fit: ` : ''}training MSE ${s.model.mse.toFixed(3)} · ${s.model.splitCount} ${s.model.splitCount === 1 ? 'split' : 'splits'} · ${s.model.regionCount} ${s.model.regionCount === 1 ? 'region' : 'regions'}`;
  }

  function render(scene, state = {}, compact = false) {
    const comparison = scene === 2, W = compact ? 420 : 780;
    const H = compact ? comparison ? 413 : 460 : 540;
    const s = comparison ? null : current(scene, state);
    const p = presentation(scene, state);
    const kind = ['tree', 'boost', 'compare'][scene];
    const nodeSize = compact ? 13 : 17;
    let body = '';
    const pointOverlays = [];

    function allCuts(tree, box = G.bounds) {
      if ('value' in tree) return [];
      const left = box.map(b => b.slice()), right = box.map(b => b.slice());
      left[tree.feature][1] = tree.threshold;
      right[tree.feature][0] = tree.threshold;
      return [{ feature: tree.feature, threshold: tree.threshold, bounds: box, parentId: tree.id }, ...allCuts(tree.left, left), ...allCuts(tree.right, right)];
    }

    function map(model, box, id, pending, amount) {
      const X = x => box.x + (x - G.bounds[0][0]) / (G.bounds[0][1] - G.bounds[0][0]) * box.w;
      const Y = y => box.y + box.h - (y - G.bounds[1][0]) / (G.bounds[1][1] - G.bounds[1][0]) * box.h;
      const cells = P.refinement(model.trees, G.bounds);
      let out = `<g data-map="${id}" data-mse="${model.mse}" data-split-count="${model.splitCount}" data-region-count="${model.regionCount}" data-predictions="${model.predictions.join(',')}">`;
      for (const cell of cells) {
        const [[lo, hi], [bottom, top]] = cell.bounds;
        out += `<rect x="${X(lo)}" y="${Y(top)}" width="${X(hi) - X(lo)}" height="${Y(bottom) - Y(top)}" fill="${timeColor(cell.value)}" data-region="${esc(cell.leafId)}" data-bounds="${cell.bounds.flat().join(',')}" data-value="${cell.value}"/>`;
      }
      const cut = (q, color, fraction, active) => {
        const [[lo, hi], [bottom, top]] = q.bounds;
        const feature = q.feature;
        const x1 = feature === 0 ? X(q.threshold) : X(lo), y1 = feature === 0 ? Y(bottom) : Y(q.threshold);
        const x2 = feature === 0 ? x1 : X(lo + (hi - lo) * fraction), y2 = feature === 0 ? Y(bottom + (top - bottom) * fraction) : y1;
        return (active ? line(x1, y1, x2, y2, 'white', 6, 'stroke-linecap="round"') : '') + line(x1, y1, x2, y2, color, active ? 3 : 1.6, `data-cut-feature="${feature}" data-cut-threshold="${q.threshold}" data-cut-bounds="${q.bounds.flat().join(',')}" data-pending="${active}" stroke-linecap="round"`);
      };
      for (const tree of model.trees) for (const q of allCuts(tree)) out += cut(q, C.blue, 1, false);
      if (pending) out += cut(pending, C.rust, clamp(amount / .82), true);
      out += `<rect x="${box.x}" y="${box.y}" width="${box.w}" height="${box.h}" fill="none" stroke="${C.rule}" stroke-width="1"/>`;
      for (const value of [0, 6, 12]) {
        out += line(X(value), box.y + box.h, X(value), box.y + box.h + 4);
        out += text(X(value), box.y + box.h + (compact ? 16 : 21), value, 'class="tick" text-anchor="middle"');
      }
      for (const value of [0, 2, 4]) {
        out += line(box.x - 4, Y(value), box.x, Y(value));
        out += text(box.x - 8, Y(value) + 4, value, 'class="tick" text-anchor="end"');
      }
      out += text(box.x + box.w / 2, box.y + box.h + (compact ? 34 : 47), compact ? 'Distance x₁ (km)' : 'Distance x₁ (km)', 'class="axis" text-anchor="middle"');
      const ax = box.x - (compact ? 29 : 37), ay = box.y + box.h / 2;
      out += text(ax, ay, 'Stops x₂', `class="axis" text-anchor="middle" transform="rotate(-90 ${ax} ${ay})"`);
      G.observations.forEach((observation, i) => {
        const px = X(observation.x[0]), py = Y(observation.x[1]);
        const prediction = model.predictions[i], residual = observation.y - prediction;
        const radius = compact ? comparison ? 3.3 : 3.8 : 5;
        out += `<circle cx="${px}" cy="${py}" r="${radius + 1.2}" fill="white" stroke="${C.blue}" stroke-width=".5"/>`;
        out += `<circle cx="${px}" cy="${py}" r="${radius}" fill="${timeColor(observation.y)}" stroke="${C.blue}" stroke-width=".5" data-observation="${observation.id}" data-x="${observation.x.join(',')}" data-y="${observation.y}" data-prediction="${prediction}" data-residual="${residual}"/>`;
        const tooltipWidth = compact ? comparison ? 149 : 167 : 216;
        const tooltipHeight = compact ? 61 : 77, padding = compact ? 8 : 11, lineHeight = compact ? 16 : 21;
        const tooltipX = clamp(px + 11, box.x + 3, box.x + box.w - tooltipWidth - 3);
        const tooltipY = py - tooltipHeight - 10 >= box.y ? py - tooltipHeight - 10 : py + 10;
        const label = `Delivery ${observation.id}: distance ${observation.x[0]} km, ${observation.x[1]} stops. Observed y ${observation.y} minutes. Prediction F ${fmt(prediction, 2)} minutes. Residual y minus F ${signed(residual)} minutes.`;
        // All transparent hit targets and tooltips are appended after the plots,
        // so a tooltip cannot be obscured by a later marker or comparison map.
        pointOverlays.push(`<g class="growth-observation" tabindex="0" role="img" aria-label="${esc(label)}" data-hover-observation="${id}-${observation.id}"><circle class="observation-hit" cx="${px}" cy="${py}" r="${compact ? 9 : 11}" fill="transparent"/><g class="observation-tooltip" aria-hidden="true"><rect x="${tooltipX}" y="${tooltipY}" width="${tooltipWidth}" height="${tooltipHeight}" rx="3" fill="white" stroke="${C.blue}" stroke-width="1"/>${[
          `Observed y = ${observation.y} min`,
          `Prediction F = ${fmt(prediction, 2)} min`,
          `y − F = ${signed(residual)} min`
        ].map((value, row) => text(tooltipX + padding, tooltipY + padding + (compact ? 10 : 13) + row * lineHeight, value, 'class="tooltip-value"')).join('')}</g></g>`);
      });
      for (const cell of cells) {
        const [[lo, hi], [bottom, top]] = cell.bounds;
        // Put predictions in the open space between the observed rows. The
        // uppermost, one-stop-high strip has room above the row of dots.
        let labelY = (bottom + top) / 2;
        if (bottom < 0 && top === 2) labelY = 1.62;
        else if (bottom === 2) labelY = 2.38;
        else if (bottom < 0 && top === 3.5) labelY = 2.03;
        else if (bottom === 3.5) labelY = 4.28;
        else if (bottom < 0 && top === 4.5) labelY = 2.1;
        let labelX = (lo + hi) / 2;
        if (pending && pending.feature === 0 && Math.abs(X(labelX) - X(pending.threshold)) < (compact ? 22 : 31) && pending.bounds[1][0] <= labelY && labelY <= pending.bounds[1][1]) labelX = (lo + pending.threshold) / 2;
        if (pending && pending.feature === 1 && Math.abs(Y(labelY) - Y(pending.threshold)) < (compact ? 13 : 20) && pending.bounds[0][0] <= labelX && labelX <= pending.bounds[0][1]) labelY = (pending.threshold + top) / 2;
        const tightTopStrip = compact && !comparison && bottom === 3.5;
        out += text(X(labelX), Y(labelY) + (compact ? 4 : 6), fmt(cell.value), `class="prediction" fill="${valueInk(cell.value)}" text-anchor="middle"${tightTopStrip ? ' style="font-size:12px"' : ''}`);
      }
      return out + '</g>';
    }

    function treeDiagram(model, next, amount) {
      const final = G.treeStages[max(0)].tree;
      const area = compact ? { x: 20, w: 380, y: 31, gap: 34 } : { x: 22, w: 340, y: 88, gap: 91 };
      const positions = new Map(), leaves = [];
      function collect(node, depth) {
        if ('value' in node) { leaves.push({ node, depth }); return; }
        collect(node.left, depth + 1); collect(node.right, depth + 1);
      }
      collect(final, 0);
      leaves.forEach(({ node, depth }, i) => positions.set(node.id, { x: area.x + (i + .5) * area.w / leaves.length, y: area.y + depth * area.gap }));
      function layout(node, depth) {
        if ('value' in node) return positions.get(node.id);
        const left = layout(node.left, depth + 1), right = layout(node.right, depth + 1);
        const p = { x: (left.x + right.x) / 2, y: area.y + depth * area.gap };
        positions.set(node.id, p); return p;
      }
      layout(final, 0);
      const pending = next && next.split, displayed = next ? next.tree : model.tree;
      const reveal = clamp(amount / .8);
      function draw(node, inPending = false) {
        const at = positions.get(node.id);
        if ('value' in node) {
          return inPending ? '' : text(at.x, at.y + 5, fmt(node.value, 1), 'class="leaf" text-anchor="middle"');
        }
        const active = pending && pending.parentId === node.id;
        const opacity = active ? reveal : 1;
        let out = `<g data-tree-node="${node.id}" opacity="${opacity}">`;
        out += text(at.x, at.y + 4, splitLabel(node), `class="node" text-anchor="middle" fill="${active ? C.rust : C.ink}"`);
        for (const [child, label] of [[node.left, '≤'], [node.right, '>']]) {
          const end = positions.get(child.id), t = active ? reveal : 1;
          const startY = at.y + (compact ? 8 : 13), endY = end.y - (compact ? 9 : 15);
          out += line(at.x, startY, at.x + (end.x - at.x) * t, startY + (endY - startY) * t, active ? C.rust : C.rule, compact ? 1.3 : 1.7);
          if (!compact) out += text((at.x + end.x) / 2 + (label === '≤' ? -12 : 12), (startY + endY) / 2 + 2, label, 'class="branch" text-anchor="middle"');
          out += draw(child, active || inPending);
        }
        return out + '</g>';
      }
      let out = text(compact ? 20 : 22, compact ? 12 : 28, 'One tree', 'class="panel-title"');
      out += draw(displayed);
      if (next) out += text(compact ? 20 : 24, compact ? 151 : 399, `Drawing split ${s.index + 1}: ${splitLabel(pending)}`, 'class="pending-note"');
      else out += text(compact ? 20 : 24, compact ? 151 : 399, 'Leaf predictions, rounded to 0.1 min.', 'class="small-note"');
      return out;
    }

    function stumpDiagrams(model, next, amount) {
      const spacing = compact ? 198 : 187;
      let out = '';
      for (let i = 0; i < Math.min(G.stumps.length, s.index + (next ? 1 : 0)); i++) {
        const stump = G.stumps[i], pending = i >= s.index, reveal = pending ? clamp(amount / .8) : 1;
        const cx = (compact ? 109 : 104) + i % 2 * spacing;
        const top = (compact ? 13 : 43) + Math.floor(i / 2) * (compact ? 80 : 192);
        const rootY = top + (compact ? 22 : 44), leafY = rootY + (compact ? 32 : 77), dx = compact ? 42 : 49;
        out += `<g data-stump="${i + 1}" data-fitted="${!pending}" opacity="${reveal}">`;
        out += text(cx, top, `h${sub(i + 1)}: fit y − F${sub(i)}`, 'class="stump-title" text-anchor="middle"');
        out += text(cx, rootY, splitLabel(stump), `class="node" text-anchor="middle" fill="${pending ? C.rust : C.ink}"`);
        for (const [value, direction] of [[stump.left.value, -1], [stump.right.value, 1]]) {
          const t = pending ? reveal : 1;
          out += line(cx, rootY + 5, cx + direction * dx * t, rootY + 5 + (leafY - rootY - 20) * t, pending ? C.rust : C.rule, compact ? 1.3 : 1.7);
          if (!pending) out += text(cx + direction * dx, leafY, signed(value), 'class="correction" text-anchor="middle"');
        }
        out += '</g>';
      }
      if (s.index === 0 && !next) out += text(compact ? 210 : 195, compact ? 73 : 175, `F₀ = ${fmt(G.baseline, 2)}`, 'class="leaf" text-anchor="middle"');
      if (!compact) out += text(195, 417, s.index ? `${fmt(G.baseline, 2)} + ${Array.from({ length: s.index }, (_, i) => `h${sub(i + 1)}`).join(' + ')}` : 'Begin with the mean.', 'class="sum" text-anchor="middle"');
      return out;
    }

    function metrics(model, x, y, short = false) {
      if (short) return text(x, y, `Training MSE ${model.mse.toFixed(3)}`, 'class="metric" text-anchor="middle"')
        + text(x, y + (compact ? 22 : 30), `${model.splitCount} splits · ${model.regionCount} regions`, 'class="metric" text-anchor="middle"');
      return text(x, y, `MSE ${model.mse.toFixed(3)} · ${model.splitCount} ${model.splitCount === 1 ? 'split' : 'splits'} · ${model.regionCount} ${model.regionCount === 1 ? 'region' : 'regions'}`, 'class="metric" text-anchor="middle"');
    }

    if (comparison) {
      const tree = G.treeStages[max(0)], boost = G.boostStages[max(1)];
      const first = compact ? { x: 39, y: 42, w: 157, h: 217 } : { x: 66, y: 66, w: 298, h: 298 };
      const second = { ...first, x: compact ? 252 : 461 };
      for (const [model, box, id, title] of [[tree, first, 'tree', 'One tree'], [boost, second, 'boost', 'Four stumps']]) {
        body += text(box.x + box.w / 2, compact ? 20 : 29, title, 'class="panel-title" text-anchor="middle"');
        body += map(model, box, id, null, 0);
        body += metrics(model, box.x + box.w / 2, compact ? 318 : 430, true);
      }
    } else {
      const box = compact ? { x: 65, y: 180, w: 331, h: 157 } : { x: 434, y: 65, w: 315, h: 317 };
      const pending = s.next ? scene === 0 ? s.next.split : { ...s.next.stump, bounds: G.bounds } : null;
      body += scene === 0 ? treeDiagram(s.model, s.next, s.amount) : stumpDiagrams(s.model, s.next, s.amount);
      body += text(box.x, compact ? 171 : 28, 'Predicted time F(x₁, x₂)', 'class="panel-title"');
      body += map(s.model, box, kind, pending, s.amount);
      body += metrics(s.model, box.x + box.w / 2, compact ? 395 : 453);
    }
    const legendY = compact ? comparison ? 374 : 421 : 496;
    const legendX = compact ? 100 : 416;
    body += colorLegend(legendX, legendY, compact ? 244 : 280, 'Minutes: points y · regions F', compact, 'observed-y,predicted-F');
    const dotY = compact ? legendY + 4 : 504;
    body += `<circle cx="${compact ? 21 : 36}" cy="${dotY}" r="${compact ? 3.8 : 5}" fill="${timeColor(16)}" stroke="${C.blue}" stroke-width=".6"/>`;
    body += text(compact ? 29 : 49, dotY + 4, compact ? 'Hover / focus' : 'Hover or focus a delivery for y, F, and y − F', 'class="legend"');
    body += pointOverlays.join('');
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" fill="${C.ink}" role="img" aria-labelledby="growth-title growth-description" data-kind="${kind}"${s ? ` data-stage="${s.index}" data-progress="${s.progress}" data-mse="${s.model.mse}" data-split-count="${s.model.splitCount}"` : ''}><title id="growth-title">${esc(p.heading)}</title><desc id="growth-description">${esc(p.takeaway + ' ' + readout(scene, state))} Region color and numbers show predicted time F on a fixed 8 to 24 minute scale. Dots show observed time y on the same color scale. Hover or focus a dot to read observed time, predicted time, and residual y minus F; tooltip values are rounded to two decimals.</desc><style>text{font-family:Georgia,serif;font-size:${nodeSize}px}.panel-title{font-size:${compact ? 14 : 21}px}.node,.leaf,.correction{paint-order:stroke;stroke:white;stroke-width:${compact ? 3 : 5}px;stroke-linejoin:round}.prediction{font-size:${compact && comparison ? 12 : compact ? 14 : 20}px}.leaf,.correction{font-size:${compact ? 13 : 18}px}.correction{fill:${C.rust}}.tick,.axis,.legend,.small-note,.pending-note,.observation-key,.stump-title,.branch{font-family:Arial,sans-serif;font-size:${compact ? 11 : 14}px}.tick,.axis,.legend,.small-note,.observation-key{fill:${C.muted}}.pending-note{fill:${C.rust}}.metric{font-family:Arial,sans-serif;font-size:${compact ? 13 : 18}px}.observation-value{font-family:Arial,sans-serif;font-size:${compact ? 10 : 13}px;paint-order:stroke;stroke:white;stroke-width:3px}.observation-key{font-size:${compact ? 10 : 13}px}.legend{font-size:${compact ? 10.5 : 13}px}.sum{font-size:20px}.growth-observation{outline:none;pointer-events:none}.observation-hit{pointer-events:all;cursor:help;stroke:none}.growth-observation:focus .observation-hit{stroke:white;stroke-width:2;stroke-dasharray:2 2}.observation-tooltip{visibility:hidden;pointer-events:none}.growth-observation:hover .observation-tooltip,.growth-observation:focus .observation-tooltip{visibility:visible}.tooltip-value{font-family:Arial,sans-serif;font-size:${compact ? 10.5 : 13}px;fill:${C.ink}}</style><rect width="${W}" height="${H}" fill="white"/>${body}</svg>`;
  }

  function transitionPresentation() {
    return { heading: 'Two predictors, one outcome', equation: '', takeaway: '' };
  }

  function transitionReadout() { return ''; }

  function renderTransition() {
    return '<div class="transition-copy"><p>Now predict delivery time using two inputs: distance and number of stops.</p><p>Using 16 new deliveries, compare how one tree and a sum of stumps partition the input space.</p></div>';
  }

  return { render, readout, presentation, max, renderTransition, transitionPresentation, transitionReadout, timeColor, timeDomain };
});
