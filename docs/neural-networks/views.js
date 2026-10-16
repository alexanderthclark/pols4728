/* Scroll-driven SVG views. No controls, timers, or hidden mathematical state. */
(function (root) {
  'use strict';

  const M = root.NNMath;
  const C = {
    ink: '#242424', blue: '#234E70', rust: '#A54F32', teal: '#007F73',
    purple: '#7B4F9D', muted: '#606060', paper: '#FFFFFF', rule: '#767676',
    pale: '#DADADA', gold: '#A56716'
  };
  const unitColors = [C.teal, C.rust, C.purple, C.blue];
  const finite = (x, fallback) => Number.isFinite(x) ? x : fallback;
  const clamp = (x, low, high) => Math.max(low, Math.min(high, x));
  const fmt = (x, digits) => {
    if (Math.abs(x) < 0.0005) return '0';
    return Number(x.toFixed(digits === undefined ? 2 : digits)).toString().replace('-', '−');
  };
  const escape = value => String(value).replace(/[&<>\"]/g, function (ch) {
    return ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[ch];
  });

  function builder(svg, width, height, description, compact) {
    const parts = [];
    svg.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', description);
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    parts.push('<style>' +
      '.nv-label{font-family:Georgia,"Times New Roman",serif;font-size:' + (compact ? 22 : 15) + 'px;fill:' + C.ink + '}' +
      '.nv-small{font-family:Arial,sans-serif;font-size:' + (compact ? 20 : 12) + 'px;fill:' + C.muted + '}' +
      '.nv-number{font-family:Arial,sans-serif;font-size:' + (compact ? 20 : 12) + 'px;fill:' + C.ink + ';font-variant-numeric:tabular-nums}' +
      '.nv-head{font-family:Arial,sans-serif;font-size:' + (compact ? 21 : 14) + 'px;fill:' + C.ink + '}' +
      '.nv-halo{paint-order:stroke;stroke:#fff;stroke-width:5px;stroke-linejoin:round}' +
      '</style>');
    return {
      add: s => parts.push(s),
      line: (x1, y1, x2, y2, attrs) => parts.push('<line x1="' + x1 + '" y1="' + y1 + '" x2="' + x2 + '" y2="' + y2 + '" ' + (attrs || '') + '/>'),
      text: (x, y, value, attrs) => parts.push('<text x="' + x + '" y="' + y + '" ' + (attrs || 'class="nv-small"') + '>' + escape(value) + '</text>'),
      circle: (x, y, r, attrs) => parts.push('<circle cx="' + x + '" cy="' + y + '" r="' + r + '" ' + (attrs || '') + '/>'),
      path: (d, attrs) => parts.push('<path d="' + d + '" ' + (attrs || '') + '/>'),
      finish: () => { svg.innerHTML = parts.join(''); }
    };
  }

  function exactPoints(model, domain) {
    return [domain[0]].concat(model.neurons.map(n => n.t).filter(x => x > domain[0] && x < domain[1]))
      .concat([domain[1]]).sort((a, b) => a - b)
      .filter((x, i, xs) => i === 0 || x - xs[i - 1] > 1e-9);
  }

  function modelFor(frame) {
    if (frame.model) return frame.model;
    if (frame.kind === 'affine') return null;
    if (frame.kind === 'relu') return M.createShallow(0, [{ t: 0, a: 1 }]);
    if (frame.kind === 'sine') return M.modelFromKnots(M.initialKnots(5));
    return M.createShallow(0, [{ t: -1, a: 1 }, { t: 0, a: -2 }, { t: 1, a: 1 }]);
  }

  function domainFor(frame) {
    if (Array.isArray(frame.domain)) return frame.domain;
    if (frame.kind === 'sine') return [0, M.TAU];
    return [-2, 2];
  }

  function drawPlot(svg, frame, model) {
    const kind = frame.kind || 'shallow';
    const domain = domainFor(frame);
    const xValue = clamp(finite(frame.x, kind === 'sine' ? 1.5 : 0.5), domain[0], domain[1]);
    const value = kind === 'affine' ? x => x : x => M.evaluate(model, x);
    const xs = kind === 'affine' ? domain : exactPoints(model, domain);
    let range;
    if (Array.isArray(frame.range)) range = frame.range;
    else if (kind === 'sine') range = [-1.35, 1.35];
    else if (kind === 'affine') range = [-2.35, 2.35];
    else {
      const ys = xs.map(value);
      if (frame.components && model) model.neurons.forEach(n => {
        xs.forEach(x => ys.push(n.a * M.relu(x - n.t)));
      });
      const low = Math.min(-0.25, ...ys), high = Math.max(1.25, ...ys);
      const pad = Math.max(0.2, 0.11 * (high - low));
      range = [low - pad, high + pad];
    }
    const b = builder(svg, 620, frame.compact ? 350 : 390,
      kind === 'sine' ? 'A shallow ReLU network, shown as a piecewise linear blue curve, approximates the dashed sine curve.' :
          'One input x is mapped to one output. The curve shows the function computed by the neural network.', frame.compact);
    const left = 53, right = 589, top = 66, bottom = frame.compact ? 293 : 323;
    const px = x => left + (x - domain[0]) / (domain[1] - domain[0]) * (right - left);
    const py = y => bottom - (y - range[0]) / (range[1] - range[0]) * (bottom - top);
    const path = (points, fn) => points.map((x, i) => (i ? 'L' : 'M') + px(x).toFixed(3) + ',' + py(fn(x)).toFixed(3)).join(' ');
    const plotId = 'nv-plot-clip';
    b.add('<defs><clipPath id="' + plotId + '"><rect x="' + left + '" y="' + top + '" width="' + (right - left) + '" height="' + (bottom - top) + '"/></clipPath></defs>');

    if (kind === 'sine') {
      b.line(54, 23, 81, 23, 'stroke="' + C.rust + '" stroke-width="2.3" stroke-dasharray="6 4"');
      b.text(88, 27, 'sin(x)', 'class="nv-label"');
      b.line(frame.compact ? 205 : 176, 23, frame.compact ? 232 : 203, 23, 'stroke="' + C.blue + '" stroke-width="3"');
      b.text(frame.compact ? 242 : 210, 27, 'neural network', 'class="nv-small"');
    } else if (kind === 'affine') b.text(left, 25, 'A straight line', 'class="nv-head"');
    else if (kind === 'relu') b.text(left, 25, 'One ReLU adds one joint', 'class="nv-head"');
    else b.text(left, 25, model.neurons.length + ' hidden ' + (model.neurons.length === 1 ? 'neuron' : 'neurons') + ' · one piecewise linear function', 'class="nv-head"');

    // Only a left and bottom axis; short ticks carry the scale without a grid.
    b.line(left, top - 6, left, bottom, 'stroke="' + C.rule + '" stroke-width="1"');
    b.line(left, bottom, right + 6, bottom, 'stroke="' + C.rule + '" stroke-width="1"');
    let ticks;
    if (kind === 'sine') ticks = [[0, '0'], [Math.PI / 2, 'π/2'], [Math.PI, 'π'], [3 * Math.PI / 2, '3π/2'], [M.TAU, '2π']];
    else ticks = [-2, -1, 0, 1, 2].filter(x => x >= domain[0] && x <= domain[1]).map(x => [x, fmt(x)]);
    ticks.forEach(t => {
      b.line(px(t[0]), bottom, px(t[0]), bottom + 5, 'stroke="' + C.rule + '"');
      b.text(px(t[0]), bottom + 22, t[1], 'class="nv-number" text-anchor="middle"');
    });
    const yTicks = kind === 'sine' ? [-1, 0, 1] : [-4, -2, -1, 0, 1, 2, 4].filter(y => y >= range[0] && y <= range[1]);
    yTicks.forEach(y => {
      b.line(left - 4, py(y), left, py(y), 'stroke="' + C.rule + '"');
      b.text(left - 10, py(y) + 4, fmt(y), 'class="nv-number" text-anchor="end"');
    });
    b.text(right + 9, bottom + 5, 'x', 'class="nv-label"');
    b.text(left - 1, top - 18, 'ŷ', 'class="nv-label" text-anchor="middle"');

    b.add('<g clip-path="url(#' + plotId + ')">');
    if (frame.showAffine) b.path(path(domain, x => x), 'fill="none" stroke="' + C.rule + '" stroke-width="1.7" stroke-dasharray="4 5"');
    if (frame.components && model) model.neurons.forEach((n, i) => {
      b.path(path(exactPoints(M.createShallow(0, [n]), domain), x => n.a * M.relu(x - n.t)),
        'fill="none" stroke="' + unitColors[i % unitColors.length] + '" stroke-width="1.8" stroke-dasharray="5 4" opacity=".55"');
    });
    if (kind === 'sine') {
      const sineXs = Array.from({ length: 301 }, (_, i) => M.TAU * i / 300);
      b.path(path(sineXs, Math.sin), 'fill="none" stroke="' + C.rust + '" stroke-width="2.1" stroke-dasharray="7 5"');
    }
    if (frame.oldModel && frame.oldModel !== model) {
      b.path(path(exactPoints(frame.oldModel, domain), x => M.evaluate(frame.oldModel, x)), 'fill="none" stroke="' + C.rule + '" stroke-width="1.8" stroke-dasharray="2 5" opacity=".65"');
    }
    b.path(path(xs, value), 'fill="none" stroke="' + C.blue + '" stroke-width="3.2" stroke-linejoin="round" stroke-linecap="round"');
    let joints;
    if (model) joints = model.neurons.filter(n => !n.baseline && n.t > domain[0] && n.t < domain[1]).map(n => n.t);
    else joints = [];
    joints.forEach((t, i) => b.circle(px(t), py(value(t)), joints.length > 15 ? 2.5 : 4.2,
      'fill="' + (kind === 'sine' ? C.rust : unitColors[i % unitColors.length]) + '" stroke="#fff" stroke-width="1.2"'));
    if (kind !== 'sine' && frame.showPoint !== false) {
      b.line(px(xValue), bottom, px(xValue), py(value(xValue)), 'stroke="' + C.teal + '" stroke-width="1.2" stroke-dasharray="3 4"');
      b.circle(px(xValue), py(value(xValue)), 5.4, 'fill="' + C.teal + '" stroke="#fff" stroke-width="1.8"');
    }
    if (kind === 'sine' && (frame.zoom || Number.isFinite(frame.highlightX))) {
      const h = clamp(finite(frame.highlightX, xValue), domain[0], domain[1]);
      const prediction = value(h), truth = Math.sin(h);
      b.line(px(h), py(prediction), px(h), py(truth), 'stroke="' + C.rust + '" stroke-width="2.5"');
      b.circle(px(h), py(truth), 4.1, 'fill="#fff" stroke="' + C.rust + '" stroke-width="1.9"');
      b.circle(px(h), py(prediction), 3.3, 'fill="' + C.blue + '" stroke="#fff" stroke-width="1.3"');
    }
    b.add('</g>');

    if (kind === 'sine' && frame.zoom) drawInset(b, frame, model, px, py);
    if (kind !== 'sine') {
      b.text(left, frame.compact ? 339 : 376, 'x = ' + fmt(xValue) + '   →   ŷ = ' + fmt(value(xValue)), 'class="nv-number"');
      if (frame.components && !frame.compact) b.text(right, 376, 'Dashed: each weighted neuron', 'class="nv-small" text-anchor="end"');
    } else {
      if (!frame.compact) b.text(left, 376, 'One input x', 'class="nv-small"');
      if (!frame.compact) b.text(right, 376, 'Solid line: linear between its joints', 'class="nv-small" text-anchor="end"');
    }
    b.finish();
  }

  function drawInset(b, frame, model, mainX, mainY) {
    const center = clamp(finite(frame.highlightX, finite(frame.x, 1.5)), 0.45, M.TAU - 0.45);
    const xlo = center - 0.43, xhi = center + 0.43;
    const samples = Array.from({ length: 101 }, (_, i) => xlo + (xhi - xlo) * i / 100);
    const allY = samples.map(Math.sin).concat(samples.map(x => M.evaluate(model, x)));
    if (frame.oldModel) allY.push(...samples.map(x => M.evaluate(frame.oldModel, x)));
    const low = Math.min(...allY) - 0.035, high = Math.max(...allY) + 0.035;
    const x0 = 367, x1 = 581, y0 = 82, y1 = 183;
    const ix = x => x0 + (x - xlo) / (xhi - xlo) * (x1 - x0);
    const iy = y => y1 - (y - low) / (high - low) * (y1 - y0);
    const line = (points, fn) => points.map((x, i) => (i ? 'L' : 'M') + ix(x).toFixed(3) + ',' + iy(fn(x)).toFixed(3)).join(' ');
    b.add('<rect x="354" y="49" width="240" height="151" fill="#fff" stroke="' + C.rule + '" stroke-width=".9"/>');
    b.text(367, 70, 'A closer look', 'class="nv-small"');
    b.line(x0, y1, x1, y1, 'stroke="' + C.rule + '" stroke-width=".8"');
    if (frame.oldModel) b.path(line(exactPoints(frame.oldModel, [xlo, xhi]), x => M.evaluate(frame.oldModel, x)), 'fill="none" stroke="' + C.rule + '" stroke-width="1.5" stroke-dasharray="2 4" opacity=".7"');
    b.path(line(samples, Math.sin), 'fill="none" stroke="' + C.rust + '" stroke-width="1.8" stroke-dasharray="5 4"');
    b.path(line(exactPoints(model, [xlo, xhi]), x => M.evaluate(model, x)), 'fill="none" stroke="' + C.blue + '" stroke-width="2.5" stroke-linejoin="round"');
    const focus = clamp(finite(frame.highlightX, center), xlo, xhi);
    const truth = Math.sin(focus), pred = M.evaluate(model, focus);
    b.line(ix(focus), iy(truth), ix(focus), iy(pred), 'stroke="' + C.rust + '" stroke-width="2"');
    b.circle(ix(focus), iy(truth), 3.2, 'fill="#fff" stroke="' + C.rust + '" stroke-width="1.5"');
    b.circle(ix(focus), iy(pred), 2.7, 'fill="' + C.blue + '"');
    b.line(mainX(focus), mainY(pred) - 8, 470, 207, 'stroke="' + C.rule + '" stroke-width=".9" stroke-dasharray="3 4"');
  }

  function drawNetwork(svg, frame, model) {
    const kind = frame.kind || 'shallow';
    if (frame.compact) return drawCompactNetwork(svg, frame, model);
    const affine = kind === 'affine';
    const neurons = affine ? [] : model.neurons;
    const count = neurons.length;
    const x = finite(frame.x, kind === 'sine' ? 1.5 : 0.5);
    const values = affine ? [] : M.contributions(model, x);
    const result = affine ? x : M.evaluate(model, x);
    const b = builder(svg, 400, 390, affine ? 'An affine function maps input x directly to the output.' :
      'A neural network with one input, ' + count + ' hidden ReLU ' + (count === 1 ? 'neuron' : 'neurons') + ', and one linear output. Each input weight is one. Hidden biases are minus the joint locations; output weights are changes in slope.');
    const inputX = 34, hiddenX = 192, outX = 365, middleY = 204;
    b.text(200, 25, affine ? '1 input · 1 output' : '1 input · ' + count + ' hidden · 1 output', 'class="nv-head" text-anchor="middle"');
    b.text(inputX, 61, 'Input', 'class="nv-small" text-anchor="middle"');
    b.text(outX, 61, 'Linear', 'class="nv-small" text-anchor="middle"');
    if (affine) {
      b.line(inputX + 18, middleY, outX - 18, middleY, 'stroke="' + C.blue + '" stroke-width="2"');
      b.text(199, middleY - 15, 'weight = 1', 'class="nv-number" text-anchor="middle"');
    } else if (count === 0) {
      b.text(195, 177, 'No hidden neurons yet', 'class="nv-small" text-anchor="middle"');
    } else {
      b.text(hiddenX, 61, 'ReLU', 'class="nv-small" text-anchor="middle"');
      b.text(94, 84, 'w = 1', 'class="nv-small" text-anchor="middle"');
      b.text(286, 84, 'output weights', 'class="nv-small" text-anchor="middle"');
      const lowY = 106, highY = 309;
      const gap = count > 1 ? (highY - lowY) / (count - 1) : 0;
      const radius = count > 17 ? 4.1 : count > 10 ? 6 : count > 6 ? 9 : 13;
      const positions = neurons.map((n, i) => count === 1 ? middleY : lowY + i * gap);
      neurons.forEach((n, i) => {
        const cy = positions[i], active = values[i].activation > 1e-8;
        const color = count > 8 ? C.blue : unitColors[i % unitColors.length];
        const opacity = active ? 0.65 : 0.22;
        b.line(inputX + 17, middleY, hiddenX - radius, cy, 'stroke="' + (active ? C.blue : C.rule) + '" stroke-width="1.1" opacity="' + opacity + '"');
        b.line(hiddenX + radius, cy, outX - 18, middleY,
          'stroke="' + (n.a < 0 ? C.rust : C.blue) + '" stroke-width="1.15" opacity="' + opacity + '"' + (n.a < 0 ? ' stroke-dasharray="3 2"' : ''));
        const labelSize = count > 17 ? 5.5 : count > 10 ? 7.5 : count > 6 ? 10 : 12;
        b.text(163, cy + labelSize * 0.32, 'b=' + fmt(-n.t), 'class="nv-number nv-halo" text-anchor="end" style="font-size:' + labelSize + 'px"');
        b.text(223, cy + labelSize * 0.32, 'a=' + fmt(n.a), 'class="nv-number nv-halo" style="font-size:' + labelSize + 'px"');
        b.circle(hiddenX, cy, radius, 'fill="' + (active ? color : '#fff') + '" stroke="' + color + '" stroke-width="1.2"');
        if (count <= 10) b.text(hiddenX, cy + 3.4, 'R', 'class="nv-number" text-anchor="middle" style="font-size:' + (count > 6 ? 9 : 11) + 'px;fill:' + (active ? '#fff' : color) + '"');
        if (count <= 6) b.text(hiddenX, cy + radius + 15, fmt(values[i].activation), 'class="nv-number nv-halo" text-anchor="middle" style="font-size:11px"');
      });
    }
    b.circle(inputX, middleY, 17, 'fill="#fff" stroke="' + C.ink + '" stroke-width="1.5"');
    b.text(inputX, middleY + 5, 'x', 'class="nv-label" text-anchor="middle"');
    b.text(inputX, middleY + 39, fmt(x), 'class="nv-number" text-anchor="middle"');
    b.circle(outX, middleY, 18, 'fill="' + C.blue + '" stroke="' + C.blue + '" stroke-width="1.5"');
    b.text(outX, middleY + 5, 'ŷ', 'class="nv-label" text-anchor="middle" style="fill:#fff"');
    b.text(outX, middleY + 40, fmt(result), 'class="nv-number" text-anchor="middle"');
    b.text(200, 353, affine ? 'Output bias = 0' : 'Output bias c = ' + fmt(model.c), 'class="nv-number" text-anchor="middle"');
    b.text(200, 376, affine ? 'One affine transformation' : count > 10 ? 'More joints, more hidden neurons' : 'R = ReLU · numbers inside the graph show activations', 'class="nv-small" text-anchor="middle"');
    b.finish();
  }

  function drawCompactNetwork(svg, frame, model) {
    const affine = frame.kind === 'affine';
    const neurons = affine ? [] : model.neurons;
    const count = neurons.length;
    const x = finite(frame.x, frame.kind === 'sine' ? 1.5 : 0.5);
    const values = affine ? [] : M.contributions(model, x);
    const result = affine ? x : M.evaluate(model, x);
    const b = builder(svg, 620, 220,
      affine ? 'One affine transformation from input to output.' :
        'One input, one hidden layer containing ' + count + ' ReLU neurons, and one linear output. Input weights are one; hidden biases and output weights are labelled.', true);
    const center = 310, inputY = 23, hiddenY = 110, outputY = 200;
    const radius = count > 17 ? 5.5 : count > 10 ? 7.5 : count > 6 ? 9 : 12;
    const positions = neurons.map((n, i) => count < 2 ? center : 48 + i * 524 / (count - 1));
    if (affine) {
      b.line(center, inputY + 16, center, outputY - 16, 'stroke="' + C.blue + '" stroke-width="2"');
      b.text(340, hiddenY + 6, 'weight = 1', 'class="nv-number"');
    } else neurons.forEach((n, i) => {
      const cx = positions[i], active = values[i].activation > 1e-8;
      const color = count > 8 ? C.blue : unitColors[i % unitColors.length];
      const opacity = active ? 0.65 : 0.24;
      b.line(center, inputY + 16, cx, hiddenY - radius, 'stroke="' + C.blue + '" stroke-width="1.3" opacity="' + opacity + '"');
      b.line(cx, hiddenY + radius, center, outputY - 16,
        'stroke="' + (n.a < 0 ? C.rust : C.blue) + '" stroke-width="1.3" opacity="' + opacity + '"' + (n.a < 0 ? ' stroke-dasharray="4 3"' : ''));
      const font = count > 10 ? 15 : count > 6 ? 17 : 19;
      b.text(cx, 84, 'b=' + fmt(-n.t), 'class="nv-number nv-halo" text-anchor="middle" style="font-size:' + font + 'px"');
      b.text(cx, 155, 'a=' + fmt(n.a), 'class="nv-number nv-halo" text-anchor="middle" style="font-size:' + font + 'px"');
      b.circle(cx, hiddenY, radius, 'fill="' + (active ? color : '#fff') + '" stroke="' + color + '" stroke-width="1.6"');
      if (count <= 10) b.text(cx, hiddenY + 5, 'R', 'class="nv-number" text-anchor="middle" style="font-size:17px;fill:' + (active ? '#fff' : color) + '"');
      if (count <= 6) b.text(cx, 135, fmt(values[i].activation), 'class="nv-number nv-halo" text-anchor="middle" style="font-size:17px"');
    });
    b.circle(center, inputY, 16, 'fill="#fff" stroke="' + C.ink + '" stroke-width="1.6"');
    b.text(center, inputY + 6, 'x', 'class="nv-label" text-anchor="middle"');
    b.text(center + 31, inputY + 6, '= ' + fmt(x), 'class="nv-number"');
    b.circle(center, outputY, 17, 'fill="' + C.blue + '"');
    b.text(center, outputY + 6, 'ŷ', 'class="nv-label" text-anchor="middle" style="fill:#fff"');
    b.text(center + 32, outputY + 6, '= ' + fmt(result), 'class="nv-number"');
    b.text(12, 27, affine ? 'Affine' : count + ' ReLU neurons', 'class="nv-small"');
    b.text(12, 207, 'c = ' + fmt(affine ? 0 : model.c), 'class="nv-number"');
    if (!affine && count < 7) b.text(604, 27, 'input weights = 1', 'class="nv-small" text-anchor="end"');
    b.finish();
  }

  function render(plotSvg, networkSvg, frame) {
    frame = frame || { kind: 'shallow' };
    const model = modelFor(frame);
    if (plotSvg) drawPlot(plotSvg, frame, model);
    if (networkSvg) {
      if (frame.showNetwork === false) {
        networkSvg.innerHTML = '';
        networkSvg.setAttribute('aria-hidden', 'true');
      } else {
        networkSvg.removeAttribute('aria-hidden');
        drawNetwork(networkSvg, frame, model);
      }
    }
  }

  root.NNViews = Object.freeze({ render: render });
})(typeof window !== 'undefined' ? window : globalThis);
