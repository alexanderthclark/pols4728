/* Sigmoid features and a fitted binary representation, driven only by scrolling. */
(function (root) {
  'use strict';

  const C = {
    ink: '#242424', blue: '#234E70', rust: '#A54F32', teal: '#007F73',
    purple: '#7B4F9D', muted: '#606060', rule: '#767676', pale: '#DADADA'
  };
  const unitColors = [C.teal, C.rust, C.purple, C.blue];
  const thresholds = [0.2, 0.4, 0.6, 0.8];
  const mathCache = new Map();
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  const finite = (x, otherwise) => Number.isFinite(x) ? x : otherwise;
  const sigmoid = x => x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
  const fmt = (x, digits) => Number(x.toFixed(digits === undefined ? 2 : digits)).toString().replace('-', '−');
  const texNumber = x => Number(x.toFixed(2)).toString();
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

  function mathMarkup(tex) {
    if (!mathCache.has(tex)) {
      mathCache.set(tex, root.katex.renderToString(tex, {
        output: 'htmlAndMathml', displayMode: false, throwOnError: true,
        strict: 'error', trust: false
      }));
    }
    return mathCache.get(tex);
  }

  function canvas(svg, width, height, description, compact) {
    svg.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', description);
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    svg.removeAttribute('aria-hidden');
    const body = ['<style>' +
      '.ne-text{font-family:Arial,sans-serif;fill:' + C.ink + ';font-size:' + (compact ? 22 : 15) + 'px}' +
      '.ne-tick{font-family:Arial,sans-serif;fill:' + C.muted + ';font-size:' + (compact ? 21 : 13) + 'px}' +
      '.ne-math .katex{font-size:1em;line-height:1.1}' +
      '.ne-halo{paint-order:stroke;stroke:#fff;stroke-width:5px;stroke-linejoin:round}' +
      '</style>'];
    const b = {
      add: text => body.push(text),
      line: (x1, y1, x2, y2, attrs) => body.push('<line x1="' + x1 + '" y1="' + y1 + '" x2="' + x2 + '" y2="' + y2 + '" ' + (attrs || '') + '/>'),
      text: (x, y, text, attrs) => body.push('<text x="' + x + '" y="' + y + '" class="ne-text" ' + (attrs || '') + '>' + esc(text) + '</text>'),
      circle: (x, y, r, attrs) => body.push('<circle cx="' + x + '" cy="' + y + '" r="' + r + '" ' + (attrs || '') + '/>'),
      path: (d, attrs) => body.push('<path d="' + d + '" ' + (attrs || '') + '/>'),
      math: (x, y, tex, width, font, attrs) => {
        const size = font || (compact ? 24 : 18), height = size * 2.8;
        body.push('<foreignObject class="ne-math" data-tex="' + esc(tex) + '" x="' + (x - width / 2) + '" y="' + (y - height / 2) + '" width="' + width + '" height="' + height + '" overflow="visible" ' + (attrs || '') + '>' +
          '<div xmlns="http://www.w3.org/1999/xhtml" style="height:100%;display:flex;align-items:center;justify-content:center;color:inherit;font-size:' + size + 'px;white-space:nowrap">' + mathMarkup(tex) + '</div></foreignObject>');
      },
      finish: () => { svg.innerHTML = body.join(''); }
    };
    return b;
  }

  function axis(b, box, xr, yr, options) {
    const opt = options || {}, compact = Boolean(opt.compact);
    const px = x => box.left + (x - xr[0]) / (xr[1] - xr[0]) * (box.right - box.left);
    const py = y => box.bottom - (y - yr[0]) / (yr[1] - yr[0]) * (box.bottom - box.top);
    b.line(box.left, box.top, box.left, box.bottom, 'stroke="' + C.rule + '" stroke-width="1.1"');
    b.line(box.left, box.bottom, box.right, box.bottom, 'stroke="' + C.rule + '" stroke-width="1.1"');
    (opt.xTicks || [xr[0], (xr[0] + xr[1]) / 2, xr[1]]).forEach(x => {
      b.line(px(x), box.bottom, px(x), box.bottom + 5, 'stroke="' + C.rule + '"');
      b.text(px(x), box.bottom + (compact ? 26 : 22), fmt(x), 'text-anchor="middle" style="fill:' + C.muted + ';font-size:' + (compact ? 21 : 13) + 'px"');
    });
    (opt.yTicks || [yr[0], (yr[0] + yr[1]) / 2, yr[1]]).forEach(y => {
      b.line(box.left - 5, py(y), box.left, py(y), 'stroke="' + C.rule + '"');
      b.text(box.left - 9, py(y) + (compact ? 7 : 4), fmt(y), 'text-anchor="end" style="fill:' + C.muted + ';font-size:' + (compact ? 21 : 13) + 'px"');
    });
    if (opt.xTex) b.math(box.right + (compact ? 18 : 15), box.bottom + 2, opt.xTex, 54, compact ? 24 : 18);
    if (opt.yTex) b.math(box.left, box.top - 22, opt.yTex, 90, compact ? 24 : 18);
    return { px, py };
  }

  function curve(xs, fn, px, py) {
    return xs.map((x, i) => (i ? 'L' : 'M') + px(x).toFixed(3) + ',' + py(fn(x)).toFixed(3)).join(' ');
  }

  function sigmoidPlot(svg, frame) {
    const compact = frame.compact, weights = frame.sigmoidWeights || [6, -6, 6, -6];
    const slope = finite(frame.sigmoidSlope, 30), bias = finite(frame.sigmoidBias, -3);
    const ts = frame.sigmoidThresholds || thresholds;
    const count = weights.length;
    const b = canvas(svg, 620, compact ? 350 : 390,
      'Constructed probability curve from ' + count + ' sigmoid hidden neuron' + (count === 1 ? '' : 's') + ' and a sigmoid output. ' +
      'Existing feature parameters stay fixed while the new output weights change.', compact);
    svg.removeAttribute('data-morph');
    const box = { left: 62, right: 575, top: compact ? 76 : 77, bottom: compact ? 287 : 321 };
    const p = axis(b, box, [0, 1], [0, 1], {
      compact, xTicks: [0, 0.5, 1], yTicks: [0, 0.5, 1], xTex: 'x', yTex: '\\hat p(x)'
    });
    const xs = Array.from({ length: 301 }, (_, i) => i / 300);
    const value = x => sigmoid(bias + weights.reduce((sum, a, j) => sum + a * sigmoid(slope * (x - ts[j])), 0));
    b.line(63, 20, 92, 20, 'stroke="' + C.blue + '" stroke-width="3.5"');
    b.text(101, compact ? 27 : 25, count + ' hidden neuron' + (count === 1 ? '' : 's'));
    b.line(box.left, p.py(0.5), box.right, p.py(0.5), 'stroke="' + C.pale + '" stroke-width="1" stroke-dasharray="3 5"');
    b.path(curve(xs, value, p.px, p.py), 'fill="none" stroke="' + C.blue + '" stroke-width="3.5" data-sigmoid-probability="true" data-hidden-slope="' + slope + '" data-output-bias="' + bias + '" data-weights="' + weights.join(',') + '"');
    b.text(62, compact ? 342 : 376, 'Illustrative functions; these weights are chosen.',
      'style="font-size:' + (compact ? 20 : 13) + 'px;fill:' + C.muted + '"');
    b.finish();
  }

  function sigmoidContributions(svg, frame) {
    const compact = frame.compact, width = compact ? 620 : 500, height = compact ? 220 : 390;
    const weights = frame.sigmoidWeights || [6, -6, 6, -6];
    const slope = finite(frame.sigmoidSlope, 30), bias = finite(frame.sigmoidBias, -3);
    const ts = frame.sigmoidThresholds || thresholds;
    const b = canvas(svg, width, height,
      'Each curve is one weighted sigmoid hidden feature. These contributions and an intercept of minus three are added before the final sigmoid.', compact);
    const box = { left: compact ? 61 : 51, right: compact ? 574 : 453, top: compact ? 68 : 98, bottom: compact ? 174 : 322 };
    b.text(box.left, compact ? 24 : 25, 'Weighted hidden features');
    const legendY = compact ? 45 : 57;
    weights.forEach((a, j) => {
      const x = box.left + j * ((box.right - box.left) / weights.length) + 6;
      b.line(x, legendY, x + 18, legendY, 'stroke="' + unitColors[j] + '" stroke-width="2.4"' + (j % 2 ? ' stroke-dasharray="5 3"' : ''));
      b.math(x + (compact ? 57 : 48), legendY, '\\phi_' + (j + 1) + 'h_' + (j + 1), compact ? 75 : 65, compact ? 22 : 18,
        'style="color:' + unitColors[j] + '"');
    });
    const p = axis(b, box, [0, 1], [-6.5, 6.5], {
      compact, xTicks: [0, 0.5, 1], yTicks: [-6, 0, 6], xTex: 'x'
    });
    b.line(box.left, p.py(0), box.right, p.py(0), 'stroke="' + C.pale + '" stroke-width="1"');
    const xs = Array.from({ length: 301 }, (_, i) => i / 300);
    weights.forEach((a, j) => {
      b.path(curve(xs, x => a * sigmoid(slope * (x - ts[j])), p.px, p.py),
        'fill="none" stroke="' + unitColors[j] + '" stroke-width="2.7"' + (j % 2 ? ' stroke-dasharray="6 4"' : '') +
        ' data-hidden-contribution="' + j + '" data-output-weight="' + a + '" data-hidden-slope="' + slope + '" data-hidden-threshold="' + ts[j] + '"');
    });
    if (!compact) b.math(252, 374, '\\text{Add the intercept }\\phi_0=' + texNumber(bias) + '\\text{, then apply }\\sigma.', 480, 16);
    b.finish();
  }

  function hidden(model, input) {
    return model.hiddenWeights.map((w, j) => Math.max(0, w[0] * input[0] + w[1] * input[1] + model.hiddenBias[j]));
  }

  function logit(model, h) {
    return model.outputBias + model.outputWeights[0] * h[0] + model.outputWeights[1] * h[1];
  }

  function ranges(values, hiddenSpace) {
    return [0, 1].map(j => {
      const lo = Math.min(...values.map(v => v[j])), hi = Math.max(...values.map(v => v[j]));
      const pad = Math.max(0.1, (hi - lo) * 0.08);
      return hiddenSpace ? [0, Math.max(0.1, hi + pad)] : [lo - pad, hi + pad];
    });
  }

  // Clip a convex polygon to a*x + b*y + c >= 0. ReLU's four activation
  // regions make the input-space boundary exactly piecewise linear.
  function clip(poly, a, b, c) {
    const result = [];
    if (!poly.length) return result;
    poly.forEach((point, i) => {
      const next = poly[(i + 1) % poly.length];
      const here = a * point[0] + b * point[1] + c, there = a * next[0] + b * next[1] + c;
      if (here >= -1e-10) result.push(point);
      if ((here > 1e-10 && there < -1e-10) || (here < -1e-10 && there > 1e-10)) {
        const t = here / (here - there);
        result.push([point[0] + t * (next[0] - point[0]), point[1] + t * (next[1] - point[1])]);
      }
    });
    return result;
  }

  function lineInPolygon(poly, a, b, c) {
    const hits = [];
    poly.forEach((point, i) => {
      const next = poly[(i + 1) % poly.length];
      const here = a * point[0] + b * point[1] + c, there = a * next[0] + b * next[1] + c;
      if (Math.abs(here) < 1e-8) hits.push(point);
      if (here * there < 0) {
        const t = here / (here - there);
        hits.push([point[0] + t * (next[0] - point[0]), point[1] + t * (next[1] - point[1])]);
      }
    });
    const unique = hits.filter((p, i) => hits.findIndex(q => Math.hypot(p[0] - q[0], p[1] - q[1]) < 1e-7) === i);
    return unique.length > 1 ? [unique[0], unique[unique.length - 1]] : [];
  }

  function regions(model, xr, yr, representation) {
    const rectangle = [[xr[0], yr[0]], [xr[1], yr[0]], [xr[1], yr[1]], [xr[0], yr[1]]];
    if (representation) return [{ poly: rectangle, a: model.outputWeights[0], b: model.outputWeights[1], c: model.outputBias }];
    return [0, 1, 2, 3].map(mask => {
      let poly = rectangle, a = 0, b = 0, c = model.outputBias;
      model.hiddenWeights.forEach((w, j) => {
        const active = Boolean(mask & (1 << j)), sign = active ? 1 : -1;
        poly = clip(poly, sign * w[0], sign * w[1], sign * model.hiddenBias[j]);
        if (active) {
          a += model.outputWeights[j] * w[0];
          b += model.outputWeights[j] * w[1];
          c += model.outputWeights[j] * model.hiddenBias[j];
        }
      });
      return { poly, a, b, c };
    }).filter(r => r.poly.length > 2);
  }

  function decision(b, model, xr, yr, p, representation, opacity) {
    if (opacity <= 0) return;
    const space = representation ? 'representation' : 'input';
    const path = poly => poly.map((v, i) => (i ? 'L' : 'M') + p.px(v[0]).toFixed(4) + ',' + p.py(v[1]).toFixed(4)).join(' ') + ' Z';
    b.add('<g data-classification-boundary="' + space + '" data-boundary-space="' + space + '" data-boundary-opacity="' + opacity + '" opacity="' + opacity + '">');
    regions(model, xr, yr, representation).forEach(r => {
      [0, 1].forEach(label => {
        const sign = label ? 1 : -1, poly = clip(r.poly, sign * r.a, sign * r.b, sign * r.c);
        if (poly.length > 2) b.path(path(poly), 'fill="' + (label ? C.rust : C.blue) + '" fill-opacity=".065" stroke="none"');
      });
      const segment = lineInPolygon(r.poly, r.a, r.b, r.c);
      if (segment.length) b.line(p.px(segment[0][0]), p.py(segment[0][1]), p.px(segment[1][0]), p.py(segment[1][1]),
        'stroke="' + C.ink + '" stroke-width="2.2" stroke-dasharray="7 5" data-zero-logit="true"');
    });
    b.add('</g>');
  }

  function marker(b, x, y, label, radius, attrs) {
    if (label) b.add('<rect x="' + (x - radius) + '" y="' + (y - radius) + '" width="' + (radius * 2) + '" height="' + (radius * 2) + '" fill="' + C.rust + '" stroke="#fff" stroke-width="1.1" ' + (attrs || '') + '/>');
    else b.circle(x, y, radius, 'fill="' + C.blue + '" stroke="#fff" stroke-width="1.1" ' + (attrs || ''));
  }

  function legend(b, compact, y, width) {
    marker(b, 79, y - 5, 0, compact ? 6 : 5);
    b.text(93, y, 'Class 0');
    marker(b, compact ? 228 : 191, y - 5, 1, compact ? 6 : 5);
    b.text(compact ? 243 : 204, y, 'Class 1');
    if (width > 550) b.text(width - 36, y, 'Same observations', 'text-anchor="end" style="font-size:' + (compact ? 20 : 13) + 'px;fill:' + C.muted + '"');
  }

  function classificationPlot(svg, frame, model) {
    const compact = frame.compact, height = compact ? 350 : 390;
    const embeddings = model.inputs.map(x => hidden(model, x));
    const inputRanges = ranges(model.inputs, false), hRanges = ranges(embeddings, true);
    const morph = clamp(finite(frame.morph, 0), 0, 1);
    const reveal = frame.scene === 'class-head' || frame.scene === 'class-compare';
    const opacity = reveal ? clamp(finite(frame.boundaryOpacity, 1), 0, 1) : 0;
    const inputScene = frame.scene === 'class-input' || frame.scene === 'class-trace';
    const description = morph === 1
      ? 'The same two classes in their learned coordinates h one and h two. ' + (opacity > 0 ? 'The fitted logistic classification head separates them with a straight boundary.' : 'The classification boundary has not yet been revealed.')
      : morph === 0 ? 'The binary training observations in their original two input coordinates. Their fitted two-ReLU-neuron network has a piecewise linear input-space boundary.'
        : 'The same observations move from original input coordinates to their two learned hidden activations. This is a change of display coordinates, not a training animation. No straight classification boundary is displayed.';
    const b = canvas(svg, 620, height, description, compact);
    svg.setAttribute('data-morph', morph);
    const box = { left: 72, right: 566, top: compact ? 80 : 86, bottom: compact ? 278 : 317 };
    legend(b, compact, 25, 620);
    const pxIn = x => box.left + (x - inputRanges[0][0]) / (inputRanges[0][1] - inputRanges[0][0]) * (box.right - box.left);
    const pyIn = y => box.bottom - (y - inputRanges[1][0]) / (inputRanges[1][1] - inputRanges[1][0]) * (box.bottom - box.top);
    const pxH = x => box.left + (x - hRanges[0][0]) / (hRanges[0][1] - hRanges[0][0]) * (box.right - box.left);
    const pyH = y => box.bottom - (y - hRanges[1][0]) / (hRanges[1][1] - hRanges[1][0]) * (box.bottom - box.top);
    const axisOptions = { compact, xTex: morph === 0 ? 'x_1' : morph === 1 ? 'h_1' : undefined, yTex: morph === 0 ? 'x_2' : morph === 1 ? 'h_2' : 'x_2\\to h_2' };
    // Endpoint tick values remain literal data coordinates. Intermediate frames
    // omit ticks because interpolation takes place in normalized display space.
    const endpointRanges = morph < 0.5 ? inputRanges : hRanges;
    if (morph > 0 && morph < 1) { axisOptions.xTicks = []; axisOptions.yTicks = []; }
    axis(b, box, endpointRanges[0], endpointRanges[1], axisOptions);
    if (morph > 0 && morph < 1) b.math((box.left + box.right) / 2, box.bottom + 29, 'x_1\\longrightarrow h_1', 180, compact ? 24 : 19);
    if (inputScene && morph === 0) decision(b, model, inputRanges[0], inputRanges[1], { px: pxIn, py: pyIn }, false, 1);
    if (reveal && morph === 1) decision(b, model, hRanges[0], hRanges[1], { px: pxH, py: pyH }, true, opacity);
    const trace = frame.scene === 'class-trace' ? clamp(Math.floor(finite(frame.traceIndex, 0)), 0, model.inputs.length - 1) : -1;
    model.inputs.forEach((x, i) => {
      const h = embeddings[i], px = pxIn(x[0]) * (1 - morph) + pxH(h[0]) * morph;
      const py = pyIn(x[1]) * (1 - morph) + pyH(h[1]) * morph;
      marker(b, px, py, model.labels[i], compact ? 4.9 : 4.3,
        'data-id="observation-' + i + '" data-point-id="' + i + '" data-label="' + model.labels[i] + '" data-space="' + (morph === 0 ? 'input' : morph === 1 ? 'representation' : 'interpolated') +
        '" data-input-x1="' + x[0] + '" data-input-x2="' + x[1] + '" data-h1="' + h[0] + '" data-h2="' + h[1] + '"');
      if (i === trace) {
        b.circle(px, py, compact ? 12 : 10, 'fill="none" stroke="' + C.ink + '" stroke-width="2" data-traced-point="true"');
        b.text(clamp(px + 13, box.left + 15, box.right - 145), clamp(py - 14, box.top + 15, box.bottom - 10), 'Follow this point',
          'style="font-size:' + (compact ? 20 : 13) + 'px;paint-order:stroke;stroke:#fff;stroke-width:5px;stroke-linejoin:round"');
      }
    });
    const label = morph === 0 ? 'Original coordinates' : morph === 1 ? 'Learned coordinates' : 'Moving to the learned coordinates';
    b.text(72, compact ? 342 : 376, label, 'style="font-size:' + (compact ? 21 : 14) + 'px;fill:' + C.muted + '"');
    b.finish();
  }

  function classificationNetwork(svg, frame, model) {
    const compact = frame.compact, width = compact ? 620 : 500, height = compact ? 220 : 390;
    const tracing = frame.scene === 'class-trace';
    const index = clamp(Math.floor(finite(frame.traceIndex, 0)), 0, model.inputs.length - 1);
    const x = model.inputs[index], h = hidden(model, x), probability = sigmoid(logit(model, h));
    const probabilityText = probability < 0.001 ? 'below 0.001' : probability > 0.999 ? 'above 0.999' : fmt(probability, 3);
    const b = canvas(svg, width, height,
      'The fitted binary network has two inputs, two ReLU hidden neurons, and one sigmoid classification head. ' +
      (tracing ? 'For observation ' + (index + 1) + ', the hidden coordinates are ' + fmt(h[0]) + ' and ' + fmt(h[1]) + ', and its fitted probability of class one is ' + probabilityText + '.' : 'Its representation and classification head were fitted together.'), compact);
    const inX = compact ? 63 : 65, hX = compact ? 300 : 250, outX = compact ? 554 : 436;
    const y1 = compact ? 65 : 126, y2 = compact ? 143 : 261, outputY = (y1 + y2) / 2, r = compact ? 23 : 25;
    b.text(inX, compact ? 24 : 34, 'Inputs', 'text-anchor="middle"');
    b.text(hX, compact ? 24 : 34, 'ReLU features', 'text-anchor="middle"');
    b.text(outX, compact ? 24 : 34, 'Head', 'text-anchor="middle"');
    [y1, y2].forEach(iy => [y1, y2].forEach(hy => b.line(inX + r, iy, hX - r, hy, 'stroke="' + C.rule + '" stroke-width="1.5" opacity=".55"')));
    [y1, y2].forEach((hy, j) => b.line(hX + r, hy, outX - r, outputY,
      'stroke="' + C.blue + '" stroke-width="2"' + (model.outputWeights[j] < 0 ? ' stroke-dasharray="5 4"' : '')));
    [y1, y2].forEach((y, j) => {
      b.circle(inX, y, r, 'fill="#fff" stroke="' + C.ink + '" stroke-width="1.5"');
      b.math(inX, y, 'x_' + (j + 1), 50, compact ? 25 : 22);
      b.circle(hX, y, r, 'fill="#fff" stroke="' + C.teal + '" stroke-width="2"');
      b.math(hX, y, 'h_' + (j + 1), 50, compact ? 25 : 22, 'style="color:' + C.teal + '"');
      if (tracing) {
        b.math(inX, y + r + 13, '=' + texNumber(x[j]), compact ? 102 : 85, compact ? 22 : 18);
        b.math(hX, y + r + 13, '=' + texNumber(h[j]), compact ? 110 : 92, compact ? 22 : 18,
          'data-hidden-value="' + j + '" style="color:' + C.teal + '"');
      }
    });
    b.circle(outX, outputY, r + 2, 'fill="#fff" stroke="' + C.blue + '" stroke-width="2.2"');
    b.math(outX, outputY, '\\sigma', 48, compact ? 26 : 25, 'style="color:' + C.blue + '"');
    if (!compact) {
      b.math(250, 337, tracing ? '(h_1,h_2)=(' + texNumber(h[0]) + ',' + texNumber(h[1]) + ')' : 'h_j=\\max(0,\\theta_{j0}+\\theta_{j1}x_1+\\theta_{j2}x_2)', 490, 18);
      b.text(250, 375, tracing ? 'The two activations are this point’s new coordinates.' : 'Both hidden features and the head are fitted together.', 'text-anchor="middle" style="font-size:13px;fill:' + C.muted + '"');
    } else b.math(310, 205, tracing ? '(h_1,h_2)=(' + texNumber(h[0]) + ',' + texNumber(h[1]) + ')' : '\\text{Two ReLU features; one sigmoid head}', 580, 21);
    b.finish();
  }

  function inputReference(svg, frame, model) {
    const compact = frame.compact, width = compact ? 620 : 500, height = compact ? 220 : 390;
    const b = canvas(svg, width, height,
      'Reference view of the same observations in their original input coordinates. ' +
      (frame.scene === 'class-move' ? 'The classifier is not shown while the main view changes coordinates.' : 'The dashed boundary is the exact input-space preimage of the same fitted logistic head.'), compact);
    const r = ranges(model.inputs, false);
    const box = { left: compact ? 64 : 64, right: compact ? 566 : 454, top: compact ? 53 : 82, bottom: compact ? 166 : 317 };
    if (!compact) b.text(box.left, 25, 'Original coordinates');
    const p = axis(b, box, r[0], r[1], { compact, xTex: 'x_1', yTex: 'x_2' });
    const reveal = frame.scene === 'class-head' || frame.scene === 'class-compare';
    if (reveal) decision(b, model, r[0], r[1], p, false, clamp(finite(frame.boundaryOpacity, 1), 0, 1));
    model.inputs.forEach((x, i) => marker(b, p.px(x[0]), p.py(x[1]), model.labels[i], compact ? 3.8 : 3.7,
      'data-reference-point-id="' + i + '" data-label="' + model.labels[i] + '"'));
    if (!compact) b.text(64, 376, reveal ? 'The same head, through the learned features.' : 'Fixed reference while the main points move.',
      'style="font-size:13px;fill:' + C.muted + '"');
    b.finish();
  }

  function render(functionSvg, networkSvg, frame) {
    if (frame.kind === 'sigmoid') {
      if (functionSvg) sigmoidPlot(functionSvg, frame);
      if (networkSvg) sigmoidContributions(networkSvg, frame);
      return;
    }
    const model = root.NNClassificationData;
    if (!model) throw new Error('The fitted binary network must be loaded before its views.');
    if (functionSvg) classificationPlot(functionSvg, frame, model);
    if (networkSvg) {
      if (frame.scene === 'class-input' || frame.scene === 'class-trace') classificationNetwork(networkSvg, frame, model);
      else inputReference(networkSvg, frame, model);
    }
  }

  root.NNExtensionViews = Object.freeze({ render });
})(typeof window !== 'undefined' ? window : globalThis);
