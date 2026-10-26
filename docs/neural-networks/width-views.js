/* Scroll-driven input-space views of frozen, independently fitted widths. */
(function (root) {
  'use strict';

  const C = { ink: '#242424', blue: '#234E70', rust: '#A54F32', teal: '#007F73', muted: '#606060', rule: '#767676', pale: '#DADADA', error: '#9B6A00' };
  const mathCache = new Map();
  const clamp = value => Math.max(0, Math.min(1, Number(value) || 0));
  const esc = text => String(text).replace(/[&<>\"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '\"': '&quot;' })[c]);
  const rounded = value => Number(value.toFixed(3));
  const percent = value => (100 * value).toFixed(2) + '%';

  function math(tex) {
    if (!mathCache.has(tex)) mathCache.set(tex, root.katex.renderToString(tex, {
      output: 'htmlAndMathml', displayMode: false, throwOnError: true, strict: 'error', trust: false
    }));
    return mathCache.get(tex);
  }

  function builder(svg, compact, description) {
    const width = compact ? 620 : 900, height = compact ? 540 : 480;
    svg.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', description);
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    svg.removeAttribute('aria-hidden');
    ['data-width-model-index','data-width-previous-index','data-width-model-mix','data-noise-mix','data-test-opacity','data-width-boundary-opacity','data-width-error-index','data-width-error-scale','data-measured-error','data-double-descent-schematic','data-schematic-progress'].forEach(name => svg.removeAttribute(name));
    const body = ['<style>.nw-text{font-family:Arial,sans-serif;fill:' + C.ink + ';font-size:' + (compact ? 22 : 16) + 'px}.nw-tick{font-family:Arial,sans-serif;fill:' + C.muted + ';font-size:' + (compact ? 20 : 14) + 'px}.nw-math .katex{font-size:1em;line-height:1.1}</style>'];
    return {
      add: markup => body.push(markup),
      line: (x1, y1, x2, y2, attrs) => body.push('<line x1="' + rounded(x1) + '" y1="' + rounded(y1) + '" x2="' + rounded(x2) + '" y2="' + rounded(y2) + '" ' + (attrs || '') + '/>'),
      text: (x, y, text, attrs) => body.push('<text x="' + rounded(x) + '" y="' + rounded(y) + '" class="nw-text" ' + (attrs || '') + '>' + esc(text) + '</text>'),
      circle: (x, y, radius, attrs) => body.push('<circle cx="' + rounded(x) + '" cy="' + rounded(y) + '" r="' + radius + '" ' + (attrs || '') + '/>'),
      path: (d, attrs) => body.push('<path d="' + d + '" ' + (attrs || '') + '/>'),
      math: (x, y, tex, boxWidth, size, attrs) => {
        const font = size || (compact ? 25 : 19), boxHeight = font * 2.8;
        body.push('<foreignObject class="nw-math" data-tex="' + esc(tex) + '" x="' + rounded(x - boxWidth / 2) + '" y="' + rounded(y - boxHeight / 2) + '" width="' + boxWidth + '" height="' + boxHeight + '" overflow="visible" ' + (attrs || '') + '><div xmlns="http://www.w3.org/1999/xhtml" style="height:100%;display:flex;align-items:center;justify-content:center;color:inherit;font-size:' + font + 'px;white-space:nowrap">' + math(tex) + '</div></foreignObject>');
      },
      finish: () => { svg.innerHTML = body.join(''); }
    };
  }

  function score(model, input) {
    return model.hiddenWeights.reduce((value, weights, j) => value + model.outputWeights[j] * Math.max(0, model.hiddenBias[j] + weights[0] * input[0] + weights[1] * input[1]), model.outputBias);
  }

  function marker(b, x, y, label, radius, hollow, attrs) {
    const color = label === 0 ? C.blue : C.rust;
    const style = 'fill="' + (hollow ? '#fff' : color) + '" stroke="' + (hollow ? color : '#fff') + '" stroke-width="' + (hollow ? 1.8 : 1) + '" ' + (attrs || '');
    if (label === 1) b.add('<rect x="' + rounded(x - radius) + '" y="' + rounded(y - radius) + '" width="' + 2 * radius + '" height="' + 2 * radius + '" ' + style + '/>');
    else b.circle(x, y, radius, style);
  }

  function axis(b, box, domain, compact) {
    const px = x => box.left + (x - domain[0][0]) / (domain[0][1] - domain[0][0]) * (box.right - box.left);
    const py = y => box.bottom - (y - domain[1][0]) / (domain[1][1] - domain[1][0]) * (box.bottom - box.top);
    b.line(box.left, box.top, box.left, box.bottom, 'stroke="' + C.rule + '"');
    b.line(box.left, box.bottom, box.right, box.bottom, 'stroke="' + C.rule + '"');
    [0, 1, 2, 3].filter(x => x >= domain[0][0] && x <= domain[0][1]).forEach(x => {
      b.line(px(x), box.bottom, px(x), box.bottom + 5, 'stroke="' + C.rule + '"');
      b.text(px(x), box.bottom + (compact ? 26 : 23), x, 'text-anchor="middle" style="font-size:' + (compact ? 20 : 14) + 'px;fill:' + C.muted + '"');
    });
    [-1.5, -1, -.5, 0, .5, 1, 1.5].filter(y => y >= domain[1][0] && y <= domain[1][1]).forEach(y => {
      b.line(box.left - 5, py(y), box.left, py(y), 'stroke="' + C.rule + '"');
      b.text(box.left - 9, py(y) + (compact ? 7 : 5), y, 'text-anchor="end" style="font-size:' + (compact ? 20 : 14) + 'px;fill:' + C.muted + '"');
    });
    b.math(box.right + (compact ? 19 : 16), box.bottom + 2, 'x_1', 50);
    b.math(box.left, box.top - 23, 'x_2', 50);
    return { px, py };
  }

  function boundary(b, model, p, opacity, baseline) {
    if (opacity <= 0) return;
    b.add('<g data-width-boundary="' + model.width + '" data-width-baseline="' + Boolean(baseline) + '" opacity="' + opacity + '">');
    (model.boundaries || []).forEach((line, index) => {
      if (line.length < 2) return;
      const d = line.map((point, i) => (i ? 'L' : 'M') + rounded(p.px(point[0])) + ',' + rounded(p.py(point[1]))).join(' ');
      b.path(d, 'fill="none" stroke="' + (baseline ? C.rule : C.ink) + '" stroke-width="' + (baseline ? 1.6 : 2.6) + '"' + (baseline ? ' stroke-dasharray="7 6"' : '') + ' data-fitted-boundary="true" data-contour-index="' + index + '"');
    });
    b.add('</g>');
  }

  function network(b, model, compact, previous, mix) {
    const width = model.width;
    const inputX = compact ? 106 : 671, hiddenX = compact ? 303 : 766, outputX = compact ? 508 : 859;
    const mid = compact ? 450 : 238, span = compact ? 122 : 230;
    const inputYs = compact ? [422, 478] : [202, 274];
    const radius = compact ? 19 : 18;
    const individual = width <= 8;
    const nodes = individual ? Array.from({ length: width }, (_, i) => i) : [0, 1, 2, width - 2, width - 1];
    const targetPositions = individual ? nodes.map((_, i) => mid - span / 2 + span * i / Math.max(1, nodes.length - 1)) : [mid - span / 2, mid - span / 3, mid - span / 6, mid + span / 3, mid + span / 2];
    const adding = individual && previous.width < width;
    const positions = targetPositions.map((target, j) => {
      const start = j < previous.width ? mid - span / 2 + span * j / Math.max(1, previous.width - 1) : mid;
      return adding ? start * (1 - mix) + target * mix : target;
    });
    const visibility = j => adding && j >= previous.width ? mix : 1;
    const hiddenRadius = compact ? (width <= 4 ? 13 : 7.5) : (width <= 4 ? 17 : 11);
    b.add('<g data-width-network="' + width + '" data-hidden-count="' + width + '" data-hidden-layers="1" data-input-count="2" data-output-count="1">');
    b.text(hiddenX, compact ? 369 : 52, width + ' hidden neurons', 'text-anchor="middle" data-width-counter="' + width + '"');
    nodes.forEach((j, i) => {
      inputYs.forEach(y => b.line(inputX + radius, y, hiddenX - hiddenRadius, positions[i], 'stroke="' + C.pale + '" stroke-width="1.1" opacity="' + visibility(j) + '"'));
      b.line(hiddenX + hiddenRadius, positions[i], outputX - radius, mid, 'stroke="' + C.pale + '" stroke-width="1.1" opacity="' + visibility(j) + '"');
    });
    inputYs.forEach((y, i) => {
      b.circle(inputX, y, radius, 'fill="#fff" stroke="' + C.ink + '" stroke-width="1.4"');
      b.math(inputX, y, 'x_' + (i + 1), 45, compact ? 24 : 19);
    });
    nodes.forEach((j, i) => {
      b.circle(hiddenX, positions[i], hiddenRadius, 'fill="#fff" stroke="' + C.teal + '" stroke-width="1.8" opacity="' + visibility(j) + '" data-hidden-node="' + j + '"');
      if (width <= 4 && !compact) b.math(hiddenX, positions[i], 'h_' + (j + 1), 38, 16, 'style="color:' + C.teal + '" opacity="' + visibility(j) + '"');
    });
    if (!individual) b.math(hiddenX, mid + span / 12, '\\vdots', 38, compact ? 24 : 21, 'style="color:' + C.teal + '"');
    b.circle(outputX, mid, radius, 'fill="#fff" stroke="' + C.blue + '" stroke-width="1.8"');
    b.math(outputX, mid, '\\sigma', 45, compact ? 25 : 22, 'style="color:' + C.blue + '"');
    if (compact) {
      b.text(inputX, 527, 'Inputs', 'text-anchor="middle" style="font-size:20px;fill:' + C.muted + '"');
      b.text(outputX, 527, 'Sigmoid head', 'text-anchor="middle" style="font-size:20px;fill:' + C.muted + '"');
    } else {
      b.text(inputX, 389, 'Inputs', 'text-anchor="middle" style="font-size:14px;fill:' + C.muted + '"');
      b.text(outputX, 389, 'Head', 'text-anchor="middle" style="font-size:14px;fill:' + C.muted + '"');
      b.math(hiddenX, 423, '\\mathbf h\\in\\mathbb R^{' + width + '}', 200, 21);
    }
    b.add('</g>');
  }

  function render(svg, frame) {
    const data = root.NNWidthData;
    if (!data || !data.models || !data.models.length) throw new Error('Frozen width-experiment data must load before its view.');
    const index = Math.max(0, Math.min(data.models.length - 1, Math.floor(Number(frame.modelIndex) || 0)));
    const previousIndex = frame.previousModelIndex === undefined ? index : Math.max(0, Math.min(data.models.length - 1, Math.floor(Number(frame.previousModelIndex) || 0)));
    const model = data.models[index], previous = data.models[previousIndex];
    const mix = previousIndex === index ? 1 : clamp(frame.modelMix);
    const noise = frame.noiseMix === undefined ? 1 : clamp(frame.noiseMix), testOpacity = clamp(frame.testOpacity);
    const boundaryOpacity = frame.boundaryOpacity === undefined ? 1 : clamp(frame.boundaryOpacity);
    const compact = Boolean(frame.compact), box = compact ? { left: 64, right: 568, top: 71, bottom: 302 } : { left: 63, right: 600, top: 80, bottom: 381 };
    const description = 'The same binary training observations in original input coordinates. ' + model.width + ' ReLU hidden neurons feed one sigmoid head. ' + (noise < 1 ? 'The observations move from their noise-free positions to the fixed noisy sample. ' : 'The noisy training sample stays fixed across fitted widths. ') + (testOpacity > 0 ? 'Hollow markers are independent test observations; outlined halos mark their classification mistakes. ' : '') + 'Each displayed boundary comes from frozen fitted network weights.';
    const b = builder(svg, compact, description);
    svg.setAttribute('data-width-model-index', index);
    svg.setAttribute('data-width-previous-index', previousIndex);
    svg.setAttribute('data-width-model-mix', mix);
    svg.setAttribute('data-noise-mix', noise);
    svg.setAttribute('data-test-opacity', testOpacity);
    svg.setAttribute('data-width-boundary-opacity', boundaryOpacity);
    const p = axis(b, box, data.domain, compact);
    const legendY = compact ? 23 : 26;
    marker(b, box.left + 5, legendY - 5, 0, 5, false);
    b.text(box.left + 19, legendY, 'Class 0');
    marker(b, box.left + (compact ? 165 : 126), legendY - 5, 1, 5, false);
    b.text(box.left + (compact ? 179 : 140), legendY, 'Class 1');
    if (testOpacity > 0) b.text(box.right, legendY, 'Hollow: test data', 'text-anchor="end" style="font-size:' + (compact ? 18 : 14) + 'px;fill:' + C.muted + '" opacity="' + testOpacity + '"');
    b.add('<defs><clipPath id="width-data-clip"><rect x="' + box.left + '" y="' + box.top + '" width="' + (box.right - box.left) + '" height="' + (box.bottom - box.top) + '"/></clipPath></defs><g clip-path="url(#width-data-clip)">');
    if (frame.compareBaseline && index > 0) boundary(b, data.models[0], p, .3 * boundaryOpacity, true);
    if (previousIndex !== index) boundary(b, previous, p, (1 - mix) * boundaryOpacity, false);
    boundary(b, model, p, mix * boundaryOpacity, false);
    if (testOpacity > 0) {
      b.add('<g data-width-test-points="true" opacity="' + testOpacity + '">');
      (data.testInputs || []).forEach((input, i) => {
        const label = data.testLabels[i], x = p.px(input[0]), y = p.py(input[1]);
        const currentWrong = Number(score(model, input) >= 0) !== label;
        const previousWrong = Number(score(previous, input) >= 0) !== label;
        const haloOpacity = previousIndex === index ? Number(currentWrong) : (1 - mix) * Number(previousWrong) + mix * Number(currentWrong);
        if (haloOpacity > 0) b.circle(x, y, compact ? 9 : 8, 'fill="none" stroke="' + C.error + '" stroke-width="1.8" stroke-dasharray="2 2" opacity="' + haloOpacity + '" data-test-error="' + i + '"');
        marker(b, x, y, label, compact ? 4.6 : 4.1, true, 'data-test-point-id="' + i + '" data-label="' + label + '" data-input-x1="' + input[0] + '" data-input-x2="' + input[1] + '"');
      });
      b.add('</g>');
    }
    data.inputs.forEach((input, i) => {
      const clean = -(Math.sin(input[0]) ** 2) + .9 * data.labels[i];
      const y = clean * (1 - noise) + input[1] * noise;
      marker(b, p.px(input[0]), p.py(y), data.labels[i], compact ? 5.1 : 4.7, false, 'data-width-point-id="' + i + '" data-label="' + data.labels[i] + '" data-input-x1="' + input[0] + '" data-input-x2="' + input[1] + '" data-display-x2="' + y + '"');
    });
    b.add('</g>');
    if (frame.showErrors) {
      const y = compact ? 345 : 431;
      b.text(box.left, y, 'Training error ' + percent(model.trainingError), 'data-training-error="' + model.trainingError + '" style="font-size:' + (compact ? 21 : 17) + 'px"');
      if (testOpacity > 0) b.text(box.right, y, 'Test error ' + percent(model.testError), 'text-anchor="end" data-test-error-rate="' + model.testError + '" opacity="' + testOpacity + '" style="font-size:' + (compact ? 21 : 17) + 'px"');
    } else if (!compact) b.text(box.left, 431, noise < 1 ? 'Adding noise to the same two class curves' : 'Same noisy observations at every width', 'style="font-size:15px;fill:' + C.muted + '"');
    network(b, model, compact, previous, mix);
    b.finish();
  }

  function renderError(svg, frame) {
    const data = root.NNWidthData;
    if (!data || !data.models || !data.models.length) throw new Error('Frozen width-experiment data must load before its error view.');
    const compact = Boolean(frame.compact), models = data.models;
    const index = frame.modelIndex === undefined ? models.length - 1 : Math.max(0, Math.min(models.length - 1, Math.floor(Number(frame.modelIndex) || 0)));
    const model = models[index];
    const b = builder(svg, compact, 'Measured training and independent-test classification error versus hidden-layer width. Width has a logarithmic scale. Only measured fits are shown; this sample does not demonstrate a full double-descent curve.');
    const box = compact ? { left: 75, right: 572, top: 105, bottom: 416 } : { left: 95, right: 835, top: 105, bottom: 376 };
    const minLog = Math.log2(models[0].width), maxLog = Math.log2(models[models.length - 1].width);
    const ymax = Math.max(1, Math.ceil(Math.max(...models.flatMap(m => [m.trainingError, m.testError])) * 400) / 4) + .1;
    const ymin = -.08 * ymax;
    const px = width => box.left + (Math.log2(width) - minLog) / (maxLog - minLog) * (box.right - box.left);
    const py = error => box.bottom - (error - ymin) / (ymax - ymin) * (box.bottom - box.top);
    svg.setAttribute('data-width-error-index', index);
    svg.setAttribute('data-width-error-scale', 'log2');
    svg.setAttribute('data-measured-error', 'true');
    b.line(box.left, box.top, box.left, box.bottom, 'stroke="' + C.rule + '"');
    b.line(box.left, box.bottom, box.right, box.bottom, 'stroke="' + C.rule + '"');
    [2, 8, 32, 128, 256].filter(width => width >= models[0].width && width <= models[models.length - 1].width).forEach(width => {
      const x = px(width);
      b.line(x, box.bottom, x, box.bottom + 5, 'stroke="' + C.rule + '"');
      b.text(x, box.bottom + (compact ? 29 : 25), width, 'text-anchor="middle" style="font-size:' + (compact ? 21 : 15) + 'px;fill:' + C.muted + '"');
    });
    const tickStep = ymax > 2 ? .5 : .25;
    for (let value = 0; value <= ymax; value += tickStep) {
      b.line(box.left, py(value), box.right, py(value), 'stroke="' + C.pale + '" stroke-dasharray="3 5"');
      b.text(box.left - 11, py(value) + (compact ? 7 : 5), Number(value.toFixed(2)), 'text-anchor="end" style="font-size:' + (compact ? 21 : 15) + 'px;fill:' + C.muted + '"');
    }
    b.math((box.left + box.right) / 2, box.bottom + (compact ? 70 : 58), '\\text{Hidden neurons }m', 420, compact ? 26 : 22);
    b.math(box.left + (compact ? 142 : 162), box.top - 26, '\\text{Classification error (\\%)}', compact ? 330 : 360, compact ? 24 : 21);
    const shown = models.slice(0, index + 1);
    [['trainingError', 'Training', C.blue, '7 5'], ['testError', 'Test', C.rust, '']].forEach(([key, label, color, dash], j) => {
      const legendX = box.left + j * (compact ? 220 : 220);
      b.line(legendX, 27, legendX + 31, 27, 'stroke="' + color + '" stroke-width="2.5"' + (dash ? ' stroke-dasharray="' + dash + '"' : ''));
      b.text(legendX + 42, compact ? 34 : 32, label);
      const d = shown.map((fit, i) => (i ? 'L' : 'M') + rounded(px(fit.width)) + ',' + rounded(py(fit[key] * 100))).join(' ');
      b.path(d, 'fill="none" stroke="' + color + '" stroke-width="2.7"' + (dash ? ' stroke-dasharray="' + dash + '"' : '') + ' data-error-series="' + key + '"');
      shown.forEach((fit, i) => b.circle(px(fit.width), py(fit[key] * 100), i === index ? (compact ? 7 : 6) : 3.5, 'fill="' + color + '" stroke="#fff" stroke-width="1.2" data-error-width="' + fit.width + '" data-error-value="' + fit[key] + '" data-error-kind="' + key + '"'));
    });
    b.line(px(model.width), box.top, px(model.width), box.bottom, 'stroke="' + C.rule + '" stroke-width="1" stroke-dasharray="2 5" opacity=".4"');
    b.math(box.right - (compact ? 32 : 48), box.top - 25, 'm=' + model.width, compact ? 130 : 160, compact ? 24 : 22);
    b.text(box.left, compact ? 528 : 469, 'Measured fits · same training sample and fitting procedure', 'style="font-size:' + (compact ? 18 : 15) + 'px;fill:' + C.muted + '"');
    b.finish();
  }

  function renderSchematic(svg, frame) {
    const compact = Boolean(frame.compact), progress = clamp(frame.schematicProgress ?? frame.errorReveal ?? frame.progress ?? 1);
    const b = builder(svg, compact, 'Conceptual double-descent diagram, not measured from this sample. Test error first falls, rises near the interpolation threshold, then falls again as model capacity increases. Training error reaches zero at the interpolation threshold.');
    const box = compact ? { left: 74, right: 574, top: 111, bottom: 417 } : { left: 95, right: 835, top: 92, bottom: 376 };
    const px = x => box.left + x * (box.right - box.left), py = y => box.bottom - y * (box.bottom - box.top);
    const xy = (x, y) => rounded(px(x)) + ',' + rounded(py(y));
    svg.setAttribute('data-double-descent-schematic', 'true');
    svg.setAttribute('data-schematic-progress', progress);
    b.text(box.left, compact ? 29 : 27, 'Schematic — not measured from this sample', 'style="font-size:' + (compact ? 20 : 17) + 'px;fill:' + C.muted + '"');
    b.line(box.left, box.top, box.left, box.bottom, 'stroke="' + C.rule + '"');
    b.line(box.left, box.bottom, box.right, box.bottom, 'stroke="' + C.rule + '"');
    b.math(box.left + (compact ? 98 : 110), box.top - 25, '\\text{Prediction error}', 270, compact ? 25 : 22);
    b.math((box.left + box.right) / 2, box.bottom + (compact ? 70 : 58), '\\text{Model capacity}', 360, compact ? 27 : 23);
    const threshold = .48;
    b.line(px(threshold), box.top + 9, px(threshold), box.bottom, 'stroke="' + C.rule + '" stroke-dasharray="3 6"');
    b.text(px(threshold) + (compact ? 14 : 17), box.top + (compact ? 18 : 14), 'Interpolation', 'style="font-size:' + (compact ? 20 : 15) + 'px;fill:' + C.muted + '"');
    b.text(px(threshold) + (compact ? 14 : 17), box.top + (compact ? 43 : 34), 'threshold', 'style="font-size:' + (compact ? 20 : 15) + 'px;fill:' + C.muted + '"');
    b.add('<defs><clipPath id="width-schematic-reveal"><rect x="' + box.left + '" y="' + box.top + '" width="' + rounded((box.right - box.left) * progress) + '" height="' + (box.bottom - box.top + 3) + '"/></clipPath></defs><g clip-path="url(#width-schematic-reveal)">');
    const train = 'M' + xy(.025, .77) + ' C' + xy(.12, .35) + ' ' + xy(.33, .035) + ' ' + xy(threshold, 0) + ' L' + xy(.98, 0);
    const test = 'M' + xy(.025, .88) + ' C' + xy(.10, .51) + ' ' + xy(.16, .34) + ' ' + xy(.24, .34) + ' C' + xy(.34, .34) + ' ' + xy(.41, .84) + ' ' + xy(threshold, .84) + ' C' + xy(.57, .84) + ' ' + xy(.61, .2) + ' ' + xy(.98, .17);
    b.path(train, 'fill="none" stroke="' + C.blue + '" stroke-width="2.8" stroke-dasharray="7 5" data-schematic-series="training"');
    b.path(test, 'fill="none" stroke="' + C.rust + '" stroke-width="3" data-schematic-series="test"');
    b.add('</g>');
    if (progress > .8) {
      b.text(px(.83), py(.18) - (compact ? 14 : 11), 'Test', 'style="fill:' + C.rust + '"');
      b.text(px(.76), py(0) - (compact ? 13 : 10), 'Training', 'style="fill:' + C.blue + '"');
    }
    b.text(box.left, compact ? 528 : 469, 'A possible pattern across model capacities; not a guarantee.', 'style="font-size:' + (compact ? 18 : 15) + 'px;fill:' + C.muted + '"');
    b.finish();
  }

  root.NNWidthViews = Object.freeze({ render, renderError, renderSchematic });
})(typeof window !== 'undefined' ? window : globalThis);
