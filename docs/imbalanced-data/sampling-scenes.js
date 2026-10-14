(() => {
  'use strict';
  const C = {full: '#234e70', sample: '#a54f32', corrected: '#007f73', truth: '#777777', ink: '#242424', muted: '#606060', grid: '#e4e7e9'};
  const clamp = (value, low = 0, high = 1) => Math.max(low, Math.min(high, value));
  const safe = value => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const number = value => Number(value).toLocaleString('en-US');
  const fixed = (value, digits = 4) => Number(value).toFixed(digits).replace(/^-/, '−');
  const signed = value => `${value < 0 ? '−' : '+'}${Math.abs(value).toFixed(4)}`;
  const text = (x, y, value, color = C.ink, size = 18, anchor = 'start', extra = '') => `<text x="${x}" y="${y}" fill="${color}" text-anchor="${anchor}" style="font-size:${size}px;font-variant-numeric:tabular-nums" ${extra}>${safe(value)}</text>`;
  const line = (x1, y1, x2, y2, color = C.grid, extra = '') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" ${extra}/>`;
  const rect = (x, y, width, height, color, extra = '') => `<rect x="${x}" y="${y}" width="${Math.max(0, width)}" height="${Math.max(0, height)}" fill="${color}" ${extra}/>`;
  const path = (points, color, extra = '') => `<path d="${points.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(3)},${p[1].toFixed(3)}`).join(' ')}" fill="none" stroke="${color}" stroke-width="3.5" ${extra}/>`;
  const svg = (body, label, narrow) => `<svg class="scene-chart sampling-chart" viewBox="0 0 ${narrow ? 380 : 900} 420" role="img" aria-label="${safe(label)}" xmlns="http://www.w3.org/2000/svg"><g font-family="LM, Georgia, serif">${body}</g></svg>`;
  const truth = (example, x) => example.trueCoefficients.intercept + example.trueCoefficients.linear * x + example.trueCoefficients.quadratic * x * x;
  const predict = (fit, x) => fit.intercept + fit.slope * x;

  function countBar(x, y, width, height, fit, scale) {
    const positiveWidth = width * fit.positive / scale;
    const negativeWidth = width * (fit.n - fit.positive) / scale;
    return rect(x, y, width, height, '#eef1f3', 'rx="2"') + rect(x, y, positiveWidth, height, C.sample) + rect(x + positiveWidth, y, negativeWidth, height, C.full);
  }

  function protocolFigure(data, narrow) {
    const W = narrow ? 380 : 900;
    let body = text(W / 2, narrow ? 20 : 27, `${number(data.protocol.nTrain)} training cases per example`, C.ink, narrow ? 17 : 23, 'middle');
    body += text(W / 2, narrow ? 43 : 55, `${number(data.protocol.nTest)} separate test cases per example`, C.muted, narrow ? 14 : 18, 'middle');
    [data.cases.correct, data.cases.curved].forEach((example, i) => {
      const left = narrow ? 12 : 36 + i * 432, top = narrow ? 63 + i * 174 : 82;
      const width = narrow ? 356 : 396, inner = left + 16, barWidth = width - 32;
      body += rect(left, top, width, narrow ? 163 : 303, 'white', 'stroke="#d8dfe3" rx="5"');
      body += text(inner, top + (narrow ? 25 : 31), i === 0 ? 'True linear log odds' : 'True curved log odds', C.ink, narrow ? 18 : 23);
      if (narrow) {
        body += text(inner, top + 51, `Full data: ${number(example.full.n)} cases`, C.full, 16);
        body += text(inner, top + 73, `${number(example.full.positive)} positive · ${number(example.full.n - example.full.positive)} negative`, C.muted, 14);
        body += countBar(inner, top + 82, barWidth, 11, example.full, example.full.n);
        body += text(inner, top + 117, `Retained draw: ${number(example.draw.n)} cases`, C.sample, 16);
        body += text(inner, top + 139, `${number(example.draw.positive)} positive · ${number(example.draw.n - example.draw.positive)} negative`, C.muted, 14);
        body += countBar(inner, top + 147, barWidth, 10, example.draw, example.full.n);
      } else {
        body += text(inner, top + 73, `Full data: ${number(example.full.n)} cases`, C.full, 21);
        body += text(inner, top + 104, `${number(example.full.positive)} positive · ${number(example.full.n - example.full.positive)} negative`, C.muted, 18);
        body += countBar(inner, top + 117, barWidth, 18, example.full, example.full.n);
        body += text(inner, top + 170, 'Keep every positive; retain 10% of negatives.', C.muted, 16);
        body += line(left + width / 2, top + 184, left + width / 2, top + 203, C.sample, 'stroke-width="2"');
        body += `<path d="M${left + width / 2 - 5},${top + 198} L${left + width / 2},${top + 204} L${left + width / 2 + 5},${top + 198}" fill="none" stroke="${C.sample}" stroke-width="2"/>`;
        body += text(inner, top + 235, `Retained draw: ${number(example.draw.n)} cases`, C.sample, 21);
        body += text(inner, top + 265, `${number(example.draw.positive)} positive · ${number(example.draw.n - example.draw.positive)} negative`, C.muted, 18);
        body += countBar(inner, top + 277, barWidth, 16, example.draw, example.full.n);
      }
    });
    body += text(W / 2, narrow ? 416 : 411, 'Bars: orange = positive; blue = negative; pale = removed.', C.muted, narrow ? 12.3 : 16, 'middle');
    const label = `Two realized training datasets of ${number(data.protocol.nTrain)} cases each and separate untouched test datasets of ${number(data.protocol.nTest)} cases each. ${[data.cases.correct, data.cases.curved].map(example => `${example.label}: full ${number(example.full.n)} cases, ${number(example.full.positive)} positives; retained draw ${number(example.draw.n)} cases, ${number(example.draw.positive)} positives.`).join(' ')}`;
    return svg(body, label, narrow);
  }

  function fitFigure(example, data, narrow) {
    const W = narrow ? 380 : 900, left = narrow ? 43 : 78, right = narrow ? 353 : 849;
    const top = narrow ? 65 : 43, bottom = narrow ? 226 : 251;
    const xmin = Math.min(...example.x), xmax = Math.max(...example.x);
    const fits = [example.full.fit, example.draw.fit, example.corrected.fit];
    const values = example.x.flatMap(x => [truth(example, x), ...fits.map(fit => predict(fit, x))]);
    const lo = Math.floor(Math.min(...values)) - .5, hi = Math.ceil(Math.max(...values)) + .5;
    const fx = x => left + (right - left) * (x - xmin) / (xmax - xmin);
    const fy = value => bottom - (bottom - top) * (value - lo) / (hi - lo);
    let body = text(left, 22, 'Log odds', C.muted, narrow ? 15 : 19);
    const legend = (x, y, label, color, dashed = false) => line(x, y - 5, x + (narrow ? 15 : 25), y - 5, color, `stroke-width="3" ${dashed ? 'stroke-dasharray="4 3"' : ''}`) + text(x + (narrow ? 21 : 32), y, label, color, narrow ? 12.5 : 17);
    if (narrow) {
      body += legend(43, 45, 'Truth', C.truth, true) + legend(116, 45, 'Full', C.full) + legend(181, 45, 'Sampled', C.sample) + legend(278, 45, 'Corrected', C.corrected, true);
    } else {
      body += legend(331, 22, 'Truth', C.truth, true) + legend(448, 22, 'Full', C.full) + legend(553, 22, 'Sampled', C.sample) + legend(704, 22, 'Corrected', C.corrected, true);
    }
    for (let value = Math.ceil(lo / 2) * 2; value < hi; value += 2) {
      body += line(left, fy(value), right, fy(value));
      body += text(left - 10, fy(value) + 5, value < 0 ? `−${-value}` : value, C.muted, narrow ? 13 : 17, 'end');
    }
    body += line(left, top, left, bottom, '#aeb9c1') + line(left, bottom, right, bottom, '#aeb9c1');
    example.x.forEach(x => { body += text(fx(x), bottom + (narrow ? 22 : 25), x, C.muted, narrow ? 14 : 18, 'middle'); });
    body += text(right, bottom + (narrow ? 40 : 46), 'Predictor x', C.muted, narrow ? 13 : 17, 'end');
    const curve = Array.from({length: 161}, (_, i) => { const x = xmin + (xmax - xmin) * i / 160; return [fx(x), fy(truth(example, x))]; });
    body += path(curve, C.truth, 'stroke-dasharray="4 4"');
    body += path([[fx(xmin), fy(predict(example.full.fit, xmin))], [fx(xmax), fy(predict(example.full.fit, xmax))]], C.full);
    body += path([[fx(xmin), fy(predict(example.draw.fit, xmin))], [fx(xmax), fy(predict(example.draw.fit, xmax))]], C.sample);
    body += path([[fx(xmin), fy(predict(example.corrected.fit, xmin))], [fx(xmax), fy(predict(example.corrected.fit, xmax))]], C.corrected, 'stroke-dasharray="8 4"');

    const columns = narrow ? [14, 178, 250, 361] : [80, 425, 590, 842];
    const headerY = narrow ? 294 : 314, rowTop = narrow ? 323 : 345, gap = narrow ? 32 : 29;
    body += text(columns[0], headerY, 'Fit', C.muted, narrow ? 14 : 17);
    body += text(columns[1], headerY, 'Intercept', C.muted, narrow ? 13 : 17, 'end');
    body += text(columns[2], headerY, 'Slope', C.muted, narrow ? 13 : 17, 'end');
    body += text(columns[3], headerY, narrow ? 'Test loss' : 'Untouched test log loss', C.muted, narrow ? 13 : 17, 'end');
    body += line(columns[0], headerY + 9, columns[3], headerY + 9);
    [['Full', example.full, C.full], ['Sampled', example.draw, C.sample], ['Corrected', example.corrected, C.corrected]].forEach(([name, fit, color], i) => {
      const y = rowTop + i * gap;
      body += text(columns[0], y, name, color, narrow ? 16 : 21);
      body += text(columns[1], y, fixed(fit.fit.intercept), color, narrow ? 17 : 22, 'end');
      body += text(columns[2], y, fixed(fit.fit.slope), color, narrow ? 17 : 22, 'end');
      body += text(columns[3], y, fixed(fit.testLogLoss, 6), color, narrow ? 17 : 22, 'end');
    });
    body += text(W / 2, narrow ? 415 : 416, `Same untouched ${number(data.protocol.nTest)}-case test set; lower loss is better.`, C.muted, narrow ? 12.5 : 15, 'middle');
    return svg(body, `${example.label}. Gray dashed curve is the data-generating truth. Blue is the full-data fitted line; orange is the sampled fitted line; teal is the prior-corrected sampled line. ${[['Full', example.full], ['Sampled', example.draw], ['Corrected', example.corrected]].map(([label, fit]) => `${label}: intercept ${fixed(fit.fit.intercept)}, slope ${fixed(fit.fit.slope)}, untouched test log loss ${fixed(fit.testLogLoss, 6)}.`).join(' ')}`, narrow);
  }

  function histogram(values, lo, hi, bins = 42) {
    const frequencies = Array(bins).fill(0);
    values.forEach(value => { frequencies[clamp(Math.floor((value - lo) / (hi - lo) * bins), 0, bins - 1)]++; });
    return frequencies;
  }

  function repeatFigure(data, narrow) {
    const cases = [data.cases.correct, data.cases.curved], colors = [C.full, C.sample];
    const deltas = cases.map(example => example.repeats.map(draw => draw.slope - example.full.fit.slope));
    const all = deltas.flat(), min = Math.min(0, ...all), max = Math.max(0, ...all);
    const padding = Math.max((max - min) * .07, .001), lo = min - padding, hi = max + padding;
    const W = narrow ? 380 : 900, left = narrow ? 43 : 80, right = narrow ? 353 : 848;
    const fx = value => left + (right - left) * (value - lo) / (hi - lo);
    let body = text(W / 2, narrow ? 19 : 26, 'Solid marker = median · dashed black line = zero', C.muted, narrow ? 12.3 : 17, 'middle');
    cases.forEach((example, i) => {
      const color = colors[i], summary = example.summary.slopeDelta;
      const titleY = narrow ? 53 + i * 179 : 62 + i * 152;
      const chartTop = narrow ? 112 + i * 179 : 99 + i * 152;
      const chartBottom = chartTop + (narrow ? 61 : 62);
      body += text(left, titleY, i === 0 ? 'True linear log odds' : 'True curved log odds', color, narrow ? 18 : 22);
      if (narrow) {
        body += text(left, titleY + 23, `Mean Δ ${signed(summary.mean)} · median ${signed(summary.median)}`, C.ink, 14.5);
        body += text(left, titleY + 45, `Middle 95%: ${signed(summary.q025)} to ${signed(summary.q975)}`, C.muted, 14);
      } else {
        body += text(right, titleY, `Mean Δ ${signed(summary.mean)} · median ${signed(summary.median)}`, color, 19, 'end');
        body += text(left, titleY + 26, `Middle 95%: ${signed(summary.q025)} to ${signed(summary.q975)}`, C.muted, 17);
      }
      body += rect(fx(summary.q025), chartTop - 3, fx(summary.q975) - fx(summary.q025), chartBottom - chartTop + 6, color, 'opacity=".10"');
      const frequencies = histogram(deltas[i], lo, hi), peak = Math.max(...frequencies), barWidth = (right - left) / frequencies.length;
      frequencies.forEach((frequency, bin) => {
        const h = (chartBottom - chartTop) * frequency / peak;
        body += rect(left + bin * barWidth + .5, chartBottom - h, Math.max(1, barWidth - 1), h, color, 'opacity=".65"');
      });
      body += line(left, chartBottom, right, chartBottom, '#aeb9c1');
      body += line(fx(0), chartTop - 4, fx(0), chartBottom + 4, C.ink, 'stroke-width="1.7" stroke-dasharray="4 3"');
      body += line(fx(summary.median), chartTop - 4, fx(summary.median), chartBottom + 4, color, 'stroke-width="2.4"');
    });
    const tickY = narrow ? 376 : 341;
    const step = (hi - lo) / 5, ticks = [0];
    for (let i = 0; i <= 5; i++) {
      const value = lo + step * i;
      if (Math.abs(fx(value) - fx(0)) > (narrow ? 36 : 52)) ticks.push(value);
    }
    ticks.sort((a, b) => a - b).forEach(value => { body += text(fx(value), tickY, value === 0 ? '0' : fixed(value, 2), C.muted, narrow ? 13 : 17, 'middle'); });
    body += text(W / 2, narrow ? 405 : 379, 'Δ slope = sampled fit − full-data fit', C.ink, narrow ? 16 : 21, 'middle');
    if (!narrow) body += text(W / 2, 410, 'Both histograms use the same horizontal scale; shaded bands contain the middle 95% of draws.', C.muted, 16, 'middle');
    return svg(body, `Empirical slope-change distributions across ${number(data.protocol.repetitions)} independent negative-downsampling draws from each fixed full dataset. ${cases.map(example => `${example.label}: mean change ${signed(example.summary.slopeDelta.mean)}, median ${signed(example.summary.slopeDelta.median)}, middle 95 percent ${signed(example.summary.slopeDelta.q025)} to ${signed(example.summary.slopeDelta.q975)}.`).join(' ')} These are conditional repeat-draw ranges, not confidence intervals.`, narrow);
  }

  window.ImbalanceScenes = window.ImbalanceScenes || {};
  window.ImbalanceScenes.sampling = progress => {
    const stage = Math.round(clamp(Number(progress) || 0, 0, 3)), data = window.LogisticSamplingRuns, narrow = window.innerWidth <= 650;
    if (!data?.cases?.correct || !data?.cases?.curved) return {stage, eyebrow: '03 / Actual random samples', title: 'Fit the models to actual random samples.', description: 'The numerical run results are loading.', svg: svg(text(narrow ? 190 : 450, 210, 'Loading numerical results…', C.muted, 20, 'middle'), 'Loading numerical sampling results.', narrow), equation: '', note: ''};
    const correct = data.cases.correct, curved = data.cases.curved;
    const titles = ['Now run logistic regression.', 'Correct model: similar slopes.', 'Curved truth: a different slope.', 'Repeat the downsampling.'];
    const descriptions = [
      'Fit each training dataset. Keep all positives, randomly retain 10% of negatives, then refit.',
      'The slopes differ slightly. Intercept correction returns the sampled line close to the full-data fit.',
      'Fit a straight line to curved log odds. Correcting its intercept leaves the sampled slope changed.',
      'Repeat on the same full data. Slope changes center near zero for linear truth and stay negative for curved truth.'
    ];
    const equations = [
      `All positives + a random ${(100 * data.protocol.retention).toFixed(0)}% of negatives → refit logistic regression`,
      `Intercept correction: add log(0.10)`,
      'Key takeaway: the original class mix wins here on log loss.',
      `Independent downsampling draws: ${number(data.protocol.repetitions)} per fixed full dataset`
    ];
    const notes = [
      'Each example has its own training and test data. The first retained draw is the one used on the next two slides.',
      'Finite samples fluctuate. Correction changes the intercept and keeps the sampled slope.',
      'Even with misspecification, fitting the original imbalanced data has lower test loss here. Other goals, such as recall, can favor rebalancing.',
      `Middle 95% of ${number(data.protocol.repetitions)} downsampling draws from each fixed full dataset; this is not a confidence interval.`
    ];
    return {stage, eyebrow: '03 / Actual random samples', title: titles[stage], description: descriptions[stage], svg: stage === 0 ? protocolFigure(data, narrow) : stage === 3 ? repeatFigure(data, narrow) : fitFigure(stage === 1 ? correct : curved, data, narrow), equation: equations[stage], note: notes[stage]};
  };

  const details = document.getElementById('sampling-run-details'), data = window.LogisticSamplingRuns;
  if (details && data) {
    details.innerHTML = ['correct', 'curved'].map(key => {
      const example = data.cases[key], delta = example.summary.slopeDelta;
      const fits = [['Full data', example.full], ['First downsample', example.draw], ['First downsample, corrected', example.corrected]];
      return `<h3>${key === 'correct' ? 'Correctly specified example' : 'Curved example'}</h3><div class="table-scroll"><table class="comparison"><caption>Actual fits, evaluated on the same untouched test sample</caption><thead><tr><th scope="col">Fit</th><th scope="col">Intercept</th><th scope="col">Slope</th><th scope="col">Test log loss</th></tr></thead><tbody>${fits.map(([name, entry]) => `<tr><th scope="row">${name}</th><td>${fixed(entry.fit.intercept, 6)}</td><td>${fixed(entry.fit.slope, 6)}</td><td>${fixed(entry.testLogLoss, 8)}</td></tr>`).join('')}</tbody></table></div><div class="table-scroll"><table class="comparison"><caption>Realized counts in the first retained draw</caption><thead><tr><th scope="col">x</th><th scope="col">Full positives</th><th scope="col">Full negatives</th><th scope="col">Retained positives</th><th scope="col">Retained negatives</th></tr></thead><tbody>${example.full.counts.map((row, i) => `<tr><th scope="row">${row.x}</th><td>${number(row.positive)}</td><td>${number(row.negative)}</td><td>${number(example.draw.counts[i].positive)}</td><td>${number(example.draw.counts[i].negative)}</td></tr>`).join('')}</tbody></table></div><p>Over ${data.protocol.repetitions} further downsampling draws, the mean sampled-minus-full slope change is ${signed(delta.mean)}. The middle 95% range is ${signed(delta.q025)} to ${signed(delta.q975)}, conditional on this fixed training dataset.</p>`;
    }).join('');
  }
})();
