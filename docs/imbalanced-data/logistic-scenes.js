(() => {
  'use strict';

  const C = {blue: '#234e70', orange: '#a54f32', teal: '#007f73', ink: '#242424', muted: '#606060', grid: '#e4e7e9', removed: '#edf0f2'};
  const L = Math.log(99), SHIFT = Math.log(10);
  const groups = [{x: 0, positive: 100, negative: 9900}, {x: 1, positive: 5000, negative: 5000}];
  const clamp = (value, low = 0, high = 1) => Math.max(low, Math.min(high, value));
  const ease = value => { const t = clamp(value); return t * t * (3 - 2 * t); };
  const safe = value => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const count = value => value.toLocaleString('en-US', {maximumFractionDigits: 1});
  const signed = value => `${value < -0.00001 ? '−' : value > 0.00001 ? '+' : ''}${Math.abs(value).toFixed(2)}`;
  const text = (x, y, value, color = C.ink, size = 18, anchor = 'start', extra = '') => `<text x="${x}" y="${y}" fill="${color}" text-anchor="${anchor}" style="font-size:${size}px;font-variant-numeric:tabular-nums" ${extra}>${safe(value)}</text>`;
  const line = (x1, y1, x2, y2, color = C.grid, extra = '') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" ${extra}/>`;
  const path = (points, color, extra = '') => `<path d="${points.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(3)},${p[1].toFixed(3)}`).join(' ')}" fill="none" stroke="${color}" stroke-width="3.5" ${extra}/>`;
  const dot = (x, y, color, size = 5.5, extra = '') => `<circle cx="${x}" cy="${y}" r="${size}" fill="${color}" stroke="white" stroke-width="1.6" ${extra}/>`;

  const story = [
    ['The true log odds are linear.', 'Positive shares of 1% and 50% put these two groups on the straight line.'],
    ['Every point is a group of cases.', 'At each x, the positive-to-negative ratio gives the odds. Its logarithm puts the group on the line.'],
    ['Sample both classes equally.', 'Both expected counts shrink together. The ratio and true log odds stay unchanged.'],
    ['Thin only the negatives.', 'Start again. Keep every positive and retain 10% of negatives at random.'],
    ['Every point rises by 2.30.', 'The odds multiply by ten. Every logit rises by log(10) ≈ 2.30; the slope stays the same.'],
    ['Prior correction shifts it back.', 'Subtract 2.30 to restore the population line: the prior correction in King and Zeng’s rare-events work.']
  ];

  function state(progress) {
    const p = clamp(Number(progress) || 0, 0, 5);
    const stage = Math.round(p);
    // Every integer stage shows its completed action, including with motion off.
    // Changes begin only after the corresponding stage's copy has appeared.
    let positiveShare = 1, negativeShare = 1;
    if (stage === 2) {
      const sample = p <= 2 ? ease((p - 1.5) / .5) : 1 - ease((p - 2) / .5);
      positiveShare = negativeShare = 1 - .5 * sample;
    } else if (stage >= 3) {
      negativeShare = Math.exp(-SHIFT * ease((p - 2.5) / .5));
    }
    const offset = Math.log(positiveShare / negativeShare);
    const corrected = stage === 5 ? ease((p - 4.5) / .5) : 0;
    return {p, stage, positiveShare, negativeShare, offset, corrected, groupsReveal: ease(p / .9)};
  }

  function mosaic(x, y, width, height, positive, negative) {
    const posWidth = width * positive / 10000;
    const negWidth = width * negative / 10000;
    let body = `<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="2" fill="${C.removed}"/>`;
    body += `<rect x="${x}" y="${y}" width="${posWidth}" height="${height}" fill="${C.orange}"/>`;
    body += `<rect x="${x + posWidth}" y="${y}" width="${negWidth}" height="${height}" fill="${C.blue}"/>`;
    for (let i = 1; i < 10; i++) body += line(x + width * i / 10, y, x + width * i / 10, y + height, 'white', 'stroke-width="1.4"');
    return body;
  }

  function figure(s, narrow) {
    const width = narrow ? 380 : 900, height = narrow ? 380 : 438;
    const left = narrow ? 43 : 92, right = narrow ? 350 : 842;
    const top = narrow ? 53 : 44, bottom = narrow ? 195 : 258;
    const fx = x => narrow ? 100 + x * 180 : 200 + x * 500;
    const fy = value => bottom - (value + 5.5) / 9 * (bottom - top);
    const cardTop = narrow ? 242 : 322, cardWidth = narrow ? 162 : 350;
    const sampleVisible = s.stage >= 3, correcting = s.stage === 5;
    const currentOffset = s.offset * (1 - s.corrected);
    const font = narrow ? 13 : 17;
    let body = text(left, narrow ? 17 : 23, 'Log odds', C.muted, narrow ? 15 : 19);

    const legend = (x, y, label, color, opacity = 1, dash = '') => `<g opacity="${opacity}">${line(x, y - 5, x + (narrow ? 16 : 25), y - 5, color, `stroke-width="3" ${dash}`)}${text(x + (narrow ? 21 : 32), y, label, color, narrow ? 12 : 17)}</g>`;
    if (narrow) {
      body += legend(44, 38, 'Population', C.blue);
      if (sampleVisible) body += legend(159, 38, 'Sampled', C.orange, correcting ? .5 : 1);
      if (correcting) body += legend(267, 38, 'Corrected', C.teal, 1, 'stroke-dasharray="5 3"');
    } else {
      body += legend(sampleVisible ? 450 : 690, 23, 'Population', C.blue);
      if (sampleVisible) body += legend(595, 23, 'Sampled', C.orange, correcting ? .5 : 1);
      if (correcting) body += legend(725, 23, 'Corrected', C.teal, 1, 'stroke-dasharray="5 3"');
    }

    [-4, -2, 0, 2].forEach(value => {
      body += line(left, fy(value), right, fy(value), value === 0 ? '#ccd4d9' : C.grid);
      body += text(left - (narrow ? 10 : 15), fy(value) + 5, value < 0 ? `−${-value}` : value, C.muted, font, 'end');
    });
    body += line(left, top, left, bottom, '#aeb9c1') + line(left, bottom, right, bottom, '#aeb9c1');
    groups.forEach(group => {
      const x = fx(group.x), original = L * (group.x - 1), sample = original + s.offset;
      if (s.groupsReveal > 0) {
        const connectorTop = fy(sampleVisible ? sample : original) + 8;
        const connectorBottom = bottom + (narrow ? 8 : 9);
        if (connectorTop < connectorBottom) body += line(x, connectorTop, x, connectorBottom, '#cdd7de', `stroke-dasharray="3 4" opacity="${s.groupsReveal}"`);
        body += line(x, narrow ? 222 : 296, x, cardTop, '#cdd7de', `stroke-dasharray="3 4" opacity="${s.groupsReveal}"`);
      }
      body += text(x, bottom + (narrow ? 22 : 26), `x = ${group.x}`, C.muted, narrow ? 14 : 18, 'middle');
    });

    body += path([[fx(0), fy(-L)], [fx(1), fy(0)]], C.blue);
    if (sampleVisible) body += path([[fx(0), fy(-L + s.offset)], [fx(1), fy(s.offset)]], C.orange, correcting ? 'opacity=".35"' : '');
    if (correcting) body += path([[fx(0), fy(-L + currentOffset)], [fx(1), fy(currentOffset)]], C.teal, 'stroke-dasharray="8 4"');

    groups.forEach(group => {
      const x = fx(group.x), value = L * (group.x - 1);
      body += dot(x, fy(value), C.blue);
      if (sampleVisible) body += dot(x, fy(value + s.offset), C.orange, 5.5, correcting ? 'opacity=".45"' : '');
      if (correcting) body += dot(x, fy(value + currentOffset), C.teal, 6);
    });

    if (sampleVisible && s.offset > .005) {
      const positions = s.stage === 4 ? [0, 1] : [1];
      positions.forEach(xValue => {
        const x = fx(xValue) + (narrow ? 10 : 15), original = L * (xValue - 1);
        const y1 = fy(original + s.offset), y2 = fy(original + (correcting ? currentOffset : 0));
        const color = correcting ? C.teal : C.orange;
        body += line(x, y1, x, y2, color, 'stroke-width="1.8"');
        body += line(x - 4, y1, x + 4, y1, color, 'stroke-width="1.8"');
        body += line(x - 4, y2, x + 4, y2, color, 'stroke-width="1.8"');
        if (xValue === 1) body += text(x + (narrow ? 8 : 12), (y1 + y2) / 2 + 5, correcting ? `−${(s.offset * s.corrected).toFixed(2)}` : `+${s.offset.toFixed(2)}`, color, narrow ? 14 : 18);
      });
    }
    if (s.stage === 4) body += text(right, bottom - 10, 'Same slope: 4.60', C.muted, narrow ? 12 : 18, 'end');

    // Equal-size insets compare conditional class ratios, not predictor frequencies.
    const caption = s.stage >= 2 ? 'Expected retained counts · pale tiles = removed' : '10,000 cases at each x · illustrative groups';
    body += `<g opacity="${s.groupsReveal}">`;
    body += text(width / 2, narrow ? 234 : 311, caption, C.muted, narrow ? 12 : 16, 'middle');
    groups.forEach(group => {
      const x = fx(group.x) - cardWidth / 2, positive = group.positive * s.positiveShare, negative = group.negative * s.negativeShare;
      const probability = positive / (positive + negative), logOdds = Math.log(positive / negative);
      body += `<rect x="${x}" y="${cardTop}" width="${cardWidth}" height="${narrow ? 113 : 103}" rx="5" fill="white" stroke="#d8dfe3"/>`;
      if (narrow) {
        body += text(x + cardWidth / 2, cardTop + 23, `${sampleVisible ? "q" : "p"} = ${(100 * probability).toFixed(1)}%`, C.muted, 14, 'middle');
        body += mosaic(x + 9, cardTop + 34, cardWidth - 18, 12, positive, negative);
        body += text(x + 9, cardTop + 66, `${count(positive)} pos.`, C.orange, 14);
        body += text(x + 9, cardTop + 87, `${count(negative)} neg.`, C.blue, 14);
        body += text(x + cardWidth / 2, cardTop + 107, `log odds ${signed(logOdds)}`, C.muted, 11.7, 'middle');
      } else {
        body += text(x + 14, cardTop + 23, `${sampleVisible ? "q" : "p"} = ${(100 * probability).toFixed(1)}%`, C.ink, 18);
        body += mosaic(x + 14, cardTop + 34, cardWidth - 28, 16, positive, negative);
        body += text(x + 14, cardTop + 74, `${count(positive)} positive`, C.orange, 15.5);
        body += text(x + cardWidth - 14, cardTop + 74, `${count(negative)} negative`, C.blue, 15.5, 'end');
        body += text(x + cardWidth / 2, cardTop + 95, `log odds ${signed(logOdds)}`, C.muted, 16, 'middle');
      }
    });
    body += '</g>';
    if (s.groupsReveal < 1) {
      body += `<g opacity="${1 - s.groupsReveal}">`;
      body += text(width / 2, narrow ? 283 : 351, 'The straight line is the data-generating truth.', C.ink, narrow ? 15 : 21, 'middle');
      body += text(width / 2, narrow ? 314 : 386, 'Most population cases have x = 0.', C.muted, narrow ? 14 : 18, 'middle');
      body += '</g>';
    }

    const label = `True linear log odds. ${story[s.stage][0]} ${groups.map(group => {
      const pos = group.positive * s.positiveShare, neg = group.negative * s.negativeShare;
      return `At x ${group.x}: ${count(pos)} expected positives, ${count(neg)} expected negatives; log odds ${signed(Math.log(pos / neg))}.`;
    }).join(' ')}${correcting ? ' The corrected line returns to the population line.' : ''}`;
    return `<svg class="scene-chart logistic-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${safe(label)}" xmlns="http://www.w3.org/2000/svg"><g font-family="LM, Georgia, serif">${body}</g></svg>`;
  }

  function logistic(progress) {
    const s = state(progress), stage = s.stage;
    let equation = 'logit p(x) = log(99) · (x − 1)';
    let note = 'Exact population relationship. Most population cases have x = 0.';
    if (stage === 1) {
      equation = 'log odds = log(positive count / negative count)';
      note = 'Equal inset sizes compare ratios. Most population cases have x = 0.';
    } else if (stage === 2) {
      equation = `Both classes retained at ${(100 * s.positiveShare).toFixed(0)}% → expected log odds unchanged`;
      note = 'Expected counts are shown. Finite random samples fluctuate around this unchanged relationship.';
    } else if (stage === 3) {
      equation = `logit q(x) = logit p(x) − log(s); s = ${(100 * s.negativeShare).toFixed(1)}%`;
      note = 'The same retention rate applies at every x. Counts show expectations.';
    } else if (stage === 4) {
      equation = 'logit q(x) = log(99) · (x − 1) + log(10)';
      note = 'Orange shows the sampled line and sampled-group ratios. A constant shift preserves the slope and ranking.';
    } else if (stage === 5) {
      equation = s.corrected >= .999 ? 'Population log odds = sampled log odds + log(0.10)' : `Apply prior correction: sampled log odds − ${(s.offset * s.corrected).toFixed(2)}`;
      note = '<a href="https://gking.harvard.edu/files/0s.pdf">King &amp; Zeng (2001), §4.1</a> · Sampling correction and finite-sample bias correction are separate.';
    }
    return {stage, eyebrow: '01 / Logistic regression', title: story[stage][0], description: story[stage][1], svg: figure(s, window.innerWidth <= 650), equation, note};
  }

  window.ImbalanceScenes = window.ImbalanceScenes || {};
  window.ImbalanceScenes.logistic = logistic;
})();
