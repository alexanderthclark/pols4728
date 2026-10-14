(() => {
  'use strict';
  const C = window.TreeColors;
  const safe = value => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const pct = value => `${(100 * value).toFixed(3).replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '')}%`;
  const value = n => n.toFixed(6);
  const text = (x, y, label, color = C.ink, size = 18, anchor = 'start') => `<text x="${x}" y="${y}" fill="${color}" text-anchor="${anchor}" style="font-size:${size}px;font-variant-numeric:tabular-nums">${safe(label)}</text>`;
  const line = (x1, y1, x2, y2, color = C.grid, extra = '') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" ${extra}/>`;
  const dot = (x, y, color, radius = 4.5, extra = '') => `<circle cx="${x}" cy="${y}" r="${radius}" fill="${color}" stroke="white" stroke-width="1.5" ${extra}/>`;
  const path = (points, color, width = 3, extra = '') => `<path d="${points.map((point, i) => `${i ? 'L' : 'M'}${point[0].toFixed(4)},${point[1].toFixed(4)}`).join(' ')}" fill="none" stroke="${color}" stroke-width="${width}" ${extra}/>`;

  function globalPlot(p, results, model, narrow) {
    const left = narrow ? 43 : 70, right = narrow ? 351 : 540;
    const top = narrow ? 52 : 60, bottom = narrow ? 212 : 289;
    const maximum = Math.log(2), fx = q => left + (right - left) * q, fy = h => bottom - (bottom - top) * h / maximum;
    const parentH = model.entropy(p);
    let body = text(left, narrow ? 20 : 27, 'Entropy H(p)', C.ink, narrow ? 17 : 21);
    body += text(right, narrow ? 20 : 27, `Parent p = ${pct(p)}`, C.muted, narrow ? 14 : 18, 'end');
    [0, .35, maximum].forEach(h => {
      body += line(left, fy(h), right, fy(h));
      body += text(left - 9, fy(h) + 5, h === maximum ? 'log 2' : h === 0 ? '0' : '0.35', C.muted, narrow ? 12.5 : 16, 'end');
    });
    body += line(left, top, left, bottom, '#aeb9c1') + line(left, bottom, right, bottom, '#aeb9c1');
    const arc = Array.from({length: 241}, (_, i) => { const q = i / 240; return [fx(q), fy(model.entropy(q))]; });
    body += path(arc, C.arc, 2.5, 'class="entropy-arc"');

    ['A', 'B'].forEach(name => {
      const result = results[name], color = C[name];
      const rightPoint = [fx(result.pRight), fy(model.entropy(result.pRight))];
      const leftPoint = [fx(result.pLeft), fy(model.entropy(result.pLeft))];
      const childH = result.leftContribution + result.rightContribution;
      body += path([rightPoint, leftPoint], color, 3, `class="entropy-chord" ${name === 'B' ? 'stroke-dasharray="7 4"' : ''} data-split="${name}" data-left-share="${result.pLeft}" data-right-share="${result.pRight}" data-child-entropy="${childH}" data-gain="${result.gain}"`);
      body += dot(...rightPoint, color, narrow ? 4 : 5, `class="entropy-endpoint" data-split="${name}" data-tree-child="R" data-node-p="${result.pRight}"`);
      body += dot(...leftPoint, color, narrow ? 4 : 5, `class="entropy-endpoint" data-split="${name}" data-tree-child="L" data-node-p="${result.pLeft}"`);
      body += dot(fx(p), fy(childH), color, narrow ? 3.5 : 4.5, `class="entropy-chord-parent" data-split="${name}" data-entropy="${childH}"`);

      const labelSize = narrow ? 13 : 16;
      body += text(leftPoint[0] - (name === 'A' ? 9 : -8), leftPoint[1] - (name === 'A' ? 12 : 11), `${name} L`, color, labelSize, name === 'A' ? 'end' : 'start');
      // At rare prevalence, separate the crowded right-child labels with leaders.
      if (p < .08) {
        const labelX = left + (narrow ? 28 : 36), labelY = rightPoint[1] + (name === 'A' ? -23 : 16);
        body += line(rightPoint[0] + 3, rightPoint[1], labelX - 5, labelY - 4, color, 'stroke-width="1"');
        body += text(labelX, labelY, `${name} R`, color, labelSize);
      } else {
        body += text(rightPoint[0] + 9, rightPoint[1] + (name === 'A' ? -13 : 20), `${name} R`, color, labelSize);
      }
    });
    const low = Math.min(...Object.values(results).map(result => result.leftContribution + result.rightContribution));
    body += line(fx(p), fy(parentH), fx(p), fy(low), C.ink, 'stroke-width="1.2" stroke-dasharray="2 3"');
    body += dot(fx(p), fy(parentH), C.ink, narrow ? 4.5 : 5.5, `class="entropy-parent" data-node-p="${p}" data-entropy="${parentH}"`);
    [0, .5, 1].forEach(q => { body += text(fx(q), bottom + (narrow ? 22 : 26), q === .5 ? '50%' : `${q * 100}%`, C.muted, narrow ? 14 : 18, 'middle'); });
    body += text((left + right) / 2, narrow ? 257 : 340, 'Positive share in a node', C.muted, narrow ? 14 : 18, 'middle');

    if (narrow) {
      body += text(left, 279, `A: L ${pct(results.A.pLeft)}; R ${pct(results.A.pRight)}`, C.A, 12.5);
      body += text(right, 279, `B: L ${pct(results.B.pLeft)}; R ${pct(results.B.pRight)}`, C.B, 12.5, 'end');
    } else {
      body += text(left, 369, `A leaves: L ${pct(results.A.pLeft)} · R ${pct(results.A.pRight)}`, C.A, 17);
      body += text(left, 392, `B leaves: L ${pct(results.B.pLeft)} · R ${pct(results.B.pRight)}`, C.B, 17);
    }
    return body;
  }

  function magnifiedSlice(p, results, model, narrow) {
    const parentH = model.entropy(p);
    if (narrow) {
      const left = 43, right = 351, fx = h => left + (right - left) * h / parentH;
      let body = text(left, 306, `H(parent) = ${value(parentH)}`, C.ink, 14);
      ['A', 'B'].forEach((name, i) => {
        const result = results[name], childH = result.leftContribution + result.rightContribution, color = C[name];
        const labelY = 328 + i * 50, barY = 343 + i * 50;
        body += text(left, labelY, `${name} chord ${value(childH)} · gain ${value(result.gain)}`, color, 14);
        body += line(left, barY, fx(childH), barY, color, 'stroke-width="8" stroke-opacity=".3"');
        body += line(fx(childH), barY, right, barY, color, `stroke-width="8" class="entropy-magnified-gap" data-split="${name}" data-gain="${result.gain}"`);
        body += dot(fx(childH), barY, color, 5);
        body += line(right, barY - 8, right, barY + 8, C.ink, 'stroke-width="1.5"');
      });
      body += text(190, 423, 'Darker segment = entropy decrease', C.muted, 12.4, 'middle');
      return body;
    }
    const left = 614, right = 862, top = 88, bottom = 289;
    const fy = h => bottom - (bottom - top) * h / parentH;
    let body = '';
    body += text(left, 52, `H(parent) = ${value(parentH)}`, C.muted, 18);
    body += line(left + 24, top, right, top, C.ink, 'stroke-width="1.5"');
    body += line(left + 24, bottom, right, bottom, C.grid);
    body += text(left + 14, bottom + 5, '0', C.muted, 16, 'end');
    ['A', 'B'].forEach((name, i) => {
      const result = results[name], color = C[name], childH = result.leftContribution + result.rightContribution, x = 638 + i * 120;
      const childY = fy(childH), centerY = (top + childY) / 2;
      body += line(x, bottom, x, childY, color, 'stroke-width="9" stroke-opacity=".3"');
      body += line(x, childY, x, top, color, `stroke-width="9" class="entropy-magnified-gap" data-split="${name}" data-gain="${result.gain}"`);
      body += dot(x, childY, color, 6);
      body += dot(x, top, C.ink, 4.5);
      body += text(x + 14, centerY - 8, `gain ${name}`, color, 17);
      body += text(x + 14, centerY + 15, value(result.gain), color, 20);
      body += text(x + 44, 318, `${name} chord height`, color, 17, 'middle');
      body += text(x + 44, 343, value(childH), color, 21, 'middle');
    });
    body += text((left + right) / 2, 375, 'Lower chord → larger gain', C.ink, 18, 'middle');
    return body;
  }

  function figure(prevalence, narrow = false) {
    const model = window.TreeBalance, p = Number(prevalence);
    const results = {A: model.evaluate('A', p), B: model.evaluate('B', p)}, parentH = model.entropy(p);
    const W = narrow ? 380 : 900, H = narrow ? 430 : 400;
    const label = `Binary entropy curve on the fixed range zero to one. Parent positive share ${pct(p)}, entropy ${value(parentH)}. ${['A', 'B'].map(name => {
      const result = results[name];
      return `Candidate ${name}: left tree child ${pct(result.pLeft)} positive, right tree child ${pct(result.pRight)} positive; weighted child entropy ${value(result.leftContribution + result.rightContribution)}; gain ${value(result.gain)}.`;
    }).join(' ')} Each chord joins its two child nodes on the entropy curve. At the parent probability, the chord height is the weighted child entropy. The difference from parent entropy is the gain. The magnified view rescales zero to the current parent entropy equally for both candidates. L and R identify tree children, not horizontal positions.`;
    return `<svg class="tree-chart tree-entropy-chords" viewBox="0 0 ${W} ${H}" role="img" aria-label="${safe(label)}" data-parent-p="${p}" data-parent-entropy="${parentH}" xmlns="http://www.w3.org/2000/svg"><g font-family="LM, Georgia, serif">${globalPlot(p, results, model, narrow)}${magnifiedSlice(p, results, model, narrow)}</g></svg>`;
  }

  window.TreeEntropyChords = {figure};
})();
