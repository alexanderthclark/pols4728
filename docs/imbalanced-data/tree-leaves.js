(() => {
  'use strict';

  const C = window.TreeColors;
  const safe = value => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const number = (value, precision = 1) => Number(value).toLocaleString('en-US', {maximumFractionDigits: precision});
  const percent = value => {
    const pct = 100 * value;
    if (pct > 0 && pct < .001) return '<0.001%';
    return `${pct.toFixed(3).replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '')}%`;
  };
  const text = (x, y, value, color = C.ink, size = 18, anchor = 'start') => `<text x="${x}" y="${y}" fill="${color}" text-anchor="${anchor}" style="font-size:${size}px;font-variant-numeric:tabular-nums">${safe(value)}</text>`;
  const line = (x1, y1, x2, y2, color = C.border, extra = '') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" ${extra}/>`;
  const rect = (x, y, width, height, fill, extra = '') => `<rect x="${x}" y="${y}" width="${Math.max(0, width)}" height="${Math.max(0, height)}" fill="${fill}" ${extra}/>`;

  function counts(p) {
    const model = window.TreeBalance;
    const prevalence = Number(p);
    // Use the same validated prevalence and fixed routing rules as the gain plots.
    model.evaluate('A', prevalence);
    const positive = 100, negative = positive * (1 - prevalence) / prevalence;
    const parent = {positive, negative, total: positive + negative, positiveShare: prevalence};
    const child = (positives, negatives) => ({positive: positives, negative: negatives, total: positives + negatives, positiveShare: positives / (positives + negatives)});
    return Object.fromEntries(Object.entries(model.splits).map(([name, split]) => {
      const leftPositive = positive * split.positiveLeft, leftNegative = negative * split.negativeLeft;
      return [name, {name, parent: {...parent}, left: child(leftPositive, leftNegative), right: child(positive - leftPositive, negative - leftNegative)}];
    }));
  }

  function compositionBar(x, y, width, height, share) {
    return rect(x, y, width, height, C.negative, 'rx="2"') + rect(x, y, width * share, height, C.positive);
  }

  function leafCard(x, y, width, leaf, side, narrow) {
    const height = narrow ? 98 : 174;
    const center = x + width / 2, inset = narrow ? 8 : 13;
    let body = rect(x, y, width, height, 'white', `stroke="${C.border}" rx="4"`);
    if (narrow) {
      body += text(center, y + 16, side === 'left' ? 'Left' : 'Right', C.ink, 15, 'middle');
      if (side === 'right') body += text(center, y + 32, 'everyone else', C.muted, 14, 'middle');
      body += text(center, y + 55, percent(leaf.positiveShare), C.positive, 23, 'middle');
      body += text(center, y + 69, 'positive', C.muted, 14, 'middle');
      body += compositionBar(x + inset, y + 76, width - 2 * inset, 8, leaf.positiveShare);
      body += text(x + inset, y + 96, `${number(leaf.positive)} +`, C.positive, 15);
      body += text(x + width - inset, y + 96, `${number(leaf.negative)} −`, C.negative, 15, 'end');
    } else {
      body += text(center, y + 24, side === 'left' ? 'Left' : 'Right · everyone else', C.ink, side === 'left' ? 20 : 17, 'middle');
      body += text(center, y + 60, percent(leaf.positiveShare), C.positive, 31, 'middle');
      body += text(center, y + 82, 'positive', C.muted, 17, 'middle');
      body += compositionBar(x + inset, y + 95, width - 2 * inset, 16, leaf.positiveShare);
      body += text(center, y + 138, `${number(leaf.positive)} positive`, C.positive, 18, 'middle');
      body += text(center, y + 164, `${number(leaf.negative)} negative`, C.negative, 18, 'middle');
    }
    return body;
  }

  function figure(p, narrow = false) {
    const data = counts(p), model = window.TreeBalance, W = narrow ? 380 : 900, H = 400;
    const parent = data.A.parent;
    let body = text(W / 2, narrow ? 20 : 24, narrow
      ? `${number(parent.positive)} positive · ${number(parent.negative)} negative (${percent(parent.positiveShare)})`
      : `Parent: ${number(parent.positive)} positives · ${number(parent.negative)} negatives · ${percent(parent.positiveShare)} positive`, C.ink, narrow ? 15.5 : 20, 'middle');

    ['A', 'B'].forEach((name, i) => {
      const split = model.splits[name], result = data[name], candidateColor = C[name];
      const panelX = narrow ? 12 : 16 + i * 444, panelY = narrow ? 31 + i * 180 : 43;
      const panelWidth = narrow ? 356 : 424, panelHeight = narrow ? 167 : 335;
      const center = panelX + panelWidth / 2;
      body += rect(panelX, panelY, panelWidth, panelHeight, C.panels[name], `stroke="${candidateColor}" stroke-width="1.5" class="tree-candidate-panel" data-split="${name}" rx="5"`);
      const leftRate = percent(split.positiveLeft), negativeRate = percent(split.negativeLeft);
      if (narrow) {
        body += text(center, panelY + 20, `Candidate ${name} · split on X_${name}`, candidateColor, 18, 'middle');
        body += text(center, panelY + 45, `Left: ${leftRate} of all +; ${negativeRate} of all −`, C.muted, 14, 'middle');
        const cardY = panelY + 66, cardWidth = 163, leftX = panelX + 9, rightX = panelX + 184;
        body += line(center, panelY + 52, leftX + cardWidth / 2, cardY, candidateColor, 'stroke-width="1.3" stroke-opacity=".65"');
        body += line(center, panelY + 52, rightX + cardWidth / 2, cardY, candidateColor, 'stroke-width="1.3" stroke-opacity=".65"');
        body += leafCard(leftX, cardY, cardWidth, result.left, 'left', true);
        body += leafCard(rightX, cardY, cardWidth, result.right, 'right', true);
      } else {
        body += text(center, panelY + 28, `Candidate ${name}`, candidateColor, 24, 'middle');
        body += rect(center - 78, panelY + 41, 156, 32, 'white', `stroke="${candidateColor}" rx="3"`);
        body += text(center, panelY + 64, `Split on X_${name}`, candidateColor, 20, 'middle');
        body += text(center, panelY + 99, `Left gets ${leftRate} of all positives; ${negativeRate} of all negatives`, C.muted, 15.5, 'middle');
        const cardY = panelY + 141, cardWidth = 190, leftX = panelX + 14, rightX = panelX + 220;
        body += line(center, panelY + 111, leftX + cardWidth / 2, cardY, candidateColor, 'stroke-width="1.5" stroke-opacity=".65"');
        body += line(center, panelY + 111, rightX + cardWidth / 2, cardY, candidateColor, 'stroke-width="1.5" stroke-opacity=".65"');
        body += leafCard(leftX, cardY, cardWidth, result.left, 'left', false);
        body += leafCard(rightX, cardY, cardWidth, result.right, 'right', false);
      }
    });
    const legendY = narrow ? 397 : 395;
    const positiveX = narrow ? 20 : 255, negativeX = narrow ? 119 : 423, noteX = narrow ? 228 : 601;
    body += rect(positiveX, legendY - 10, narrow ? 8 : 12, narrow ? 8 : 12, C.positive);
    body += text(positiveX + (narrow ? 13 : 19), legendY, narrow ? '+ positive' : 'Positive cases', C.positive, narrow ? 13.5 : 16);
    body += rect(negativeX, legendY - 10, narrow ? 8 : 12, narrow ? 8 : 12, C.negative);
    body += text(negativeX + (narrow ? 13 : 19), legendY, narrow ? '− negative' : 'Negative cases', C.negative, narrow ? 13.5 : 16);
    body += text(noteX, legendY, 'expected counts', C.muted, narrow ? 13.5 : 16);

    const label = `Parent contains ${number(parent.positive, 3)} positives and ${number(parent.negative, 3)} negatives, ${percent(parent.positiveShare)} positive. ${['A', 'B'].map(name => {
      const result = data[name], split = model.splits[name];
      return `Candidate ${name} splits on X_${name}. Its fixed left routing rates are ${percent(split.positiveLeft)} of all positives and ${percent(split.negativeLeft)} of all negatives. Left leaf: ${number(result.left.positive, 3)} positives and ${number(result.left.negative, 3)} negatives, ${percent(result.left.positiveShare)} positive. Right leaf, everyone else: ${number(result.right.positive, 3)} positives and ${number(result.right.negative, 3)} negatives, ${percent(result.right.positiveShare)} positive.`;
    }).join(' ')} Counts are expected; displayed intermediate counts are rounded.`;
    return `<svg class="tree-chart tree-leaves-chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${safe(label)}" xmlns="http://www.w3.org/2000/svg"><g font-family="LM, Georgia, serif">${body}</g></svg>`;
  }

  window.TreeLeaves = {figure, counts};
})();
