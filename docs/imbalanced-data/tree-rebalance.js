(() => {
  'use strict';
  const safe = value => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const number = value => Number(value).toLocaleString('en-US', {maximumFractionDigits: 1});
  const percent = value => `${(100 * value).toFixed(3).replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '')}%`;

  function figure(retention, narrow = false) {
    const C = window.TreeColors, raw = Number(retention);
    if (!Number.isFinite(raw)) throw new RangeError('Negative retention must be a finite number.');
    const s = Math.max(1 / 99, Math.min(1, raw));
    const positives = 100, originalNegatives = 9900, negatives = originalNegatives * s;
    const removed = originalNegatives - negatives, p = positives / (positives + negatives);
    const W = narrow ? 380 : 900, H = narrow ? 440 : 430, bannerHeight = narrow ? 70 : 68;
    const text = (x, y, label, color = C.ink, size = 18, anchor = 'start') => `<text x="${x}" y="${y}" fill="${color}" text-anchor="${anchor}" style="font-size:${size}px;font-variant-numeric:tabular-nums">${safe(label)}</text>`;
    const rect = (x, y, width, height, color, extra = '') => `<rect x="${x}" y="${y}" width="${Math.max(0, width)}" height="${height}" fill="${color}" ${extra}/>`;

    let banner = text(narrow ? 12 : 16, narrow ? 20 : 24, 'All 100 positives kept', C.positive, narrow ? 17 : 22);
    banner += text(W - (narrow ? 12 : 16), narrow ? 20 : 24, `Parent ${percent(p)} positive`, C.ink, narrow ? 14 : 19, 'end');
    banner += text(narrow ? 12 : 16, narrow ? 40 : 48, narrow
      ? `Keep ${number(negatives)} (${percent(s)})`
      : `Keep ${number(negatives)} / 9,900 negatives (${percent(s)})`, C.negative, narrow ? 14 : 18);
    banner += text(W - (narrow ? 12 : 16), narrow ? 40 : 48, narrow
      ? `Remove ${number(removed)} (${percent(1 - s)})`
      : `Remove ${number(removed)} / 9,900 negatives (${percent(1 - s)})`, C.muted, narrow ? 14 : 18, 'end');

    const poolX = narrow ? 12 : 16, poolY = narrow ? 47 : 55;
    const poolWidth = W - poolX * 2, poolHeight = narrow ? 11 : 10, retainedWidth = poolWidth * s;
    banner += rect(poolX, poolY, retainedWidth, poolHeight, C.negative, `class="tree-negative-pool-retained" data-count="${negatives}" data-share="${s}"`);
    banner += rect(poolX + retainedWidth, poolY, poolWidth - retainedWidth, poolHeight, '#edf0f2', `class="tree-negative-pool-removed" data-count="${removed}" data-share="${1 - s}"`);
    for (let i = 1; i < 20; i++) {
      const x = poolX + poolWidth * i / 20;
      banner += `<line x1="${x}" y1="${poolY}" x2="${x}" y2="${poolY + poolHeight}" stroke="white" stroke-width="1"/>`;
    }
    if (narrow) banner += text(W / 2, 66, 'Original pool: 9,900 negatives · pale = removed', C.muted, 12.5, 'middle');

    // Reuse the live leaf renderer, replacing its duplicated parent caption with
    // the sampling banner. Candidate panels, routing rules, and legends stay intact.
    const leafSvg = window.TreeLeaves.figure(p, narrow);
    const leafAria = leafSvg.match(/aria-label="([^"]*)"/)?.[1] || '';
    const leaves = leafSvg.replace(/^<svg\b[^>]*>/, '').replace(/<\/svg>\s*$/, '').replace(/<text\b[^>]*>[\s\S]*?<\/text>/, '');
    const topInset = narrow ? 31 : 43, shift = bannerHeight - topInset;
    const narration = `Keep all ${positives} positives. Randomly retain ${percent(s)} of the original ${originalNegatives} negatives: ${number(negatives)} retained and ${number(removed)} removed, or ${percent(1 - s)} removed. The parent is now ${percent(p)} positive. The blue negative-pool segment shrinks while the pale removed segment grows. Counts show expectations; sampling is random within the negative class and both candidate routing rules stay fixed.`;
    return `<svg class="tree-chart tree-rebalance-chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${safe(narration)} ${leafAria}" data-retention="${s}" data-positive-count="${positives}" data-negative-count="${negatives}" data-removed-negative-count="${removed}" data-parent-positive-share="${p}" xmlns="http://www.w3.org/2000/svg"><g font-family="LM, Georgia, serif">${banner}<g transform="translate(0,${shift})">${leaves}</g></g></svg>`;
  }

  window.TreeRebalance = {figure};
})();
