/* Fixed data, with the two models crossfading as their equations take focus. */
(function (root) {
  'use strict';

  const C = {
    ink: '#242424', blue: '#234E70', rust: '#A54F32', muted: '#606060',
    rule: '#767676', pale: '#DADADA', blueWash: '#EEF2F5', rustWash: '#F6ECE7'
  };
  const mathCache = new Map();
  const geometryCache = new WeakMap();
  const viewCache = new WeakMap();
  const clamp = x => Math.max(0, Math.min(1, Number.isFinite(x) ? x : 0));
  const esc = s => String(s).replace(/[&<>\"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
  const n = x => Number(x.toFixed(5));

  function math(tex) {
    if (!mathCache.has(tex)) mathCache.set(tex, root.katex.renderToString(tex, {
      output: 'htmlAndMathml', displayMode: false, throwOnError: true,
      strict: 'error', trust: false
    }));
    return mathCache.get(tex);
  }

  function mathLabel(x, y, tex, width, font) {
    const height = font * 2.8;
    return '<foreignObject class="nc-math" data-tex="' + esc(tex) + '" x="' + n(x - width / 2) + '" y="' + n(y - height / 2) + '" width="' + width + '" height="' + height + '" overflow="visible">' +
      '<div xmlns="http://www.w3.org/1999/xhtml" style="height:100%;display:flex;align-items:center;justify-content:center;color:inherit;font-size:' + font + 'px;white-space:nowrap">' + math(tex) + '</div></foreignObject>';
  }

  // Intersections are roots of the actual score, not a fitted spline or a
  // threshold on interpolated probabilities. All geometry is computed once.
  function zero(score, a, b) {
    let lo = a, hi = b, flo = score(a[0], a[1]);
    const fhi = score(b[0], b[1]);
    if (Math.abs(flo) < 1e-12) return a;
    if (Math.abs(fhi) < 1e-12) return b;
    for (let i = 0; i < 34; i++) {
      const mid = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2];
      const fm = score(mid[0], mid[1]);
      if ((fm >= 0) === (flo >= 0)) { lo = mid; flo = fm; }
      else hi = mid;
    }
    return [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2];
  }

  function positiveAffinePolygon(score) {
    const rect = [[0, 0], [1, 0], [1, 1], [0, 1]], polygon = [];
    rect.forEach((a, i) => {
      const b = rect[(i + 1) % rect.length], fa = score(a[0], a[1]), fb = score(b[0], b[1]);
      if (fa >= 0) polygon.push(a);
      if ((fa >= 0) !== (fb >= 0)) polygon.push(zero(score, a, b));
    });
    return polygon;
  }

  function contours(score) {
    const steps = 96, vertices = [], roots = new Map(), nodes = new Map(), segments = [];
    for (let j = 0; j <= steps; j++) for (let i = 0; i <= steps; i++) {
      const point = [i / steps, j / steps];
      vertices.push({ id: j * (steps + 1) + i, point, value: score(point[0], point[1]) });
    }
    function intersection(a, b) {
      const edge = Math.min(a.id, b.id) + ':' + Math.max(a.id, b.id);
      if (!roots.has(edge)) {
        const point = zero(score, a.point, b.point);
        // A zero at a grid vertex must share one identity across all its edges.
        const key = point.map(v => v.toFixed(10)).join(',');
        if (!nodes.has(key)) nodes.set(key, { point, links: [] });
        roots.set(edge, key);
      }
      return roots.get(edge);
    }
    function triangle(vs) {
      const hits = [];
      vs.forEach((a, i) => {
        const b = vs[(i + 1) % 3];
        if ((a.value >= 0) !== (b.value >= 0)) hits.push(intersection(a, b));
      });
      if (hits.length === 2 && hits[0] !== hits[1]) {
        const edge = { a: hits[0], b: hits[1], used: false };
        nodes.get(edge.a).links.push(edge); nodes.get(edge.b).links.push(edge);
        segments.push(edge);
      }
    }
    for (let j = 0; j < steps; j++) for (let i = 0; i < steps; i++) {
      const k = j * (steps + 1) + i;
      const a = vertices[k], b = vertices[k + 1], c = vertices[k + steps + 2], d = vertices[k + steps + 1];
      triangle([a, b, c]); triangle([a, c, d]);
    }
    const paths = [];
    function trace(start) {
      let key = start;
      const points = [nodes.get(key).point];
      while (true) {
        const edge = nodes.get(key).links.find(e => !e.used);
        if (!edge) break;
        edge.used = true; key = edge.a === key ? edge.b : edge.a;
        points.push(nodes.get(key).point);
        if (key === start) break;
      }
      if (points.length > 1) paths.push({ points, closed: key === start });
    }
    nodes.forEach((node, key) => { if (node.links.length === 1 && !node.links[0].used) trace(key); });
    segments.forEach(edge => { if (!edge.used) trace(edge.a); });
    return paths;
  }

  function geometry(model) {
    if (!geometryCache.has(model)) {
      const samples = Array.from({ length: 301 }, (_, i) => i / 300);
      const output = {};
      ['logistic', 'network'].forEach(name => {
        const score = model[name + 'Score2d'];
        const boundaries = contours(score);
        output[name] = {
          curve: samples.map(x => [x, model[name + 'Probability1d'](x)]),
          boundaries,
          positive: name === 'logistic' ? [positiveAffinePolygon(score)] : boundaries.filter(p => p.closed).map(p => p.points)
        };
      });
      geometryCache.set(model, output);
    }
    return geometryCache.get(model);
  }

  function path(points, px, py, closed) {
    return points.map((point, i) => (i ? 'L' : 'M') + n(px(point[0])) + ',' + n(py(point[1]))).join(' ') + (closed ? ' Z' : '');
  }

  function marker(point, x, y, compact, attributes) {
    const radius = compact ? 5.7 : 4.7, color = point.y ? C.rust : C.blue;
    const shape = point.y
      ? '<path d="M0,' + (-radius * 1.2) + 'L' + (radius * 1.12) + ',' + radius + 'L' + (-radius * 1.12) + ',' + radius + 'Z"/>'
      : '<circle r="' + radius + '"/>';
    return '<g data-comparison-point-id="' + esc(point.id) + '" data-label="' + point.y + '" ' + attributes + ' transform="translate(' + n(x) + ' ' + n(y) + ')" fill="' + color + '" stroke="#fff" stroke-width="1.2">' + shape + '</g>';
  }

  function build(svg, kind, compact, model, allGeometry) {
    const height = compact ? 285 : 400;
    const box = { left: 66, right: 575, top: compact ? 47 : 53, bottom: compact ? 236 : 344 };
    const px = x => box.left + x * (box.right - box.left);
    const py = y => box.bottom - y * (box.bottom - box.top);
    const font = compact ? 23 : 18, tick = compact ? 20 : 13;
    const body = ['<style>.nc-math .katex{font-size:1em;line-height:1.1}.nc-tick{font-family:Arial,sans-serif;fill:' + C.muted + ';font-size:' + tick + 'px}</style>'];
    svg.setAttribute('viewBox', '0 0 620 ' + height);
    svg.setAttribute('role', 'img');
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    svg.setAttribute('data-comparison-view', kind);
    svg.removeAttribute('aria-hidden');
    svg.setAttribute('aria-label', kind === 'probability'
      ? 'The same binary observations and two probability models. Logistic regression has a monotone sigmoid curve; the neural network can rise and fall.'
      : 'The same two-dimensional binary observations and two classifiers. Logistic regression has a straight boundary; the sigmoid network encloses a rounded region.');

    if (kind === 'boundary') {
      ['logistic', 'network'].forEach(name => {
        const g = allGeometry[name];
        body.push('<g data-comparison-model="' + name + '" opacity="0">');
        body.push('<path data-comparison-region="' + name + '" data-predicted-class="0" d="' + path([[0, 0], [1, 0], [1, 1], [0, 1]], px, py, true) + '" fill="' + C.blueWash + '"/>');
        body.push('<path data-comparison-region="' + name + '" data-predicted-class="1" d="' + g.positive.map(points => path(points, px, py, true)).join(' ') + '" fill="' + C.rustWash + '"/>');
        body.push('<path data-comparison-boundary="' + name + '" data-score-level="0" data-boundary-coordinates="' + esc(JSON.stringify(g.boundaries)) + '" d="' + g.boundaries.map(p => path(p.points, px, py, p.closed)).join(' ') + '" fill="none" stroke="' + C.ink + '" stroke-width="' + (compact ? 3.6 : 2.8) + '" stroke-linejoin="round"/>');
        body.push('</g>');
      });
    }
    body.push('<g data-comparison-axes="true" stroke="' + C.rule + '" stroke-width="1.1"><line x1="' + box.left + '" y1="' + box.top + '" x2="' + box.left + '" y2="' + box.bottom + '"/><line x1="' + box.left + '" y1="' + box.bottom + '" x2="' + box.right + '" y2="' + box.bottom + '"/></g>');
    [0, 0.5, 1].forEach(value => {
      body.push('<line x1="' + n(px(value)) + '" y1="' + box.bottom + '" x2="' + n(px(value)) + '" y2="' + (box.bottom + 5) + '" stroke="' + C.rule + '"/><text class="nc-tick" x="' + n(px(value)) + '" y="' + (box.bottom + (compact ? 28 : 22)) + '" text-anchor="middle">' + value + '</text>');
      body.push('<line x1="' + (box.left - 5) + '" y1="' + n(py(value)) + '" x2="' + box.left + '" y2="' + n(py(value)) + '" stroke="' + C.rule + '"/><text class="nc-tick" x="' + (box.left - 9) + '" y="' + n(py(value) + (compact ? 7 : 4)) + '" text-anchor="end">' + value + '</text>');
    });
    body.push(mathLabel(box.right + 19, box.bottom + 1, kind === 'probability' ? 'x' : 'x_1', 50, font));
    body.push(mathLabel(kind === 'probability' ? box.left + 23 : box.left, box.top - 23, kind === 'probability' ? 'y,\\;\\hat p(x)' : 'x_2', kind === 'probability' ? 160 : 80, font));

    if (kind === 'probability') {
      body.push('<line data-probability-threshold="0.5" x1="' + box.left + '" y1="' + n(py(0.5)) + '" x2="' + box.right + '" y2="' + n(py(0.5)) + '" stroke="' + C.pale + '" stroke-width="1.2" stroke-dasharray="5 5"/>');
      ['logistic', 'network'].forEach(name => {
        body.push('<g data-comparison-model="' + name + '" opacity="0"><path data-comparison-curve="' + name + '" data-sample-count="301" d="' + path(allGeometry[name].curve, px, py, false) + '" fill="none" stroke="' + C.ink + '" stroke-width="' + (compact ? 4 : 3.2) + '" stroke-linecap="round" stroke-linejoin="round"/></g>');
      });
    }
    body.push('<g data-comparison-points="true">');
    if (kind === 'probability') model.points1d.forEach(point => body.push(marker(point, px(point.x), py(point.y), compact, 'data-x="' + point.x + '" data-y="' + point.y + '"')));
    else model.points2d.forEach(point => body.push(marker(point, px(point.x1), py(point.x2), compact, 'data-x1="' + point.x1 + '" data-x2="' + point.x2 + '"')));
    body.push('</g>');
    svg.innerHTML = body.join('');
    return { compact, kind, model, layers: Array.from(svg.querySelectorAll('[data-comparison-model]')) };
  }

  function renderView(svg, kind, compact, model, allGeometry, mix, opacity) {
    if (!svg) return;
    let cached = viewCache.get(svg);
    // Other lesson scenes may reuse the SVG; re-entering must rebuild it.
    if (!cached || cached.compact !== compact || cached.kind !== kind || cached.model !== model || !svg.querySelector('[data-comparison-points]')) {
      cached = build(svg, kind, compact, model, allGeometry);
      viewCache.set(svg, cached);
    }
    svg.setAttribute('data-comparison-mix', mix);
    svg.setAttribute('data-comparison-plot-opacity', opacity);
    cached.layers.forEach(layer => {
      const value = layer.getAttribute('data-comparison-model') === 'network' ? mix : 1 - mix;
      layer.setAttribute('opacity', value);
      layer.setAttribute('aria-hidden', value === 0 ? 'true' : 'false');
    });
  }

  function render(probabilitySvg, boundarySvg, frame) {
    const model = root.NNComparisonModel;
    if (!model) throw new Error('Comparison models must load before their views.');
    const compact = Boolean(frame.compact), mix = clamp(frame.comparisonMix);
    const opacity = frame.comparisonPlotOpacity === undefined ? 1 : clamp(frame.comparisonPlotOpacity);
    const allGeometry = geometry(model);
    renderView(probabilitySvg, 'probability', compact, model, allGeometry, mix, opacity);
    renderView(boundarySvg, 'boundary', compact, model, allGeometry, mix, opacity);
  }

  root.NNComparisonViews = Object.freeze({ render });
})(typeof window !== 'undefined' ? window : globalThis);
