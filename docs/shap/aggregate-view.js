/**
 * Scroll-controlled geometry for one waterfall → a beeswarm → mean |SHAP|.
 * No animation clocks: calling with the same data/progress restores the same SVG.
 */
const NS = 'http://www.w3.org/2000/svg';
const INK = '#242424', BLUE = '#234E70', RUST = '#A54F32';
const MUTED = '#606060', RULE = '#D6D6D6';
const SCENES = ['contribution-axis', 'one-person-dots', 'many-people', 'beeswarm',
  'feature-color', 'signed-average', 'absolute-values', 'global-importance', 'importance-ranking'];
const views = new WeakMap();
let sequence = 0;

const clamp = value => Math.max(0, Math.min(1, value));
const lerp = (a, b, t) => a + (b - a) * t;
const ease = value => { const t = clamp(value); return t * t * (3 - 2 * t); };
const format = (value, decimals = 2) => new Intl.NumberFormat('en-US', {
  maximumFractionDigits: decimals,
}).format(Math.abs(value) < 1e-9 ? 0 : value).replace('-', '−');
const signed = value => `${value > 0 ? '+' : ''}${format(value)}`;

function element(tag, attributes = {}, content) {
  const node = document.createElementNS(NS, tag);
  attrs(node, attributes);
  if (content !== undefined) node.textContent = content;
  return node;
}
function attrs(node, attributes) {
  for (const [key, value] of Object.entries(attributes)) {
    if (node.getAttribute(key) !== String(value)) node.setAttribute(key, String(value));
  }
}
function text(content = '', attributes = {}) {
  return element('text', { fill: INK, ...attributes }, content);
}
function mixColor(a, b, t) {
  const channels = color => color.startsWith('rgb')
    ? color.match(/[\d.]+/g).slice(0, 3).map(Number)
    : color.match(/[0-9a-f]{2}/gi).map(value => parseInt(value, 16));
  return `rgb(${channels(a).map((value, index) => Math.round(lerp(value, channels(b)[index], t))).join(',')})`;
}
function valueColor(value, feature, fallbackDomain) {
  const domain = feature.domain || fallbackDomain;
  const t = domain[1] === domain[0] ? .5 : clamp((value - domain[0]) / (domain[1] - domain[0]));
  return t < .5 ? mixColor(BLUE, '#D0CBC5', t * 2) : mixColor('#D0CBC5', RUST, (t - .5) * 2);
}
function niceStep(range, count) {
  const rough = range / count;
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const fraction = rough / magnitude;
  return (fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10) * magnitude;
}

/** Pack only vertically. Every point retains its exact quantitative x position. */
export function packBeeswarm(points, radius) {
  const diameter = radius * 2 + .1;
  const placed = [];
  const offsets = new Array(points.length).fill(0);
  const sorted = points.map((x, index) => ({ x, index })).sort((a, b) => a.x - b.x || a.index - b.index);
  for (const point of sorted) {
    const neighbors = placed.filter(other => point.x - other.x < diameter);
    const candidates = [0];
    for (const other of neighbors) {
      const dy = Math.sqrt(Math.max(0, diameter ** 2 - (point.x - other.x) ** 2));
      candidates.push(other.y + dy, other.y - dy);
    }
    candidates.sort((a, b) => Math.abs(a) - Math.abs(b) || a - b);
    const y = candidates.find(candidate => neighbors.every(other =>
      (point.x - other.x) ** 2 + (candidate - other.y) ** 2 >= diameter ** 2 - 1e-6)) || 0;
    offsets[point.index] = y;
    placed.push({ x: point.x, y });
  }
  return offsets;
}

function createView(container, features, population) {
  const id = `shap-aggregate-${++sequence}`;
  const svg = element('svg', { class: 'shap-aggregate', width: '100%', role: 'img',
    'aria-labelledby': `${id}-title ${id}-description` });
  const title = element('title', { id: `${id}-title` });
  const description = element('desc', { id: `${id}-description` });
  svg.append(title, description);
  const defs = element('defs');
  const gradient = element('linearGradient', { id: `${id}-value-gradient` });
  [['0%', BLUE], ['50%', '#D0CBC5'], ['100%', RUST]].forEach(([offset, color]) =>
    gradient.append(element('stop', { offset, 'stop-color': color })));
  defs.append(gradient); svg.append(defs);
  const heading = text('', { class: 'aggregate-heading' });
  const subheading = text('', { class: 'aggregate-subheading', fill: MUTED });
  svg.append(heading, subheading);
  const guides = element('g', { 'aria-hidden': 'true' });
  const baseline = element('line', { stroke: MUTED, 'stroke-dasharray': '3 5' });
  const prediction = element('line', { stroke: INK, 'stroke-dasharray': '2 4' });
  const zero = element('line', { stroke: '#767676', 'stroke-width': 1, 'stroke-dasharray': '3 4' });
  guides.append(baseline, prediction, zero); svg.append(guides);
  const rows = features.map((feature, index) => {
    const group = element('g', { 'data-feature-index': index });
    const guide = element('line', { stroke: RULE, 'stroke-width': .65 });
    const label = text('', { class: 'aggregate-feature-label', 'text-anchor': 'end' });
    const observed = text('', { class: 'aggregate-observed', 'text-anchor': 'end', fill: MUTED });
    const bar = element('path', { class: 'aggregate-local-bar' });
    const barLabel = text('', { 'text-anchor': 'middle', class: 'aggregate-local-value' });
    const connector = element('line', { stroke: MUTED, 'stroke-dasharray': '3 4' });
    const meanBar = element('rect', { class: 'aggregate-mean-bar', fill: BLUE });
    const meanLabel = text('', { class: 'aggregate-mean-value', fill: BLUE });
    group.append(guide, connector, meanBar, bar, label, observed, barLabel, meanLabel);
    svg.append(group);
    return { group, guide, label, observed, bar, barLabel, connector, meanBar, meanLabel };
  });
  const dotLayer = element('g', { class: 'aggregate-dots', 'aria-hidden': 'true' });
  const dots = features.map((feature, featureIndex) => population.map((person, personIndex) => {
    const circle = element('circle', { class: 'aggregate-dot', 'data-feature-index': featureIndex,
      'data-person-index': personIndex, 'data-person-id': person.id,
      'stroke-width': .3, stroke: 'white' });
    circle.append(element('title', {}, `${person.name || person.id}; ${feature.label}: input ${format(person.values[featureIndex])}; SHAP ${signed(person.shapValues[featureIndex])} thousand dollars per year.`));
    dotLayer.append(circle);
    return circle;
  }));
  svg.append(dotLayer);
  const localDots = features.map((_, index) => {
    const dot = element('circle', { class: 'aggregate-local-dot', 'data-feature-index': index,
      stroke: 'white', 'stroke-width': .8, 'aria-hidden': 'true' });
    svg.append(dot); return dot;
  });
  const means = features.map((_, index) => {
    const marker = element('path', { class: 'aggregate-mean-marker', fill: INK, 'data-feature-index': index });
    const label = text('Mean = 0', { class: 'aggregate-signed-mean', 'text-anchor': 'end', fill: INK });
    svg.append(marker, label); return { marker, label };
  });
  const axis = element('g', { class: 'aggregate-axis', 'aria-hidden': 'true' });
  const axisLine = element('line', { stroke: '#767676' });
  axis.append(axisLine);
  const ticks = Array.from({ length: 5 }, () => {
    const group = element('g');
    const line = element('line', { stroke: '#767676' });
    const label = text('', { 'text-anchor': 'middle', fill: MUTED });
    group.append(line, label); axis.append(group); return { group, line, label };
  });
  const axisLabel = text('', { class: 'aggregate-axis-label', 'text-anchor': 'middle', fill: MUTED });
  axis.append(axisLabel); svg.append(axis);
  const signLegend = element('g', { class: 'aggregate-sign-legend' });
  const negativeKey = element('circle', { r: 3, fill: RUST });
  const negativeText = text('Lowers prediction', { fill: MUTED });
  const positiveKey = element('circle', { r: 3, fill: BLUE });
  const positiveText = text('Raises prediction', { fill: MUTED });
  signLegend.append(negativeKey, negativeText, positiveKey, positiveText); svg.append(signLegend);
  const valueLegend = element('g', { class: 'aggregate-value-legend' });
  const legendLabel = text('Feature value:', { fill: MUTED });
  const legendLow = text('Low', { fill: MUTED });
  const legendHigh = text('High', { fill: MUTED });
  const legendRamp = element('rect', { height: 7, rx: 0, fill: `url(#${id}-value-gradient)` });
  valueLegend.append(legendLabel, legendLow, legendHigh, legendRamp); svg.append(valueLegend);
  const footer = text('', { class: 'aggregate-footer', fill: MUTED });
  svg.append(footer);
  const view = { svg, title, description, heading, subheading, baseline, prediction, zero, rows, dots,
    localDots, means, axisLine, ticks, axisLabel, signLegend, negativeKey, negativeText, positiveKey,
    positiveText, valueLegend, legendLabel, legendLow, legendHigh, legendRamp, footer,
    featureCount: features.length, population, packing: null, onSelect: null, scene: null };
  svg.addEventListener('click', event => {
    const dot = event.target.closest?.('.aggregate-dot');
    if (dot && view.scene === 'feature-color' && view.onSelect) {
      view.onSelect(view.population[Number(dot.getAttribute('data-person-index'))]);
    }
  });
  container.replaceChildren(svg);
  views.set(container, view);
  return view;
}

export function renderAggregate(container, { scene, progress = 1, features, observation,
  population, featureOrder, onSelect, reducedMotion = false } = {}) {
  if (!container || !features?.length || !observation || !population?.length || !SCENES.includes(scene)) {
    throw new TypeError('An aggregate plot needs its scene, feature metadata, local observation, and population.');
  }
  let view = views.get(container);
  if (!view || !container.contains(view.svg) || view.population !== population || view.featureCount !== features.length) {
    view = createView(container, features, population);
  }
  view.scene = scene; view.onSelect = typeof onSelect === 'function' ? onSelect : null;
  const phase = SCENES.indexOf(scene);
  const p = reducedMotion ? 1 : ease(Number.isFinite(progress) ? progress : 1);
  const width = Math.max(280, Math.min(820, Math.round(container.getBoundingClientRect().width || 760)));
  const compact = width < 540;
  const fontSize = compact ? 11 : 13;
  const order = featureOrder || observation.shapValues.map((_, index) => index)
    .sort((a, b) => Math.abs(observation.shapValues[b]) - Math.abs(observation.shapValues[a]) || a - b);
  const importance = features.map((_, feature) => population.reduce((sum, person) =>
    sum + Math.abs(person.shapValues[feature]), 0) / population.length);
  const signedMeans = features.map((_, feature) => population.reduce((sum, person) =>
    sum + person.shapValues[feature], 0) / population.length);
  const ranked = features.map((_, index) => index).sort((a, b) => importance[b] - importance[a] || a - b);
  const cumulative = [];
  let running = observation.baseValue;
  [...order].reverse().forEach(index => {
    cumulative[index] = { start: running, end: running + observation.shapValues[index] };
    running += observation.shapValues[index];
  });
  const extrema = [observation.baseValue, observation.prediction,
    ...cumulative.flatMap(entry => [entry.start, entry.end])];
  const rawMin = Math.min(...extrema), rawMax = Math.max(...extrema);
  const rawRange = Math.max(rawMax - rawMin, Math.abs(observation.baseValue) * .03, 1);
  const waterfallStep = niceStep(rawRange, compact ? 3 : 5);
  const waterfallMin = Math.floor((rawMin - rawRange * .1) / waterfallStep) * waterfallStep;
  const waterfallMax = Math.ceil((rawMax + rawRange * .1) / waterfallStep) * waterfallStep;
  const extent = Math.max(1, ...population.flatMap(person => person.shapValues.map(Math.abs)),
    ...observation.shapValues.map(Math.abs));
  const bound = Math.ceil(extent / 10) * 10;
  const rebase = phase === 0 ? p : 1;
  const left = lerp(compact ? 136 : 221, compact ? 98 : 184, rebase);
  const right = width - (compact ? 16 : 24);
  const top = lerp(compact ? 64 : 84, compact ? 62 : 76, rebase);
  const rowHeight = lerp(compact ? 48 : 64, compact ? 52 : 70, rebase);
  const axisY = top + (features.length - 1) * rowHeight + lerp(compact ? 30 : 68, compact ? 30 : 40, rebase);
  const height = axisY + lerp(compact ? 66 : 104, compact ? 66 : 78, rebase);
  const labelX = left - 15;
  const summaryStep = niceStep(Math.max(...importance), 4);
  const summaryMax = Math.ceil(Math.max(...importance) / summaryStep) * summaryStep;
  const domainMin = phase === 8 ? lerp(-bound, 0, p) : lerp(waterfallMin, -bound, rebase);
  const domainMax = phase === 8 ? lerp(bound, summaryMax, p) : lerp(waterfallMax, bound, rebase);
  const x = value => left + (value - domainMin) / (domainMax - domainMin) * (right - left);
  const finalX = value => (compact ? 98 : 184) + (value + bound) / (bound * 2) * (right - (compact ? 98 : 184));
  const radius = compact ? 1 : 1.35;
  const cacheKey = `${width}:${bound}:${population.length}`;
  if (view.packing?.key !== cacheKey) {
    view.packing = { key: cacheKey,
      signed: features.map((_, index) => packBeeswarm(population.map(person => finalX(person.shapValues[index])), radius)),
      domains: features.map((_, index) => [Math.min(...population.map(person => person.values[index])), Math.max(...population.map(person => person.values[index]))]),
    };
  }
  const y = index => top + lerp(order.indexOf(index), ranked.indexOf(index), phase === 8 ? p : 0) * rowHeight;
  const dotProgress = phase === 1 ? p : phase > 1 ? 1 : 0;
  const swarmProgress = phase === 3 ? p : phase > 3 ? 1 : 0;
  const colorProgress = phase === 4 ? p : phase > 4 ? 1 : 0;
  const absoluteProgress = phase === 6 ? p : phase > 6 ? 1 : 0;
  const meanProgress = phase === 7 ? p : phase > 7 ? 1 : 0;
  const signedMeanProgress = phase === 5 ? p : phase === 6 ? 1 - p : 0;
  attrs(view.svg, { viewBox: `0 0 ${width} ${height}`, 'data-scene': scene,
    'data-progress': format(progress, 4), 'data-reduced-motion': reducedMotion,
    'data-population-count': population.length, style: `font-size:${fontSize}px` });
  const titles = ['Give every contribution its own zero', 'One person, one dot per feature',
    'One dot per person, in every row', 'Stack overlapping dots into a beeswarm',
    'Color shows the input value', 'Opposing contributions cancel in a signed mean',
    'Measure each contribution’s distance from zero', 'Average those distances, feature by feature',
    'Rank features by mean absolute SHAP value'];
  const shortTitles = ['Put contributions on one axis', 'One person becomes three dots',
    `${population.length} people, three feature rows`, 'Spread overlaps vertically',
    'Color shows the input value', 'Signed contributions can cancel',
    'Take each distance from zero', 'Average the absolute values', 'Rank the average magnitudes'];
  view.title.textContent = titles[phase];
  view.description.textContent = `${titles[phase]}. ${population.length} illustrative profiles; baseline ${format(observation.baseValue)} thousand dollars per year. `
    + features.map((feature, index) => `${feature.label}: signed mean ${format(signedMeans[index])}; mean absolute SHAP value ${format(importance[index])}.`).join(' ')
    + (phase >= 4 ? 'Color indicates low to high input value within each feature. Horizontal position indicates SHAP contribution, not input value.' : 'Blue contributions raise the model prediction; rust contributions lower it.')
    + (phase >= 6 ? 'Negative contributions move to their absolute distances from zero. The horizontal unit is unchanged.' : 'Vertical displacement only separates overlapping observations.');
  attrs(view.heading, { x: 0, y: compact ? 17 : 22, 'font-size': compact ? 15 : 19 });
  view.heading.textContent = compact ? shortTitles[phase] : titles[phase];
  attrs(view.subheading, { x: 0, y: compact ? 35 : 43, 'font-size': compact ? 10 : 12 });
  view.subheading.textContent = phase < 2 ? `${observation.name || 'One person'} · reference ${format(observation.baseValue)}, prediction ${format(observation.prediction)}`
    : phase === 5 ? `${population.length} profiles · the signed mean is 0 for every feature here`
      : phase >= 7 ? 'Mean absolute SHAP · $1,000/year'
        : `${population.length} illustrative profiles · one fixed model and background`;
  attrs(view.baseline, { x1: x(lerp(observation.baseValue, 0, rebase)), x2: x(lerp(observation.baseValue, 0, rebase)),
    y1: top - 21, y2: axisY, opacity: 1 - rebase });
  attrs(view.prediction, { x1: x(lerp(observation.prediction, 0, rebase)), x2: x(lerp(observation.prediction, 0, rebase)),
    y1: top - 21, y2: axisY, opacity: 1 - rebase });
  attrs(view.zero, { x1: x(0), x2: x(0), y1: top - (compact ? 19 : 25), y2: axisY,
    opacity: rebase });
  const selectedIndex = population.findIndex(person => person.id === observation.id ||
    person.values.every((value, index) => Math.abs(value - observation.values[index]) < 1e-9));
  for (let index = 0; index < features.length; index += 1) {
    const feature = features[index], row = view.rows[index], cy = y(index), phi = observation.shapValues[index];
    const start = lerp(cumulative[index].start, 0, rebase);
    const end = lerp(cumulative[index].end, phi, rebase);
    const startX = lerp(x(start), x(phi), dotProgress), endX = x(end);
    const direction = phi < 0 ? -1 : 1;
    const head = Math.min(8 * (1 - dotProgress), Math.abs(endX - startX));
    const barHalf = lerp(14, radius * 1.8, dotProgress);
    const signColor = phi < 0 ? RUST : BLUE;
    attrs(row.guide, { x1: left, x2: right, y1: cy, y2: cy, opacity: phase >= 2 ? .65 * (1 - meanProgress) : 0 });
    attrs(row.label, { x: labelX, y: cy + (phase === 0 ? lerp(-8, 4, p) : 4), 'font-size': compact ? 11 : 14 });
    row.label.textContent = compact ? feature.shortLabel || feature.label : feature.label;
    if (!compact && row.label.textContent.length > 23) {
      row.label.replaceChildren();
      const words = feature.label.split(' ');
      row.label.append(element('tspan', { x: labelX, dy: '-.35em' }, words.slice(0, -1).join(' ')),
        element('tspan', { x: labelX, dy: '1.15em' }, words.at(-1)));
    }
    attrs(row.observed, { x: labelX, y: cy + 16, 'font-size': compact ? 10 : 12, opacity: 1 - rebase });
    row.observed.textContent = `Observed: ${format(observation.values[index])} ${feature.unit || ''}`;
    attrs(row.bar, { d: `M${startX},${cy - barHalf} L${endX - direction * head},${cy - barHalf} L${endX},${cy} L${endX - direction * head},${cy + barHalf} L${startX},${cy + barHalf} Z`,
      fill: signColor, opacity: phase < 2 ? 1 - dotProgress : 0 });
    attrs(row.barLabel, { x: (startX + endX) / 2, y: cy - 21, fill: signColor,
      opacity: phase < 2 ? 1 - dotProgress : 0, 'font-size': compact ? 12 : 14 });
    row.barLabel.textContent = signed(phi);
    const rowPosition = order.indexOf(index);
    attrs(row.connector, { x1: x(end), x2: x(end), y1: cy - 14, y2: cy - rowHeight + 14,
      opacity: rowPosition > 0 && phase === 0 ? 1 - p : 0 });
    attrs(row.meanBar, { x: x(0), y: cy - (compact ? 8 : 11),
      width: Math.max(0, (x(importance[index]) - x(0)) * meanProgress), height: compact ? 16 : 22,
      opacity: meanProgress });
    attrs(row.meanLabel, { x: x(importance[index]) + 7, y: cy + 4,
      opacity: meanProgress, 'font-size': compact ? 12 : 15 });
    row.meanLabel.textContent = format(importance[index]);
    attrs(view.localDots[index], { cx: x(phi), cy,
      r: lerp(3, compact ? 3.1 : 4.1, dotProgress), fill: signColor,
      opacity: phase === 1 ? p : phase === 2 ? 1 - p : 0 });
    const meanX = x(lerp(signedMeans[index], importance[index], absoluteProgress));
    const diamond = compact ? 4 : 5;
    attrs(view.means[index].marker, { d: `M${meanX},${cy - diamond} L${meanX + diamond},${cy} L${meanX},${cy + diamond} L${meanX - diamond},${cy} Z`,
      opacity: signedMeanProgress, fill: INK });
    attrs(view.means[index].label, { x: labelX, y: cy + (compact ? 18 : 23),
      opacity: signedMeanProgress, 'font-size': compact ? 10 : 12 });
    view.means[index].label.textContent = `Mean = ${format(lerp(signedMeans[index], importance[index], absoluteProgress))}`;
    population.forEach((person, personIndex) => {
      const circle = view.dots[index][personIndex];
      const contribution = person.shapValues[index];
      const reveal = phase === 2 ? ease((p - personIndex / population.length * .55) / .45) : phase > 2 ? 1 : 0;
      const horizontal = lerp(lerp(observation.shapValues[index], contribution, reveal),
        Math.abs(contribution), absoluteProgress);
      const position = lerp(horizontal, importance[index], meanProgress);
      // Folding changes only x. Preserve each person's vertical identity; dots
      // can overlap when opposite signed contributions share an absolute value.
      const offset = view.packing.signed[index][personIndex] * swarmProgress * (1 - meanProgress);
      attrs(circle, { cx: x(position), cy: cy + offset, r: radius,
        fill: mixColor(contribution < 0 ? RUST : BLUE,
          valueColor(person.values[index], feature, view.packing.domains[index]), colorProgress),
        opacity: reveal * (1 - meanProgress),
        'data-selected': personIndex === selectedIndex,
        'data-shap-value': contribution,
        'data-input-value': person.values[index],
        'data-swarm-offset': format(offset, 4),
      });
    });
  }
  attrs(view.axisLine, { x1: left, x2: right, y1: axisY, y2: axisY });
  view.ticks.forEach((tick, index) => {
    const value = lerp(domainMin, domainMax, index / 4), cx = x(value);
    attrs(tick.group, { opacity: value < -1e-6 ? 1 - absoluteProgress : 1 });
    attrs(tick.line, { x1: cx, x2: cx, y1: axisY, y2: axisY + 5 });
    attrs(tick.label, { x: cx, y: axisY + 19, 'font-size': compact ? 10 : 12 });
    tick.label.textContent = format(value, rebase > 0 && rebase < 1 ? 1 : 2);
  });
  attrs(view.axisLabel, { x: (left + right) / 2, y: axisY + (compact ? 35 : 40),
    'font-size': compact ? 10 : 12 });
  view.axisLabel.textContent = rebase < .5 ? 'Model output · $1,000/year'
    : phase >= 6 && absoluteProgress > .5 ? 'Absolute SHAP · $1,000/year' : 'SHAP value · $1,000/year';
  const legendY = axisY + (compact ? 51 : 59);
  attrs(view.signLegend, { opacity: (phase >= 1 && phase <= 4 ? clamp(1 - colorProgress * 2) : 0), 'font-size': compact ? 9 : 11 });
  attrs(view.negativeKey, { cx: 4, cy: legendY - 3 });
  attrs(view.negativeText, { x: 12, y: legendY });
  attrs(view.positiveKey, { cx: compact ? 133 : 170, cy: legendY - 3 });
  attrs(view.positiveText, { x: compact ? 141 : 178, y: legendY });
  attrs(view.valueLegend, { opacity: clamp(colorProgress * 2 - 1) * (1 - meanProgress), 'font-size': compact ? 10 : 12 });
  attrs(view.legendLabel, { x: 0, y: legendY });
  attrs(view.legendLow, { x: compact ? 83 : 102, y: legendY });
  attrs(view.legendRamp, { x: compact ? 108 : 133, y: legendY - 7, width: compact ? 69 : 100 });
  attrs(view.legendHigh, { x: compact ? 184 : 242, y: legendY });
  attrs(view.footer, { x: 0, y: legendY, 'font-size': compact ? 10 : 12,
    opacity: phase === 0 ? 1 - p : phase >= 7 ? meanProgress : 0 });
  view.footer.textContent = phase === 0 ? `Reference ${format(observation.baseValue)} → prediction ${format(observation.prediction)}`
    : 'Bar length: average absolute contribution';
  return view.svg;
}

export function destroyAggregate(container) {
  const view = views.get(container);
  if (view?.svg.parentNode === container) view.svg.remove();
  views.delete(container);
}
