/**
 * Scroll-controlled geometry for one waterfall → a beeswarm → mean |SHAP|.
 * No animation clocks: calling with the same data/progress restores the same SVG.
 */
import { waterfallGeometry } from './waterfall.js';

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
function wrapWords(content, maximumLength) {
  const lines = [''];
  for (const word of content.split(/\s+/)) {
    const last = lines.length - 1;
    if (lines[last] && `${lines[last]} ${word}`.length > maximumLength) lines.push(word);
    else lines[last] += `${lines[last] ? ' ' : ''}${word}`;
  }
  return lines;
}
function setLines(node, lines, x, y, lineHeight) {
  node.replaceChildren(...lines.map((line, index) =>
    element('tspan', { x, y: y + index * lineHeight }, line)));
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
  const legendLabel = text('Input value:', { fill: MUTED });
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
  population, featureOrder, onSelect, reducedMotion = false, backgroundSize = 8 } = {}) {
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
  const geometry = waterfallGeometry(container.getBoundingClientRect().width, features.length);
  const { width, compact, top, rowHeight, axisY, height, labelLineHeight } = geometry;
  const fontSize = compact ? 14 : 16;
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
  const left = geometry.plotLeft;
  const right = geometry.plotRight;
  const labelX = geometry.labelWidth;
  const summaryStep = niceStep(Math.max(...importance), 4);
  const summaryMax = Math.ceil(Math.max(...importance) / summaryStep) * summaryStep;
  const domainMin = phase === 8 ? lerp(-bound, 0, p) : lerp(waterfallMin, -bound, rebase);
  const domainMax = phase === 8 ? lerp(bound, summaryMax, p) : lerp(waterfallMax, bound, rebase);
  const x = value => left + (value - domainMin) / (domainMax - domainMin) * (right - left);
  const finalX = value => left + (value + bound) / (bound * 2) * (right - left);
  const radius = compact ? 1.35 : 1.7;
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
  view.description.textContent = `${titles[phase]}. ${population.length} explained profiles use one fixed model and ${backgroundSize} background rows; baseline ${format(observation.baseValue)} thousand dollars per year. `
    + features.map((feature, index) => `${feature.label}: signed mean ${format(signedMeans[index])}; mean absolute SHAP value ${format(importance[index])}.`).join(' ')
    + (phase >= 4 && phase <= 6 ? 'Color indicates low to high input value within each feature. ' : phase < 4 ? 'Blue contributions raise the model prediction; rust contributions lower it. ' : '')
    + (phase >= 7 ? 'Each bar shows the mean absolute SHAP value across the explained profiles. All bar lengths are nonnegative; the horizontal unit is thousand dollars per year.'
      : phase === 6 ? 'Negative SHAP values fold to their positive distances from zero. These are individual absolute SHAP values, before averaging; the horizontal unit is thousand dollars per year.'
        : phase === 0 ? 'Read the initial waterfall from bottom to top. Cumulative model-output positions become signed SHAP contributions measured from zero.'
          : 'Horizontal position shows a signed SHAP contribution, not an input value. Vertical displacement only separates overlapping profiles.');
  const headingSize = compact ? 22 : 26;
  attrs(view.heading, { x: 0, y: compact ? 24 : 30, 'font-size': headingSize });
  setLines(view.heading, wrapWords(compact ? shortTitles[phase] : titles[phase],
    Math.floor(width / (headingSize * .51))), 0, compact ? 24 : 30, compact ? 26 : 30);
  attrs(view.subheading, { x: 0, y: compact ? 78 : 62, 'font-size': fontSize });
  const subtitle = phase < 2 ? `${observation.name || 'One person'} · baseline ${format(observation.baseValue)}, prediction ${format(observation.prediction)}`
    : phase === 5 ? `${population.length} explained profiles · signed mean 0 in each row`
      : phase >= 7 ? `${population.length} profiles · average absolute contributions`
        : `${population.length} explained profiles · ${backgroundSize} background rows`;
  setLines(view.subheading, wrapWords(subtitle, Math.floor(width / (fontSize * .51))),
    0, compact ? 78 : 62, compact ? 20 : 22);
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
    const barHalf = lerp(geometry.barHeight / 2, radius * 1.8, dotProgress);
    const signColor = phi < 0 ? RUST : BLUE;
    attrs(row.guide, { x1: left, x2: right, y1: cy, y2: cy, opacity: phase >= 2 ? .65 * (1 - meanProgress) : 0 });
    const labelLines = wrapWords(compact ? feature.shortLabel || feature.label : feature.label,
      compact ? 17 : 22);
    const labelY = cy + lerp(-8, 6 - (labelLines.length - 1) * labelLineHeight / 2, rebase);
    attrs(row.label, { x: labelX, y: labelY, 'font-size': compact ? 16 : 18 });
    setLines(row.label, labelLines, labelX, labelY, labelLineHeight);
    attrs(row.observed, { x: labelX, y: cy - 8 + labelLines.length * labelLineHeight,
      'font-size': fontSize, opacity: 1 - rebase });
    row.observed.textContent = `Input: ${format(observation.values[index])}${feature.unit ? ` ${feature.unit}` : ''}`;
    attrs(row.bar, { d: `M${startX},${cy - barHalf} L${endX - direction * head},${cy - barHalf} L${endX},${cy} L${endX - direction * head},${cy + barHalf} L${startX},${cy + barHalf} Z`,
      fill: signColor, opacity: phase < 2 ? 1 - dotProgress : 0 });
    attrs(row.barLabel, { x: (startX + endX) / 2, y: cy - geometry.barHeight / 2 - 10, fill: signColor,
      opacity: phase < 2 ? 1 - dotProgress : 0, 'font-size': compact ? 18 : 20 });
    row.barLabel.textContent = signed(phi);
    const rowPosition = order.indexOf(index);
    attrs(row.connector, { x1: x(end), x2: x(end), y1: cy - geometry.barHeight / 2, y2: cy - rowHeight + geometry.barHeight / 2,
      opacity: rowPosition > 0 && phase === 0 ? 1 - p : 0 });
    attrs(row.meanBar, { x: x(0), y: cy - (compact ? 13 : 16),
      width: Math.max(0, (x(importance[index]) - x(0)) * meanProgress), height: compact ? 26 : 32,
      opacity: meanProgress });
    attrs(row.meanLabel, { x: x(importance[index]) + 8, y: cy + 6,
      opacity: meanProgress, 'font-size': compact ? 18 : 20 });
    row.meanLabel.textContent = format(importance[index]);
    attrs(view.localDots[index], { cx: x(phi), cy,
      r: lerp(3, compact ? 3.1 : 4.1, dotProgress), fill: signColor,
      opacity: phase === 1 ? p : phase === 2 ? 1 - p : 0 });
    const meanX = x(lerp(signedMeans[index], importance[index], absoluteProgress));
    const diamond = compact ? 4 : 5;
    attrs(view.means[index].marker, { d: `M${meanX},${cy - diamond} L${meanX + diamond},${cy} L${meanX},${cy + diamond} L${meanX - diamond},${cy} Z`,
      opacity: signedMeanProgress, fill: INK });
    attrs(view.means[index].label, { x: labelX, y: cy + (compact ? 29 : 44),
      opacity: signedMeanProgress, 'font-size': fontSize });
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
    attrs(tick.label, { x: cx, y: axisY + 25, 'font-size': fontSize });
    tick.label.textContent = format(value, rebase > 0 && rebase < 1 ? 1 : 2);
  });
  const axisTitle = rebase < .5 ? 'Model output'
    : phase >= 7 ? 'Mean absolute SHAP value'
      : phase === 6 && absoluteProgress > .5 ? 'Absolute SHAP value' : 'SHAP value';
  const axisLabelX = compact ? width / 2 : (left + right) / 2;
  attrs(view.axisLabel, { x: axisLabelX, y: axisY + 51, 'font-size': fontSize });
  setLines(view.axisLabel, compact ? [axisTitle, '$1,000/year'] : [`${axisTitle} · $1,000/year`],
    axisLabelX, axisY + 51, 19);
  const legendY = axisY + (compact ? 98 : 90);
  attrs(view.signLegend, { opacity: (phase >= 1 && phase <= 4 ? clamp(1 - colorProgress * 2) : 0), 'font-size': fontSize });
  attrs(view.negativeKey, { cx: 4, cy: legendY - 3 });
  attrs(view.negativeText, { x: 14, y: legendY });
  attrs(view.positiveKey, { cx: compact ? width / 2 + 8 : 206, cy: legendY - 3 });
  attrs(view.positiveText, { x: compact ? width / 2 + 18 : 216, y: legendY });
  attrs(view.valueLegend, { opacity: clamp(colorProgress * 2 - 1) * (1 - meanProgress), 'font-size': fontSize });
  attrs(view.legendLabel, { x: 0, y: legendY });
  attrs(view.legendLow, { x: compact ? 92 : 110, y: legendY });
  attrs(view.legendRamp, { x: compact ? 126 : 154, y: legendY - 9, width: compact ? 80 : 120, height: 9 });
  attrs(view.legendHigh, { x: compact ? 214 : 282, y: legendY });
  attrs(view.footer, { x: 0, y: legendY, 'font-size': phase === 0 ? compact ? 16 : 18 : fontSize,
    opacity: phase === 0 ? 1 - p : phase >= 7 ? meanProgress : 0 });
  view.footer.textContent = phase === 0 ? `Baseline ${format(observation.baseValue)} → prediction ${format(observation.prediction)}`
    : 'Bar length: mean absolute contribution';
  return view.svg;
}

export function destroyAggregate(container) {
  const view = views.get(container);
  if (view?.svg.parentNode === container) view.svg.remove();
  views.delete(container);
}
