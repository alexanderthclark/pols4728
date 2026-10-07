/**
 * A course-styled SHAP waterfall, with one common quantitative horizontal scale.
 *
 * renderWaterfall(element, { features, observation, unit, revealed: features.length });
 * observation supplies values[], shapValues[], baseValue, and prediction.
 * Feature metadata may supply label/name/key, unit, decimals, and valueLabels.
 * `revealed` adds contributions from the bottom (smallest magnitude) upward.
 * Call again after a container resize to retain readable text on narrow screens.
 * The caller supplies its own table/text equivalent and reduced-motion styling.
 */

const SVG_NS = 'http://www.w3.org/2000/svg';
const colors = {
  ink: '#242424',
  positive: '#234E70',
  negative: '#A54F32',
  muted: '#606060',
  rule: '#767676',
  paper: '#FFFFFF',
};
let sequence = 0;

function svgElement(name, attributes = {}, content) {
  const element = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, String(value));
  }
  if (content !== undefined) element.textContent = content;
  return element;
}

function number(value, maximumFractionDigits = 0) {
  return new Intl.NumberFormat('en-US', { maximumFractionDigits }).format(value).replace('-', '−');
}

function signed(value) {
  if (value === 0) return '0';
  return `${value < 0 ? '−' : '+'}${number(Math.abs(value), 1)}`;
}

function featureLabel(feature, index) {
  return feature.label || feature.name || feature.key || `Feature ${index + 1}`;
}

function featureValue(feature, value) {
  const label = feature.valueLabels?.[value];
  if (label !== undefined) return String(label);
  const decimals = Number.isInteger(feature.decimals) ? feature.decimals : 1;
  const formatted = typeof value === 'number' ? number(value, decimals) : String(value);
  return feature.unit ? `${formatted} ${feature.unit}` : formatted;
}

function niceStep(range, count) {
  const rough = range / count;
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const fraction = rough / magnitude;
  return (fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10) * magnitude;
}

function labelLines(label, maximumLength) {
  const words = label.split(/\s+/);
  const lines = [''];
  for (const word of words) {
    const last = lines.length - 1;
    if (lines[last] && `${lines[last]} ${word}`.length > maximumLength) lines.push(word);
    else lines[last] += `${lines[last] ? ' ' : ''}${word}`;
  }
  return lines;
}

export function renderWaterfall(container, {
  features, observation, unit = 'model output units', revealed = features.length,
}) {
  if (!container || !Array.isArray(features) || !observation) {
    throw new TypeError('A waterfall needs a container, feature metadata, and an observation.');
  }
  const { baseValue, prediction, shapValues, values } = observation;
  if (!Number.isFinite(baseValue) || !Number.isFinite(prediction)
      || !Array.isArray(shapValues) || shapValues.length !== features.length
      || shapValues.some(value => !Number.isFinite(value))
      || !Array.isArray(values) || values.length !== features.length) {
    throw new TypeError('The observation must contain finite SHAP values, a baseline, prediction, and feature values.');
  }

  const visibleCount = Math.max(0, Math.min(features.length, Math.trunc(revealed)));
  const measuredWidth = container.getBoundingClientRect().width;
  const width = Math.max(280, Math.min(820, Math.round(measuredWidth || 760)));
  const compact = width < 540;
  const labelWidth = compact ? 120 : 205;
  const plotLeft = labelWidth + 16;
  const plotRight = width - 20;
  const rowHeight = compact ? 48 : 64;
  const top = compact ? 64 : 84;
  const axisY = top + Math.max(features.length, 1) * rowHeight + 4;
  const height = axisY + (compact ? 84 : 104);
  const barHeight = 28;
  const entries = features.map((feature, index) => ({
    feature,
    index,
    value: values[index],
    phi: shapValues[index],
    label: featureLabel(feature, index),
  })).sort((a, b) => Math.abs(b.phi) - Math.abs(a.phi) || a.index - b.index);

  let running = baseValue;
  for (let index = entries.length - 1; index >= 0; index -= 1) {
    entries[index].start = running;
    running += entries[index].phi;
    entries[index].end = running;
    entries[index].visible = entries.length - index <= visibleCount;
  }
  const points = [baseValue, prediction, ...entries.flatMap(entry => [entry.start, entry.end])];
  const rawMin = Math.min(...points);
  const rawMax = Math.max(...points);
  const rawRange = Math.max(rawMax - rawMin, Math.abs(baseValue) * 0.03, 1);
  const step = niceStep(rawRange, compact ? 3 : 5);
  const domainMin = Math.floor((rawMin - rawRange * 0.10) / step) * step;
  const domainMax = Math.ceil((rawMax + rawRange * 0.10) / step) * step;
  const x = value => plotLeft + (value - domainMin) / (domainMax - domainMin) * (plotRight - plotLeft);

  const id = `shap-waterfall-${++sequence}`;
  const svg = svgElement('svg', {
    viewBox: `0 0 ${width} ${height}`,
    width: '100%',
    role: 'img',
    'aria-labelledby': `${id}-title ${id}-description`,
    class: 'shap-waterfall waterfall-plot',
    style: 'display:block;overflow:visible;font-family:inherit',
  });
  svg.append(svgElement('title', { id: `${id}-title` }, 'SHAP waterfall for one prediction'));
  const spokenUnit = unit === '$1,000/year' ? 'thousand dollars per year' : unit;
  const description = [
    `The reference prediction is ${number(baseValue, 1)} ${spokenUnit}.`,
    ...entries.filter(entry => entry.visible).reverse().map(entry =>
      `${entry.label}, observed value ${featureValue(entry.feature, entry.value)}, contributes ${signed(entry.phi)} ${spokenUnit}.`),
    `The model prediction is ${number(prediction, 1)} ${spokenUnit}. Contributions are ordered by absolute magnitude.`,
  ].join(' ');
  svg.append(svgElement('desc', { id: `${id}-description` }, description));

  const text = (content, attributes) => svgElement('text', {
    fill: colors.ink,
    'font-size': compact ? 13 : 15,
    ...attributes,
  }, content);

  svg.append(text(compact ? 'Feature contributions' : 'One prediction, feature by feature', {
    x: 0, y: 21, 'font-size': compact ? 17 : 20,
  }));
  svg.append(text(compact ? `Model output in ${unit}.` : `Horizontal position is model output in ${unit}.`, {
    x: 0, y: compact ? 42 : 45, fill: colors.muted, 'font-size': compact ? 11 : 14,
  }));

  // Baseline and output reference lines use the same axis as every contribution.
  svg.append(svgElement('line', {
    x1: x(baseValue), x2: x(baseValue), y1: top - 22, y2: axisY,
    stroke: colors.rule, 'stroke-width': 1, 'stroke-dasharray': '3 5',
  }));
  if (visibleCount === entries.length) {
    svg.append(svgElement('line', {
      x1: x(prediction), x2: x(prediction), y1: top - 22, y2: axisY,
      stroke: colors.ink, 'stroke-width': 1, 'stroke-dasharray': '2 4',
    }));
  }

  for (let index = 0; index < entries.length; index += 1) {
    const entry = entries[index];
    const y = top + index * rowHeight;
    const label = svgElement('g', { class: 'waterfall-feature-label' });
    const lines = labelLines(entry.label, compact ? 17 : 28);
    lines.forEach((line, lineIndex) => label.append(text(line, {
      x: labelWidth, y: y - 8 + lineIndex * 16, 'text-anchor': 'end',
    })));
    label.append(text(`Observed: ${featureValue(entry.feature, entry.value)}`, {
      x: labelWidth, y: y - 8 + lines.length * 16,
      'text-anchor': 'end', fill: colors.muted, 'font-size': compact ? 11 : 13,
    }));
    svg.append(label);
    if (!entry.visible) continue;

    const startX = x(entry.start);
    const endX = x(entry.end);
    const direction = entry.phi < 0 ? -1 : 1;
    const head = Math.min(8, Math.abs(endX - startX));
    const color = entry.phi < 0 ? colors.negative : colors.positive;
    const bar = svgElement('g', {
      class: `waterfall-contribution ${entry.phi < 0 ? 'is-negative' : 'is-positive'}`,
      'data-feature-index': entry.index,
    });
    bar.append(svgElement('title', {}, `${entry.label}: ${signed(entry.phi)} ${spokenUnit}`));
    if (entry.phi === 0) {
      bar.append(svgElement('line', { x1: startX, x2: startX, y1: y - barHeight / 2, y2: y + barHeight / 2, stroke: colors.rule, 'stroke-width': 2 }));
    } else {
      bar.append(svgElement('path', {
        d: `M ${startX} ${y - barHeight / 2} L ${endX - direction * head} ${y - barHeight / 2} L ${endX} ${y} L ${endX - direction * head} ${y + barHeight / 2} L ${startX} ${y + barHeight / 2} Z`,
        fill: color,
      }));
    }
    // Values sit above bars, so even a very small contribution has a readable label.
    bar.append(text(signed(entry.phi), {
      x: (startX + endX) / 2, y: y - barHeight / 2 - 7,
      'text-anchor': 'middle', fill: color, 'font-size': compact ? 13 : 15,
    }));
    if (index > 0 && entries[index - 1].visible) {
      bar.append(svgElement('line', {
        x1: endX, x2: endX, y1: y - barHeight / 2, y2: y - rowHeight + barHeight / 2,
        stroke: colors.rule, 'stroke-width': 1, 'stroke-dasharray': '3 4',
      }));
    }
    svg.append(bar);
  }

  svg.append(svgElement('line', { x1: plotLeft, x2: plotRight, y1: axisY, y2: axisY, stroke: colors.rule, 'stroke-width': 1 }));
  for (let tick = domainMin; tick <= domainMax + step * 0.01; tick += step) {
    const tickX = x(tick);
    svg.append(svgElement('line', { x1: tickX, x2: tickX, y1: axisY, y2: axisY + 5, stroke: colors.rule, 'stroke-width': 1 }));
    svg.append(text(number(tick), { x: tickX, y: axisY + 23, 'text-anchor': 'middle', fill: colors.muted, 'font-size': compact ? 11 : 13 }));
  }
  // These direct labels avoid collisions between reference values on narrow axes.
  const referenceLabelX = compact ? 0 : plotLeft;
  svg.append(text(`Reference: ${number(baseValue, 1)} ${unit}`, { x: referenceLabelX, y: axisY + 48, 'font-size': compact ? 12 : 14 }));
  if (visibleCount === entries.length) {
    svg.append(text(`Prediction: ${number(prediction, 1)} ${unit}`, { x: referenceLabelX, y: axisY + 69, 'font-size': compact ? 12 : 14 }));
  } else {
    const partial = entries.filter(entry => entry.visible).reduce((sum, entry) => sum + entry.phi, baseValue);
    svg.append(text(`Running total: ${number(partial, 1)} ${unit}`, { x: referenceLabelX, y: axisY + 69, 'font-size': compact ? 12 : 14 }));
  }
  container.replaceChildren(svg);
  return svg;
}
