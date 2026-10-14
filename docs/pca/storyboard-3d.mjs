import { THREE_D_POINTS, THREE_D_BASIS, THREE_D_RESIDUALS, dotND } from './model.mjs';
import { RESIDUAL_BASIS, residualCamera, projectWithCamera, residualPointAt } from './storyboard-geometry.mjs';

const colors = { ink: '#242424', blue: '#234E70', rust: '#A54F32', teal: '#007F73', muted: '#606060', rule: '#767676', light: '#d6d6d6' };
const columns = [0, 1, 2].map(j => THREE_D_BASIS.map(row => row[j]));
const clamp = value => Math.min(1, Math.max(0, value));
const scaled = (vector, amount) => vector.map(value => amount * value);

function add(parent, name, attributes = {}, text) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  parent.append(node);
  return node;
}

function line(parent, first, last, color = colors.rule, width = 1, dash = '') {
  return add(parent, 'line', { x1: first[0], y1: first[1], x2: last[0], y2: last[1], stroke: color, 'stroke-width': width, ...(dash ? { 'stroke-dasharray': dash } : {}) });
}

function label(parent, position, runs, color = colors.muted, size = 18, anchor = 'start') {
  const compact = matchMedia('(max-width: 800px)').matches;
  const text = add(parent, 'text', { x: position[0], y: position[1], fill: color, 'font-size': compact ? Math.max(28, size * 1.55) : size, 'text-anchor': anchor });
  for (const run of Array.isArray(runs) ? runs : [runs]) {
    if (typeof run === 'string') add(text, 'tspan', {}, run);
    else {
      add(text, 'tspan', { class: 'math-symbol', 'font-style': 'italic' }, run.text);
      if (run.sub !== undefined) add(text, 'tspan', { class: 'math-script', 'font-size': '.7em', 'baseline-shift': 'sub' }, run.sub);
    }
  }
  return text;
}

function point(parent, position, index, selected, hollow = false) {
  add(parent, 'circle', { cx: position[0], cy: position[1], r: hollow ? 4 : 4.5, fill: hollow ? '#fff' : colors.blue, stroke: hollow ? colors.rule : colors.blue, 'stroke-width': hollow ? 1.4 : .7, 'data-point-index': index });
  if (index === selected) add(parent, 'circle', { cx: position[0], cy: position[1], r: 9.5, fill: 'none', stroke: colors.rust, 'stroke-width': 2, 'data-highlight-index': index });
}

function arrow(parent, first, last, color, dash = '') {
  line(parent, first, last, color, 2.2, dash);
  const theta = Math.atan2(last[1] - first[1], last[0] - first[0]);
  for (const offset of [-.45, .45]) line(parent, [last[0] - 9 * Math.cos(theta + offset), last[1] - 9 * Math.sin(theta + offset)], last, color, 2.2);
}

function residualAxes(parent, map, opacity = 1) {
  const group = add(parent, 'g', { opacity });
  const extent = 2.35;
  for (const [axis, name] of [[RESIDUAL_BASIS.a, 'a'], [RESIDUAL_BASIS.b, 'b']]) {
    line(group, map(scaled(axis, -extent)), map(scaled(axis, extent)), colors.rule);
    const end = map(scaled(axis, extent));
    label(group, name === 'a' ? [end[0] - 4, end[1] + 24] : [end[0] + 9, end[1] + 25], [{ text: name }], colors.muted, 19, name === 'a' ? 'end' : 'start');
    for (const tick of [-2, -1, 1, 2]) {
      const position = map(scaled(axis, tick));
      const other = name === 'a' ? RESIDUAL_BASIS.b : RESIDUAL_BASIS.a;
      const first = map(axis.map((value, j) => tick * value - .065 * other[j]));
      const last = map(axis.map((value, j) => tick * value + .065 * other[j]));
      line(group, first, last, colors.rule);
      label(group, name === 'a' ? [position[0], position[1] + 21] : [position[0] - 9, position[1] + 5], String(tick).replace('-', '−'), colors.muted, 14, name === 'a' ? 'middle' : 'end');
    }
  }
}

/** Rotate the camera around unchanged residuals, then zoom to their 2D view. */
export function drawResidualTransition(parent, { progress = 0, camera = 25, selected = 23, points = THREE_D_RESIDUALS } = {}) {
  const amount = clamp(progress);
  const frame = residualCamera(amount, camera);
  const map = point => projectWithCamera(point, frame);
  label(parent, [360, 40], amount === 1 ? 'Residual plane · viewed face on' : 'Turn to face the residual plane', colors.ink, 21, 'middle');
  const shade = add(parent, 'g', { opacity: 1 - amount });
  const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, b]) => columns[1].map((value, j) => 2.8 * a * value + 2.3 * b * columns[2][j]));
  add(shade, 'polygon', { points: corners.map(point => map(point).join(',')).join(' '), fill: colors.blue, 'fill-opacity': .07, stroke: colors.blue, 'stroke-width': 1 });
  const originalAxes = add(parent, 'g', { opacity: (1 - amount) ** 3 });
  for (let j = 0; j < 3; j++) {
    const vector = [0, 0, 0]; vector[j] = 5.7;
    line(originalAxes, map(scaled(vector, -1)), map(vector), colors.light);
    const end = map(vector);
    label(originalAxes, [end[0] + 10, end[1] + 5], [{ text: 'x', sub: j + 1 }], colors.muted, 19);
  }
  line(originalAxes, map(scaled(columns[0], -6)), map(scaled(columns[0], 6)), colors.blue, 1.3);
  arrow(originalAxes, map([0, 0, 0]), map(columns[0]), colors.blue);
  const primaryEnd = map(scaled(columns[0], 6));
  label(originalAxes, [primaryEnd[0] + 12, primaryEnd[1] - 7], [{ text: 'u', sub: 1 }], colors.blue, 21);
  const original = add(parent, 'g', { opacity: (1 - amount) ** 4 });
  if (points.length === THREE_D_POINTS.length) THREE_D_POINTS.forEach((point3D, index) => {
    line(original, map(point3D), map(points[index]), colors.rule, 1, '3 4');
    point(original, map(point3D), index, -1, true);
  });
  residualAxes(parent, map, amount);
  points.map((position, index) => ({ position, index })).sort((a, b) => dotND(a.position, frame.normal) - dotND(b.position, frame.normal)).forEach(({ position, index }) => point(parent, map(position), index, selected));
  label(parent, [360, 401], 'The same residuals · only the view changes', colors.muted, 18, 'middle');
  return { camera: frame, points: points.map(position => [...position]), projected: points.map(map) };
}

/** Remove the second contribution in the already established face-on plane. */
export function drawThirdDirection(parent, { progress = 0, selected = 23, points = THREE_D_RESIDUALS } = {}) {
  const amount = clamp(progress);
  const frame = residualCamera(1);
  const map = point => projectWithCamera(point, frame);
  const displayed = points.map(point => residualPointAt(point, amount));
  label(parent, [360, 40], 'Subtract PC2 · the PC3 line remains', colors.ink, 21, 'middle');
  residualAxes(parent, map);
  for (const [index, length, color, dash] of [[1, 2.2, colors.rust, '6 5'], [2, 2.2, colors.teal, '2 4']]) {
    const axis = columns[index];
    line(parent, map(scaled(axis, -length)), map(scaled(axis, length)), color, 1.3, dash);
    arrow(parent, map([0, 0, 0]), map(axis), color, dash);
    const end = map(scaled(axis, length));
    label(parent, [end[0] + 10, end[1] - 8], [{ text: 'u', sub: index + 1 }], color, 21);
  }
  points.forEach((position, index) => {
    line(parent, map(position), map(displayed[index]), colors.rule, 1, '3 4');
    point(parent, map(position), index, -1, true);
  });
  displayed.forEach((position, index) => point(parent, map(position), index, selected));
  label(parent, [360, 401], '○ Before subtraction     ● Remaining observation', colors.muted, 18, 'middle');
  label(parent, [360, 435], amount === 1 ? 'Only the PC3 coordinate remains' : 'Each point travels parallel to PC2', colors.ink, 18, 'middle');
  return { points: displayed, projected: displayed.map(map), camera: frame };
}
