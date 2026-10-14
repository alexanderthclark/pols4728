import { svdExample } from './model.mjs';

// Keep one vertical scale throughout the multiplication. Hollow marks preserve
// the unit pattern while the filled marks grow into the actual score column.
export function drawSVDFactors(parent, helpers, componentIndex, progress = 1) {
  const { add, line, label, colors, fmt } = helpers;
  const { scores, P, singularValues } = svdExample();
  const index = String(componentIndex + 1);
  const color = [colors.blue, colors.rust, colors.teal][componentIndex];
  const d = singularValues[componentIndex];
  const multiplier = 1 + Math.max(0, Math.min(1, progress)) * (d - 1);
  const pattern = P.map(row => row[componentIndex]);
  const range = Math.ceil(Math.max(...scores.map(row => Math.abs(row[componentIndex]))));
  const top = 112, bottom = 338, zero = (top + bottom) / 2;
  const x = i => 96 + i / (pattern.length - 1) * 565;
  const y = value => zero - value / range * (bottom - top) / 2;
  const indexed = text => ({ text, math: true, sub: index });
  label(parent, 360, 36, 'Scale the unit pattern into the score column', colors.ink, 21, 'middle');
  label(parent, 360, 79, [indexed('p'), ' → ', indexed('z')], color, 24, 'middle');
  line(parent, [96, top], [96, bottom], colors.rule);
  line(parent, [96, zero], [661, zero], colors.rule);
  for (const tick of [-range, 0, range]) {
    line(parent, [92, y(tick)], [100, y(tick)], colors.rule);
    label(parent, 81, y(tick) + 6, String(tick).replace('-', '−'), colors.muted, 15, 'end');
  }
  pattern.forEach((value, i) => {
    line(parent, [x(i), zero], [x(i), y(value)], colors.light, 1);
    add(parent, 'circle', { cx: x(i), cy: y(value), r: 4, fill: '#fff', stroke: colors.rule, 'stroke-width': 1.2 });
    line(parent, [x(i), zero], [x(i), y(multiplier * value)], color, 1.3);
    add(parent, 'circle', { cx: x(i), cy: y(multiplier * value), r: 4.3, fill: color, stroke: '#fff', 'stroke-width': .7 });
  });
  for (const i of [0, 11, 23]) label(parent, x(i), 364, String(i + 1), colors.muted, 15, 'middle');
  label(parent, 360, 395, 'Observation index · fixed vertical scale', colors.muted, 17, 'middle');
  label(parent, 360, 435, ['Length: 1 → ', fmt(multiplier)], color, 21, 'middle');
}
