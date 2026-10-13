import { svdExample } from './model.mjs';

// Both columns contain the same ordered observations. Their separate, fixed
// vertical scales make the normalization visible without implying a time axis.
export function drawSVDFactors(parent, helpers, componentIndex) {
  const { add, line, label, colors, fmt } = helpers;
  const { scores, P, singularValues } = svdExample();
  const subscript = ['₁', '₂', '₃'][componentIndex];
  const color = [colors.blue, colors.rust, colors.teal][componentIndex];
  const singularValue = singularValues[componentIndex];
  const top = 147;
  const bottom = 298;
  const zero = (top + bottom) / 2;

  label(parent, 360, 36, 'Separate a score pattern from its size', colors.ink, 21, 'middle');

  function columnPlot({ left, right, values, range, ticks, title, scale, length }) {
    const center = (left + right) / 2;
    const x = index => left + index / (values.length - 1) * (right - left);
    const y = value => zero - value / range * (bottom - top) / 2;
    label(parent, center, 82, title, colors.ink, 20, 'middle');
    label(parent, center, 117, scale, colors.muted, 16, 'middle');
    line(parent, [left, top], [left, bottom], colors.rule, 1);
    line(parent, [left, zero], [right, zero], colors.rule, 1);

    ticks.forEach(tick => {
      const tickY = y(tick);
      line(parent, [left - 4, tickY], [left + 4, tickY], colors.rule, 1);
      label(parent, left - 11, tickY + 7,
        tick === 0 ? '0' : fmt(tick, range === 6 ? 0 : 1), colors.muted, 14, 'end');
    });

    values.forEach((value, index) => {
      const position = [x(index), y(value)];
      line(parent, [position[0], zero], position, color, 1.2);
      add(parent, 'circle', { cx: position[0], cy: position[1], r: 4.3,
        fill: color, stroke: '#fff', 'stroke-width': 0.8 });
    });
    [0, 11, values.length - 1].forEach(index => {
      line(parent, [x(index), bottom], [x(index), bottom + 4], colors.rule, 1);
      label(parent, x(index), 326, String(index + 1), colors.muted, 14, 'middle');
    });
    label(parent, center, 354, 'Observation index', colors.muted, 16, 'middle');
    label(parent, center, 392, length, color, 19, 'middle');
  }

  columnPlot({
    left: 98, right: 312,
    values: scores.map(row => row[componentIndex]),
    range: 6, ticks: [-6, 0, 6],
    title: `Scores z${subscript} = Xu${subscript}`,
    scale: 'Scale: −6 to 6',
    length: `Length d${subscript} = ${fmt(singularValue)}`,
  });
  columnPlot({
    left: 460, right: 674,
    values: P.map(row => row[componentIndex]),
    // PC3 reaches about 0.514, so ±0.6 gives every dot honest headroom.
    range: .6, ticks: [-.5, 0, .5],
    title: `Unit pattern p${subscript}`,
    scale: 'Scale: −0.6 to 0.6',
    length: 'Length = 1',
  });
  label(parent, 360, 435,
    `z${subscript} = ${fmt(singularValue)} p${subscript}`, colors.ink, 22, 'middle');
}
