import {
  LINE_POINTS, POINTS, THREE_D_POINTS, THREE_D_BASIS, THREE_D_RESIDUALS,
  THREE_D_RESIDUAL_COORDINATES, svdExample,
  pcaSummary, coordinates, changeBasis, reconstructBasis, sampleVariance,
} from './model.mjs';
import { drawSVDFactors } from './svd-view.mjs';

const $ = id => document.getElementById(id);
const steps = [...document.querySelectorAll('.step')];
const svg = $('diagram');
const colors = { ink: '#242424', blue: '#234E70', rust: '#A54F32', teal: '#007F73', muted: '#606060', rule: '#767676', light: '#d6d6d6' };
const motion = matchMedia('(prefers-reduced-motion: reduce)');
const saved = new Map();
let current = 0;
let angle = 0;
let targetAngle = 0;
let dataset = 'line';
let view = 'original';
let components = 2;
let camera = 25;
let removal = 0;
let piece = 0;
// Rotate the face-on reference frame away from the known principal axes.
const residualPoints = coordinates(THREE_D_RESIDUAL_COORDINATES, 30);
let animation;
let scrollFrame;
let navigationUntil = 0;
let navigationLocked = false;
let initializing = true;

function fmt(value, digits = 2) {
  return (Math.abs(value) < .000001 ? 0 : value).toFixed(digits).replace('-', '−');
}
function el(name, attributes = {}, text) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}
function add(parent, name, attributes, text) {
  const node = el(name, attributes, text);
  parent.append(node);
  return node;
}
function line(parent, a, b, color = colors.rule, width = 1, dash = '') {
  return add(parent, 'line', { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: color, 'stroke-width': width, ...(dash ? { 'stroke-dasharray': dash } : {}) });
}
function label(parent, x, y, text, color = colors.muted, size = 18, anchor = 'start') {
  const compact = matchMedia('(max-width: 800px)').matches;
  return add(parent, 'text', { x, y, fill: color, 'font-size': compact ? Math.max(28, size * 1.55) : size, 'text-anchor': anchor }, text);
}
function arrow(parent, a, b, color, dash = '') {
  line(parent, a, b, color, 2.2, dash);
  const theta = Math.atan2(b[1] - a[1], b[0] - a[0]);
  const wing = offset => [b[0] - 9 * Math.cos(theta + offset), b[1] - 9 * Math.sin(theta + offset)];
  line(parent, wing(.45), b, color, 2.2);
  line(parent, wing(-.45), b, color, 2.2);
}
function point(parent, position, selected = false, hollow = false, radius = 5) {
  add(parent, 'circle', { cx: position[0], cy: position[1], r: radius, fill: hollow ? '#fff' : colors.blue, stroke: hollow ? colors.rule : colors.blue, 'stroke-width': hollow ? 1.4 : .7 });
  if (selected) add(parent, 'circle', { cx: position[0], cy: position[1], r: radius + 5, fill: 'none', stroke: colors.rust, 'stroke-width': 2 });
}
function cross(parent, position, color = colors.blue) {
  line(parent, [position[0] - 4, position[1] - 4], [position[0] + 4, position[1] + 4], color, 2);
  line(parent, [position[0] - 4, position[1] + 4], [position[0] + 4, position[1] - 4], color, 2);
}
function plot(parent, cx, cy, scale, extent, names = ['x₁', 'x₂'], ticks = true) {
  const map = p => [cx + scale * p[0], cy - scale * p[1]];
  line(parent, map([-extent, 0]), map([extent, 0]), colors.rule);
  line(parent, map([0, -extent]), map([0, extent]), colors.rule);
  label(parent, cx + scale * extent - 4, cy + 24, names[0], colors.muted, 19, 'end');
  label(parent, cx + 9, cy - scale * extent + 8, names[1], colors.muted, 19);
  if (ticks) {
    const values = extent < 4 ? [-2, -1, 1, 2] : [-4, -2, 2, 4];
    for (const v of values) {
      line(parent, map([v, -.065]), map([v, .065]), colors.rule);
      line(parent, map([-.065, v]), map([.065, v]), colors.rule);
      label(parent, cx + scale * v, cy + 21, String(v).replace('-', '−'), colors.muted, 14, 'middle');
      label(parent, cx - 9, cy - scale * v + 5, String(v).replace('-', '−'), colors.muted, 14, 'end');
    }
  }
  return map;
}
function basis(parent, map, u, v, length, second = true, names = ['u', 'v']) {
  const origin = map([0, 0]);
  const end = (direction, sign = 1) => map(direction.map(value => value * length * sign));
  line(parent, end(u, -1), end(u), colors.blue, 1.3);
  arrow(parent, origin, map(u), colors.blue);
  const firstLabel = names[0] === 'w' ? map(u) : end(u);
  label(parent, firstLabel[0] + 12, firstLabel[1] - 5, names[0], colors.blue, 21);
  if (second) {
    line(parent, end(v, -1), end(v), colors.rust, 1.3, '6 5');
    arrow(parent, origin, map(v), colors.rust, '6 5');
    const secondLabel = names[0] === 'w' ? map(v) : end(v);
    label(parent, secondLabel[0] + 10, secondLabel[1] - 6, names[1], colors.rust, 21);
  }
}
function scoreStrip(parent, scores, y, title) {
  const cx = 360;
  const scale = 45;
  label(parent, 36, y - 61, title, colors.ink, 18);
  line(parent, [cx - 6 * scale, y], [cx + 6 * scale, y], colors.rule);
  for (const tick of [-6, -4, -2, 0, 2, 4, 6]) {
    line(parent, [cx + tick * scale, y - 4], [cx + tick * scale, y + 4], colors.rule);
    label(parent, cx + tick * scale, y + 24, String(tick).replace('-', '−'), colors.muted, 15, 'middle');
  }
  scores.forEach((score, index) => {
    // A small vertical stack reveals coincident scores without changing their x position.
    const repeated = scores.slice(0, index).filter(other => Math.abs(other - score) < .001).length;
    point(parent, [cx + score * scale, y - repeated * 10], index === scores.length - 1, false, scores.length > 5 ? 3.5 : 5);
  });
}
function varianceCurve(parent, summary, points, min = 0, maxAngle = 180) {
  const x = 70, y = 414, width = 570, height = 66;
  const max = summary.totalVariance;
  label(parent, x, y - height - 19, 'Score variance as the direction turns', colors.ink, 17);
  line(parent, [x, y], [x + width, y], colors.rule);
  line(parent, [x, y], [x, y - height], colors.rule);
  label(parent, x - 10, y + 5, '0', colors.muted, 14, 'end');
  label(parent, x - 10, y - height + 5, fmt(max, max < 2 ? 2 : 0), colors.muted, 14, 'end');
  const coords = [];
  for (let theta = min; theta <= maxAngle; theta += 2) coords.push(`${x + (theta - min) / (maxAngle - min) * width},${y - pcaSummary(theta, points).variance / max * height}`);
  add(parent, 'polyline', { points: coords.join(' '), fill: 'none', stroke: colors.blue, 'stroke-width': 2 });
  for (const tick of (min === 0 ? [0, 45, 90, 135, 180] : [-90, -30, 0, 60, 90])) label(parent, x + (tick - min) / (maxAngle - min) * width, y + 23, `${tick}°`, colors.muted, 14, 'middle');
  const currentX = x + (angle - min) / (maxAngle - min) * width;
  const currentY = y - summary.variance / max * height;
  line(parent, [currentX, y], [currentX, currentY], colors.rust, 1, '3 4');
  point(parent, [currentX, currentY], true, false, 4);
}
function draw2D(parent, points, summary) {
  const scene = steps[current].dataset.scene;
  const isPair = scene === 'coordinates' || (scene === 'explore' && view === 'coordinates');
  const isReconstruction = scene === 'reconstruction' || (scene === 'explore' && view === 'reconstruction');
  const isCurve = ['maximum', 'cloud', 'reconstruction', 'explore'].includes(scene) && !isPair;
  const isStrip = ['projection', 'variance', 'quadratic'].includes(scene);
  const isScores = scene === 'scores';
  const showBasis = current > 0;
  const extent = points === LINE_POINTS ? 3.1 : 6.2;
  const selected = points.length - 1;
  if (isPair) {
    label(parent, 178, 34, 'Original coordinates', colors.ink, 20, 'middle');
    label(parent, 540, 34, 'New coordinates · XU', colors.ink, 20, 'middle');
    const scale = points === LINE_POINTS ? 43 : 23;
    const left = plot(parent, 178, 226, scale, extent);
    const right = plot(parent, 540, 226, scale, extent, ['z₁', 'z₂']);
    basis(parent, left, summary.u, summary.v, extent - .4);
    points.forEach((p, i) => point(parent, left(p), i === selected, false, points.length > 5 ? 4 : 5));
    coordinates(points, angle).forEach((p, i) => point(parent, right(p), i === selected, false, points.length > 5 ? 4 : 5));
    label(parent, 360, 415, `Ringed point: (${fmt(points[selected][0])}, ${fmt(points[selected][1])}) → (${fmt(summary.scores[selected])}, ${fmt(summary.otherScores[selected])})`, colors.ink, 18, 'middle');
    return;
  }
  const cx = isScores ? 233 : 360;
  const cy = isCurve || isStrip ? 174 : 220;
  const scale = points === LINE_POINTS ? (isCurve || isStrip ? 43 : 52) : 22;
  const map = plot(parent, cx, cy, scale, extent);
  if (points === LINE_POINTS) {
    line(parent, map([-2.7, -2.7]), map([2.7, 2.7]), colors.light, 1.5, '4 5');
    if (!showBasis) label(parent, map([1.6, 2.4])[0], map([1.6, 2.4])[1], 'x₁ = x₂', colors.ink, 21);
  }
  if (showBasis) basis(parent, map, summary.u, summary.v, extent - .3, !isReconstruction);
  if (isReconstruction) {
    points.forEach((p, i) => {
      line(parent, map(p), map(summary.reconstruction[i]), colors.rule, 1, '3 3');
      point(parent, map(p), i === selected, true, 4);
      cross(parent, map(summary.reconstruction[i]));
    });
    label(parent, 70, 305, '○ Original observation     × Reconstructed from one score', colors.muted, 17);
  } else {
    if (isStrip || isScores) {
      points.forEach((p, i) => {
        if (scene !== 'projection' || i === selected) {
          line(parent, map(p), map(summary.reconstruction[i]), colors.rule, 1, '3 4');
          if (scene === 'projection') cross(parent, map(summary.reconstruction[i]), colors.rust);
        }
      });
    }
    points.forEach((p, i) => point(parent, map(p), current >= 2 && i === selected, false, points.length > 5 ? 4 : 5));
  }
  if (scene === 'basis') label(parent, 360, 430, 'The axes turn. The observations stay fixed.', colors.muted, 20, 'middle');
  if (isStrip) scoreStrip(parent, summary.scores, 386, 'One score per observation · Xu');
  if (isScores) {
    label(parent, 565, 75, 'Xu', colors.blue, 28, 'middle');
    add(parent, 'path', { d: 'M514 98H502V327H514M614 98H626V327H614', stroke: colors.ink, 'stroke-width': 1.3, fill: 'none' });
    summary.scores.forEach((value, i) => label(parent, 564, 131 + i * 44, fmt(value), i === selected ? colors.rust : colors.ink, 23, 'middle'));
    label(parent, 360, 428, 'A direction has 2 entries. Its score column has 5.', colors.muted, 19, 'middle');
  }
  if (isCurve) varianceCurve(parent, summary, points);
}

function drawResidualPlane(parent, summary) {
  label(parent, 360, 32, 'Residual plane · viewed face on', colors.ink, 21, 'middle');
  const map = plot(parent, 360, 179, 57, 2.35, ['a', 'b']);
  basis(parent, map, summary.u, summary.v, 2.2, true, ['w', 'w⊥']);
  residualPoints.forEach(p => point(parent, map(p), false, false, 4.5));
  varianceCurve(parent, summary, residualPoints, -90, 90);
}

function draw3D(parent) {
  const scene = steps[current].dataset.scene;
  const scores = changeBasis(THREE_D_POINTS, THREE_D_BASIS);
  const isCoordinates = scene === 'explore' && view === 'coordinates';
  const isReconstruction = scene === 'plane' || (scene === 'explore' && view === 'reconstruction');
  const isDeflation = scene === 'deflation';
  const isPiece = scene === 'pieces';
  const cols = [0, 1, 2].map(j => THREE_D_BASIS.map(row => row[j]));
  let displayed = isCoordinates ? scores : THREE_D_POINTS;
  if (isDeflation) displayed = THREE_D_POINTS.map((p, i) => p.map((v, j) => (1 - removal) * v + removal * THREE_D_RESIDUALS[i][j]));
  if (isPiece) displayed = scores.map(row => cols[piece].map(v => row[piece] * v));
  const reconstructed = reconstructBasis(scores, THREE_D_BASIS, components);
  const yaw = camera * Math.PI / 180;
  const pitch = 22 * Math.PI / 180;
  const scale = 31;
  const project3D = ([x, y, z]) => [
    360 + scale * (x * Math.cos(yaw) - y * Math.sin(yaw)),
    215 - scale * ((x * Math.sin(yaw) + y * Math.cos(yaw)) * Math.sin(pitch) + z * Math.cos(pitch)),
  ];
  const depth = ([x, y, z]) => (x * Math.sin(yaw) + y * Math.cos(yaw)) * Math.cos(pitch) - z * Math.sin(pitch);
  const titles = {
    'three-d': 'Find PC1 · the direction of greatest variance',
    deflation: 'Subtract the projection along PC1',
    pieces: `The PC${piece + 1} contribution · one piece of X`,
  };
  label(parent, 360, 32, titles[scene] || (isCoordinates ? 'Principal coordinates · Z = XU' : isReconstruction ? 'Reconstruct the original observations' : 'Original cloud · three coordinates'), colors.ink, 21, 'middle');
  if (isDeflation || (isReconstruction && components === 2)) {
    const axes = isDeflation ? [cols[1], cols[2]] : [cols[0], cols[1]];
    const extents = isDeflation ? [2.8, 2.3] : [5.9, 2];
    const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, b]) => axes[0].map((v, i) => a * extents[0] * v + b * extents[1] * axes[1][i]));
    add(parent, 'polygon', { points: corners.map(p => project3D(p).join(',')).join(' '), fill: colors.blue, 'fill-opacity': .07, stroke: colors.blue, 'stroke-width': 1 });
  }
  const axisNames = isCoordinates ? ['z₁', 'z₂', 'z₃'] : ['x₁', 'x₂', 'x₃'];
  [0, 1, 2].forEach(j => {
    const d = [0, 0, 0]; d[j] = 5.7;
    const end = project3D(d);
    line(parent, project3D(d.map(v => -v)), end, colors.light, 1);
    label(parent, end[0] + 10, end[1] + 5, axisNames[j], colors.muted, 19);
  });
  const axisColors = [colors.blue, colors.rust, colors.teal];
  if (!isCoordinates) cols.forEach((u, j) => {
    if ((['three-d', 'deflation'].includes(scene) && j !== 0) || (isPiece && j !== piece)) return;
    const end = project3D(u.map(v => v * (j === 0 ? 6 : 3)));
    const dash = ['', '6 5', '2 4'][j];
    line(parent, project3D(u.map(v => v * (j === 0 ? -6 : -3))), end, axisColors[j], 1.3, dash);
    arrow(parent, project3D([0, 0, 0]), project3D(u), axisColors[j], dash);
    label(parent, end[0] + 12, end[1] - 7, `u${['₁', '₂', '₃'][j]}`, axisColors[j], 21);
  });
  if (isDeflation || isPiece) {
    THREE_D_POINTS.forEach((p, i) => {
      if (isDeflation) line(parent, project3D(p), project3D(displayed[i]), colors.rule, 1, '3 4');
      point(parent, project3D(p), false, true, 4);
    });
  }
  displayed.map((p, i) => ({ p, i })).sort((a, b) => depth(b.p) - depth(a.p)).forEach(({ p, i }) => {
    if (isReconstruction) {
      line(parent, project3D(p), project3D(reconstructed[i]), colors.rule, 1, '3 3');
      point(parent, project3D(p), false, true, 4);
      cross(parent, project3D(reconstructed[i]));
    } else if (isPiece) cross(parent, project3D(p), axisColors[piece]);
    else point(parent, project3D(p), false, false, 4.5);
  });
  const legend = isDeflation ? '○ Original     ● After subtraction' : isPiece ? '○ Original     × Selected contribution' : isReconstruction ? `○ Original     × Reconstruction · ${components} kept` : scene === 'three-d' ? 'PC1 variance = 9 · total variance = 10.25' : 'Turning the camera preserves score variance';
  label(parent, 360, 401, legend, colors.muted, 18, 'middle');
  if (isDeflation) label(parent, 360, 435, 'Shaded plane: perpendicular to PC1', colors.ink, 18, 'middle');
  if (scene === 'plane' && components === 2) label(parent, 360, 435, 'Shaded plane: spanned by PC1 and PC2', colors.ink, 18, 'middle');
}

function threeSummary() {
  const scores = changeBasis(THREE_D_POINTS, THREE_D_BASIS);
  const variances = [0, 1, 2].map(j => sampleVariance(scores.map(row => row[j])));
  const total = variances.reduce((a, b) => a + b, 0);
  const retained = variances.slice(0, components).reduce((a, b) => a + b, 0) / total;
  return { variances, total, retained };
}
function render() {
  const scene = steps[current].dataset.scene;
  const isThree = dataset === 'three';
  const isResidual = ['residual-plane', 'second-pc'].includes(scene);
  const isSVD = scene === 'svd';
  const isExplore = scene === 'explore';
  const points = isResidual ? residualPoints : dataset === 'line' ? LINE_POINTS : POINTS;
  const summary = pcaSummary(angle, points);
  $('angle-control').hidden = (isThree && !isResidual) || current === 0;
  $('data-control').hidden = !isExplore;
  $('view-control').hidden = !isExplore;
  $('components-control').hidden = !(scene === 'plane' || (isExplore && isThree && view === 'reconstruction'));
  $('camera-control').hidden = !isThree || isResidual || isSVD;
  $('removal-control').hidden = scene !== 'deflation';
  $('piece-control').hidden = !['pieces', 'svd'].includes(scene);
  $('statistics').hidden = current < 4;
  $('angle').min = isResidual ? -90 : 0;
  $('angle').max = isResidual ? 90 : 180;
  $('angle').value = targetAngle;
  $('angle-label').textContent = isResidual ? 'Direction in the plane φ' : 'Direction angle θ';
  $('angle-value').value = `${fmt(angle, 0)}°`;
  $('dataset').value = dataset;
  $('view').value = view;
  $('components').value = components;
  $('camera-value').value = `${camera}°`;
  $('camera').value = camera;
  $('removal-value').value = `${Math.round(removal * 100)}%`;
  $('removal').value = removal * 100;
  $('piece').value = piece;
  document.querySelector('.stage').classList.toggle('is-three', isThree);
  document.querySelector('.stage').classList.toggle('is-explore', isExplore);
  document.querySelectorAll('[data-angle]').forEach((button, index) => {
    const degrees = (isResidual ? [-30, 0, 60] : [0, 45, 135])[index];
    button.dataset.angle = degrees;
    button.setAttribute('aria-pressed', String(Math.abs(targetAngle - degrees) < .1));
    button.textContent = `${String(degrees).replace('-', '−')}°${isResidual ? (index === 0 ? ' · PC2' : index === 2 ? ' · PC3' : '') : current >= 6 && degrees !== 0 ? ` · PC${index}` : ''}`;
  });
  let caption;
  function statistics(labels, values) {
    ['variance-label', 'second-label', 'third-label'].forEach((id, i) => { $(id).textContent = labels[i]; });
    ['variance', 'second-variance', 'third-value'].forEach((id, i) => { $(id).value = values[i]; });
  }
  if (isResidual) {
    statistics(['Var(Rw)', 'Other direction', 'Residual total'], [fmt(summary.variance), fmt(summary.otherVariance), '1.25']);
    caption = `The fully deflated cloud viewed face on. At φ = ${fmt(angle, 0)}°, score variance is ${fmt(summary.variance)}; perpendicular variance is ${fmt(summary.otherVariance)}. The maximum is 1 at −30° (PC2); PC3 is perpendicular at 60°, with variance 0.25.`;
  } else if (scene === 'deflation') {
    const left = 9 * (1 - removal) ** 2;
    statistics(['PC1 variance left', 'In the plane', 'Remaining total'], [fmt(left), '1.25', fmt(left + 1.25)]);
    caption = `${Math.round(removal * 100)}% of each PC1 score is subtracted. The remaining PC1 variance is ${fmt(left)}. At 100%, every point lies in the plane perpendicular to PC1; its total variance is 1.25.`;
  } else if (isSVD) {
    const { singularValues } = svdExample();
    const d = singularValues[piece];
    statistics([`Length d${['₁', '₂', '₃'][piece]}`, 'Length of pⱼ', 'Score variance'], [fmt(d), '1.00', fmt(d * d / 23)]);
    caption = `PC${piece + 1}: zⱼ = dⱼpⱼ. The score column has length ${fmt(d)}; its normalized pattern has length 1. Their plots have different labelled vertical scales. Squared length divided by 23 gives score variance ${fmt(d * d / 23)}.`;
  } else if (isThree) {
    const three = threeSummary();
    const isReconstruction = scene === 'plane' || (isExplore && view === 'reconstruction');
    if (scene === 'pieces') {
      statistics(['Piece variance', 'Original total', 'Share of total'], [fmt(three.variances[piece]), '10.25', `${fmt(three.variances[piece] / three.total * 100, 1)}%`]);
      caption = `Crosses show each observation’s PC${piece + 1} contribution zᵢ${['₁', '₂', '₃'][piece]}u${['₁', '₂', '₃'][piece]}. It lies along one direction and contributes variance ${fmt(three.variances[piece])}. The three pieces add to the original hollow-circle observations.`;
    } else if (scene === 'three-d') {
      statistics(['Var(Xu₁)', 'Remaining', 'Original total'], ['9.00', '1.25', '10.25']);
      caption = 'Begin with PC1 alone: u₁ = (1, 1, 1)ᵀ/√3, with score variance 9. The original total is 10.25, so 1.25 remains in directions perpendicular to PC1.';
    } else {
      statistics(['Var(PC1)', 'Var(PC2)', isReconstruction ? 'Variance kept' : 'Var(PC3)'], ['9.00', '1.00', isReconstruction ? `${fmt(three.retained * 100, 1)}%` : '0.25']);
      caption = isReconstruction
        ? `${components} component${components > 1 ? 's' : ''} retain${components > 1 ? '' : 's'} ${fmt(100 * three.retained, 1)}% of original variance 10.25. ${components === 1 ? 'The retained line follows PC1.' : components === 2 ? 'The retained plane contains PC1 and PC2; the residual plane from subtraction was perpendicular to PC1.' : 'All three components recover the original points.'}`
        : 'Three perpendicular principal directions. Their score variances are 9, 1, and 0.25; the full basis preserves all the observations.';
    }
  } else {
    const isReconstruction = scene === 'reconstruction' || (isExplore && view === 'reconstruction');
    $('variance-label').textContent = 'Var(Xu)';
    $('second-label').textContent = 'Var(Xv)';
    $('variance').value = fmt(summary.variance);
    $('second-variance').value = fmt(summary.otherVariance);
    $('third-label').textContent = isReconstruction ? 'Variance kept' : 'Total variance';
    $('third-value').value = isReconstruction ? `${fmt(summary.retainedFraction * 100, 1)}%` : fmt(summary.totalVariance);
    const captions = {
      line: 'Five centered observations lie exactly on x₁ = x₂. Both axes use the same unit scale.',
      basis: `u = (${fmt(summary.u[0])}, ${fmt(summary.u[1])})ᵀ. Solid u and dashed v are perpendicular unit directions.`,
      projection: `The ringed point has score ${fmt(summary.scores.at(-1))}. The dashed perpendicular ends at its projection onto u.`,
      scores: `The five entries of Xu are ${summary.scores.map(value => fmt(value)).join(', ')}. Each entry measures one observation along u.`,
      variance: `The scores have mean zero. Their squared values sum to ${fmt(summary.variance * (points.length - 1))}; divide by n − 1 = ${points.length - 1} to get ${fmt(summary.variance)}.`,
      quadratic: `Variance from the scores and from uᵀSu both equal ${fmt(summary.variance)}. The sample covariance matrix has all four entries equal to 2.5.`,
      maximum: `The greatest variance is 5 at 45°. The perpendicular direction at 135° has variance 0. The total stays 5.`,
      coordinates: `Both coordinates preserve every point. Cov(Xu, Xv) = ${fmt(summary.crossCovariance)}. At 45°, the second score is zero for everyone.`,
      cloud: `A new cloud of 24 points. Cov(Xu, Xv) = ${fmt(summary.crossCovariance)}. At the principal basis, the score variances are 9 and 1.`,
      reconstruction: `One direction keeps ${fmt(100 * summary.retainedFraction, 1)}% of total variance. Squared reconstruction errors sum to ${fmt(summary.meanSquaredError * points.length)}.`,
      explore: isReconstruction
        ? `One coordinate keeps ${fmt(100 * summary.retainedFraction, 1)}% of total variance. Reconstruction SSE = ${fmt(summary.meanSquaredError * points.length)}.`
        : `Var(Xu) = ${fmt(summary.variance)}; Var(Xv) = ${fmt(summary.otherVariance)}; covariance = ${fmt(summary.crossCovariance)}. Total variance stays ${fmt(summary.totalVariance)}.`,
    };
    caption = captions[scene];
  }
  $('caption').textContent = caption;
  if (scene === 'projection') $('one-score-worked').textContent = `At ${fmt(angle, 0)}°, z = 2 cos ${fmt(angle, 0)}° + 2 sin ${fmt(angle, 0)}° ≈ ${fmt(2 * summary.u[0] + 2 * summary.u[1])}.`;
  svg.replaceChildren(el('title', { id: 'diagram-title' }, steps[current].dataset.title), el('desc', { id: 'diagram-description' }, caption));
  const drawing = add(svg, 'g', { 'aria-hidden': 'true' });
  if (isSVD) drawSVDFactors(drawing, { add, line, label, point, colors, fmt }, piece);
  else if (isResidual) drawResidualPlane(drawing, summary);
  else if (isThree) draw3D(drawing);
  else draw2D(drawing, points, summary);
}

function remember() {
  saved.set(current, { angle: targetAngle, dataset, view, components, camera, removal, piece });
}
function changeAngle(value, animate = false) {
  cancelAnimationFrame(animation);
  targetAngle = value;
  if (!animate || motion.matches) {
    angle = value;
    render();
    return;
  }
  const from = angle;
  const start = performance.now();
  // Update scene layout before navigation measures the sticky figure.
  render();
  function frame(time) {
    const t = Math.min(1, (time - start) / 450);
    const eased = t * t * (3 - 2 * t);
    angle = from + (value - from) * eased;
    render();
    if (t < 1) animation = requestAnimationFrame(frame);
  }
  animation = requestAnimationFrame(frame);
}
function setStep(index, animate = true) {
  const changed = current !== index;
  if (changed) remember();
  current = index;
  const scene = steps[index].dataset.scene;
  const fallback = {
    angle: scene === 'residual-plane' ? 0 : scene === 'second-pc' ? -30 : index === 0 || index === 4 ? 0 : index < 6 ? 30 : 45,
    dataset: index < 8 ? 'line' : index < 10 || scene === 'explore' ? 'cloud' : 'three',
    view: scene === 'plane' ? 'reconstruction' : 'original',
    components: scene === 'plane' ? 2 : 1,
    camera: 25,
    removal: 0,
    piece: 0,
  };
  const state = saved.get(index) || fallback;
  dataset = state.dataset;
  view = state.view;
  components = state.components;
  camera = state.camera;
  removal = state.removal;
  piece = state.piece;
  $('step-select').value = index;
  $('step-count').textContent = `${String(index + 1).padStart(2, '0')} / ${steps.length}`;
  $('previous').disabled = index === 0;
  $('next').disabled = index === steps.length - 1;
  changeAngle(state.angle, animate && changed && dataset !== 'three');
}
function readingLine() {
  const stage = document.querySelector('.stage');
  const mobile = matchMedia('(max-width: 800px)').matches;
  const sticky = getComputedStyle(stage).position === 'sticky';
  return mobile && sticky ? parseFloat(getComputedStyle(stage).top) + stage.offsetHeight + 22 : Math.max(100, innerHeight * .38);
}
function goTo(index, push = true, focus = true) {
  index = Math.max(0, Math.min(steps.length - 1, index));
  navigationUntil = performance.now() + 100;
  navigationLocked = true;
  setStep(index);
  if (push && location.hash !== `#${steps[index].id}`) history.pushState(null, '', `#${steps[index].id}`);
  const mobile = matchMedia('(max-width: 800px)').matches;
  const alignment = mobile ? readingLine() : 84;
  const target = steps[index].getBoundingClientRect().top + scrollY - alignment + 5;
  window.scrollTo({ top: Math.max(0, target), behavior: 'instant' });
  if (focus) {
    const heading = steps[index].querySelector('h2');
    heading.tabIndex = -1;
    heading.focus({ preventScroll: true });
  }
}
function readScroll() {
  scrollFrame = undefined;
  if (initializing || navigationLocked || performance.now() < navigationUntil) return;
  const lineY = readingLine();
  let index = 0;
  steps.forEach((step, i) => { if (step.getBoundingClientRect().top <= lineY + 5) index = i; });
  if (index !== current) {
    setStep(index);
    if (scrollY > document.querySelector('.title-page').offsetHeight) history.replaceState(null, '', `#${steps[index].id}`);
  }
}
steps.forEach((step, index) => {
  const option = document.createElement('option');
  option.value = index;
  option.textContent = `${String(index + 1).padStart(2, '0')} · ${step.dataset.title}`;
  $('step-select').append(option);
});
$('step-select').addEventListener('change', event => goTo(Number(event.target.value), true, false));
$('previous').addEventListener('click', () => goTo(current - 1, true, false));
$('next').addEventListener('click', () => goTo(current + 1, true, false));
$('angle').addEventListener('input', event => { changeAngle(Number(event.target.value)); remember(); });
$('removal').addEventListener('change', () => { $('status').textContent = $('caption').textContent; });
$('angle').addEventListener('change', () => { $('status').textContent = $('caption').textContent; });
document.querySelectorAll('[data-angle]').forEach(button => button.addEventListener('click', () => {
  changeAngle(Number(button.dataset.angle), true);
  remember();
  $('status').textContent = `Direction set to ${targetAngle} degrees.`;
}));
for (const id of ['dataset', 'view', 'components', 'camera', 'removal', 'piece']) {
  $(id).addEventListener(['camera', 'removal'].includes(id) ? 'input' : 'change', event => {
    const value = event.target.value;
    if (id === 'dataset') { dataset = value; view = 'original'; components = dataset === 'three' ? 2 : 1; }
    if (id === 'view') view = value;
    if (id === 'components') components = Number(value);
    if (id === 'camera') camera = Number(value);
    if (id === 'removal') removal = Number(value) / 100;
    if (id === 'piece') piece = Number(value);
    remember(); render();
    if (!['camera', 'removal'].includes(id)) $('status').textContent = $('caption').textContent;
  });
}
document.querySelectorAll('a[href^="#"]').forEach(link => link.addEventListener('click', event => {
  const index = steps.findIndex(step => `#${step.id}` === link.getAttribute('href'));
  if (index >= 0) { event.preventDefault(); goTo(index); }
}));
document.addEventListener('keydown', event => {
  if (event.altKey || event.ctrlKey || event.metaKey || event.target.closest('input, select, textarea, button, [contenteditable]')) return;
  if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
    event.preventDefault(); goTo(current + (event.key === 'ArrowRight' ? 1 : -1));
  }
  if (['PageDown', 'PageUp', 'ArrowDown', 'ArrowUp', 'Home', 'End', ' '].includes(event.key)) navigationLocked = false;
});
// A selected chapter stays selected through browser focus/scroll anchoring.
// Intentional wheel, swipe, or page-key scrolling returns control to the story.
addEventListener('wheel', () => { navigationLocked = false; }, { passive: true });
addEventListener('touchmove', event => {
  if (!event.target.closest('input, select, button')) navigationLocked = false;
}, { passive: true });
addEventListener('pointerdown', event => {
  if (event.clientX >= document.documentElement.clientWidth) navigationLocked = false;
}, { passive: true });
addEventListener('scroll', () => { if (!scrollFrame) scrollFrame = requestAnimationFrame(readScroll); }, { passive: true });
addEventListener('resize', () => { render(); readScroll(); });
motion.addEventListener('change', () => { if (motion.matches) changeAngle(targetAngle); });
function restoreHash() {
  const index = steps.findIndex(step => `#${step.id}` === location.hash);
  if (index >= 0) goTo(index, false, false);
  else if (!location.hash || location.hash === '#page-title') window.scrollTo({ top: 0, behavior: 'instant' });
}
addEventListener('popstate', restoreHash);
setStep(0, false);
document.fonts.ready.then(() => { initializing = false; if (location.hash) restoreHash(); else readScroll(); });
