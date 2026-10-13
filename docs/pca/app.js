import {
  LINE_POINTS, POINTS, THREE_D_POINTS, THREE_D_BASIS,
  pcaSummary, coordinates, changeBasis, reconstructBasis, sampleVariance,
} from './model.mjs';

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
let animation;
let scrollFrame;
let navigationUntil = 0;

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
function basis(parent, map, u, v, length, second = true) {
  const origin = map([0, 0]);
  const end = (direction, sign = 1) => map(direction.map(value => value * length * sign));
  line(parent, end(u, -1), end(u), colors.blue, 1.3);
  arrow(parent, origin, map(u), colors.blue);
  label(parent, end(u)[0] + 12, end(u)[1] - 5, 'u', colors.blue, 21);
  if (second) {
    line(parent, end(v, -1), end(v), colors.rust, 1.3, '6 5');
    arrow(parent, origin, map(v), colors.rust, '6 5');
    label(parent, end(v)[0] + 10, end(v)[1] - 6, 'v', colors.rust, 21);
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
function varianceCurve(parent, summary, points) {
  const x = 70, y = 414, width = 570, height = 66;
  const max = summary.totalVariance;
  label(parent, x, y - height - 19, 'Score variance as the direction turns', colors.ink, 17);
  line(parent, [x, y], [x + width, y], colors.rule);
  line(parent, [x, y], [x, y - height], colors.rule);
  label(parent, x - 10, y + 5, '0', colors.muted, 14, 'end');
  label(parent, x - 10, y - height + 5, fmt(max, 0), colors.muted, 14, 'end');
  const coords = [];
  for (let theta = 0; theta <= 180; theta += 2) coords.push(`${x + theta / 180 * width},${y - pcaSummary(theta, points).variance / max * height}`);
  add(parent, 'polyline', { points: coords.join(' '), fill: 'none', stroke: colors.blue, 'stroke-width': 2 });
  for (const tick of [0, 45, 90, 135, 180]) label(parent, x + tick / 180 * width, y + 23, `${tick}°`, colors.muted, 14, 'middle');
  const currentX = x + angle / 180 * width;
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

function draw3D(parent) {
  const scene = steps[current].dataset.scene;
  const scores = changeBasis(THREE_D_POINTS, THREE_D_BASIS);
  const isCoordinates = view === 'coordinates';
  const isReconstruction = scene === 'plane' || view === 'reconstruction';
  const displayed = isCoordinates ? scores : THREE_D_POINTS;
  const reconstructed = reconstructBasis(scores, THREE_D_BASIS, components);
  const yaw = camera * Math.PI / 180;
  const pitch = 22 * Math.PI / 180;
  const scale = 31;
  const project3D = ([x, y, z]) => [
    360 + scale * (x * Math.cos(yaw) - y * Math.sin(yaw)),
    215 - scale * ((x * Math.sin(yaw) + y * Math.cos(yaw)) * Math.sin(pitch) + z * Math.cos(pitch)),
  ];
  const depth = ([x, y, z]) => (x * Math.sin(yaw) + y * Math.cos(yaw)) * Math.cos(pitch) - z * Math.sin(pitch);
  const cols = [0, 1, 2].map(j => THREE_D_BASIS.map(row => row[j]));
  label(parent, 360, 32, isCoordinates ? 'Principal coordinates · Z = XU' : 'Original coordinates · three variables', colors.ink, 21, 'middle');
  if (isReconstruction && components === 2) {
    const corners = [[-5.9, -2], [5.9, -2], [5.9, 2], [-5.9, 2]].map(([a, b]) => cols[0].map((v, i) => a * v + b * cols[1][i]));
    add(parent, 'polygon', { points: corners.map(p => project3D(p).join(',')).join(' '), fill: colors.blue, 'fill-opacity': .07, stroke: colors.blue, 'stroke-width': 1 });
  }
  const axisNames = isCoordinates ? ['z₁', 'z₂', 'z₃'] : ['x₁', 'x₂', 'x₃'];
  [0, 1, 2].forEach((j) => {
    const d = [0, 0, 0]; d[j] = 5.7;
    const end = project3D(d);
    line(parent, project3D(d.map(v => -v)), end, colors.light, 1);
    arrow(parent, project3D([0, 0, 0]), end, colors.rule);
    label(parent, end[0] + 10, end[1] + 5, axisNames[j], colors.muted, 19);
  });
  const axisColors = [colors.blue, colors.rust, colors.teal];
  if (!isCoordinates) cols.forEach((u, j) => {
    const end = project3D(u.map(v => v * (j === 0 ? 6 : 3)));
    const dash = ['', '6 5', '2 4'][j];
    line(parent, project3D(u.map(v => v * (j === 0 ? -6 : -3))), end, axisColors[j], 1.3, dash);
    arrow(parent, project3D([0, 0, 0]), project3D(u), axisColors[j], dash);
    label(parent, end[0] + 12, end[1] - 7, `u${['₁', '₂', '₃'][j]}`, axisColors[j], 21);
  });
  if (isReconstruction && components === 1) line(parent, project3D(cols[0].map(v => -6 * v)), project3D(cols[0].map(v => 6 * v)), colors.blue, 2);
  displayed.map((p, i) => ({ p, i })).sort((a, b) => depth(b.p) - depth(a.p)).forEach(({ p, i }) => {
    if (isReconstruction) {
      line(parent, project3D(p), project3D(reconstructed[i]), colors.rule, 1, '3 3');
      point(parent, project3D(p), false, true, 4);
      cross(parent, project3D(reconstructed[i]));
    } else point(parent, project3D(p), false, false, 4.5);
  });
  label(parent, 360, 419, isReconstruction ? `○ Original     × Reconstruction · ${components} component${components > 1 ? 's' : ''} kept` : '3D cloud · turning the view preserves variance', colors.muted, 18, 'middle');
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
  const isExplore = scene === 'explore';
  const points = dataset === 'line' ? LINE_POINTS : POINTS;
  const summary = pcaSummary(angle, points);
  $('angle-control').hidden = isThree || current === 0;
  $('data-control').hidden = !isExplore;
  $('view-control').hidden = !(scene === 'three-d' || isExplore);
  $('components-control').hidden = !(scene === 'plane' || (isExplore && isThree && view === 'reconstruction'));
  $('camera-control').hidden = !isThree;
  $('statistics').hidden = current < 4;
  $('angle').value = targetAngle;
  $('angle-value').value = `${fmt(angle, 0)}°`;
  $('dataset').value = dataset;
  $('view').value = view;
  $('components').value = components;
  $('camera-value').value = `${camera}°`;
  $('camera').value = camera;
  document.querySelector('.stage').classList.toggle('is-three', isThree);
  document.querySelector('.stage').classList.toggle('is-explore', isExplore);
  document.querySelectorAll('[data-angle]').forEach(button => {
    button.setAttribute('aria-pressed', String(Math.abs(targetAngle - Number(button.dataset.angle)) < .1));
    const degrees = Number(button.dataset.angle);
    button.textContent = current >= 6 && degrees !== 0 ? `${degrees}° · ${degrees === 45 ? 'PC1' : 'PC2'}` : `${degrees}°`;
  });
  $('variance-label').textContent = isThree ? 'Var(PC1)' : 'Var(Xu)';
  $('second-label').textContent = isThree ? 'Var(PC2)' : 'Var(Xv)';
  let caption;
  if (isThree) {
    const three = threeSummary();
    const isReconstruction = scene === 'plane' || view === 'reconstruction';
    $('variance').value = fmt(three.variances[0]);
    $('second-variance').value = fmt(three.variances[1]);
    $('third-label').textContent = isReconstruction ? 'Variance kept' : 'Var(PC3)';
    $('third-value').value = isReconstruction ? `${fmt(three.retained * 100, 1)}%` : fmt(three.variances[2]);
    caption = isReconstruction
      ? `${components} component${components > 1 ? 's' : ''} retain${components > 1 ? '' : 's'} ${fmt(100 * three.retained, 1)}% of total variance ${fmt(three.total)}. Dropping PC3 discards a new coordinate, not the original variable x₃.`
      : 'Three perpendicular principal directions. Their score variances are 9, 1, and 0.25; the full basis preserves all the observations.';
  } else {
    const isReconstruction = scene === 'reconstruction' || (isExplore && view === 'reconstruction');
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
  if (isThree) draw3D(drawing); else draw2D(drawing, points, summary);
}

function remember() {
  saved.set(current, { angle: targetAngle, dataset, view, components, camera });
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
  const fallback = {
    angle: index === 0 || index === 4 ? 0 : index < 6 ? 30 : 45,
    dataset: index < 8 ? 'line' : index < 10 || index === 12 ? 'cloud' : 'three',
    view: index === 11 ? 'reconstruction' : 'original',
    components: index === 11 ? 2 : 1,
    camera: 25,
  };
  const state = saved.get(index) || fallback;
  dataset = state.dataset;
  view = state.view;
  components = state.components;
  camera = state.camera;
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
  if (performance.now() < navigationUntil) return;
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
$('angle').addEventListener('change', () => { $('status').textContent = $('caption').textContent; });
document.querySelectorAll('[data-angle]').forEach(button => button.addEventListener('click', () => {
  changeAngle(Number(button.dataset.angle), true);
  remember();
  $('status').textContent = `Direction set to ${targetAngle} degrees.`;
}));
for (const id of ['dataset', 'view', 'components', 'camera']) {
  $(id).addEventListener(id === 'camera' ? 'input' : 'change', event => {
    const value = event.target.value;
    if (id === 'dataset') { dataset = value; view = 'original'; components = dataset === 'three' ? 2 : 1; }
    if (id === 'view') view = value;
    if (id === 'components') components = Number(value);
    if (id === 'camera') camera = Number(value);
    remember(); render();
    if (id !== 'camera') $('status').textContent = $('caption').textContent;
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
});
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
document.fonts.ready.then(() => { if (location.hash) restoreHash(); });
