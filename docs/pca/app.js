import {
  LINE_POINTS, POINTS, THREE_D_POINTS, THREE_D_BASIS, THREE_D_RESIDUALS,
  THREE_D_RESIDUAL_COORDINATES, svdExample,
  pcaSummary, coordinates, changeBasis, reconstructBasis, sampleVariance,
} from './model.mjs';
import { drawSVDFactors } from './svd-view.mjs?v=20261014-storyboard';
import { drawResidualTransition, drawThirdDirection } from './storyboard-3d.mjs?v=20261014-storyboard';
import { assembleComponents } from './storyboard-geometry.mjs?v=20261014-scroll';
import { scrollSceneAt, scrollLocalAtProgress } from './scroll-timeline.mjs?v=20261015-continuity';
import { mVar, mNum, mOp, mSub, mSup, mTranspose, mNorm, mTuple, mFunction, inlineMath, setNotation, notationText } from './notation.mjs?v=20261013-math-2';

const steps = [...document.querySelectorAll('.step')];
history.scrollRestoration = 'manual';
const stageTemplate = document.getElementById('stage-template');
const stages = steps.map(step => {
  const stage = stageTemplate.content.firstElementChild.cloneNode(true);
  stage.setAttribute('aria-label', `${step.dataset.title}: interactive diagram`);
  stage.querySelectorAll('[id]').forEach(node => {
    node.dataset.control = node.id;
    node.id = `${step.id}-${node.id}`;
  });
  for (const attribute of ['for', 'aria-labelledby']) {
    stage.querySelectorAll(`[${attribute}]`).forEach(node => {
      node.setAttribute(attribute, node.getAttribute(attribute).split(' ').map(id => `${step.id}-${id}`).join(' '));
    });
  }
  step.append(stage);
  const calculation = document.createElement('div');
  calculation.className = 'score-calculation';
  calculation.dataset.control = 'score-calculation';
  calculation.hidden = true;
  stage.querySelector('.diagram-wrap').after(calculation);
  return stage;
});
// One pinned screen serves the guided story. The original sections retain their
// anchors and scroll distances; the explorer and reference notes stay in flow.
const guidedCount = steps.findIndex(step => step.dataset.scene === 'explore');
const movieCopies = steps.slice(0, guidedCount).map(step => [...step.children]
  .filter(node => !node.classList.contains('stage')).map(node => node.cloneNode(true)));
const movieSequence = document.createElement('div');
movieSequence.className = 'movie-sequence';
const movieShell = document.createElement('div');
movieShell.className = 'movie-shell';
movieShell.setAttribute('aria-label', 'PCA scrolling story');
const movieCopy = document.createElement('div');
movieCopy.className = 'movie-copy';
const movieStage = document.createElement('div');
movieStage.className = 'movie-stage';
const movieTrack = document.createElement('div');
movieTrack.className = 'movie-track';
steps[0].before(movieSequence);
movieSequence.append(movieShell, movieTrack);
movieShell.append(movieCopy, movieStage);
steps.slice(0, guidedCount).forEach(step => { movieTrack.append(step); step.setAttribute('aria-hidden', 'true'); });
document.body.classList.add('is-cinematic');
let shownMovieStep = -1;
let scrubTransition;
let scrubProgress = -1;
let scrubLocal = .12;
let timedScrollProgress;
let inGuidedSequence = false;
function setMovieCopy(index, animate = false) {
  const content = document.createElement('div');
  content.className = 'movie-copy-content';
  content.append(...movieCopies[index].map(node => node.cloneNode(true)));
  movieCopy.replaceChildren(content);
  const heading = content.querySelector('h2');
  heading.id = 'movie-heading';
  movieCopy.setAttribute('aria-labelledby', heading.id);
  if (animate && !motion.matches) content.animate([{ opacity: .25, transform: 'translateY(6px)' }, { opacity: 1, transform: 'translateY(0)' }], { duration: 240, easing: 'ease-out' });
}
function showMovieStep(index) {
  for (const stage of movieStage.children) steps[shownMovieStep].append(stage);
  if (index >= guidedCount) return;
  shownMovieStep = index;
  setMovieCopy(index, !initializing);
  movieStage.append(stages[index]);
}
function measureMovieCopy() {
  // Reserve the largest prose height once per viewport, so the plot never
  // moves when a shorter caption replaces a longer one.
  movieCopy.style.height = 'auto';
  let height = 0;
  for (const copy of movieCopies) {
    movieCopy.replaceChildren(...copy.map(node => node.cloneNode(true)));
    height = Math.max(height, movieCopy.getBoundingClientRect().height);
  }
  movieCopy.style.height = '';
  movieShell.style.setProperty('--movie-copy-height', `${Math.ceil(height)}px`);
  if (shownMovieStep >= 0) setMovieCopy(shownMovieStep);
}
const $ = id => stages[current].querySelector(`[data-control="${id}"]`) || document.getElementById(id);
let svg;
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
let targetRemoval = 0;
let piece = 0;
// Rotate the face-on reference frame away from the known principal axes.
const residualPoints = coordinates(THREE_D_RESIDUAL_COORDINATES, 30);
const measurementScenes = ['scores', 'variance', 'projection', 'quadratic', 'maximum'];
const familiarPresets = ['scores', 'variance', 'projection', 'quadratic'];
let animation;
let finishAnimation;
let entrancePending;
let projectionMix = 1;
let basisReveal = 1;
let coordinateMix = 1;
let dropMix = 1;
let faceOn = 1;
let secondRemoval = 1;
let reconstructionMix = 1;
let reconstructionStart;
let svdMix = 1;
let assemblyMix = 1;
const entrancePlayed = new Set();
let scrollFrame;
let navigationUntil = 0;
let navigationLocked = false;
let initializing = true;

function fmt(value, digits = 2) {
  return (Math.abs(value) < .000001 ? 0 : value).toFixed(digits).replace('-', '−');
}
function revealBetween(value, start, end) {
  const t = Math.max(0, Math.min(1, (value - start) / (end - start)));
  return t * t * (3 - 2 * t);
}
function scoreCollection() {
  const collecting = steps[current].dataset.scene === 'scores' && steps[current].hasAttribute('data-motion-projection');
  const t = collecting ? projectionMix : 1;
  return {
    collecting, t,
    shift: revealBetween(t, 0, .32),
    projections: revealBetween(t, .32, .72),
    heading: revealBetween(t, .32, .48),
    column: revealBetween(t, .48, .66),
    legend: revealBetween(t, .72, .9),
    variance: revealBetween(t, .84, 1),
  };
}
const xu = () => `${mVar('X')}${mVar('u')}`;
const xv = () => `${mVar('X')}${mVar('v')}`;
const indexed = (symbol, index) => mSub(mVar(symbol), mNum(index));
const directionNotation = (u, digits = 2, operator = '=') => inlineMath(`${mVar('u')}${mOp(operator)}${mTranspose(mTuple(u.map(value => fmt(value, digits))))}`);
const varianceNotation = body => inlineMath(mFunction('Var', body));
const covarianceNotation = value => inlineMath(`${mFunction('Cov', `${xu()}${mOp(',')}${xv()}`)}${mOp('=')}${mNum(fmt(value))}`);
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
  const node = add(parent, 'text', { x, y, fill: color, 'font-size': compact ? Math.max(28, size * 1.55) : size, 'text-anchor': anchor });
  for (const run of Array.isArray(text) ? text : [text]) {
    if (typeof run === 'string') { add(node, 'tspan', {}, run); continue; }
    add(node, 'tspan', { ...(run.math ? { class: 'math-symbol', 'font-style': run.italic === false ? 'normal' : 'italic' } : {}) }, run.text);
    for (const [script, shift] of [['sub', 'sub'], ['sup', 'super']]) {
      if (run[script] !== undefined) add(node, 'tspan', { class: 'math-script', 'font-size': '.7em', 'baseline-shift': shift }, run[script]);
    }
  }
  return node;
}
function arrow(parent, a, b, color, dash = '') {
  line(parent, a, b, color, 2.2, dash);
  const theta = Math.atan2(b[1] - a[1], b[0] - a[0]);
  const wing = offset => [b[0] - 9 * Math.cos(theta + offset), b[1] - 9 * Math.sin(theta + offset)];
  line(parent, wing(.45), b, color, 2.2);
  line(parent, wing(-.45), b, color, 2.2);
}
function point(parent, position, selected = false, hollow = false, radius = 5) {
  const node = add(parent, 'circle', { cx: position[0], cy: position[1], r: radius, fill: hollow ? '#fff' : colors.blue, stroke: hollow ? colors.rule : colors.blue, 'stroke-width': hollow ? 1.4 : .7 });
  if (selected) add(parent, 'circle', { cx: position[0], cy: position[1], r: radius + 5, fill: 'none', stroke: colors.rust, 'stroke-width': 2 });
  return node;
}
function cross(parent, position, color = colors.blue) {
  line(parent, [position[0] - 4, position[1] - 4], [position[0] + 4, position[1] + 4], color, 2);
  line(parent, [position[0] - 4, position[1] + 4], [position[0] + 4, position[1] - 4], color, 2);
}
function plot(parent, cx, cy, scale, extent, names = ['x₁', 'x₂'], ticks = true) {
  const map = p => [cx + scale * p[0], cy - scale * p[1]];
  line(parent, map([-extent, 0]), map([extent, 0]), colors.rule);
  line(parent, map([0, -extent]), map([0, extent]), colors.rule);
  const axisRuns = name => [{ text: name[0], math: true, ...(name.length > 1 ? { sub: name.slice(1).replace('₁', '1').replace('₂', '2').replace('₃', '3') } : {}) }];
  label(parent, cx + scale * extent - 4, cy + 24, axisRuns(names[0]), colors.muted, 19, 'end');
  label(parent, cx + 9, cy - scale * extent + 8, axisRuns(names[1]), colors.muted, 19);
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
function basis(parent, map, u, v, length, second = true, names = ['u', 'v'], reveal = 1, palette = [colors.blue, colors.rust]) {
  const origin = map([0, 0]);
  const end = (direction, sign = 1) => map(direction.map(value => value * length * sign));
  line(parent, end(u, -1), end(u), palette[0], 1.3);
  arrow(parent, origin, map(u), palette[0]);
  const firstLabel = names[0] === 'w' ? map(u) : end(u);
  // Keep a near-vertical direction label apart from the original y-axis label.
  const firstVertical = Math.abs(u[0]) < .25 && u[1] > 0;
  label(parent, firstLabel[0] + (firstVertical ? -16 : 12), firstLabel[1] - 5, [{text:names[0],math:true}], palette[0], 21, firstVertical ? 'end' : 'start');
  if (second) {
    const growing = v.map(value => value * reveal);
    const group = add(parent, 'g', { opacity: reveal });
    line(group, end(growing, -1), end(growing), palette[1], 1.3, '6 5');
    arrow(group, origin, map(growing), palette[1], '6 5');
    const secondLabel = names[0] === 'w' ? map(v) : end(v);
    const secondVertical = Math.abs(v[0]) < .25 && v[1] > 0;
    label(group, secondLabel[0] + (secondVertical ? -16 : 10), secondLabel[1] - 6, [{text:names[1].replace('⊥', ''),math:true,...(names[1].includes('⊥') ? {sup:'⊥'} : {})}], palette[1], 21, secondVertical ? 'end' : 'start');
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
  const isCurve = ['maximum', 'explore'].includes(scene) && !isPair;
  const isStrip = ['variance', 'quadratic'].includes(scene);
  const isScores = scene === 'scores';
  const collection = scoreCollection();
  const showAllProjections = ['scores', 'variance'].includes(scene);
  const showBasis = current > 0;
  const extent = points === LINE_POINTS ? 3.1 : 6.2;
  const selected = isPair && points === POINTS ? points.length - 2 : points.length - 1;
  if (isPair) {
    label(parent, 178, 34, 'Original coordinates', colors.ink, 20, 'middle');
    label(parent, 540, 34, coordinateMix < 1 ? 'Changing coordinates' : ['New coordinates · ', {text:'XU',math:true}], colors.ink, 20, 'middle');
    const scale = points === LINE_POINTS ? 43 : 23;
    const left = plot(parent, 178, 226, scale, extent);
    const right = plot(parent, 540, 226, scale, extent, coordinateMix < 1 ? [' ', ' '] : ['z₁', 'z₂']);
    basis(parent, left, summary.u, summary.v, extent - .4);
    points.forEach((p, i) => point(parent, left(p), i === selected, false, points.length > 5 ? 4 : 5));
    coordinates(points, angle * coordinateMix).forEach((p, i) => point(parent, right(p), i === selected, false, points.length > 5 ? 4 : 5));
    label(parent, 360, 415, coordinateMix < 1 ? 'The same points, expressed in the chosen basis' : 'Every distance is preserved', colors.ink, 18, 'middle');
    return;
  }
  if (scene === 'reconstruction') {
    const left = plot(parent, 178, 226, 23, extent);
    const right = plot(parent, 540, 226, 23, extent, ['z₁', 'z₂']);
    label(parent, 178, 34, 'Reconstruction', colors.ink, 20, 'middle');
    label(parent, 540, 34, 'Second score → 0', colors.ink, 20, 'middle');
    basis(parent, left, summary.u, summary.v, extent - .4, false);
    points.forEach((p, i) => {
      const moving = p.map((v, j) => v + dropMix * (summary.reconstruction[i][j] - v));
      const z = [summary.scores[i], summary.otherScores[i]];
      const collapsed = [z[0], (1 - dropMix) * z[1]];
      line(parent, left(p), left(moving), colors.rule, 1, '3 4');
      point(parent, left(p), false, true, 3.8);
      cross(parent, left(moving), i === selected ? colors.rust : colors.blue);
      line(parent, right(z), right(collapsed), colors.rule, 1, '3 4');
      point(parent, right(z), false, true, 3.8);
      point(parent, right(collapsed), i === selected, false, 4);
    });
    label(parent, 360, 415, 'The first coordinate stays fixed; the second goes to zero', colors.ink, 18, 'middle');
    return;
  }
  const cx = isScores ? 360 - 127 * collection.shift : 360;
  const cy = isCurve || isStrip ? 174 : 220;
  const scale = points === LINE_POINTS ? (isCurve || isStrip ? 43 : 52) : 22;
  const map = plot(parent, cx, cy, scale, extent);
  if (points === LINE_POINTS) {
    line(parent, map([-2.7, -2.7]), map([2.7, 2.7]), colors.light, 1.5, '4 5');
    if (!showBasis) label(parent, map([1.6, 2.4])[0], map([1.6, 2.4])[1], [{text:'x',math:true,sub:'1'}, ' = ', {text:'x',math:true,sub:'2'}], colors.ink, 21);
  }
  if (showBasis) basis(parent, map, summary.u, summary.v, extent - .3, !isReconstruction && !measurementScenes.includes(scene), ['u', 'v'], scene === 'basis' ? basisReveal : 1);
  if (isReconstruction) {
    points.forEach((p, i) => {
      line(parent, map(p), map(summary.reconstruction[i]), colors.rule, 1, '3 3');
      point(parent, map(p), i === selected, true, 4);
      cross(parent, map(summary.reconstruction[i]));
    });
    label(parent, 70, 305, '○ Original     × Reconstruction', colors.muted, 17);
  } else if (showAllProjections) {
    points.forEach((p, i) => {
      // The highlighted projection is already complete in the preceding scene.
      // Preserve it while making room, then project the remaining observations.
      const amount = collection.collecting ? i === selected ? 1 : collection.projections : projectionMix;
      const projected = p.map((value, j) => value + amount * (summary.reconstruction[i][j] - value));
      const marks = add(parent, 'g', { opacity: collection.collecting && i !== selected ? revealBetween(collection.t, .32, .38) : 1 });
      line(marks, map(p), map(projected), colors.rule, 1, '3 4');
      cross(marks, map(projected), i === selected ? colors.rust : colors.blue);
      const hollow = collection.collecting ? collection.projections : 1;
      const original = point(parent, map(p), i === selected, false, 5 - .5 * hollow);
      original.setAttribute('fill', `rgb(${[35, 78, 112].map(channel => channel + hollow * (255 - channel)).join(',')})`);
      original.setAttribute('stroke', `rgb(${[35, 78, 112].map(channel => channel + hollow * (118 - channel)).join(',')})`);
      original.setAttribute('stroke-width', .7 + .7 * hollow);
    });
  } else {
    if (['projection', 'quadratic'].includes(scene)) {
      points.forEach((p, i) => {
        if (scene !== 'projection' || i === selected) {
          const projected = p.map((value, j) => value + projectionMix * (summary.reconstruction[i][j] - value));
          line(parent, map(p), map(projected), colors.rule, 1, '3 4');
          if (scene === 'projection') cross(parent, map(projected), colors.rust);
        }
      });
    }
    points.forEach((p, i) => point(parent, map(p), scene !== 'line' && i === selected, false, points.length > 5 ? 4 : 5));
  }
  if (scene === 'basis') label(parent, 360, 430, 'Add a perpendicular direction', colors.muted, 20, 'middle');
  if (isStrip) scoreStrip(parent, summary.scores, 386, ['One score per observation · ', {text:'Xu',math:true}]);
  if (isScores) {
    const heading = add(parent, 'g', { opacity: collection.heading });
    label(heading, cx, 32, 'Data and projections', colors.ink, 19, 'middle');
    const column = add(parent, 'g', { opacity: collection.column });
    label(column, 565, 75, [{text:'Xu',math:true}], colors.blue, 28, 'middle');
    add(column, 'path', { d: 'M514 98H502V327H514M614 98H626V327H614', stroke: colors.ink, 'stroke-width': 1.3, fill: 'none' });
    summary.scores.forEach((value, i) => {
      const row = add(column, 'g', { opacity: !collection.collecting || i === selected ? 1 : revealBetween(collection.t, .62 + i * .035, .78 + i * .035) });
      label(row, 564, 131 + i * 44, fmt(value), i === selected ? colors.rust : colors.ink, 23, 'middle');
    });
    const legend = add(parent, 'g', { opacity: collection.legend });
    label(legend, 360, 428, ['○ Data point     × Projection onto ', {text:'u',math:true}], colors.muted, 18, 'middle');
  }
  if (isCurve) varianceCurve(parent, summary, points);
}

function drawResidualPlane(parent, summary) {
  const map = plot(parent, 360, 179, 57, 2.35, ['a', 'b']);
  basis(parent, map, summary.u, summary.v, 2.2, true, ['w', 'w⊥'], 1, [colors.rust, colors.teal]);
  residualPoints.forEach((p, i) => point(parent, map(p), i === 20, false, 4.5));
  varianceCurve(parent, summary, residualPoints, -90, 90);
}

function draw3D(parent) {
  const scene = steps[current].dataset.scene;
  const scores = changeBasis(THREE_D_POINTS, THREE_D_BASIS);
  const isCoordinates = scene === 'explore' && view === 'coordinates';
  const isReconstruction = scene === 'plane' || (scene === 'explore' && view === 'reconstruction');
  const isDeflation = scene === 'deflation';
  const isAssembly = scene === 'pieces';
  const isPiece = scene === 'explore' && view === 'pieces';
  const assembly = isAssembly ? assembleComponents(THREE_D_POINTS, THREE_D_BASIS, assemblyMix) : null;
  const cols = [0, 1, 2].map(j => THREE_D_BASIS.map(row => row[j]));
  let displayed = isCoordinates ? scores : THREE_D_POINTS;
  if (isDeflation) displayed = THREE_D_POINTS.map((p, i) => p.map((v, j) => (1 - removal) * v + removal * THREE_D_RESIDUALS[i][j]));
  if (isPiece) displayed = scores.map(row => cols[piece].map(v => row[piece] * v));
  if (isAssembly) displayed = assembly.points;
  const target = reconstructBasis(scores, THREE_D_BASIS, components);
  const reconstructed = target.map((p, i) => p.map((v, j) => {
    const start = reconstructionStart || THREE_D_POINTS;
    return start[i][j] + reconstructionMix * (v - start[i][j]);
  }));
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
  };
  const title = isAssembly ? assemblyMix === 1 ? 'All three pieces recover the data' : `Add the PC${assembly.active + 1} contribution`
    : isPiece ? [`The PC${piece + 1} contribution · one piece of `, {text:'X',math:true}] : titles[scene] || (isCoordinates ? ['Principal coordinates · ', {text:'Z',math:true}, ' = ', {text:'XU',math:true}] : isReconstruction ? 'Reconstruct the original observations' : 'Original cloud · three coordinates');
  label(parent, 360, 40, title, colors.ink, 21, 'middle');
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
    label(parent, end[0] + 10, end[1] + 5, [{text:axisNames[j][0],math:true,sub:String(j + 1)}], colors.muted, 19);
  });
  const axisColors = [colors.blue, colors.rust, colors.teal];
  if (!isCoordinates) cols.forEach((u, j) => {
    if ((['three-d', 'deflation'].includes(scene) && j !== 0) || (isPiece && j !== piece) || (isAssembly && j > assembly.active)) return;
    const end = project3D(u.map(v => v * (j === 0 ? 6 : 3)));
    const dash = ['', '6 5', '2 4'][j];
    line(parent, project3D(u.map(v => v * (j === 0 ? -6 : -3))), end, axisColors[j], 1.3, dash);
    arrow(parent, project3D([0, 0, 0]), project3D(u), axisColors[j], dash);
    label(parent, end[0] + 12, isAssembly ? Math.max(80, end[1] - 7) : end[1] - 7, [{text:'u',math:true,sub:String(j + 1)}], axisColors[j], 21);
  });
  if (isDeflation || isPiece || isAssembly) {
    THREE_D_POINTS.forEach((p, i) => {
      if (isDeflation) line(parent, project3D(p), project3D(displayed[i]), colors.rule, 1, '3 4');
      point(parent, project3D(p), false, true, 4);
    });
  }
  if (isAssembly && assemblyMix < 1) {
    const before = scores.map(row => cols[0].map((_, d) => row.reduce((sum, score, j) => sum + (j < assembly.active ? score * cols[j][d] : 0), 0)));
    before.forEach((p, i) => line(parent, project3D(p), project3D(displayed[i]), axisColors[assembly.active], 1, '3 4'));
  }
  displayed.map((p, i) => ({ p, i })).sort((a, b) => depth(b.p) - depth(a.p)).forEach(({ p, i }) => {
    if (isReconstruction) {
      line(parent, project3D(p), project3D(reconstructed[i]), colors.rule, 1, '3 3');
      point(parent, project3D(p), false, true, 4);
      cross(parent, project3D(reconstructed[i]), i === 20 ? colors.rust : colors.blue);
    } else if (isPiece || isAssembly) cross(parent, project3D(p), axisColors[isAssembly ? assembly.active : piece]);
    else point(parent, project3D(p), i === 20, false, 4.5);
  });
  const legend = isDeflation ? '○ Original     ● After subtraction' : isAssembly ? '○ Original     × Sum of contributions' : isPiece ? '○ Original     × Selected contribution' : isReconstruction ? `○ Original     × Reconstruction · ${components} kept` : scene === 'three-d' ? 'PC1 variance = 9 · total variance = 10.25' : 'Turning the camera preserves score variance';
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
function varianceBudget(parts, total) {
  const budget = $('variance-budget');
  budget.hidden = false;
  budget.replaceChildren();
  const heading = document.createElement('p');
  heading.className = 'budget-heading';
  heading.textContent = `Original total variance: ${fmt(total, total % 1 ? 2 : 0)}`;
  const track = document.createElement('div');
  track.className = 'budget-track';
  track.setAttribute('aria-hidden', 'true');
  const key = document.createElement('div');
  key.className = 'budget-key';
  for (const part of parts) {
    const segment = document.createElement('span');
    segment.style.width = `${Math.max(0, part.value) / total * 100}%`;
    segment.style.backgroundColor = part.color || colors.blue;
    if (part.removed) segment.className = 'budget-omitted';
    track.append(segment);
    const item = document.createElement('div');
    const name = document.createElement('span');
    setNotation(name, part.label);
    const value = document.createElement('output');
    value.value = fmt(part.value);
    item.append(name, value);
    key.append(item);
  }
  budget.append(heading, track, key);
}
function render() {
  const scene = steps[current].dataset.scene;
  const isThree = dataset === 'three';
  const isResidual = ['residual-plane', 'second-pc', 'third-pc'].includes(scene);
  const isExplore = scene === 'explore';
  const isSVD = scene === 'svd' || (isExplore && isThree && view === 'svd');
  const points = isResidual ? residualPoints : dataset === 'line' ? LINE_POINTS : POINTS;
  const summary = pcaSummary(angle, points);
  stages[current].querySelector('.stage-controls').hidden = !isExplore;
  $('angle-control').hidden = !isExplore || isThree;
  $('data-control').hidden = !isExplore;
  $('view-control').hidden = !isExplore;
  $('components-control').hidden = !(isExplore && isThree && view === 'reconstruction');
  $('camera-control').hidden = !isExplore || !isThree || isSVD;
  $('piece-control').hidden = !(isExplore && isThree && ['pieces', 'svd'].includes(view));
  $('view').querySelectorAll('[data-three-only]').forEach(option => { option.hidden = option.disabled = !isThree; });
  $('statistics').hidden = !isExplore;
  $('variance-budget').hidden = true;
  $('score-calculation').hidden = true;
  $('angle').min = isResidual ? -90 : 0;
  $('angle').max = isResidual ? 90 : 180;
  $('angle').value = angle;
  setNotation($('angle-label'), isResidual ? ['Direction in the plane ', inlineMath(mVar('φ'))] : measurementScenes.includes(scene) ? 'Direction angle' : ['Direction angle ', inlineMath(mVar('θ'))]);
  $('angle-value').value = `${fmt(angle, 0)}°`;
  $('dataset').value = dataset;
  $('view').value = view;
  $('components').value = components;
  $('camera-value').value = `${fmt(camera, 0)}°`;
  $('camera').value = camera;
  $('piece').value = piece;
  const hasEntrance = entranceFor(current) !== null;
  $('motion-controls').hidden = !hasEntrance;
  setNotation($('motion-cue'), scene === 'basis' ? 'Add a perpendicular direction'
    : scene === 'reconstruction' ? dropMix < 1 ? 'Discarding the second coordinate' : 'Second coordinate discarded'
    : scene === 'residual-plane' ? faceOn < 1 ? 'Turning to face the residual plane' : 'The same residuals, viewed face on'
    : scene === 'third-pc' ? secondRemoval < 1 ? 'Subtracting PC2' : 'Only PC3 remains'
    : isSVD ? svdMix < 1 ? 'Scale the unit pattern' : 'The pattern has become the score column'
    : scene === 'plane' ? reconstructionMix < 1 ? 'Projecting onto the retained subspace' : 'Original observations reconstructed'
    : scene === 'coordinates' ? coordinateMix < 1 ? 'Expressing the same points in the new basis' : 'All coordinates kept'
    : scene === 'pieces' ? assemblyMix < 1 ? `Adding PC${assembleComponents(THREE_D_POINTS, THREE_D_BASIS, assemblyMix).active + 1}` : 'All three contributions added'
    : scene === 'deflation'
    ? removal < 1 ? 'Removing PC1' : 'PC1 removed'
    : steps[current].hasAttribute('data-motion-projection')
      ? scene === 'projection' ? 'Follow one point to its signed score' : 'Collect one score per observation'
      : steps[current].hasAttribute('data-motion-camera')
        ? 'Turn the view to inspect the long axis'
        : isResidual ? `Direction in the plane: ${fmt(angle, 0)}°` : ['Direction ', inlineMath(mVar('u')), `: ${fmt(angle, 0)}°`]);
  stages[current].classList.toggle('is-three', isThree);
  stages[current].classList.toggle('is-explore', isExplore);
  stages[current].querySelectorAll('[data-angle]').forEach((button, index) => {
    const degrees = (isResidual ? [-30, 0, 60] : familiarPresets.includes(scene) ? [0, 90, 45] : [0, 45, 135])[index];
    button.dataset.angle = degrees;
    button.setAttribute('aria-pressed', String(Math.abs(targetAngle - degrees) < .1));
    const suffix = isResidual ? (index === 0 ? ' · PC2' : index === 2 ? ' · PC3' : '') : familiarPresets.includes(scene) ? [' · x₁', ' · x₂', ' · diagonal'][index] : ['maximum', 'basis', 'coordinates', 'cloud', 'reconstruction', 'explore'].includes(scene) && degrees !== 0 ? ` · PC${index}` : '';
    setNotation(button, familiarPresets.includes(scene) && !isResidual && index < 2
      ? [`${String(degrees).replace('-', '−')}° · `, inlineMath(indexed('x', index + 1))]
      : `${String(degrees).replace('-', '−')}°${suffix}`);
  });
  let caption;
  function statistics(labels, values) {
    ['variance-label', 'second-label', 'third-label'].forEach((id, i) => { setNotation($(id), labels[i]); });
    ['variance', 'second-variance', 'third-value'].forEach((id, i) => { $(id).value = values[i]; });
  }
  if (isResidual) {
    statistics([varianceNotation(`${mVar('R')}${mVar('w')}`), 'Other direction', 'Residual total'], [fmt(summary.variance), fmt(summary.otherVariance), '1.25']);
    caption = ['The fully deflated cloud viewed face on. At ', inlineMath(`${mVar('φ')}${mOp('=')}${mNum(fmt(angle, 0))}${mOp('°')}`), `, score variance is ${fmt(summary.variance)}; perpendicular variance is ${fmt(summary.otherVariance)}. The maximum is 1 at −30° (PC2); PC3 is perpendicular at 60°, with variance 0.25.`];
  } else if (scene === 'deflation') {
    const left = 9 * (1 - removal) ** 2;
    statistics(['PC1 variance left', 'In the plane', 'Remaining total'], [fmt(left), '1.25', fmt(left + 1.25)]);
    caption = removal < 1
      ? 'Subtracting the PC1 contribution moves each point into the plane perpendicular to PC1.'
      : 'PC1 is removed. Every residual lies in the plane perpendicular to PC1, with total variance 1.25.';
  } else if (isSVD) {
    const { singularValues } = svdExample();
    const d = singularValues[piece];
    statistics([['Length ', inlineMath(indexed('d', piece + 1))], ['Length of ', inlineMath(mSub(mVar('p'), mVar('j')))], 'Score variance'], [fmt(d), '1.00', fmt(d * d / 23)]);
    caption = [`PC${piece + 1}: `, inlineMath(`${mSub(mVar('z'), mVar('j'))}${mOp('=')}${mSub(mVar('d'), mVar('j'))}${mSub(mVar('p'), mVar('j'))}`), `. The score column has length ${fmt(d)}; its normalized pattern has length 1. Both use the same vertical scale. Squared length divided by 23 gives score variance ${fmt(d * d / 23)}.`];
  } else if (isThree) {
    const three = threeSummary();
    const isReconstruction = scene === 'plane' || (isExplore && view === 'reconstruction');
    if (isExplore && view === 'pieces') {
      statistics(['Piece variance', 'Original total', 'Share of total'], [fmt(three.variances[piece]), '10.25', `${fmt(three.variances[piece] / three.total * 100, 1)}%`]);
      caption = [`Crosses show each observation’s PC${piece + 1} contribution `, inlineMath(`${mSub(mVar('z'), `${mVar('i')}${mNum(piece + 1)}`)}${indexed('u', piece + 1)}`), `. It lies along one direction and contributes variance ${fmt(three.variances[piece])}. The three pieces add to the original hollow-circle observations.`];
    } else if (scene === 'three-d') {
      statistics([varianceNotation(`${mVar('X')}${indexed('u', 1)}`), 'Remaining', 'Original total'], ['9.00', '1.25', '10.25']);
      caption = ['Begin with PC1 alone: ', inlineMath(`${indexed('u', 1)}${mOp('=')}<mfrac>${mTranspose(mTuple([1, 1, 1]))}<msqrt>${mNum(3)}</msqrt></mfrac>`), ', with score variance 9. The original total is 10.25, so 1.25 remains in directions perpendicular to PC1.'];
    } else {
      statistics([varianceNotation(`${mVar('X')}${indexed('u', 1)}`), varianceNotation(`${mVar('X')}${indexed('u', 2)}`), isReconstruction ? 'Variance kept' : varianceNotation(`${mVar('X')}${indexed('u', 3)}`)], ['9.00', '1.00', isReconstruction ? `${fmt(three.retained * 100, 1)}%` : '0.25']);
      caption = isReconstruction
        ? `${components} component${components > 1 ? 's' : ''} retain${components > 1 ? '' : 's'} ${fmt(100 * three.retained, 1)}% of original variance 10.25. ${components === 1 ? 'The retained line follows PC1.' : components === 2 ? 'The retained plane contains PC1 and PC2; the residual plane from subtraction was perpendicular to PC1.' : 'All three components recover the original points.'}`
        : 'Three perpendicular principal directions. Their score variances are 9, 1, and 0.25; the full basis preserves all the observations.';
    }
  } else {
    const isReconstruction = scene === 'reconstruction' || (isExplore && view === 'reconstruction');
    setNotation($('variance-label'), varianceNotation(xu()));
    setNotation($('second-label'), varianceNotation(xv()));
    $('variance').value = fmt(summary.variance);
    $('second-variance').value = fmt(summary.otherVariance);
    $('third-label').textContent = isReconstruction ? 'Variance kept' : 'Total variance';
    $('third-value').value = isReconstruction ? `${fmt(summary.retainedFraction * 100, 1)}%` : fmt(summary.totalVariance);
    if (measurementScenes.includes(scene)) statistics([inlineMath(mSup(mNorm(xu()), mNum(2))), inlineMath(`${mVar('n')}${mOp('−')}${mNum(1)}`), varianceNotation(xu())], [fmt(summary.scores.reduce((sum, score) => sum + score * score, 0)), String(points.length - 1), fmt(summary.variance)]);
    const captions = {
      line: ['Five centered points on ', inlineMath(`${indexed('x', 1)}${mOp('=')}${indexed('x', 2)}`), '. PC1 comes from the unit direction with the greatest variance of projected scores.'],
      basis: [directionNotation(summary.u), '. Solid ', inlineMath(mVar('u')), ' and dashed ', inlineMath(mVar('v')), ' are perpendicular unit directions.'],
      projection: [`The ringed point has score ${fmt(summary.scores.at(-1))}. The dashed perpendicular ends at its projection onto `, inlineMath(mVar('u')), '.'],
      scores: [directionNotation(summary.u), `. Squared score length ${fmt(summary.scores.reduce((sum, score) => sum + score * score, 0))} divided by 4 gives variance ${fmt(summary.variance)}. The circles are data points; crosses are their projections.`],
      variance: [`Current squared score length: ${fmt(summary.scores.reduce((sum, score) => sum + score * score, 0))}. The maximum is 20 at 45°, giving sample variance `, inlineMath(`<mfrac>${mNum(20)}${mNum(4)}</mfrac>${mOp('=')}${mNum(5)}`), '.'],
      quadratic: ['Variance from the scores and from ', inlineMath(`${mTranspose(mVar('u'))}${mVar('S')}${mVar('u')}`), ` both equal ${fmt(summary.variance)}. The sample covariance matrix has all four entries equal to 2.5.`],
      maximum: `The greatest variance is 5 at 45°. The perpendicular direction at 135° has variance 0.`,
      coordinates: ['Both coordinates preserve every point. ', covarianceNotation(summary.crossCovariance), points === LINE_POINTS ? '. At 45°, the second score is zero for everyone.' : '. At 45°, the score variances are 9 and 1.'],
      cloud: ['A new cloud of 24 points. ', covarianceNotation(summary.crossCovariance), '. At the principal basis, the score variances are 9 and 1.'],
      reconstruction: `One direction keeps ${fmt(100 * summary.retainedFraction, 1)}% of total variance. Squared reconstruction errors sum to ${fmt(summary.meanSquaredError * points.length)}.`,
      explore: isReconstruction
        ? `One coordinate keeps ${fmt(100 * summary.retainedFraction, 1)}% of total variance. Reconstruction SSE = ${fmt(summary.meanSquaredError * points.length)}.`
        : [inlineMath(`${mFunction('Var', xu())}${mOp('=')}${mNum(fmt(summary.variance))}`), '; ', inlineMath(`${mFunction('Var', xv())}${mOp('=')}${mNum(fmt(summary.otherVariance))}`), `; covariance = ${fmt(summary.crossCovariance)}. Total variance stays ${fmt(summary.totalVariance)}.`],
    };
    caption = captions[scene];
  }
  if (scene === 'projection') {
    // Selecting the horizontal unit direction returns the first data column.
    // Keep this identity clear of the axis labels and carry it into collection.
    $('score-calculation').hidden = false;
    $('score-calculation').style.opacity = Math.max(0, (projectionMix - .65) / .35);
    setNotation($('score-calculation'), inlineMath(`${xu()}${mOp('=')}${indexed('x', 1)}`));
  } else if (scene === 'scores') {
    const squared = summary.scores.reduce((sum, value) => sum + value * value, 0);
    $('score-calculation').hidden = false;
    const formula = inlineMath(`${mFunction('Var', xu())}${mOp('=')}<mfrac>${mNum(fmt(squared))}${mNum(points.length - 1)}</mfrac>${mOp('=')}${mNum(fmt(summary.variance))}`);
    $('score-calculation').style.opacity = 1;
    const collection = scoreCollection();
    if (collection.collecting) {
      const readouts = document.createElement('div');
      readouts.className = 'score-transition';
      const single = inlineMath(`${xu()}${mOp('=')}${indexed('x', 1)}`);
      [single, formula].forEach((notation, i) => {
        const span = document.createElement('span');
        const opacity = i === 0 ? 1 - collection.variance : collection.variance;
        span.style.opacity = opacity;
        span.setAttribute('aria-hidden', String(opacity < .5));
        setNotation(span, notation);
        readouts.append(span);
      });
      $('score-calculation').replaceChildren(readouts);
    } else setNotation($('score-calculation'), formula);
  } else if (isSVD) {
    const d = svdExample().singularValues[piece];
    $('score-calculation').hidden = false;
    $('score-calculation').style.opacity = Math.max(0, (svdMix - .75) / .25);
    setNotation($('score-calculation'), inlineMath(`${mFunction('Var', `${mVar('X')}${indexed('u', piece + 1)}`)}${mOp('=')}<mfrac>${mSup(indexed('d', piece + 1), mNum(2))}${mNum(23)}</mfrac>${mOp('=')}${mNum(fmt(d * d / 23))}`));
  }
  if (['basis', 'cloud', 'coordinates'].includes(scene)) {
    const values = summary;
    varianceBudget([
      { label: varianceNotation(xu()), value: values.variance, color: colors.blue },
      { label: varianceNotation(xv()), value: values.otherVariance, color: colors.rust },
    ], summary.totalVariance);
  } else if (scene === 'reconstruction') {
    const left = summary.otherVariance * (1 - dropMix) ** 2;
    varianceBudget([
      { label: 'First coordinate kept', value: summary.variance, color: colors.blue },
      { label: 'Second coordinate left', value: left, color: colors.rust },
      { label: 'Discarded', value: summary.otherVariance - left, removed: true },
    ], summary.totalVariance);
  } else if (scene === 'deflation') {
    const left = 9 * (1 - removal) ** 2;
    varianceBudget([
      { label: 'PC1 left', value: left, color: colors.blue },
      { label: 'In the plane', value: 1.25, color: colors.rust },
      { label: 'Removed', value: 9 - left, removed: true },
    ], 10.25);
  } else if (isResidual) {
    const third = scene === 'third-pc';
    const first = third ? (1 - secondRemoval) ** 2 : summary.variance;
    const second = third ? .25 : summary.otherVariance;
    varianceBudget([
      { label: third ? 'PC2 left' : varianceNotation(`${mVar('R')}${mVar('w')}`), value: first, color: colors.rust },
      { label: third ? 'PC3' : 'Perpendicular direction', value: second, color: colors.teal },
      { label: 'Removed', value: 10.25 - first - second, removed: true },
    ], 10.25);
  } else if (isThree && !isExplore && !isSVD) {
    const three = threeSummary();
    if (scene === 'plane') {
      const scores = changeBasis(THREE_D_POINTS, THREE_D_BASIS);
      const target = reconstructBasis(scores, THREE_D_BASIS, components);
      const start = reconstructionStart || THREE_D_POINTS;
      const moving = target.map((p, i) => p.map((v, j) => start[i][j] + reconstructionMix * (v - start[i][j])));
      const movingScores = changeBasis(moving, THREE_D_BASIS);
      const remaining = [0, 1, 2].map(j => sampleVariance(movingScores.map(p => p[j])));
      const kept = remaining.slice(0, components).reduce((a, b) => a + b, 0);
      const tail = remaining.slice(components).reduce((a, b) => a + b, 0);
      varianceBudget([
        { label: `${components} component${components === 1 ? '' : 's'} kept`, value: kept, color: colors.blue },
        { label: 'Omitted coordinates left', value: tail, color: colors.teal },
        { label: 'Discarded', value: 10.25 - kept - tail, removed: true },
      ], 10.25);
    } else if (scene === 'pieces') {
      const { weights } = assembleComponents(THREE_D_POINTS, THREE_D_BASIS, assemblyMix);
      const parts = three.variances.map((value, j) => ({ label: `PC${j + 1} added`, value: value * weights[j] ** 2, color: [colors.blue, colors.rust, colors.teal][j] }));
      parts.push({ label: 'Not yet recovered', value: 10.25 - parts.reduce((sum, part) => sum + part.value, 0), removed: true });
      varianceBudget(parts, 10.25);
    } else varianceBudget(three.variances.map((value, j) => ({ label: `PC${j + 1}`, value, color: [colors.blue, colors.rust, colors.teal][j] })), 10.25);
  }
  const brief = {
    line: 'The observations are centered: their mean is the origin.',
    basis: 'Both perpendicular unit directions together describe each point.',
    coordinates: 'The total stays fixed while variation shifts between coordinates.',
    cloud: 'Both principal coordinates now contain variation.',
    reconstruction: dropMix < 1 ? 'Only the second coordinate changes.' : 'One component retains 9 of the original total variance 10: 90%.',
    'three-d': 'PC1 captures 9 of the original total variance 10.25.',
    deflation: removal < 1 ? 'Points move parallel to PC1; the camera stays fixed.' : 'PC1 is gone. The remaining variance is 1.25.',
    'residual-plane': faceOn < 1 ? 'Only the view changes; the residual points stay fixed.' : 'Next, the direction rotates within this plane to find PC2.',
    'second-pc': 'The maximum is PC2. Its perpendicular direction within the plane is PC3.',
    'third-pc': secondRemoval < 1 ? 'Subtract PC2 along parallel paths.' : 'Only the PC3 contribution remains, with variance 0.25.',
    plane: reconstructionMix < 1 ? 'Return to the original cloud, then omit the unused contributions.' : `${components} component${components === 1 ? '' : 's'} retain ${fmt(100 * threeSummary().retained, 1)}% of the original total variance.`,
    pieces: assemblyMix < 1 ? 'Each new contribution moves the points along its principal direction.' : 'The sum of all three contributions recovers every original observation.',
    svd: 'Each observation keeps its place in the pattern while its score is scaled.',
  };
  setNotation($('caption'), (isSVD ? brief.svd : brief[scene]) || caption);
  $('variance-budget').style.opacity = scene === 'coordinates' ? Math.max(0, Math.min(1, (coordinateMix - .6) / .4)) : 1;
  const captionText = notationText($('caption'));
  const description = ['scores', 'variance'].includes(scene) ? `${captionText} Xu contains the scores ${summary.scores.map(value => fmt(value)).join(', ')}.` : captionText;
  svg.replaceChildren(el('title', { id: `${steps[current].id}-diagram-title` }, steps[current].dataset.title), el('desc', { id: `${steps[current].id}-diagram-description` }, description));
  const drawing = add(svg, 'g', { 'aria-hidden': 'true' });
  if (isSVD) drawSVDFactors(drawing, { add, line, label, point, colors, fmt }, piece, svdMix);
  else if (scene === 'third-pc') drawThirdDirection(drawing, { progress: secondRemoval, selected: 20 });
  else if (scene === 'residual-plane' && faceOn < 1) drawResidualTransition(drawing, { progress: faceOn, camera, selected: 20 });
  else if (isResidual) drawResidualPlane(drawing, summary);
  else if (isThree) draw3D(drawing);
  else draw2D(drawing, points, summary);
}

function remember() {
  saved.set(current, { angle: targetAngle, dataset, view, components, camera, removal: targetRemoval, piece });
}
function stopAnimation(complete = false) {
  cancelAnimationFrame(animation);
  animation = undefined;
  const finish = finishAnimation;
  finishAnimation = undefined;
  if (complete && finish) finish();
}
function changeAngle(value, animate = false) {
  stopAnimation();
  targetAngle = value;
  if (!animate || motion.matches) {
    angle = value;
    render();
    return;
  }
  const from = angle;
  const start = performance.now();
  finishAnimation = () => { angle = value; render(); };
  // Keep the current diagram and its controls in sync during rotation.
  render();
  function frame(time) {
    const t = Math.min(1, (time - start) / 450);
    const eased = t * t * (3 - 2 * t);
    angle = from + (value - from) * eased;
    render();
    if (t < 1) animation = requestAnimationFrame(frame);
    else { animation = undefined; finishAnimation = undefined; }
  }
  animation = requestAnimationFrame(frame);
}
function animateReconstruction(startPoints) {
  reconstructionStart = startPoints;
  const finish = () => { reconstructionMix = 1; render(); };
  if (motion.matches) { finish(); return; }
  reconstructionMix = 0;
  const start = performance.now();
  finishAnimation = finish;
  render();
  function frame(time) {
    const t = Math.min(1, (time - start) / 1600);
    reconstructionMix = t * t * (3 - 2 * t);
    render();
    if (t < 1) animation = requestAnimationFrame(frame);
    else { animation = undefined; finishAnimation = undefined; finish(); }
  }
  animation = requestAnimationFrame(frame);
}
function entranceFor(index) {
  const step = steps[index];
  if (step.dataset.scene === 'explore' && dataset === 'three' && view === 'svd') return { svdMix: 0 };
  if (step.hasAttribute('data-motion-basis')) return { basisReveal: 0 };
  if (step.hasAttribute('data-motion-coordinates')) return { coordinateMix: 0 };
  if (step.hasAttribute('data-motion-drop')) return { dropMix: 0 };
  if (step.hasAttribute('data-motion-face-on')) return { faceOn: 0 };
  if (step.hasAttribute('data-motion-second-removal')) return { secondRemoval: 0 };
  if (step.hasAttribute('data-motion-reconstruction')) return { reconstructionMix: 0 };
  if (step.hasAttribute('data-motion-svd')) return { svdMix: 0 };
  if (step.hasAttribute('data-motion-assembly')) return { assemblyMix: 0 };
  if (step.hasAttribute('data-motion-projection')) return { projectionMix: 0 };
  if (step.hasAttribute('data-motion-removal')) return { removal: 0 };
  if (step.hasAttribute('data-motion-camera')) return { camera: Number(step.dataset.motionCamera) };
  if (step.hasAttribute('data-motion-angle-from')) return { angle: Number(step.dataset.motionAngleFrom) };
  return null;
}
function prepareEntrance() {
  const from = entranceFor(current);
  if (!from || entrancePlayed.has(current)) return;
  const to = { angle: targetAngle, projectionMix: 1, removal, camera, basisReveal: 1, coordinateMix: 1, dropMix: 1, faceOn: 1, secondRemoval: 1, reconstructionMix: 1, svdMix: 1, assemblyMix: 1 };
  if ('removal' in from) to.removal = targetRemoval = 1;
  if ('camera' in from) to.camera = 25;
  const apply = state => {
    if ('angle' in state) angle = state.angle;
    if ('projectionMix' in state) projectionMix = state.projectionMix;
    if ('removal' in state) removal = state.removal;
    if ('camera' in state) camera = state.camera;
    if ('basisReveal' in state) basisReveal = state.basisReveal;
    if ('coordinateMix' in state) coordinateMix = state.coordinateMix;
    if ('dropMix' in state) dropMix = state.dropMix;
    if ('faceOn' in state) faceOn = state.faceOn;
    if ('secondRemoval' in state) secondRemoval = state.secondRemoval;
    if ('reconstructionMix' in state) reconstructionMix = state.reconstructionMix;
    if ('svdMix' in state) svdMix = state.svdMix;
    if ('assemblyMix' in state) assemblyMix = state.assemblyMix;
    render();
  };
  const finish = () => {
    apply(to);
    entrancePlayed.add(current);
    entrancePending = undefined;
    remember();
  };
  if (motion.matches) { finish(); return; }
  entrancePending = { index: current, from, to, apply, finish };
  apply(from);
}
function maybePlayEntrance(force = false) {
  const pending = entrancePending;
  if (!pending || pending.index !== current || animation) return;
  const rect = stages[current].querySelector('.diagram-wrap').getBoundingClientRect();
  const visible = Math.max(0, Math.min(rect.bottom, innerHeight) - Math.max(rect.top, 76));
  // Begin when the graphic is readable, rather than while its prose is still entering.
  if (!force && (visible < rect.height * .65 || rect.top > innerHeight * .65)) return;
  entrancePending = undefined;
  entrancePlayed.add(current);
  const index = current;
  const start = performance.now();
  const duration = 'assemblyMix' in pending.from ? 6000 : 'faceOn' in pending.from ? 2400 : 'basisReveal' in pending.from ? 1400 : 2000;
  if (index < guidedCount) timedScrollProgress = 0;
  finishAnimation = () => {
    if (index < guidedCount) timedScrollProgress = 1;
    pending.finish();
  };
  function frame(time) {
    if (current !== index) return;
    const t = Math.min(1, (time - start) / duration);
    const eased = 'assemblyMix' in pending.from ? t : t * t * (3 - 2 * t);
    if (index < guidedCount) timedScrollProgress = eased;
    const state = {};
    for (const key of Object.keys(pending.from)) state[key] = pending.from[key] + eased * (pending.to[key] - pending.from[key]);
    pending.apply(state);
    if (t < 1) animation = requestAnimationFrame(frame);
    else {
      animation = undefined;
      finishAnimation = undefined;
      pending.finish();
      $('status').textContent = notationText($('caption'));
    }
  }
  animation = requestAnimationFrame(frame);
}
function setStep(index, animate = true) {
  const changed = current !== index;
  const sourceCamera = camera;
  const sourceScene = steps[current].dataset.scene;
  // Finish the old figure before changing the globals used by its drawing functions.
  stopAnimation(true);
  if (entrancePending) {
    entrancePending.apply(entrancePending.to);
    remember();
  }
  entrancePending = undefined;
  if (changed) remember();
  current = index;
  scrubTransition = undefined;
  scrubProgress = -1;
  timedScrollProgress = undefined;
  showMovieStep(index);
  projectionMix = 1;
  basisReveal = coordinateMix = dropMix = faceOn = secondRemoval = reconstructionMix = svdMix = assemblyMix = 1;
  reconstructionStart = undefined;
  svg = $('diagram');
  const scene = steps[index].dataset.scene;
  const fallback = {
    angle: Number(steps[index].dataset.fixedAngle ?? steps[index].dataset.startAngle ?? (scene === 'residual-plane' || scene === 'line' ? 0 : scene === 'second-pc' ? -30 : 45)),
    dataset: steps[index].dataset.example ?? (['line', 'scores', 'variance', 'projection', 'quadratic', 'maximum', 'basis', 'coordinates'].includes(scene) ? 'line' : ['cloud', 'reconstruction', 'explore'].includes(scene) ? 'cloud' : 'three'),
    view: scene === 'plane' ? 'reconstruction' : 'original',
    components: scene === 'plane' ? 2 : 1,
    camera: scene === 'residual-plane' && sourceScene === 'deflation' ? sourceCamera : scene === 'plane' ? -25 : 25,
    removal: 0,
    piece: 0,
  };
  // The guided story always returns to its authored demonstration. Only the
  // final explorer preserves manual choices between visits.
  const state = { ...(scene === 'explore' ? saved.get(index) || fallback : fallback) };
  if (scene === 'deflation') state.removal = 1;
  if (scene === 'residual-plane' && sourceScene === 'deflation' && !entrancePlayed.has(index)) state.camera = sourceCamera;
  dataset = state.dataset;
  view = state.view;
  components = state.components;
  camera = state.camera;
  removal = targetRemoval = state.removal;
  piece = state.piece;
  $('step-select').value = index;
  $('step-count').textContent = `${String(index + 1).padStart(2, '0')} / ${steps.length}`;
  $('previous').disabled = index === 0;
  $('next').textContent = index === steps.length - 1 ? 'Takeaways' : 'Next';
  changeAngle(Number(steps[index].dataset.fixedAngle ?? state.angle));
  if (animate) prepareEntrance();
}
function readingLine() {
  return Math.max(110, innerHeight * .3);
}
function releaseNavigation() {
  if (navigationLocked && current < guidedCount && timedScrollProgress !== undefined) {
    // Timed playback and scrolling share the same playhead. Resume at the frame
    // the reader just watched, rather than at the chapter's initial scroll mark.
    const local = scrollLocalAtProgress(timedScrollProgress);
    stopAnimation();
    const top = steps[current].getBoundingClientRect().top + scrollY - document.querySelector('.masthead').getBoundingClientRect().height;
    const height = steps[current].getBoundingClientRect().height;
    // Passive wheel/touch handlers may run after the browser has already moved
    // the document. Preserve that movement as well as the displayed playhead.
    const alreadyScrolled = scrollY - (top + scrubLocal * height);
    scrubLocal = local;
    window.scrollTo({ top: top + local * height + alreadyScrolled, behavior: 'instant' });
  }
  timedScrollProgress = undefined;
  navigationLocked = false;
  navigationUntil = 0;
}
function goToCover(push = true) {
  stopAnimation(true);
  if (entrancePending) entrancePending.finish();
  navigationLocked = false;
  navigationUntil = 0;
  inGuidedSequence = false;
  timedScrollProgress = undefined;
  if (push && location.hash !== '#page-title') history.pushState(null, '', '#page-title');
  window.scrollTo({ top: 0, behavior: 'instant' });
}
function goTo(index, push = true, focus = true) {
  if (index >= steps.length) return goToTakeaways(push, focus);
  index = Math.max(0, Math.min(steps.length - 1, index));
  navigationUntil = performance.now() + 100;
  navigationLocked = true;
  entrancePlayed.delete(index);
  setStep(index);
  inGuidedSequence = index < guidedCount;
  scrubLocal = .12;
  if (push && location.hash !== `#${steps[index].id}`) history.pushState(null, '', `#${steps[index].id}`);
  const mobile = matchMedia('(max-width: 800px)').matches;
  const alignment = index < guidedCount ? document.querySelector('.masthead').getBoundingClientRect().height : mobile ? 76 : 84;
  const target = steps[index].getBoundingClientRect().top + scrollY - alignment + (index < guidedCount ? steps[index].getBoundingClientRect().height * .12 : 5);
  window.scrollTo({ top: Math.max(0, target), behavior: 'instant' });
  requestAnimationFrame(() => maybePlayEntrance(index < guidedCount));
  if (focus) {
    const heading = index < guidedCount ? movieCopy.querySelector('h2') : steps[index].querySelector('h2');
    heading.tabIndex = -1;
    heading.focus({ preventScroll: true });
  }
}
function goToTakeaways(push = true, focus = true) {
  inGuidedSequence = false;
  navigationUntil = performance.now() + 100;
  navigationLocked = true;
  setStep(steps.length - 1, false);
  if (push && location.hash !== '#takeaways') history.pushState(null, '', '#takeaways');
  const section = document.getElementById('takeaways');
  const alignment = matchMedia('(max-width: 800px)').matches ? 76 : 84;
  window.scrollTo({ top: Math.max(0, section.getBoundingClientRect().top + scrollY - alignment + 5), behavior: 'instant' });
  if (focus) {
    const heading = section.querySelector('h2');
    heading.tabIndex = -1;
    heading.focus({ preventScroll: true });
  }
}
function readScroll() {
  scrollFrame = undefined;
  if (initializing) return;
  if (navigationLocked || performance.now() < navigationUntil) { maybePlayEntrance(); return; }
  const lineY = readingLine();
  const details = document.getElementById('mathematical-details');
  if (details.getBoundingClientRect().top <= lineY + 5) {
    inGuidedSequence = false;
    if (!document.getElementById(location.hash.slice(1))?.closest('#mathematical-details')) history.replaceState(null, '', '#mathematical-details');
    return;
  }
  if (document.getElementById('takeaways').getBoundingClientRect().top <= lineY + 5) {
    inGuidedSequence = false;
    if (current !== steps.length - 1) setStep(steps.length - 1, false);
    if (location.hash !== '#takeaways') history.replaceState(null, '', '#takeaways');
    return;
  }
  if (steps[guidedCount].getBoundingClientRect().top > lineY + 5) {
    const offset = document.querySelector('.masthead').getBoundingClientRect().height - movieSequence.getBoundingClientRect().top;
    inGuidedSequence = offset >= 0;
    const state = scrollSceneAt(offset, steps[0].getBoundingClientRect().height, guidedCount);
    scrubLocal = state.local;
    if (state.index !== current) setStep(state.index, false);
    stopAnimation();
    if (motion.matches) {
      if (scrubProgress !== 1) {
        entrancePlayed.delete(current);
        prepareEntrance();
        scrubProgress = 1;
      }
    } else {
      if (!scrubTransition) {
        scrubProgress = -1;
        entrancePlayed.delete(current);
        prepareEntrance();
        scrubTransition = entrancePending;
        entrancePending = undefined;
      }
      if (scrubTransition && Math.abs(state.progress - scrubProgress) > .00001) {
        const values = {};
        for (const key of Object.keys(scrubTransition.from)) values[key] = scrubTransition.from[key] + state.progress * (scrubTransition.to[key] - scrubTransition.from[key]);
        scrubTransition.apply(values);
      }
      scrubProgress = state.progress;
    }
    if (offset >= 0 && location.hash !== `#${steps[current].id}`) history.replaceState(null, '', `#${steps[current].id}`);
    return;
  }
  inGuidedSequence = false;
  let index = 0;
  steps.forEach((step, i) => { if (step.getBoundingClientRect().top <= lineY + 5) index = i; });
  if (index !== current || location.hash === '#takeaways') {
    if (index !== current) setStep(index);
    if (scrollY > document.querySelector('.title-page').offsetHeight) history.replaceState(null, '', `#${steps[index].id}`);
  }
  maybePlayEntrance();
}
stages.forEach((stage, index) => {
  const local = id => stage.querySelector(`[data-control="${id}"]`);
  const activate = () => {
    if (current !== index) setStep(index, false);
    stopAnimation(true);
    if (entrancePending) entrancePending.finish();
    removal = targetRemoval;
    basisReveal = coordinateMix = dropMix = faceOn = secondRemoval = reconstructionMix = svdMix = assemblyMix = 1;
    entrancePending = undefined;
    entrancePlayed.add(index);
    projectionMix = 1;
    targetAngle = angle;
  };
  steps.forEach((step, optionIndex) => {
    const option = document.createElement('option');
    option.value = optionIndex;
    option.textContent = `${String(optionIndex + 1).padStart(2, '0')} · ${step.dataset.title}`;
    local('step-select').append(option);
  });
  local('step-select').addEventListener('change', event => {
    const destination = Number(event.target.value);
    local('step-select').value = index;
    goTo(destination);
  });
  local('previous').addEventListener('click', () => goTo(index - 1));
  local('next').addEventListener('click', () => goTo(index + 1));
  local('motion-replay').addEventListener('click', () => {
    activate();
    navigationLocked = true;
    scrubTransition = undefined;
    targetAngle = Number(steps[index].dataset.fixedAngle ?? steps[index].dataset.startAngle ?? (steps[index].dataset.scene === 'second-pc' ? -30 : 45));
    if (!steps[index].hasAttribute('data-motion-angle-from')) angle = targetAngle;
    reconstructionStart = undefined;
    if (steps[index].hasAttribute('data-motion-camera')) camera = 25;
    entrancePlayed.delete(index);
    prepareEntrance();
    maybePlayEntrance(true);
  });
  local('angle').addEventListener('input', event => {
    const value = Number(event.target.value);
    activate(); changeAngle(value); remember();
  });
  for (const id of ['angle', 'camera']) {
    const takeControl = () => { activate(); remember(); render(); };
    local(id).addEventListener('pointerdown', takeControl);
    local(id).addEventListener('keydown', event => {
      if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End', 'PageUp', 'PageDown'].includes(event.key)) takeControl();
    });
  }
  local('angle').addEventListener('change', () => {
    activate(); $('status').textContent = notationText($('caption'));
  });
  stage.querySelectorAll('[data-angle]').forEach(button => button.addEventListener('click', () => {
    const value = Number(button.dataset.angle);
    activate(); changeAngle(value, true); remember();
    $('status').textContent = `Direction set to ${targetAngle} degrees.`;
  }));
  for (const id of ['dataset', 'view', 'components', 'camera', 'piece']) {
    local(id).addEventListener(id === 'camera' ? 'input' : 'change', event => {
      const value = event.target.value;
      activate();
      const previousReconstruction = id === 'components' && steps[index].dataset.scene === 'plane'
        ? reconstructBasis(changeBasis(THREE_D_POINTS, THREE_D_BASIS), THREE_D_BASIS, components) : null;
      if (id === 'dataset') { dataset = value; view = 'original'; components = dataset === 'three' ? 2 : 1; }
      if (id === 'view') view = value;
      if (id === 'components') components = Number(value);
      if (id === 'camera') camera = Number(value);
      if (id === 'piece') piece = Number(value);
      remember(); render();
      if (previousReconstruction) animateReconstruction(previousReconstruction);
      if ((id === 'piece' || id === 'view') && dataset === 'three' && view === 'svd') {
        entrancePlayed.delete(index);
        prepareEntrance();
        maybePlayEntrance(true);
      }
      if (id !== 'camera') $('status').textContent = notationText($('caption'));
    });
  }
});
document.querySelectorAll('a[href^="#"]').forEach(link => link.addEventListener('click', event => {
  const index = steps.findIndex(step => `#${step.id}` === link.getAttribute('href'));
  if (index >= 0) { event.preventDefault(); goTo(index); }
  else if (link.getAttribute('href') === '#takeaways') { event.preventDefault(); goToTakeaways(); }
  else if (link.getAttribute('href') === '#page-title') { event.preventDefault(); goToCover(); }
  else {
    const detail = document.getElementById(link.getAttribute('href').slice(1));
    if (detail?.tagName === 'DETAILS') detail.open = true;
  }
}));
document.addEventListener('keydown', event => {
  if (event.altKey || event.ctrlKey || event.metaKey || event.target.closest('input, select, textarea, [contenteditable]')) return;
  if (event.target.closest('button')) {
    if (['PageDown', 'PageUp', 'ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) releaseNavigation();
    return;
  }
  if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
    event.preventDefault();
    if (location.hash === '#takeaways' && event.key === 'ArrowLeft') goTo(steps.length - 1);
    else goTo(current + (event.key === 'ArrowRight' ? 1 : -1));
  }
  if (['PageDown', 'PageUp', 'ArrowDown', 'ArrowUp', 'Home', 'End', ' '].includes(event.key)) releaseNavigation();
});
// Preserve an explicit chapter selection until the reader scrolls intentionally.
addEventListener('wheel', releaseNavigation, { passive: true });
addEventListener('touchmove', event => {
  if (!event.target.closest('input, select, button')) releaseNavigation();
}, { passive: true });
addEventListener('pointerdown', event => {
  if (event.clientX >= document.documentElement.clientWidth) navigationLocked = false;
}, { passive: true });
addEventListener('scroll', () => { if (!scrollFrame) scrollFrame = requestAnimationFrame(readScroll); }, { passive: true });
function renderAll() {
  const selected = current;
  for (let index = 0; index < steps.length; index++) setStep(index, false);
  setStep(selected, false);
}
addEventListener('resize', () => {
  const index = current;
  const preserveGuidedPosition = inGuidedSequence;
  const local = scrubLocal;
  const locked = navigationLocked;
  const focusCopy = movieCopy.contains(document.activeElement);
  stopAnimation(true);
  if (entrancePending) entrancePending.finish();
  renderAll();
  measureMovieCopy();
  if (index < guidedCount && preserveGuidedPosition) {
    const top = steps[index].getBoundingClientRect().top + scrollY - document.querySelector('.masthead').getBoundingClientRect().height;
    window.scrollTo({ top: Math.max(0, top + local * steps[index].getBoundingClientRect().height), behavior: 'instant' });
    navigationLocked = locked;
    navigationUntil = locked ? performance.now() + 100 : 0;
    if (locked) timedScrollProgress = 1;
  }
  if (focusCopy) {
    const heading = movieCopy.querySelector('h2');
    heading.tabIndex = -1;
    heading.focus({ preventScroll: true });
  }
  if (!locked) readScroll();
});
motion.addEventListener('change', () => {
  if (motion.matches) {
    stopAnimation(true);
    if (entrancePending) entrancePending.finish();
    angle = targetAngle;
    if (current < guidedCount) { entrancePlayed.delete(current); prepareEntrance(); }
    render();
  }
  scrubProgress = -1;
  scrubTransition = undefined;
  readScroll();
});
function restoreHash() {
  const index = steps.findIndex(step => `#${step.id}` === location.hash);
  if (index >= 0) goTo(index, false, false);
  else if (location.hash === '#takeaways') goToTakeaways(false, false);
  else if (document.getElementById(location.hash.slice(1))?.closest('#mathematical-details')) {
    inGuidedSequence = false;
    stopAnimation(true);
    const target = document.getElementById(location.hash.slice(1));
    if (target.tagName === 'DETAILS') target.open = true;
    navigationLocked = true;
    target.scrollIntoView({ block: 'start', behavior: 'instant' });
  }
  else if (!location.hash || location.hash === '#page-title') goToCover(false);
}
addEventListener('popstate', restoreHash);
renderAll();
const pageLoaded = document.readyState === 'complete' ? Promise.resolve() : new Promise(resolve => addEventListener('load', resolve, { once: true }));
Promise.all([document.fonts.ready, pageLoaded]).then(() => requestAnimationFrame(() => {
  measureMovieCopy();
  initializing = false;
  if (location.hash) restoreHash();
  else { prepareEntrance(); readScroll(); }
}));
