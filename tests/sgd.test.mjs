import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import test from 'node:test';
import vm from 'node:vm';
import { copy, data, gradient, initialize, loss, mse, predict, ramp, random, shuffle, target, update } from '../docs/sgd/model.mjs';
import { epochState, renderEpochCounter } from '../docs/sgd/epoch-counter.mjs';
import { scenes, foundationScenes, batchSteps } from '../docs/sgd/scenes.mjs';
import { render } from '../docs/sgd/views.mjs';
import {renderText,plainText} from '../docs/sgd/math.mjs';
import { constructionKinds, constructionState } from '../docs/sgd/construction.mjs';

globalThis.katex = createRequire(import.meta.url)('../docs/neural-networks/vendor/katex/katex.min.js');

const experiment = JSON.parse(readFileSync(new URL('../docs/sgd/experiment.json', import.meta.url)));
const close = (actual, expected, tolerance = 1e-12) =>
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} should equal ${expected}`);

// Inspect the real SVG output and numerical frame without making layout claims.
function renderScene(id, progress, width = 850, height = 430, detailHeight = 110) {
  const svg = (name, h) => ({
    id: name, innerHTML: '', attrs: {},
    getBoundingClientRect() { return { width, height: h }; },
    setAttribute(key, value) { this.attrs[key] = value; }
  });
  const plot = svg('plot', height), detail = svg('detail', detailHeight);
  const scene = scenes.find(item => item.id === id);
  assert.ok(scene, `Unknown scene ${id}`);
  const frame = render(plot, detail, scene, progress, experiment);
  assert.doesNotMatch(plot.innerHTML + detail.innerHTML, /NaN|Infinity|undefined/);
  return { frame, plot: plot.innerHTML, detail: detail.innerHTML, plotLabel: plot.attrs['aria-label'] };
}

test('the static opening introduces the noiseless training data before constructing ReLUs', async () => {
  assert.equal(scenes[0].id, 'step-target');
  assert.equal(scenes[0].kind, 'step-target');
  assert.equal(scenes[1].id, 'two-hinges');
  assert.equal(scenes[2].id, 'pair');
  const opening = renderScene('step-target', 0);
  const staticFrame = ({ progress, ...frame }) => frame;
  assert.equal(opening.detail, '', 'The data-generating process comes before any network diagram');
  assert.equal(opening.frame.dataCount, data.length);
  assert.equal(opening.frame.dataCount, 2048);
  const shown = opening.frame.shownPoints;
  assert.ok(shown.length > 0 && shown.length < data.length, 'The display is a subset of the training observations');
  assert.equal(new Set(shown.map(point => point.index)).size, shown.length);
  assert.deepEqual([...new Set(shown.map(point => point.y))].sort(), [0, 1]);
  const plotted = [...opening.plot.matchAll(/<circle\b[^>]*data-observation="(\d+)"[^>]*>/g)]
    .map(match => Object.fromEntries([...match[0].matchAll(/([\w-]+)="([^"]*)"/g)].map(attribute => attribute.slice(1))));
  assert.equal(plotted.length, shown.length);
  for (const point of shown) {
    assert.deepEqual({ x: point.x, y: point.y }, data[point.index]);
    assert.equal(point.y, target(point.x));
    const dot = plotted.find(item => Number(item['data-observation']) === point.index);
    assert.ok(dot, `Observation ${point.index} must be plotted`);
    assert.equal(Number(dot['data-x']), point.x);
    assert.equal(Number(dot['data-y']), point.y);
  }
  for (const progress of [.25, .5, .75, 1, .75, .25, 0]) {
    const result = renderScene('step-target', progress);
    assert.deepEqual({ ...result, frame: staticFrame(result.frame) }, { ...opening, frame: staticFrame(opening.frame) }, 'The standalone DGP slide remains static while scrolling');
  }
  const marker = kind => {
    const element = opening.plot.match(new RegExp(`<circle\\b[^>]*data-target-boundary="${kind}"[^>]*>`))?.[0];
    assert.ok(element, `${kind} must be visibly marked`);
    return Object.fromEntries([...element.matchAll(/([\w-]+)="([^"]*)"/g)].map(match => match.slice(1)));
  };
  const closed = marker('closed-zero'), open = marker('open-one');
  assert.equal(closed.cx, open.cx);
  assert.ok(Number(closed.cy) > Number(open.cy), 'Both threshold endpoints share x=0, with one above zero');
  assert.ok(closed.fill && !['white', '#fff', '#ffffff', 'none'].includes(closed.fill));
  assert.equal(open.fill, 'white');
  for (const dot of plotted) {
    assert.equal(Number(dot.cy), Number(dot['data-y']) === 0 ? Number(closed.cy) : Number(open.cy), 'Every displayed observation lies exactly on its target branch');
  }
  assert.match(opening.plotLabel, /step/i);
  assert.match(opening.plotLabel, /x\s*(?:≤|<=)\s*0/);
  assert.match(opening.plotLabel, /x\s*>\s*0/);
  assert.match(opening.plotLabel, /no noise/i);
  assert.match(opening.plotLabel, /evenly spaced/i);

  const ui = await controller();
  const first = staticFrame(ui.at('step-target', .25));
  assert.deepEqual(staticFrame(ui.at('step-target', .75)), first);
  assert.deepEqual(staticFrame(ui.at('step-target', .25)), first);
  assert.equal(ui.nodes.get('epoch-counter').hidden, true);
  ui.reduce(true);
  assert.deepEqual(staticFrame(ui.at('step-target', .1)), first);
  ui.emit('document:keydown', { key: 'ArrowRight', target: { tagName: 'DIV' }, preventDefault() {} });
  assert.equal(ui.nodes.get('stage').dataset.scene, 'two-hinges');
  ui.emit('document:keydown', { key: 'ArrowRight', target: { tagName: 'DIV' }, preventDefault() {} });
  assert.equal(ui.nodes.get('stage').dataset.scene, 'pair');
});

test('the width-two approximation steepens reversibly while the target step stays fixed', async () => {
  const scene = scenes.find(item => item.id === 'two-hinges');
  assert.equal(scene.title, 'Width of Two is Mathematically Sufficient');
  assert.equal(scene.caption, '');
  assert.equal(scene.note, '');
  const progressStops = [0, .25, .5, .75, 1];
  const frames = progressStops.map(progress => renderScene('two-hinges', progress));
  const targetPaths = result => result.plot.match(/<path\b[^>]*data-target="step"[^>]*>/g) || [];
  const reference = targetPaths(frames[0]);
  assert.equal(reference.length, 1);
  close(frames[0].frame.steepness, 2);
  close(frames.at(-1).frame.steepness, 24);
  let previousSlope = 0, previousMSE = Infinity;
  for (const preview of frames) {
    const { model, width, hinges, regionSlopes, steepness } = preview.frame;
    assert.equal(width, 2);
    assert.equal(model.units.length, 2, 'A sharper approximation must not require additional neurons');
    assert.equal(preview.detail, '');
    assert.deepEqual(targetPaths(preview), reference, 'Only the approximation moves; the target step is fixed');
    assert.doesNotMatch(preview.plot, /Flat|Rising|Hinge \d|data-hinge=|data-region=/, 'The plot compares two functions without hinge or region annotations');
    assert.match(preview.plot, /data-network="two-hinges"/);
    assert.deepEqual(hinges, [0, 1 / steepness]);
    assert.deepEqual(regionSlopes, [0, steepness, 0]);
    assert.deepEqual(model.units.map(unit => -unit.b / unit.w || 0), hinges);
    const epsilon = 1e-6;
    for (const x of [-1, -.2, 0, .25 / steepness, .5 / steepness, 1 / steepness, .75, 1]) {
      close(predict(model, x), ramp(x, steepness));
    }
    for (const [index, x] of [-.5, .5 / steepness, .75].entries()) {
      close((predict(model, x + epsilon) - predict(model, x - epsilon)) / (2 * epsilon), regionSlopes[index], 2e-9);
    }
    for (const hinge of hinges) {
      assert.ok(Math.abs(predict(model, hinge - epsilon) - predict(model, hinge + epsilon)) <= steepness * epsilon + 1e-12, 'The two ReLUs remain continuous at both joins');
    }
    close(predict(model, 0), 0);
    close(predict(model, 1 / steepness), 1);
    close(predict(model, .5 / steepness), .5);
    assert.notEqual(predict(model, .5 / steepness), target(.5 / steepness), 'A finite ramp approximates the discontinuity without claiming exact equality');
    const approximationMSE = mse(model);
    assert.ok(steepness > previousSlope, 'Forward scrolling makes the ramp steeper');
    assert.ok(approximationMSE < previousMSE, 'The fixed-width approximation gets closer to the true target');
    previousSlope = steepness;
    previousMSE = approximationMSE;
  }
  const networkPath = result => result.plot.match(/<path\b[^>]*data-network="two-hinges"[^>]*>/)?.[0].match(/\bd="([^"]+)"/)?.[1];
  assert.ok(networkPath(frames[0]) && networkPath(frames.at(-1)));
  assert.notEqual(networkPath(frames[0]), networkPath(frames.at(-1)), 'The visible approximation must animate with the mathematical model');
  for (const [index, progress] of progressStops.entries()) {
    assert.deepEqual(renderScene('two-hinges', progress), frames[index]);
  }
  for (const [index, progress] of [...progressStops.entries()].reverse()) {
    assert.deepEqual(renderScene('two-hinges', progress), frames[index], 'Reverse scrolling restores the same curve');
  }
  const ui = await controller(), first = ui.at('two-hinges', .25);
  ui.at('two-hinges', .75);
  assert.deepEqual(ui.at('two-hinges', .25), first);
  ui.at('pair', .5);
  assert.deepEqual(ui.at('two-hinges', .25), first, 'Returning from the ReLU construction restores the same approximation');
  assert.equal(ui.nodes.get('epoch-counter').hidden, true);
  ui.reduce(true);
  const reduced = ui.at('two-hinges', .1);
  assert.deepEqual(reduced.model, frames.at(-1).frame.model, 'Reduced motion shows the final steep approximation');
  close(reduced.steepness, 24);
  assert.deepEqual(ui.at('two-hinges', .75), reduced);
});

test('the DGP cases brace uses the bundled large-delimiter font and every declared font resolves locally', () => {
  const opening = scenes.find(scene => scene.id === 'step-target');
  const html = globalThis.katex.renderToString(opening.tex, { displayMode: true, output: 'htmlAndMathml', throwOnError: true, strict: 'error' });
  assert.match(html, /class="[^"]*delimsizing[^"]*"/);
  assert.match(html, /class="[^"]*delim-size4[^"]*"/, 'The cases brace needs the Size4 glyphs');
  const styleUrl = new URL('../docs/neural-networks/vendor/katex/katex.min.css', import.meta.url);
  const css = readFileSync(styleUrl, 'utf8');
  assert.match(css, /\.delim-size4>span\{[^}]*font-family:KaTeX_Size4/);
  const faces = [...css.matchAll(/@font-face\s*\{([^}]+)\}/g)].map(match => match[1]);
  assert.ok(faces.some(face => /font-family:KaTeX_Size4(?:;|$)/.test(face)), 'The large-brace family must have an actual font-face declaration');
  for (const face of faces) {
    const sources = [...face.matchAll(/url\(["']?([^"')]+)["']?\)/g)].map(match => match[1]);
    assert.ok(sources.length > 0);
    for (const source of sources) {
      const fontUrl = new URL(source, styleUrl);
      assert.equal(fontUrl.protocol, 'file:', 'Math fonts must be bundled with the lesson');
      const font = readFileSync(fontUrl);
      assert.equal(font.subarray(0, 4).toString('ascii'), 'wOF2', `${source} must contain a WOFF2 font`);
      assert.equal(font.readUInt32BE(8), font.length, `${source} must not be truncated`);
    }
  }
});

test('the target reference accompanies the combined output without obscuring the individual ReLUs', () => {
  const targetPath = result => result.plot.match(/<path data-target="step"[^>]*>/g) || [];
  for (const id of ['pair', 'input-first', 'input-second', 'flip']) {
    for (const progress of [0, .5, 1]) assert.equal(targetPath(renderScene(id, progress)).length, 0);
  }
  for (const progress of [0, .3]) assert.equal(targetPath(renderScene('add', progress)).length, 0);
  const revealed = renderScene('add', .5);
  assert.ok(revealed.frame.reveal > 0);
  assert.equal(targetPath(revealed).length, 1);
  assert.match(revealed.plotLabel, /target step/);
  const reference = targetPath(renderScene('add', 1))[0];
  assert.ok(reference);
  for (const progress of [.5, .75, 1, .75, .5]) {
    assert.deepEqual(targetPath(renderScene('add', progress)), [reference], 'Revealing the sum must not move or duplicate the target');
  }
  const sum = renderScene('add', 1).frame;
  const next = scenes[scenes.findIndex(scene => scene.id === 'add') + 1];
  const steepening = renderScene(next.id, 0).frame;
  for (const x of [-.3, 0, .1, .5, 1, 1.2]) {
    const combined = sum.c + sum.a1 * Math.max(0, sum.w1 * x + sum.b1) + sum.a2 * Math.max(0, sum.w2 * x + sum.b2);
    close(combined, ramp(x, steepening.a));
  }
  assert.equal(targetPath(renderScene('add', .3)).length, 0, 'Reverse scrolling restores the separate-contribution view');
});

test('adding the ReLUs reveals their complete sum without a moving point or vertical measurement bars', () => {
  const scene = scenes.find(item => item.id === 'add');
  assert.equal(scene.caption, '');
  assert.equal(scene.note, '');
  assert.ok(scene.reading, 'The reading view retains an explanation of the sum');
  const attributes = element => Object.fromEntries([...element.matchAll(/([\w-]+)="([^"]*)"/g)].map(match => match.slice(1)));
  const curve = (preview, color) => {
    const paths = (preview.plot.match(/<path\b[^>]*>/g) || []).map(attributes).filter(path => path.stroke === color);
    assert.equal(paths.length, 1, `Exactly one ${color} curve is visible`);
    return paths[0];
  };
  const points = path => [...path.d.matchAll(/[ML]([-\d.e+]+),([-\d.e+]+)/g)].map(match => match.slice(1).map(Number));
  let completeSum;
  for (const progress of [.4, .55, .7, 1, .7, .4]) {
    const preview = renderScene('add', progress);
    assert.ok(preview.frame.reveal > 0);
    const sum = curve(preview, '#007f73');
    const first = curve(preview, '#234e70'), second = curve(preview, '#a54f32');
    const sumPoints = points(sum), firstPoints = points(first), secondPoints = points(second);
    assert.ok(sumPoints.length > 2, 'The sum is a curve rather than a moving endpoint');
    assert.equal(sumPoints.length, firstPoints.length);
    assert.equal(sumPoints.length, secondPoints.length);
    const zeroY = sumPoints[0][1];
    for (let i = 0; i < sumPoints.length; i++) {
      close(sumPoints[i][0], firstPoints[i][0]);
      close(sumPoints[i][0], secondPoints[i][0]);
      close(sumPoints[i][1], firstPoints[i][1] + secondPoints[i][1] - zeroY);
    }
    assert.ok(sumPoints.at(-1)[0] > sumPoints[0][0]);
    assert.ok(sumPoints.at(-1)[1] < zeroY, 'The summed curve rises to its upper plateau');
    completeSum ??= sum.d;
    assert.equal(sum.d, completeSum, 'The entire sum is shown throughout the fade, including on reverse scrolling');
    close(Number(sum['stroke-opacity']), preview.frame.reveal);
    for (const component of [first, second]) {
      assert.ok(component['stroke-dasharray'], 'Both weighted ReLUs remain visible as dashed curves');
      assert.ok(Number(component['stroke-opacity']) > 0);
    }
    const circles = (preview.plot.match(/<circle\b[^>]*>/g) || []).map(attributes);
    assert.equal(circles.length, 2, 'Only the fixed open and closed target endpoints remain');
    assert.ok(circles.every(circle => circle.fill !== '#007f73'), 'There is no moving teal endpoint');
    const coloredMeasurements = (preview.plot.match(/<line\b[^>]*>/g) || []).map(attributes)
      .filter(line => ['#234e70', '#a54f32'].includes(line.stroke));
    assert.deepEqual(coloredMeasurements, [], 'The figure no longer measures individual heights with vertical bars');
    assert.doesNotMatch(preview.plot, />\s*[-\d.]+\s*[+−]\s*[-\d.]+\s*=\s*[-\d.]+\s*</);
    assert.doesNotMatch(preview.detail, /Hidden layer|>Output</);

    for (const figure of [preview.plot, preview.detail]) {
      assert.match(figure, /<foreignObject\b/);
      assert.match(figure, /construction-math/);
      assert.match(figure, /class="katex"/);
      assert.match(figure, /<annotation encoding="application\/x-tex">/);
      assert.doesNotMatch(figure, /<text\b[^>]*>[^<]*[θφΣ₀₁₂]/, 'Math symbols must be typeset rather than assembled with Unicode subscripts');
    }
  }
});

const svgAttributes = element => Object.fromEntries([...element.matchAll(/([\w-]+)="([^"]*)"/g)].map(match => match.slice(1)));
const figureGeometry = figure => [...figure.replace(/<foreignObject\b[\s\S]*?<\/foreignObject>/g, '').matchAll(/<(path|line|circle)\b[^>]*>/g)]
  .map(match => ({ tag: match[1], ...svgAttributes(match[0]) }))
  .filter(element => !Object.hasOwn(element, 'data-observation'))
  .map(element => Object.fromEntries(['tag', 'd', 'x1', 'y1', 'x2', 'y2', 'cx', 'cy', 'r'].filter(key => key in element).map(key => [key, element[key]])));
const targetCoordinates = preview => {
  const element = preview.plot.match(/<path\b[^>]*data-target="step"[^>]*>/)?.[0];
  assert.ok(element, 'The target remains visible while the framing changes');
  return [...svgAttributes(element).d.matchAll(/-?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?/gi)].map(match => Number(match[0]));
};

test('the sum keeps its graph and network geometry across both construction-to-data boundaries', () => {
  assert.ok(constructionKinds.includes('steepen'));
  assert.ok(constructionKinds.includes('sgd-opening'));
  const ids = scenes.map(scene => scene.id);
  assert.equal(ids[ids.indexOf('add') + 1], 'steepen');
  assert.equal(ids[ids.indexOf('steepen') + 1], 'sgd-transition');
  for (const [from, to] of [['add', 'steepen'], ['steepen', 'sgd-transition']]) {
    for (const [width, height] of [[850, 430], [288, 240]]) {
      const before = renderScene(from, 1, width, height), after = renderScene(to, 0, width, height);
      assert.deepEqual(figureGeometry(after.plot), figureGeometry(before.plot), `${from} → ${to}: the graph must not jump at the boundary`);
      assert.deepEqual(figureGeometry(after.detail), figureGeometry(before.detail), `${from} → ${to}: the network must not jump at the boundary`);
      for (const parameter of ['w1', 'b1', 'w2', 'b2', 'a1', 'a2', 'c']) {
        close(after.frame[parameter], before.frame[parameter]);
      }
    }
  }
});

test('the numeric slope becomes k before the ramp steepens and the framing changes gradually', () => {
  const start = renderScene('steepen', 0), substitution = renderScene('steepen', .22), framed = renderScene('steepen', .38), end = renderScene('steepen', 1);
  close(start.frame.a, 2);
  close(start.frame.symbol, 0);
  close(start.frame.reframe, 0);
  close(substitution.frame.symbol, 1);
  close(framed.frame.reframe, 1);
  close(end.frame.a, 32);
  const bounds = [targetCoordinates(start), targetCoordinates(framed)];
  assert.notDeepEqual(bounds[0], bounds[1], 'The new framing must actually move the visible axes and target');
  let previousSlope = 2;
  for (const progress of [0, .04, .1, .16, .22, .3, .38, .4, .55, .7, .88, 1]) {
    const preview = renderScene('steepen', progress), frame = preview.frame;
    close(frame.w1, frame.a);
    close(frame.w2, frame.a);
    close(frame.b1, 0);
    close(frame.b2, -1);
    close(frame.a1, 1);
    close(frame.a2, -1);
    close(frame.c, 0);
    close(frame.dataReveal, 0);
    if (progress <= .4) close(frame.a, 2);
    if (progress > .4 && progress < .88) assert.ok(frame.a > previousSlope && frame.a < 32, 'Steepening starts after the substitution and continues smoothly');
    previousSlope = frame.a;
    if (progress > .1 && progress < .38) {
      assert.ok(frame.reframe > 0 && frame.reframe < 1);
      const coordinates = targetCoordinates(preview);
      assert.equal(coordinates.length, bounds[0].length);
      coordinates.forEach((value, i) => {
        assert.ok(value >= Math.min(bounds[0][i], bounds[1][i]) - 1e-9 && value <= Math.max(bounds[0][i], bounds[1][i]) + 1e-9, 'Intermediate target coordinates remain between the two framings');
      });
      assert.notDeepEqual(coordinates, bounds[0]);
      assert.notDeepEqual(coordinates, bounds[1]);
    }
    for (const x of [-1, 0, .5 / frame.a, 1 / frame.a, 1]) {
      close(frame.c + frame.a1 * Math.max(0, frame.w1 * x + frame.b1) + frame.a2 * Math.max(0, frame.w2 * x + frame.b2), ramp(x, frame.a));
    }
  }
  for (const id of ['steepen', 'sgd-transition']) {
    const scene = scenes.find(item => item.id === id);
    assert.equal(scene.caption, '');
    assert.equal(scene.note, '');
  }
});

test('the fitting introduction reveals real finite observations from the same DGP before any update', () => {
  const original = JSON.stringify(experiment), start = renderScene('sgd-transition', 0), end = renderScene('sgd-transition', 1);
  close(start.frame.dataReveal, 0);
  close(end.frame.dataReveal, 1);
  for (const progress of [0, .08, .2, .42, .75, 1, .75, .2, 0]) {
    const preview = renderScene('sgd-transition', progress), frame = preview.frame;
    assert.equal(frame.trainingDataCount, data.length);
    assert.equal(frame.trainingDataCount, 2048);
    assert.equal(frame.completedUpdates, 0);
    close(frame.a, 32);
    for (const parameter of ['w1', 'b1', 'w2', 'b2', 'a1', 'a2', 'c']) close(frame[parameter], start.frame[parameter]);
    if (progress > .08 && progress < .42) assert.ok(frame.dataReveal > 0 && frame.dataReveal < 1);
    const shown = frame.displayedObservations;
    assert.equal(shown.length, 32);
    assert.equal(new Set(shown.map(point => point.index)).size, shown.length);
    assert.deepEqual([...new Set(shown.map(point => point.y))].sort(), [0, 1]);
    const plotted = [...preview.plot.matchAll(/<circle\b[^>]*data-observation="(\d+)"[^>]*>/g)].map(match => svgAttributes(match[0]));
    assert.equal(plotted.length, frame.dataReveal > 0 ? shown.length : 0);
    const sum = svgAttributes(preview.plot.match(/<path\b[^>]*data-curve="sum"[^>]*>/)?.[0] || "");
    close(Number(sum['stroke-opacity']), 1 - frame.dataReveal);
    if (frame.dataReveal > 0) {
      const observationGroup = svgAttributes(preview.plot.match(/<g\b[^>]*data-observations="finite-data"[^>]*>/)?.[0] || "");
      close(Number(observationGroup.opacity), frame.dataReveal);
    }
    for (const point of shown) {
      assert.deepEqual({ x: point.x, y: point.y }, data[point.index]);
      assert.equal(point.y, target(point.x));
      if (frame.dataReveal === 0) continue;
      const dot = plotted.find(item => Number(item['data-observation']) === point.index);
      assert.ok(dot, `Observation ${point.index} must be plotted`);
      assert.equal(Number(dot['data-x']), point.x);
      assert.equal(Number(dot['data-y']), point.y);
      const coordinates = targetCoordinates(preview);
      close(Number(dot.cy), point.y === 0 ? coordinates[1] : coordinates[4]);
      close(Number(dot.cx), coordinates[0] + (point.x + 1) / 2 * (coordinates[5] - coordinates[0]));
    }
    if (frame.dataReveal > 0) assert.match(preview.detail, /32 of 2,048 observations shown/);
  }
  assert.equal(JSON.stringify(experiment), original, 'Introducing the fitting problem does not change training data or recorded updates');
});

test('scrolling reverses the construction-to-data transition and reduced motion completes its stages', async () => {
  const ui = await controller();
  const equationProgress = frame => {
    const style = ui.nodes.get('equation').style;
    close(Number(style['--coefficient-progress']), frame.symbol);
    close(Number(style['--data-progress']), frame.dataReveal);
  };
  const stops = [['add', .9], ['steepen', 0], ['steepen', .16], ['steepen', .3], ['steepen', .55], ['steepen', .9], ['sgd-transition', 0], ['sgd-transition', .2], ['sgd-transition', .75]];
  const seen = stops.map(([id, progress]) => {
    const frame = ui.at(id, progress);
    assert.equal(ui.nodes.get('stage').dataset.construction, 'true');
    equationProgress(frame);
    assert.equal(ui.nodes.get('epoch-counter').hidden, true, 'The fitting introduction has no training counter yet');
    return { id, progress, frame, plot: ui.nodes.get('plot').innerHTML, detail: ui.nodes.get('detail').innerHTML, equation: ui.nodes.get('equation').innerHTML };
  });
  for (const saved of seen.reverse()) {
    assert.deepEqual(ui.at(saved.id, saved.progress), saved.frame);
    equationProgress(saved.frame);
    assert.equal(ui.nodes.get('plot').innerHTML, saved.plot);
    assert.equal(ui.nodes.get('detail').innerHTML, saved.detail);
    assert.equal(ui.nodes.get('equation').innerHTML, saved.equation);
  }
  ui.reduce(true);
  for (const id of ['steepen', 'sgd-transition']) {
    const frame = ui.at(id, .1);
    close(frame.a, 32);
    close(frame.symbol, 1);
    close(frame.reframe, 1);
    close(frame.dataReveal, id === 'sgd-transition' ? 1 : 0);
    equationProgress(frame);
    assert.deepEqual(ui.at(id, .75), frame);
    equationProgress(frame);
  }
  ui.at('steepen', .1);
  ui.emit('document:keydown', { key: 'ArrowRight', target: { tagName: 'DIV' }, preventDefault() {} });
  assert.equal(ui.nodes.get('stage').dataset.scene, 'sgd-transition');
  close(JSON.parse(ui.nodes.get('stage').dataset.frame).dataReveal, 1);
  ui.emit('document:keydown', { key: 'ArrowLeft', target: { tagName: 'DIV' }, preventDefault() {} });
  assert.equal(ui.nodes.get('stage').dataset.scene, 'steepen');
  close(JSON.parse(ui.nodes.get('stage').dataset.frame).a, 32);
});

test('each of the first two epochs partitions the same data and preserves the preceding update', () => {
  for (const width of [2, 12]) {
    const trace = experiment.traces[width];
    const rng = random(10001);
    for (const epoch of [1, 2]) {
      const snapshots = trace.filter(snapshot => snapshot.epoch === epoch);
      assert.equal(snapshots.length, 32);
      const indices = snapshots.flatMap(snapshot => snapshot.indices);
      assert.equal(indices.length, data.length);
      assert.equal(new Set(indices).size, data.length);
      assert.deepEqual(indices, shuffle(rng));
      snapshots.forEach((snapshot, index) => {
        assert.equal(snapshot.batch, index + 1);
        assert.equal(snapshot.step, (epoch - 1) * 32 + index);
        assert.equal(snapshot.indices.length, 64);
        if (snapshot.step > 0) {
          assert.deepEqual(snapshot.model, trace[snapshot.step - 1].after);
        }
      });
    }
    assert.notDeepEqual(trace[0].indices, trace[32].indices);
  }
});

const parameters = model => model.units.flatMap(unit => [unit.w, unit.b, unit.a]).concat(model.c);
const parameterNames = ['θ₁₁', 'θ₁₀', 'φ₁', 'θ₂₁', 'θ₂₀', 'φ₂', 'φ₀'];
function perturb(model, index, amount) {
  const changed = copy(model);
  if (index === model.units.length * 3) changed.c += amount;
  else changed.units[Math.floor(index / 3)][['w', 'b', 'a'][index % 3]] += amount;
  return changed;
}

test('the half-MSE objective averages half-squared point errors while reporting ordinary MSE', () => {
  const first = experiment.firstUpdate;
  const rows = first.indices.map(index => data[index]);
  const pointLosses = rows.map(row => .5 * (predict(first.before, row.x) - row.y) ** 2);
  close(pointLosses.reduce((sum, value) => sum + value, 0) / rows.length, first.batchBefore);
  close(first.batchBefore, loss(first.before, rows));
  close(mse(first.before, rows), 2 * first.batchBefore);
  assert.equal(first.batchBefore.toFixed(4), '0.2129');
  assert.equal(mse(first.before, rows).toFixed(4), '0.4257');
  const g = gradient(first.before, rows);
  const eps = 1e-6;
  parameters(g).forEach((value, index) => {
    const plus = perturb(first.before, index, eps), minus = perturb(first.before, index, -eps);
    close((loss(plus, rows) - loss(minus, rows)) / (2 * eps), value, 1e-9);
    close(rows.reduce((sum, row) => sum + parameters(gradient(first.before, [row]))[index], 0) / rows.length, value);
  });
  close(experiment.traces[2][0].eta, .05);
  close(first.before.c - experiment.traces[2][0].eta * g.c, first.after.c);
  assert.equal(first.after.c.toFixed(4), '0.5084');
  for (const width of [2, 12]) {
    for (const snapshot of experiment.traces[width]) {
      close(snapshot.loss, mse(snapshot.model));
      close(snapshot.afterLoss, mse(snapshot.after));
    }
  }
  for (const run of experiment.runs) close(run.loss, mse(run.model));
});

// Preserve the former squared-error derivative independently of the current
// objective. Replaying it at the former rate detects accidental changes to the
// learning-rate decay, update order, or gradients anywhere in training.
function legacyMSEGradient(model, rows) {
  const result = { c: 0, units: model.units.map(() => ({ w: 0, b: 0, a: 0 })) };
  for (const row of rows) {
    const errorDerivative = 2 * (predict(model, row.x) - row.y) / rows.length;
    result.c += errorDerivative;
    model.units.forEach((unit, index) => {
      const preactivation = unit.w * row.x + unit.b;
      result.units[index].a += errorDerivative * Math.max(0, preactivation);
      if (preactivation > 0) {
        result.units[index].w += errorDerivative * unit.a * row.x;
        result.units[index].b += errorDerivative * unit.a;
      }
    });
  }
  return result;
}

test('half-MSE with twice the learning rate preserves every recorded update across all 300 epochs', () => {
  for (const width of [2, 12]) {
    const model = initialize(width, 1), rng = random(10001);
    const recorded = new Map(experiment.traces[width].map(snapshot => [snapshot.step, snapshot]));
    for (let epoch = 0; epoch < 300; epoch++) {
      const order = shuffle(rng);
      for (let batch = 0; batch < 32; batch++) {
        const step = epoch * 32 + batch, snapshot = recorded.get(step);
        const rows = order.slice(batch * 64, (batch + 1) * 64).map(index => data[index]);
        const oldRate = .025 / (1 + step / 6000);
        if (snapshot) {
          assert.deepEqual(snapshot.model, model, `Width ${width}, step ${step}: preceding weights`);
          assert.equal(snapshot.eta, 2 * oldRate);
        }
        update(model, legacyMSEGradient(model, rows), oldRate);
        if (snapshot) assert.deepEqual(snapshot.after, model, `Width ${width}, step ${step}: updated weights`);
      }
    }
    assert.deepEqual(model, experiment.runs.find(run => run.width === width && run.seed === 1).model);
  }
});

test('epoch counters advance only after every update in the pass has completed', () => {
  for (const completed of [1, 2, 150, 300]) {
    const boundary = completed * 32;
    const before = epochState(boundary - 0.001);
    assert.equal(before.completedEpochs, completed - 1);
    assert.equal(before.currentEpoch, completed);
    assert.equal(before.batch, 32);
    assert.equal(before.finished, false);
    const after = epochState(boundary);
    assert.equal(after.completedEpochs, completed);
    assert.equal(after.currentEpoch, Math.min(300, completed + 1));
    assert.equal(after.batch, completed === 300 ? 32 : 1);
    assert.equal(after.finished, completed === 300);
  }
  assert.equal(epochState(1.9).usedUpdates, 1);
  assert.equal(epochState(-2).usedUpdates, 0);
  assert.equal(epochState(10000).usedUpdates, 9600);
  assert.throws(() => epochState(Number.NaN), TypeError);
});

test('counter text and accessible state rewind with the actual completed-update count', () => {
  const element = {
    classList: { add() {} }, dataset: {}, attrs: {}, innerHTML: '',
    style: { setProperty() {} },
    setAttribute(key, value) { this.attrs[key] = value; }
  };
  const forwards = new Map();
  for (const updates of [0, 1, 31, 32, 33, 63, 64, 9599, 9600]) {
    renderEpochCounter(element, epochState(updates));
    forwards.set(updates, element.innerHTML);
    assert.equal(Number(element.dataset.completedUpdates), updates);
    assert.equal(Number(element.dataset.completedEpochs), Math.floor(updates / 32));
    assert.equal(Number(element.dataset.completedBatches), updates === 9600 ? 32 : updates % 32);
    assert.match(element.attrs['aria-label'], updates === 9600 ? /Training complete: 300 epochs/ : /Epoch \d+ of 300: \d+ of 32 batches used/);
  }
  for (const updates of [...forwards.keys()].reverse()) {
    renderEpochCounter(element, epochState(updates));
    assert.equal(element.innerHTML, forwards.get(updates));
    assert.equal(Number(element.dataset.completedUpdates), updates);
  }
});

test('the partition diagram puts every actual observation inside exactly its batch', () => {
  const { plot, detail } = renderScene('partition', 1);
  const groups = new Map([...plot.matchAll(/data-batch="(\d+)" data-count="64"><rect x="([^"]+)" y="([^"]+)" width="([^"]+)" height="([^"]+)"/g)]
    .map(match => [Number(match[1]), match.slice(2).map(Number)]));
  assert.equal(groups.size, 32);
  const observations = [...plot.matchAll(/data-observation="(\d+)" x="([^"]+)" y="([^"]+)" width="([^"]+)" height="([^"]+)"/g)];
  assert.equal(observations.length, 2048);
  assert.equal(new Set(observations.map(match => match[1])).size, 2048);
  const membership = new Map(experiment.traces[2].slice(0, 32).flatMap((snapshot, index) => snapshot.indices.map(id => [id, index + 1])));
  observations.forEach(match => {
    const [id, x, y, width, height] = match.slice(1).map(Number);
    const [left, top, groupWidth, groupHeight] = groups.get(membership.get(id));
    assert.ok(x >= left && x + width <= left + groupWidth, `Observation ${id} escaped its batch horizontally`);
    assert.ok(y >= top && y + height <= top + groupHeight, `Observation ${id} escaped its batch vertically`);
  });
  assert.equal(detail, '');
});

test('the same point loss becomes the first term of the sum before averaging', () => {
  const first = experiment.firstUpdate;
  const point = renderScene('point-loss', 1).frame;
  assert.equal(point.pointIndex, first.indices[0]);
  const row = data[point.pointIndex];
  close(point.pointPrediction, predict(first.before, row.x));
  close(point.pointResidual, point.pointPrediction - row.y);
  close(point.pointLoss, .5 * point.pointResidual ** 2);
  const start = renderScene('batch-loss', 0).frame;
  assert.equal(start.lossesAdded, 1);
  assert.equal(start.normalized, false);
  close(start.batchSum, point.pointLoss);
  const end = renderScene('batch-loss', 1).frame;
  assert.equal(end.lossesAdded, 64);
  assert.equal(end.normalized, true);
  close(end.batchSum / 64, end.batchHalfMSE);
  close(end.batchHalfMSE, first.batchBefore);
  close(end.batchMSE, 2 * first.batchBefore);
  const reversed = renderScene('batch-loss', 0).frame;
  assert.deepEqual(reversed, start);
});

test('the supplied batch gradient has one component per parameter at the unchanged current weights', () => {
  const first = experiment.firstUpdate;
  const batch = first.indices.map(index => data[index]);
  const expected = parameters(gradient(first.before, batch));
  const residuals = batch.map(row => predict(first.before, row.x) - row.y);
  const meanResidual = residuals.reduce((sum, residual) => sum + residual, 0) / batch.length;
  for (const progress of [0, .25, .65, 1, .25, 0]) {
    const { frame, plot } = renderScene('batch-gradient', progress);
    assert.deepEqual(frame.model, first.before);
    assert.equal(frame.completedUpdates, 0);
    assert.equal(frame.residualCount, batch.length);
    assert.equal(frame.residualCount, 64);
    assert.deepEqual(frame.residuals, residuals, 'Residual coordinates must follow observations in the batch');
    assert.equal(frame.parameterCount, parameters(first.before).length);
    assert.equal(frame.parameterCount, 7);
    assert.equal(frame.gradient.length, parameters(first.before).length);
    frame.gradient.forEach((value, index) => close(value, expected[index]));
    close(frame.gradientC, expected[6]);
    close(frame.gradientC, meanResidual);
    close(frame.meanResidual, meanResidual);
    close(frame.outputBiasGradient, meanResidual);
    assert.match(plot, /data-space="observations"/);
    assert.match(plot, /data-space="parameters"/);
    const displayedGradient = [...plot.matchAll(/data-parameter="([^"]+)" data-gradient="([^"]+)"/g)];
    assert.equal(displayedGradient.length, 7);
    displayedGradient.forEach((match, index) => {
      assert.equal(match[1], parameterNames[index]);
      close(Number(match[2]), expected[index]);
    });
    const displayedResiduals = [...plot.matchAll(/data-observation="(\d+)" data-batch-position="(\d+)" data-residual="([^"]+)"/g)];
    assert.equal(displayedResiduals.length, 64);
    displayedResiduals.forEach((match, index) => {
      assert.equal(Number(match[1]), first.indices[index]);
      assert.equal(Number(match[2]), index + 1);
      close(Number(match[3]), residuals[index]);
    });
  }
  assert.deepEqual(expected.slice(0, 3), [0, 0, 0]);
  assert.equal(expected.filter(value => value !== 0).length, 4);
});

test('widening the network changes the parameter-gradient dimension without changing the batch residual dimension', () => {
  const first = experiment.traces[2][0];
  for (const width of [2, 12]) {
    const snapshot = experiment.traces[width][0];
    assert.deepEqual(snapshot.indices, first.indices);
    const rows = snapshot.indices.map(index => data[index]);
    const residuals = rows.map(row => predict(snapshot.model, row.x) - row.y);
    const g = gradient(snapshot.model, rows);
    assert.equal(residuals.length, 64);
    assert.equal(parameters(g).length, width === 2 ? 7 : 37);
    assert.equal(parameters(g).length, parameters(snapshot.model).length);
    const meanResidual = residuals.reduce((sum, residual) => sum + residual, 0) / rows.length;
    // The output bias moves every prediction by the same amount. Its one
    // gradient coordinate therefore averages all 64 residuals in either model.
    close(g.c, meanResidual);
    const eps = 1e-6;
    const plus = { ...snapshot.model, c: snapshot.model.c + eps };
    const minus = { ...snapshot.model, c: snapshot.model.c - eps };
    close((loss(plus, rows) - loss(minus, rows)) / (2 * eps), meanResidual, 1e-9);
  }
});

test('all seven parameters take the same simultaneous update, including the three zero-gradient coordinates', () => {
  const first = experiment.firstUpdate;
  const before = parameters(first.before), after = parameters(first.after);
  const g = parameters(first.gradient);
  const moving = g.map((value, index) => value !== 0 ? index : null).filter(index => index !== null);
  assert.equal(moving.length, 4);
  const states = new Map();
  for (const progress of [0, .3, .55, .8, 1]) {
    const { frame, detail } = renderScene('update', progress);
    const current = parameters(frame.model), fraction = frame.updateProgress;
    assert.ok(fraction >= 0 && fraction <= 1);
    assert.equal(frame.displayedParameters.length, 7);
    current.forEach((value, index) => {
      close(value, before[index] + fraction * (after[index] - before[index]));
      const displayed = frame.displayedParameters[index];
      assert.equal(displayed.label, parameterNames[index]);
      close(displayed.before, before[index]);
      close(displayed.current, value);
      close(displayed.gradient, g[index]);
      assert.equal(displayed.unchanged, g[index] === 0);
      if (g[index] === 0) assert.equal(value, before[index]);
      else close((value - before[index]) / (after[index] - before[index]), fraction, 1e-10);
    });
    const rendered = [...detail.matchAll(/data-parameter="[^"]+"/g)];
    assert.equal(rendered.length, 7, 'All parameter values must accompany the moving model curve');
    if (progress > .12 && progress < .82) {
      assert.ok(fraction > 0 && fraction < 1);
      moving.forEach(index => assert.notEqual(current[index], before[index]));
    }
    states.set(progress, frame);
  }
  assert.deepEqual(states.get(0).model, first.before);
  assert.deepEqual(states.get(1).model, first.after);
  for (const progress of [...states.keys()].reverse()) {
    assert.deepEqual(renderScene('update', progress).frame, states.get(progress));
  }
});

test('batch and epoch transitions continue from the exact preceding learned weights', () => {
  const trace = experiment.traces[2];
  const nextStart = renderScene('next-batch', 0).frame;
  const nextEnd = renderScene('next-batch', 1).frame;
  assert.equal(nextStart.completedUpdates, 1);
  assert.equal(nextEnd.completedUpdates, 2);
  assert.deepEqual(nextStart.model, trace[0].after);
  assert.deepEqual(nextEnd.model, trace[1].after);
  close(nextStart.outputBias, nextStart.model.c);
  close(nextEnd.outputBias, nextEnd.model.c);
  const partialNext = renderScene('next-batch', .65).frame;
  assert.ok(partialNext.updateProgress > 0 && partialNext.updateProgress < 1);
  assert.equal(partialNext.completedUpdates, 1);
  parameters(partialNext.model).forEach((value, index) => {
    const before = parameters(trace[1].model)[index], after = parameters(trace[1].after)[index];
    close(value, before + partialNext.updateProgress * (after - before));
  });
  assert.equal(nextEnd.batchOverlap, 0);
  for (const [id, progress, step, completed] of [
    ['batches-two', 0, 2, 2], ['batches-two', 1, 31, 32],
    ['epoch-two', 0, 32, 32], ['epoch-two', 1, 32, 33],
    ['train-two', 0, 33, 33], ['train-two', 1, 9599, 9600]
  ]) {
    const frame = renderScene(id, progress).frame;
    assert.equal(frame.step, step);
    assert.equal(frame.completedUpdates, completed);
    const snapshot = trace.find(item => item.step === step);
    const expected = completed === step ? snapshot.model : snapshot.after;
    close(frame.model.c, expected.c);
    frame.model.units.forEach((unit, index) => {
      for (const key of ['w', 'b', 'a']) close(unit[key], expected.units[index][key]);
    });
  }
  assert.equal(renderScene('train-wide', 0).frame.completedUpdates, 32);
  assert.equal(renderScene('train-wide', 1).frame.completedUpdates, 9600);
});

test('the detailed update and training sequence keeps its target axes and boundary predictions fixed', () => {
  const sequence = ['update', 'next-batch', 'batches-two', 'epoch-two', 'train-two'];
  const originalWidth = globalThis.innerWidth, originalHeight = globalThis.innerHeight;
  const attributes = element => Object.fromEntries([...element.matchAll(/([\w-]+)="([^"]*)"/g)].map(match => match.slice(1)));
  const geometry = plot => {
    const clip = plot.match(/<clipPath id="clip-plot">(<rect\b[^>]*\/>)/)?.[1];
    assert.ok(clip, 'The prediction plot must declare its fixed plotting area');
    const target = [...plot.matchAll(/<path\b[^>]*\/>/g)].map(match => attributes(match[0]))
      .filter(path => path.stroke === '#606060' && path['stroke-dasharray'] === '6 5');
    assert.equal(target.length, 1, 'One stationary target step anchors the prediction plot');
    const endpoints = [...plot.matchAll(/<circle\b[^>]*\/>/g)].map(match => attributes(match[0]))
      .filter(circle => circle.fill === '#606060' || circle.stroke === '#606060');
    assert.equal(endpoints.length, 2, 'Both open and closed target endpoints remain visible');
    return { clip: attributes(clip), target: target[0].d, endpoints };
  };
  const prediction = plot => {
    const paths = [...plot.matchAll(/<path\b[^>]*\/>/g)].map(match => attributes(match[0]))
      .filter(path => path.stroke === '#234e70' && path['clip-path'] === 'url(#clip-plot)' && !path['stroke-dasharray']);
    assert.equal(paths.length, 1, 'A single blue curve shows the current model');
    return paths[0].d;
  };
  try {
    for (const [viewportWidth, viewportHeight, plotWidth, plotHeight] of [[320, 568, 288, 240], [1280, 720, 760, 340], [1920, 1080, 1130, 650]]) {
      globalThis.innerWidth = viewportWidth;
      globalThis.innerHeight = viewportHeight;
      const reference = geometry(renderScene(sequence[0], 0, plotWidth, plotHeight).plot);
      const seen = [];
      for (const id of sequence) {
        for (const progress of [0, .35, .65, 1]) {
          const preview = renderScene(id, progress, plotWidth, plotHeight);
          assert.deepEqual(geometry(preview.plot), reference, `${id} must not shift or rescale the target at ${viewportWidth}px`);
          seen.push({ id, progress, preview });
        }
      }
      for (let i = 1; i < sequence.length; i++) {
        const outgoing = renderScene(sequence[i - 1], 1, plotWidth, plotHeight);
        const incoming = renderScene(sequence[i], 0, plotWidth, plotHeight);
        assert.deepEqual(incoming.frame.model, outgoing.frame.model, 'Each scene starts from the exact preceding learned weights');
        assert.equal(prediction(incoming.plot), prediction(outgoing.plot), 'The visible model must not jump when the batch or epoch changes');
        assert.equal(incoming.frame.completedUpdates, outgoing.frame.completedUpdates);
      }
      for (const stop of seen.reverse()) {
        assert.deepEqual(renderScene(stop.id, stop.progress, plotWidth, plotHeight), stop.preview, 'Reverse scrolling restores the same coordinates and numerical state');
      }
    }
  } finally {
    if (originalWidth === undefined) delete globalThis.innerWidth;
    else globalThis.innerWidth = originalWidth;
    if (originalHeight === undefined) delete globalThis.innerHeight;
    else globalThis.innerHeight = originalHeight;
  }
});

test('the loss history zooms from one epoch to the full run while its axes and learned model stay fixed', async () => {
  const originalWidth = globalThis.innerWidth, originalHeight = globalThis.innerHeight;
  const attributes = element => Object.fromEntries([...element.matchAll(/([\w-]+)="([^"]*)"/g)].map(match => match.slice(1)));
  const history = detail => {
    const domain = detail.match(/>([\d,]+) updates<\/text>/)?.[1];
    assert.ok(domain, 'The loss plot must label the visible update domain');
    const lines = [...detail.matchAll(/<line\b[^>]*\/>/g)].map(match => attributes(match[0]));
    const axis = lines.find(line => line.stroke === '#767676');
    assert.ok(axis, 'The loss history keeps a visible horizontal axis');
    const marker = [...detail.matchAll(/<circle\b[^>]*\/>/g)].map(match => attributes(match[0]))
      .find(circle => circle.fill === '#234e70');
    assert.ok(marker, 'The current loss must remain marked throughout the zoom');
    return { domain: Number(domain.replaceAll(',', '')), axis, grid: lines.filter(line => line.stroke === '#eee'), marker };
  };
  try {
    for (const [viewportWidth, viewportHeight, width, detailHeight] of [[320, 568, 288, 110], [1280, 720, 360, 330], [1920, 1080, 510, 650]]) {
      globalThis.innerWidth = viewportWidth;
      globalThis.innerHeight = viewportHeight;
      const preview = (id, progress) => renderScene(id, progress, width, 430, detailHeight);
      const previous = preview('batches-two', 1), start = preview('epoch-two', 0), first = history(start.detail);
      assert.equal(first.domain, 32);
      assert.deepEqual(first, history(previous.detail), 'The loss plot enters the reshuffle at the exact last point and scale of epoch one');
      const baselineModel = start.frame.model, seen = [];
      let previousDomain = first.domain, previousX = Number(first.marker.cx);
      for (const progress of [0, .18, .24, .3, .36, .44]) {
        const current = preview('epoch-two', progress), graph = history(current.detail);
        assert.deepEqual(graph.axis, first.axis, 'Zooming the update domain must not move the horizontal axis');
        assert.deepEqual(graph.grid, first.grid, 'The vertical loss scale and grid remain fixed');
        assert.deepEqual(current.frame.model, baselineModel, 'Reshuffling and reframing the history must not train the model');
        assert.equal(current.frame.completedUpdates, 32);
        assert.equal(graph.marker.cy, first.marker.cy, 'The current loss stays constant while the history zooms');
        assert.ok(graph.domain >= previousDomain && graph.domain <= 9600);
        assert.ok(Number(graph.marker.cx) <= previousX, 'The same 32 completed updates move continuously toward the left as the domain expands');
        close(Number(graph.marker.cx), Number(graph.axis.x1) + 32 / graph.domain * (Number(graph.axis.x2) - Number(graph.axis.x1)));
        previousDomain = graph.domain;
        previousX = Number(graph.marker.cx);
        seen.push({ progress, current });
      }
      assert.equal(previousDomain, 9600, 'The full-run framing is established before the next update');
      assert.ok(history(seen[2].current.detail).domain > 32 && history(seen[2].current.detail).domain < 9600, 'The history has an intermediate scale rather than a sudden jump');
      const end = preview('epoch-two', 1), next = preview('train-two', 0);
      assert.equal(history(end.detail).domain, 9600);
      assert.deepEqual(history(next.detail), history(end.detail), 'The longer training sequence inherits the same loss point, axes, and full-run domain');
      for (const stop of seen.reverse()) assert.deepEqual(preview('epoch-two', stop.progress), stop.current, 'Reversing the reshuffle restores its exact intermediate zoom');
    }
    const ui = await controller();
    ui.at('epoch-two', .3);
    const intermediate = ui.nodes.get('detail').innerHTML;
    ui.at('epoch-two', .8);
    ui.at('train-two', .4);
    ui.at('epoch-two', .3);
    assert.equal(ui.nodes.get('detail').innerHTML, intermediate, 'Actual controller scroll events restore the same intermediate loss-history framing');
  } finally {
    if (originalWidth === undefined) delete globalThis.innerWidth;
    else globalThis.innerWidth = originalWidth;
    if (originalHeight === undefined) delete globalThis.innerHeight;
    else globalThis.innerHeight = originalHeight;
  }
});

test('single-parameter probes compare independent nudges of the trained model with honest full-data losses', () => {
  const original = JSON.stringify(experiment);
  const baseline = experiment.traces[2].at(-1).after;
  const trained = experiment.runs.find(run => run.width === 2 && run.seed === 1);
  const baselineValues = parameters(baseline), baselineMSE = mse(baseline);
  assert.deepEqual(baseline, trained.model);
  assert.deepEqual(renderScene('parameter-probe', 0).frame.model, renderScene('train-two', 1).frame.model);
  const seen = new Map();
  for (let index = 0; index < 7; index++) {
    for (const [localProgress, expectedDelta] of [[0, 0], [.25, -.02], [.5, 0], [.75, .02]]) {
      const progress = (index + localProgress) / 7;
      const frame = renderScene('parameter-probe', progress).frame;
      assert.deepEqual(frame.baselineModel, baseline);
      assert.equal(frame.parameterIndex, index);
      assert.equal(frame.parameterName, parameterNames[index]);
      assert.equal(frame.completedUpdates, 9600, 'Trying a parameter is not an additional training update');
      close(frame.delta, expectedDelta);
      parameters(frame.model).forEach((value, coordinate) => {
        if (coordinate === index) close(value, baselineValues[coordinate] + expectedDelta);
        else assert.equal(value, baselineValues[coordinate], 'Each trial must hold every other parameter fixed');
      });
      close(frame.baselineMSE, baselineMSE);
      close(frame.currentMSE, mse(frame.model));
      close(frame.currentMSE, mse(perturb(baseline, index, expectedDelta)));
      assert.equal(frame.trialResults.length, 7);
      frame.trialResults.forEach((result, coordinate) => {
        assert.equal(result.parameterIndex, coordinate);
        assert.equal(result.parameterName, parameterNames[coordinate]);
        assert.equal(result.baselineValue, baselineValues[coordinate]);
        close(result.minus, mse(perturb(baseline, coordinate, -.02)));
        close(result.plus, mse(perturb(baseline, coordinate, .02)));
        // This observed result supports the scene's local claim; it does not
        // assert that coordinate changes can never improve other networks.
        assert.ok(result.minus >= baselineMSE - 1e-12);
        assert.ok(result.plus >= baselineMSE - 1e-12);
      });
      seen.set(progress, frame);
    }
  }
  const end = renderScene('parameter-probe', 1);
  assert.deepEqual(end.frame.model, baseline, 'Finish by restoring the actual trained model');
  assert.equal(end.frame.delta, 0);
  assert.equal(end.frame.completedUpdates, 9600);
  assert.equal([...end.detail.matchAll(/data-parameter="[^"]+"/g)].length, 7, 'The static end state must show every independent trial');
  assert.equal([...end.detail.matchAll(/>MSE after<\/text>/g)].length, 2, 'Both endpoint columns must identify their values as resulting MSE');
  for (const label of [renderText(String.raw`\(-0.02\) nudge`), renderText(String.raw`\(+0.02\) nudge`), 'Baseline MSE:', 'Current trial MSE:', 'Change in MSE:']) {
    assert.ok(end.detail.includes(label), `The trial summary must distinguish ${label}`);
  }
  const mathRows = [...end.detail.matchAll(/<g data-parameter="([^"]+)"[^>]*>([\s\S]*?)<\/g>/g)];
  const expectedTex = [String.raw`\theta_{11}`, String.raw`\theta_{10}`, String.raw`\phi_1`, String.raw`\theta_{21}`, String.raw`\theta_{20}`, String.raw`\phi_2`, String.raw`\phi_0`];
  mathRows.forEach((row, index) => {
    assert.equal(row[1], parameterNames[index]);
    assert.match(row[2], /<foreignObject\b/);
    assert.match(row[2], /class="katex"/);
    assert.ok(row[2].includes(`<annotation encoding="application/x-tex">${expectedTex[index]}</annotation>`), 'Each parameter must retain its typeset symbol and subscript');
  });
  assert.match(end.plot, /class="katex"/, 'The current curve trial must use the same mathematical notation');
  for (const [progress, frame] of [...seen].reverse()) {
    assert.deepEqual(renderScene('parameter-probe', progress).frame, frame);
  }
  assert.equal(JSON.stringify(experiment), original, 'The probe must not change the stored experiment or later training scenes');
});

// This harness runs the real controller and renderers. Browser geometry and
// actual scrolling are reviewed separately; here scroll events must alter and
// restore the scientific frames and counter, without relying on a timer.
async function controller(chapter = 'full') {
  const lesson = chapter === 'batches' ? foundationScenes : scenes;
  const html = readFileSync(new URL(`../docs/sgd/${chapter === 'batches' ? 'batches' : 'index'}.html`, import.meta.url), 'utf8');
  const source = readFileSync(new URL('../docs/sgd/story.mjs', import.meta.url), 'utf8').replace(/^import[^\n]+\n/gm, '');
  const listeners = new Map(), frames = [], nodes = new Map();
  let context;
  function on(type, callback) { listeners.set(type, [...(listeners.get(type) || []), callback]); }
  class Element {
    constructor(id = '') {
      this.id = id; this.dataset = {}; this.attrs = {}; this.children = [];
      this.textContent = ''; this.innerHTML = ''; this.hidden = false;
      this.classes = new Set();
      this.classList = {
        add: key => this.classes.add(key), remove: key => this.classes.delete(key),
        toggle: (key, value) => value ? this.classes.add(key) : this.classes.delete(key)
      };
      this.style = { setProperty(key, value) { this[key] = value; } };
    }
    setAttribute(key, value) { this.attrs[key] = String(value); }
    removeAttribute(key) { delete this.attrs[key]; }
    addEventListener(type, callback) { on(`${this.id}:${type}`, callback); }
    scrollIntoView() {}
    getBoundingClientRect() { return { top: 0, width: 288, height: this.id === 'detail' ? 90 : 240 }; }
  }
  for (const match of html.matchAll(/\bid="([^"]+)"/g)) nodes.set(match[1], new Element(match[1]));
  for (const id of ['stage', 'body', 'skip']) nodes.set(id, new Element(id));
  nodes.get('body').dataset.chapter = chapter;
  nodes.get('procedure').children = batchSteps.map((_, index) => new Element(`procedure-${index}`));
  let offset = 0;
  const steps = lesson.map(scene => {
    const step = new Element(scene.id);
    step.offset = offset; step.height = (scene.length || 1) * 710; offset += step.height;
    step.getBoundingClientRect = () => ({ top: step.offset - context.scrollY, height: step.height });
    return step;
  });
  const media = { matches: false, addEventListener: (type, callback) => on(`media:${type}`, callback) };
  context = {
    allScenes: scenes, foundationScenes, constructionKinds, constructionState, render, renderText, plainText, epochState, renderEpochCounter,
    console, Math, Map, JSON, Number, String, innerWidth: 320, innerHeight: 568,
    scrollY: 0, location: { hash: '' }, matchMedia: () => media,
    requestAnimationFrame: callback => frames.push(callback), addEventListener: on,
    fetch: async () => {
      assert.notEqual(chapter, 'batches', 'Focused explanation must not require experiment data');
      return { ok: true, json: async () => experiment };
    }
  };
  context.document = {
    body: nodes.get('body'), fonts: { ready: Promise.resolve() },
    getElementById: id => nodes.get(id),
    querySelector: selector => nodes.get(selector === '.stage' ? 'stage' : 'skip'),
    querySelectorAll: () => steps,
    addEventListener: (type, callback) => on(`document:${type}`, callback)
  };
  context.window = context; context.katex = { renderToString: tex => tex };
  context.scrollTo = options => {
    context.scrollY = options.top; context.lastScroll = options;
    for (const callback of listeners.get('scroll') || []) callback();
  };
  vm.createContext(context);
  await vm.runInContext(`(async()=>{${source}\n})()`, context);
  const flush = () => { let count = 0; while (frames.length) { assert.ok(count++ < 20); frames.shift()(); } };
  const emit = (type, event) => { for (const callback of listeners.get(type) || []) callback(event); flush(); };
  flush();
  const at = (id, progress) => {
    const step = steps.find(item => item.id === id);
    context.scrollY = step.offset + progress * step.height; emit('scroll');
    assert.equal(nodes.get('stage').dataset.scene, id);
    return JSON.parse(nodes.get('stage').dataset.frame);
  };
  return {
    at, nodes, context, emit,
    reduce(value) { media.matches = value; emit('media:change'); }
  };
}

test('scroll events advance and restore the batch update and epoch counter together', async () => {
  const ui = await controller();
  const stops = ['update', 'next-batch', 'batches-two', 'epoch-two', 'train-two']
    .flatMap(id => [0, .35, .6, .9].map(progress => [id, progress]));
  const seen = [], counter = ui.nodes.get('epoch-counter');
  let previous = 0;
  for (const [id, progress] of stops) {
    const frame = ui.at(id, progress);
    assert.equal(counter.hidden, false);
    const used = Number(counter.dataset.completedUpdates);
    assert.equal(used, Math.floor(frame.completedUpdates));
    assert.ok(used >= previous, `${id} regressed to ${used} after ${previous}`);
    previous = used;
    seen.push({ id, progress, frame, counter: counter.innerHTML });
  }
  for (const stop of seen.reverse()) {
    assert.deepEqual(ui.at(stop.id, stop.progress), stop.frame);
    assert.equal(counter.innerHTML, stop.counter);
  }
});

test('parameter probes retain the completed epoch count and a complete reduced-motion summary', async () => {
  const ui = await controller(), counter = ui.nodes.get('epoch-counter');
  const stops = [0, .25 / 7, .75 / 7, 3.25 / 7, 3.75 / 7, 6.25 / 7, 6.75 / 7, .999];
  const seen = new Map();
  for (const progress of stops) {
    const frame = ui.at('parameter-probe', progress);
    assert.equal(counter.hidden, false);
    assert.equal(frame.completedUpdates, 9600);
    assert.equal(counter.dataset.completedUpdates, '9600');
    assert.equal(counter.dataset.completedEpochs, '300');
    seen.set(progress, { frame, counter: counter.innerHTML });
  }
  for (const [progress, state] of [...seen].reverse()) {
    assert.deepEqual(ui.at('parameter-probe', progress), state.frame);
    assert.equal(counter.innerHTML, state.counter);
  }
  ui.reduce(true);
  const reduced = ui.at('parameter-probe', .1);
  assert.deepEqual(reduced.model, experiment.traces[2].at(-1).after);
  assert.equal(reduced.delta, 0);
  assert.equal(reduced.trialResults.length, 7);
  assert.equal([...ui.nodes.get('detail').innerHTML.matchAll(/data-parameter="[^"]+"/g)].length, 7);
  assert.equal(counter.dataset.completedEpochs, '300');
  ui.emit('document:keydown', { key: 'ArrowRight', target: { tagName: 'DIV' }, preventDefault() {} });
  assert.equal(ui.nodes.get('stage').dataset.scene, 'initialize-wide');
  assert.equal(counter.dataset.completedUpdates, '0');
  assert.equal(counter.dataset.completedEpochs, '0');
  ui.emit('document:keydown', { key: 'ArrowLeft', target: { tagName: 'DIV' }, preventDefault() {} });
  assert.equal(ui.nodes.get('stage').dataset.scene, 'parameter-probe');
  assert.equal(counter.dataset.completedEpochs, '300');
});

test('reduced motion and keyboard navigation retain complete losses and epoch boundaries', async () => {
  const ui = await controller();
  ui.reduce(true);
  assert.equal(ui.at('batch-loss', .1).normalized, true);
  const gradientFrame = ui.at('batch-gradient', .1);
  assert.equal(gradientFrame.gradient.length, 7);
  assert.equal(gradientFrame.completedUpdates, 0);
  assert.equal(ui.at('batches-two', .1).completedUpdates, 32);
  assert.equal(ui.nodes.get('epoch-counter').dataset.completedEpochs, '1');
  ui.emit('document:keydown', { key: 'ArrowRight', target: { tagName: 'DIV' }, preventDefault() {} });
  assert.equal(ui.context.lastScroll.behavior, 'instant');
  assert.equal(ui.nodes.get('stage').dataset.scene, 'epoch-two');
  assert.equal(ui.nodes.get('epoch-counter').dataset.completedUpdates, '33');
  ui.emit('document:keydown', { key: 'ArrowLeft', target: { tagName: 'DIV' }, preventDefault() {} });
  assert.equal(ui.nodes.get('stage').dataset.scene, 'batches-two');
  assert.equal(ui.nodes.get('epoch-counter').dataset.completedUpdates, '32');
  const focused = await controller('batches');
  focused.at('epoch-repeat', .5);
  assert.equal(focused.nodes.get('epoch-counter').hidden, true);
});

test('the batch gradient leads directly into the same simultaneous update and reverses with keyboard and reduced motion', async () => {
  const first = experiment.firstUpdate, original = JSON.stringify(experiment);
  const index = scenes.findIndex(scene => scene.id === 'batch-gradient');
  assert.equal(index, 14);
  assert.equal(scenes[index + 1].id, 'update');
  assert.equal(scenes.length, 25);
  assert.equal(scenes.some(scene => ['batch-gradient-comparison', 'parameter-step', 'finish'].includes(scene.id)), false);
  assert.equal(scenes.at(-1).id, 'compare');
  const supplied = renderScene('batch-gradient', 1).frame;
  const start = renderScene('update', 0).frame;
  assert.deepEqual(start.model, supplied.model, 'The update starts from the exact weights used to compute the displayed gradient');
  assert.deepEqual(start.model, first.before);
  assert.deepEqual(start.displayedParameters.map(parameter => parameter.gradient), supplied.gradient, 'The update uses the seven gradient components just shown');
  assert.equal(start.completedUpdates, supplied.completedUpdates);
  assert.equal(start.completedUpdates, 0);

  const ui = await controller(), counter = ui.nodes.get('epoch-counter'), seen = [];
  const opacity = () => Number(ui.nodes.get('stage').style['--gradient-handoff-opacity']);
  for (const [id, progress] of [['batch-gradient', .25], ['batch-gradient', .995], ['update', 0], ['update', .05], ['update', .1], ['update', .25], ['update', .5], ['update', .995]]) {
    const frame = ui.at(id, progress);
    assert.equal(counter.dataset.completedUpdates, String(Math.floor(frame.completedUpdates)));
    if (id === 'update' && progress <= .1) {
      assert.deepEqual(frame.model, first.before, 'The incoming diagram is revealed before any parameter moves');
      assert.equal(frame.completedUpdates, 0);
    }
    if (id === 'update' && progress >= .1) assert.equal(opacity(), 1, 'The model is fully visible before its update begins');
    seen.push({ id, progress, frame, counter: counter.innerHTML, opacity: opacity() });
  }
  assert.equal(seen.find(stop => stop.id === 'update' && stop.progress === 0).opacity, 0);
  const arriving = seen.find(stop => stop.id === 'update' && stop.progress === .05);
  assert.ok(arriving.opacity > 0 && arriving.opacity < 1);
  assert.deepEqual(seen.at(-1).frame.model, first.after);
  assert.equal(seen.at(-1).frame.completedUpdates, 1);
  for (const stop of seen.reverse()) {
    assert.deepEqual(ui.at(stop.id, stop.progress), stop.frame, 'Reverse scrolling restores the same parameters and gradient');
    assert.equal(counter.innerHTML, stop.counter);
    assert.equal(opacity(), stop.opacity, 'The diagram handoff reverses with the same scroll position');
  }

  ui.at('batch-gradient', .5);
  ui.emit('document:keydown', { key: 'ArrowRight', target: { tagName: 'DIV' }, preventDefault() {} });
  assert.equal(ui.nodes.get('stage').dataset.scene, 'update');
  assert.deepEqual(JSON.parse(ui.nodes.get('stage').dataset.frame).model, renderScene('update', .15).frame.model);
  assert.equal(counter.dataset.completedUpdates, '0');
  ui.emit('document:keydown', { key: 'ArrowLeft', target: { tagName: 'DIV' }, preventDefault() {} });
  assert.equal(ui.nodes.get('stage').dataset.scene, 'batch-gradient');
  assert.deepEqual(JSON.parse(ui.nodes.get('stage').dataset.frame).model, first.before);
  assert.equal(counter.dataset.completedUpdates, '0');

  ui.reduce(true);
  const reducedGradient = ui.at('batch-gradient', .1);
  assert.deepEqual(reducedGradient.model, first.before);
  assert.deepEqual(reducedGradient.gradient, supplied.gradient);
  assert.equal(reducedGradient.completedUpdates, 0);
  assert.equal(opacity(), 1, 'Reduced motion must keep the completed gradient diagram visible');
  ui.emit('document:keydown', { key: 'ArrowRight', target: { tagName: 'DIV' }, preventDefault() {} });
  assert.equal(ui.context.lastScroll.behavior, 'instant');
  assert.equal(ui.nodes.get('stage').dataset.scene, 'update');
  const reducedUpdate = JSON.parse(ui.nodes.get('stage').dataset.frame);
  assert.deepEqual(reducedUpdate.model, first.after);
  assert.deepEqual(reducedUpdate.displayedParameters.map(parameter => parameter.gradient), supplied.gradient);
  assert.equal(reducedUpdate.completedUpdates, 1);
  assert.equal(opacity(), 1, 'Reduced motion must keep the completed update visible');
  assert.equal(counter.dataset.completedUpdates, '1');
  assert.deepEqual(ui.at('update', .6), reducedUpdate, 'Reduced motion presents the completed simultaneous update');
  ui.emit('document:keydown', { key: 'ArrowLeft', target: { tagName: 'DIV' }, preventDefault() {} });
  assert.equal(ui.nodes.get('stage').dataset.scene, 'batch-gradient');
  assert.deepEqual(JSON.parse(ui.nodes.get('stage').dataset.frame), reducedGradient);
  assert.equal(counter.dataset.completedUpdates, '0');
  assert.equal(JSON.stringify(experiment), original, 'Moving between the gradient and update never mutates the stored experiment');
});

test('both static reading routes contain the same substantive captions and definitions', () => {
  for (const [file, lesson] of [['index.html', scenes], ['batches.html', foundationScenes]]) {
    const html = readFileSync(new URL(`../docs/sgd/${file}`, import.meta.url), 'utf8');
    const steps = html.slice(html.indexOf('<div id="steps">'), html.indexOf('<section class="methods"'));
    const reading = html.slice(html.indexOf('<details id="reading">'), html.indexOf('</details>'));
    assert.equal((steps.match(/class="step"/g) || []).length, lesson.length);
    assert.equal((reading.match(/<section>/g) || []).length, lesson.length);
    for (const scene of lesson) {
      const narration = scene.reading ?? scene.caption;
      const note = scene.readingNote ?? scene.note;
      assert.ok(steps.includes(renderText(narration)) && steps.includes(renderText(note)), `${file}: ${scene.id} fallback`);
      assert.ok(reading.includes(renderText(narration)) && reading.includes(renderText(note)), `${file}: ${scene.id} reading`);
    }
  }
});

test('inline prose math keeps LaTeX semantics and escapes ordinary text', () => {
  const source = String.raw`Bias \(\phi_0\) < 1 & residual \(r_i\).`;
  const html = renderText(source);
  assert.match(html, /<annotation encoding="application\/x-tex">\\phi_0<\/annotation>/);
  assert.match(html, /<annotation encoding="application\/x-tex">r_i<\/annotation>/);
  assert.ok(html.includes('&lt; 1 &amp; residual'));
  assert.doesNotMatch(plainText(source), /\\[()]/);
  assert.match(plainText(source), /Bias/);
});

test('every lesson diagram renders symbolic notation with KaTeX rather than SVG text', () => {
  const previousWidth = globalThis.innerWidth, previousHeight = globalThis.innerHeight;
  const bareMath = /[αθφϕℓΣ∑₀-₉ᵢⱼ⁰¹²³⁴⁵⁶⁷⁸⁹≤≥≠±×½⋯]|\b(?:x|y|h_[12]|B\d+)\b|=/;
  try {
    for (const viewport of [{width:320,height:568},{width:1280,height:720},{width:1920,height:1080}]) {
      globalThis.innerWidth=viewport.width; globalThis.innerHeight=viewport.height;
      for (const scene of [...scenes,...foundationScenes]) for (const progress of [0,.5,1]) {
        const svg=id=>({id,innerHTML:'',getBoundingClientRect:()=>({width:viewport.width<=700?288:760,height:id==='plot'?300:140}),setAttribute(){}});
        const plot=svg('plot'),detail=svg('detail');
        render(plot,detail,scene,progress,experiment);
        for (const match of (plot.innerHTML+detail.innerHTML).matchAll(/<text\b[^>]*>([\s\S]*?)<\/text>/g)) {
          const visible=match[1].replace(/<[^>]*>/g,'');
          assert.doesNotMatch(visible,bareMath,`${scene.id}: math must use KaTeX, found ${visible}`);
        }
      }
    }
  } finally { globalThis.innerWidth=previousWidth; globalThis.innerHeight=previousHeight; }
});
