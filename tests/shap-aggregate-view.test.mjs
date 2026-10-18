import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { buildPopulation } from '../docs/shap/aggregate.mjs';
import { renderAggregate, packBeeswarm } from '../docs/shap/aggregate-view.js';

const data = JSON.parse(await readFile(new URL('../docs/shap/data.json', import.meta.url), 'utf8'));
const model = JSON.parse(await readFile(new URL('../docs/shap/model.json', import.meta.url), 'utf8'));
const population = buildPopulation(data, model);
const observation = population[0];
const scenes = ['contribution-axis', 'one-person-dots', 'many-people', 'beeswarm',
  'feature-color', 'signed-average', 'absolute-values', 'global-importance', 'importance-ranking'];
const widths = [760, 360];
const expectedImportance = [3.24, 7.2, 1.2];
const closeTo = (actual, expected) => assert.ok(Number.isFinite(actual)
  && Math.abs(actual - expected) < 1e-8, `${actual} should equal ${expected}`);
const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;

// Exercise the real SVG renderer without pretending this small DOM measures
// typography or browser layout. Browser checks cover those separately.
class MockElement {
  constructor(tagName = 'div', width = 760) {
    this.tagName = tagName;
    this.attributes = new Map();
    this.children = [];
    this.parentNode = null;
    this.width = width;
    this._text = '';
  }
  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  getAttribute(name) { return this.attributes.get(name) ?? null; }
  append(...children) {
    for (const child of children) { child.parentNode = this; this.children.push(child); }
  }
  replaceChildren(...children) {
    for (const child of this.children) child.parentNode = null;
    this.children = []; this._text = ''; this.append(...children);
  }
  contains(node) { return this === node || this.children.some(child => child.contains(node)); }
  addEventListener() {}
  getBoundingClientRect() { return { width: this.width }; }
  set textContent(value) { this.replaceChildren(); this._text = String(value); }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
}
globalThis.document = { createElementNS: (_, tagName) => new MockElement(tagName) };

function all(node, predicate) {
  return [node, ...node.children.flatMap(child => all(child, predicate))].filter(predicate);
}
const byClass = (node, name) => all(node, element =>
  element.getAttribute('class')?.split(' ').includes(name));
const number = (node, attribute) => Number(node.getAttribute(attribute));
const pathNumbers = node => node.getAttribute('d').match(/-?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?/gi).map(Number);
const coordinates = dots => dots.map(dot => ({ x: number(dot, 'cx'), y: number(dot, 'cy') }));
const featureDots = (svg, feature) => byClass(svg, 'aggregate-dot')
  .filter(dot => number(dot, 'data-feature-index') === feature);
const row = (svg, feature) => svg.children.find(node => node.tagName === 'g'
  && number(node, 'data-feature-index') === feature
  && node.getAttribute('data-feature-index') !== null);
const rowPart = (svg, feature, className) => byClass(row(svg, feature), className)[0];
const zeroX = svg => number(all(svg, node => node.tagName === 'line'
  && node.getAttribute('stroke') === '#767676'
  && node.getAttribute('stroke-dasharray') === '3 4')[0], 'x1');
function axisScale(svg) {
  const ticks = byClass(svg, 'aggregate-axis')[0].children.filter(node => node.tagName === 'g');
  const value = tick => Number(tick.children.find(node => node.tagName === 'text').textContent.replace('−', '-'));
  const pixel = tick => number(tick.children.find(node => node.tagName === 'line'), 'x1');
  return (pixel(ticks.at(-1)) - pixel(ticks[0])) / (value(ticks.at(-1)) - value(ticks[0]));
}

function snapshot(node, ignoreProgress = false) {
  return {
    tag: node.tagName,
    attributes: Object.fromEntries([...node.attributes].filter(([name]) =>
      !ignoreProgress || !['data-progress', 'data-reduced-motion'].includes(name))),
    text: node._text,
    children: node.children.map(child => snapshot(child, ignoreProgress)),
  };
}
function setup(width) {
  const container = new MockElement('div', width);
  return (scene, progress, reducedMotion = false) => renderAggregate(container, {
    scene, progress, reducedMotion, features: data.features, observation, population,
  });
}

test('every aggregate scene has reproducible intermediate SVG states when scrolling forward and backward', () => {
  for (const width of widths) {
    const render = setup(width), states = new Map();
    let originalSvg;
    for (const scene of scenes) {
      const sceneStates = [];
      for (const progress of [0, .5, 1]) {
        const svg = render(scene, progress);
        originalSvg ??= svg;
        assert.equal(svg, originalSvg, 'updates retain the same SVG instead of restarting a transition');
        sceneStates.push(snapshot(svg, true));
        assert.doesNotMatch(JSON.stringify(sceneStates.at(-1)), /NaN|Infinity|undefined/);
      }
      assert.notDeepEqual(sceneStates[0], sceneStates[1]);
      assert.notDeepEqual(sceneStates[1], sceneStates[2]);
      states.set(scene, sceneStates);
    }
    for (const scene of [...scenes].reverse()) {
      for (const progress of [1, .5, 0]) {
        assert.deepEqual(snapshot(render(scene, progress), true), states.get(scene)[progress * 2]);
      }
    }
  }
});

test('waterfall segments move from cumulative endpoints to zero before collapsing into contribution dots', () => {
  // The default waterfall adds ability, then experience, then neighborhood
  // from its baseline: 47 → 46 → 48 → 30.
  const cumulative = [[47, 46], [48, 30], [46, 48]];
  for (const width of widths) {
    const render = setup(width);
    for (const progress of [0, .5, 1]) {
      const svg = render('contribution-axis', progress), zero = zeroX(svg), unit = axisScale(svg);
      cumulative.forEach(([start, end], feature) => {
        const coordinates = pathNumbers(rowPart(svg, feature, 'aggregate-local-bar'));
        closeTo(coordinates[0], zero + unit * start * (1 - progress));
        closeTo(coordinates[4], zero + unit * (end * (1 - progress) + observation.shapValues[feature] * progress));
      });
    }
    for (const progress of [0, .5, 1]) {
      const svg = render('one-person-dots', progress), zero = zeroX(svg), unit = axisScale(svg);
      const dots = byClass(svg, 'aggregate-local-dot');
      data.features.forEach((_, feature) => {
        const coordinates = pathNumbers(rowPart(svg, feature, 'aggregate-local-bar'));
        closeTo(coordinates[0], zero + unit * observation.shapValues[feature] * progress);
        closeTo(coordinates[4], zero + unit * observation.shapValues[feature]);
        closeTo(number(dots[feature], 'cx'), coordinates[4]);
        closeTo(number(dots[feature], 'opacity'), progress);
      });
    }
  }
});

test('beeswarm packing changes only vertical position and preserves signed contribution coordinates', () => {
  for (const width of widths) {
    const render = setup(width);
    const start = coordinates(byClass(render('many-people', 1), 'aggregate-dot'));
    const mid = coordinates(byClass(render('beeswarm', .5), 'aggregate-dot'));
    const end = coordinates(byClass(render('beeswarm', 1), 'aggregate-dot'));
    assert.ok(end.some((point, index) => Math.abs(point.y - start[index].y) > 1));
    for (let index = 0; index < start.length; index += 1) {
      closeTo(mid[index].x, start[index].x);
      closeTo(end[index].x, start[index].x);
      closeTo(mid[index].y, (start[index].y + end[index].y) / 2);
    }
    const svg = render('beeswarm', 1), unit = (width < 540 ? width - 16 - 98 : width - 24 - 184) / 40;
    for (const dot of byClass(svg, 'aggregate-dot')) {
      closeTo(number(dot, 'cx'), zeroX(svg) + number(dot, 'data-shap-value') * unit);
    }
  }
});

test('feature color interpolates to blue for low inputs and rust for high inputs regardless of contribution sign', () => {
  for (const width of widths) {
    const render = setup(width);
    const before = byClass(render('feature-color', 0), 'aggregate-dot').map(dot => dot.getAttribute('fill'));
    const halfway = byClass(render('feature-color', .5), 'aggregate-dot').map(dot => dot.getAttribute('fill'));
    const after = byClass(render('feature-color', 1), 'aggregate-dot');
    const rgb = value => value.match(/^rgb\((\d+),(\d+),(\d+)\)$/)?.slice(1).map(Number);
    for (let index = 0; index < after.length; index += 1) {
      const dot = after[index], feature = number(dot, 'data-feature-index');
      const input = number(dot, 'data-input-value'), [low, high] = data.features[feature].domain;
      const channels = rgb(dot.getAttribute('fill'));
      assert.equal(channels?.length, 3);
      assert.ok(channels.every(channel => channel >= 0 && channel <= 255));
      if (input === low) assert.deepEqual(channels, [35, 78, 112]);
      if (input === high) assert.deepEqual(channels, [165, 79, 50]);
      assert.deepEqual(rgb(halfway[index]), rgb(before[index]).map((channel, channelIndex) =>
        Math.round((channel + channels[channelIndex]) / 2)));
    }
    const negativeLow = after.find(dot => number(dot, 'data-feature-index') === 0
      && number(dot, 'data-input-value') === 0 && number(dot, 'data-shap-value') < 0);
    assert.equal(negativeLow.getAttribute('fill'), 'rgb(35,78,112)');
  }
});

test('absolute folding reflects negative contributions through zero without changing vertical identities', () => {
  for (const width of widths) {
    const render = setup(width);
    const svg = render('absolute-values', 0), zero = zeroX(svg);
    const start = coordinates(byClass(svg, 'aggregate-dot'));
    const halfway = coordinates(byClass(render('absolute-values', .5), 'aggregate-dot'));
    const end = coordinates(byClass(render('absolute-values', 1), 'aggregate-dot'));
    start.forEach((point, index) => {
      closeTo(end[index].x, zero + Math.abs(point.x - zero));
      closeTo(halfway[index].x, (point.x + end[index].x) / 2);
      closeTo(halfway[index].y, point.y);
      closeTo(end[index].y, point.y);
    });
    assert.ok(end.every(point => point.x >= zero - 1e-9));
  }
});

test('the mean diamond and displayed mean follow the actual folding dots at every intermediate stage', () => {
  for (const width of widths) {
    const render = setup(width);
    for (const progress of [0, .5, 1]) {
      const svg = render('absolute-values', progress), zero = zeroX(svg);
      const markers = byClass(svg, 'aggregate-mean-marker');
      const labels = byClass(svg, 'aggregate-signed-mean');
      for (let feature = 0; feature < data.features.length; feature += 1) {
        const averageX = mean(featureDots(svg, feature).map(dot => number(dot, 'cx')));
        closeTo(pathNumbers(markers[feature])[0], averageX);
        const unit = (width < 540 ? width - 16 - 98 : width - 24 - 184) / 40;
        closeTo(Number(labels[feature].textContent.replace('Mean = ', '')), (averageX - zero) / unit);
        closeTo((averageX - zero) / unit, expectedImportance[feature] * progress);
      }
    }
  }
});

test('importance bars preserve computed mean magnitudes while the final scene ranks rows and rescales the axis', () => {
  for (const width of widths) {
    const render = setup(width);
    for (const progress of [0, .5, 1]) {
      const svg = render('global-importance', progress), zero = zeroX(svg);
      for (let feature = 0; feature < data.features.length; feature += 1) {
        const bar = rowPart(svg, feature, 'aggregate-mean-bar');
        const label = rowPart(svg, feature, 'aggregate-mean-value');
        const unit = (width < 540 ? width - 16 - 98 : width - 24 - 184) / 40;
        closeTo(number(bar, 'x'), zero);
        closeTo(number(bar, 'width') / unit, expectedImportance[feature] * progress);
        assert.equal(Number(label.textContent), expectedImportance[feature]);
      }
    }
    const rankedPositions = [], rankedScales = [];
    for (const progress of [0, .5, 1]) {
      const svg = render('importance-ranking', progress);
      const unit = axisScale(svg);
      rankedScales.push(unit);
      const positions = data.features.map((_, feature) => number(rowPart(svg, feature, 'aggregate-mean-bar'), 'y'));
      rankedPositions.push(positions);
      data.features.forEach((_, feature) => closeTo(number(rowPart(svg, feature, 'aggregate-mean-bar'), 'width') / unit,
        expectedImportance[feature]));
    }
    assert.deepEqual([0, 1, 2].sort((a, b) => rankedPositions[0][a] - rankedPositions[0][b]), [1, 2, 0]);
    assert.deepEqual([0, 1, 2].sort((a, b) => rankedPositions[2][a] - rankedPositions[2][b]), [1, 0, 2]);
    data.features.forEach((_, feature) => closeTo(rankedPositions[1][feature],
      (rankedPositions[0][feature] + rankedPositions[2][feature]) / 2));
    assert.ok(rankedScales[0] < rankedScales[1] && rankedScales[1] < rankedScales[2]);
    closeTo(rankedScales[2] / rankedScales[0], 5);
  }
});

test('reduced motion immediately presents the completed geometry for every scene and requested progress', () => {
  for (const width of widths) {
    const render = setup(width);
    for (const scene of scenes) {
      const completed = snapshot(render(scene, 1), true);
      for (const progress of [0, .5, 1]) {
        assert.deepEqual(snapshot(render(scene, progress, true), true), completed);
      }
    }
  }
});

test('deterministic vertical packing separates exact duplicate dots without changing their x inputs', () => {
  const points = [...Array(30).fill(100), 101, 101, 103, 150, 150];
  const original = [...points], radius = 1.35;
  const offsets = packBeeswarm(points, radius);
  assert.deepEqual(points, original);
  assert.deepEqual(packBeeswarm(points, radius), offsets);
  assert.equal(offsets.length, points.length);
  assert.equal(new Set(offsets.slice(0, 30)).size, 30);
  assert.ok(offsets.some(value => value < 0) && offsets.some(value => value > 0));
  for (let left = 0; left < points.length; left += 1) {
    for (let right = left + 1; right < points.length; right += 1) {
      const distance = Math.hypot(points[left] - points[right], offsets[left] - offsets[right]);
      assert.ok(distance >= 2 * radius + .1 - 1e-6, `points ${left} and ${right} overlap`);
    }
  }
});
