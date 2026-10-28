'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const M = require('../docs/neural-networks/math.js');
const S = require('../docs/neural-networks/state.js');
const D = require('../docs/neural-networks/classification-data.js');

function close(actual, expected, tolerance = 1e-10, message = '') {
  assert.ok(Number.isFinite(actual), `Nonfinite value: ${actual}. ${message}`);
  assert.ok(Number.isFinite(expected), `Nonfinite expected value: ${expected}. ${message}`);
  assert.ok(Math.abs(actual - expected) <= tolerance,
    `${actual} differs from ${expected} by more than ${tolerance}. ${message}`);
}

function grid(lo, hi, intervals = 200) {
  return Array.from({ length: intervals + 1 }, (_, i) => lo + (hi - lo) * i / intervals);
}

function allNumbersFinite(value, path = 'frame') {
  if (typeof value === 'number') {
    assert.ok(Number.isFinite(value), `${path} must be finite, received ${value}`);
  } else if (Array.isArray(value)) {
    value.forEach((entry, i) => allNumbersFinite(entry, `${path}[${i}]`));
  } else if (value && typeof value === 'object') {
    Object.entries(value).forEach(([key, entry]) => allNumbersFinite(entry, `${path}.${key}`));
  }
}

function sampledError(model, intervals = 10000) {
  let error = 0;
  for (const x of grid(0, M.TAU, intervals)) {
    error = Math.max(error, Math.abs(M.evaluate(model, x) - Math.sin(x)));
  }
  return error;
}

test('the scroll story preserves the original 16 scenes and adds classification and generalization scenes', () => {
  assert.equal(S.scenes.length, 30);
  assert.equal(new Set(S.scenes).size, 30);
  assert.equal(S.scenes[0], 'affine');
  assert.deepEqual(S.scenes.slice(0, 16), ['affine', 'relu', 'hinge', 'hill-rise', 'hill-fall', 'hill-flat', 'shallow', 'width-one', 'width-two', 'width-three', 'width-many', 'sine-coarse', 'sine-zoom', 'sine-joint', 'sine-wide', 'universal']);
  assert.deepEqual(S.scenes.slice(16, 24), ['conflict', 'sigmoid-one', 'sigmoid-two', 'class-input', 'class-trace', 'class-move', 'class-head', 'class-compare']);
  assert.throws(() => S.frame('missing-scene', 0.5), RangeError);

  for (const scene of S.scenes) {
    for (const progress of [-1, 0, 0.125, 0.5, 0.875, 1, 2, NaN]) {
      const frame = S.frame(scene, progress);
      assert.equal(frame.scene, scene);
      assert.ok(['affine', 'relu', 'shallow', 'sine', 'sigmoid', 'classification', 'width-experiment'].includes(frame.kind));
      assert.equal(typeof frame.x, 'number');
      assert.ok(!Object.hasOwn(frame, 'depth'));
      assert.ok(frame.progress >= 0 && frame.progress <= 1);
      assert.ok((frame.intro || frame.caption).length > 0, `${scene} needs an introduction or caption`);
      assert.ok(frame.equation.length > 0, `${scene} needs an equation`);
      allNumbersFinite(frame);
      if (frame.model) {
        assert.equal(frame.model.inputSize, 1);
        assert.equal(frame.model.hiddenLayers, 1);
        assert.equal(frame.model.outputSize, 1);
        assert.ok(!Object.hasOwn(frame.model, 'depth'));
        assert.ok(frame.model.neurons.every(unit => typeof unit.t === 'number' && typeof unit.a === 'number'));
        const inputs = frame.kind === 'sine' ? grid(0, M.TAU) : grid(-2, 2);
        for (const x of inputs) {
          assert.ok(Number.isFinite(M.evaluate(frame.model, x)), `${scene} output at ${x}`);
          allNumbersFinite(M.contributions(frame.model, x), `${scene}.contributions`);
        }
      }
    }
  }
});

test('one hinge changes its location, steepness, and sign while remaining a single ReLU', () => {
  const anchors = [[0, 1, -.8], [.4, 1, .6], [.7, 1.6, .6], [1, -1.2, .6]];
  for (const [progress, a, t] of anchors) {
    const frame = S.frame('hinge', progress);
    close(frame.hingeA, a);
    close(frame.hingeT, t);
    close(frame.model.neurons[0].a, a);
    close(frame.model.neurons[0].t, t);
  }
  const locations = new Set(), weights = [];
  for (const progress of grid(0, 1, 100)) {
    const frame = S.frame('hinge', progress), unit = frame.model.neurons[0];
    assert.equal(frame.model.width, 1);
    assert.equal(frame.model.neurons.length, 1);
    assert.equal(frame.model.hiddenLayers, 1);
    close(frame.model.c, 0);
    close(unit.w, 1);
    close(unit.b, -frame.hingeT);
    close(unit.a, frame.hingeA);
    close(unit.t, frame.hingeT);
    assert.deepEqual(frame.hingeExamples.map(example => ({ a: example.a, t: example.t })),
      [{ a: 1, t: -.8 }, { a: 1, t: .6 }], 'Reference curves are separate fixed examples');
    assert.equal(frame.hingePhase, progress < .4 ? 'threshold' : progress < .7 ? 'steepen' : 'flip');
    for (const x of [unit.t - 1, unit.t - .01, unit.t]) close(M.evaluate(frame.model, x), 0);
    for (const offset of [.1, .5, 1]) close(M.evaluate(frame.model, unit.t + offset), unit.a * offset);
    const epsilon = 1e-5, x = unit.t + .25;
    close((M.evaluate(frame.model, x + epsilon) - M.evaluate(frame.model, x - epsilon)) / (2 * epsilon), unit.a, 1e-8,
      'The active piece has the displayed output slope, including negative slopes.');
    locations.add(unit.t.toFixed(4)); weights.push(unit.a);
  }
  assert(locations.size > 10, 'Scrolling moves the joint through distinct locations');
  assert(weights.some(a => a > 1.5), 'Scrolling steepens the hinge');
  assert(weights.some(a => a < -1), 'Scrolling reverses the hinge slope');
  const epsilon = 1e-7;
  for (const boundary of [.4, .7]) {
    for (const parameter of ['hingeA', 'hingeT']) {
      const before = S.frame('hinge', boundary - epsilon)[parameter];
      const at = S.frame('hinge', boundary)[parameter];
      const after = S.frame('hinge', boundary + epsilon)[parameter];
      close(before, at, 1e-5, 'The parameter values do not jump at phase boundaries.');
      close(after, at, 1e-5, 'The parameter values do not jump at phase boundaries.');
    }
  }
  const original = S.frame('hinge', .2);
  S.frame('hinge', .99); S.frame('hinge', .65);
  assert.deepEqual(S.frame('hinge', .2), original, 'Returning to earlier progress restores the same hinge');
});

test('the actual coarse sine curve has five visible joints and width six', () => {
  const frame = S.frame('sine-coarse', 0);
  const model = frame.model;
  assert.equal(frame.width, 6);
  assert.equal(model.width, 6);
  assert.equal(model.knots.length, 7);
  assert.deepEqual(model.knots, [...S.initialKnots]);
  assert.equal(model.neurons[0].baseline, true);
  assert.equal(model.neurons[0].t, -1);
  assert.equal(model.neurons.slice(1).filter(unit => Math.abs(unit.a) > 1e-10).length, 5);

  for (let i = 0; i < model.knots.length; i += 1) {
    close(M.evaluate(model, model.knots[i]), Math.sin(model.knots[i]));
  }
  for (let i = 0; i < model.knots.length - 1; i += 1) {
    const left = model.knots[i], right = model.knots[i + 1];
    for (const fraction of [0.1, 0.37, 0.75]) {
      const x = left + fraction * (right - left);
      close(M.evaluate(model, x), Math.sin(left) + fraction * (Math.sin(right) - Math.sin(left)));
    }
  }
});

test('the three-neuron hill has slopes 0, +1, -1, 0 and continuous scene boundaries', () => {
  const full = S.frame('hill-flat', 1).model;
  assert.equal(full.width, 3);
  const epsilon = 1e-4;
  for (const [x, slope] of [[-1.5, 0], [-0.5, 1], [0.5, -1], [1.5, 0]]) {
    close((M.evaluate(full, x + epsilon) - M.evaluate(full, x - epsilon)) / (2 * epsilon), slope);
  }
  for (const [x, y] of [[-2, 0], [-1, 0], [-0.5, 0.5], [0, 1], [0.5, 0.5], [1, 0], [2, 0]]) {
    close(M.evaluate(full, x), y);
  }
  const boundaries = [
    [S.frame('hill-rise', 1), S.frame('hill-fall', 0)],
    [S.frame('hill-fall', 1), S.frame('hill-flat', 0)],
    [S.frame('hill-flat', 1), S.frame('shallow', 0)],
  ];
  for (const [before, after] of boundaries) {
    for (const x of grid(-2, 2)) close(M.evaluate(before.model, x), M.evaluate(after.model, x));
  }
});

test('width one, two, three, and six give genuine slope jumps in one hidden layer', () => {
  const units = [{t:-1.5,a:.8},{t:-.9,a:-1.2},{t:-.3,a:1.1},{t:.3,a:-1},{t:.9,a:.9},{t:1.5,a:-.6}];
  for (const [scene, width] of [
    ['width-one', 1], ['width-two', 2], ['width-three', 3], ['width-many', 6],
  ]) {
    const frame = S.frame(scene, 1);
    assert.equal(frame.kind, 'shallow');
    assert.equal(frame.width, width);
    assert.equal(frame.model.width, width);
    assert.equal(frame.model.hiddenLayers, 1);
    close(frame.x, 0.4);
    close(frame.model.c, 0);
    assert.deepEqual(frame.model.neurons.map(({t,a}) => ({t,a})), units.slice(0,width));
    assert.equal(new Set(frame.model.neurons.map(unit => unit.t)).size, width);
    assert.equal(frame.model.neurons.filter(unit => unit.a !== 0).length, width);
    assert.match(frame.caption, /upper bound/);
    assert.match(frame.equation, /\\text\{joints\}\\le m/);
    assert.match(frame.equation, /\\text\{pieces\}\\le m\+1/);
    const epsilon = 1e-3;
    for (const unit of frame.model.neurons) {
      const leftSlope = (M.evaluate(frame.model, unit.t-epsilon)-M.evaluate(frame.model, unit.t-2*epsilon))/epsilon;
      const rightSlope = (M.evaluate(frame.model, unit.t+2*epsilon)-M.evaluate(frame.model, unit.t+epsilon))/epsilon;
      close(rightSlope-leftSlope, unit.a);
    }
  }
  let previous = 3;
  for (const progress of grid(0,1,100)) {
    const frame = S.frame('width-many', progress);
    assert.ok(frame.width >= previous && frame.width <= 6);
    assert.equal(frame.model.width, frame.width);
    assert.equal(frame.model.hiddenLayers, 1);
    previous = frame.width;
  }
  assert.equal(S.frame('width-many', 0).width, 3);
  assert.equal(S.frame('width-many', 0.25).width, 4);
  assert.equal(S.frame('width-many', 0.5).width, 5);
  assert.equal(S.frame('width-many', 1).width, 6);
  for (const name of ['tent', 'deepValue', 'deepActivations', 'deepJoints']) {
    assert.ok(!Object.hasOwn(M, name));
  }
});

test('inserting one sine joint starts at the old curve and ends exactly on sine', () => {
  const coarse = S.frame('sine-coarse', 1).model;
  const start = S.frame('sine-joint', 0);
  const end = S.frame('sine-joint', 1);
  assert.equal(start.width, coarse.width + 1);
  assert.equal(end.width, coarse.width + 1);
  assert.equal(start.model.width, coarse.width + 1);
  assert.equal(end.model.width, coarse.width + 1);
  assert.equal(start.x, S.jointX);
  assert.equal(end.x, S.jointX);

  const inserted = end.model.neurons.filter(unit => !coarse.neurons.some(old => Math.abs(old.t - unit.t) < 1e-10));
  assert.equal(inserted.length, 1);
  close(inserted[0].t, S.jointX);
  close(start.model.neurons.find(unit => Math.abs(unit.t - S.jointX) < 1e-10).a, 0);
  assert.ok(Math.abs(M.evaluate(coarse, S.jointX) - Math.sin(S.jointX)) > 0.1);
  close(M.evaluate(end.model, S.jointX), Math.sin(S.jointX));
  const changedExisting = coarse.neurons.filter(old => {
    const updated = end.model.neurons.find(unit => Math.abs(old.t - unit.t) < 1e-10);
    return updated && Math.abs(updated.a - old.a) > 1e-8;
  });
  assert.ok(changedExisting.length > 0, 'The construction must update existing output weights too.');

  for (const x of grid(0, M.TAU)) close(M.evaluate(start.model, x), M.evaluate(coarse, x));
  for (const progress of [0, 0.1, 0.4, 0.75, 1]) {
    const frame = S.frame('sine-joint', progress);
    const signedError = M.evaluate(frame.model, S.jointX) - Math.sin(S.jointX);
    close(signedError, (1 - progress) * (M.evaluate(coarse, S.jointX) - Math.sin(S.jointX)));
    for (const x of S.initialKnots) close(M.evaluate(frame.model, x), Math.sin(x));
  }
});

test('the widening scene progresses from width 7 to 26 with decreasing actual error', () => {
  let previous = S.approximation(6);
  let previousError = sampledError(previous.model);
  for (let width = 7; width <= 26; width += 1) {
    const progress = width === 26 ? 1 : (width - 7 + 0.25) / 20;
    const frame = S.frame('sine-wide', progress);
    const current = S.approximation(width);
    assert.equal(frame.width, width);
    assert.equal(frame.model.width, width);
    assert.equal(current.model.width, width);
    assert.equal(current.knots.length, previous.knots.length + 1);
    for (const x of previous.knots) assert.ok(current.knots.includes(x));
    const additions = current.knots.filter(x => !previous.knots.includes(x));
    assert.equal(additions.length, 1);
    close(additions[0], M.maxError(previous.knots).x);
    close(M.evaluate(current.model, additions[0]), Math.sin(additions[0]));
    close(current.error, M.maxError(current.knots).error);
    const actualError = sampledError(current.model);
    assert.ok(actualError < previousError, `Maximum error must decrease at width ${width}.`);
    previous = current;
    previousError = actualError;
  }
  assert.equal(S.frame('sine-wide', 0).width, 7);
  assert.equal(S.frame('sine-wide', 1).width, 26);
  assert.equal(S.frame('universal', 0).width, 26);
  assert.ok(previousError < 0.01);
});

test('weight blending is linear in function space, including outside the plotted domain', () => {
  for (const [beforeWidth, afterWidth] of [[6, 7], [7, 26]]) {
    const before = S.approximation(beforeWidth).model;
    const after = S.approximation(afterWidth).model;
    for (const alpha of [-0.5, 0, 0.1, 0.4, 0.75, 1, 1.5]) {
      const p = Math.max(0, Math.min(1, alpha));
      const blended = S.blend(before, after, alpha);
      assert.equal(blended.width, after.width);
      allNumbersFinite(blended, 'blended');
      for (const x of grid(-2, M.TAU + 2, 400)) {
        close(M.evaluate(blended, x),
          (1 - p) * M.evaluate(before, x) + p * M.evaluate(after, x));
      }
    }
  }
});

test('adding fixed sigmoid features changes a monotone probability into one hump', () => {
  const sigmoid = value => 1 / (1 + Math.exp(-value));
  const probability = (frame, x) => sigmoid(frame.sigmoidBias + frame.sigmoidWeights.reduce(
    (sum, weight, j) => sum + weight * sigmoid(30 * (x - frame.sigmoidThresholds[j])), 0));
  const peaks = frame => {
    const values = grid(0, 1, 2000).map(x => probability(frame, x));
    return values.slice(1, -1).filter((value, i) => value > values[i] && value > values[i + 2]).length;
  };
  const one = S.frame('sigmoid-one', 1);
  const two = S.frame('sigmoid-two', 1);
  assert.deepEqual([one.width, two.width], [1, 2]);
  let previous = 0;
  for (const x of grid(0, 1, 1000)) {
    const value = probability(one, x);
    assert(value > previous, 'A single hidden sigmoid remains monotone');
    previous = value;
    for (const frame of [one, two]) {
      assert(probability(frame, x) > 0 && probability(frame, x) < 1, 'The output sigmoid is a probability');
    }
  }
  assert.equal(peaks(one), 0);
  assert.equal(peaks(two), 1);
  for (const [beforeScene, afterScene] of [['sigmoid-one', 'sigmoid-two']]) {
    const before = S.frame(beforeScene, 1), after = S.frame(afterScene, 0);
    for (const x of grid(0, 1)) close(probability(before, x), probability(after, x));
    for (const progress of grid(0, 1, 20)) {
      const current = S.frame(afterScene, progress);
      assert.deepEqual(current.sigmoidWeights.slice(0, before.width), before.sigmoidWeights,
        'Additional neurons preserve every existing contribution');
      assert.deepEqual(current.sigmoidThresholds, before.sigmoidThresholds);
      close(current.sigmoidBias, before.sigmoidBias);
    }
  }
});

test('the fitted binary network exposes all nine frozen parameters to three decimals', () => {
  const frame = S.frame('class-input', .5);
  const expected = [
    D.hiddenBias[0], ...D.hiddenWeights[0],
    D.hiddenBias[1], ...D.hiddenWeights[1],
    D.outputBias, ...D.outputWeights
  ].map(value => Number(value.toFixed(3)));
  assert.equal(S.scenes.indexOf('class-input'), 19, 'The fitted network now appears on slide 20');
  for (const source of [frame.equation, frame.equationCompact]) {
    assert.match(source, /h_1/);
    assert.match(source, /h_2/);
    assert.equal((source.match(/\\operatorname\{ReLU\}/g) || []).length, 2,
      'Both hidden neurons retain their ReLU activation');
    assert.equal((source.match(/\\operatorname\{sigmoid\}/g) || []).length, 1,
      'The fitted classification head uses a sigmoid');
    assert.equal((source.match(/\\approx/g) || []).length, 3,
      'Rounded parameter formulas are marked as approximations');
    const values = [...source.replace(/\s/g, '').matchAll(/[+-]?\d+\.\d{3}/g)].map(match => Number(match[0]));
    assert.deepEqual(values, expected, 'The displayed coefficients match the frozen fit in neuron and input order');
  }
});

test('classification separates coordinate movement from the later boundary reveal at every progress value', () => {
  let previousMorph = -1, previousOpacity = -1;
  for (const progress of grid(0, 1, 100)) {
    const moving = S.frame('class-move', progress);
    assert.equal(moving.kind, 'classification');
    assert(moving.morph >= previousMorph && moving.morph <= 1);
    assert.equal(moving.boundaryOpacity, 0, 'Even the completed movement has no head boundary');
    previousMorph = moving.morph;
    const head = S.frame('class-head', progress);
    assert.equal(head.morph, 1, 'The head acts on completed hidden coordinates');
    assert(head.boundaryOpacity >= previousOpacity && head.boundaryOpacity <= 1);
    previousOpacity = head.boundaryOpacity;
  }
  assert.equal(S.frame('class-move', 0).morph, 0);
  assert.equal(S.frame('class-move', 1).morph, 1);
  assert.equal(S.frame('class-head', 0).boundaryOpacity, 0);
  assert.equal(S.frame('class-head', 1).boundaryOpacity, 1);
  for (const scene of ['class-input', 'class-trace']) {
    const frame = S.frame(scene, 1);
    assert.equal(frame.morph, 0);
    assert.equal(frame.boundaryOpacity, 0);
  }
  const compare = S.frame('class-compare', .5);
  assert.equal(compare.morph, 1);
  assert.equal(compare.boundaryOpacity, 1);
  const earlier = S.frame('class-move', .37);
  S.frame('class-head', 1); S.frame('class-compare', 1);
  assert.deepEqual(S.frame('class-move', .37), earlier, 'Reverse scrolling returns to the same intermediate representation');
});
