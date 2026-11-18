import test from 'node:test';
import assert from 'node:assert/strict';
import {attentionInterpretation, deletionExperiment} from '../docs/attention/interpretation-model.mjs';
import {createInterpretationScenes} from '../docs/attention/interpretation-scenes.mjs';

const close = (actual, expected, tolerance = 1e-12) => assert.ok(Math.abs(actual - expected) < tolerance,
  `expected ${actual} to equal ${expected} within ${tolerance}`);

test('constructed keys realize the displayed weights and two-coordinate output', () => {
  const model = attentionInterpretation();
  [.6, .3, .1].forEach((weight, j) => close(model.weights[j], weight));
  close(model.weights.reduce((sum, weight) => sum + weight, 0), 1);
  close(model.output[0], -.74);
  close(model.output[1], 2.9);
  close(model.logit, -.74);
  close(model.probability, .323004143761477);
});

test('logit terms sum to the score but the largest weight is not its largest signed term', () => {
  const model = attentionInterpretation();
  [.06, 0, -.8].forEach((term, j) => close(model.contributions[j], term));
  close(model.contributions.reduce((sum, value) => sum + value, 0), model.logit);
  assert.equal(model.tokens[model.weights.indexOf(Math.max(...model.weights))], 'film');
  assert.equal(model.tokens[model.contributions.indexOf(Math.min(...model.contributions))], 'awful');
});

test('changing the last value preserves keys and weights while reversing the predicted class', () => {
  const negative = attentionInterpretation();
  const positive = attentionInterpretation({tone: 'great'});
  assert.deepEqual(positive.keys, negative.keys);
  assert.deepEqual(positive.weights, negative.weights);
  close(positive.output[0], .86);
  close(positive.output[1], 2.9);
  close(positive.logit, .86);
  close(positive.probability, .7026606543447315);
  assert.ok(negative.probability < .5 && positive.probability > .5);
});

test('deletion recomputes the mixture instead of just subtracting a direct term', () => {
  const result = deletionExperiment();
  assert.deepEqual(result.deleted.tokens, ['was', 'awful']);
  [.75, .25].forEach((weight, j) => close(result.deleted.weights[j], weight));
  close(result.deleted.logit, -2);
  close(result.originalTerm, .06);
  close(result.logitChange, -1.26);
  assert.notEqual(result.logitChange, -result.originalTerm);
  assert.deepEqual(result.deleted.keys, result.original.keys.slice(1));
  assert.deepEqual(result.deleted.values, result.original.values.slice(1));
});

test('all eight scenes render deterministically, including reversible discrete comparisons', () => {
  const colors = {blue: '#234e70', rust: '#a54f32', teal: '#007f73', ink: '#242424', muted: '#606060', rule: '#dededb'};
  const drawing = {C: colors,
    m: (tex, ...args) => `<math data-tex="${tex}" data-args="${args.join(',')}"/>`,
    line: (...args) => `<line data-args="${args.join(',')}"/>`,
    rect: (...args) => `<rect data-args="${args.join(',')}"/>`};
  const scenes = createInterpretationScenes(drawing);
  assert.equal(scenes.length, 8);
  for (const scene of scenes) {
    const initial = scene.draw(.1);
    assert.match(initial, /interpretation-label/);
    assert.doesNotMatch(initial, /NaN|undefined/);
    scene.draw(.9);
    assert.equal(scene.draw(.1), initial);
  }
  assert.match(scenes[5].draw(.1), /32\.3\\%/);
  assert.match(scenes[5].draw(.9), /70\.3\\%/);
  assert.match(scenes[6].draw(.9), /−2\.00/);
  for (const index of [0, 2, 3]) assert.notEqual(scenes[index].draw(.1), scenes[index].draw(.9));
  assert.match(scenes[2].draw(.5), /h_\{\\mathrm\{partial\}\}/);
  assert.match(scenes[2].draw(1), /h=\\left\[-0\.74,\\;2\.90\\right\]/);
  assert.match(scenes[3].draw(1), /32\.3\\%/);
});
