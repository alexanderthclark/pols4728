import test from 'node:test';
import assert from 'node:assert/strict';
import {attentionInterpretation, deletionExperiment} from '../docs/attention/interpretation-model.mjs';
import {createInterpretationScenes} from '../docs/attention/interpretation-scenes.mjs';

const close = (actual, expected, tolerance = 1e-12) => assert.ok(Math.abs(actual - expected) < tolerance,
  `expected ${actual} to equal ${expected} within ${tolerance}`);

test('constructed queries and keys realize all displayed attention rows', () => {
  const model = attentionInterpretation();
  const expected = [[.80, .15, .05], [.60, .30, .10], [.40, .45, .15]];
  assert.deepEqual(model.queries, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
  expected.forEach((row, i) => {
    row.forEach((weight, j) => {
      close(model.attentionMatrix[i][j], weight);
      close(model.scores[i][j], Math.log(weight));
      assert.ok(model.attentionMatrix[i][j] > 0);
    });
    close(model.attentionMatrix[i].reduce((sum, weight) => sum + weight, 0), 1);
  });
  assert.notEqual(model.attentionMatrix[0][1], model.attentionMatrix[1][0]);
  close(model.attentionMatrix.reduce((sum, row) => sum + row[0], 0), 1.8);
});

test('each token gets its own output and mean pooling yields the displayed summary', () => {
  const model = attentionInterpretation();
  const expectedOutputs = [[-.32, 3.45], [-.74, 2.90], [-1.16, 2.35]];
  expectedOutputs.forEach((row, i) => row.forEach((value, coordinate) =>
    close(model.outputs[i][coordinate], value)));
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

test('changing the last value preserves queries, keys and attention while reversing the predicted class', () => {
  const negative = attentionInterpretation();
  const positive = attentionInterpretation({tone: 'great'});
  assert.deepEqual(positive.queries, negative.queries);
  assert.deepEqual(positive.keys, negative.keys);
  assert.deepEqual(positive.scores, negative.scores);
  assert.deepEqual(positive.attentionMatrix, negative.attentionMatrix);
  assert.deepEqual(positive.weights, negative.weights);
  assert.deepEqual(positive.readout, negative.readout);
  assert.deepEqual(negative.projections, [.1, 0, -8]);
  assert.deepEqual(positive.projections, [.1, 0, 8]);
  close(positive.output[0], .86);
  close(positive.output[1], 2.9);
  close(positive.logit, .86);
  close(positive.probability, .7026606543447315);
  assert.ok(negative.probability < .5 && positive.probability > .5);
});

test('deletion removes the destination row and source column before renormalizing and pooling', () => {
  const result = deletionExperiment();
  assert.deepEqual(result.deleted.tokens, ['was', 'awful']);
  assert.equal(result.deleted.attentionMatrix.length, 2);
  result.deleted.attentionMatrix.forEach(row => {
    assert.equal(row.length, 2);
    [.75, .25].forEach((weight, j) => close(row[j], weight));
  });
  result.deleted.outputs.forEach(row => {
    close(row[0], -2);
    close(row[1], 1.25);
  });
  [.75, .25].forEach((weight, j) => close(result.deleted.weights[j], weight));
  close(result.deleted.output[0], -2);
  close(result.deleted.output[1], 1.25);
  close(result.deleted.logit, -2);
  close(result.originalTerm, .06);
  close(result.logitChange, -1.26);
  assert.notEqual(result.logitChange, -result.originalTerm);
  assert.deepEqual(result.deleted.queries, [[0, 1, 0], [0, 0, 1]]);
  assert.deepEqual(result.deleted.keys, result.original.keys.slice(1));
  assert.deepEqual(result.deleted.values, result.original.values.slice(1));
  assert.deepEqual(result.deleted.scores, result.original.scores.slice(1).map(row => row.slice(1)));
});

test('all thirteen scenes render deterministically, including reversible discrete comparisons', () => {
  const colors = {blue: '#234e70', rust: '#a54f32', teal: '#007f73', ink: '#242424', muted: '#606060', rule: '#dededb'};
  const drawing = {C: colors,
    m: (tex, ...args) => `<math data-tex="${tex}" data-args="${args.join(',')}"/>`,
    line: (...args) => `<line data-args="${args.join(',')}"/>`,
    rect: (...args) => `<rect data-args="${args.join(',')}"/>`};
  const scenes = createInterpretationScenes(drawing);
  assert.equal(scenes.length, 13);
  assert.deepEqual(scenes.map(scene => scene.id), [
    'matrix', 'entry', 'row', 'values', 'mixture', 'outputs', 'roles',
    'pooling', 'readout', 'terms', 'substitution', 'deletion', 'uses',
  ]);
  const sceneById = id => {
    const scene = scenes.find(candidate => candidate.id === id);
    assert.ok(scene, `missing scene ${id}`);
    return scene;
  };
  for (const scene of scenes) {
    const initial = scene.draw(.1);
    assert.match(initial, /interpretation-label/);
    assert.doesNotMatch(initial, /NaN|undefined/);
    scene.draw(.9);
    assert.equal(scene.draw(.1), initial);
  }
  assert.match(sceneById('substitution').draw(.1), /32\.3\\%/);
  assert.match(sceneById('substitution').draw(.9), /70\.3\\%/);
  assert.match(sceneById('deletion').draw(.9), /−2\.00/);
  for (const id of ['mixture', 'readout']) {
    assert.notEqual(sceneById(id).draw(.1), sceneById(id).draw(.9));
  }
  assert.match(sceneById('mixture').draw(.5), /u_\{1,\\mathrm\{partial\}\}/);
  assert.match(sceneById('mixture').draw(1), /u_1=\\left\[-0\.32,\\;3\.45\\right\]/);
  assert.match(sceneById('readout').draw(1), /32\.3\\%/);
});

