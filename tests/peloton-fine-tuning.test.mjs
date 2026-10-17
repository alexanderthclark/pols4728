import test from 'node:test';
import assert from 'node:assert/strict';
import { lesson } from '../docs/peloton-fine-tuning/lesson.mjs';
import { softmax, forward, crossEntropy, gradients, update, initialParameters, completedParameters, observations, probeSentences, interpolateParameters } from '../docs/peloton-fine-tuning/model.mjs';

test('softmax is a probability distribution and is stable under a large offset', () => {
  const probabilities = softmax([1000, 1001, 999]);
  assert.ok(probabilities.every(value => value > 0 && value < 1));
  assert.ok(Math.abs(probabilities.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  softmax([0, 1, -1]).forEach((value, k) => assert.ok(Math.abs(value - probabilities[k]) < 1e-12));
});

test('the surrogate gradient agrees with finite differences for encoder, head, and bias', () => {
  const analytic = gradients();
  const epsilon = 1e-6;
  for (const name of ['encoder', 'head', 'bias']) {
    const entries = name === 'bias' ? initialParameters[name].map((_, j) => [j]) : initialParameters[name].flatMap((row, i) => row.map((_, j) => [i, j]));
    for (const indices of entries) {
      const perturb = amount => {
        const parameters = structuredClone(initialParameters);
        if (indices.length === 1) parameters[name][indices[0]] += amount;
        else parameters[name][indices[0]][indices[1]] += amount;
        return crossEntropy(forward(parameters).probabilities);
      };
      const numerical = (perturb(epsilon) - perturb(-epsilon)) / (2 * epsilon);
      const exact = indices.length === 1 ? analytic[name][indices[0]] : analytic[name][indices[0]][indices[1]];
      assert.ok(Math.abs(numerical - exact) < 1e-7, `${name}:${indices}`);
    }
  }
});

test('the displayed full update changes both parts and reduces this example loss', () => {
  const before = forward(), after = forward(completedParameters);
  assert.ok(after.probabilities[0] > before.probabilities[0]);
  assert.ok(crossEntropy(after.probabilities) < crossEntropy(before.probabilities));
  assert.notDeepEqual(completedParameters.encoder, initialParameters.encoder);
  assert.notDeepEqual(completedParameters.head, initialParameters.head);
  assert.deepEqual(update(initialParameters, 0.8, { freezeEncoder: true }).encoder, initialParameters.encoder);
  assert.deepEqual(interpolateParameters(initialParameters, completedParameters, 0), initialParameters);
  assert.deepEqual(interpolateParameters(initialParameters, completedParameters, 1), completedParameters);
});

test('schematic split is exactly stratified 80/20 in all three categories', () => {
  assert.equal(observations.filter(observation => observation.validation).length, 6);
  for (let category = 0; category < 3; category++) {
    const row = observations.filter(observation => observation.category === category);
    assert.equal(row.length, 10);
    assert.equal(row.filter(observation => observation.validation).length, 2);
  }
});

test('the same observations move continuously and reverse scroll reproduces an earlier state', () => {
  const before = lesson.render(2.2), middle = lesson.render(2.5), after = lesson.render(2.8);
  const x = svg => Number(svg.match(/data-observation="0" cx="([\d.]+)"/)[1]);
  assert.ok(x(before) > x(middle) && x(middle) > x(after));
  lesson.render(9.7);
  assert.equal(lesson.render(2.2), before);
  assert.notEqual(lesson.render(7.7), lesson.render(8.2));
});

test('every scene has desktop and dedicated compact SVG without invalid values', () => {
  assert.equal(lesson.scenes.length, 14);
  for (let p = 0; p <= lesson.scenes.length - 1; p += 0.25) {
    for (const compact of [false, true]) {
      const svg = lesson.render(p, { compact });
      assert.doesNotMatch(svg, /NaN|Infinity|foreignObject|undefined/);
      assert.ok([...svg.matchAll(/font-size="([\d.]+)"/g)].every(match => Number(match[1]) >= 20));
      assert.equal([...svg.matchAll(/data-observation="\d+"/g)].length, 30);
    }
  }
  assert.notEqual(lesson.render(5), lesson.render(5, { compact: true }));
});

test('numerical update and four-epoch progression are complete at their scene anchors', () => {
  const completed = forward(completedParameters);
  for (const compact of [false, true]) {
    const updateSVG = lesson.render(8, { compact });
    assert.match(updateSVG, /One gradient step: 100%/);
    assert.ok(updateSVG.includes(`p(Coaching) = ${completed.probabilities[0].toFixed(3)}`));
    const epochSVG = lesson.render(10, { compact });
    assert.equal([...epochSVG.matchAll(/data-epoch="[1-4]" data-complete="1"/g)].length, 4);
  }
});

test('intermediate scroll contains one annotation state rather than overlapping captions', () => {
  const svg = lesson.render(6.6);
  assert.match(svg, /Target label: Coaching/);
  assert.doesNotMatch(svg, /Logits are scores, not probabilities/);
  assert.equal([...svg.matchAll(/data-observation="\d+"/g)].length, 30);
});

test('provenance is explicit and supplied probe sentences retain their wording', () => {
  const svg = lesson.render(1);
  for (const sentence of probeSentences) assert.ok(svg.includes(sentence));
  assert.match(lesson.dataNote, /not DistilBERT outputs/);
  assert.match(lesson.dataNote, /illustrative categories/);
  assert.match(lesson.scenes[11].reading, /No empirical score/);
  lesson.scenes.forEach((scene, i) => {
    const words = scene.body.replace(/<[^>]+>/g, '').split(/\s+/).length;
    assert.ok(words <= (i === 0 ? 60 : 50), `${scene.id}: ${words} words`);
    assert.ok(scene.reading.length > scene.body.length);
  });
});
