import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { buildPopulation, explainValues, globalImportance } from '../docs/shap/aggregate.mjs';

const data = JSON.parse(await readFile(new URL('../docs/shap/data.json', import.meta.url), 'utf8'));
const model = JSON.parse(await readFile(new URL('../docs/shap/model.json', import.meta.url), 'utf8'));
const population = buildPopulation(data, model);
const closeTo = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-10,
  `${actual} should equal ${expected}`);
const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;

test('new explanations exactly reproduce every stored coalition, order, and SHAP vector', () => {
  for (const observation of data.observations) {
    const calculated = explainValues(observation.values, data, model);
    const { label, ...originalShape } = calculated;
    assert.equal(label, observation.name);
    assert.deepEqual(originalShape, observation);
  }
});

test('the summary cohort is a deterministic, balanced 125-profile grid containing the original person', () => {
  assert.equal(population.length, 125);
  assert.equal(new Set(population.map(observation => observation.id)).size, 125);
  assert.equal(new Set(population.map(observation => observation.values.join(','))).size, 125);
  assert.equal(population[0].id, data.defaultObservationId);
  assert.deepEqual(population[0].values, [1, -1, 1]);
  assert.deepEqual(buildPopulation(data, model), population);
  const expectedLevels = [[0, .25, .5, .75, 1], [-1, -.5, 0, .5, 1], [0, .25, .5, .75, 1]];
  expectedLevels.forEach((levels, feature) => {
    for (const level of levels) assert.equal(population.filter(observation =>
      observation.values[feature] === level).length, 25);
  });
  for (const observation of population) assert.ok(observation.name && observation.label);
});

test('every person uses the same baseline and satisfies exact SHAP efficiency and analytic contributions', () => {
  for (const observation of population) {
    const [ability, neighborhood, experience] = observation.values;
    const expected = [
      (ability - .5) * (10 + 12 * neighborhood),
      12 * neighborhood * (ability + .5),
      4 * (experience - .5),
    ];
    assert.equal(observation.baseValue, 47);
    observation.shapValues.forEach((value, feature) => closeTo(value, expected[feature]));
    closeTo(observation.baseValue + observation.shapValues.reduce((sum, value) => sum + value, 0),
      40 + 10 * ability + 24 * ability * neighborhood + 4 * experience);
  }
  closeTo(mean(population.map(observation => observation.prediction)), 47);
});

test('averaging signed contributions cancels activity that mean absolute SHAP retains', () => {
  const importance = globalImportance(population);
  [3.24, 7.2, 1.2].forEach((expected, feature) => {
    closeTo(importance[feature], expected);
    closeTo(mean(population.map(observation => observation.shapValues[feature])), 0);
  });
  assert.deepEqual([0, 1, 2].sort((left, right) => importance[right] - importance[left]), [1, 0, 2]);
  assert.ok(population.some(observation => observation.shapValues[0] < 0));
  assert.ok(population.some(observation => observation.shapValues[0] > 0));
});

test('global importance describes the chosen cohort, whose members need not be the reference profiles', () => {
  const favorable = population.filter(observation => observation.values[1] === 1);
  const adverse = population.filter(observation => observation.values[1] === -1);
  const favorableImportance = globalImportance(favorable);
  const adverseImportance = globalImportance(adverse);
  closeTo(favorableImportance[0], 6.6);
  closeTo(adverseImportance[0], .6);
  closeTo(favorableImportance[1], 12);
  closeTo(adverseImportance[1], 12);
  assert.equal(favorable[0].baseValue, adverse[0].baseValue);
  assert.equal(data.background.length, 8);
});

test('coefficients are evaluated from the supplied fixed model and source objects remain unchanged', () => {
  const before = JSON.stringify({ data, model });
  const adjusted = { ...model, coefficients: { ...model.coefficients, experience: 8 } };
  const observation = explainValues([1, -1, 1], data, adjusted);
  assert.equal(observation.prediction, 34);
  assert.equal(observation.baseValue, 49);
  assert.deepEqual(observation.shapValues, [-1, -18, 4]);
  assert.equal(JSON.stringify({ data, model }), before);
});

test('invalid feature values or incomplete SHAP populations cannot silently enter a summary', () => {
  assert.throws(() => explainValues([1, 2, 1], data, model), RangeError);
  assert.throws(() => explainValues([1, NaN, 1], data, model), RangeError);
  assert.throws(() => explainValues([1, -1], data, model), RangeError);
  assert.throws(() => globalImportance([]), TypeError);
  assert.throws(() => globalImportance([{ shapValues: [1, 2] }, { shapValues: [1] }]), TypeError);
  assert.throws(() => globalImportance([{ shapValues: [Infinity] }]), TypeError);
});
