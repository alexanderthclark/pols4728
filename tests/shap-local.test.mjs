import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const data = JSON.parse(await readFile(new URL('../docs/shap/data.json', import.meta.url), 'utf8'));
const model = JSON.parse(await readFile(new URL('../docs/shap/model.json', import.meta.url), 'utf8'));
const featureCount = data.features.length;
const fullMask = (1 << featureCount) - 1;
const tolerance = 1e-9;

function closeTo(actual, expected, context) {
  assert.ok(Number.isFinite(actual) && Number.isFinite(expected), `${context}: values must be finite`);
  assert.ok(Math.abs(actual - expected) <= tolerance,
    `${context}: ${actual} should equal ${expected}`);
}

const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
const factorial = n => n < 2 ? 1 : n * factorial(n - 1);
const sizeOf = mask => Array.from({ length: featureCount }, (_, feature) => Boolean(mask & (1 << feature)))
  .filter(Boolean).length;

// Evaluate the stated illustrative equation independently of the website's
// model, masking, and marginal-contribution helpers.
function predict([ability, neighborhood, experience]) {
  return 40 + 10 * ability + 24 * ability * neighborhood + 4 * experience;
}

function reconstruct(observation) {
  return Array.from({ length: fullMask + 1 }, (_, mask) => {
    const predictions = data.background.map(background => predict(background.values.map((value, feature) =>
      mask & (1 << feature) ? observation.values[feature] : value)));
    return { predictions, value: mean(predictions) };
  });
}

const reconstructed = new Map(data.observations.map(observation => [observation.id, reconstruct(observation)]));

test('the illustrative interaction equation has the complete equally weighted eight-profile background', () => {
  assert.equal(featureCount, 3);
  assert.equal(model.type, 'IllustrativeEarningsInteraction');
  assert.equal(model.intercept, 40);
  assert.deepEqual(model.coefficients, { ability: 10, abilityNeighborhood: 24, experience: 4 });
  assert.deepEqual(model.inputFeatures, ['ability', 'neighborhood', 'experience']);
  assert.deepEqual(model.inputFeatures, data.features.map(feature => feature.id));
  assert.deepEqual(model.featureDomains, { ability: [0, 1], neighborhood: [-1, 1], experience: [0, 1] });
  assert.equal(data.background.length, 8);
  assert.equal(data.background.length, data.method.backgroundSize);
  assert.equal(new Set(data.background.map(row => row.id)).size, data.background.length);
  assert.equal(new Set(data.observations.map(row => row.id)).size, data.observations.length);
  assert.ok(data.observations.some(row => row.id === data.defaultObservationId));
  const expectedBackground = new Set();
  for (const ability of [0, 1]) for (const neighborhood of [-1, 1]) for (const experience of [0, 1]) {
    expectedBackground.add([ability, neighborhood, experience].join(','));
  }
  assert.deepEqual(new Set(data.background.map(row => row.values.join(','))), expectedBackground);
  for (const row of [...data.background, ...data.observations]) {
    assert.equal(row.values.length, featureCount, row.id);
    row.values.forEach((value, feature) => {
      const [lower, upper] = model.featureDomains[model.inputFeatures[feature]];
      assert.ok(Number.isFinite(value) && value >= lower && value <= upper,
        `${row.id}: feature ${feature} is within its declared domain`);
    });
  }
  for (const row of data.background) closeTo(predict(row.values), row.modelPrediction, `${row.id}: background prediction`);
  closeTo(mean(data.background.map(row => predict(row.values))), 47, 'reference mean of illustrative model predictions');
});

test('every coalition is the mean prediction of correctly substituted background profiles', () => {
  for (const observation of data.observations) {
    const expected = reconstructed.get(observation.id);
    assert.equal(observation.coalitions.length, fullMask + 1, observation.id);
    assert.deepEqual(observation.coalitions.map(coalition => coalition.mask).sort((a, b) => a - b),
      Array.from({ length: fullMask + 1 }, (_, mask) => mask));
    for (const coalition of observation.coalitions) {
      assert.equal(coalition.predictions.length, data.background.length,
        `${observation.id}, mask ${coalition.mask}: all background profiles are evaluated`);
      coalition.predictions.forEach((prediction, row) => {
        closeTo(prediction, expected[coalition.mask].predictions[row],
          `${observation.id}, mask ${coalition.mask}, background profile ${row}`);
      });
      closeTo(coalition.value, expected[coalition.mask].value,
        `${observation.id}, mask ${coalition.mask}: mean model prediction`);
    }
    closeTo(observation.baseValue, 47, `${observation.id}: common reference baseline`);
    closeTo(observation.baseValue, expected[0].value, `${observation.id}: empty coalition`);
    closeTo(observation.prediction, predict(observation.values), `${observation.id}: focal prediction`);
    closeTo(expected[fullMask].value, observation.prediction, `${observation.id}: all features fixed`);
    for (const prediction of expected[fullMask].predictions) {
      closeTo(prediction, observation.prediction, `${observation.id}: every fully fixed profile equals the focal observation`);
    }
  }
});

test('independent subset factorial weights recover SHAP values, closed forms, and additive predictions', () => {
  for (const observation of data.observations) {
    const coalitions = reconstructed.get(observation.id);
    assert.equal(observation.shapValues.length, featureCount, observation.id);
    const [ability, neighborhood, experience] = observation.values;
    const closedForms = [
      10 * (ability - .5) + 12 * (ability - .5) * neighborhood,
      12 * neighborhood * (ability + .5),
      4 * (experience - .5),
    ];
    const shares = data.features.map((_, feature) => {
      let share = 0;
      let totalWeight = 0;
      for (let mask = 0; mask <= fullMask; mask += 1) {
        if (mask & (1 << feature)) continue;
        const size = sizeOf(mask);
        const weight = factorial(size) * factorial(featureCount - size - 1) / factorial(featureCount);
        totalWeight += weight;
        share += weight * (coalitions[mask | (1 << feature)].value - coalitions[mask].value);
      }
      closeTo(totalWeight, 1, `${observation.id}, feature ${feature}: marginal weights sum to one`);
      closeTo(share, observation.shapValues[feature], `${observation.id}, feature ${feature}: subset-weighted SHAP value`);
      closeTo(share, closedForms[feature], `${observation.id}, feature ${feature}: analytic SHAP value`);
      return share;
    });
    closeTo(observation.baseValue + shares.reduce((sum, value) => sum + value, 0),
      observation.prediction, `${observation.id}: baseline plus SHAP contributions`);
  }
  for (const key of ['maxShapAgreementError', 'maxBaselineAgreementError', 'maxAdditivityError']) {
    assert.ok(Number.isFinite(data.validation[key]) && data.validation[key] >= 0
      && data.validation[key] <= tolerance, `${key}: recorded Python SHAP validation must agree within tolerance`);
  }
});

test('all six unique reveal orders contain valid joining contexts and telescope to the prediction', () => {
  const expectedOrders = new Set(['0,1,2', '0,2,1', '1,0,2', '1,2,0', '2,0,1', '2,1,0']);
  for (const observation of data.observations) {
    const coalitions = reconstructed.get(observation.id);
    assert.equal(observation.orders.length, factorial(featureCount), observation.id);
    assert.deepEqual(new Set(observation.orders.map(order => order.features.join(','))), expectedOrders);
    const byFeature = Array.from({ length: featureCount }, () => []);
    for (const order of observation.orders) {
      assert.equal(order.marginals.length, featureCount, observation.id);
      assert.equal(order.steps.length, featureCount, observation.id);
      let beforeMask = 0;
      order.features.forEach((feature, position) => {
        const afterMask = beforeMask | (1 << feature);
        assert.equal(beforeMask & (1 << feature), 0, 'A joining feature must not already be revealed');
        const marginal = coalitions[afterMask].value - coalitions[beforeMask].value;
        closeTo(order.marginals[position], marginal, `${observation.id}, order ${order.features}: marginal ${position}`);
        assert.equal(order.steps[position].feature, feature);
        assert.equal(order.steps[position].beforeMask, beforeMask);
        assert.equal(order.steps[position].afterMask, afterMask);
        closeTo(order.steps[position].value, marginal, `${observation.id}: stored joining step`);
        byFeature[feature].push(marginal);
        beforeMask = afterMask;
      });
      assert.equal(beforeMask, fullMask);
      closeTo(order.marginals.reduce((sum, value) => sum + value, 0),
        observation.prediction - observation.baseValue, `${observation.id}: order path total`);
    }
    byFeature.forEach((marginals, feature) => {
      closeTo(mean(marginals), observation.shapValues[feature], `${observation.id}, feature ${feature}: six-order mean`);
    });
  }
});

test('the adverse-neighborhood example reverses ability’s marginal sign while preserving complementarity', () => {
  const observation = data.observations.find(row => row.id === data.defaultObservationId);
  assert.deepEqual(observation.values, [1, -1, 1]);
  const { feature, beforeMask, afterMask } = data.defaultComparison;
  const coalitions = reconstructed.get(observation.id);
  assert.equal(feature, 0);
  assert.equal(beforeMask, 2);
  assert.equal(beforeMask & (1 << feature), 0);
  assert.equal(afterMask, beforeMask | (1 << feature));
  closeTo(observation.prediction, 30, 'default prediction');
  const beforeNeighborhood = coalitions[1].value - coalitions[0].value;
  const afterNeighborhood = coalitions[3].value - coalitions[2].value;
  closeTo(beforeNeighborhood, 5, 'ability marginal with neighborhood integrated out');
  closeTo(afterNeighborhood, -7, 'ability marginal after adverse neighborhood is fixed');
  assert.ok(beforeNeighborhood > 0 && afterNeighborhood < 0, 'The context must reverse the marginal contribution’s sign');
  observation.shapValues.forEach((value, featureIndex) => {
    closeTo(value, [-1, -18, 2][featureIndex], `default SHAP contribution ${featureIndex}`);
  });
  assert.ok(Math.abs(afterNeighborhood - observation.shapValues[feature]) > 1,
    'A single adverse-context marginal must differ from the final SHAP average');
  for (const mask of [0, 1, 2, 3]) {
    closeTo(coalitions[mask | 4].value - coalitions[mask].value, 2,
      'Experience contributes two relative to background-average experience in every context');
  }
  const abilityGainAdverse = predict([1, -1, 0]) - predict([0, -1, 0]);
  const abilityGainFavorable = predict([1, 1, 0]) - predict([0, 1, 0]);
  assert.ok(abilityGainFavorable > abilityGainAdverse,
    'Ability and neighborhood remain complementary despite the negative marginal in the adverse context');
});
