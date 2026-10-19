import { hybridValues, mean } from './calculation.mjs';

// The people summarized below are a separate, equally weighted explanatory
// cohort. Every explanation still uses the original eight reference profiles.
const SCORE_LEVELS = [0, .25, .5, .75, 1];
const NEIGHBORHOOD_LEVELS = [-1, -.5, 0, .5, 1];
const FEATURE_ORDERS = [
  [0, 1, 2], [0, 2, 1], [1, 0, 2],
  [1, 2, 0], [2, 0, 1], [2, 1, 0],
];

function predict(values, model) {
  const [ability, neighborhood, experience] = values;
  const coefficients = model.coefficients;
  return model.intercept + coefficients.ability * ability
    + coefficients.abilityNeighborhood * ability * neighborhood
    + coefficients.experience * experience;
}

function validateGame(data, model) {
  if (model?.type !== 'IllustrativeEarningsInteraction'
    || data.features?.length !== 3 || data.background?.length !== 8
    || model.inputFeatures?.some((id, feature) => id !== data.features[feature].id)
    || model.inputFeatures?.length !== 3
    || !Number.isFinite(model.intercept)
    || !['ability', 'abilityNeighborhood', 'experience']
      .every(key => Number.isFinite(model.coefficients?.[key]))) {
    throw new TypeError('Use the existing three-feature earnings model and eight reference profiles.');
  }
  for (const row of data.background) {
    if (row.values?.length !== 3 || row.values.some(value => !Number.isFinite(value))
      || row.weight !== 1 / data.background.length) {
      throw new TypeError('Every reference profile must contain three finite values with equal weight.');
    }
  }
}

/** Explain one profile with the fixed model and original finite background. */
export function explainValues(values, data, model) {
  validateGame(data, model);
  if (values?.length !== data.features.length || values.some((value, feature) => {
    const [lower, upper] = data.features[feature].domain;
    return !Number.isFinite(value) || value < lower || value > upper;
  })) throw new RangeError('Profile values must be within the three declared feature domains.');

  const stored = data.observations.find(observation =>
    observation.values.every((value, feature) => value === values[feature]));
  const id = stored?.id ?? `profile-${values.join('_')}`;
  const name = stored?.name ?? `Profile (${values.join(', ')})`;
  const coalitions = Array.from({ length: 1 << values.length }, (_, mask) => {
    const predictions = data.background.map(background =>
      predict(hybridValues(background.values, values, mask), model));
    return { mask, value: mean(predictions), predictions };
  });
  const orders = FEATURE_ORDERS.map(features => {
    let beforeMask = 0;
    const steps = features.map(feature => {
      const afterMask = beforeMask | (1 << feature);
      const step = {
        feature, beforeMask, afterMask,
        value: coalitions[afterMask].value - coalitions[beforeMask].value,
      };
      beforeMask = afterMask;
      return step;
    });
    return { features: [...features], marginals: steps.map(step => step.value), steps };
  });
  // Averaging the six exact revealing orders is equivalent to the subset
  // factorial weights taught earlier in the story.
  const shapValues = values.map((_, feature) => mean(orders.map(order =>
    order.steps.find(step => step.feature === feature).value)));
  return {
    id, name, label: name, values: [...values],
    prediction: predict(values, model), baseValue: coalitions[0].value,
    coalitions, orders, shapValues,
  };
}

/** All 5 × 5 × 5 input combinations; the selected example appears first. */
export function buildPopulation(data, model) {
  validateGame(data, model);
  const population = [];
  for (const ability of SCORE_LEVELS) {
    for (const neighborhood of NEIGHBORHOOD_LEVELS) {
      for (const experience of SCORE_LEVELS) {
        population.push(explainValues([ability, neighborhood, experience], data, model));
      }
    }
  }
  const defaultIndex = population.findIndex(observation => observation.id === data.defaultObservationId);
  if (defaultIndex > 0) population.unshift(...population.splice(defaultIndex, 1));
  return population;
}

/** Mean absolute contribution, in feature order and the prediction's units. */
export function globalImportance(observations) {
  const featureCount = observations[0]?.shapValues?.length;
  if (!featureCount || observations.some(observation =>
    observation.shapValues?.length !== featureCount
    || observation.shapValues.some(value => !Number.isFinite(value)))) {
    throw new TypeError('Global importance needs nonempty, equally sized finite SHAP vectors.');
  }
  return Array.from({ length: featureCount }, (_, feature) =>
    mean(observations.map(observation => Math.abs(observation.shapValues[feature]))));
}
