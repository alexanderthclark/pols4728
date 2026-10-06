// The website displays a finite interventional game computed by the Python script.
// These helpers keep the feature masks and rendered hybrid inputs consistent.
export function mean(values) {
  if (!values.length || values.some(value => !Number.isFinite(value))) throw new TypeError('An average needs finite values.');
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export function hybridValues(backgroundValues, observationValues, mask) {
  if (backgroundValues.length !== observationValues.length) throw new RangeError('Feature rows must have equal lengths.');
  return backgroundValues.map((value, index) => mask & (1 << index) ? observationValues[index] : value);
}

export function coalition(observation, mask) {
  const result = observation.coalitions.find(group => group.mask === mask);
  if (!result) throw new RangeError(`No coalition ${mask}.`);
  return result;
}

export function marginal(observation, feature, beforeMask) {
  if (beforeMask & (1 << feature)) throw new RangeError('The joining feature cannot already be revealed.');
  return coalition(observation, beforeMask | (1 << feature)).value - coalition(observation, beforeMask).value;
}

export function precedingMasks(feature, featureCount) {
  return Array.from({ length: 1 << featureCount }, (_, mask) => mask).filter(mask => !(mask & (1 << feature)));
}

export function validateData(data) {
  if (data.features?.length !== 3 || data.background?.length !== 8 || !data.observations?.length) throw new Error('This story expects three features and eight background rows.');
  for (const observation of data.observations) {
    if (observation.coalitions.length !== 8 || observation.orders.length !== 6) throw new Error('Missing coalition or order calculations.');
    for (const group of observation.coalitions) {
      if (group.predictions.length !== data.background.length || Math.abs(mean(group.predictions) - group.value) > 1e-8) throw new Error('A coalition average does not match its row predictions.');
    }
    const sum = observation.baseValue + observation.shapValues.reduce((total, value) => total + value, 0);
    if (Math.abs(sum - observation.prediction) > 1e-8) throw new Error('SHAP contributions do not reconstruct the prediction.');
  }
  return data;
}
