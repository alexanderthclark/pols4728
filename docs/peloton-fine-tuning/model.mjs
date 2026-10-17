// A fully specified teaching calculation. These are not DistilBERT weights.
export const illustrativeClasses = ['Coaching', 'Music', 'Encouragement'];
export const probeSentences = [
  "Let's increase that resistance to zone five.",
  'This song reminds me of college.',
  'Great job everyone!',
];
export const trainingSentence = 'Keep your shoulders relaxed.';
export const initialParameters = Object.freeze({
  encoder: [[0.7, 0.1], [-0.2, 0.6]],
  head: [[0.4, 0.1], [-0.2, 0.3], [0.1, -0.3]],
  bias: [0, 0, 0],
});
export const featureVector = [1, 0.5];

export function softmax(logits) {
  const offset = Math.max(...logits);
  const values = logits.map(value => Math.exp(value - offset));
  const total = values.reduce((sum, value) => sum + value, 0);
  return values.map(value => value / total);
}

export function forward(parameters = initialParameters, input = featureVector) {
  const hidden = parameters.encoder.map(row => Math.tanh(row.reduce((sum, value, j) => sum + value * input[j], 0)));
  const logits = parameters.head.map((row, k) => row.reduce((sum, value, j) => sum + value * hidden[j], parameters.bias[k]));
  return { hidden, logits, probabilities: softmax(logits) };
}

export function crossEntropy(probabilities, target = 0) {
  return -Math.log(probabilities[target]);
}

export function gradients(parameters = initialParameters, input = featureVector, target = 0) {
  const state = forward(parameters, input);
  const dz = state.probabilities.map((value, k) => value - Number(k === target));
  const head = dz.map(value => state.hidden.map(hidden => value * hidden));
  const dh = state.hidden.map((_, j) => parameters.head.reduce((sum, row, k) => sum + row[j] * dz[k], 0));
  const da = dh.map((value, j) => value * (1 - state.hidden[j] ** 2));
  const encoder = da.map(value => input.map(x => value * x));
  return { encoder, head, bias: dz };
}

export function update(parameters = initialParameters, rate = 0.8, { freezeEncoder = false } = {}) {
  const gradient = gradients(parameters);
  return {
    encoder: parameters.encoder.map((row, i) => row.map((value, j) => value - (freezeEncoder ? 0 : rate * gradient.encoder[i][j]))),
    head: parameters.head.map((row, i) => row.map((value, j) => value - rate * gradient.head[i][j])),
    bias: parameters.bias.map((value, i) => value - rate * gradient.bias[i]),
  };
}

export function interpolateParameters(before, after, fraction) {
  const t = Math.max(0, Math.min(1, fraction));
  return {
    encoder: before.encoder.map((row, i) => row.map((value, j) => value + t * (after.encoder[i][j] - value))),
    head: before.head.map((row, i) => row.map((value, j) => value + t * (after.head[i][j] - value))),
    bias: before.bias.map((value, i) => value + t * (after.bias[i] - value)),
  };
}

// Thirty invented observations, ten per illustrative class. Eight in each row
// belong to training and two to validation: exact 80/20 stratification.
export const observations = Array.from({ length: 30 }, (_, id) => ({
  id,
  category: Math.floor(id / 10),
  validation: id % 10 >= 8,
}));

export const completedParameters = update();
