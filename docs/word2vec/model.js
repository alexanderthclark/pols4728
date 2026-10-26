// Course convention: W_in has one input embedding per COLUMN;
// W_out has one output embedding per ROW. This six-word example illustrates
// the mechanism of skip-gram; it is not a learned semantic word map.
export const vocabulary = [
  'voters', 'support', 'reform', 'senate', 'backs', 'proposal',
];

export const initial = {
  Win: [
    [0.2, 0.6, -0.3, 0.4, 0.1, -0.2],
    [-0.1, 0.4, 0.5, -0.4, 0.2, 0.3],
  ],
  Wout: [
    [0.5, 0.1],
    [-0.2, 0.3],
    [0.4, 0.6],
    [-0.3, 0.2],
    [0.2, -0.4],
    [0.1, 0.5],
  ],
  b: [0, 0, 0, 0, 0, 0],
};

export function forward(model, center = 1) {
  const h = model.Win.map(row => row[center]);
  // This minimal model omits biases; b is a fixed-zero compatibility field.
  const z = model.Wout.map(row => (
    row.reduce((sum, weight, d) => sum + weight * h[d], 0)
  ));
  // Subtracting a common maximum leaves softmax unchanged and prevents overflow.
  const maximum = Math.max(...z);
  const exponentials = z.map(value => Math.exp(value - maximum));
  const normalizer = exponentials.reduce((sum, value) => sum + value, 0);
  const p = exponentials.map(value => value / normalizer);
  return { h, z, p };
}

export function step(model, center = 1, target = 2, eta = 0.3) {
  const { h, p } = forward(model, center);
  // For ell = -log p(target | center), d ell / d z_j = p_j - 1[j=target].
  const error = p.map((probability, j) => probability - Number(j === target));
  // Compute every gradient using the OLD parameters before constructing updates.
  const hiddenGradient = h.map((_, d) => (
    model.Wout.reduce((sum, row, j) => sum + row[d] * error[j], 0)
  ));
  return {
    Win: model.Win.map((row, d) => row.map((weight, c) => (
      c === center ? weight - eta * hiddenGradient[d] : weight
    ))),
    Wout: model.Wout.map((row, j) => row.map((weight, d) => (
      weight - eta * error[j] * h[d]
    ))),
    b: model.b.slice(),
  };
}

export function interpolateModel(a, b, t) {
  const blend = (left, right) => left + t * (right - left);
  return {
    Win: a.Win.map((row, d) => row.map((value, c) => blend(value, b.Win[d][c]))),
    Wout: a.Wout.map((row, j) => row.map((value, d) => blend(value, b.Wout[j][d]))),
    b: a.b.map((value, j) => blend(value, b.b[j])),
  };
}

// Repeated updates on the SAME pair make the update's direction easy to see.
// This is a mechanism demonstration, not training on a complete corpus.
export const trainingPath = [interpolateModel(initial, initial, 0)];
for (let update = 0; update < 16; update += 1) {
  trainingPath.push(step(trainingPath[trainingPath.length - 1]));
}
