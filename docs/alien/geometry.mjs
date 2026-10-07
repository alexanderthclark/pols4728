// Pure geometry shared by the crowd, prediction stacks, and error decomposition.
// Each point keeps its original index as it moves between diagrams.

const finite = (value, name) => {
  if (!Number.isFinite(value)) throw new TypeError(`${name} must be finite.`);
};
const positive = (value, name) => {
  finite(value, name);
  if (value <= 0) throw new RangeError(`${name} must be positive.`);
};
const integer = (value, name, minimum = 0) => {
  if (!Number.isSafeInteger(value) || value < minimum) {
    throw new RangeError(`${name} must be a safe integer of at least ${minimum}.`);
  }
};

export function predictionLayout(predictions, {
  left = 180, right = 620, baseline = 380, columns = 20, gap = 5,
} = {}) {
  if (!Array.isArray(predictions)) throw new TypeError('Predictions must be an array.');
  finite(left, 'left');
  finite(right, 'right');
  finite(baseline, 'baseline');
  integer(columns, 'columns', 1);
  positive(gap, 'gap');
  if (right <= left) throw new RangeError('right must be greater than left.');
  const totals = [0, 0];
  for (const prediction of predictions) {
    if (prediction !== 0 && prediction !== 1) throw new RangeError('Predictions must be 0 or 1.');
    totals[prediction]++;
  }
  const ranks = [0, 0];
  return predictions.map((prediction, index) => {
    const rank = ranks[prediction]++;
    const row = Math.floor(rank / columns);
    const column = rank % columns;
    // Center the last partial row too: a group's center always stays on 0 or 1.
    const rowLength = Math.min(columns, totals[prediction] - row * columns);
    const anchor = prediction === 0 ? left : right;
    return {
      index, prediction,
      x: anchor + (column - (rowLength - 1) / 2) * gap,
      y: baseline - (row + 1) * gap,
    };
  });
}

export function crowdLayout(count, {
  left = 145, top = 110, columns = 40, gap = 12,
} = {}) {
  integer(count, 'count');
  finite(left, 'left');
  finite(top, 'top');
  integer(columns, 'columns', 1);
  positive(gap, 'gap');
  return Array.from({length: count}, (_, index) => ({
    index,
    x: left + (index % columns) * gap,
    y: top + Math.floor(index / columns) * gap,
  }));
}

export function decompositionSegments(stats, {x = 90, width = 640, max = 1} = {}) {
  if (!stats || typeof stats !== 'object') throw new TypeError('Statistics must be an object.');
  finite(x, 'x');
  positive(width, 'width');
  positive(max, 'max');
  const values = {biasSquared: stats.biasSquared, variance: stats.variance, noise: stats.noise ?? 0};
  for (const [key, value] of Object.entries(values)) {
    finite(value, key);
    if (value < 0) throw new RangeError(`${key} must be nonnegative.`);
  }
  finite(stats.mse, 'mse');
  if (stats.mse < 0 || stats.mse > max) throw new RangeError('mse must be between zero and max.');
  const sum = values.biasSquared + values.variance + values.noise;
  const tolerance = 1e-12 * Math.max(1, sum, stats.mse);
  if (Math.abs(sum - stats.mse) > tolerance) {
    throw new RangeError('Squared bias, variance, and noise must sum to mse.');
  }
  let start = x;
  const segments = Object.entries(values).map(([key, value]) => {
    const segmentWidth = value / max * width;
    const segment = {key, value, x: start, width: segmentWidth};
    start += segmentWidth;
    return segment;
  });
  return {segments, totalWidth: stats.mse / max * width};
}
