import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
import assert from 'node:assert/strict';
const model = require("../docs/imbalanced-data/misspec-model.js");

function close(actual, expected, label, tolerance = 1e-9) {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${label}: ${actual} vs ${expected}`);
}

function converged(fit, label) {
  assert.equal(fit.converged, true, `${label} convergence flag`);
  assert.ok(Math.max(...fit.score.map(Math.abs)) < 1e-11, `${label} population score equations`);
}

const example = model.example();
const { rows, naturalFit, sampledFit, correctedFit, weightedFit, summary } = example;
converged(naturalFit, "Population fit");
converged(sampledFit, "Sampled fit");
converged(weightedFit, "Inverse selection weighted fit");

// Verify the population independently using known level masses and true logit.
const masses = [0.68, 0.20, 0.08, 0.03, 0.01];
let independentPrior = 0;
rows.forEach((row, x) => {
  const eta = -6 + 0.3 * x + 0.5 * x * x;
  const p = 1 / (1 + Math.exp(-eta));
  independentPrior += masses[x] * p;
  close(row.trueLogit, eta, "Curved true log odds", 1e-14);
  close(row.p, p, "Known conditional population probability", 1e-14);
  close(row.mass, masses[x], "Known level mass", 1e-14);
  if (x > 0) assert.ok(row.trueLogit > rows[x - 1].trueLogit, "The true relationship is monotone");
  const sampledP = p / (p + 0.1 * (1 - p));
  close(row.sampleP, sampledP, "Exact outcome dependent sampling probability", 1e-14);
  // The sampling identity is exact even though the log odds curve is nonlinear.
  close(Math.log(row.sampleP / (1 - row.sampleP)) - eta, Math.log(10), "Pointwise sampling log odds shift", 1e-11);
});
close(summary.populationPrior, independentPrior, "Exact population prevalence", 1e-14);
assert.ok(summary.populationPrior < 0.03, "The positive class is rare in the population");
close(summary.sampledPrior, independentPrior / (independentPrior + 0.1 * (1 - independentPrior)), "Exact sampled prevalence", 1e-14);

// Independent reference values from solving the population score equations.
close(naturalFit.intercept, -7.2602586695650375, "Population intercept");
close(naturalFit.slope, 2.230060566267418, "Population slope");
close(sampledFit.intercept, -4.412987007339833, "Sampled intercept");
close(sampledFit.slope, 1.932862169678228, "Sampled slope");
assert.ok(naturalFit.slope - sampledFit.slope > 0.25, "Misspecification changes the fitted slope materially");
close(correctedFit.intercept, sampledFit.intercept - Math.log(10), "Known intercept correction");
close(correctedFit.slope, sampledFit.slope, "Intercept correction retains the sampled slope", 1e-14);
assert.ok(Math.abs(correctedFit.intercept - naturalFit.intercept) > 0.5, "The intercept correction does not recover the population fit");
assert.ok(Math.max(...correctedFit.score.map(Math.abs)) > 1e-4, "The corrected fit does not solve the population score equations");
assert.ok(summary.correctedLogLoss > summary.naturalLogLoss + 0.0008, "The corrected sampled fit has worse population log loss");
close(summary.naturalLogLoss, 0.05336109770409519, "Population optimal linear log loss");
close(summary.correctedLogLoss, 0.0542169882503761, "Corrected sampled population log loss");

// Directly recompute population cross entropy without using the model helper.
const independentLoss = rows.reduce((sum, row) => {
  const p = 1 / (1 + Math.exp(-(correctedFit.intercept + correctedFit.slope * row.x)));
  return sum - row.mass * (row.p * Math.log(p) + (1 - row.p) * Math.log(1 - p));
}, 0);
close(summary.correctedLogLoss, independentLoss, "Independent expected cross entropy", 1e-14);

// Inverse inclusion weighting restores the full population objective even
// under misspecification, and therefore recovers its best linear approximation.
close(weightedFit.intercept, naturalFit.intercept, "Weighted recovery of intercept");
close(weightedFit.slope, naturalFit.slope, "Weighted recovery of slope");
close(weightedFit.loss, naturalFit.loss, "Weighted recovery of objective", 1e-14);

// When the model is correct, outcome dependent sampling moves only the
// intercept. An equal inclusion rate for both classes also changes no fit.
for (const selection of [{ s1: 1, s0: 0.1 }, { s1: 0.4, s0: 0.08 }]) {
  const linear = model.example({ intercept: -5, linear: 1.1, quadratic: 0, ...selection });
  close(linear.naturalFit.intercept, -5, "Correctly specified population intercept");
  close(linear.naturalFit.slope, 1.1, "Correctly specified population slope");
  close(linear.sampledFit.intercept, -5 + Math.log(selection.s1 / selection.s0), "Correctly specified sampled intercept");
  close(linear.sampledFit.slope, 1.1, "Correctly specified sampled slope");
  close(linear.correctedFit.intercept, -5, "Correct model intercept recovery");
  close(linear.correctedFit.slope, 1.1, "Correct model slope recovery");
}
const equal = model.example({ s1: 0.3, s0: 0.3 });
close(equal.sampledFit.intercept, equal.naturalFit.intercept, "Equal sampling leaves intercept unchanged");
close(equal.sampledFit.slope, equal.naturalFit.slope, "Equal sampling leaves slope unchanged");

// The optimizer is not relying on its default initial estimate; the line
// search converges to the same convex optimum from a deliberately poor start.
const distantStart = model.fit(example.naturalRows, { initial: { intercept: 3, slope: -1 } });
converged(distantStart, "Backtracking fit from poor initial values");
close(distantStart.intercept, naturalFit.intercept, "Initial value independent intercept");
close(distantStart.slope, naturalFit.slope, "Initial value independent slope");
assert.throws(() => model.example({ s0: 0 }), RangeError);
assert.throws(() => model.fit([{ x: 0, positiveMass: 1, negativeMass: 1 }, { x: 0, positiveMass: 1, negativeMass: 1 }]), RangeError);

console.log("All exact population misspecification checks passed.");
