import assert from "node:assert/strict";
import test from "node:test";
import { traceReadoutBridge } from "../lib/attention-bridge.ts";
import { DETAIL_VALUES, traceWholeModel } from "../lib/attention-detail.ts";

const close = (actual: number, expected: number) =>
  assert.ok(Math.abs(actual - expected) < 1e-12, `${actual} differs from ${expected}`);

test("the default bridge reproduces the original toy's representations and prediction", () => {
  const bridge = traceReadoutBridge();
  const original = traceWholeModel();
  assert.deepEqual(bridge.values, DETAIL_VALUES);
  assert.deepEqual(bridge.headOutputs, original.headOutputs);
  assert.deepEqual(bridge.averagedOutputs, original.averagedOutputs);
  assert.deepEqual(bridge.features, original.averagedOutputs);
  assert.deepEqual(bridge.credit, original.credit);
  close(bridge.logit, original.logit);
  close(bridge.probability, original.probability);
});

test("changing the video value changes the prediction while preserving attention credit", () => {
  for (const readout of ["mean", "position"] as const) {
    const positive = traceReadoutBridge(false, readout);
    const negative = traceReadoutBridge(true, readout);
    assert.equal(negative.values[2], -positive.values[2]);
    for (const index of [0, 1, 3]) {
      assert.equal(negative.values[index], positive.values[index]);
    }
    assert.deepEqual(negative.incoming, positive.incoming);
    assert.deepEqual(negative.credit, positive.credit);
    assert.ok(negative.probability < positive.probability);
    positive.features.forEach((value, index) => {
      assert.ok(negative.features[index] < value);
    });
  }
});

test("two heads and four query rows provide eight units of incoming attention", () => {
  const original = traceWholeModel();
  for (const readout of ["mean", "position"] as const) {
    const bridge = traceReadoutBridge(false, readout);
    close(bridge.incoming.reduce((total, value) => total + value, 0), 8);
    close(bridge.credit.reduce((total, value) => total + value, 0), 1);
    bridge.credit.forEach((share, source) => {
      assert.ok(share > 0 && share < 1);
      const incoming = original.traces.reduce((headSum, rows) =>
        headSum + rows.reduce((querySum, row) => querySum + row.weights[source], 0), 0,
      );
      close(bridge.incoming[source], incoming);
      close(share, incoming / 8);
    });
  }
});

test("flattened features retain query order and both readouts use the stated coefficients", () => {
  const mean = traceReadoutBridge(false, "mean");
  const position = traceReadoutBridge(false, "position");
  assert.deepEqual(mean.coefficients, [0.25, 0.25, 0.25, 0.25]);
  assert.deepEqual(position.coefficients, [0.1, 0.2, 0.3, 0.4]);
  assert.deepEqual(mean.features, position.features);
  assert.deepEqual(mean.credit, position.credit);
  assert.ok(Math.abs(mean.probability - position.probability) > 1e-4);
  for (const result of [mean, position]) {
    assert.equal(result.features.length, 4);
    assert.equal(result.coefficients.length, result.features.length);
    assert.equal(result.weightedFeatures.length, result.features.length);
    result.features.forEach((feature, query) => {
      close(feature, (result.headOutputs[0][query] + result.headOutputs[1][query]) / 2);
      close(result.weightedFeatures[query], feature * result.coefficients[query]);
    });
    close(result.logit, result.bias + result.weightedFeatures.reduce((sum, value) => sum + value, 0));
    close(result.probability, 1 / (1 + Math.exp(-result.logit)));
  }
});
