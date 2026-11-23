import { DETAIL_VALUES, traceWholeModel } from "./attention-detail.ts";

export type BridgeReadout = "mean" | "position";

/**
 * A fully specified readout illustration using the existing fixed attention.
 *
 * Changing a value leaves Q, K, and the attention matrices unchanged. The
 * position readout demonstrates flattening followed by a position-dependent
 * linear classifier; the mean readout reproduces the original pooled toy.
 * Neither readout is asserted to reconstruct LiDDA's unpublished classifier,
 * and changing an internal value is not an intervention on an actual ad.
 */
export function traceReadoutBridge(
  negativeVideo = false,
  readout: BridgeReadout = "mean",
) {
  if (readout !== "mean" && readout !== "position") {
    throw new RangeError("Choose the mean or position readout.");
  }

  const whole = traceWholeModel();
  const values = DETAIL_VALUES.map((value, index) =>
    negativeVideo && index === 2 ? -value : value,
  );
  const headOutputs = whole.traces.map((rows) =>
    rows.map((row) => row.weights.reduce(
      (total, weight, index) => total + weight * values[index], 0,
    )),
  );
  const averagedOutputs = values.map((_, query) =>
    headOutputs.reduce((total, outputs) => total + outputs[query], 0)
      / headOutputs.length,
  );

  // With one scalar per query, row-major flattening preserves the four entries.
  // With d_v coordinates per query, this vector would have N * d_v entries.
  const features = [...averagedOutputs];
  const coefficients = readout === "mean"
    ? [0.25, 0.25, 0.25, 0.25]
    : [0.1, 0.2, 0.3, 0.4];
  const weightedFeatures = features.map((value, index) =>
    value * coefficients[index],
  );
  const bias = -2.4;
  const logit = bias + weightedFeatures.reduce((total, value) => total + value, 0);
  const probability = 1 / (1 + Math.exp(-logit));

  return {
    values,
    headOutputs,
    averagedOutputs,
    features,
    coefficients,
    weightedFeatures,
    incoming: [...whole.incoming],
    credit: [...whole.credit],
    logit,
    probability,
    bias,
    readout,
  };
}
