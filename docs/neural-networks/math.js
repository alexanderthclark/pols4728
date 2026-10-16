/* Exact, dependency-free mathematical models for the ReLU explainer. */
(function (root, factory) {
  "use strict";
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.NNMath = api;
})(typeof window !== "undefined" ? window : globalThis, function () {
  "use strict";

  const TAU = 2 * Math.PI;
  const DOMAIN = Object.freeze([0, TAU]);
  const EPS = 1e-10;
  const relu = (x) => Math.max(0, x);

  function finite(value, name) {
    if (!Number.isFinite(value)) throw new TypeError(name + " must be finite.");
    return value;
  }

  /** Make a normalized hinge network: f(x) = c + sum_j a_j ReLU(x-t_j). */
  function createShallow(c, neurons) {
    finite(c, "Output bias");
    if (!Array.isArray(neurons)) throw new TypeError("Neurons must be an array.");
    const units = neurons.map(function (unit, index) {
      const t = finite(unit.t, "Neuron threshold");
      const a = finite(unit.a, "Neuron output weight");
      return { index: index, t: t, a: a, w: 1, b: -t };
    });
    return { c: c, bias: c, neurons: units, width: units.length };
  }

  /** Inspect every hidden activation and its signed contribution to the output. */
  function contributions(model, x) {
    finite(x, "Input");
    return model.neurons.map(function (unit, index) {
      const preactivation = x - unit.t;
      const activation = relu(preactivation);
      const value = unit.a * activation;
      return {
        index: index,
        t: unit.t,
        a: unit.a,
        preactivation: preactivation,
        activation: activation,
        active: preactivation > 0,
        value: value,
        weighted: value,
      };
    });
  }

  function evaluate(model, x) {
    finite(x, "Input");
    return model.neurons.reduce(function (sum, unit) {
      return sum + unit.a * relu(x - unit.t);
    }, model.c);
  }

  /** Return uniformly spaced interpolation nodes, including both endpoints. */
  function initialKnots(interiorCount) {
    if (interiorCount === undefined) interiorCount = 5;
    if (!Number.isInteger(interiorCount) || interiorCount < 0) {
      throw new RangeError("The number of interior joints must be a nonnegative integer.");
    }
    const segments = interiorCount + 1;
    return Array.from({ length: segments + 1 }, function (_, k) {
      return TAU * k / segments;
    });
  }

  function normalizeKnots(knots) {
    if (!Array.isArray(knots)) throw new TypeError("Knots must be an array.");
    const points = [0, TAU];
    knots.forEach(function (x) {
      finite(x, "Knot position");
      if (x < -EPS || x > TAU + EPS) {
        throw new RangeError("Sine interpolation knots must lie in [0, 2π].");
      }
      points.push(Math.max(0, Math.min(TAU, x)));
    });
    points.sort(function (a, b) { return a - b; });
    return points.filter(function (x, index) {
      return index === 0 || x - points[index - 1] > EPS;
    });
  }

  /**
   * Express the exact linear interpolant to sin(x) using one hidden layer.
   * The first hidden neuron is ReLU(x+1), active across the entire domain.
   * Its output weight supplies the initial slope. Every interior knot adds
   * one neuron whose output weight is the change in adjacent segment slopes.
   * There is no input-to-output skip connection.
   */
  function modelFromKnots(knots) {
    const nodes = normalizeKnots(knots);
    const values = nodes.map(Math.sin);
    const slopes = nodes.slice(0, -1).map(function (x, i) {
      return (values[i + 1] - values[i]) / (nodes[i + 1] - x);
    });
    const units = [{ t: -1, a: slopes[0] }];
    nodes.slice(1, -1).forEach(function (x, i) {
      units.push({ t: x, a: slopes[i + 1] - slopes[i] });
    });
    const model = createShallow(values[0] - slopes[0] * (nodes[0] + 1), units);
    model.neurons[0].baseline = true;
    model.knots = nodes;
    model.values = values;
    model.slopes = slopes;
    model.domain = DOMAIN.slice();
    model.interiorJoints = nodes.length - 2;
    model.segments = nodes.length - 1;
    return model;
  }

  /** Insert one new interpolation node; existing nodes and input are retained. */
  function refineAt(knots, x) {
    finite(x, "New joint position");
    if (x < 0 || x > TAU) throw new RangeError("The new joint must lie in [0, 2π].");
    return normalizeKnots(knots.concat([x]));
  }

  /** Dense-grid approximation error; error is absolute and signedError is f-sin. */
  function maxError(knots, samples) {
    if (samples === undefined) samples = 2000;
    if (!Number.isInteger(samples) || samples < 2) {
      throw new RangeError("At least two error samples are required.");
    }
    const model = modelFromKnots(knots);
    let result = { x: 0, error: 0, maxError: 0, signedError: 0, rms: 0, samples: samples };
    let squares = 0;
    for (let i = 0; i < samples; i += 1) {
      const x = TAU * i / (samples - 1);
      const signedError = evaluate(model, x) - Math.sin(x);
      const error = Math.abs(signedError);
      squares += signedError * signedError;
      if (error > result.error) {
        result.x = x;
        result.error = error;
        result.maxError = error;
        result.signedError = signedError;
      }
    }
    result.rms = Math.sqrt(squares / samples);
    return result;
  }

  /** Tent map on [0,1], implemented by two ReLU units and a linear output. */
  function tent(x) {
    return 2 * relu(x) - 4 * relu(x - 0.5);
  }

  function checkDepth(depth) {
    if (!Number.isInteger(depth) || depth < 1 || depth > 20) {
      throw new RangeError("Depth must be an integer from 1 through 20.");
    }
  }

  /** Evaluate d compositions of the tent map, with two neurons in each layer. */
  function deepValue(x, depth) {
    finite(x, "Input");
    checkDepth(depth);
    let value = x;
    for (let i = 0; i < depth; i += 1) value = tent(value);
    return value;
  }

  /**
   * Each layer uses preactivations [u, u-1/2] and activations ReLU thereof.
   * The next layer receives u=2h_1-4h_2; the final output uses the same sum.
   * Hidden-to-hidden weights are [[2,-4],[2,-4]], biases [0,-1/2].
   */
  function deepActivations(x, depth) {
    finite(x, "Input");
    checkDepth(depth);
    const layers = [];
    let input = x;
    for (let i = 0; i < depth; i += 1) {
      const preactivations = [input, input - 0.5];
      const activations = preactivations.map(relu);
      const output = 2 * activations[0] - 4 * activations[1];
      layers.push({
        depth: i + 1,
        input: input,
        preactivations: preactivations,
        activations: activations,
        output: output,
      });
      input = output;
    }
    return layers;
  }

  /** Exact locations of the 2^d-1 interior joints of T composed d times. */
  function deepJoints(depth) {
    checkDepth(depth);
    const segments = Math.pow(2, depth);
    return Array.from({ length: segments - 1 }, function (_, i) {
      return (i + 1) / segments;
    });
  }

  return Object.freeze({
    TAU: TAU,
    DOMAIN: DOMAIN,
    relu: relu,
    createShallow: createShallow,
    evaluate: evaluate,
    contributions: contributions,
    initialKnots: initialKnots,
    modelFromKnots: modelFromKnots,
    refineAt: refineAt,
    maxError: maxError,
    tent: tent,
    deepValue: deepValue,
    deepActivations: deepActivations,
    deepJoints: deepJoints,
  });
});
