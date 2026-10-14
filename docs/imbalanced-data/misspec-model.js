(function (root, factory) {
  "use strict";
  var model = factory();
  if (typeof module === "object" && module.exports) module.exports = model;
  if (root) root.MisspecModel = model;
})(typeof window !== "undefined" ? window : null, function () {
  "use strict";

  // An exact discrete population, not a simulated finite sample. The curved
  // log odds increase at every x, and positives comprise about 2.56% overall.
  var DEFAULT_SPEC = Object.freeze({
    x: Object.freeze([0, 1, 2, 3, 4]),
    masses: Object.freeze([0.68, 0.20, 0.08, 0.03, 0.01]),
    intercept: -6,
    linear: 0.3,
    quadratic: 0.5,
    s1: 1,
    s0: 0.1,
    retention: 0.1
  });

  function finite(value, label) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new TypeError(label + " must be a finite number.");
    }
    return value;
  }

  function positive(value, label) {
    finite(value, label);
    if (value <= 0) throw new RangeError(label + " must be positive.");
    return value;
  }

  function inclusion(value, label) {
    positive(value, label);
    if (value > 1) throw new RangeError(label + " must be at most one.");
    return value;
  }

  function sigmoid(eta) {
    if (eta >= 0) return 1 / (1 + Math.exp(-eta));
    var e = Math.exp(eta);
    return e / (1 + e);
  }

  function softplus(eta) {
    return Math.max(eta, 0) + Math.log1p(Math.exp(-Math.abs(eta)));
  }

  function trueLogit(x, spec) {
    spec = spec || DEFAULT_SPEC;
    return spec.intercept + spec.linear * x + spec.quadratic * x * x;
  }

  function logOdds(fitted, x) {
    return fitted.intercept + fitted.slope * x;
  }

  function predict(fitted, x) {
    return sigmoid(logOdds(fitted, x));
  }

  function resolveSpec(options) {
    var spec = Object.assign({}, DEFAULT_SPEC, options || {});
    if (options && options.retention !== undefined && options.s0 === undefined) spec.s0 = options.retention;
    // The walkthrough retains every positive; retention names the share of
    // negatives retained, while s1/s0 also support general class selection.
    spec.retention = spec.s0;
    if (!Array.isArray(spec.x) || !Array.isArray(spec.masses) || spec.x.length < 2 || spec.x.length !== spec.masses.length) {
      throw new TypeError("x and masses must be equally sized arrays with at least two levels.");
    }
    finite(spec.intercept, "True intercept");
    finite(spec.linear, "True linear coefficient");
    finite(spec.quadratic, "True quadratic coefficient");
    inclusion(spec.s1, "Positive inclusion probability");
    inclusion(spec.s0, "Negative inclusion probability");
    var total = spec.masses.reduce(function (sum, mass) { return sum + positive(mass, "Level mass"); }, 0);
    spec.x = spec.x.map(function (x) { return finite(x, "Predictor level"); });
    if (new Set(spec.x).size !== spec.x.length) throw new RangeError("Predictor levels must be distinct.");
    spec.masses = spec.masses.map(function (mass) { return mass / total; });
    return spec;
  }

  function rows(options) {
    var spec = resolveSpec(options);
    return spec.x.map(function (x, i) {
      var eta = trueLogit(x, spec);
      var p = sigmoid(eta);
      var mass = spec.masses[i];
      var positiveMass = mass * p;
      var negativeMass = mass * (1 - p);
      var samplePositiveMass = positiveMass * spec.s1;
      var sampleNegativeMass = negativeMass * spec.s0;
      var sampleMass = samplePositiveMass + sampleNegativeMass;
      return {
        x: x,
        mass: mass,
        p: p,
        trueLogit: eta,
        positiveMass: positiveMass,
        negativeMass: negativeMass,
        sampleMass: sampleMass,
        samplePositiveMass: samplePositiveMass,
        sampleNegativeMass: sampleNegativeMass,
        sampleP: samplePositiveMass / sampleMass,
        sampledLogit: eta + Math.log(spec.s1 / spec.s0)
      };
    });
  }

  // Each row stores population masses of positives and negatives at a known x.
  // Normalization makes the objective expected cross entropy in natural logs.
  function prepare(inputRows, options) {
    if (!Array.isArray(inputRows) || inputRows.length < 2) throw new TypeError("At least two population rows are required.");
    options = options || {};
    var positiveWeight = positive(options.positiveWeight === undefined ? 1 : options.positiveWeight, "Positive weight");
    var negativeWeight = positive(options.negativeWeight === undefined ? 1 : options.negativeWeight, "Negative weight");
    var total = 0;
    var posTotal = 0;
    var negTotal = 0;
    var result = inputRows.map(function (row) {
      var x = finite(row.x, "Predictor level");
      var pos = finite(row.positiveMass, "Positive mass") * positiveWeight;
      var neg = finite(row.negativeMass, "Negative mass") * negativeWeight;
      if (pos < 0 || neg < 0 || pos + neg <= 0) throw new RangeError("Class masses must be nonnegative with positive total mass at every level.");
      total += pos + neg;
      posTotal += pos;
      negTotal += neg;
      return { x: x, positiveMass: pos, negativeMass: neg };
    });
    if (posTotal <= 0 || negTotal <= 0) throw new RangeError("Both outcome classes must have positive mass.");
    if (new Set(result.map(function (row) { return row.x; })).size < 2) throw new RangeError("A slope requires at least two distinct predictor levels.");
    return result.map(function (row) {
      return { x: row.x, positiveMass: row.positiveMass / total, negativeMass: row.negativeMass / total };
    });
  }

  function objective(data, fitted) {
    var loss = 0;
    var g0 = 0;
    var g1 = 0;
    var h00 = 0;
    var h01 = 0;
    var h11 = 0;
    data.forEach(function (row) {
      var eta = logOdds(fitted, row.x);
      var p = sigmoid(eta);
      var total = row.positiveMass + row.negativeMass;
      // Separate stable positive and negative losses avoid subtracting large
      // nearly equal values when a predicted probability is close to one.
      loss += row.positiveMass * softplus(-eta) + row.negativeMass * softplus(eta);
      var residual = total * p - row.positiveMass;
      var curvature = total * p * (1 - p);
      g0 += residual;
      g1 += row.x * residual;
      h00 += curvature;
      h01 += row.x * curvature;
      h11 += row.x * row.x * curvature;
    });
    return { loss: loss, score: [g0, g1], hessian: [h00, h01, h11] };
  }

  function logLoss(inputRows, fitted, options) {
    return objective(prepare(inputRows, options), fitted).loss;
  }

  function score(inputRows, fitted, options) {
    return objective(prepare(inputRows, options), fitted).score;
  }

  function fit(inputRows, options) {
    options = options || {};
    var data = prepare(inputRows, options);
    var prior = data.reduce(function (sum, row) { return sum + row.positiveMass; }, 0);
    var fitted = options.initial ? {
      intercept: finite(options.initial.intercept, "Initial intercept"),
      slope: finite(options.initial.slope, "Initial slope")
    } : { intercept: Math.log(prior) - Math.log1p(-prior), slope: 0 };
    var maxIterations = 100;
    var tolerance = 1e-12;
    for (var iteration = 0; iteration <= maxIterations; iteration++) {
      var current = objective(data, fitted);
      if (Math.max(Math.abs(current.score[0]), Math.abs(current.score[1])) <= tolerance) {
        return {
          intercept: fitted.intercept,
          slope: fitted.slope,
          coefficients: [fitted.intercept, fitted.slope],
          loss: current.loss,
          score: current.score,
          iterations: iteration,
          converged: true
        };
      }
      var h = current.hessian;
      var determinant = h[0] * h[2] - h[1] * h[1];
      if (!Number.isFinite(determinant) || determinant <= 0) throw new RangeError("The logistic fit has singular curvature; check predictor variation and separation.");
      var delta0 = (h[2] * current.score[0] - h[1] * current.score[1]) / determinant;
      var delta1 = (h[0] * current.score[1] - h[1] * current.score[0]) / determinant;
      var descent = current.score[0] * delta0 + current.score[1] * delta1;
      var step = 1;
      var candidate;
      var accepted = false;
      // Backtracking ensures that each Newton update reduces cross entropy.
      for (var backtrack = 0; backtrack < 50; backtrack++) {
        candidate = { intercept: fitted.intercept - step * delta0, slope: fitted.slope - step * delta1 };
        var candidateLoss = objective(data, candidate).loss;
        var roundingAllowance = 4 * Number.EPSILON * Math.max(1, Math.abs(current.loss));
        if (Number.isFinite(candidateLoss) && candidateLoss <= current.loss - 0.0001 * step * descent + roundingAllowance) {
          accepted = true;
          break;
        }
        step /= 2;
      }
      if (!accepted) throw new RangeError("The logistic fit failed to find a decreasing Newton step.");
      fitted = candidate;
    }
    throw new RangeError("The logistic fit did not converge; a finite optimum may not exist.");
  }

  function example(options) {
    var spec = resolveSpec(options);
    var allRows = rows(spec);
    var naturalRows = allRows.map(function (row) {
      return { x: row.x, positiveMass: row.positiveMass, negativeMass: row.negativeMass };
    });
    var sampledRows = allRows.map(function (row) {
      return { x: row.x, positiveMass: row.samplePositiveMass, negativeMass: row.sampleNegativeMass };
    });
    var naturalFit = fit(naturalRows);
    var sampledFit = fit(sampledRows);
    var logOddsShift = Math.log(spec.s1 / spec.s0);
    var correctedFit = {
      intercept: sampledFit.intercept - logOddsShift,
      slope: sampledFit.slope,
      coefficients: [sampledFit.intercept - logOddsShift, sampledFit.slope]
    };
    correctedFit.loss = logLoss(naturalRows, correctedFit);
    correctedFit.score = score(naturalRows, correctedFit);
    var weightedFit = fit(sampledRows, { positiveWeight: 1 / spec.s1, negativeWeight: 1 / spec.s0 });
    var populationPrior = allRows.reduce(function (sum, row) { return sum + row.positiveMass; }, 0);
    var retainedShare = allRows.reduce(function (sum, row) { return sum + row.sampleMass; }, 0);
    var samplePos = allRows.reduce(function (sum, row) { return sum + row.samplePositiveMass; }, 0);
    var trueLogLoss = allRows.reduce(function (sum, row) {
      return sum + row.positiveMass * softplus(-row.trueLogit) + row.negativeMass * softplus(row.trueLogit);
    }, 0);
    return {
      spec: spec,
      rows: allRows,
      naturalRows: naturalRows,
      sampledRows: sampledRows,
      naturalFit: naturalFit,
      sampledFit: sampledFit,
      correctedFit: correctedFit,
      weightedFit: weightedFit,
      summary: {
        populationPrior: populationPrior,
        sampledPrior: samplePos / retainedShare,
        retainedShare: retainedShare,
        logOddsShift: logOddsShift,
        interceptCorrection: -logOddsShift,
        slopeChange: sampledFit.slope - naturalFit.slope,
        relativeSlopeChange: sampledFit.slope / naturalFit.slope - 1,
        naturalLogLoss: naturalFit.loss,
        correctedLogLoss: correctedFit.loss,
        sampledNaturalLogLoss: logLoss(naturalRows, sampledFit),
        trueLogLoss: trueLogLoss,
        excessCorrectedLogLoss: correctedFit.loss - naturalFit.loss
      }
    };
  }

  return Object.freeze({
    DEFAULT_SPEC: DEFAULT_SPEC,
    sigmoid: sigmoid,
    trueLogit: trueLogit,
    logOdds: logOdds,
    predict: predict,
    rows: rows,
    fit: fit,
    logLoss: logLoss,
    score: score,
    example: example
  });
});
