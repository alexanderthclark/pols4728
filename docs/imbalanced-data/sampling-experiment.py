"""Finite-sample logistic regression and independent negative-retention draws.

Internal artifact-generation code. It does not prescribe software or a random
seed to students. Run once: the representative sample is the first draw, and
there is no search over draws or seeds.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
import statsmodels
import statsmodels.api as sm


OUT = Path(__file__).resolve().parent
N = 200_000
REPEATS = 500
RETENTION = 0.1
INTERNAL_SEED = 982451653  # Fixed before any fitted results were inspected.
RNG = np.random.default_rng(INTERNAL_SEED)
SPECS = {
    "correct": {
        "label": "Correctly specified linear log odds",
        "x": [0, 1], "masses": [0.99, 0.01],
        "intercept": -float(np.log(99)), "linear": float(np.log(99)),
        "quadratic": 0.0,
        "formula": "log(99) × (x − 1)",
    },
    "curved": {
        "label": "Curved true log odds fitted with a straight line",
        "x": [0, 1, 2, 3, 4], "masses": [0.68, 0.20, 0.08, 0.03, 0.01],
        "intercept": -6.0, "linear": 0.3, "quadratic": 0.5,
        "formula": "−6 + 0.3x + 0.5x²",
    },
}


def sigmoid(eta: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0, -eta))


def counts_for(x: np.ndarray, y: np.ndarray, levels: int) -> list[dict]:
    positives = np.bincount(x[y == 1], minlength=levels)
    negatives = np.bincount(x[y == 0], minlength=levels)
    return [
        {"x": k, "positive": int(positives[k]), "negative": int(negatives[k]),
         "total": int(positives[k] + negatives[k]),
         "p": float(positives[k] / (positives[k] + negatives[k]))}
        for k in range(levels)
    ]


def score_from_counts(counts: list[dict], intercept: float, slope: float) -> list[float]:
    xs = np.array([row["x"] for row in counts], dtype=float)
    pos = np.array([row["positive"] for row in counts], dtype=float)
    total = np.array([row["total"] for row in counts], dtype=float)
    residual = total * sigmoid(intercept + slope * xs) - pos
    return [float(residual.sum() / total.sum()), float((xs * residual).sum() / total.sum())]


def loss_from_counts(counts: list[dict], intercept: float, slope: float) -> float:
    xs = np.array([row["x"] for row in counts], dtype=float)
    pos = np.array([row["positive"] for row in counts], dtype=float)
    neg = np.array([row["negative"] for row in counts], dtype=float)
    eta = intercept + slope * xs
    return float((pos * np.logaddexp(0, -eta) + neg * np.logaddexp(0, eta)).sum() / (pos + neg).sum())


def population_loss(spec: dict, intercept: float, slope: float) -> float:
    xs = np.array(spec["x"], dtype=float)
    mass = np.array(spec["masses"], dtype=float)
    true_p = sigmoid(spec["intercept"] + spec["linear"] * xs + spec["quadratic"] * xs**2)
    eta = intercept + slope * xs
    return float((mass * (true_p * np.logaddexp(0, -eta) + (1 - true_p) * np.logaddexp(0, eta))).sum())


def fit_counts(counts: list[dict], test_counts: list[dict], spec: dict) -> dict:
    # One binary record for each (x, outcome) combination, frequency-weighted
    # by its count. This is exactly the individual-row Bernoulli likelihood,
    # with no penalty or approximation to the actual finite dataset.
    design, outcome, frequency = [], [], []
    for row in counts:
        for y, key in [(1, "positive"), (0, "negative")]:
            if row[key] > 0:
                design.append([1.0, float(row["x"])])
                outcome.append(y)
                frequency.append(row[key])
    model = sm.GLM(np.asarray(outcome), np.asarray(design),
                   family=sm.families.Binomial(), freq_weights=np.asarray(frequency))
    result = model.fit(method="IRLS", maxiter=100, tol=1e-10, tol_criterion="params")
    intercept, slope = map(float, result.params)
    score = score_from_counts(counts, intercept, slope)
    assert result.converged, "An actual logistic fit failed to converge."
    assert max(map(abs, score)) < 1e-10, "Independent score equations are not satisfied."
    return {
        "intercept": intercept, "slope": slope,
        "coefficients": [intercept, slope],
        "standardErrors": [float(v) for v in result.bse],
        "converged": bool(result.converged),
        "iterations": int(result.fit_history["iteration"]),
        "normalizedScore": score,
        "trainLoss": loss_from_counts(counts, intercept, slope),
        "testLoss": loss_from_counts(test_counts, intercept, slope),
        "populationLoss": population_loss(spec, intercept, slope),
    }


def corrected(fitted: dict, full_counts: list[dict], test_counts: list[dict], spec: dict) -> dict:
    intercept = fitted["intercept"] + float(np.log(RETENTION))
    slope = fitted["slope"]
    return {
        "intercept": intercept, "slope": slope, "coefficients": [intercept, slope],
        "correction": float(np.log(RETENTION)),
        "fullTrainLoss": loss_from_counts(full_counts, intercept, slope),
        "testLoss": loss_from_counts(test_counts, intercept, slope),
        "populationLoss": population_loss(spec, intercept, slope),
        "normalizedScoreOnFull": score_from_counts(full_counts, intercept, slope),
    }


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    lo, hi = np.quantile(array, [0.025, 0.975], method="linear")
    return {"mean": float(array.mean()), "median": float(np.median(array)), "sd": float(array.std(ddof=1)),
            "q025": float(lo), "q975": float(hi), "p025": float(lo), "p975": float(hi), "lower": float(lo), "upper": float(hi),
            "min": float(array.min()), "max": float(array.max())}


def csv_row(case: str, run: int, counts: list[dict], fitted: dict, correction: dict) -> dict:
    row = {
        "case": case, "run": run,
        "retained_size": sum(r["total"] for r in counts),
        "retained_positives": sum(r["positive"] for r in counts),
        "retained_negatives": sum(r["negative"] for r in counts),
        "intercept": fitted["intercept"], "slope": fitted["slope"],
        "corrected_intercept": correction["intercept"],
        "train_loss": fitted["trainLoss"], "test_loss": fitted["testLoss"],
        "corrected_test_loss": correction["testLoss"],
        "population_loss": fitted["populationLoss"],
        "corrected_population_loss": correction["populationLoss"],
        "iterations": fitted["iterations"],
        "normalized_score_max": max(map(abs, fitted["normalizedScore"])),
    }
    for r in counts:
        row[f"negative_x{r['x']}"] = r["negative"]
    return row


def main() -> None:
    started = time.perf_counter()
    cases, raw_rows, representative_rows, saved_arrays = {}, [], [], {}
    # Both full datasets and untouched holdouts are generated before any
    # downsampling. Their exact arrays are retained for reproducibility.
    generated = {}
    for key, spec in SPECS.items():
        xs = np.array(spec["x"])
        eta = spec["intercept"] + spec["linear"] * xs + spec["quadratic"] * xs**2
        p_by_x = sigmoid(eta)
        full_x = RNG.choice(xs, size=N, p=spec["masses"])
        full_y = RNG.binomial(1, p_by_x[full_x]).astype(np.uint8)
        test_x = RNG.choice(xs, size=N, p=spec["masses"])
        test_y = RNG.binomial(1, p_by_x[test_x]).astype(np.uint8)
        generated[key] = (full_x, full_y, test_x, test_y)
        saved_arrays.update({f"{key}_full_x": full_x, f"{key}_full_y": full_y,
                             f"{key}_test_x": test_x, f"{key}_test_y": test_y})
    np.savez_compressed(OUT / "individual-data.npz", **saved_arrays)

    for key, spec in SPECS.items():
        full_x, full_y, test_x, test_y = generated[key]
        full_counts = counts_for(full_x, full_y, len(spec["x"]))
        test_counts = counts_for(test_x, test_y, len(spec["x"]))
        full_fit = fit_counts(full_counts, test_counts, spec)
        negative_x = full_x[full_y == 0]
        positive_counts = np.bincount(full_x[full_y == 1], minlength=len(spec["x"]))

        def sample_counts() -> list[dict]:
            # Independent Bernoulli inclusion of every actual negative row.
            retained = negative_x[RNG.random(len(negative_x)) < RETENTION]
            negatives = np.bincount(retained, minlength=len(spec["x"]))
            result = []
            for x in spec["x"]:
                pos, neg = int(positive_counts[x]), int(negatives[x])
                result.append({"x": x, "positive": pos, "negative": neg,
                               "total": pos + neg, "p": pos / (pos + neg)})
            assert all(row["positive"] == original["positive"] for row, original in zip(result, full_counts))
            assert all(row["negative"] <= original["negative"] for row, original in zip(result, full_counts))
            return result

        # First draw is representative. It is never replaced after inspection.
        draw_counts = sample_counts()
        draw_fit = fit_counts(draw_counts, test_counts, spec)
        corrected_fit = corrected(draw_fit, full_counts, test_counts, spec)
        representative_rows.append(csv_row(key, 0, draw_counts, draw_fit, corrected_fit))

        aggregate = {"slopes": [], "intercepts": [], "correctedIntercepts": [],
                     "slopeChanges": [], "correctedInterceptChanges": [],
                     "testLosses": [], "correctedTestLosses": [],
                     "populationLosses": [], "correctedPopulationLosses": [],
                     "retainedSizes": []}
        scores, repeats = [], []
        for run in range(1, REPEATS + 1):
            run_counts = sample_counts()
            fitted = fit_counts(run_counts, test_counts, spec)
            correction = corrected(fitted, full_counts, test_counts, spec)
            raw_rows.append(csv_row(key, run, run_counts, fitted, correction))
            repeats.append({
                "draw": run, "intercept": fitted["intercept"], "slope": fitted["slope"],
                "correctedIntercept": correction["intercept"],
                "n": sum(row["total"] for row in run_counts),
                "positive": sum(row["positive"] for row in run_counts),
                "negativeCounts": [row["negative"] for row in run_counts],
                "testLogLoss": fitted["testLoss"], "correctedTestLogLoss": correction["testLoss"],
                "populationLogLoss": fitted["populationLoss"],
                "correctedPopulationLogLoss": correction["populationLoss"],
                "converged": fitted["converged"], "iterations": fitted["iterations"],
                "normalizedScoreMax": max(map(abs, fitted["normalizedScore"])),
            })
            for name, value in [
                ("slopes", fitted["slope"]), ("intercepts", fitted["intercept"]),
                ("correctedIntercepts", correction["intercept"]),
                ("slopeChanges", fitted["slope"] - full_fit["slope"]),
                ("correctedInterceptChanges", correction["intercept"] - full_fit["intercept"]),
                ("testLosses", fitted["testLoss"]),
                ("correctedTestLosses", correction["testLoss"]),
                ("populationLosses", fitted["populationLoss"]),
                ("correctedPopulationLosses", correction["populationLoss"]),
                ("retainedSizes", sum(row["total"] for row in run_counts)),
            ]:
                aggregate[name].append(value)
            scores.append(max(map(abs, fitted["normalizedScore"])))

        truth = dict(spec)
        xs = np.array(spec["x"])
        truth_eta = spec["intercept"] + spec["linear"] * xs + spec["quadratic"] * xs**2
        truth["trueLogits"] = [float(v) for v in truth_eta]
        truth["positiveProbabilities"] = [float(v) for v in sigmoid(truth_eta)]
        truth["populationPositiveShare"] = float(np.dot(spec["masses"], sigmoid(truth_eta)))
        dataset_bytes = full_x.tobytes() + full_y.tobytes()
        test_bytes = test_x.tobytes() + test_y.tobytes()
        cases[key] = {
            "label": spec["label"], "x": spec["x"], "masses": spec["masses"], "truth": truth,
            "trueCoefficients": {"intercept": spec["intercept"], "linear": spec["linear"], "quadratic": spec["quadratic"]},
            "full": {"counts": full_counts, "n": N, "positive": int(full_y.sum()),
                     "fit": full_fit, "testLogLoss": full_fit["testLoss"], "populationLogLoss": full_fit["populationLoss"]},
            "draw": {"counts": draw_counts, "n": sum(row["total"] for row in draw_counts),
                     "positive": int(full_y.sum()), "fit": draw_fit,
                     "testLogLoss": draw_fit["testLoss"], "populationLogLoss": draw_fit["populationLoss"]},
            "corrected": {"fit": corrected_fit, "testLogLoss": corrected_fit["testLoss"], "populationLogLoss": corrected_fit["populationLoss"]},
            "testCounts": test_counts, "repeats": repeats,
            "summary": {name: summarize(aggregate[source]) for name, source in [
                ("slope", "slopes"), ("intercept", "intercepts"),
                ("correctedIntercept", "correctedIntercepts"), ("slopeDelta", "slopeChanges"),
                ("correctedInterceptDelta", "correctedInterceptChanges"),
                ("testLogLoss", "testLosses"), ("correctedTestLogLoss", "correctedTestLosses"),
                ("populationLogLoss", "populationLosses"), ("correctedPopulationLogLoss", "correctedPopulationLosses"),
                ("retainedSize", "retainedSizes"),
            ]},
            "checks": {"allFitsConverged": True, "maxRepeatedNormalizedScore": float(max(scores)),
                       "allPositiveCountsUnchanged": True, "drawAndRepeatsUseSameFullDataset": True,
                       "fullDatasetSha256": hashlib.sha256(dataset_bytes).hexdigest(),
                       "testDatasetSha256": hashlib.sha256(test_bytes).hexdigest()},
        }
        print(key, json.dumps({
            "fullFit": full_fit["coefficients"], "drawFit": draw_fit["coefficients"],
            "correctedFit": corrected_fit["coefficients"],
            "repeatedSlope": cases[key]["summary"]["slope"],
            "repeatedCorrectedIntercept": cases[key]["summary"]["correctedIntercept"],
            "testLoss": {"full": full_fit["testLoss"], "drawCorrected": corrected_fit["testLoss"],
                         "meanRepeatedCorrected": cases[key]["summary"]["correctedTestLogLoss"]["mean"]},
        }))

    fieldnames = ["case", "run", "retained_size", "retained_positives", "retained_negatives",
                  "intercept", "slope", "corrected_intercept", "train_loss", "test_loss",
                  "corrected_test_loss", "population_loss", "corrected_population_loss",
                  "iterations", "normalized_score_max"] + [f"negative_x{x}" for x in range(5)]
    for filename, data in [("repeated-downsamples.csv", raw_rows), ("representative-draws.csv", representative_rows)]:
        with (OUT / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    runtime = time.perf_counter() - started
    output = {
        "protocol": {
            "nTrain": N, "nTest": N, "retention": RETENTION, "repetitions": REPEATS,
            "datasetSize": N, "holdoutSize": N,
            "repeatedDrawsPerFixedDataset": REPEATS, "representativeDrawsPerDataset": 1,
            "positiveRetention": 1.0, "negativeRetention": RETENTION,
            "generation": "Draw each x independently from the stated masses; draw each label independently Bernoulli(p(x)).",
            "holdout": "One separate 200,000-row holdout per case, generated before any downsampling and never resampled or used to select a fit.",
            "downsampling": "Retain every actual positive; independently retain each actual negative with probability 0.1. Draw 500 new independent masks on the same fixed full dataset after the first representative draw.",
            "representativeSelection": "First draw, specified before fitting; no selection of a favorable run or seed.",
            "estimator": "statsmodels GLM, Binomial family, unpenalized intercept plus x; IRLS maximum 100 iterations and parameter convergence tolerance 1e-10; independent normalized score tolerance 1e-10.",
            "likelihood": "Binary (x,y) records frequency-weighted by actual integer counts, exactly equivalent to fitting all individual rows.",
            "correction": "Add log(0.1) to the sampled intercept; retain the sampled slope.",
            "uncertainty": "Mean and empirical 2.5th/97.5th percentiles of 500 downsampling draws, conditional on one fixed full dataset; these are not uncertainty intervals over newly generated populations.",
            "logLossUnits": "Natural-log units (nats) per observation.",
            "internalReproducibility": {"seed": INTERNAL_SEED, "generator": "NumPy default_rng / PCG64", "studentInstruction": False},
            "dependencies": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "statsmodels": statsmodels.__version__},
            "runtimeSeconds": runtime,
        },
        "cases": cases,
    }
    with (OUT / "sampling-results.json").open("w") as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Finished 1,004 logistic fits in {runtime:.3f} seconds.")


if __name__ == "__main__":
    main()
