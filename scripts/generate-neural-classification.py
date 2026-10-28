#!/usr/bin/env python3
"""Fit and export the shallow-network story's constructed binary example.

Run from any working directory with NumPy and SciPy installed. No downloads,
private course files, or saved training state are required. ``--check`` repeats
the fit and checks the checked-in export. Reproducibility is numerical across
library versions; the recorded versions reproduce the export byte for byte.

These are new synthetic observations and a new fit inspired by two adjacent
curved groups in the course notes, not recovered observations or weights from
the notes. Both hidden units and the logistic head are fitted jointly.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit


POINTS_PER_CLASS = 41
CLASS_OFFSET = 0.9
L2 = 0.0001
STARTS = 32
OUTPUT = Path(__file__).resolve().parents[1] / "docs/neural-networks/classification-data.js"


def dataset():
    x1 = np.linspace(0, np.pi, POINTS_PER_CLASS)
    inputs = np.concatenate([
        np.column_stack((x1, -np.sin(x1) ** 2 + CLASS_OFFSET * label))
        for label in (0, 1)
    ])
    return inputs, np.repeat([0, 1], POINTS_PER_CLASS)


def forward(parameters, inputs):
    weights = parameters[:4].reshape(2, 2)
    preactivations = inputs @ weights.T + parameters[4:6]
    hidden = np.maximum(0, preactivations)
    logits = hidden @ parameters[6:8] + parameters[8]
    return preactivations, hidden, logits


def objective(parameters, inputs, labels):
    preactivations, hidden, logits = forward(parameters, inputs)
    # The intercept of the head is unpenalized. All eight other parameters
    # receive the same L2 penalty, including the hidden intercepts.
    cross_entropy = np.mean(np.logaddexp(0, logits) - labels * logits)
    loss = cross_entropy + L2 * np.sum(parameters[:-1] ** 2) / 2
    output_gradient = (expit(logits) - labels) / len(labels)
    hidden_gradient = (
        output_gradient[:, None] * parameters[6:8] * (preactivations > 0)
    )
    gradient = np.concatenate([
        (hidden_gradient.T @ inputs).ravel(),
        hidden_gradient.sum(axis=0),
        hidden.T @ output_gradient,
        [output_gradient.sum()],
    ])
    gradient[:-1] += L2 * parameters[:-1]
    return loss, gradient


def fit(inputs, labels):
    candidates = []
    for start in range(STARTS):
        initial = np.random.default_rng(start).normal(size=9)
        result = minimize(
            objective, initial, args=(inputs, labels), jac=True,
            method="L-BFGS-B",
            options={"maxiter": 3000, "gtol": 1e-10, "ftol": 1e-14},
        )
        _, hidden, logits = forward(result.x, inputs)
        signed_logits = (2 * labels - 1) * logits
        distances = np.linalg.norm(hidden[:, None, :] - hidden[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        # All observations must be correctly classified with substantial
        # margin. Distinct coordinates keep each plotted point trackable.
        # This selects a fit of the same fixed data; it never changes labels.
        if np.min(signed_logits) > 4 and np.min(distances) > 1e-6:
            candidates.append((float(result.fun), start, result))
    if not candidates:
        raise RuntimeError("No fit meets the teaching example's verification criteria.")
    _, start, result = min(candidates, key=lambda entry: (entry[0], entry[1]))
    return np.round(result.x, 12), start, result


def rounded(value):
    return np.round(np.asarray(value), 12).tolist()


def build_data():
    raw_inputs, labels = dataset()
    parameters, start, result = fit(raw_inputs, labels)
    # Metrics refer to the frozen, rounded coordinates and weights actually
    # read by the browser, not extra-precision values held by the optimizer.
    inputs = np.round(raw_inputs, 12)
    preactivations, hidden, logits = forward(parameters, inputs)
    signed_logits = (2 * labels - 1) * logits
    correct = int(np.count_nonzero((logits >= 0) == labels))
    assert correct == len(labels)
    assert np.min(signed_logits) > 4
    assert np.linalg.matrix_rank(hidden - hidden.mean(axis=0)) == 2
    assert np.all(np.any(preactivations > 0, axis=0))
    assert np.all(np.any(preactivations < 0, axis=0))

    # The two class convex hulls share (pi/2, 0): average the two class-0
    # endpoints; for class 1 mix its endpoints (0.05 each) and center (0.9).
    # Therefore no affine score can be strictly negative on all class-0
    # observations and strictly positive on all class-1 observations.
    witness = {
        "class0": {"indices": [0, 40], "weights": [0.5, 0.5]},
        "class1": {"indices": [41, 61, 81], "weights": [0.05, 0.9, 0.05]},
        "intersection": rounded([np.pi / 2, 0]),
    }
    return {
        "inputs": inputs.tolist(),
        "labels": labels.tolist(),
        "hiddenWeights": parameters[:4].reshape(2, 2).tolist(),
        "hiddenBias": parameters[4:6].tolist(),
        "outputWeights": parameters[6:8].tolist(),
        "outputBias": float(parameters[8]),
        "nonseparabilityWitness": witness,
        "provenance": {
            "description": "New constructed binary example inspired by adjacent curved groups in the course notes; not the notes' original fitted network.",
            "dataFormula": "x1 = j*pi/40, x2 = -sin(x1)^2 + 0.9*y; j = 0,...,40; y in {0,1}",
            "architecture": [2, 2, 1],
            "hiddenActivation": "ReLU",
            "outputActivation": "sigmoid",
            "training": "Joint full-batch minimization of mean binary cross-entropy plus an L2 penalty on all parameters except the output intercept, using analytic backpropagation gradients and scipy.optimize.minimize (L-BFGS-B).",
            "l2Coefficient": L2,
            "initialization": "32 independent standard-normal parameter initializations from NumPy default_rng with internal start identifiers 0 through 31.",
            "selection": "Lowest penalized training loss among fits with minimum signed logit greater than 4 and all hidden-coordinate pairs distinct by more than 1e-6.",
            "selectedStart": start,
            "optimizerConverged": bool(result.success),
            "optimizerIterations": int(result.nit),
            "numpyVersion": np.__version__,
            "scipyVersion": scipy.__version__,
            "generator": "scripts/generate-neural-classification.py",
            "interpretation": "Training accuracy only; neither population separation nor performance on unseen data is established.",
        },
        "metrics": {
            "trainingCount": len(labels),
            "correct": correct,
            "accuracy": correct / len(labels),
            "minSignedLogit": float(np.min(signed_logits)),
            "minGeometricMargin": float(np.min(signed_logits) / np.linalg.norm(parameters[6:8])),
            "meanCrossEntropy": float(np.mean(np.logaddexp(0, logits) - labels * logits)),
            "hiddenCoordinateRank": 2,
            "originalLinearSeparable": False,
        },
    }


def export(data):
    return (
        "/* Generated by scripts/generate-neural-classification.py.\n"
        " * New fitted teaching example; do not hand-edit coordinates or weights. */\n"
        "(function (root, factory) {\n"
        '  "use strict";\n'
        "  const api = factory();\n"
        '  if (typeof module === "object" && module.exports) module.exports = api;\n'
        "  if (root) root.NNClassificationData = api;\n"
        '})(typeof window !== "undefined" ? window : globalThis, function () {\n'
        '  "use strict";\n'
        "  return Object.freeze(" + json.dumps(data, indent=2, allow_nan=False) + ");\n"
        "});\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Refit and compare with the checked-in export.")
    args = parser.parse_args()
    data = build_data()
    text = export(data)
    if args.check:
        if OUTPUT.read_text() != text:
            raise SystemExit("Export differs: regenerate using the recorded dependency versions.")
    else:
        OUTPUT.write_text(text)
    print(json.dumps({"selectedStart": data["provenance"]["selectedStart"], **data["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
