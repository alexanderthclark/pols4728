#!/usr/bin/env python3
"""Reproduce and verify the noisy shallow-network width illustration.

``--check`` needs NumPy only. It recomputes all metrics on the recorded 40,000
independent observations and verifies the complete, unsmoothed zero contours
against the frozen weights. ``--refit`` additionally needs PyTorch and repeats
the jointly trained illustration. Add ``--refit-confirmation`` to repeat all
three independent confirmation samples too. No private notes, downloads, or
scratch files are needed. CPU/library versions can change the numerical fit;
the checked-in models are the authoritative classroom illustration.

The development example was chosen before the fresh test draw. Confirmation
results are retained, including a two-unit optimization failure. This is an
illustration of capacity and generalization, not an empirical double-descent
curve or a promise that more units always improve a boundary or its predictions.
"""

import argparse
import itertools
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

WIDTHS = [2, 4, 8, 16, 32, 64, 128, 256]
TRAIN_SEED = 6200
TEST_SEED = 64000
INITIALIZATION_SEED = 4100
CONFIRMATION_SEEDS = [6400, 6401, 6402]
NOISE = 0.18
EPOCHS = 12000
STARTS = 4
TEST_COUNT = 40000
DOMAIN = [[-0.22, float(np.pi + 0.22)], [-1.6, 1.4]]
MEAN = np.array([np.pi / 2, -0.05])
SCALE = np.array([np.pi / 2, 1.0])
OUTPUT = Path(__file__).resolve().parents[1] / "docs/neural-networks/width-data.js"
CONFIRMATION_OUTPUT = Path(__file__).with_name("neural-width-confirmation.json")


def dataset(seed, points_per_class=41, grid=True):
    rng = np.random.default_rng(seed)
    labels = np.repeat([0, 1], points_per_class)
    x1 = (np.tile(np.linspace(0, np.pi, points_per_class), 2) if grid
          else rng.uniform(0, np.pi, 2 * points_per_class))
    x2 = -np.sin(x1) ** 2 + 0.9 * labels + rng.normal(0, NOISE, len(labels))
    return np.column_stack([x1, x2]), labels


def rounded(values):
    return np.round(np.asarray(values), 12).tolist()


def logits(model, inputs):
    inputs = np.asarray(inputs, dtype=float)
    if not len(inputs):
        return np.empty(0)
    weights = np.asarray(model["hiddenWeights"])
    bias = np.asarray(model["hiddenBias"])
    output = np.asarray(model["outputWeights"])
    # Batching avoids making the browser export depend on a large temporary
    # matrix when the 40,000-observation test is independently recomputed.
    return np.concatenate([
        np.maximum(inputs[j:j + 4096] @ weights.T + bias, 0) @ output
        + model["outputBias"] for j in range(0, len(inputs), 4096)
    ])


def metrics(model, inputs, labels):
    score = logits(model, inputs)
    mistakes = int(np.count_nonzero((score >= 0) != labels))
    return mistakes, mistakes / len(labels), float(np.mean(
        np.logaddexp(0, score) - labels * score))


def raw_model(parameters, width, start):
    p = np.asarray(parameters, dtype=float)
    weights = p[:2 * width].reshape(width, 2) / SCALE
    bias = p[2 * width:3 * width] - weights @ MEAN
    return {
        "width": width,
        "hiddenWeights": rounded(weights),
        "hiddenBias": rounded(bias),
        "outputWeights": rounded(p[3 * width:4 * width]),
        "outputBias": round(float(p[-1]), 12),
        "selectedStart": int(start),
    }


def fit(inputs, labels, width):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(INITIALIZATION_SEED)
    x = torch.tensor((inputs - MEAN) / SCALE, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    weights = torch.randn(STARTS, width, 2, requires_grad=True)
    bias = (torch.randn(STARTS, width) * 0.5).requires_grad_()
    output = (torch.randn(STARTS, width) / np.sqrt(width)).requires_grad_()
    intercept = torch.zeros(STARTS, requires_grad=True)
    optimizer = torch.optim.Adam([weights, bias, output, intercept], lr=0.01)
    for step in range(EPOCHS):
        optimizer.zero_grad()
        hidden = torch.relu(torch.einsum("nd,kwd->knw", x, weights) + bias[:, None, :])
        score = torch.einsum("knw,kw->kn", hidden, output) + intercept[:, None]
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            score, y.expand_as(score), reduction="none").mean(dim=1)
        losses.sum().backward()
        optimizer.step()
        # This precise placement reproduces the development run: step 9000
        # still uses .01, and the subsequent 2999 updates use .001.
        if step == 9000:
            for group in optimizer.param_groups:
                group["lr"] = 0.001
    with torch.no_grad():
        hidden = torch.relu(torch.einsum("nd,kwd->knw", x, weights) + bias[:, None, :])
        score = torch.einsum("knw,kw->kn", hidden, output) + intercept[:, None]
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            score, y.expand_as(score), reduction="none").mean(dim=1)
    start = int(losses.argmin())
    parameters = np.concatenate([
        weights[start].detach().numpy().ravel(), bias[start].detach().numpy(),
        output[start].detach().numpy(), [intercept[start].item()],
    ])
    return raw_model(parameters, width, start)


def clipped_line(normal, intercept):
    """Return a point, unit tangent, and the part of a line inside DOMAIN."""
    normal = np.asarray(normal, dtype=float)
    norm2 = float(normal @ normal)
    if norm2 < 1e-26:
        return None
    origin = -intercept * normal / norm2
    direction = np.array([-normal[1], normal[0]]) / np.sqrt(norm2)
    low, high = -np.inf, np.inf
    for axis, (minimum, maximum) in enumerate(DOMAIN):
        if abs(direction[axis]) < 1e-14:
            if not minimum - 1e-10 <= origin[axis] <= maximum + 1e-10:
                return None
        else:
            a, b = sorted([(minimum - origin[axis]) / direction[axis],
                           (maximum - origin[axis]) / direction[axis]])
            low, high = max(low, a), min(high, b)
    if low > high + 1e-12:
        return None
    return origin, direction, low, high


def line_roots(model, normal, intercept):
    """Exact score roots along one line, split at every ReLU breakpoint."""
    clipped = clipped_line(normal, intercept)
    if clipped is None:
        return []
    origin, direction, low, high = clipped
    weights = np.asarray(model["hiddenWeights"])
    bias = np.asarray(model["hiddenBias"])
    denominator = weights @ direction
    changing = np.abs(denominator) > 1e-13
    cuts = -(weights[changing] @ origin + bias[changing]) / denominator[changing]
    cuts = cuts[(cuts > low + 1e-11) & (cuts < high - 1e-11)]
    positions = np.sort(np.r_[low, cuts, high])
    positions = positions[np.r_[True, np.diff(positions) > 1e-11]]
    points = origin + positions[:, None] * direction
    scores = logits(model, points)
    roots = [point for point, score in zip(points, scores) if abs(score) < 1e-9]
    for j in np.flatnonzero(scores[:-1] * scores[1:] < 0):
        fraction = scores[j] / (scores[j] - scores[j + 1])
        roots.append(points[j] + fraction * (points[j + 1] - points[j]))
    return roots


def boundaries(model):
    """Extract all zero-set branches as exact ReLU line segments.

    Every corner lies on a hidden neuron's activation line. Enumerating its
    score roots, plus roots on the four domain edges, visits every crossed
    activation region. Within each such convex region the output logit is
    affine, so joining its two extreme vertices gives an exact zero segment.
    This introduces no spline, visual smoothing, or marching-grid shortcuts.
    """
    weights = np.asarray(model["hiddenWeights"])
    bias = np.asarray(model["hiddenBias"])
    output = np.asarray(model["outputWeights"])
    lines = list(zip(weights, bias)) + [
        ([1, 0], -DOMAIN[0][0]), ([1, 0], -DOMAIN[0][1]),
        ([0, 1], -DOMAIN[1][0]), ([0, 1], -DOMAIN[1][1]),
    ]
    unique = {}
    for normal, intercept in lines:
        for point in line_roots(model, normal, intercept):
            unique.setdefault(tuple(np.round(point, 10)), point)
    points = list(unique.values())
    regions = {}
    for index, point in enumerate(points):
        activation = weights @ point + bias
        zeros = np.flatnonzero(np.abs(activation) < 1e-7)
        if len(zeros) > 8:
            raise ValueError("Degenerate coincident activation lines need explicit handling.")
        for sides in itertools.product([False, True], repeat=len(zeros)):
            active = activation > 0
            active[zeros] = sides
            key = np.packbits(active).tobytes()
            regions.setdefault(key, (active, set()))[1].add(index)
    edges = set()
    for active, members in regions.values():
        if len(members) < 2:
            continue
        gradient = output[active] @ weights[active]
        if np.linalg.norm(gradient) < 1e-12:
            raise ValueError("An identically zero region has no unique decision boundary.")
        tangent = np.array([-gradient[1], gradient[0]])
        ordered = sorted(members, key=lambda j: float(points[j] @ tangent))
        for a, b in zip(ordered[:-1], ordered[1:]):
            if np.linalg.norm(points[a] - points[b]) > 1e-10:
                edges.add(tuple(sorted([a, b])))
    neighbors = {j: set() for edge in edges for j in edge}
    for a, b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    unused = set(edges)
    paths = []

    def walk(start, following):
        path = [start, following]
        unused.remove(tuple(sorted([start, following])))
        while len(neighbors[path[-1]]) == 2:
            candidates = [j for j in neighbors[path[-1]]
                          if tuple(sorted([path[-1], j])) in unused]
            if not candidates:
                break
            following = min(candidates)
            unused.remove(tuple(sorted([path[-1], following])))
            path.append(following)
        return rounded([points[j] for j in path])

    for start in sorted(neighbors):
        if len(neighbors[start]) != 2:
            for following in sorted(neighbors[start]):
                if tuple(sorted([start, following])) in unused:
                    paths.append(walk(start, following))
    while unused:
        paths.append(walk(*min(unused)))
    return paths


def decorate(model, inputs, labels, test_inputs, test_labels):
    for prefix, xx, yy in [("training", inputs, labels), ("test", test_inputs, test_labels)]:
        mistakes, error, bce = metrics(model, xx, yy)
        model[prefix + "Mistakes"] = mistakes
        model[prefix + "Error"] = error
        model[prefix + "BCE"] = bce
    model["boundaries"] = boundaries(model)
    return model


def fit_confirmation():
    """Repeat the original 12-block float32 batched confirmation training.

    The three samples share four initialization arrays but are optimized
    independently. Keeping bmm/reduction order matches the historical run;
    floating-point/library differences can still alter nonlinear optimization.
    """
    import torch
    torch.set_num_threads(1)
    datasets = [dataset(seed) for seed in CONFIRMATION_SEEDS]
    x = torch.tensor(np.stack([(xx - MEAN) / SCALE for xx, _ in datasets]), dtype=torch.float32)
    y = torch.tensor(np.stack([yy for _, yy in datasets]), dtype=torch.float32)
    x = x[:, None].expand(-1, STARTS, -1, -1).reshape(12, 82, 2)
    y = y[:, None].expand(-1, STARTS, -1).reshape(12, 82)
    models = []
    for width in WIDTHS:
        torch.manual_seed(INITIALIZATION_SEED)
        initial_weights = torch.randn(STARTS, width, 2)
        initial_bias = torch.randn(STARTS, width) * 0.5
        initial_output = torch.randn(STARTS, width) / np.sqrt(width)
        weights = initial_weights.repeat(3, 1, 1).requires_grad_()
        bias = initial_bias.repeat(3, 1).requires_grad_()
        output = initial_output.repeat(3, 1).requires_grad_()
        intercept = torch.zeros(12, requires_grad=True)
        optimizer = torch.optim.Adam([weights, bias, output, intercept], lr=0.01)
        for step in range(EPOCHS):
            optimizer.zero_grad()
            hidden = torch.relu(torch.bmm(x, weights.transpose(1, 2)) + bias[:, None, :])
            score = (hidden * output[:, None, :]).sum(dim=2) + intercept[:, None]
            losses = torch.nn.functional.binary_cross_entropy_with_logits(
                score, y, reduction="none").mean(dim=1)
            losses.sum().backward()
            optimizer.step()
            if step == 9000:
                for group in optimizer.param_groups:
                    group["lr"] = 0.001
        with torch.no_grad():
            hidden = torch.relu(torch.bmm(x, weights.transpose(1, 2)) + bias[:, None, :])
            score = (hidden * output[:, None, :]).sum(dim=2) + intercept[:, None]
            losses = torch.nn.functional.binary_cross_entropy_with_logits(
                score, y, reduction="none").mean(dim=1)
        for index, seed in enumerate(CONFIRMATION_SEEDS):
            candidates = losses[index * STARTS:(index + 1) * STARTS]
            start = int(candidates.argmin())
            chosen = index * STARTS + start
            parameters = np.concatenate([
                weights[chosen].detach().numpy().ravel(), bias[chosen].detach().numpy(),
                output[chosen].detach().numpy(), [intercept[chosen].item()],
            ])
            models.append({"trainingSeed": seed, "width": width,
                           "normalizedParameters": parameters.tolist(),
                           "selectedStart": start,
                           "candidateTrainingBCE": candidates.tolist()})
    return confirmation_record(models, torch.__version__)


def confirmation_record(models, torch_version="2.8.0"):
    return {
        "provenance": {
            "description": "All 24 frozen confirmation fits; retained separately from browser data so every reported confirmation metric can be replayed.",
            "trainingSeeds": CONFIRMATION_SEEDS, "widths": WIDTHS,
            "noiseSD": NOISE, "testSeed": TEST_SEED, "testCount": TEST_COUNT,
            "normalization": {"mean": MEAN.tolist(), "scale": SCALE.tolist()},
            "parameterLayout": "Flattened hidden weights (width,2), hidden biases (width), output weights (width), output bias (1); inputs are standardized using the recorded mean and scale.",
            "training": "Joint Adam on binary cross-entropy, float32 CPU; three datasets times four starts in 12 independent blocks using torch.bmm and elementwise output multiplication followed by sum. 12,000 updates: lr.01 through step9000, lr.001 thereafter; no explicit penalty.",
            "initializationSeed": INITIALIZATION_SEED,
            "selection": "Lowest final training BCE among the four starts, before evaluating the independent test draw. The same four initialization arrays are reused across the three datasets.",
            "torchVersion": torch_version,
            "reproduction": "--check replays all metrics directly from these frozen parameters without retraining. --refit --refit-confirmation repeats the original batched protocol but floating-point/library changes can produce numerically different fits; byte-for-byte retraining is not promised.",
        },
        "models": models,
    }


def summarize_confirmation(record):
    test_inputs, test_labels = dataset(TEST_SEED, TEST_COUNT // 2, False)
    rows = []
    for saved in record["models"]:
        inputs, labels = dataset(saved["trainingSeed"])
        model = raw_model(saved["normalizedParameters"], saved["width"], saved["selectedStart"])
        row = {"trainingSeed": saved["trainingSeed"], "width": saved["width"]}
        for prefix, xx, yy in [("training", inputs, labels), ("test", test_inputs, test_labels)]:
            mistakes, error, bce = metrics(model, xx, yy)
            row.update({prefix + "Mistakes": mistakes, prefix + "Error": error, prefix + "BCE": bce})
        rows.append(row)
    return rows


def build_data(models, confirmation, torch_version="2.8.0"):
    inputs, labels = dataset(TRAIN_SEED)
    # Export precision is also the precision used to compute the metrics.
    inputs = np.asarray(rounded(inputs))
    test_inputs, test_labels = dataset(TEST_SEED, TEST_COUNT // 2, False)
    display_indices = np.arange(0, TEST_COUNT, 250)
    return {
        "inputs": inputs.tolist(), "labels": labels.tolist(),
        "testInputs": rounded(test_inputs[display_indices]),
        "testLabels": test_labels[display_indices].tolist(),
        "testIndices": display_indices.tolist(), "testCount": TEST_COUNT,
        "domain": DOMAIN,
        "models": [decorate(model, inputs, labels, test_inputs, test_labels) for model in models],
        "provenance": {
            "description": "New jointly fitted binary ReLU examples inspired by the notes' 2-versus-256-neuron comparison, not the notes' unavailable original fitted models.",
            "dataFormula": "x1=j*pi/40; x2=-sin(x1)^2+0.9*y+epsilon; epsilon~Normal(0,0.18^2); j=0,...,40; y in {0,1}.",
            "testFormula": "Same two classes and vertical-noise distribution, with independent x1~Uniform(0,pi), 20,000 observations per class.",
            "trainingCount": 82, "trainingSeed": TRAIN_SEED, "testSeed": TEST_SEED,
            "developmentValidationSeed": 62000, "noiseSD": NOISE,
            "hiddenActivation": "ReLU", "outputActivation": "sigmoid",
            "training": "Joint full-batch Adam minimization of mean binary cross-entropy; no explicit regularization; float32 CPU training; 12,000 updates. Learning rate .01 for updates 0 through 9000, then .001.",
            "initialization": "Four normal starts per width: hidden weights SD1, hidden biases SD.5, output weights SD1/sqrt(width), output bias0. PyTorch internal initialization identifier 4100 is reset for each width.",
            "initializationSeed": INITIALIZATION_SEED, "starts": STARTS, "epochs": EPOCHS,
            "selection": "Lowest final training binary cross-entropy among four starts. No validation or test outcomes choose among the starts.",
            "developmentSelection": "Development sample 6200 and the common fitting setup were selected using development comparisons before the independent test draw 64000. This is a selected teaching illustration, not a representative Monte Carlo estimate.",
            "normalization": "Training inputs were (x-[pi/2,-.05])/[pi/2,1]. Exported weights and biases are algebraically transformed to act directly on original x1,x2.",
            "displayTestSelection": "Every 250th observation from the independent 40,000-point test sample, starting at 0: 160 points, 80 per class. Selection ignores predictions and errors. Reported test error always uses all 40,000 observations.",
            "boundaries": "Exact piecewise-linear zero-logit segments obtained by activation-region intersections, including every branch in the displayed domain; vertices rounded to 12 decimal places. No geometric smoothing.",
            "generator": "scripts/generate-neural-width.py",
            "numpyVersion": np.__version__, "torchVersion": torch_version,
            "interpretation": "This selected example illustrates rounder boundaries and fewer test classification mistakes with a wider model. Improvements are nonmonotonic and not guaranteed. Correct labels do not imply exact or well-calibrated probabilities. Every displayed width already classifies this training sample perfectly, so this is not an empirical double-descent curve.",
        },
        "confirmationSummary": {
            "trainingSeeds": CONFIRMATION_SEEDS, "testSeed": TEST_SEED,
            "trainingCount": 82, "testCount": TEST_COUNT, "rows": confirmation,
            "interpretation": "All three fresh training draws and every width are retained. The four-start two-neuron optimizer fails to classify the training sample perfectly for draw 6400; its comparison mixes optimization and approximation. Draw 6401 has lower test classification error with 256 units but worse test cross-entropy. Draw 6402 has only a small classification improvement and worse test cross-entropy. These checks do not establish universal smoothing, better probability estimates, or double descent.",
        },
    }


def export(data):
    return (
        "/* Generated by scripts/generate-neural-width.py; frozen fitted models.\n"
        " * Selected teaching illustration; no geometric smoothing or fabricated errors. */\n"
        "(function (root, factory) {\n  \"use strict\";\n  const api = factory();\n"
        "  if (typeof module === \"object\" && module.exports) module.exports = api;\n"
        "  if (root) root.NNWidthData = api;\n"
        "})(typeof window !== \"undefined\" ? window : globalThis, function () {\n"
        "  \"use strict\";\n  return Object.freeze(" + json.dumps(data, indent=2, allow_nan=False)
        + ");\n});\n"
    )


def read_export():
    text = OUTPUT.read_text()
    return json.loads(text.split("return Object.freeze(", 1)[1].rsplit(");", 2)[0])


def check(data, confirmation=None):
    assert [m["width"] for m in data["models"]] == WIDTHS
    inputs, labels = dataset(TRAIN_SEED)
    np.testing.assert_allclose(data["inputs"], inputs, atol=6e-13, rtol=0)
    np.testing.assert_array_equal(data["labels"], labels)
    test_inputs, test_labels = dataset(TEST_SEED, TEST_COUNT // 2, False)
    indices = np.arange(0, TEST_COUNT, 250)
    np.testing.assert_array_equal(data["testIndices"], indices)
    np.testing.assert_allclose(data["testInputs"], test_inputs[indices], atol=6e-13, rtol=0)
    np.testing.assert_array_equal(data["testLabels"], test_labels[indices])
    assert data["testCount"] == TEST_COUNT
    max_residual = 0.0
    for model in data["models"]:
        for prefix, xx, yy in [("training", np.asarray(data["inputs"]), labels), ("test", test_inputs, test_labels)]:
            mistakes, error, bce = metrics(model, xx, yy)
            assert model[prefix + "Mistakes"] == mistakes
            assert model[prefix + "Error"] == error
            assert abs(model[prefix + "BCE"] - bce) < 1e-12
        assert model["trainingMistakes"] == 0
        regenerated = boundaries(model)
        assert len(regenerated) == len(model["boundaries"])
        for saved, rebuilt in zip(model["boundaries"], regenerated):
            np.testing.assert_allclose(saved, rebuilt, atol=1e-10, rtol=0)
            points = np.asarray(saved)
            assert len(points) >= 2
            for axis, (minimum, maximum) in enumerate(DOMAIN):
                assert np.all((points[:, axis] >= minimum - 1e-10) & (points[:, axis] <= maximum + 1e-10))
            probes = np.concatenate([points] + [
                points[:-1] * (1 - fraction) + points[1:] * fraction
                for fraction in [0.25, 0.5, 0.75]])
            residual = float(np.max(np.abs(logits(model, probes))))
            max_residual = max(max_residual, residual)
            assert residual < 1e-5, (model["width"], residual)
    rows = data["confirmationSummary"]["rows"]
    if confirmation is None:
        confirmation = json.loads(CONFIRMATION_OUTPUT.read_text())
    replayed = summarize_confirmation(confirmation)
    recorded = {(r["trainingSeed"], r["width"]): r for r in rows}
    for saved in confirmation["models"]:
        assert saved["selectedStart"] == int(np.argmin(saved["candidateTrainingBCE"]))
    for replay in replayed:
        saved = recorded[(replay["trainingSeed"], replay["width"])]
        for key in ["trainingMistakes", "trainingError", "testMistakes", "testError"]:
            assert saved[key] == replay[key]
        for key in ["trainingBCE", "testBCE"]:
            assert abs(saved[key] - replay[key]) < 1e-10
    assert len(rows) == len(WIDTHS) * len(CONFIRMATION_SEEDS)
    assert {(r["trainingSeed"], r["width"]) for r in rows} == set(itertools.product(CONFIRMATION_SEEDS, WIDTHS))
    for row in rows:
        assert row["trainingMistakes"] / 82 == row["trainingError"]
        assert row["testMistakes"] / TEST_COUNT == row["testError"]
    assert next(r for r in rows if r["trainingSeed"] == 6400 and r["width"] == 2)["trainingMistakes"] == 8
    return {"models": len(WIDTHS), "trainingCount": 82, "testCount": TEST_COUNT,
            "displayTestCount": len(indices), "confirmationModelsReplayed": len(replayed),
            "maximumBoundaryLogitResidual": max_residual,
            "endpointTestMistakes": [data["models"][0]["testMistakes"], data["models"][-1]["testMistakes"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Verify frozen models, metrics, and complete contours without training.")
    mode.add_argument("--refit", action="store_true", help="Repeat the fixed training protocol and regenerate the illustration.")
    parser.add_argument("--refit-confirmation", action="store_true", help="With --refit, repeat all three independent confirmation draws too.")
    args = parser.parse_args()
    if args.refit_confirmation and not args.refit:
        parser.error("--refit-confirmation requires --refit")
    frozen = read_export() if OUTPUT.exists() else None
    if frozen is None and not (args.refit and args.refit_confirmation):
        parser.error("No frozen export exists; use --refit --refit-confirmation to regenerate everything.")
    if args.refit:
        inputs, labels = dataset(TRAIN_SEED)
        models = [fit(inputs, labels, width) for width in WIDTHS]
        confirmation = (fit_confirmation() if args.refit_confirmation
                        else json.loads(CONFIRMATION_OUTPUT.read_text()))
        import torch
        frozen = build_data(models, summarize_confirmation(confirmation), torch.__version__)
        report = check(frozen, confirmation)
        OUTPUT.write_text(export(frozen))
        if args.refit_confirmation:
            CONFIRMATION_OUTPUT.write_text(json.dumps(confirmation, indent=2, allow_nan=False) + "\n")
    else:
        report = check(frozen)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
