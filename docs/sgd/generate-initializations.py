#!/usr/bin/env python3
"""Record real scikit-learn fits without changing the animated SGD experiment.

Requires numpy, scipy, and scikit-learn. The generated file records their
versions, the complete estimator configuration, and the exact source-data hash.
Run with --solver adam or --solver sgd; the solver is deliberately explicit.
Use --add-initializations to recover the initial states of an existing export
without rerunning its full training or changing its final fitted functions.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import warnings

import numpy as np
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor


HERE = Path(__file__).resolve().parent
WIDTHS = (2, 12)
SEEDS = tuple(range(1, 9))


def export_model(estimator):
    """Use the same scalar-network storage as model.mjs."""
    return {
        "c": float(estimator.intercepts_[1][0]),
        "units": [
            {"w": float(w), "b": float(b), "a": float(a)}
            for w, b, a in zip(
                estimator.coefs_[0][0],
                estimator.intercepts_[0],
                estimator.coefs_[1][:, 0],
            )
        ],
    }


def predict_exported(model, x):
    return model["c"] + sum(
        unit["a"] * np.maximum(0, unit["w"] * x + unit["b"])
        for unit in model["units"]
    )


class RecordingMLPRegressor(MLPRegressor):
    """Copy sklearn's initialized parameters before its first training update.

    sklearn has no public initialize-only method. This small recording hook
    delegates initialization unchanged, so it neither consumes random draws nor
    substitutes an independently implemented initialization formula.
    """

    def _initialize(self, *args, **kwargs):
        super()._initialize(*args, **kwargs)
        self.initial_model_ = export_model(self)


def initial_fields(estimator, plot_x, train_x, train_y):
    model = estimator.initial_model_
    return {
        "initialModel": model,
        "initialPredictions": predict_exported(model, plot_x).tolist(),
        "initialTrainMSE": float(np.mean((predict_exported(model, train_x) - train_y) ** 2)),
    }


def add_initializations(result):
    """Augment a saved export, preserving every original fit and setting."""
    config = result["config"]
    runtime = config["runtime"]
    for name, installed in (("scikitLearn", sklearn.__version__), ("numpy", np.__version__)):
        if runtime[name] != installed:
            raise ValueError(f"Recover initial states with the saved {name} version {runtime[name]}, not {installed}.")
    source_bytes = (HERE / config["source"]).read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != config["sourceSHA256"]:
        raise ValueError("The source dataset differs from the saved fits.")
    rows = json.loads(source_bytes)["data"]
    x = np.asarray([row["x"] for row in rows], dtype=np.float64)
    y = np.asarray([row["y"] for row in rows], dtype=np.float64)
    plot_x = np.asarray(result["x"], dtype=np.float64)
    order = np.random.RandomState(config["batchOrderSeed"]).permutation(len(x))
    batch = order[:config["batchSize"]] if config["solver"] == "sgd" else order
    for run in result["runs"]:
        settings = dict(
            hidden_layer_sizes=(run["width"],),
            random_state=run["seed"],
            **config["estimatorSettings"],
        )
        recording = RecordingMLPRegressor(**settings)
        reference = MLPRegressor(**settings)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            for estimator in (recording, reference):
                if config["solver"] == "sgd":
                    estimator.fit(x[batch, None], y[batch])
                else:
                    estimator.partial_fit(x[batch, None], y[batch])
        # The observer must not change initialization, RNG use, or the first
        # update. Compare against an ordinary estimator using its public API.
        for observed, original in zip(
            recording.coefs_ + recording.intercepts_,
            reference.coefs_ + reference.intercepts_,
        ):
            np.testing.assert_array_equal(observed, original)
        run.update(initial_fields(recording, plot_x, x, y))
    return result


def generate(solver, epochs, batch_size, batch_order_seed, learning_rate):
    source = HERE / "experiment.json"
    source_bytes = source.read_bytes()
    experiment = json.loads(source_bytes)
    rows = experiment["data"]
    x = np.asarray([row["x"] for row in rows], dtype=np.float64)
    y = np.asarray([row["y"] for row in rows], dtype=np.float64)
    if len(x) != 2048 or not np.array_equal(y, (x > 0).astype(float)):
        raise ValueError("Expected the existing 2,048-point noiseless step dataset.")
    if len(x) % batch_size:
        raise ValueError("Batch size must divide the dataset.")

    # Keep batch order identical across every width and initialization. sklearn's
    # random_state normally affects BOTH initialization and shuffling, and the
    # initialization consumes a width-dependent number of random draws. External
    # permutations plus shuffle=False isolate the initialization comparison.
    order_rng = np.random.RandomState(batch_order_seed)
    orders = [order_rng.permutation(len(x)) for _ in range(epochs)]
    order_bytes = np.asarray(orders, dtype="<i4").tobytes()
    plot_x = np.linspace(-1, 1, 401)
    metric_x = -1 + 2 * (np.arange(4000) + 0.5) / 4000
    metric_y = (metric_x > 0).astype(float)
    updates = epochs * len(x) // batch_size
    settings = dict(
        activation="relu",
        solver=solver,
        alpha=0.0,
        batch_size=batch_size,
        learning_rate="constant",
        learning_rate_init=learning_rate,
        max_iter=1 if solver == "sgd" else epochs,
        warm_start=solver == "sgd",
        shuffle=False,
        early_stopping=False,
        tol=0.0,
        n_iter_no_change=updates + 1,
        momentum=0.0,
        nesterovs_momentum=False,
    )
    runs = []
    for width in WIDTHS:
        for run, seed in enumerate(SEEDS, 1):
            estimator = RecordingMLPRegressor(
                hidden_layer_sizes=(width,), random_state=seed, **settings
            )
            if solver == "sgd":
                # Each public fit call makes exactly one minibatch update.
                # warm_start retains parameters; fit recreates the optimizer at
                # the requested learning rate. With momentum=0, plain SGD has
                # no optimizer state to retain. This reproduces model.mjs's
                # per-update schedule without modifying sklearn's optimizer.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"Stochastic Optimizer: Maximum iterations \(1\) reached.*",
                        category=ConvergenceWarning,
                    )
                    step = 0
                    for order in orders:
                        for start in range(0, len(x), batch_size):
                            batch = order[start:start + batch_size]
                            estimator.set_params(
                                learning_rate_init=learning_rate / (1 + step / 6000)
                            )
                            estimator.fit(x[batch, None], y[batch])
                            step += 1
                expected_calls = updates
            else:
                # partial_fit preserves Adam's moment estimates across epochs.
                for order in orders:
                    estimator.partial_fit(x[order, None], y[order])
                expected_calls = epochs
            model = export_model(estimator)
            prediction = estimator.predict(plot_x[:, None])
            np.testing.assert_allclose(
                predict_exported(model, plot_x), prediction, rtol=1e-13, atol=1e-13
            )
            train_mse = float(np.mean((estimator.predict(x[:, None]) - y) ** 2))
            grid_mse = float(np.mean((estimator.predict(metric_x[:, None]) - metric_y) ** 2))
            if len(estimator.loss_curve_) != expected_calls or estimator.t_ != epochs * len(x):
                raise AssertionError("Every run must complete the same training budget.")
            runs.append({
                "width": width,
                "run": run,
                "seed": seed,
                "trainMSE": train_mse,
                "gridMSE": grid_mse,
                "model": model,
                "predictions": prediction.tolist(),
                **initial_fields(estimator, plot_x, x, y),
                "epochs": epochs,
                "updates": updates,
                "observationVisits": int(estimator.t_),
                "converged": None,
                "stopReason": "fixed epoch budget",
            })
            print(f"width={width:2} run={run}: train MSE={train_mse:.8f}; grid MSE={grid_mse:.8f}", flush=True)

    return {
        "version": 1,
        "config": {
            "estimator": "sklearn.neural_network.MLPRegressor",
            "runtime": {
                "python": platform.python_version(),
                "scikitLearn": sklearn.__version__,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "source": "experiment.json",
            "sourceSHA256": hashlib.sha256(source_bytes).hexdigest(),
            "n": len(x),
            "widths": list(WIDTHS),
            "seeds": list(SEEDS),
            "runsPerWidth": len(SEEDS),
            "epochs": epochs,
            "batchSize": batch_size,
            "updatesPerRun": updates,
            "solver": solver,
            "learningRateSchedule": f"{learning_rate:g} / (1 + step / 6000)" if solver == "sgd" else f"Adam with initial rate {learning_rate:g}",
            "fitAPI": "One warm-start fit per minibatch; max_iter=1; no optimizer state because momentum=0" if solver == "sgd" else "One partial_fit call per epoch; optimizer state preserved",
            "estimatorSettings": settings,
            "estimatorDefaults": MLPRegressor().get_params(),
            "initialization": "scikit-learn default; random_state varies only the initial weights and biases",
            "sampling": "Identical external permutation each epoch for all widths and initializations; shuffle=False inside MLPRegressor",
            "batchOrderSeed": batch_order_seed,
            "batchOrderSHA256": hashlib.sha256(order_bytes).hexdigest(),
            "stopRule": "Exactly the stated number of epochs and minibatch updates; no validation or tolerance-based early stopping; convergence is not asserted",
            "objective": "Half mean squared error; no regularization",
            "metric": "Mean squared error on all 2,048 training observations",
            "gridMetric": "Mean squared error on 4,000 evenly spaced midpoints of [-1,1]",
            "plotGrid": "401 evenly spaced points on [-1,1], including endpoints",
        },
        "x": plot_x.tolist(),
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("adam", "sgd"))
    parser.add_argument("--add-initializations", action="store_true")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batch-order-seed", type=int, default=10001)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--output", type=Path, default=HERE / "initializations.json")
    args = parser.parse_args()
    if args.add_initializations:
        if args.solver is not None:
            parser.error("--add-initializations uses the solver and settings from the existing output.")
        result = add_initializations(json.loads(args.output.read_text()))
        args.output.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
        print(f"Added initial states to {len(result['runs'])} saved fits in {args.output}")
        return
    if args.solver is None:
        parser.error("--solver is required when generating new fits.")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("Epochs and batch size must be positive.")
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 0.001 if args.solver == "adam" else 0.05
    if learning_rate <= 0:
        parser.error("The learning rate must be positive.")
    result = generate(args.solver, args.epochs, args.batch_size, args.batch_order_seed, learning_rate)
    args.output.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"Saved {len(result['runs'])} fits to {args.output}")


if __name__ == "__main__":
    main()
