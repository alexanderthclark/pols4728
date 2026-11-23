"""Reproduce the fixed-setting XGBoost diagnostics for the bear lesson.

This program evaluates XGBoost only. It does not run TabPFN or tune models.
All variants within a seed share labels, noise draws, and train/test indices.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
import sklearn
import xgboost
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from bear_simulation import simulate


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CASE_LABELS = {
    "signal_only": "Bear type and strategy only",
    "continuous_noise": "Add continuous noise columns",
    "binary_noise": "Binarize the same noise columns",
    "interaction_feature": "Add the bear-type × strategy product",
    "two_bins": "Keep continuous noise; use two histogram bins",
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def root_description(model: XGBClassifier, case_id: str) -> tuple[str, str]:
    root = json.loads(model.get_booster().get_dump(dump_format="json")[0])
    split = root.get("split")
    if split is None:
        return "leaf", "leaf"
    index = int(split.removeprefix("f"))
    if index < 2:
        return ["brown_bear", "lie_down"][index], "signal"
    if case_id == "interaction_feature" and index == model.n_features_in_ - 1:
        return "brown_bear_times_lie_down", "product"
    return f"noise_{index - 1}", "noise"


def moments(values: list[float]) -> tuple[float, float | None]:
    mean = float(np.mean(values))
    # A singleton run has no estimated between-seed standard deviation.
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else None
    return mean, sd


def summarize(records: list[dict], depths: list[int]) -> tuple[list, list]:
    cases, deltas = [], []
    for depth in depths:
        baseline = {
            r["seed"]: r for r in records
            if r["depth"] == depth and r["id"] == "continuous_noise"
        }
        for case_id, label in CASE_LABELS.items():
            rows = [r for r in records if r["depth"] == depth and r["id"] == case_id]
            mean_auc, sd_auc = moments([r["auc"] for r in rows])
            mean_brier, sd_brier = moments([r["brier"] for r in rows])
            roots = Counter(r["first_root_kind"] for r in rows)
            cases.append({
                "id": case_id, "label": label, "depth": depth,
                "max_bin": rows[0]["max_bin"], "n_seeds": len(rows),
                "mean_auc": mean_auc, "sd_auc": sd_auc,
                "mean_brier": mean_brier, "sd_brier": sd_brier,
                "root_signal_fraction": roots["signal"] / len(rows),
                "root_noise_fraction": roots["noise"] / len(rows),
                "root_product_fraction": roots["product"] / len(rows),
                "root_leaf_fraction": roots["leaf"] / len(rows),
            })
            paired = [{
                "seed": r["seed"],
                "auc_delta": r["auc"] - baseline[r["seed"]]["auc"],
                "brier_delta": r["brier"] - baseline[r["seed"]]["brier"],
            } for r in rows]
            auc_mean, auc_sd = moments([r["auc_delta"] for r in paired])
            brier_mean, brier_sd = moments([r["brier_delta"] for r in paired])
            deltas.append({
                "id": case_id, "depth": depth, "baseline": "continuous_noise",
                "mean_auc_delta": auc_mean, "sd_auc_delta": auc_sd,
                "mean_brier_delta": brier_mean, "sd_brier_delta": brier_sd,
                "per_seed": paired,
            })
    return cases, deltas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=positive_int, default=10_000)
    parser.add_argument("--noise-columns", type=positive_int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    parser.add_argument("--depths", type=positive_int, nargs="+", default=[2, 6])
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument(
        "--web-output", type=Path,
        default=REPO / "docs/tabular-foundation-models/bear-results.mjs",
        help="ES module generated from the same results as the JSON and CSV.",
    )
    args = parser.parse_args()
    if args.n < 20:
        parser.error("--n must be at least 20 for a train/test experiment")
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("--seeds must be distinct nonnegative integers")
    if len(set(args.depths)) != len(args.depths):
        parser.error("--depths must be distinct")

    CASE_LABELS["continuous_noise"] = f"Add {args.noise_columns} continuous noise columns"
    records = []
    for seed in args.seeds:
        variants, y = simulate(seed, args.n, n_noise=args.noise_columns)
        train, test = train_test_split(
            np.arange(args.n), test_size=0.2, random_state=seed
        )
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            raise ValueError(f"Seed {seed} produced a split with only one outcome class")
        for case_id, X in variants.items():
            bins = 2 if case_id == "two_bins" else 256
            for depth in args.depths:
                model = XGBClassifier(
                    n_estimators=200, max_depth=depth, learning_rate=0.1,
                    tree_method="hist", max_bin=bins, n_jobs=2,
                    random_state=seed, eval_metric="logloss",
                )
                model.fit(X[train], y[train])
                probability = model.predict_proba(X[test])[:, 1]
                root_feature, root_kind = root_description(model, case_id)
                records.append({
                    "seed": seed, "id": case_id, "depth": depth,
                    "max_bin": bins, "n_train": len(train), "n_test": len(test),
                    "n_features": X.shape[1],
                    "auc": float(roc_auc_score(y[test], probability)),
                    "brier": float(brier_score_loss(y[test], probability)),
                    "first_root_feature": root_feature, "first_root_kind": root_kind,
                })
        print(f"Completed seed {seed}", flush=True)

    cases, deltas = summarize(records, args.depths)
    p_match = float(1 / (1 + np.exp(-1)))
    metadata = {
        "experiment": "Bear survival: interaction and irrelevant-feature diagnostics",
        "model": "XGBoost", "tabpfn_evaluated": False,
        "n_total": args.n, "n_train": records[0]["n_train"],
        "n_signal_features": 2, "n_noise": args.noise_columns, "n_features": 2 + args.noise_columns,
        "n_test": records[0]["n_test"], "seeds": args.seeds,
        "depths": args.depths, "n_estimators": 200, "learning_rate": 0.1,
        "tree_method": "hist", "default_max_bin": 256, "n_jobs": 2,
        "test_size": 0.2, "split_stratified": False,
        "split_random_state": "same integer as data seed",
        "tuned": False,
        "uncertainty": "SD is the sample standard deviation across seeds, not a confidence interval.",
        "dgp": f"B,A independently Bernoulli(0.5); Z1,...,Z{args.noise_columns} independently Uniform(0,1); E independently standard Logistic; Y=1[-1+2*1[B=A]+E>0].",
        "noise_extension": "Original B, A, first three noise columns and logistic error preserve their original draw order. Extra distractors use an independent SeedSequence([seed,4728,20261123]) stream, drawn column-wise.",
        "p_survival_match": p_match, "p_survival_mismatch": 1 - p_match,
        "population_marginal_survival": 0.5,
        "paired_design": "Within each seed, all cases share B, A, continuous noise draws, Y, and train/test indices.",
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__,
            "scipy": version("scipy"), "joblib": version("joblib"), "threadpoolctl": version("threadpoolctl"),
        },
    }
    result = {"metadata": metadata, "cases": cases, "records": records, "paired_deltas": deltas}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    (args.output_dir / "bear-results.json").write_text(text + "\n", encoding="utf-8")
    with (args.output_dir / "bear-records.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    args.web_output.parent.mkdir(parents=True, exist_ok=True)
    args.web_output.write_text(
        "// Generated by examples/tabular-foundation-models/bear_diagnostics.py.\n"
        "// Fixed-setting XGBoost diagnostics; no TabPFN run or hyperparameter tuning.\n"
        f"export const bearResults = {text};\n", encoding="utf-8",
    )
    for case in cases:
        print(f"depth={case['depth']} {case['id']}: AUC={case['mean_auc']:.4f}, Brier={case['mean_brier']:.4f}")


if __name__ == "__main__":
    main()
