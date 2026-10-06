#!/usr/bin/env python3
"""Fit and explain a standardized, synthetic Bread and Peace teaching example.

The two substantive predictors follow Hibbs; the rows and coefficients do not
reproduce his historical data or estimates. No election forecasts are made.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import shap

OUTPUT = Path(__file__).resolve().parents[2] / "docs" / "shap" / "bread-peace.json"


def build():
    # Empirical SD uses denominator n, consistently for all three columns.
    # Six opposite-sign rows and two same-sign rows give correlation -0.5.
    inputs = np.array([
        [-1, -1], [-1, 1], [-1, 1], [-1, 1],
        [1, -1], [1, -1], [1, -1], [1, 1],
    ], dtype=float)
    residual = np.array([0, -1, 0, 1, -1, 0, 1, 0], dtype=float) / np.sqrt(2)
    outcome = inputs @ np.array([0.5, -0.5]) + residual
    design = np.column_stack([np.ones(len(inputs)), inputs])
    fitted = np.linalg.lstsq(design, outcome, rcond=None)[0]
    np.testing.assert_allclose(fitted, [0, 0.5, -0.5], atol=1e-12, rtol=0)
    # Remove harmless floating-point fit noise from the saved teaching equation.
    intercept, income_coefficient, fatalities_coefficient = np.round(fitted, 12)
    coefficients = np.array([income_coefficient, fatalities_coefficient])

    def predict(rows):
        return intercept + np.asarray(rows) @ coefficients

    standardized = np.column_stack([inputs, outcome])
    np.testing.assert_allclose(standardized.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(standardized.std(axis=0), 1, atol=1e-12)
    correlations = np.corrcoef(standardized.T)
    reduced = np.linalg.lstsq(inputs[:, :1], outcome, rcond=None)[0][0]
    reduced = float(np.round(reduced, 12))
    fatalities_only = float(np.round(np.linalg.lstsq(inputs[:, 1:], outcome, rcond=None)[0][0], 12))
    np.testing.assert_allclose(reduced, correlations[0, 2], atol=1e-12)
    np.testing.assert_allclose(fatalities_only, correlations[1, 2], atol=1e-12)
    omitted_term = float(correlations[0, 1] * fatalities_coefficient)
    np.testing.assert_allclose(reduced, income_coefficient + omitted_term, atol=1e-12)
    np.testing.assert_allclose(fatalities_only, fatalities_coefficient + correlations[0, 1] * income_coefficient, atol=1e-12)

    observations = []
    profiles = [
        ("growth-low-fatalities", "Growth, lower fatalities", [1, -1]),
        ("growth-high-fatalities", "Growth, higher fatalities", [1, 1]),
        ("weak-growth-high-fatalities", "Weak growth, higher fatalities", [-1, 1]),
    ]
    for profile_id, name, values in profiles:
        x = np.array(values, dtype=float)
        coalitions = []
        for mask in range(4):
            hybrid = inputs.copy()
            for feature in range(2):
                if mask & (1 << feature):
                    hybrid[:, feature] = x[feature]
            predictions = predict(hybrid)
            coalitions.append({"mask": mask, "value": float(predictions.mean()),
                               "predictions": predictions.tolist()})
        orders = []
        marginals_by_feature = [[], []]
        for order in itertools.permutations(range(2)):
            before = 0
            marginals = []
            for feature in order:
                after = before | (1 << feature)
                value = coalitions[after]["value"] - coalitions[before]["value"]
                marginals.append(value)
                marginals_by_feature[feature].append(value)
                before = after
            orders.append({"features": list(order), "marginals": marginals})
        values_shap = np.mean(marginals_by_feature, axis=1)
        np.testing.assert_allclose(values_shap, coefficients * x, atol=1e-12)
        prediction = float(predict(x.reshape(1, -1))[0])
        np.testing.assert_allclose(sum(values_shap), prediction, atol=1e-12)
        observations.append({
            "id": profile_id, "name": name, "values": values,
            "prediction": prediction, "baseValue": coalitions[0]["value"],
            "reducedPrediction": float(reduced * x[0]),
            "coalitions": coalitions, "orders": orders,
            "shapValues": values_shap.tolist(),
        })

    masker = shap.maskers.Independent(inputs, max_samples=len(inputs))
    explanation = shap.Explainer(predict, masker, algorithm="exact")(
        np.array([row["values"] for row in observations]), max_evals=4
    )
    agreement = float(np.max(np.abs(
        explanation.values - np.array([row["shapValues"] for row in observations])
    )))
    baseline_error = float(np.max(np.abs(explanation.base_values)))
    np.testing.assert_allclose(agreement, 0, atol=1e-12)
    np.testing.assert_allclose(baseline_error, 0, atol=1e-12)

    return {
        "schemaVersion": 1,
        "title": "Stylized Bread and Peace OLS",
        "dataSource": "Synthetic teaching rows, not historical elections or Hibbs's estimates.",
        "source": {
            "author": "Douglas A. Hibbs",
            "title": "Obama's Reelection Prospects under Bread and Peace Voting in the 2012 US Presidential Election",
            "url": "https://www.cambridge.org/core/journals/ps-political-science-and-politics/article/abs/obamas-reelection-prospects-under-bread-and-peace-voting-in-the-2012-us-presidential-election/085A6DB3D1D1310250B5E2566AB352CF",
        },
        "features": [
            {"id": "income", "label": "Real income growth", "shortLabel": "Income",
             "description": "Standardized real disposable income growth; +1 is one empirical SD above the reference mean."},
            {"id": "fatalities", "label": "War fatalities", "shortLabel": "Fatalities",
             "description": "Standardized war fatalities; -1 is one empirical SD below the reference mean, not a negative count."},
        ],
        "standardization": {"means": [0, 0, 0], "standardDeviations": [1, 1, 1],
                            "columns": ["income", "fatalities", "vote"], "ddof": 0},
        "model": {"fitted": True, "estimator": "OLS on synthetic standardized rows",
                  "intercept": float(intercept), "coefficients": coefficients.tolist(),
                  "equation": "y-hat(x) = 0.5 × income - 0.5 × fatalities",
                  "predictionUnit": "SD of incumbent-party two-party vote share"},
        "refit": {"featureCorrelation": float(correlations[0, 1]),
                  "incomeVoteCorrelation": float(correlations[0, 2]),
                  "fatalitiesVoteCorrelation": float(correlations[1, 2]),
                  "incomeOnlyCoefficient": reduced, "fatalitiesOnlyCoefficient": fatalities_only,
                  "omittedVariableTerm": omitted_term},
        "background": [
            {"id": f"reference-{index + 1}", "values": values.tolist(),
             "outcome": float(outcome[index]), "modelPrediction": float(predict(values.reshape(1, -1))[0])}
            for index, values in enumerate(inputs)
        ],
        "method": {"definition": "finite-background interventional replacement",
                   "backgroundSize": len(inputs), "weightPerRow": 1 / len(inputs)},
        "defaultObservationId": profiles[0][0],
        "observations": observations,
        "validation": {"maxShapAgreementError": agreement,
                       "maxBaselineAgreementError": baseline_error,
                       "shapVersion": shap.__version__},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Rebuild and compare without writing.")
    args = parser.parse_args()
    data = build()
    if args.check:
        if json.loads(OUTPUT.read_text()) != data:
            raise SystemExit("Saved Bread and Peace data differ from the reproducible build.")
        print("Bread and Peace OLS, standardization, omitted-variable identity, and exact SHAP checks passed.")
    else:
        OUTPUT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        print(f"Wrote {OUTPUT}")
