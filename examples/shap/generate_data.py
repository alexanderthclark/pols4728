#!/usr/bin/env python3
"""Rebuild exact local SHAP explanations for a transparent illustrative predictor.

The model and the reference profiles are designed teaching examples, not estimates
from people or observed earnings. The website reads the saved JSON and has no
Python or SHAP-package runtime dependency.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "docs" / "shap"
FEATURES = [
    {
        "id": "ability",
        "label": "Ability",
        "shortLabel": "Ability",
        "unit": "index",
        "decimals": 2,
        "domain": [0, 1],
        "description": "Illustrative ability or skill score, scaled from 0 to 1; not a measured person-level quantity.",
    },
    {
        "id": "neighborhood",
        "label": "Neighborhood opportunity",
        "shortLabel": "Neighborhood",
        "unit": "index",
        "decimals": 2,
        "domain": [-1, 1],
        "description": "Illustrative neighborhood opportunity score from −1 (adverse) to +1 (favorable).",
    },
    {
        "id": "experience",
        "label": "Experience",
        "shortLabel": "Experience",
        "unit": "index",
        "decimals": 2,
        "domain": [0, 1],
        "description": "Illustrative experience score, scaled from 0 to 1; not a number of years.",
    },
]
MODEL = {
    "schemaVersion": 1,
    "type": "IllustrativeEarningsInteraction",
    "intercept": 40,
    "coefficients": {"ability": 10, "abilityNeighborhood": 24, "experience": 4},
    "inputFeatures": ["ability", "neighborhood", "experience"],
    "featureDomains": {"ability": [0, 1], "neighborhood": [-1, 1], "experience": [0, 1]},
    "predictionUnit": "thousand dollars per year",
    "equation": "40 + 10A + 24AN + 4E",
    "fitted": False,
}
PROFILES = [
    {"id": "adverse-neighborhood", "name": "Adverse neighborhood", "values": [1, -1, 1]},
    {"id": "favorable-neighborhood", "name": "Favorable neighborhood", "values": [1, 1, 1]},
    {"id": "mixed-profile", "name": "Mixed profile", "values": [0.75, -0.5, 0.5]},
]


def predict(inputs):
    """Evaluate the single fixed predictor; there is no training or refitting."""
    inputs = np.asarray(inputs, dtype=np.float64)
    ability, neighborhood, experience = inputs[:, 0], inputs[:, 1], inputs[:, 2]
    return (
        MODEL["intercept"]
        + MODEL["coefficients"]["ability"] * ability
        + MODEL["coefficients"]["abilityNeighborhood"] * ability * neighborhood
        + MODEL["coefficients"]["experience"] * experience
    )


def explain_observation(background, x):
    """Enumerate all 8 coalitions and the 6 equally likely feature orders."""
    n_features = len(x)
    coalitions = []
    for mask in range(2**n_features):
        hybrid = background.copy()
        for feature in range(n_features):
            if mask & (1 << feature):
                hybrid[:, feature] = x[feature]
        predictions = predict(hybrid)
        coalitions.append(
            {"mask": mask, "value": float(predictions.mean()), "predictions": predictions.tolist()}
        )

    contributions_by_feature = [[] for _ in range(n_features)]
    orders = []
    for order in itertools.permutations(range(n_features)):
        before = 0
        marginals, steps = [], []
        for feature in order:
            after = before | (1 << feature)
            marginal = coalitions[after]["value"] - coalitions[before]["value"]
            marginals.append(marginal)
            steps.append(
                {"feature": feature, "beforeMask": before, "afterMask": after, "value": marginal}
            )
            contributions_by_feature[feature].append(marginal)
            before = after
        orders.append({"features": list(order), "marginals": marginals, "steps": steps})

    shap_values = np.array([np.mean(values) for values in contributions_by_feature])

    # Independently check the subset-weighted expression, rather than only orders.
    weighted_values = np.zeros(n_features, dtype=np.float64)
    for feature in range(n_features):
        for mask in range(2**n_features):
            if mask & (1 << feature):
                continue
            size = mask.bit_count()
            weight = (
                math.factorial(size)
                * math.factorial(n_features - size - 1)
                / math.factorial(n_features)
            )
            weighted_values[feature] += weight * (
                coalitions[mask | (1 << feature)]["value"] - coalitions[mask]["value"]
            )
    np.testing.assert_allclose(shap_values, weighted_values, rtol=0, atol=1e-12)
    return coalitions, orders, shap_values


def build(output_dir, check_only=False):
    import shap

    # All eight combinations of the declared endpoint reference values, weighted equally.
    # An unrevealed row's columns remain together when other columns are replaced.
    background = np.array(list(itertools.product([0, 1], [-1, 1], [0, 1])), dtype=np.float64)
    observations = []
    for profile in PROFILES:
        x = np.array(profile["values"], dtype=np.float64)
        coalitions, orders, shap_values = explain_observation(background, x)
        prediction = float(predict(x.reshape(1, -1))[0])
        np.testing.assert_allclose(coalitions[-1]["predictions"], prediction, rtol=0, atol=1e-12)
        np.testing.assert_allclose(coalitions[0]["value"] + shap_values.sum(), prediction, rtol=0, atol=1e-12)
        observations.append(
            {
                **profile,
                "prediction": prediction,
                "baseValue": coalitions[0]["value"],
                "coalitions": coalitions,
                "orders": orders,
                "shapValues": shap_values.tolist(),
            }
        )

    expected = {
        "adverse-neighborhood": {"prediction": 30, "values": [47, 52, 35, 28, 49, 54, 37, 30], "phi": [-1, -18, 2]},
        "favorable-neighborhood": {"prediction": 78, "values": [47, 52, 59, 76, 49, 54, 61, 78], "phi": [11, 18, 2]},
        "mixed-profile": {"prediction": 40.5, "values": [47, 49.5, 41, 40.5, 47, 49.5, 41, 40.5], "phi": [1, -7.5, 0]},
    }
    for observation in observations:
        reference = expected[observation["id"]]
        np.testing.assert_allclose(observation["prediction"], reference["prediction"], rtol=0, atol=1e-12)
        np.testing.assert_allclose([coalition["value"] for coalition in observation["coalitions"]], reference["values"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(observation["shapValues"], reference["phi"], rtol=0, atol=1e-12)

    # The official package gets the same fixed callable and all eight reference rows.
    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.Explainer(
        predict, masker, algorithm="exact", feature_names=[feature["label"] for feature in FEATURES]
    )
    inputs = np.array([profile["values"] for profile in PROFILES], dtype=np.float64)
    explanation = explainer(inputs, max_evals=8, batch_size=100)
    calculated = np.array([observation["shapValues"] for observation in observations])
    baseline = np.array([observation["baseValue"] for observation in observations])
    np.testing.assert_allclose(calculated, explanation.values, rtol=0, atol=1e-12)
    np.testing.assert_allclose(baseline, explanation.base_values, rtol=0, atol=1e-12)

    # Explicitly protect the pedagogical sign change in the default profile.
    default_values = [coalition["value"] for coalition in observations[0]["coalitions"]]
    assert default_values[1] - default_values[0] == 5
    assert default_values[3] - default_values[2] == -7
    assert calculated[0, 0] < 0 and calculated[1, 0] > 0

    data = {
        "schemaVersion": 1,
        "title": "Explain one prediction with SHAP",
        "predictionUnit": MODEL["predictionUnit"],
        "features": FEATURES,
        "defaultObservationId": PROFILES[0]["id"],
        "defaultComparison": {"feature": 0, "beforeMask": 2, "afterMask": 3},
        "dataset": {
            "name": "Illustrative earnings profiles",
            "type": "synthetic",
            "description": "Designed reference profiles and observations, not data collected from people.",
            "sourceRows": len(background),
            "backgroundConstruction": "All A,E in {0,1} and N in {−1,+1}; each of the eight combinations has weight 1/8.",
        },
        "model": {
            "type": MODEL["type"],
            "equation": MODEL["equation"],
            "fitted": False,
            "interpretation": "A transparent, fixed illustrative predictor of yearly earnings in $1,000 units.",
            "interaction": "The term 24AN deliberately lets neighborhood opportunity change how ability enters the prediction.",
            "limitations": "The equation and scores are invented for explanation. They are not fitted empirical estimates, validated income predictions, or claims about causal effects of ability or neighborhoods.",
        },
        "method": {
            "name": "Exact interventional SHAP over a finite illustrative background",
            "backgroundSize": len(background),
            "backgroundSelection": "All eight endpoint profiles, equally weighted.",
            "coalitionDefinition": "Fix the revealed columns at this profile's values in every background row; retain all other columns from that row; average the fixed predictor's outputs.",
            "modelRefittedForCoalitions": False,
            "backgroundRowsFilteredOnRevealedValues": False,
            "unknownColumnsSampledSeparately": False,
            "orderCount": math.factorial(3),
            "coalitionCount": 8,
            "unitsNote": "Predictions, coalition averages, marginal contributions, and SHAP values are in thousand dollars per year. The three inputs are illustrative normalized scores.",
        },
        "validation": {
            "shapPackage": shap.__version__,
            "shapAlgorithm": "ExactExplainer with an Independent masker containing all eight background profiles",
            "maxShapAgreementError": float(np.max(np.abs(calculated - explanation.values))),
            "maxBaselineAgreementError": float(np.max(np.abs(baseline - explanation.base_values))),
            "maxAdditivityError": max(
                abs(observation["baseValue"] + sum(observation["shapValues"]) - observation["prediction"])
                for observation in observations
            ),
            "defaultAbilityMarginalBeforeAnyReveal": default_values[1] - default_values[0],
            "defaultAbilityMarginalAfterNeighborhood": default_values[3] - default_values[2],
            "defaultExpectedShapValues": [-1, -18, 2],
        },
        "buildVersions": {"python": platform.python_version(), "numpy": np.__version__, "shap": shap.__version__},
        "background": [
            {
                "id": f"background-{index + 1}",
                "name": f"Reference profile {index + 1}",
                "values": values.tolist(),
                "weight": 1 / len(background),
                "modelPrediction": float(predict(values.reshape(1, -1))[0]),
            }
            for index, values in enumerate(background)
        ],
        "observations": observations,
    }
    serialized = {
        "data.json": json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        "model.json": json.dumps(MODEL, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    }
    if check_only:
        for filename, contents in serialized.items():
            existing = json.loads((output_dir / filename).read_text(encoding="utf-8"))
            generated = json.loads(contents)
            if filename == "data.json":
                existing.pop("buildVersions", None)
                generated.pop("buildVersions", None)
            assert existing == generated, f"Saved {filename} differs from this rebuild."
        print("Saved JSON matches the deterministic rebuild; all numerical checks passed.")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename, contents in serialized.items():
            (output_dir / filename).write_text(contents, encoding="utf-8")
        print(f"Saved {len(observations)} profiles, eight coalitions each, and {len(background)} background profiles to {output_dir}.")
    print(json.dumps(data["validation"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="Check committed data without writing files.")
    arguments = parser.parse_args()
    build(arguments.output_dir, check_only=arguments.check)
