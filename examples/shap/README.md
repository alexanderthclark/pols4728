# Reproduce the local SHAP story

The [local SHAP page](../../docs/shap/) begins with a synthetic Bread and Peace
OLS warm-up, then explains a fixed illustrative predictor of yearly earnings.
Both examples use constructed data, rather than empirical election observations
or measured people. Their model outputs and SHAP calculations are reproducible.

The browser reads saved JSON. Python and the SHAP package are used only to rebuild
and verify the data; students do not need Python to read the page. This example
contains only local explanations, with no global importance or beeswarm plots.

## Standardized Bread and Peace OLS warm-up

Douglas Hibbs's Bread and Peace model supplies the two substantive predictors:
real disposable income growth and war fatalities. The warm-up uses eight
synthetic training rows and a direct OLS fit; it does not reproduce Hibbs's
historical data, income-growth time weighting, coefficients, or forecasts.
Its [source reference](https://www.cambridge.org/core/journals/ps-political-science-and-politics/article/abs/obamas-reelection-prospects-under-bread-and-peace-voting-in-the-2012-us-presidential-election/085A6DB3D1D1310250B5E2566AB352CF)
describes the underlying Bread and Peace interpretation.

All three columns (income, fatalities, and observed vote) have mean zero and
empirical SD one, using denominator `n`. Six rows have opposite-sign predictors
and two have same-sign predictors, yielding feature correlation −0.5. Outcomes
combine `0.5 × income − 0.5 × fatalities` with an orthogonal residual of variance
0.25. An OLS fit recovers zero intercept and slopes +0.5 and −0.5. The predictions
are in outcome SD units; they are not standardized again after fitting.

An income-only OLS refit has slope 0.75, equal to the income–vote correlation.
This is `0.5 + (−0.5)(−0.5)`: the retained full-model slope plus feature
correlation times the omitted feature's slope. The reduced model is used only
to demonstrate that refitting is a different calculation from our SHAP game.

The same eight input rows form the equally weighted background for exact
interventional SHAP. For a default synthetic election with income 1 and fatalities
−1, the four coalition values are `[0, 0.5, 0.5, 1]` for masks 0, 1, 2, 3.
Income contributes +0.5 in both revealing orders; fatalities also contribute
+0.5. The other saved profiles are `[1, 1]` and `[-1, 1]`, with predictions 0
and −1 and corresponding contributions `[0.5, -0.5]` and `[-0.5, -0.5]`.
Coefficients and the background stay fixed throughout each explanation.

`generate_bread_peace.py` fits OLS, checks centering, standard deviations, residual
orthogonality, the omitted-variable identity, and exact agreement with the SHAP
package. It saves training outcomes, fitted predictions, the reduced-model
comparison, all four coalitions, both orders, and three profiles in
`docs/shap/bread-peace.json`. Browser tables average predictions, never training
outcomes or prediction errors. For this additive model, inserting the zero
background means gives the same coalition averages as averaging the rows.

```sh
examples/shap/.venv/bin/python examples/shap/generate_bread_peace.py
examples/shap/.venv/bin/python examples/shap/generate_bread_peace.py --check
```

## Fixed predictor and input scores

The predictor is:

```text
f(x) = 40 + 10 × ability + 24 × ability × neighborhood + 4 × experience
```

Its output is in thousands of dollars per year. The features are named `ability`,
`neighborhood`, and `experience`; `neighborhood` denotes neighborhood opportunity.
These are illustrative normalized scores. `S` and `F` denote sets of features,
with `F = {ability, neighborhood, experience}`, and `i` denotes the feature
joining `S`. The scores have the following domains:

| Input | Domain | Interpretation |
| --- | --- | --- |
| Ability | 0 to 1 | Low to high illustrative ability or skill. |
| Neighborhood opportunity | −1 to +1 | Adverse to favorable illustrative opportunity. |
| Experience | 0 to 1 | Low to high illustrative experience; not a count of years. |

The interaction coefficient of 24 deliberately makes the role of ability depend
on neighborhood opportunity. With `neighborhood = −1`, the ability terms become
`10 × ability − 24 × ability = −14 × ability`; increasing ability lowers this
predictor's output. With `neighborhood = +1`, they become `34 × ability`.
This is a property of the stipulated teaching model,
not a claim about real earnings or causal effects.

There is no training, train/test split, refitting, or predictive validation in the
earnings interaction example. Its JSON specifies `fitted: false` and stores the exact equation and
coefficients so readers can inspect every result.

## Background and profiles to explain

The reference background is exactly eight profiles: all combinations of
`ability ∈ {0, 1}`, `neighborhood ∈ {−1, +1}`, and `experience ∈ {0, 1}`.
Each has weight 1/8.
Their average prediction is 47, or $47,000 per year. This is the average over the
declared reference profiles, not an observed population average.

The three profiles to explain are:

| Profile | Ability | Neighborhood | Experience | Prediction | SHAP values for ability, neighborhood, experience |
| --- | --- | --- | --- | --- | --- |
| Adverse neighborhood | 1 | −1 | 1 | 30 | −1, −18, +2 |
| Favorable neighborhood | 1 | +1 | 1 | 78 | +11, +18, +2 |
| Mixed profile | 0.75 | −0.5 | 0.5 | 40.5 | +1, −7.5, 0 |

Predictions and contributions in this table are in $1,000 units. The default
profile is `adverse-neighborhood`. Its explanation satisfies
`47 − 1 − 18 + 2 = 30`.

## How a marginal contribution is calculated

For the profile `x` being explained, let `S` denote the features already revealed.
For each background profile `z`, create a hybrid input: use `x`'s values for
columns in `S` and that original reference row's values for all other columns.
Evaluate the same fixed predictor on each of the eight inputs, then average:

```text
v_x(S) = (1/8) × sum over reference rows z of f(x_S, z_-S)
```

For the default profile, revealing neighborhood means fixing `neighborhood = −1` in every
row while retaining each row's ability and experience. The eight resulting
predictions average to 35. Revealing ability too fixes `ability = 1` in every row;
the new predictions average to 28. Ability's marginal contribution after
neighborhood is therefore `28 − 35 = −7`, or −$7,000.

When nothing else is revealed, revealing ability changes the average from 47 to
52, a marginal of +5. Thus the same profile has a positive ability marginal in
one context and a negative marginal in another.

This calculation averages over unrevealed inputs using the stated reference
profiles. It does not set an unrevealed input to zero, filter background rows to
match a revealed input, or fit a new predictor without an input. All unrevealed
values within a row remain together. The values fixed from `x` are combined with
those reference-row values using the finite-background interventional definition.

All eight coalitions are enumerated. Each of the six feature orders records the
change in the coalition average when each feature is added. A feature's SHAP
value is the average of its six recorded marginals. For the default profile,
ability enters before neighborhood in three orders, contributing +5 each time;
it enters after neighborhood in three orders, contributing −7 each time. Its
SHAP value is `(3 × 5 + 3 × −7) / 6 = −1`.

With no features revealed, the value is the reference average, 47. With all
features revealed, every hybrid input is the explained profile, so the average
is that profile's prediction. The SHAP contributions sum to the difference.
These calculations are exact for this equation and this equally weighted
background. Changing the reference profiles would generally change the baseline
and contributions. Display labels may be rounded; the JSON stores unrounded
floating-point results.

## Rebuild and verify

The committed numerical build used Python 3.12.14, NumPy 2.2.6, and SHAP 0.49.1.
From the repository root:

```bash
python3 -m venv examples/shap/.venv
examples/shap/.venv/bin/python -m pip install -r examples/shap/requirements.txt
examples/shap/.venv/bin/python examples/shap/generate_data.py
examples/shap/.venv/bin/python examples/shap/generate_data.py --check
```

The generated files are:

- `docs/shap/data.json`: feature and method metadata, all eight reference profiles,
  the three observations, every coalition average and per-reference prediction,
  all six orders, and exact SHAP contributions.
- `docs/shap/model.json`: the fixed equation, coefficients, input order, domains,
  and output units.

The official package check uses the same prediction callable and all eight
reference rows:

```python
masker = shap.maskers.Independent(background, max_samples=len(background))
explainer = shap.Explainer(predict, masker, algorithm="exact")
explanation = explainer(profiles, max_evals=8)
```

`Independent` retains the joint values of all unrevealed inputs in each original
reference row. Setting `max_samples` to the background size prevents an extra
subsampling step.

The generator checks all coalition values against the stipulated equation, the
subset-weighted SHAP formula independently of the six orders, additivity, the
ability marginal's sign change, and agreement with the official SHAP package's
values and baseline. Its validation results and package versions are saved in
`data.json`. `--check` rebuilds the results and compares them with the saved JSON
without writing files.

## Export a standard waterfall in the course style

`style.py` supplies `course_waterfall(explanation)`, a reusable wrapper around
the standard `shap.plots.waterfall(..., show=False)` call. It reads
`design/tokens.json` for the white background, course colors, Computer Modern
preview typography, and print sizes. Blue shows positive contributions and rust
negative contributions, with signed numbers and arrow directions retaining their
meaning. Feature ordering, contribution widths, and the baseline and prediction
markers remain the standard SHAP waterfall format.

The helper applies colors through Matplotlib's artist methods after SHAP creates
the plot. It restores Matplotlib settings and interactive state, and does not
modify SHAP's private styling configuration. Matplotlib 3.11.2 is pinned in the
example requirements and was used to verify the export.

To export the default profile as SVG from the repository root:

```bash
examples/shap/.venv/bin/python examples/shap/style.py --output /tmp/shap-waterfall.svg
```

Use `--profile favorable-neighborhood` or `--profile mixed-profile` to select
another saved observation. The output filename can end in `.svg`, `.png`, or
`.pdf`. Generated figures do not need to be committed to the repository.
The x axis remains in $1,000 per year, matching the saved explanation values.

To use the helper with a one-row `shap.Explanation` from another regression model,
import `course_waterfall` from `examples/shap/style.py` and pass the explanation:

```python
from style import course_waterfall

axis = course_waterfall(explanation, title="One observation", output_label="Your model output units")
axis.figure.savefig("waterfall.svg", bbox_inches="tight")
```

The module must be on the Python import path, for example when running from
`examples/shap/`. Specify the appropriate x-axis units for another model.

The equation and illustrative profiles are authored for this course example.
The SHAP calculation follows the documented package method:
[ExactExplainer](https://shap.readthedocs.io/en/latest/generated/shap.ExactExplainer.html),
[Independent masker](https://shap.readthedocs.io/en/latest/generated/shap.maskers.Independent.html),
and [waterfall plots](https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/waterfall.html).
