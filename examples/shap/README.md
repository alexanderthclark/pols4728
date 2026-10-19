# Reproduce the SHAP story

The [SHAP page](../../docs/shap/) begins with a synthetic Bread and Peace
OLS warm-up, then explains a fixed illustrative predictor of yearly earnings.
Both examples use constructed data, rather than empirical election observations
or measured people. Their model outputs and SHAP calculations are reproducible.
The unchanged opening title scene leads into 30 teaching frames: a
21-frame explanation through the local waterfall, including the early definition
of background averaging and the OLS recap,
followed by nine frames that construct a beeswarm and derive global importance.

The browser reads saved JSON for the original profiles and computes the summary
cohort's explanations from the same saved model and reference background.
Python and the SHAP package rebuild and independently verify the saved data;
the page requires no local software installation.

The first five formula frames define the prediction game before the OLS warm-up.
Frame 4 shows `v_x(S) = (1/B) Σ_{b=1}^B ŷ(x_S, z^(b)_−S)`: `B` is the number of
reference rows, `z^(b)` is background row `b`, and `x_S` fixes the included inputs
to the observation being explained. Excluded inputs retain their joint values
from each overall reference row, without conditioning on the included inputs.
The model stays fixed while predictions are averaged. Frame 5 defines the
marginal contribution `v_x(S ∪ {i}) − v_x(S)` and connects its average across
orders to feature credit. Frame 13 repeats the general background-average
definition before labeling prediction at mean excluded inputs as a
linear model shortcut. Overall background means, rather than conditional means,
are used in that shortcut; nonlinear predictors need not give the same average.

## Standardized Bread and Peace OLS warm-up

Douglas Hibbs's Bread and Peace model supplies the two substantive predictors:
real disposable income growth and war fatalities. We call income growth `bread`
and reverse-code standardized war fatalities as `peace`: higher peace means
fewer war deaths. The warm-up uses eight synthetic training rows and a direct
OLS fit; it does not reproduce Hibbs's
historical data, income-growth time weighting, coefficients, or forecasts.
Its [source reference](https://link.springer.com/article/10.1023/A:1005292312412)
describes the underlying Bread and Peace interpretation.

All three columns (bread, peace, and observed vote) have mean zero and
empirical SD one, using denominator `n`. Six rows have same-sign predictors
and two have opposite-sign predictors, yielding feature correlation +0.5. Outcomes
combine `0.5 × bread + 0.5 × peace` with an orthogonal residual of variance
0.25. An OLS fit recovers zero intercept and slopes +0.5 and +0.5. The predictions
are in outcome SD units; they are not standardized again after fitting.

The story presents the three correlations directly: bread–peace +0.5,
bread–vote +0.75, and peace–vote +0.75. With unit variances, the univariate
slopes equal the feature–vote correlations. The bivariate slopes are
`(r_bread,vote − rho × r_peace,vote) / (1 − rho²)` and its counterpart for
peace. These recover +0.5 and +0.5 without showing a data table.

A bread-only OLS refit has slope 0.75, equal to the bread–vote correlation.
This is `0.5 + (0.5)(0.5)`: the retained full-model slope plus feature
correlation times the omitted feature's slope. The reduced model is used only
to demonstrate that refitting is a different calculation from our SHAP game.
Peace-only OLS has slope +0.75, likewise its correlation with vote.

The same eight input rows form the equally weighted background for exact
interventional SHAP. For a default synthetic election with bread 1 and peace
1, the four coalition values are `[0, 0.5, 0.5, 1]` for masks 0, 1, 2, 3.
Bread contributes +0.5 in both revealing orders; peace also contributes
+0.5. The other saved profiles are `[1, -1]` and `[-1, -1]`, with predictions 0
and −1 and corresponding contributions `[0.5, -0.5]` and `[-0.5, -0.5]`.
Coefficients and the background stay fixed throughout each explanation.
The peace recoding preserves every prediction and SHAP contribution from the
equivalent income-growth/war-fatalities specification; it changes the feature's
sign and name together. Existing profile IDs retain their original wording to
preserve saved references.

`generate_bread_peace.py` fits OLS, checks centering, standard deviations, residual
orthogonality, the omitted-variable identity, and exact agreement with the SHAP
package. It saves training outcomes, fitted predictions, the reduced-model
comparison, all four coalitions, both orders, and three profiles in
`docs/shap/bread-peace.json`. Coalition values average predictions, never training
outcomes or prediction errors. For this additive model, inserting the zero
background means gives the same coalition averages as averaging the rows.
The browser uses that equivalence to write both revealing orders directly as
differences of bivariate predictions, then averages both features' marginals.
The underlying synthetic rows support numerical verification and do not appear
in the OLS story; the earnings example retains its hybrid-row tables.

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

## From the waterfall to beeswarm and global importance

The nine appended frames move each waterfall contribution to a common zero,
replace it with a dot, add people, spread overlapping dots vertically, introduce
input-value color, demonstrate signed cancellation, take absolute values,
average those values into bars, and finally rank the bars. Scrolling reaches
every teaching point. Vertical position only prevents overlap; horizontal
position measures the signed SHAP contribution. Color represents low-to-high
input value within each feature, rather than the sign of its contribution.

The beeswarm summarizes 125 equally weighted constructed profiles, all
combinations of ability and experience in `{0, .25, .5, .75, 1}` and neighborhood
opportunity in `{−1, −.5, 0, .5, 1}`. This cohort contains all three saved
profiles. It is separate from the eight endpoint reference profiles used for
every person's explanation. The predictor, those eight reference rows, and the
baseline of 47 remain fixed throughout.

`docs/shap/aggregate.mjs` reads the coefficients supplied from `model.json`,
enumerates all eight coalitions against the original background, and averages
the six revealing-order marginals for each profile. It computes the same exact
finite-background interventional game in the browser. Global importance is the
mean of the absolute SHAP contributions across the 125 explained profiles:

| Feature | Mean signed SHAP | Mean absolute SHAP |
| --- | --- | --- |
| Ability | 0 | 3.24 |
| Neighborhood opportunity | 0 | 7.2 |
| Experience | 0 | 1.2 |

All values are in $1,000/year. Taking absolute values before averaging retains
magnitude that the signed averages cancel. The final ranking is neighborhood
opportunity, ability, then experience. These summaries describe this predictor,
background, and cohort; changing the explained population or reference
background can change the ranking. The bars discard the direction, variation,
and interaction patterns that the beeswarm preserves. Contributions to these
constructed predictions are not causal effects.

Selecting a dot in the colored beeswarm opens that person's complete waterfall.
The method note also provides a profile dropdown and a waterfall button for
optional exploration.

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

The existing JavaScript SHAP tests independently check the equation, all hybrid
inputs, coalition averages, subset factorial weights, revealing orders, and
additivity. `tests/shap-aggregate.test.mjs` additionally checks exact agreement
with every stored profile, the balanced 125-profile grid, and the shared
baseline. Its independent closed-form SHAP values are
`(ability − .5) × (10 + 12 × neighborhood)`,
`12 × neighborhood × (ability + .5)`, and `4 × (experience − .5)`.
It verifies those values for every person, signed cancellation, mean absolute
importance, ranking, and the effect of choosing a different explained cohort
while keeping the same reference background.

From the repository root:

```sh
node --test tests/shap-local.test.mjs tests/shap-aggregate.test.mjs tests/shap-ols.test.mjs
```

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
