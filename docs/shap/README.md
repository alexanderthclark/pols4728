# Explaining one prediction with SHAP

This six-frame scrolling story assumes readers already know Shapley values.
It shows how a revealed feature changes the value of a group in a local SHAP
explanation: substitute the observation's values into background rows, evaluate
the same prediction function, average its outputs, then compare those averages.

The illustrative earnings predictor is `f(A,N,E) = 40 + 10A + 24AN + 4E`, in
thousands of dollars per year. Ability and experience range from 0 to 1;
neighborhood opportunity ranges from −1 to +1. The interaction can outweigh
ability's positive main term in an adverse context. The equation and profiles
are constructed for teaching; they are not empirical estimates or causal claims.
There are three features. The interaction is computed inside the predictor.

The eight equally weighted reference profiles cover every endpoint combination.
All eight rows appear in the main tables. Included columns are fixed to the
selected person; excluded columns retain the joint values from each background
row. The before/after output columns and their means show the marginal directly.
No feature is removed from the model or zeroed to represent absence.

The default profile has `(A,N,E) = (1,−1,1)`. Its baseline is 47 and prediction is
30. Ability's marginal is +5 before neighborhood is revealed and −7 after it.
Each occurs in three of the six orders, giving a final ability SHAP value of −1.
The complete explanation is `47 − 1 − 18 + 2 = 30`. Alternate favorable and mixed
profiles use the same model and background.

## Story and controls

1. Select a person and see the three inputs and prediction equation.
2. Average predictions over the unrevealed background.
3. Fix ability in every row, predict again, and compare with the baseline.
4. Start with neighborhood fixed, add ability, and compare the two averages.
5. Average across contexts; select a feature and optionally inspect all six orders.
6. Assemble the final contributions in a course-styled standard waterfall.

The concluding math section supplies the finite-background `v_x(S)` definition.
Its checkbox-controlled data table exposes every feature group, including the
empty and complete groups. All input columns remain visible. The Python panel
connects the procedure to a fixed model's `predict` method and SHAP's exact
explainer. Global importance and beeswarm plots are reserved for a later page.

Navigation buttons provide an alternative to scrolling. Native selects,
checkboxes, disclosures, and modal dialogs support keyboard operation. Dialogs
restore focus to their trigger. Reduced-motion preferences disable animated
replacements and bar reveals. Static narrative numbers remain available without
JavaScript, but the interactive figures require it.

## Run and reproduce

From the repository root:

```sh
python3 -m http.server 8765 --directory docs
node --test tests/*.test.mjs
```

Open `http://localhost:8765/shap/`. No website build step or Python installation
is needed to view the deployed page. All website assets are served locally,
including the math font shared with the earlier Shapley page.

[Python reproduction files](../../examples/shap/README.md) generate the saved
model and explanations and independently verify them against SHAP. The website
displays those results without executing Python in the browser. JavaScript tests
independently evaluate the equation for every hybrid input, check the averages,
recover SHAP via subset factorial weights, and verify all order paths and
additive predictions.

## Design and hosting

The page follows [the course guide](../../design/README.md): white backgrounds,
readable serif prose, restrained rules, locally typeset math, and the course
palette. Included/excluded headings and before/after numbers supplement color.
Blue and rust distinguish signed contributions in the waterfall, with arrows
and direct signed labels. They do not indicate good or bad people.

The browser draws the familiar waterfall from the same numbers used by Python
SHAP, with shared quantitative scales and feature rows ranked by magnitude.
The table uses all reference rows; its numbers are not decorative samples.

GitHub Pages publishes `docs/` after the checks pass on `main`. Merging this
change makes the page available at
`https://alexanderthclark.github.io/pols4728/shap/`. It also adds a course-home
link, preserving the existing logo and Shapley page.
