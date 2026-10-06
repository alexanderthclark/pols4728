# Explaining one prediction with SHAP

This scrolling story begins with the Shapley formula readers
already know. It maps its players to features and its game to one observation’s
prediction, distinguishing model training from explanation. The highlighted
value-function terms show how a revealed feature changes the value of a group: substitute the observation's values into background rows, evaluate
the same prediction function, average its outputs, then compare those averages.

The illustrative earnings predictor is `f(a,n,e) = 40 + 10a + 24an + 4e`, in
thousands of dollars per year. Ability and experience range from 0 to 1;
neighborhood opportunity ranges from −1 to +1. The interaction can outweigh
ability's positive main term in an adverse context. The equation and profiles
are constructed for teaching; they are not empirical estimates or causal claims.
There are three features, labeled with lowercase `a`, `n`, and `e`. Uppercase
`S` and `F` denote sets of features, and `i` denotes the feature joining `S`.
The interaction is computed inside the predictor.

The eight equally weighted reference profiles cover every endpoint combination.
All eight rows appear in the main tables. Included columns are fixed to the
selected person; excluded columns retain the joint values from each background
row. The before/after output columns and their means show the marginal directly.
No feature is removed from the model or zeroed to represent absence.

The default profile has `(a,n,e) = (1,−1,1)`. Its baseline is 47 and prediction is
30. Ability's marginal is +5 before neighborhood is revealed and −7 after it.
Each occurs in three of the six orders, giving a final ability SHAP value of −1.
The complete explanation is `47 − 1 − 18 + 2 = 30`. Alternate favorable and mixed
profiles use the same model and background.

## Story and controls

1. Begin with the complete Shapley formula.
2. Highlight the value function: a feature group’s prediction average.
3. Highlight the observation `x` supplying the fixed inputs.
4. Highlight `S`, the group already fixed, and define `F` and `m`.
5. Highlight `i`, the additional feature joining a group that excludes it.
6. Highlight the result: the final contribution assigned to that feature.
7. Orient training, the fixed predictor, and one observation to explain.
8. Compute `v_x(S)` from background rows, starting with `S = ∅`.
9. Compute `v_x(S ∪ {i})` by fixing ability and comparing with the baseline.
10. Start with neighborhood fixed and compare the same two value-function terms.
11. Match all four preceding groups to their factorial weights and prediction differences.
12. Add the weighted contributions; select a feature and inspect all six orders.
13. Assemble the final contributions in a course-styled standard waterfall.

The opening keeps the same equation in place and highlights only the symbols
under discussion. Definitions enter through scrolling instead of a glossary.
The person selector appears when the worked prediction example begins.
The full formula fits on one line in the desktop figure, with type sized to
the figure's width. Phones and narrow prose columns allow a readable line break.

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
