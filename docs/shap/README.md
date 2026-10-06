# Explaining one prediction with SHAP

This scrolling story begins with the Shapley formula readers
already know. It maps its players to features and its game to one observation’s
prediction, distinguishing model training from explanation. The highlighted
value-function terms show how a revealed feature changes the value of a group: substitute the observation's values into background rows, evaluate
the same prediction function, average its outputs, then compare those averages.

The first worked example is a stylized Bread and Peace model inspired by
Douglas Hibbs. Two features, real income growth and war fatalities, predict
incumbent-party two-party vote share. The story begins with centered,
standardized features and observed vote (mean zero, SD one), and three stipulated
correlations: income–fatalities −0.5, income–vote +0.75, and fatalities–vote
−0.75. These determine the two univariate OLS slopes (+0.75 and −0.75) and
the bivariate slopes (+0.5 and −0.5), with zero intercepts. The figure displays
all three model equations; the coefficient derivations are in an expandable note. Predictions
use the outcome's SD units; they are not themselves rescaled to SD one.

Removing fatalities and refitting income-only OLS gives a slope of 0.75, equal
to the vote–income correlation. The full-model coefficient of 0.5 plus the
omitted-variable term `−0.5 × −0.5 = +0.25` recovers that slope. Our SHAP
calculation instead keeps the full model fixed and averages its predictions.
For the default election, income 1 and fatalities −1, the income-only coalition
has value 0.5 and the full prediction is 1. Four coalitions and two revealing
orders give contributions +0.5 and +0.5 from a zero baseline. The scroll writes
out both orders as explicit prediction differences, such as
`y_hat(1, 0) − y_hat(0, 0)` for income joining first and
`y_hat(1, −1) − y_hat(0, −1)` for income joining second. Both calls use the
same bivariate coefficients. Zero represents the mean of an averaged-out input
in this additive model. There are no data tables in the OLS warm-up.
Alternate elections change the inputs and contributions; the fit and background
stay fixed. These are constructed examples, not historical elections or Hibbs's
estimated coefficients. The reproduction files construct eight standardized
synthetic rows to independently verify the stipulated correlations and model
fits; empirical SD uses denominator `n`. Those verification rows do not appear
in the story. The optional method note distinguishes interventional
replacement from conditional SHAP and reduced-model refitting.

The illustrative earnings predictor is
`f(x) = 40 + 10 × ability + 24 × ability × neighborhood + 4 × experience`, in
thousands of dollars per year. Ability and experience range from 0 to 1;
neighborhood opportunity ranges from −1 to +1. The interaction can outweigh
ability's positive main term in an adverse context. The equation and profiles
are constructed for teaching; they are not empirical estimates or causal claims.
There are three features, named `ability`, `neighborhood`, and `experience`;
`neighborhood` denotes neighborhood opportunity. `S` and `F` denote sets of
features, with `F = {ability, neighborhood, experience}`, and `i` denotes the
feature joining `S`. The interaction is computed inside the predictor.

The eight equally weighted reference profiles cover every endpoint combination.
All eight earnings reference rows appear in the main tables. Included columns are fixed to the
selected person; excluded columns retain the joint values from each background
row. The before/after output columns and their means show the marginal directly.
No feature is removed from the model. In the centered additive OLS warm-up,
inserting the excluded features' zero means happens to reproduce the prediction
average. Background-row averaging is the definition used for both examples.

The default profile has ability 1, neighborhood −1, and experience 1.
Its baseline is 47 and prediction is 30. Ability's marginal is +5 before
neighborhood is revealed and −7 after it.
Each occurs in three of the six orders, giving a final ability SHAP value of −1.
The complete explanation is `47 − 1 − 18 + 2 = 30`. Alternate favorable and mixed
profiles use the same model and background.

## Story and controls

1. Begin with the complete Shapley formula.
2. Highlight the observation `x` supplying the fixed inputs.
3. Define `y_hat(x) = f(x)`, distinct from the observed outcome and prediction error.
4. Highlight its value function: the average of `y_hat` over hybrid input rows.
5. Highlight `S`, the group already fixed, and define `F` and `m`.
6. Highlight `i`, the additional feature joining a group that excludes it.
7. Highlight the result and introduce `phi_i(x)` as shorthand for `phi_i(v_x)`.
8. State the three correlations for standardized income, fatalities, and observed vote.
9. Display the two univariate fits and one bivariate fit, with the derivation in the notes.
10. Choose an election; compare income-only OLS with the bivariate prediction at mean fatalities.
11. Reveal income first, then fatalities; show both before/after prediction subtractions.
12. Reverse the order and show the same bivariate equation evaluated at different inputs.
13. Average each feature's two marginals and reconstruct the full prediction.
14. Move to the three-feature earnings interaction and one person to explain.
15. Compute its `v_x(S)` from background rows, starting with `S = ∅`.
16. Fix ability and compare with the earnings baseline.
17. Start with neighborhood fixed and compare the same two value-function terms.
18. Match all four preceding groups to their factorial weights and prediction differences.
19. Add the weighted contributions; select a feature and inspect all six orders.
20. Assemble the final contributions in a course-styled standard waterfall.

The first seven frames use a single centered formula, with a short definition
underneath. The same equation stays in place and highlights the symbols under
discussion. Definitions enter through scrolling instead of a glossary, including
the observation, model prediction, value function, feature groups, and final credit.
The split layout begins with the election example in frame eight.
Each worked frame's left column uses two or three short lecture cues in larger type,
leaving the instructor room to explain the worked figure. The earnings equation
appears beside its observation in the figure. Expandable notes retain the notation,
OLS derivation, model assumptions, and factorial-count explanation for later reading.
The election selector appears after all three OLS fits are introduced. The person
selector appears when the earnings example begins. Each example retains its
selection when readers move between them.
The full formula fits on one line in the desktop figure, with type sized to
the figure's width. Phones and narrow prose columns allow a readable line break.

The concluding math section supplies the finite-background `v_x(S)` definition.
Its checkbox-controlled data table exposes every feature group, including the
empty and complete groups. All input columns remain visible. The Python panel
connects the procedure to a fixed model's `predict` method and SHAP's exact
explainer. Global importance and beeswarm plots are reserved for a later page.

Navigation buttons provide an alternative to scrolling. Formula navigation
advances immediately while keeping the equation in place. Native selects,
checkboxes, disclosures, and modal dialogs support keyboard operation. Dialogs
restore focus to their trigger, and a concise status announces stage changes.
The sum's narrative follows the selected feature as well as the selected person.
Reduced-motion preferences disable animated replacements and bar reveals.
Short landscape screens place the worked prose and figure side by side. Phone tables
keep every column visible, with larger included/excluded labels and controls.
Without JavaScript, the lecture cues, complete formula, earnings equation, worked
profiles, and calculation notes remain available in a continuous reading layout;
interactive figures require it. The opening formula and worked profiles also
appear in print.

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
additive predictions. The OLS tests independently solve its normal equations,
check standardization, distinguish refitting from masking, and reconstruct all
two-feature coalitions and orders.

## Design and hosting

The page follows [the course guide](../../design/README.md): white backgrounds,
readable serif type, restrained rules, locally typeset math, and the course
palette. Included/excluded headings and before/after numbers supplement color.
The sparse lecture copy follows the instructor's request for in-class presentation;
the expandable notes preserve the longer reasoning and qualifications.
Blue and rust distinguish signed contributions in the waterfall, with arrows
and direct signed labels. They do not indicate good or bad people.

The browser draws the familiar waterfall from the same numbers used by Python
SHAP, with shared quantitative scales and feature rows ranked by magnitude.
The table uses all reference rows; its numbers are not decorative samples.

GitHub Pages publishes `docs/` after the checks pass on `main`. Merging this
change makes the page available at
`https://alexanderthclark.github.io/pols4728/shap/`. It also adds a course-home
link, preserving the existing logo and Shapley page.
