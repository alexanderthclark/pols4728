# Explaining one prediction with SHAP

This scrolling story begins with the Shapley formula readers
already know. It maps its players to features and its game to one observation’s
prediction, distinguishing model training from explanation. The highlighted
value-function terms show how a revealed feature changes the value of a group: substitute the observation's values into background rows, evaluate
the same prediction function, average its outputs, then compare those averages.

The first worked example is a stylized Bread and Peace model inspired by
Douglas Hibbs. Two features, `bread` and `peace`, predict incumbent-party
two-party vote share. Bread is standardized real income growth; peace is minus
standardized war fatalities, so higher peace means fewer fatalities.
Features and observed vote have mean zero and SD one. Three stipulated
correlations, bread–peace +0.5, bread–vote +0.75, and peace–vote +0.75, determine
both univariate OLS slopes (+0.75) and both bivariate slopes (+0.5), with zero
intercepts. The figure displays all three model equations; the coefficient
derivations are in an expandable note. Predictions use the outcome's SD units;
they are not themselves rescaled to SD one.

Removing peace and refitting bread-only OLS gives a slope of 0.75, equal
to the vote–bread correlation. The full-model coefficient of 0.5 plus the
omitted-variable term `0.5 × 0.5 = +0.25` recovers that slope. Our SHAP
calculation instead keeps the full model fixed and averages its predictions.
For the default election, bread 1 and peace 1, the bread-only coalition
has value 0.5 and the full prediction is 1. Four coalitions and two revealing
orders give contributions +0.5 and +0.5 from a zero baseline. The scroll writes
out both orders as explicit prediction differences, such as
`y_hat(1, 0) − y_hat(0, 0)` for bread joining first and
`y_hat(1, 1) − y_hat(0, 1)` for bread joining second. Both calls use the
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
3. Define `y_hat(x)`, the fixed model’s prediction for one complete input row, distinct from observed outcome `y`.
4. Define the value function through a comparison of `v_x(S)` and `v_x(S ∪ {i})`: both average the same model’s predictions, while feature `i` moves from background values to its value in `x`.
5. Highlight the result and introduce `phi_i(x)` as shorthand for `phi_i(v_x)`.
6. Introduce the stylized Bread and Peace specification with the original Hibbs citation and course-style characters.
7. State the three correlations for standardized bread, peace, and observed vote.
8. Display the two univariate fits and one bivariate fit, with the derivation in the notes.
9. Choose an election; compare the muted bread-only refit (not used for SHAP) with the bivariate prediction at mean peace.
10. Reveal bread first, then peace; show both before/after prediction subtractions.
11. Reverse the order and show the same bivariate equation evaluated at different inputs.
12. Average each feature's two marginals and reconstruct the full prediction.
13. Introduce how ability and neighborhood interact with a centered East/West Germany analogy.
14. Move to the three-feature earnings interaction and one person to explain.
15. Compute its `v_x(S)` from background rows, starting with `S = ∅`.
16. Fix ability and compare with the earnings baseline.
17. Start with neighborhood fixed and compare the same two value-function terms.
18. Match all four preceding groups to their factorial weights and prediction differences.
19. Add the weighted contributions; select a feature and inspect all six orders.
20. Assemble the final contributions in a course-styled standard waterfall.

The opening frames distinguish `ŷ(x)`, the fixed model’s prediction for a
complete row, from `v_x(S)`, an average of that same model’s predictions across
completed background rows. Fixing every feature makes every row equal to `x`,
so the average equals `ŷ(x)`. Frame 4 defines this averaged-prediction value function by comparing both coalition values:
`S` supplies the inputs already fixed to `x`; adding feature `i` uses its value
from `x` instead of its background values. The note below the comparison
identifies `i` as the additional feature and states there is no retraining. The alias `f(x) = ŷ(x)` is
introduced with the earnings model. Frame 9 explicitly connects row averaging to the valid linear
shortcut: average `ŷ(bread, peace)` over reference peace values, giving
`ŷ(bread, 0)` because the overall background mean is zero.

The first five frames use a centered formula, with short definitions
underneath. The same equation stays in place and highlights the symbols under
discussion; frame 4 shows its before and after coalition values together.
Definitions enter through scrolling instead of a glossary, including
the observation, model prediction, value function, feature groups, and final credit.
Frame six is a full-width Bread and Peace transition, with a centered teaching
specification, a compact loaf character, and a peace-symbol character. It cites
[Hibbs (2000), “Bread and Peace Voting in U.S. Presidential Elections”](https://link.springer.com/article/10.1023/A:1005292312412),
and identifies bread as real income growth and peace as fewer war fatalities.
The peace feature reverses the standardized war-fatalities measure, so both
feature coefficients in the teaching model are positive.
The equation is a stylized, centered and standardized teaching specification,
rather than the original historical fit. The split layout begins with the
three correlations in frame seven. Frame 13 is another centered transition,
“When features interact,” showing two round body-and-face characters with
stick limbs on a tilted map of Germany. The smiling western character wears a
top hat; the frowning eastern character wears a patched cap and worn shoes.
The bottom map credit names West and East Germany. The native math term
`ability × neighborhood` connects that comparison to the illustrative earnings
equation. The full earnings model and person selector appear in frame 14.
The story has 20 frames.
Each worked frame's left column uses two or three short lecture cues in larger type,
leaving the instructor room to explain the worked figure. The earnings equation
appears beside its observation in the figure. Expandable notes retain the notation,
OLS derivation, model assumptions, and factorial-count explanation for later reading.
The election selector appears after all three OLS fits are introduced. The person
selector appears when the earnings example begins. Each example retains its
selection when readers move between them. Frame 12 adds a small prediction
identity below its order-weight caption:
`ŷ(x) = ŷ(0, 0) + ϕ_bread(x) + ϕ_peace(x)`. The zero-input baseline applies to
this centered additive model.
The full formula fits on one line in the desktop figure, with type sized to
the figure's width. Phones and narrow prose columns allow a readable line break.

The concluding math section supplies the finite-background `v_x(S)` definition.
Its checkbox-controlled data table exposes every feature group, including the
empty and complete groups. All input columns remain visible. The Python panel
connects the procedure to a fixed model's `predict` method and SHAP's exact
explainer. Global importance and beeswarm plots are reserved for a later page.

Navigation buttons provide an alternative to scrolling and advance immediately
so quick clicks cannot repeat a frame during a scroll animation. The opening
equation stays in place across its five opening frames. Native selects,
checkboxes, disclosures, and modal dialogs support keyboard operation. Dialogs
restore focus to their trigger, and a concise status announces stage changes.
Dense figures scroll within their panel when space is limited, with a visible
cue and keyboard access. Navigation resets that figure scroll for the next frame.
The compact waterfall retains readable labels and its common quantitative scale;
its displayed bar order ranks averaged credits, rather than representing one
revealing order.
The sum's narrative follows the selected feature as well as the selected person.
Reduced-motion preferences disable animated replacements and bar reveals.
Short landscape screens place the worked prose and figure side by side. Phone tables
keep every column visible, with larger included/excluded labels and controls.
The weights table uses 12 px text and compact spacing on narrow screens.
Without JavaScript, the lecture cues, complete formula, both model transitions,
earnings equation, worked profiles, and calculation notes remain available in a
continuous reading layout; interactive figures require it. The opening formula,
both model transitions, and worked profiles also appear in print.

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

The Germany illustration uses [IEG-Maps, Map 360](https://www.ieg-maps.uni-mainz.de/mapsp/mapp989d.htm), by Andreas Kunz and Joachim Robert Moeschl (CC BY-NC 4.0), as a historical geography reference. Its outline and division are schematic; the two figures represent equal ability and experience in different institutional settings. The illustration does not assign the earnings model’s numerical neighborhood scores to German territories.
