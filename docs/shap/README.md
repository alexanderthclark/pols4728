# Explaining one prediction with SHAP

The opening title is “SHAP,” with the subtitle “SHapley Additive exPlanations,”
following the expansion used in the [official documentation](https://shap.readthedocs.io/en/latest/).
The slide shows two torso-less investigators examining a faceless
Y-hat in a gently landed spaceship. Its Start link leads into the existing
30-frame explanation; the cover is separate from the story’s frame count.
The first 21 frames lead to a local waterfall; nine appended frames build
a beeswarm and derive global importance from those same local contributions.

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
intercepts. The figure displays all three model equations. Predictions use the outcome's SD units;
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
in this additive model. The OLS recap summarizes coalition inputs and predictions
without displaying the synthetic training rows.
Alternate elections change the inputs and contributions; the fit and background
stay fixed. These are constructed examples, not historical elections or Hibbs's
estimated coefficients. The reproduction files construct eight standardized
synthetic rows to independently verify the stipulated correlations and model
fits; empirical SD uses denominator `n`. Those verification rows do not appear
in the story. The optional model and background note distinguishes interventional
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

The summary extends the earnings example to 125 equally weighted constructed
profiles: all combinations of ability and experience in `{0, .25, .5, .75, 1}`
and neighborhood opportunity in `{−1, −.5, 0, .5, 1}`. These are the people
being explained. Their explanations continue to use the original eight endpoint
reference profiles, the same fixed predictor, and the baseline of 47.
The cohort does not replace the background.

## Story and controls

1. Begin with the complete Shapley formula.
2. Highlight the observation `x` supplying the fixed inputs.
3. Define `y_hat(x)`, the fixed model’s prediction for one complete input row, distinct from observed outcome `y`.
4. Define `v_x(S)` as the average of the same fixed model’s predictions over completed background rows, retaining this observation’s values for features in `S`.
5. Define the marginal contribution `v_x(S ∪ {i}) − v_x(S)`, then connect its average across orders to the feature’s credit, `phi_i(v_x)`.
6. Introduce the stylized Bread and Peace specification with the original Hibbs citation and course-style characters.
7. State the three correlations for standardized bread, peace, and observed vote.
8. Display the two univariate fits and one bivariate fit.
9. Choose an election; compare the muted bread-only refit (not used for SHAP) with the bivariate prediction at mean peace.
10. Reveal bread first, then peace; show both before/after prediction subtractions.
11. Reverse the order and show the same bivariate equation evaluated at different inputs.
12. Average each feature's two marginals and reconstruct the full prediction.
13. Re-emphasize the general average over background predictions, then label mean substitution as a linear model shortcut and map all four feature groups to predictions for the selected election.
14. Introduce how ability and neighborhood interact with a centered East/West Germany analogy.
15. Move to the three-feature earnings interaction and one person to explain.
16. Compute its `v_x(S)` from background rows, starting with `S = ∅`.
17. Fix ability and compare with the earnings baseline.
18. Start with neighborhood fixed and compare the same two value-function terms.
19. Match all four preceding groups to their factorial weights and prediction differences.
20. Add the weighted contributions; select a feature and inspect all six orders.
21. Assemble the final contributions in a course-styled standard waterfall.
22. Move each waterfall contribution to a common zero, preserving its signed length.
23. Replace each contribution bar with a dot at its SHAP value.
24. Repeat the explanation for all 125 constructed profiles, with the same model and eight reference rows.
25. Spread overlapping dots vertically into a beeswarm, keeping their horizontal contributions fixed.
26. Color dots by low-to-high input value within each feature, distinguishing input value from contribution direction.
27. Show how signed averages cancel positive and negative contributions across these profiles.
28. Fold negative contributions to their absolute distance from zero.
29. Average the 125 absolute contributions in each feature row to construct global importance bars.
30. Rank the mean absolute contributions: neighborhood opportunity 7.2, ability 3.24, and experience 1.2, in $1,000/year.

The opening frames distinguish `ŷ(x)`, the fixed model’s prediction for a
complete row, from `v_x(S)`, an average of that same model’s predictions across
completed background rows. Fixing every feature makes every row equal to `x`,
so the average equals `ŷ(x)`. Frame 4, “Average background predictions,” gives
the general definition before either worked example:
`v_x(S) = (1/B) Σ_{b=1}^B ŷ(x_S, z^(b)_−S)`. Here `B` counts the reference rows,
`z^(b)` is background row `b`, and `x_S` supplies this observation's fixed
values. Excluded inputs retain their joint values from each reference row.
These are overall background rows, not rows conditioned on the included inputs;
the same model predicts every completed row. Frame 5 connects the marginal
contribution `v_x(S ∪ {i}) − v_x(S)` directly to feature credit: `S` supplies
the inputs already fixed to `x`; adding feature `i` uses its value from `x`
instead of its background values. Averaging those prediction differences across
orders gives the feature's SHAP value. The model is not retrained. The explainer
uses `phi_i(v_x)` for the feature’s credit throughout. The alias `f(x) = ŷ(x)`
is introduced with the earnings model. Frame 9 explicitly connects row averaging
to the valid linear shortcut:
average `ŷ(bread, peace)` over reference peace values, giving `ŷ(bread, 0)`
because the overall background mean is zero.

Frame 13 returns to the general average over background predictions first in a
centered, full-width recap. It then labels
`v_x(S) = ŷ(x_S, x̄_−S)` as a linear model shortcut. This unstandardized identity
holds for a linear model including its intercept. The bar denotes overall
reference means, not conditional means. Each of the four groups `S`, including
the empty and complete groups, maps to a prediction from the same bivariate
model: included predictors retain the selected election's values, while excluded
predictors use their overall reference means, both zero in the teaching example.
The coefficients and background stay fixed. These grouped arguments retain the
inputs in their original feature positions; they do not reorder the model's
columns. Standardization only makes the means zero. Linearity makes prediction
at those means equal to the average of the background-row predictions here.
Averaging predictions remains the general definition of `v_x(S)`; substituting
mean inputs need not reproduce that average in a nonlinear model.

The first five frames use a centered formula, with short definitions
underneath. The same equation stays in place and highlights the symbols under
discussion; frame 5 defines a marginal contribution and connects its average to feature credit.
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
three correlations in frame seven. Frame eight centers the three fitted
equations in a full-width scene. Frame nine also uses the full width for the
prediction comparison, retaining the election selector and averaging equation;
frame ten returns to the worked layout. Frame 13 centers the general
definition and four-coalition recap before frame 14's centered transition,
“When features interact,” showing two round body-and-face characters with
stick limbs on a tilted map of Germany. The smiling western character wears a
top hat; the frowning eastern character wears a patched cap and worn shoes.
The bottom map credit names West and East Germany. The native math term
`ability × neighborhood` connects that comparison to the illustrative earnings
equation. The full earnings model and person selector appear in frame 15.
The story has 30 teaching frames, plus the unchanged opening title scene.
When a chart exceeds the available figure height, the figure scrolls vertically
and shows a scroll hint. Aggregate charts retain page-scroll chaining at their
edges and reserve scrollbar space so the chart layout remains stable.
Bread and Peace and the earnings calculations use the full-width pinned stage.
Essential explanations use 18px text on desktop and 16px on phones; table sources
stay at least 14px. Before-and-after predictions share one column, and each
calculation's main result appears before its reference table. The order-weight
table uses readable group names and explicit Before, After, Marginal, and Weight
headings. The earnings equation appears with the selected observation. A concise model and background
note retains the assumptions and interpretation for later reading; the story
itself introduces the notation and factorial weights.
The election selector appears after all three OLS fits are introduced. The person
selector appears when the earnings example begins. Each example retains its
selection when readers move between them. Frame 12 adds the prediction
identity below its order averages:
`ŷ(x) = ŷ(0, 0) + ϕ_bread(v_x) + ϕ_peace(v_x)`. The zero-input baseline applies to
this centered additive model.
The full formula fits on one line in the desktop figure, with type sized to
the figure's width. Phones and narrow prose columns allow a readable line break.

The concluding “Notes and exploration” section has three dropdowns: model and
background assumptions, the coalition table explorer, and an individual
waterfall explorer. The model note preserves the distinctions between explaining
a fixed model, refitting a reduced model, and conditional SHAP, together with the
constructed example's interpretation. The checkbox-controlled data table exposes
every feature group, including the empty and complete groups; all input columns
remain visible. The individual explorer opens the complete waterfall for a
selected profile. A link to the calculation reproduction files supplies the full
implementation and SHAP verification. The global-summary note distinguishes the
125 explained profiles
from the eight reference rows and defines importance as mean absolute SHAP.
The signed means are zero for all three features; taking absolute values before
averaging retains contribution magnitude. The ranking describes this predictor,
reference background, and explained cohort. It does not describe causal effects.

The waterfall and population sequence also use the full-width pinned stage.
Waterfall inputs, axes, legends, and contribution labels remain readable on
phones. The waterfall is read from the baseline upward, adding contributions
from bottom to top; its arithmetic follows that same order. Chart captions
use 18px text on desktop and 16px on phones.

Scrolling carries the complete transition from a waterfall to a beeswarm and
global importance. The feature rows keep their order until the final ranking.
Vertical packing only prevents dot overlap; its height has no numerical meaning.
The colored beeswarm uses color for the input's position within that feature's
range, while horizontal position continues to show the signed contribution.
The final axis explicitly measures mean absolute SHAP value. The bars retain
average magnitude but discard the direction, spread, and interaction patterns
visible in the beeswarm. Captions distinguish the 125 explained profiles from
the eight fixed background rows and keep the dependence of the ranking on the
model, background, and explained population visible. Optional exploration opens a person's complete
waterfall by selecting a dot in the colored beeswarm or choosing a profile in
the individual waterfall dropdown under “Notes and exploration” and selecting
its waterfall button.

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
Short landscape screens retain the full-width figures and allow the figure to
scroll when necessary. Phone tables keep the substantive columns visible, with
readable input-source labels and controls. The weights table uses 16px values
and 14px column headings on narrow screens.
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
including the local Latin Modern Math font and its license. Inline symbols,
subscripts, accents, sets, and fractions use native MathML with the same
LaTeX-style mathematical typography as the displayed equations.

[Python reproduction files](../../examples/shap/README.md) generate the saved
model and explanations and independently verify them against SHAP. The browser
reads those saved results for the original profiles. `aggregate.mjs` evaluates
the same fixed `model.json` equation and exact finite-background game for every
profile in the 125-person grid; it does not run Python. JavaScript tests
independently evaluate the equation for every hybrid input, check the averages,
recover SHAP via subset factorial weights, and verify all order paths and
additive predictions. The aggregate tests reproduce all saved explanations,
check the balanced grid and common baseline, independently verify closed-form
SHAP values across all 125 profiles, and check signed cancellation, mean absolute
values, and ranking. The OLS tests independently solve its normal equations,
check standardization, distinguish refitting from masking, and reconstruct all
two-feature coalitions and orders.

## Design and hosting

The page follows [the course guide](../../design/README.md): white backgrounds,
readable serif type, restrained rules, locally typeset math, and the course
palette. Included/excluded headings and before/after numbers supplement color.
The sparse lecture copy follows the instructor's request for in-class presentation;
the concise model and background note preserves the assumptions and qualifications.
Blue and rust distinguish signed contributions in the waterfall, with arrows
and direct signed labels. They do not indicate good or bad people.
The beeswarm introduces its separate low-to-high feature-value color scale
through scrolling; the meaning of its horizontal axis remains unchanged.

The browser draws the familiar waterfall from the same numbers used by Python
SHAP, with shared quantitative scales and feature rows ranked by magnitude.
The table uses all reference rows; its numbers are not decorative samples.

GitHub Pages publishes `docs/` after the checks pass on `main`. Merging this
change makes the page available at
`https://alexanderthclark.github.io/pols4728/shap/`. It also adds a course-home
link, preserving the existing logo and Shapley page.

The Germany illustration uses [IEG-Maps, Map 360](https://www.ieg-maps.uni-mainz.de/mapsp/mapp989d.htm), by Andreas Kunz and Joachim Robert Moeschl (CC BY-NC 4.0), as a historical geography reference. Its outline and division are schematic; the two figures represent equal ability and experience in different institutional settings. The illustration does not assign the earnings model’s numerical neighborhood scores to German territories.
