# Boosting with regression trees

A static explainer of boosting with regression trees, served at `/trees/`.
No build step or external JavaScript dependencies. The course homepage links
directly to this chapter.

The title cartoon shows two round, torso-less figures taking turns to straighten
a picture. Each adjustment crosses level by less than the previous one. The
animation plays automatically once, then holds the final near-level pose.
Leaving the title or hiding the tab pauses elapsed time.
Reduced motion shows the final pose without movement. `title-picture.js` draws
the SVG and defines the sequence; `title-motion.js` controls playback separately
from the scroll-driven teaching diagrams. The illustration is an analogy for
successive corrections, not a plot of the example's training error.
`assets/picture-hanging.svg` is the static final-pose fallback generated with
`TitlePicture.render(TitlePicture.duration)` and registered in the artwork list.

Ten scroll-linked scenes begin with observations from a synthetic delivery
dataset, reveal the mean prediction, and then form residuals for the first tree.
The figure scenes use a full-width figure, a short heading and equation, and
a concise takeaway; the transition uses only a heading and two short paragraphs. The arrangement inside the figure changes when a comparison
needs side-by-side plots; there is no permanent prose column. The page ends with only two further-reading references: Athey–Imbens (2019)
and Chapter 10 of The Elements of Statistical Learning. Supplemental prose
and expandable reading notes have been removed at the instructor’s request.
The example fits two boosting updates, then continues through eight
rounds. Aligned panels distinguish the current residual
target and its fitted stump from the original observations and accumulated
prediction. A dedicated transition introduces a new sixteen-observation dataset
with two predictors through a spare, text-only transition. It names the inputs
and introduces the comparison, without plots, an equation, a caption, or a score.
The last three frames fit this shared dataset.
One frame grows a single decision tree and its input map; the next fits four
residual stumps and builds their summed map. The final frame compares the maps,
training MSE, split-node counts, and region counts at the same four-split budget.

Scrolling through the opening reveals the mean prediction after a hold on the
observations alone. This first scene holds for 40% of its progress, reveals the
mean over the next 50%, and holds the final state for the remainder. Its plot
has no residual panel. Its observed dots expose exact y values on hover or
keyboard focus; the caption identifies the three distinct outcomes. The next scene starts with exactly the same coordinates,
then moves the observation plot down and copies its points into a residual plot
above it before revealing the split and fitted leaf means.

Scrolling through a delivery fitting or update scene advances its three-stage
sequence; scrolling backward rewinds it. These sections hold their opening state
for the first 8% of progress, complete the sequence over the next 77%, and
hold the final state for the remainder. Reduced motion advances fitting and
update scenes through completed phases rather than continuous interpolation.

Previous/Next buttons and left/right keys navigate the explanation. Previous
action, Next action, Replay sequence, and the progress slider also control each
sequence; an active replay can be paused and resumed. Manual choices stay in
place until the scroll position changes. A new scroll cancels replay and closes
the optional candidate explorer, returning control to the section's position.
All screen sizes use a reading line at 46% of viewport height, independent of
the figure's controls. Navigation settling does not overwrite a manual action, while resizing
resynchronizes the animation with the new geometry. Sliders retain their native
arrow-key behavior. Reduced motion applies requested button actions immediately.

Fitting first forms the residual targets, then separates the observations at the
chosen threshold, and finally reveals the leaf means. The second fitting scene
subtracts the previous correction from the old residuals while keeping F₁ fixed.
The optional candidate-cut explorer compares all seven partitions, reporting
their fitted residual means and SSE without changing the accumulated model.
Closing it restores the fitted winner. Update scenes attach a fixed correction,
add it to the previous function, and inspect the remaining errors. The targets
and correction above stay fixed while the prediction below changes.

The completed F₂ plot labels its two cutpoints and three intervals. The story
then moves directly to the third fitted correction, without a separate
contribution-strip recap. The later partition diagrams keep prediction labels fixed
in their regions. Slightly larger fixed dots encode observed y on the same scale
as the fitted surface; hovering or focusing a dot reports its observed time,
prediction, and signed residual y − F. Scrolling grows each model,
and Previous/Next split or stump buttons and Replay provide manual control.
New tree boundaries stop at their parent region; new stump boundaries cross
the whole input map. Reduced motion commits completed stages.

The continuation first holds F₂ while showing the third stump in dark orange
against its residual targets, with its selected 8 km split and +0.6/−1 leaf
values labeled. It then adds h₃ and follows the smaller corrections through F₈.
Each update shows its own selected cut, fixed pre-update residual targets, and
fitted orange stump below the moving prediction; these targets are distinct
from the current training MSE reported for the upper plot.
The transition to two predictors changes the coordinate roles:
delivery plots use distance horizontally and a response or residual vertically;
partition maps use both axes for inputs, with times encoded by color and direct
numerical labels. A fixed light-to-dark blue scale spans 8–24 minutes for both
observed y on the dots and predicted F in the regions. The scale is common to
both models and every fitting stage, and legends distinguish dots from regions.
Adding more trees does not add input dimensions.

## Numerical examples

All predictions are on the regression scale, with exact `<=` left-branch
semantics. Both datasets fit their leaf values and splits programmatically. Neither
comparison claims a generalization advantage from the training scores.

The synthetic delivery distances are `[1, 3, 4, 6, 7, 9, 10, 12]` km, with times
`[6, 6, 11, 11, 11, 17, 17, 17]` minutes. The initial prediction is their mean,
12 minutes. Each stump is fitted programmatically to the current residuals using
the squared-error fitter in `../shared/boosting-math.js`, with learning rate 1.

| Stage | Fitted correction | Accumulated predictions | Training MSE |
| --- | --- | --- | --- |
| Initial | None | All 12 | 18.75 |
| Round 1 | −3 for x ≤ 8; +5 otherwise | First five 9; last three 17 | 3.75 |
| Round 2 | −3 for x ≤ 3.5; +1 otherwise | 6, 6, 10, 10, 10, 18, 18, 18 | 0.75 |
| Round 3 | +0.6 for x ≤ 8; −1 otherwise | 6.6, 6.6, 10.6, 10.6, 10.6, 17, 17, 17 | 0.15 |
| Round 8 | −0.024 for x ≤ 3.5; +0.008 otherwise | 6, 6, 10.992, 10.992, 10.992, 17.008, 17.008, 17.008 | 0.000048 |

Each fitted partition uniquely minimizes SSE among the seven possible adjacent
data partitions. Midpoints label the thresholds; other thresholds between the
same adjacent observations produce the same training partition. The first stump
is fitted to `[-6, -6, -1, -1, -1, 5, 5, 5]`, and the second to
`[-3, -3, 2, 2, 2, 0, 0, 0]`. The error reductions describe training fit.
During the second update, the three longest-distance predictions rise from 17
to 18 even though their observations equal 17. Those individual errors grow
while aggregate training MSE falls. Animation progress interpolates the chosen
update; it does not refit the displayed stump to changing residual targets.

Later rounds alternate the same two thresholds, 8 and 3.5 km. In this particular
greedy sequence, every two further updates multiply the residual vector by 1/5,
and each update divides MSE by 5: after m rounds, training MSE is 18.75 / 5^m.
Thus the errors approach zero without reaching exact zero in a finite number of
rounds in exact arithmetic. F₈ is a near-zero training fit, not a perfect one.
The three-level target itself has an exact three-leaf tree representation;
the remaining error is a property of this sequence of fitted stump updates.

### Two predictors, one split budget

The new synthetic observations have distances `[2, 4, 8, 10]` km and prior stops
`[0, 1, 3, 4]`. Each combination occurs once. The observed minutes are:

| Stops / distance | 2 | 4 | 8 | 10 |
| --- | --- | --- | --- | --- |
| 0 | 8 | 10 | 13 | 16 |
| 1 | 10 | 11 | 15 | 16 |
| 3 | 12 | 14 | 18 | 21 |
| 4 | 13 | 16 | 20 | 24 |

Both models start at the mean, 14.8125. The comparison gives each model the
same four-split budget.
The tree chooses the largest reduction in training SSE over every candidate
split in every current leaf, then replaces only that leaf. Boosting fits one
stump to current residuals per round and adds it with learning rate 1.
Midpoints define cutpoints; ties follow leftmost leaf, feature, then threshold.

| Split nodes | Tree MSE | Tree regions | Stump-sum MSE | Stump-sum regions |
| --- | --- | --- | --- | --- |
| 0 | 17.902344 | 1 | 17.902344 | 1 |
| 1 | 8.523438 | 2 | 8.523438 | 2 |
| 2 | 4.390625 | 3 | 2.582031 | 4 |
| 3 | 2.390625 | 4 | 1.951823 | 6 |
| 4 | 1.625000 | 5 | 1.618490 | 9 |

The tree first cuts distance at 6, then stops at 2 separately in its right and
left children, then distance at 9 in the upper-right leaf. Stumps cut distance
at 6, stops at 2, distance at 9, and stops at 3.5 across the full domain. Their
signed leaf corrections are −3.0625/+3.0625, −2.4375/+2.4375, −11/24/+1.375,
and −1/3/+1. The final scores are similar and both positive. The stump sum uses
crossing cuts to form nine regions from four split nodes; one tree has five
leaves after four splits. Region values in the sum remain constrained.

The color scale and axes are shared across maps. Diagram updates reveal the
chosen cut before committing the new predictions and corresponding MSE/counts.
All fixed observations remain in the map. Hover details always use the same
committed prediction as the displayed surface and score, including while a new
cut is being drawn. Keyboard focus exposes the same observation details.

## Files and checks

- `model.js` supplies evaluation, branch paths, rectangular regions and common
  refinement; it is shared by the renderer and Node tests.
- `delivery.js` fits eight residual stumps and exports observations, rounds,
  MSE calculation, partial-update predictions, and frozen candidate rows through
  `candidates(round)`. Candidate values are residual predictions. It uses
  `BoostingMath` in the browser and the same module through CommonJS in tests.
- `delivery-view.js` renders the first six scenes with shared axes and exports
  their topics, captions, process labels, process maxima, and numerical readouts.
  The opening process runs from 0 to 1; fitting and update processes run from 0
  to 3, with addition during process 1 to 2. Direct renders default to a
  completed sequence and retain the older `correction` input for inspection.
- `growth.js` computes the recursive tree and residual-stump stages on the same
  two-predictor observations, with immutable stage snapshots and exact MSE.
- `growth-view.js` renders the text-only transition, two growth animations,
  and final comparison, with a shared quantitative color scale.
- `story.js` routes all ten full-width scenes and initializes controls and
  scrolling. `scenePresentation` supplies their headings, equations, and
  takeaways; the later-round equation follows the currently fitted tree.
  Scenes 0–5 delegate to `delivery-view.js`; scene 6 uses the transition API;
  scenes 7–9 delegate to the three fitting views in `growth-view.js`.
- `index.html` contains the ten scroll markers, accessible controls, and the
  two further-reading references. Numerical documentation remains in this README.
- `style.css` follows the course design guide, with a compact mobile stage,
  reduced motion, keyboard focus styling, and print references.

Run the JavaScript syntax checks in `.github/workflows/pages.yml` and
`node --test tests/*.test.mjs`. The title loads `title-picture.js`, then
`title-motion.js`. The teaching diagrams load `../shared/boosting-math.js`,
`delivery.js`, `delivery-view.js`, `model.js`, `growth.js`, `growth-view.js`, then
`story.js`.
For browser review, inspect all ten scenes at desktop and mobile widths,
scroll forward and backward through each delivery sequence, jump between
sections, step through each action, pause and resume a replay, scrub backward,
compare candidate cuts, move directly from F₂ to h₃ and back, and follow
the continuation while checking that its nonzero MSE remains visible. Then
grow both two-predictor models, replay and reverse their stages, and inspect the
final four-split comparison. Verify training scores and fixed map scales, then
hover or focus dots in both models to compare observed values and residuals.
Check reduced motion and the link back to the course home. Renderer tests verify fixed
coordinates and numerical states; a headless event/clock harness covers scroll
scrubbing, manual-control transitions, reduced motion, geometry changes, and
asynchronous candidate-panel closure. Browser review remains
necessary for layout and visual animation quality. The reading references remain available without JavaScript.
