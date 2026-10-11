# Forward stagewise additive modeling

A static companion to the boosting story, served at `/trees/`. No build step or
external JavaScript dependencies. The course home lists this page before boosting;
both explainers link to one another.

Fourteen scroll-linked scenes begin with observations from a synthetic delivery
dataset, reveal the mean prediction, and then form residuals for the first tree.
All fourteen scenes use a full-width figure, a short heading and equation, and
a concise takeaway. The arrangement inside the figure changes when a comparison
needs side-by-side plots; there is no permanent prose column. The complete
explanation and data are retained below the slides in an expandable reading
section, open without JavaScript and available in print.
The example fits two boosting updates, then continues through eight
rounds. Aligned panels distinguish the current residual
target and its fitted stump from the original observations and accumulated
prediction. The sequence then moves to two predictors, comparing two split nodes
in one tree (three regions) with two crossing stumps (four regions). A later
comparison shows the third split a single tree needs to match the four-region
prediction function, before distinguishing region counts from freedom in the
leaf values.

Scrolling through the opening reveals the mean prediction after a hold on the
observations alone. This first scene holds for 40% of its progress, reveals the
mean over the next 50%, and holds the final state for the remainder. Its plot
has no residual panel. The next scene starts with exactly the same coordinates,
then moves the observation plot down and copies its points into a residual plot
above it before revealing the split and fitted leaf means.

Scrolling through a delivery fitting or update scene advances its three-stage
sequence; scrolling backward rewinds it. These sections hold their opening state
for the first 8% of progress, complete the sequence over the next 77%, and
hold the final state for the remainder. The delivery summary similarly moves
the inspected distance from 0 to 13 km. Reduced motion advances fitting and
update scenes through completed phases rather than continuous interpolation,
and samples the contribution strips at 1, 6, and 12 km without sweeping.

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

The delivery summary uses three aligned strips for h₁, h₂, and their sum plus
the starting mean. Inspect distance, pointer probing, and Trace the sum reveal
the contributions at one shared input. In the later partition scenes, pointer
probing and focusable region buttons select the same input across the diagrams.
Separate replays animate a stump's contribution and compare the second split
in the recursive and ensemble constructions.

The continuation first holds F₂ while showing the third stump in dark orange
against its residual targets, with its selected 8 km split and +0.6/−1 leaf
values labeled. It then adds h₃ and follows the smaller corrections through F₈.
Each update shows its own selected cut, fixed pre-update residual targets, and
fitted orange stump below the moving prediction; these targets are distinct
from the current training MSE reported for the upper plot.
The transition to two predictors changes the coordinate roles:
delivery plots use distance horizontally and a response or residual vertically;
partition maps use both axes for inputs, with predictions written inside cells.
Adding more trees does not add input dimensions.

## Numerical examples

All predictions are on the regression scale, with exact `<=` left-branch
semantics. The delivery example fits its leaf values; the two-dimensional
partition examples use illustrative values.

The synthetic delivery distances are `[1, 3, 4, 6, 7, 9, 10, 12]` km, with times
`[6, 6, 11, 11, 11, 17, 17, 17]` minutes. The initial prediction is their mean,
12 minutes. Each stump is fitted programmatically to the current residuals using
the squared-error fitter in `../boosting/model.js`, with learning rate 1.

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

- The main comparison uses one tree that splits x1 at zero and then only its
  right child at x2 = 0, with leaf predictions 0, 1, and 3. The two-stump sum has
  an additional upper-left region predicting 2. Both constructions use two split
  nodes; their boundaries differ in where the second cut applies.
- A single tree that tests x2 in both children matches the sum with three split
  nodes. Its four leaf predictions are unrestricted, while the stump sum obeys
  the zero-interaction constraint: upper right = lower right + upper left − lower left.
- The one-dimensional two-cut example is retained in an afterword note. Both
  constructions yield three intervals there; it cannot demonstrate crossing cuts.
- In two dimensions, `T1 = 1{x1 > 0}` and `T2 = 2·1{x2 > 0}`. Their common refinement
  has predictions 0, 1, 2, and 3. A single 0–5 prediction color scale is shared by
  all the maps. Numbers, line styles, and leaf outlines supplement color.
- The conditional tree `U = 2·1{x1 > 0, x2 > 0}` has a horizontal split only in
  the right half-plane. Adding it changes the upper-right prediction to 5, so the
  x1 contrast is 1 below the horizontal boundary and 3 above it.

The structural overlay can retain boundaries whose jumps cancel. The page also
explains that a finite sum of these trees has a single-tree representation; the
learning procedures and constraints differ.

## Files and checks

- `model.js` supplies evaluation, branch paths, rectangular regions and common
  refinement; it is shared by the renderer and Node tests.
- `delivery.js` fits eight residual stumps and exports observations, rounds,
  MSE calculation, partial-update predictions, and frozen candidate rows through
  `candidates(round)`. Candidate values are residual predictions. It uses
  `BoostingMath` in the browser and the same module through CommonJS in tests.
- `delivery-view.js` renders the first seven scenes with shared axes and exports
  their topics, captions, process labels, process maxima, and numerical readouts.
  The opening process runs from 0 to 1; fitting and update processes run from 0
  to 3, with addition during process 1 to 2. Direct renders default to a
  completed sequence and retain the older `correction` input for inspection.
- `story.js` routes all fourteen full-width scenes and initializes controls and
  scrolling. `scenePresentation` supplies their headings, equations, and
  takeaways; the later-round equation follows the currently fitted tree.
  It also exports the original partition renderer and readout as
  `renderPartitionScene` and `partitionReadout` for focused tests.
- `index.html` contains the complete narrative and accessible controls.
- `style.css` follows the course design guide, with a compact mobile stage,
  reduced motion, keyboard focus styling, and print prose.

Run the JavaScript syntax checks in `.github/workflows/pages.yml` and
`node --test tests/*.test.mjs`. Browser loading order is `../boosting/model.js`,
`delivery.js`, `delivery-view.js`, `model.js`, then `story.js`.
For browser review, inspect all fourteen scenes at desktop and mobile widths,
scroll forward and backward through each delivery sequence, jump between
sections, step through each action, pause and resume a replay, scrub backward,
compare candidate cuts, trace contributions across both thresholds, and follow
the continuation while checking that its nonzero MSE remains visible. Then
probe the partition maps, activate regions with Enter/Space, replay
addition, toggle U, and navigate both directions. Check reduced motion and links
between the companion, boosting, and course home. Renderer tests verify fixed
coordinates and numerical states; a headless event/clock harness covers scroll
scrubbing, manual-control transitions, reduced motion, geometry changes, and
asynchronous candidate-panel closure. Browser review remains
necessary for layout and visual animation quality. The full prose remains
available without JavaScript.
