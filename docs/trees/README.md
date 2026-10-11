# Fit a correction. Add a tree.

A static companion to the boosting story, served at `/trees/`. No build step or
external JavaScript dependencies. The course home lists this page before boosting;
both explainers link to one another.

Thirteen scroll-linked scenes begin with two fitted boosting updates on a
synthetic delivery dataset. Aligned panels distinguish the current residual
target and its fitted stump from the original observations and accumulated
prediction. The sequence then moves to two predictors, comparing two split nodes
in one tree (three regions) with two crossing stumps (four regions). A later
comparison shows the third split a single tree needs to match the four-region
prediction function, before distinguishing region counts from freedom in the
leaf values.

The input sliders, pointer probing, and focusable leaf buttons select the same input across
the diagrams. Previous/Next buttons and left/right keys navigate the explanation.
Sliders retain their native arrow-key behavior. The addition control scales a
second tree; Replay animates its coefficient from zero to one. A separate replay
shows the first split in both constructions, then adds the second. Reduced motion
uses the completed comparison and immediate navigation.

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

Both fitted partitions uniquely minimize SSE among the seven possible adjacent
data partitions. Midpoints label the thresholds; other thresholds between the
same adjacent observations produce the same training partition. The first stump
is fitted to `[-6, -6, -1, -1, -1, 5, 5, 5]`, and the second to
`[-3, -3, 2, 2, 2, 0, 0, 0]`. The error reductions describe training fit.

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
- `delivery.js` fits the two residual stumps and exports observations, rounds,
  MSE calculation, and partial-update predictions. It uses `BoostingMath` in the
  browser and the same module through CommonJS in tests.
- `delivery-view.js` renders the first six scenes with shared axes and exports
  their topics, captions, and numerical readouts.
- `story.js` routes all thirteen scenes and initializes controls and scrolling.
  It also exports the original partition renderer and readout as
  `renderPartitionScene` and `partitionReadout` for focused tests.
- `index.html` contains the complete narrative and accessible controls.
- `style.css` follows the course design guide, with a compact mobile stage,
  reduced motion, keyboard focus styling, and print prose.

Run the JavaScript syntax checks in `.github/workflows/pages.yml` and
`node --test tests/*.test.mjs`. Browser loading order is `../boosting/model.js`,
`delivery.js`, `delivery-view.js`, `model.js`, then `story.js`.
For browser review, inspect all thirteen scenes at desktop and mobile widths,
replay the fitted residual corrections, compare both aligned rows, and drag
both input sliders, probe the plots, activate leaves with Enter/Space, replay and
pause addition, toggle U, and navigate both directions. Check reduced motion and
links between the companion, boosting, and course home. The full prose remains
available without JavaScript.
