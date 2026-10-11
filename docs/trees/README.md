# Trees, one partition at a time

A static companion to the boosting story, served at `/trees/`. No build step or
external JavaScript dependencies. The course home lists this page before boosting;
both explainers link to one another.

Eight scroll-linked scenes connect a decision tree to its step function and then
compare two split nodes in one tree (three regions) with two crossing stumps
(four regions). Slide 03 makes that comparison side by side; slide 05 shows the
third split a single tree needs to match the same four-region prediction function.
The next scene distinguishes region counts from freedom in the leaf values.
The input
sliders, pointer probing, and focusable leaf buttons select the same input across
the diagrams. Previous/Next buttons and left/right keys navigate the explanation.
Sliders retain their native arrow-key behavior. The addition control scales a
second tree; Replay animates its coefficient from zero to one. A separate replay
shows the first split in both constructions, then adds the second. Reduced motion
uses the completed comparison and immediate navigation.

## Numerical examples

The leaf values are illustrative, not estimated from data. All predictions are on
the regression scale, with exact `<=` left-branch semantics.

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
- `story.js` exports pure SVG renderers for inspection, then initializes controls
  and scrolling in the browser.
- `index.html` contains the complete narrative and accessible controls.
- `style.css` follows the course design guide, with a compact mobile stage,
  reduced motion, keyboard focus styling, and print prose.

Run `node --check docs/trees/story.js` and `node --test tests/*.test.mjs`.
For browser review, inspect all eight scenes at desktop and mobile widths, drag
both input sliders, probe the plots, activate leaves with Enter/Space, replay and
pause addition, toggle U, and navigate both directions. Check reduced motion and
links between the companion, boosting, and course home. The full prose remains
available without JavaScript.
