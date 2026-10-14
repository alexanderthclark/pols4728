# Imbalanced Data

A scrolling lesson based on Alexander Clark's Applied Issues notes. The
text-only opening title leads into three pinned, full-width chapters, with
the explanation above each figure. Scroll position continuously changes the
geometry; scrolling backward reverses the argument. No build step, external
JavaScript dependency, or external data call is required.

The first chapter develops logistic prior correction one step at a time:
population probabilities, case counts at a fixed predictor value, random
negative thinning, the change in odds, the same multiplier at every predictor
value, its conversion into an additive log-odds shift, and the intercept
correction. The correction assumes sampling is independent of predictors
conditional on class. The notes distinguish this exact probability identity
from finite-sample, penalized, or misspecified logistic fits. A subsequent
playground lets readers change the negative inclusion rate and switch between
probabilities and log odds.

The efficiency chapter shrinks one million observations to twenty thousand
and compares batch composition. Its fifty-fold reduction describes observation
visits per pass, not a promised wall-clock speedup. It then distinguishes
balanced training from importance weighting that restores the population
objective in expectation.

The learning chapter uses an analytically optimized classifier that can choose
only one of two binary predictors and fit a free probability in each branch.
Signal A has TPR 0.5 and FPR 0.001; B has TPR 0.9 and FPR 0.1. At a 1% event
rate, population log loss chooses A; equal class influence chooses B. The loss
curves and selected model change as the class influence moves. Evaluation at
the population prevalence compares AUC, recall, precision, and corrected
population log loss. The example demonstrates a change at convergence without
claiming that balancing universally improves a model.

All populations are synthetic and all plotted losses are analytic. The page
links further reading by Fithian and Hastie, Byrd and Lipton, and Kang et al.

## Preview and hosting

From the repository root, serve `docs/` with a static web server and open
`/imbalanced-data/`. For example, for a local development preview:

```sh
python3 -m http.server 8765 --directory docs
```

All asset paths are relative, so the page also works under
`/pols4728/imbalanced-data/`. The existing GitHub Pages workflow publishes
`docs/` after checks pass on `main`. After merging and deployment, the course
homepage links to `https://alexanderthclark.github.io/pols4728/imbalanced-data/`.

## Source and validation

- `index.html` contains the title, chapter frames, complete prose, sampling
  playground, and evaluation.
- `model.js` contains the probability transforms, exact stump losses and
  population metrics, and batch-composition calculations.
- `scrolly.js` maps scroll position to chapter progress and renders the
  logistic correction scenes.
- `chapter-scenes.js` renders the efficiency and learning scenes.
- `app.js` handles the sampling playground and evaluation figures.
- `style.css` and `scrolly.css` provide the course typography, palette, and
  responsive pinned layout.
- `fonts/` contains locally served Latin Modern Roman fonts, attribution, and
  the GUST Font License.

The chapter dots and controls support keyboard use. Motion can be switched
off, and the page respects reduced-motion preferences. Expandable notes
preserve the complete reasoning alongside the animated chapters. Desktop and
narrow-phone checks cover the pinned frames, intermediate line positions,
backward scrolling, the A/B model switch, and the motion control.

Run the analytic regression checks from the repository root:

```sh
node --test tests/imbalanced-data.test.mjs
```

The checks verify odds correction and its sign, round trips and extreme
probabilities, optimized losses computed independently from joint class/branch
probabilities, model selection, AUC, population precision, and batch
composition. The PR workflow also checks all four JavaScript files for syntax
errors and runs these checks with the existing lesson tests.


The appended decision-tree chapter uses a separate Gini example: A routes 10% of positives and 0% of negatives left; B routes 90% and 10%. Exact calculations select A at 1% training prevalence and B at 50%. A second animation and slider hold the original gains fixed while moving `min_impurity_decrease`; A or no split can result, never B. The text distinguishes candidate admissibility (`min_samples_leaf`) from gain ranking and states the limited implication for deep forests. Existing logistic, efficiency, and log-loss chapters remain intact.

`tree-model.js` computes Gini gains and child shares. `tree-scenes.js` adds five reversible scroll stages and two keyboard-operable playgrounds. `node --test tests/imbalanced-tree.test.mjs` verifies the results against independent integer child counts. The new imblearn section introduces samplers, binary sampling ratios, fold-local pipelines, and per-tree balanced forests. `imblearn_demo.py` is a complete downloadable CV example with an untouched test set and explicit forest sampling settings.

This update's chart SVGs were inspected at desktop and phone sizes. Full browser interaction QA was unavailable in the editing environment.
