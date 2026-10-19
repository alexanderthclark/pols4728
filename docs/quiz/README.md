# Prediction, selection, and causal adjustment

A sixteen-scene companion to the Fall 2026 POLS 4728 quiz and lecture notes.
The three original question sections retain their wording. Concise scene
summaries carry the visual argument; native expandable sections retain the
full derivations, tables, assumptions and optional experiments.

## The scroll sequence

The graph remains pinned on desktop, portrait phones and short landscape
screens. Scroll position constructs every intermediate state, and scrolling
backward restores it. Each scene holds its completed calculation before the
next one. No clock, autoplay, pause or replay is involved. Readers can also use
the step selector, arrow buttons or left/right keys outside form controls.
Direct links use `#step-0` through `#step-15`; chapter links show the setup.
Resizing retains the active calculation and position within it.

The sixteen scenes cover:

- Fair guesses, searching 32,768 predictors, validation/test separation and
  optional outer evaluation.
- Ten fitted curves; fixed-age bias and variance; averaging over population
  ages; reporting thresholds; and MSE at ordinary and large noise levels.
- Memorization, held-out randomized adjustment, the observed 30-point gap,
  outcome-tree predictions, the 7.5-point outcome-only estimate, and the
  count-weighted adjustment of both outcome and treatment that recovers 10.

Carnies 1 and 3 share the same training draws. The full-range curve panels use
matching scales; threshold details explicitly use different vertical scales.
The moving outcome axis follows the residual subtraction and subsequent zoom.
Every partially accumulated row, product, total and ratio uses exactly the
same included students; four decimal places suffice for the exact dyadic sums.

Compact diagrams use their own geometry rather than scaling an entire desktop
page down. The phone narrative travels through the space below the diagram.
Opening a calculation holds its completed figure while the longer text scrolls.
Reduced motion uses reversible logical thresholds; an optional title-page
button offers the same stepped presentation independently of system settings.

The validation split, fresh curve draws, single-carnie views and noise levels
are optional experiments inside the calculations. Every main teaching point
is available by scrolling alone. Fresh samples persist during forward/backward
scrolling; the numerical reference table is separate from these fresh draws.
Native disclosures keep the worked answers available without JavaScript.
The page uses local assets and fonts, without external runtime dependencies.

## Local preview and verification

From the repository root, serve `docs/` with any static web server and open
`/quiz/`. After merging, the existing Pages workflow publishes the lesson at
<https://alexanderthclark.github.io/pols4728/quiz/>.

```sh
node --check docs/quiz/model.mjs
node --check docs/quiz/motion.mjs
node --check docs/quiz/animate.js
node --check docs/quiz/story.js
node --test tests/quiz-model.test.mjs tests/quiz-motion.test.mjs
node --test tests/*.test.mjs tests/*.test.cjs
```

`model.mjs` separates calculations from presentation. Its probability and risk
tests use independent normal-probability reference values, exact reporting
boundaries and count-weighted tutoring results. The scroll tests check forward
and reverse progress, completed-state holds, reduced-motion boundaries, actual
bar geometry, all 33 prefixes of the student accumulation, moving treatment
labels and responsive residual axes. The independent audit also recomputed
population risks using a different normal CDF and numerical integrator.

Browser review covers every scene at desktop and 320px portrait/667px landscape
sizes, actual forward/backward scrolling, label bounds, optional controls,
keyboard disclosure access, stepped motion, chapter/deep links and resizing.
These checks do not establish full accessibility compliance.
