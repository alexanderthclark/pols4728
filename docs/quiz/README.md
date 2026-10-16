# Prediction, selection, and causal adjustment

A sixteen-scene scrollytelling explanation of all three questions and subparts
in the Fall 2026 POLS 4728 take-home quiz. The original question wording is
available in expandable sections. The page includes the worked answers, not
the quiz's submission instructions or unrelated semester materials.

Each figure plays a finite explanatory sequence when it enters view, with
Pause/Resume and Replay controls. Fair guesses appear one at a time; the
candidate search builds an expected histogram; ten fitted curves draw in
sequence; linked age markers move along the fitted line and its floored report.
The noise comparison moves all three MSEs on a fixed logarithmic scale.
Outcome adjustment physically subtracts each fitted prediction, then zooms to
the residual scale; treatment adjustment slides the same four groups from D
to T before drawing the slope of 10. The final sums accumulate student counts.
Replaying retains the same training draws. Motion pauses offscreen and when
the tab is hidden. Reduced motion displays completed figures immediately.

## Reading the lesson

- Question 1: chance accuracy, searching 32,768 random predictors, a validation /
  test split with no training parameters, and optional outer evaluation.
- Question 2: ten simulated training samples per sampling design; a shared set
  of fits for Carnies 1 and 3; analytic prediction bias and variance; numerical
  population averages; reporting-threshold detail; and MSE at the stated and
  very large noise levels.
- Question 3: in-sample memorization, randomized held-out adjustment, the
  observational 30-point score gap, two tree leaves, the 7.5-point outcome-only
  estimate, and count-weighted adjustment of both outcome and treatment.

The validation slider preserves all 20 observations while changing the split.
The age-curve control selects full-range views or redraws ten training samples;
the threshold detail uses the same first-carnie samples. The noise selector
recalculates population MSE for four multipliers. The initial ten curves are
fresh draws on each visit; the simulation table in the narrative records a
single reference run and remains separate from the exact-probability numerical
averages.

Desktop figures remain beside the narrative as the reader scrolls. Previous /
Next, the step selector, and left/right arrows outside controls navigate the
sixteen steps. Step links use `#step-0` through `#step-15`; browser history
restores explicit step navigation. On screens at most 900px wide or 760px tall, figures sit
inside their corresponding narrative steps, keeping all three curve panels
readable without covering the text. Chapter links work at every width.

The static answers, equations and tables remain readable without JavaScript.
Native MathML uses the existing bundled STIX Two Math font. The lesson uses
the course palette and approved identity artwork, descriptive SVG titles,
direct labels and line patterns, native keyboard controls, and reduced-motion
navigation. These choices do not establish full accessibility compliance.

## Local preview and verification

This is static HTML, CSS and JavaScript, with no build step or external runtime
dependencies. From the repository root:

```sh
python3 -m http.server 8898 --bind 127.0.0.1 --directory docs
```

Open <http://127.0.0.1:8898/quiz/>. The course homepage links to the lesson.
After merging to `main`, the existing Pages workflow publishes it at
<https://alexanderthclark.github.io/pols4728/quiz/>.

```sh
node --check docs/quiz/model.mjs
node --check docs/quiz/motion.mjs
node --check docs/quiz/animate.js
node --check docs/quiz/story.js
node --test tests/quiz-model.test.mjs tests/quiz-motion.test.mjs
node --test tests/*.test.mjs
```

`model.mjs` separates the calculations from presentation. Carnie 3's three
reporting probabilities come from the normal distribution of Carnie 1's fitted
prediction. Adaptive numerical integration averages squared bias and variance
over uniform X. The eight calculation tests use independent normal-probability
reference values, known population risks at two noise levels, exact floor
boundaries, and count-weighted tutoring results. Browser review covers all
scenes, control changes, direct links, navigation, no-JavaScript reading,
reduced motion, and desktop and 320px/390px layouts.
Eight additional clock tests check progress, pause/resume, replay,
cancellation, and protection against stale animation callbacks. Browser checks
also confirm actual movement in all sixteen scenes, frozen positions during
pause, replay from the same data, mobile entry playback, and instant completion
when reduced motion is enabled.
