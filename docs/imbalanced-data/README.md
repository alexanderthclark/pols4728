# Imbalanced Data

The opening title includes the transparent monochrome high-wire cartoon in
`assets/high-wire-refined.png`, with lower arms and a clearer grip on the pole. The original remains available in `assets/high-wire.png`.

A buildless scrollytelling lesson based on Alexander Clark's Applied Issues and Logistic Regression notes. Serve `` with any static web server. No runtime dependencies or external data calls.

The title leads directly to seven scrolling stages: the model's linear log-odds assumption, probabilities as within-group counts, cancellation of the common total, negative-only retention, expansion of the logarithm, the intercept shift, and prior correction. Large fractions lead into two grouped class ratios with 1% and 50% positives. Retaining a share s of negatives adds −log(s) to the true log odds; the sampled group total also cancels. Expected count bars thin as the reader scrolls. Two points illustrate the assumed line rather than test linearity. Equal-probability sampling remains in expanded notes. The rare-events connection cites King and Zeng (2001), separating sampling correction from finite-sample bias.

The next chapter gives a quicker, two-stage comparison for an intentionally misspecified linear logistic model with curved true log odds. Its slope changes from 2.230 to 1.933 after retaining all positives and 10% of negatives. Intercept correction cannot recover the full-data fit. Inverse-selection weighting does recover its objective. Results minimize population cross entropy, rather than simulating a random sample; complete assumptions and tables remain in expanded notes.

Four new slides show executed finite-sample logistic regressions before the decision-tree chapter: the two random training datasets, the correctly specified fit, the curved fit, and 500 further independent downsampling draws from each fixed dataset. Both fits are unpenalized; every negative has a 10% retention probability and every positive is kept. Coefficients and untouched-test log loss are displayed. The repeated-draw range describes conditional downsampling variation, not a population confidence interval. Raw counts, results, repeated-fit CSV, and the calculation script are downloadable.

Decision-tree split ranking follows the executed runs. A routes 50% of positives and 0% of negatives left; B routes 90% and 10%. Exact entropy calculations (natural logs, nats) select A at 1% training prevalence and B at 50%. A gain cutoff at the original mixture selects A or no split, never B. Additional training-efficiency, restricted-model learning, evaluation, and optional implementation examples follow the tree playground.

`scrolly.js` drives reversible, full-width pinned chapters. `logistic-scenes.js` renders the grouped log-odds walkthrough. `misspec-model.js` solves the expected-loss fits; `misspec-scenes.js` presents them. `tree-model.js` and `tree-scenes.js` provide entropy calculations and keyboard-operable controls. Motion can be switched off; expanded chapter notes preserve prose and numerical tables.

From the repository root, run `node --test tests/imbalanced-*.test.mjs`. These check the sampling identity, exact numerical fit score equations, correctly specified control, equal sampling, independent cross entropy, inverse-selection weighting, and independent child-count gains.

`node --test tests/imbalanced-scenes.test.mjs` checks intermediate and completed animation frames at narrow and wide sizes, comparing rendered logits and coefficient labels against independently computed counts and line geometry.

`sampling-data.js` contains the recorded experiment results; `sampling-scenes.js` renders the numerical-results slides.

The finite-run results comprise 1,004 executed unpenalized logistic fits. Independent audits reproduced the generated training/test data and all retention masks, checked every fit against separate score equations/Newton fits, and recomputed the untouched-test losses and repeat quantiles.

`tree-leaves.js` shows both children of each candidate with expected positive/negative counts, within-leaf positive share, and class-composition bars. The walkthrough begins at 1% positives; the continuous class-balance playground updates leaf balances alongside entropy gains. Counts hold the same 100 positives fixed while thinning negatives.

`tree-entropy-chords.js` joins child posterior proportions on the binary entropy curve. The chord at the parent’s prevalence is the size-weighted child entropy; the vertical gap is information gain. Scroll rebalancing recomputes both chords, and the continuous playground shares its prior control with the leaf diagrams and gain bars. A magnified view makes the rare-class gain difference legible.

`tree-colors.js` keeps candidate identities consistent: A is purple and B is teal across cards, chords, and gain plots. Positive and negative counts retain their separate burnt-orange and navy class colors.

The curved finite-run slide highlights a scoped takeaway: fitting the original imbalanced data wins here on untouched natural-prevalence test log loss. Expanded notes explain the population risk target under misspecification, inverse-selection weighting, and why this result does not settle recall or tree-learning objectives. The displayed downsample has 20.54% positives, not a 50/50 mixture.

`tree-rebalance.js` adds a scroll-driven negative-thinning slide before entropy chords. It keeps all 100 positives and reduces negative retention from 100% to 1/99, updating both candidate leaf diagrams. A’s left leaf stays pure; its right denominator shrinks. The walkthrough has no balance tabs; scrolling advances the entire experiment, and the chord stages replay the same thinning after explicitly returning to the original mixture.

The course page is available directly at
`https://alexanderthclark.github.io/pols4728/imbalanced-data/` and is intentionally
not linked from the course homepage. Local font attribution and the GUST Font
License remain in `fonts/`.
