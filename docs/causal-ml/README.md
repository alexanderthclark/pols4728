# Causal machine learning

A connected POLS 4728 collection, linked from the course homepage. Available lessons are AIPW (16 scenes), partially linear DML (13 scenes), DML for IV (17 scenes), causal trees (16 scenes), S/T/X meta-learners (16 scenes), DR/R meta-learners (18 scenes), and causal forests (19 scenes). All seven collection entries are available.

## Authoring

- `docs/causal-ml/lessons.mjs`: scene text, equations, full explanations, references, and collection map.
- `docs/causal-ml/model.mjs`: exact constructed examples and calculations.
- `docs/causal-ml/visuals.mjs`: scroll-driven SVG geometry; all mathematical labels use the bundled KaTeX renderer.
- `docs/causal-ml/advanced-lessons.mjs`, `advanced-model.mjs`, and `advanced-visuals.mjs`: IV and causal-tree content, numerical constructions, and geometry, using the shared renderer and scroll engine.
- `docs/causal-ml/meta-lessons.mjs`, `meta-model.mjs`, and `meta-visuals.mjs`: the five meta-learners, their actual toy fits, and their scroll-driven transformations.
- `docs/causal-ml/forest-lessons.mjs`, `forest-model.mjs`, and `forest-visuals.mjs`: the causal-forest progression, exact local-moment calculations, and adaptive-neighborhood geometry.
- `docs/causal-ml/app.mjs`: native scrolling, chapter navigation, reversible states, responsive layout, and reduced motion.
- `docs/causal-ml/style.css`: shared course typography and layout.
- `scripts/build-causal-ml.mjs`: emits fifteen HTML pages and validates every scene with the real math renderer.

Run `node scripts/build-causal-ml.mjs` after content changes and `node --test tests/causal-ml*.test.mjs` after numerical changes. A static server can serve `docs/`; open `/causal-ml/`. The reading versions contain all diagrams and pre-rendered math and do not require JavaScript. The animated pages accept `?motion=reduce` for completed states at scroll thresholds and `#step-N` to jump to a scene.

## Content and provenance

All numerical examples are constructed, not empirical findings. AIPW uses 16 noiseless observations with deliberately balanced treatment counts; its four nuisance cases average to 4, 4, 4, and 13. PLR uses 50 balanced observations and known nuisance functions; residualization recovers the coefficient 2. Cross-fitting diagrams show data flow rather than fitting learners. The inference scenes state required conditions and do not report research intervals from these arithmetic constructions.

The primary references are linked in each lesson. The course design guide, palette, Latin Modern fonts, and licensed KaTeX distribution were retained from the existing Beyond MLPs course explainer. The original two opening illustrations are native SVGs, registered in the repository's `design/assets.json`.

IV uses 24 factorial observations with known conditional means. The endogenous residual slope is 2.5; the valid-instrument ratio is 2. Nuisance perturbations, weak-first-stage sensitivity, and direct-effect violations are separate calculations. The final three scenes switch to a distinct binary-IV population: LATE is 4 and ATE is 3.8. The IV lesson keeps treatment-mean notation `m` from PLR and uses `q` for the instrument mean.

Causal trees begin with a 64-observation balanced realization of a hypothetical randomized experiment. Outcome prediction favors the baseline-knowledge split; the honest effect criterion initially favors engagement. Additional prognostic splits improve precision without changing effects. Structure and estimation realizations are separate; the final leaf estimates are 6.4 and 1.8, with standard error approximately 0.60. The constant-effect selection example and smooth-effect target example are clearly separate. Minimum counts and normal intervals are illustrative calculations, not recommendations or coverage guarantees. The criterion follows Athey and Imbens with equal-size structure and estimation samples and constant treatment probability one half.

The meta-learner lessons share 80 observations at five covariate values, with equal group sizes, treatment probabilities from 0.25 to 0.75, and balanced outcome noise. Quadratic outcome learners, linear effect learners, and a penalized S-learner are actually fitted by a small deterministic least-squares routine. These transparent learners illustrate the architectures rather than recommending a model or software choice. The S/T/X lesson contrasts sharing one response model, separating the treatment arms, and regressing imputed contrasts. Its X-learner labels follow the arm used for effect regression: propensity multiplies the control-arm effect fit, and one minus propensity multiplies the treated-arm effect fit.

The DR lesson separates conditional double robustness from second-stage learning error. Deliberately incorrect outcome and propensity functions yield a wrong CATE line, 6.5−5x, despite an accidentally correct overall average of 4. The R lesson uses the marginal outcome mean and propensity, demonstrates the weighted quotient identity, and distinguishes local orthogonality from two-sided double robustness. The losses for its three fixed candidates are 1, 1.39375, and 4.89375. Sensitivity calculations, known-nuisance fits, and data-flow schematics are explicitly identified. Nuisance cross-fitting does not replace held-out evaluation of the effect model or justify automatic CATE confidence intervals.

Causal forests use 240 observations across fifteen covariate cells. Three specified partitions isolate the normalized neighborhood-weight calculation; they are not claimed to be a trained production forest. A separate gradient split comparison is computed from the data. The supported query (0.3, 0.5) has true effect 4.8; its forest-weighted local regression has slope approximately 4.500787, compared with a mean tree slope of 4.516701. The estimator retains a local intercept. Narrowing-window, overlap, Monte Carlo, and copied-tree calculations distinguish localization, treatment information, algorithmic variation, and sampling uncertainty. Averaging the forest predictions gives 4.010822; the illustrated correct-propensity AIPW correction returns the true ATE of 4. Data-flow diagrams distinguish per-tree honesty, out-of-bag prediction, nuisance cross-fitting, and independent evaluation.

The IV, tree, meta-learner, and forest title illustrations reuse the registered native character anatomy with newly authored monochrome props. All seven lesson illustrations are registered in `design/assets.json`.

## Verification

Reviewed through November 11, 2026 (America/New_York). Thirty-one numerical checks verify the stated causal targets, cancellations, orthogonality perturbations, IV identification failures, tree criteria, leaf uncertainty, fitted meta-learners, loss identities, and forest-weighted local estimation. The build checks 115 scenes at five progress values in desktop and mobile layouts (1,150 rendered states), rejects invalid math, and checks nonfinite geometry.

The actual browser was reviewed at desktop size, 390 × 844, and 390 × 660. Final DOM geometry checks covered all 29 scenes at 1280 × 800 and 390 × 660 with no offscreen labels, page overflow, or math-rendering errors. Representative title, prediction, correction, contribution, residual, and slope scenes were visually inspected. Actual forward and reverse scrolling changed and restored geometry; the graph remained pinned. Keyboard PageDown advanced the lesson. Reduced motion displayed completed states. No browser errors or warnings were observed. These are targeted checks, not an accessibility certification.

The IV/tree extension adds browser checks for all 33 scenes at 1280 × 800 and 390 × 660, including label-collision checks. Both lessons were scrolled forward and backward at each size: geometry changed, the stage stayed pinned, and reverse scrolling restored the earlier SVG. Keyboard advancement and completed reduced-motion states were checked. The static reading versions retain pre-rendered figures and mathematics without scripts.

The meta-learner extension adds the same browser geometry and label checks for all 34 new scenes in both layouts. Actual forward and reverse scrolling changed and restored X-learner imputation and R-learner residual geometry, with the stage pinned at both screen sizes. Keyboard navigation, reduced-motion states, reading pages, and collection links were checked. The content and arithmetic received a separate mathematical review against the primary papers. The repository suite now passes 464 tests, with all five separate word2vec checks and 63 syntax checks also passing.

The forest extension adds browser geometry and label checks for all 19 scenes at 1280 × 800 and 390 × 660. Residualization and moving-query scenes were scrolled forward and backward at both sizes; their geometry and text restored exactly and the stage stayed pinned. Keyboard navigation, completed reduced-motion states, the script-free reading version, and the hub and preceding-lesson links passed. The opening artwork, neighborhood maps, split calculation, local regression, and uncertainty illustrations were visually inspected. A separate mathematical review caught and resolved an unsupported query and a residual-axis bound. All local page references and registered artwork hashes were checked.

The collection lives under `docs/causal-ml/`. Serve `docs/` locally and open `/causal-ml/`. The existing GitHub Pages workflow publishes `docs/` after changes are merged to `main`; this feature branch does not deploy the live course site.

The generated HTML pages are tracked so GitHub Pages can serve them directly. The existing check workflow rebuilds them and verifies that the committed pages are current. Run the authoring commands from the repository root.
