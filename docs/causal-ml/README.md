# Causal machine learning

A connected POLS 4728 collection, linked from the course homepage. The first release contains the navigation page, AIPW (16 scenes), and partially linear DML (13 scenes). The remaining five entries are explicitly marked planned.

## Authoring

- `docs/causal-ml/lessons.mjs`: scene text, equations, full explanations, references, and collection map.
- `docs/causal-ml/model.mjs`: exact constructed examples and calculations.
- `docs/causal-ml/visuals.mjs`: scroll-driven SVG geometry; all mathematical labels use the bundled KaTeX renderer.
- `docs/causal-ml/app.mjs`: native scrolling, chapter navigation, reversible states, responsive layout, and reduced motion.
- `docs/causal-ml/style.css`: shared course typography and layout.
- `scripts/build-causal-ml.mjs`: emits the five HTML pages and validates every scene with the real math renderer.

Run `node scripts/build-causal-ml.mjs` after content changes and `node --test tests/causal-ml.test.mjs` after numerical changes. A static server can serve `docs/`; open `/causal-ml/`. The reading versions contain all diagrams and pre-rendered math and do not require JavaScript. The animated pages accept `?motion=reduce` for completed states at scroll thresholds and `#step-N` to jump to a scene.

## Content and provenance

All numerical examples are constructed, not empirical findings or browser-trained models. AIPW uses 16 noiseless observations with deliberately balanced treatment counts; its four nuisance cases average to 4, 4, 4, and 13. PLR uses 50 balanced observations and known nuisance functions; residualization recovers the coefficient 2. Cross-fitting diagrams show data flow rather than fitting learners. The inference scenes state required conditions and do not report research intervals from these arithmetic constructions.

The primary references are linked in each lesson. The course design guide, palette, Latin Modern fonts, and licensed KaTeX distribution were retained from the existing Beyond MLPs course explainer. Both opening illustrations are newly authored native SVGs, registered in the repository's `design/assets.json`.

## Verification

Reviewed on November 10, 2026 (America/New_York). Five numerical checks verify causal targets, AIPW cancellation, structural residuals, nuisance-perturbation behavior, and reversible coordinates. The build checks every scene at five progress values in desktop and mobile layouts (290 rendered states), rejects invalid math, and checks nonfinite geometry.

The actual browser was reviewed at desktop size, 390 × 844, and 390 × 660. Final DOM geometry checks covered all 29 scenes at 1280 × 800 and 390 × 660 with no offscreen labels, page overflow, or math-rendering errors. Representative title, prediction, correction, contribution, residual, and slope scenes were visually inspected. Actual forward and reverse scrolling changed and restored geometry; the graph remained pinned. Keyboard PageDown advanced the lesson. Reduced motion displayed completed states. No browser errors or warnings were observed. These are targeted checks, not an accessibility certification.

The collection lives under `docs/causal-ml/`. Serve `docs/` locally and open `/causal-ml/`. The existing GitHub Pages workflow publishes `docs/` after changes are merged to `main`; this feature branch does not deploy the live course site.

The generated HTML pages are tracked so GitHub Pages can serve them directly. The existing check workflow rebuilds them and verifies that the committed pages are current. Run the authoring commands from the repository root.
