# PCA: a change of basis

A thirteen-step scrollytelling lesson begins with five points exactly on
`x₁ = x₂`. Rotating a perpendicular unit basis introduces a score, the column
`Xu`, and its sample variance. The lesson connects that variance to `uᵀSu`,
chooses principal directions, and shows the same observations in both coordinate
systems. A thicker 2D cloud separates a full basis change from dimension
reduction. A 3D cloud develops the choice between retaining a line, a plane,
and all three coordinates.

The opening title and diagrams use real text and native SVG. The page is static
HTML, CSS, and JavaScript modules, with local Latin Modern fonts, native MathML,
and no external runtime or build step.

## Preview

From the repository root:

```sh
python3 -m http.server 8898 --bind 127.0.0.1 --directory docs
```

Open <http://127.0.0.1:8898/pca/>. GitHub Pages publishes the `docs/` site after
checks pass on `main`; the new lesson's published path is
<https://alexanderthclark.github.io/pols4728/pca/> after merging.

## Examples and conventions

All observations are constructed, centered data. Rows of `X` are observations;
columns are variables. Sample variances and covariances consistently use
`n − 1`. Centering does not silently standardize the variables.

| Example | Observations | Principal score variances | Total variance |
| --- | ---: | --- | ---: |
| Exact line | 5 | 5, 0 | 5 |
| Thicker 2D cloud | 24 | 9, 1 | 10 |
| 3D cloud | 24 | 9, 1, 0.25 | 10.25 |

The line uses `(-2, -2)` through `(2, 2)` in unit increments. Its sample
covariance matrix has four entries equal to 2.5. At 45°, scores are
`√2 × x₁`, and all second scores are zero.

The 2D cloud has sample covariance `[[5, 4], [4, 5]]`, so the rotating
frame's variances are `5 + 4 sin(2θ)` and `5 − 4 sin(2θ)`. Its score
covariance is `4 cos(2θ)`. Retaining PC1 keeps 90% of total sample variance;
the sum of squared reconstruction errors is 23.

The 3D basis has columns:

```text
u₁ = (1, 1, 1) / √3
u₂ = (1, −1, 0) / √2
u₃ = (1, 1, −2) / √6
```

Two components span `x₁ + x₂ − 2x₃ = 0`, retain `40/41` (about 97.6%) of
the total sample variance, and give reconstruction SSE 5.75. The third score
is constructed from a centered quadratic function of the first score; the
uncorrelated scores are not claimed to be independent.

The model computes scores and summaries from the actual points. The 2D plot
uses equal x/y units; paired coordinate plots use the same scale. A unit arrow
marks each basis vector, with longer lines marking its axis. Turning the 3D
view is an orthographic camera rotation, separate from changing the basis.
Original 3D coordinates and principal score coordinates can be inspected
separately. Reconstruction sets omitted principal scores to zero before
returning to the original coordinates.

## Interaction and access

- Scroll, choose a named lesson step, or use Previous/Next. Left/right arrow
  navigation applies outside interactive controls. Step links and browser
  Back/Forward restore the reading location. Step selection lands immediately
  beside the matching figure rather than animating across intervening chapters.
- Rotate 2D directions with a labeled range control or the 0°, 45°, and 135°
  presets. Choices are retained separately for each step.
- In 3D, turn the view and choose how many components to retain. The final
  explorer switches between all three datasets and coordinate/reconstruction
  views.
- Basis directions have direct labels and distinct solid, dashed, and dotted
  strokes. Original/reconstructed points use circles and crosses.
- Phone layouts pin a shorter diagram above the narrative; short landscape
  viewports let it scroll normally. Compact SVG labels are enlarged.
- Reduced motion disables smooth navigation and basis animation. The complete
  narrative and assumptions remain readable without JavaScript.

The palette follows `design/tokens.json`, the web guidance in
`design/README.md`, and the visual reference in `design/guide.pdf`.
Latin Modern files are accompanied by `fonts/GUST-FONT-LICENSE.txt`.
These choices and visual review do not establish full accessibility compliance.

## Files and validation

- `index.html`: opening, complete narrative, MathML, and controls.
- `style.css`: course typography and desktop/mobile layouts.
- `model.mjs`: deterministic points, projections, variances, basis changes,
  and reconstructions.
- `app.js`: SVG figures, controls, scroll scenes, and navigation.

```sh
node --check docs/pca/model.mjs
node --check docs/pca/app.js
node --test tests/pca-model.test.mjs
node --test tests/*.test.mjs
```

The twelve numerical tests use independently specified covariance matrices
and expected variances. They check unit direction scaling, full-basis inversion
and distance preservation, 2D and 3D decorrelation, retained variance, and
reconstruction loss. Browser review covers desktop and narrow layouts, the
45°/135° cases, coordinate views, 3D camera/reconstruction controls, and
step/history navigation.
