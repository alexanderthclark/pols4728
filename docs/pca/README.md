# PCA: a change of basis

A nineteen-step scrollytelling lesson first defines the principal direction:
`u₁ := argmax_{‖u‖ = 1} Var(Xu) = argmax_{‖u‖ = 1} ‖Xu‖`.
For centered data, the same direction maximizes score length because variance is
its squared length divided by the fixed `n − 1`. The first eighteen steps share
one pinned visual stage in a centered column, with a short explanation above
the diagram. Scrolling advances the geometric operation; scrolling backward
reverses it. The main sequence introduces one idea per beat, with at most one
principal display equation. The final explorer and reference notes return to
ordinary document flow.

| Steps | Development |
| --- | --- |
| 1–5 | State the goal, follow one point to its signed horizontal score, collect all five scores in `Xu`, then compare horizontal, vertical, and diagonal directions separately. |
| 6–7 | Hold PC1 fixed, add its perpendicular direction, and express the same observations in the new basis. |
| 8–10 | Introduce the thicker 2D cloud, repeat the full basis change, then discard the second score while holding the principal basis fixed. |
| 11–15 | Find PC1 in 3D, subtract its contribution, turn the residual plane face-on, find PC2 within it, and subtract PC2 to reveal the PC3 line. |
| 16–18 | Return to the original cloud, reconstruct with two retained components, automatically add PC1 then PC2 then PC3, and scale a unit SVD pattern into its score column. |
| 19 | Compare the examples in the explorer. |

The variance budget uses the original dataset's total as a fixed reference.
A full basis change redistributes variance between coordinates without changing
that total. During dimension reduction or component subtraction, the removed
portion remains visible alongside the retained portion. Thus the budget still
accounts for the original total, while the remaining data have less variance.

The thicker cloud has principal score variances 9 and 1, with zero covariance.
Its second score is nonzero, unlike the exact-line example. The 3D sequence
keeps the residual plane used to search for PC2 distinct from the plane spanned
by PC1 and PC2 used to reconstruct the original observations. Reconstruction
and rank-one pieces connect to SVD through `zⱼ = dⱼpⱼ`: a unit-length pattern
scaled by a singular value gives the corresponding score column.

A closing section connects the examples to the geometric interpretation of PCA:
full basis changes preserve distances, while dimension reduction discards score
coordinates. It distinguishes the stretching map `u → Xu` from rotating the
observations, explains optimal squared-error reconstruction and the global
meaning of retained variance, and covers scale sensitivity, unusual observations,
uncorrelated versus independent scores, and linear versus curved structure.
Short paragraphs link back to the animated examples. The explorer's final
navigation button leads to this section, with direct `#takeaways` links and
browser history support.

An optional `#mathematical-details` section follows the takeaways. Five native
expandable disclosures preserve the covariance derivation, eigenvectors and
explained variance, the 2D cloud's variance formula, exact 3D directions and
residual calculations, and the full SVD with matrix dimensions. The original
`#covariance-form` and `#principal-direction` anchors remain available. This
keeps `Var(Xu) = uᵀSu`, `X = PDUᵀ`, `XU = PD`, and
`Var(Xuⱼ) = dⱼ²/(n − 1)` available without interrupting the geometric sequence.

The opening title and diagrams use real text and native SVG. The page is static
HTML, CSS, and JavaScript modules, with local Latin Modern fonts, native MathML,
and no external runtime or build step. Static and changing expressions share
structured MathML; `notation.mjs` builds the changing labels and captions.
SVG labels use real positioned subscripts and superscripts, with italic
variables in the same mathematical font as the equations.

## Preview

From the repository root:

```sh
python3 -m http.server 8898 --bind 127.0.0.1 --directory docs
```

Open <http://127.0.0.1:8898/pca/>. During the pinned-stage review, the revised
preview uses port 8898 and the saved preceding version uses
<http://127.0.0.1:8900/pca/>. All preceding work is committed at `4d61884` and
preserved on `codex/pca-before-cinema`, so this presentation change can be
compared or reverted independently. The earlier storyboard baseline at
`4b42c2f` remains available on `codex/pca-storyboard-baseline` and at
<http://127.0.0.1:8899/pca/>. To inspect the pinned-stage changes:

```sh
git diff codex/pca-before-cinema -- docs/pca
```

GitHub Pages publishes the `docs/` site after checks pass on `main`; the new
lesson's published path is <https://alexanderthclark.github.io/pols4728/pca/>
after merging. Local previews do not merge or publish the revision.

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

Removing PC1 puts every residual in `x₁ + x₂ + x₃ = 0`, with total
sample variance 1.25. A reference basis rotated 30° away from PC2/PC3 makes
the face-on direction search visible: PC2 occurs at φ = −30° (variance 1),
and PC3 at 60° (variance 0.25). The subtraction animation removes the full PC1
contribution, leaving total residual variance 1.25. Removing PC2 as well leaves
only the PC3 line.

The residual plane differs from the retained plane. Two components span
`x₁ + x₂ − 2x₃ = 0`, retain `40/41` (about 97.6%) of
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

- Scroll through the pinned, single-column sequence or use Previous/Next.
  Left/right arrow navigation applies outside interactive controls. Step links
  and browser Back/Forward restore the scene and reading location. Keyboard
  navigation focuses the visible destination heading; resizing preserves that
  focus. Each diagram and its controls retain their own accessible IDs and state.
- The first eighteen explanations occupy the same area above one shared plot
  stage. The largest prose height is measured for the current viewport, so
  changing to a shorter explanation does not shift the plot. Original narrative
  sections retain the chapter anchors and scroll distances; their source text
  remains the fallback for printing and reading without JavaScript. The explorer,
  takeaways, and mathematical notes remain in normal reading order after the
  pinned sequence releases.
- The opening follows one highlighted observation to its signed horizontal
  score before building the full `Xu` column. Horizontal, vertical, and diagonal
  comparisons remain separate scroll beats. Hollow circles mark observations;
  crosses and connecting segments mark projections. One calculation connects
  squared score length, the sample-variance denominator, and score variance.
- Adding the second basis direction keeps PC1 fixed. The paired coordinate
  views preserve every point and use equal units. In the reduction step, the
  basis stays fixed while the second coordinate shrinks to zero, so the motion
  demonstrates discarding a coordinate rather than finding a direction.
- In 3D, PC1 subtraction collapses the points into the perpendicular residual
  plane. A rigid camera rotation turns that same plane and its points face-on.
  The search for PC2 then takes place within the plane; a second subtraction
  collapses its residual points onto the PC3 line. The highlighted observation
  remains identifiable through these operations.
- Returning to the original 3D cloud introduces the retained reconstruction
  plane. The next animation adds PC1, then PC2, then PC3 along their respective
  directions, pausing after each contribution. A fixed camera and variance
  budget show how the pieces recover the complete observations. The guided
  SVD animation uses PC1 and scales a unit pattern into its actual score column
  on one fixed vertical scale.
- The guided sequence has no sliders, angle presets, or model selectors.
  Scrolling controls each authored demonstration directly and reversibly.
  Each scene holds its starting state briefly, eases through its operation,
  then holds the completed state. Replay and Previous/Next provide timed
  playback until the reader resumes scrolling. The final explorer holds the
  manual angle, camera, and reconstruction controls, plus individual 3D
  contributions and all three SVD patterns. Guided scenes always restore their
  authored basis and camera; the explorer preserves the reader's choices.
- Replay repeats the transition on the fixed stage. Resizing preserves the
  guided scene's proportional scroll position, while ordinary scrolling can
  revisit any intermediate state. Component subtraction has no removal slider
  or toggle. Reduced motion displays each scene's completed endpoint immediately.
- Basis directions have direct labels and distinct solid, dashed, and dotted
  strokes. Original/reconstructed points use circles and crosses.
- Desktop and phone layouts keep short explanations above the diagram in one
  column, with compact readouts below it. SVG labels are enlarged on phones.
- Reduced motion shows each transition's endpoint immediately. The complete
  narrative and assumptions remain readable without JavaScript.

The palette follows `design/tokens.json`, the web guidance in
`design/README.md`, and the visual reference in `design/guide.pdf`.
Latin Modern files are accompanied by `fonts/GUST-FONT-LICENSE.txt`.
These choices and visual review do not establish full accessibility compliance.

## Files and validation

- `index.html`: opening, nineteen narrative beats, takeaways, optional mathematical details, MathML, and controls.
- `style.css`: course typography and desktop/mobile layouts.
- `cinema.css`: the shared pinned stage, stable caption space, viewport fitting,
  and print fallback.
- `model.mjs`: deterministic points, projections, variances, basis changes,
  reconstructions, deflation, and SVD factors for the known principal basis.
- `svd-view.mjs`: score columns, normalized patterns, and singular values.
- `app.js`: SVG figures, controls, scroll scenes, and navigation.
- `scroll-timeline.mjs`: pure scroll-to-scene mapping with reversible progress,
  start/end holds, and smooth interpolation between them.
- `tests/pca-scroll-timeline.test.mjs`: scene boundaries, clamped sequence ends,
  interpolation, reverse scrolling, proportional resizing, and invalid inputs.

```sh
node --check docs/pca/model.mjs
node --check docs/pca/app.js
node --check docs/pca/svd-view.mjs
node --test tests/pca-model.test.mjs
node --test tests/*.test.mjs
```

The full repository suite has 275 tests. The numerical tests use independently
specified covariance matrices and expected variances. They check unit direction scaling, full-basis inversion
and distance preservation, 2D and 3D decorrelation, retained variance, and
reconstruction loss. They also check residual orthogonality and covariance,
sequential removal, additive component pieces, and the SVD factorization and
variance identity. Browser review covers desktop and narrow layouts, the
45°/135° cases, the single-score and full-column sequence, fixed-basis
coordinate dropping, variance budgets, 3D subtraction and face-on camera motion,
PC2 removal, sequential component assembly, explorer reconstruction controls,
residual plane angles, all three SVD
columns on a fixed scale, optional disclosures, and step/history navigation.
Pinned-stage review also checks a stationary plot across scroll positions,
forward and reverse geometric motion, Replay followed by scrolling, responsive
caption measurements, visible heading focus, and release into the explorer.
