# Gradient Boosting

A scrollytelling chapter about gradient boosting. After the title, twelve scenes
build one two-observation example: zero predictions, the residual vector,
the identical-input update constraint, equal-MSE contours, the fitted update,
orthogonality, a changed loss, negative-gradient targets, a fitted tree, repeated
descent, the constrained optimum, and a loss playground.
Scrolling into a step starts its animation. Gradient terminology enters only after asymmetric loss exposes a
limitation of raw residual fitting. A full-width transition introduces varying
inputs, followed by eight scenes developing the four-observation example from sections
1.4.2–1.4.3 of the boosting notes. Four later scenes compare squared error,
Huber loss, and asymmetric squared loss through targets, splits, leaf values,
and fitted paths. The chapter ends with the fitted-path comparison. A compact
footer retains the sources, optional XGBoost reading, and geometry companion link.

The opening figure occupies the full story width with sparse floating
annotations, no right-hand pane, no sliders, and no numerical disclosure.
A brief fixed asymmetric-loss equation floats in during the contour change,
then disappears. Loss buttons and three animated prediction actions support
the final playground. Later examples retain their controls tray. Each chapter has a short summary;
the original explanation and equations remain under “Why this works.”
“Show numbers” reveals the numerical vectors without shrinking the plot.
Scrolling follows the cards beneath the pinned figure at every viewport size.
Later phone scenes use a smaller plot and a scrollable controls tray so navigation
remains visible. Previous/Next and keyboard arrows also enter each animation;
a small Replay button repeats it. Reduced motion shows each endpoint directly.

The chapter is the second item on the course homepage. GitHub Pages publishes
it from `docs/boosting/` after the checks pass on `main`.
To preview, serve the repository root and open `/docs/boosting/`.
No build step or external JavaScript dependencies are required.

## Construction

- `index.html` contains the main explanation, numerical example, controls,
  general algorithm, and loss comparison.
- `story.js` follows twenty-five narrative steps and handles the transition,
  split inspection, update progress, learning rate, later rounds, and navigation.
- `prediction-view.js` retains the reusable three-observation vector and
  equal-loss circle renderers outside the main story, with consistent vectors,
  gradient norms, step lengths, and losses.
- `restricted-view.js` shows two observations with identical inputs and
  loss contours in prediction space. Shared-prediction and loss controls compare
  ordinary residuals with negative gradients and the loss-optimal shared tree
  correction. It supports half-squared error, asymmetric squared error with
  continuous τ, Huber loss, and fourth-power error.
- `worked-example.js` builds immutable gradient-boosting states using the
  unchanged `../shared/boosting-math.js` stump-fitting module.
- `worked-view.js` renders response plots, gradients, candidate split tables,
  prediction updates, later fitted rounds, and the general learning loop.
- `loss-example.js` refits loss-aware gradient boosting and the ordinary
  residual-fitting shortcut from the same loss-optimal constant. Huber uses
  clipped gradients and original-loss minimizing leaf corrections; asymmetric
  squared loss uses signed weights. The normalization agrees with half-squared
  loss at τ = 0.5.
- `loss-view.js` shows target bars, candidate SSEs, leaf-loss curves, and
  prediction/loss histories. The seven-point high-outcome case defaults to
  Huber δ = 1; the original four-point data remain available.
- `style.css` follows the course design guide and supports phone layouts.
- `geometry.html`, `geometry.js`, and `geometry.css` preserve the earlier
  three-observation prediction-space construction as a linked supplement.

The main example uses `x = [1, 2, 3, 4]`, `y = [0, 2, 1, 5]`, and mean
prediction `F0 = 2`. With learning rate 1, the first two stumps split at 3.5 and
1.5, and total half-squared loss falls from 7 to 1 to 1/3. The gradient has
one entry per training prediction; inputs and outcomes stay fixed. Split
comparisons use residual-fitting SSE, which differs from the model's
half-squared loss. A shared leaf correction can worsen an individual
prediction while reducing total loss.

At learning rate 0.5, both rounds are rebuilt from the baseline. Their losses
are 7, 5/2, and 11/8. The second round ties at cuts 1.5 and 3.5; the first
sorted cut wins. The later-round control continues the same four-observation
model through 24 freshly fitted corrections rather than introducing a new
example.

Negative-gradient targets equal ordinary residuals for half-squared error.
For general differentiable losses, fitting those targets chooses tree regions;
leaf corrections minimize the original loss within each region. The other-loss
examples show Huber and asymmetric squared loss. The footer links to XGBoost
as optional reading; it is not used in the interactive calculation.

The opening uses `x = [1, 1]` and `y = [1, 0]`. It first shows only the zero
predictions and observed outcomes, then adds the full residual `(1, 0)`.
The next step introduces the constraint: identical inputs reach the same tree
leaf, so any update is `(a, a)` and lies on the 45-degree line. Equal-MSE circles
appear in their own step while the predictions remain at zero; the following
animation fits the residual mean.
Starting at `(0, 0)`, fitting the residual mean
adds `(0.5, 0.5)`. This least-squares fit is exactly the Euclidean orthogonal
projection of the residual vector onto the available update line
`span(u)`, where `u = (1,1)`: `h = ((r·u)/(u·u))*u`.
With signed residuals `r_i = y_i-f_i`, MSE is
`0.5*(r_1^2+r_2^2)`, falling from 0.5 to 0.25.
At `(0.5, 0.5)`, the residual `(0.5, -0.5)` is perpendicular to the diagonal;
the next residual fit, or projection, is zero. A right-angle mark and tangent equal-MSE contour
show the geometric condition. A separate ghost prediction probes both directions
along the diagonal and returns to the midpoint. Its loss obeys
`MSE = 0.25 + a^2` after a shared adjustment `(a,a)` from the midpoint,
so every nonzero shared adjustment worsens MSE.

The next scene changes only the loss. Predictions remain at `(0.5,0.5)` and
residuals at `(0.5,-0.5)` throughout. The contours morph continuously from MSE
to asymmetric squared loss with τ = 0.8, and the optimum marker moves along the
diagonal. The equation defines the per-observation loss from the signed residual:
`ell(r) = 0.8*r^2` when `r >= 0`, and `0.2*r^2` when `r < 0`.
The total loss adds the two contributions. It uses fixed coefficients rather
than a general τ control or a separate symbol for shared predictions.
Asymmetric squared loss with τ = 0.8
charges underprediction four times as much as overprediction. The best shared
prediction moves to `(0.8, 0.8)`, with loss 0.16 rather than 0.25. Only after
this counterexample does the following scene introduce the negative gradient:
at `(0.5, 0.5)` it is `(0.8, -0.2)`, whose mean 0.3 supplies a useful correction.
The target comparison keeps predictions fixed at the midpoint. The next scene
fits an actual stump to those targets: identical inputs share one leaf, so
`h(x) = 0.3` for both observations. Writing `g = -grad L`, the fit is now
`h = ((g·u)/(u·u))*u = (0.3,0.3)`. The raw residual still projects to zero at
the midpoint, while the new targets have a nonzero projection.
The full negative gradient and the feasible
tree correction are drawn separately. A learning rate of 0.5 supplies the
smaller update `(0.15, 0.15)`.

Projection describes the Euclidean least-squares fit of a target vector onto
the available update line; the learning rate scales that fit afterward.
Loss contours describe a different object: levels of the original objective.
For a fixed tree leaf partition, least-squares leaf means project a target
vector onto the subspace of vectors constant within each leaf. Searching over
different tree partitions searches across those subspaces, so an unrestricted
tree-structure search is not one orthogonal projection onto a single linear
space. Recalibrating leaves to minimize a different original loss can also
produce corrections different from the least-squares projection.

The repeated-descent animation shows six rounds of computing targets, fitting
the stump, and adding half its correction. Targets and the fitted tree remain
anchored to the start of each round while its update animates; the following
round recomputes them. Earlier prediction vertices remain visible. The shared
prediction follows `F_(m+1)(1) = F_m(1) + 0.5*(0.8-F_m(1))`, ending at 0.7953125 after
six updates. A separate scene inspects its limiting optimum at 0.8; finite
shrinkage steps are not depicted as exact convergence. At that optimum the
full negative gradient `(0.32, -0.32)` is perpendicular to the diagonal, so the
projection and fitted tree correction are zero. The raw residual `(0.2, -0.8)` has mean -0.3;
following it would move away from the selected loss's optimum.

Both residual and negative-gradient targets require observed outcomes during
training. The fitted tree learns a mapping from inputs to those targets. Once
fitted, `h(x)` can supply a correction from the input alone, including at a new
observation whose outcome is unknown. The plot distinguishes computing training
targets from evaluating the learned function rather than implying gradients
are available from inputs alone.

The final opening scene provides MSE, fixed asymmetric squared loss with
τ = 0.8, Huber δ = 0.25, and fourth-power error `r^4/4`. Its initial
state is MSE at `(0.5, 0.5)`. Native MathML with the bundled STIX Two Math
font typesets the floating equations, with stacked fractions, true scripts,
and stretching case braces. Formulas wrap between the residual definition
and loss, retaining their mathematical structure. A brief floating equation
follows the selected
loss. Asymmetric loss weights positive residuals by τ and negative residuals
by `1-τ`; its best prediction pair is `(τ,τ)` for these observations.
When Huber δ < 0.5, every shared prediction in `[δ, 1-δ]` is optimal; the
loss-optimal correction chooses the closest point and preserves an optimum.
Huber and fourth-power loss generally require a different leaf correction
from the mean fitted gradient target.

Both axes use equal scales and labeled half-unit minor ticks. Current and
optimal contours show the selected loss's geometry. Early scenes draw the
residual at its actual length and contain no gradient terminology; later
comparison arrows disclose their display scaling. Reset and midpoint buttons
support comparisons at the same prediction. The zero start demonstrates the
update; initialization at the selected loss's best constant would already
solve this two-observation problem.

## Checks

From the repository root:

```sh
node --check docs/boosting/story.js
node --check docs/boosting/prediction-view.js
node --check docs/boosting/restricted-view.js
node --check docs/boosting/worked-example.js
node --check docs/boosting/worked-view.js
node --check docs/boosting/geometry.js
node --test tests/*.test.mjs
```

Inspect a wide desktop and narrow phone viewport, navigate all main scenes,
check vector and norm views, compare candidates, change learning rate, adjust
update progress and rounds, compare the four two-observation losses, vary τ,
and inspect both the zero start and residual-mean comparison,
and check keyboard and reduced-motion behavior. Review the optional geometry
link and return route. Text remains readable without JavaScript; visual states
have accessible descriptions and numerical readouts.

## Sources and artwork

The main procedure and worked calculation follow the supplied boosting notes.
The optional geometry follows Kilian Weinberger's Cornell lecture from about
3:20 onward: <https://www.youtube.com/watch?v=dosOtgSdbnY&t=180s>.
The page also links Cornell's lecture notes and the official XGBoost tutorial.

`assets/boosting-swing.png` is the approved transparent cover artwork, registered
with its hash and usage guidance in `../../design/assets.json`. The existing
STIX math font is loaded from the Shapley explainer's local asset. No shared
design rules are changed.

## Loss comparison

The seven-point example uses x = [1,2,3,4,5,6,7] and y = [0,0,1,2,3,4,15].
At the Huber δ = 1 optimum F0 = 2, raw residual fitting selects cut 6.5;
gradient fitting selects cut 3.5. The gradient-target means are −1 and 3/4;
the original-loss minimizing corrections are −5/3 and 3/2. Huber can have
an interval of minimizers; choose its midpoint deterministically.

Both compared paths share the selected loss-optimal constant, stump complexity,
learning rate, data, and round count. The residual shortcut retains ordinary
residual targets and mean-residual updates, and is evaluated using the selected
loss. It is not a comparator for every possible forward stagewise optimizer.
Different targets do not guarantee a lower loss at every finite round.

The later loss comparison reports sums, not MSE. Candidate SSEs are measured on each method's
own targets, not directly ranked against one another. The two-observation
view changes its contours with the selected loss; only squared error has
circular contours. Controls support keyboard use, and round
changes refit from the initial model rather than reusing another loss's trees.

The third dataset has identical inputs x = [1,1,1] and responses y = [0,0,10].
For Huber δ = 1, gradient boosting remains at the optimal constant 1/2 with
loss 9.25. The residual shortcut approaches the outcome mean 10/3, where
Huber loss is 71/6. Since no split is possible, this isolates a persistent
mismatch between the residual shortcut and the selected loss.
