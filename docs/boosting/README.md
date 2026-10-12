# Gradient Boosting

A scrollytelling chapter about gradient boosting. Its main path
restores the interactive observed-outcome vector, prediction
coordinates, negative gradients, and norm/radius experiment. Three vector
scenes lead into an identical-input example and eight scenes developing the four-observation example from
sections 1.4.2–1.4.3 of the boosting notes. Four additional scenes compare squared error, Huber loss, and asymmetric squared
(expectile) loss through targets, split search, original-loss leaf optimization,
and two fitted paths. The chapter closes with an exponential-loss comparison
and a brief XGBoost extension. The longer geometry sequence remains
available as a companion.

The figure occupies the full story width, with a compact controls tray on
desktop and stacked controls on phones. Each chapter has a short summary;
the original explanation and equations remain under “Why this works.”
“Show numbers” reveals the numerical vectors without shrinking the plot.
Desktop scrolling follows the cards beneath the pinned figure. Phone links
and Previous/Next return to the figure, which scrolls naturally with the page.

The chapter is the second item on the course homepage. GitHub Pages publishes
it from `docs/boosting/` after the checks pass on `main`.
To preview, serve the repository root and open `/docs/boosting/`.
No build step or external JavaScript dependencies are required.

## Construction

- `index.html` contains the main explanation, numerical example, controls,
  general algorithm, and loss comparison.
- `story.js` follows sixteen narrative steps and handles vector experiments,
  split inspection, update progress, learning rate, later rounds, and navigation.
- `prediction-view.js` restores the three-observation vector picture, fitted
  correction, and exact equal-loss circle. It exposes consistent vectors,
  gradient norms, step lengths, and losses for all three interactions.
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
section illustrates exponential loss for binary labels. XGBoost is a brief
extension, not the implementation used in the interactive calculation.

The opening vector geometry uses `x = [1, 2, 3]`, `y = [1, -1, 1]`, zero baseline,
and learning rate 0.5. Its orthographic axes are training-point predictions.
Its flat angle experiment shows a true Euclidean equal-loss circle, with
radius equal to the initial residual norm. A fixed, equal scale makes the
circle shrink when the starting model improves. Arbitrary candidate updates
are lifted to exact three-dimensional vectors; their angles, norms, and losses
are not inferred from the orthographic projection. Fitted-tree states and the
candidate experiment remain independent. Neither apparent projected angles nor falling training
loss establish predictive performance on new observations.

Between the radius view and the four-observation calculation, `x = [1, 1]`
and `y = [0, 1]` restrict every prediction and update to equal coordinates.
The default is asymmetric squared error with τ = 0.8 and prediction `(0, 0)`.
For `0 ≤ c ≤ 1`, total loss is `(1-τ)c² + τ(1-c)²`, so the best shared
prediction is `(τ, τ)`. At `(0.5, 0.5)`, raw residuals have mean zero but the
negative-gradient targets are `(-0.2, 0.8)` and their fitted mean is 0.3.
At `(0.8, 0.8)` the negative gradient `(-0.32, 0.32)` is perpendicular to
the available line, while the residual `(-0.8, 0.2)` is not.

The loss selector also provides half-squared error (and asymmetric squared
at τ = 0.5), Huber δ in {0.1, 0.25, 0.5, 1}, and fourth-power error `e^4/4`.
For Huber δ < 0.5, the optimum is the entire interval `[δ, 1-δ]` on the
shared-prediction line. A unit-rate loss-optimal correction moves to the closest
point in that interval, preserving an already optimal prediction. The gradient
targets are fitted by least squares; Huber and fourth-power error generally
need a different shared leaf correction to minimize the original loss.

Both axes use equal, fixed scales. The raw residual and negative gradient
arrows are drawn at half length, stated in the legend, while the final shared
correction is drawn at its actual length. Current and optimal equal-loss contours
make the loss geometry visible; only squared error has circular contours.
Reset and residual-mean buttons allow comparisons at the same prediction.
The zero start demonstrates an update; initialization at the selected loss's
best constant would already start at an optimum in this two-observation case.

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

Loss values are sums, not MSE. Candidate SSEs are measured on each method's
own targets, not directly ranked against one another. The three-observation exact
circles remain specific to squared error; the two-observation view changes its
contours with the selected loss. Controls support keyboard use, and round
changes refit from the initial model rather than reusing another loss's trees.

The third dataset has identical inputs x = [1,1,1] and responses y = [0,0,10].
For Huber δ = 1, gradient boosting remains at the optimal constant 1/2 with
loss 9.25. The residual shortcut approaches the outcome mean 10/3, where
Huber loss is 71/6. Since no split is possible, this isolates a persistent
mismatch between the residual shortcut and the selected loss.
