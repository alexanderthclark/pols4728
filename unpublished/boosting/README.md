# Gradient boosting, one step at a time

An unpublished scrollytelling chapter about gradient boosting. Its main path
starts with a loss and a constant prediction, differentiates training-point
predictions, fits trees to negative-gradient targets, and adds scaled
corrections. Eight scenes develop the four-observation example from sections
1.4.2–1.4.3 of the boosting notes, followed by other losses and a brief XGBoost
extension. Prediction-space geometry is an optional companion.

The chapter stays outside the `docs` directory published by GitHub Pages.
To preview, serve the repository root and open `/unpublished/boosting/`.
No build step or external JavaScript dependencies are required.

## Construction

- `index.html` contains the main explanation, numerical example, controls,
  general algorithm, and loss comparison.
- `story.js` follows the eight narrative steps and handles split inspection,
  update progress, learning rate, later rounds, and navigation.
- `worked-example.js` builds immutable gradient-boosting states using the
  unchanged `../../docs/shared/boosting-math.js` stump-fitting module.
- `worked-view.js` renders response plots, gradients, candidate split tables,
  prediction updates, later fitted rounds, and the general learning loop.
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

The optional geometry uses `x = [1, 2, 3]`, `y = [1, -1, 1]`, zero baseline,
and learning rate 0.5. Its orthographic axes are training-point predictions.
A separate flat angle experiment illustrates alignment and the exact
step-length bound. Neither apparent projected angles nor falling training
loss establish predictive performance on new observations.

## Checks

From the repository root:

```sh
node --check unpublished/boosting/story.js
node --check unpublished/boosting/worked-example.js
node --check unpublished/boosting/worked-view.js
node --check unpublished/boosting/geometry.js
node --test tests/*.test.mjs
```

Inspect a wide desktop and narrow phone viewport, navigate all main scenes,
compare candidates, change learning rate, adjust update progress and rounds,
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
design rules or published pages are changed.
