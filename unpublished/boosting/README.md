# Boosting, one step at a time

An unpublished static scrollytelling draft about gradient descent in function
space. This chapter is kept outside the `docs` directory published by GitHub
Pages. To preview it locally, serve the repository root with a static HTTP
server and open `/unpublished/boosting/`. No build step or external JavaScript
dependencies.

## Construction

- `index.html` contains the full explanation, MathML, references and controls.
- `../../docs/shared/boosting-math.js` is the canonical shared math module. It
  fits least-squares regression stumps and calculates exact vectors and losses
  for this draft and the public tree explainer. It supports both the browser
  and Node.js tests.
- `story.js` renders SVG scenes, follows the narrative while scrolling, and
  handles the experiment controls. Previous/Next buttons also navigate scenes;
  left/right arrow keys work when focus is outside an interactive control.
- `worked-example.js` builds the appended four-observation calculation from
  section 1.4.3 of the boosting notes. It uses the shared stump-fitting module
  without changing the earlier geometric example or the published tree page.
- `worked-view.js` shows the four observed responses, prediction gradients,
  candidate split tables, and two cumulative model updates.
- `style.css` follows the course design guide. The existing STIX math font is
  loaded from the Shapley explainer's local font asset.

The earlier geometric example uses `x = [1, 2, 3]`, `y = [1, -1, 1]`, an explicit zero
baseline, and learning rate 0.5. Each next stump is fit to freshly computed
residuals. Ties select the first sorted candidate threshold. Slider paths
recompute all trees when the learning rate changes.

The appended worked section uses `x = [1, 2, 3, 4]`, `y = [0, 2, 1, 5]`, and
the mean prediction `f0 = 2`. With learning rate 1, its two stumps split at
3.5 and then 1.5, and total half-squared loss falls from 7 to 1 to 1/3.
The gradient has four entries because it differentiates the four predictions,
holding observed inputs and responses fixed. Candidate tables report the SSE
for fitting the current residuals; this differs from the model's half-squared
loss. Shared leaf corrections can worsen an individual prediction while
improving the aggregate objective.

Selecting learning rate 0.5 rebuilds both rounds from the mean baseline and
fits the second tree to its newly computed residuals. The loss sequence becomes
7, 5/2, and 11/8. The second round has an exact tie between splits 1.5 and 3.5;
the shared fitting routine deterministically chooses 1.5, the first sorted
threshold. The two-round states, trees, and candidate rows are immutable.

The three-axis picture is an orthographic projection of prediction vectors.
Axes correspond to training observations, not input features. Its apparent
angles are not used to establish descent. A separate flat, equal-scale
experiment shows both orientations of a candidate learner's line and selects
`h` or `-h` to align with the residual. The selected step is a true vector
negation for obtuse input angles. It demonstrates the exact step-length bound
`0 < s < 2 |cos(theta)|`, where `s` is step length divided by remaining distance
and `theta` is the original angle of `h`. At 90 degrees neither sign improves
loss. The fitted leaf-mean trees already satisfy `r·h = ||h||²`, so their
nonzero prediction vectors need no sign correction. Displayed training loss
is always one half of the squared Euclidean error.

The XGBoost afterword describes second-order tree selection and the L2-only
optimal leaf weight. It does not claim to implement XGBoost itself.

## Checks

Run these commands from the repository root:

```sh
node --check docs/shared/boosting-math.js
node --check unpublished/boosting/story.js
node --check unpublished/boosting/worked-example.js
node --check unpublished/boosting/worked-view.js
node --test tests/*.test.mjs
```

For visual review, inspect a wide desktop viewport and a narrow phone viewport,
scroll through every scene, use both sliders and the rate selectors, and
check keyboard navigation and reduced-motion settings. The text remains
readable without JavaScript. The plot has a dynamic accessible description
and equivalent numerical readouts.

## References and artwork

The prediction-space construction follows Kilian Weinberger's Cornell lecture
from approximately 3:20 onward:
<https://www.youtube.com/watch?v=dosOtgSdbnY&t=180s>.
The page links Cornell's lecture notes and the official XGBoost model tutorial.
It uses original prose and executable calculations, not lecture screenshots.

`assets/boosting-swing.png` is the approved transparent title artwork, registered with
its hash and usage guidance in `../../design/assets.json`. No shared design rules
were changed.
