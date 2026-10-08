# Boosting, one step at a time

A static scrollytelling draft about gradient descent in function space. Open
`/boosting/` from the course site, or serve the repository's `docs` directory
with a static HTTP server. No build step or external JavaScript dependencies.

## Construction

- `index.html` contains the full explanation, MathML, references and controls.
- `model.js` fits least-squares regression stumps and calculates exact vectors
  and losses. It supports both the browser and Node.js tests.
- `story.js` renders SVG scenes, follows the narrative while scrolling, and
  handles the experiment controls. Previous/Next buttons also navigate scenes;
  left/right arrow keys work when focus is outside an interactive control.
- `style.css` follows the course design guide. The existing STIX math font is
  loaded from the Shapley explainer's local font asset.

The worked example uses `x = [1, 2, 3]`, `y = [1, -1, 1]`, an explicit zero
baseline, and learning rate 0.5. Each next stump is fit to freshly computed
residuals. Ties select the first sorted candidate threshold. Slider paths
recompute all trees when the learning rate changes.

The three-axis picture is an orthographic projection of prediction vectors.
Axes correspond to training observations, not input features. Its apparent
angles are not used to establish descent. A separate flat, equal-scale
experiment demonstrates the exact step-length bound `0 < s < 2 cos(theta)`,
where `s` is step length divided by remaining distance. Displayed training
loss is always one half of the squared Euclidean error.

The XGBoost afterword describes second-order tree selection and the L2-only
optimal leaf weight. It does not claim to implement XGBoost itself.

## Checks

```sh
node --check docs/boosting/model.js
node --check docs/boosting/story.js
node --test tests/*.test.mjs
```

For visual review, inspect a wide desktop viewport and a narrow phone viewport,
scroll through all twelve scenes, use both sliders and the rate selector, and
check keyboard navigation and reduced-motion settings. The text remains
readable without JavaScript. The plot has a dynamic accessible description
and equivalent numerical readouts.

## References and artwork

The prediction-space construction follows Kilian Weinberger's Cornell lecture
from approximately 3:20 onward:
<https://www.youtube.com/watch?v=dosOtgSdbnY&t=180s>.
The page links Cornell's lecture notes and the official XGBoost model tutorial.
It uses original prose and executable calculations, not lecture screenshots.

`assets/xg-swing.png` is the approved transparent title artwork, registered with
its hash and usage guidance in `design/assets.json`. No shared design rules
were changed.
