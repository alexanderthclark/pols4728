# Shallow neural networks

A scroll-driven introduction to shallow networks and learned representations.
The narrative contains twenty-six scenes in five parts: shallow ReLU networks
as sums of hinges, width as the number of available joints, constructive sine
approximation, sigmoid classification in Beck, King, and Zeng (2000), and a
fitted binary example with two hidden features. Scrolling backward reverses the story.
The function and network diagrams share one full-width pinned visual stage,
with a scene-specific heading, an equation, and a concise caption. There is no separate explanation
column. The opening identifies the bias input, then explains the diagram-omission
convention when that node disappears. A return link stays available at the end.
The input-x slider lets readers inspect which neurons activate at a chosen
input, with a matching point on the function plot. A count above the network
reports the active neurons out of the current hidden-layer width. A neuron
counts as active when its ReLU activation is positive, independent of its
output weight. It preserves the model
weights and the story's fixed approximation-error detail view. Scrolling
continues to drive the lesson, network width, and weight updates. The arrow
buttons and left/right arrow keys provide optional navigation between the
same scroll positions; arrow keys on the slider adjust the input instead.

The title illustration depicts a round, torso-less blacksmith shaping a heated
rod on an anvil. Its registry entry is in `design/assets.json`. The WebP asset
is a lossless conversion of the generated transparent source. The requested
rust-orange heated rod is a purposeful exception to monochrome character art.
All teaching text and equations remain HTML.

## Scope to preserve

The original construction uses one scalar input, one hidden ReLU layer, and a
linear output. Part 2 is about **width**: add neurons alongside one another in
that single hidden layer. Preserve this scope in the original three parts.
The width sequence is one, two, three, and then six hidden neurons. Part 3 keeps
the same architecture while adding joints to approximate sine.

The classification extension introduces sigmoid hidden units and a sigmoid
output, then returns to ReLU hidden units for a newly fitted binary example.
That example has exactly two inputs, two hidden neurons, and one sigmoid head.
Its two-dimensional representation consists of the actual hidden activations,
not a projection. The input inspector is hidden during these new scenes.

## Mathematics

`math.js` supplies the exact hinge networks and piecewise linear interpolation.
`state.js` maps each scene and its scroll progress to its network weights.
`views.js` draws the original function and network, including a compact phone
layout. `extension-views.js` draws the sigmoid constructions and binary example.
`story.js` connects those states and captions to scrolling and keyboard navigation. The
original narrative sections supply scroll distances; they remain the reading
route without JavaScript and in print.

The opening maps the familiar equation ŷ = wx + b to its network diagram,
using w = 1 and b = 0.5. A faint constant-1 node and its weight-b connection
initially show how the intercept enters the sum. The constant is bookkeeping,
not a second observed input. As the opening progresses, the bias node and its
connection disappear from the drawing for clarity; the intercept remains in
the equation and the model. Omitted bias connections are a diagram convention,
not a change to the model.

`state.js` supplies LaTeX for every stage equation, including compact line breaks
for longer formulas. `formula.js` renders these with local KaTeX to visible HTML
and semantic MathML. The runtime, stylesheet, and ten required fonts are
self-hosted in `vendor/katex/`; no external requests are required. Network
parameter labels in the original scenes retain selectable SVG text with italic
mathematical variables. New plot and network labels use KaTeX HTML and MathML
inside SVG foreign objects.

The single-hinge scene first moves its threshold from −0.8 to 0.6 with
output weight 1, then increases that weight to 1.6 and turns it negative.
Faint comparison curves retain earlier parameter choices. They are visual
references; the actual network in this scene still has one hidden neuron.

The sine approximation begins with five nonzero interior slope changes, at
0.9, 1.95, 3.05, 4.2, and 5.35, on the domain [0, 2π]. One always-active
ReLU supplies the initial slope, so the initial network has six hidden neurons.
Adding an interpolation node adds one hidden neuron and changes adjacent
output weights. The insertion scene linearly interpolates the old and new
weights; its final function meets sine exactly at the inserted node. The
widening scene increases the width from seven to twenty-six through greedy
refinements at the current largest sampled error.

The width example uses the domain [−2, 2] and successive prefixes of six
hinges. Their thresholds are [−1.5, −0.9, −0.3, 0.3, 0.9, 1.5] and output
weights are [0.8, −1.2, 1.1, −1, 0.9, −0.6]. For these distinct thresholds
and nonzero slope changes, width m gives m interior joints and m + 1 linear
pieces. For arbitrary weights these are upper bounds: neurons can remain
inactive on the interval, share thresholds, or cancel one another.

Part 3's final argument concerns uniform approximation of a continuous function on
a closed, bounded interval. It establishes representational capacity, not
whether training will find the displayed weights. These ReLU networks with
linear outputs remain continuous and piecewise linear.

## Classification and representations

Part 4 opens with a centered typesetting of [Beck, King, and Zeng (2000), Equation 3](https://gking.harvard.edu/files/improv.pdf#page=5), with no diagrams in that opening scene.
It preserves the paper's Bernoulli model and parameter notation, spelling out
the function name `sigmoid`. Desktop shows the expanded terms; phones use the
equivalent summation. The next scene compares the BKZ model with ordinary
logistic regression using equations side by side on desktop and stacked
on phones. Continued scrolling foregrounds logistic regression and reveals
a one-input probability curve and a two-input classification boundary.
The network then takes focus: its curve can rise and fall, and its boundary
can enclose the inner class. The observations remain fixed. `comparison-model.js`
contains the chosen weights and exact sigmoid forward passes; `comparison-views.js`
renders the two models as separate crossfading layers, not an interpolated fit.
The two-input boundary is a level set of the actual four-unit network, not an
imposed circle. The 1D example uses two sigmoid hidden units. These constructions
illustrate capacity and are not estimates from BKZ. Reduced motion switches
between all three beats, and the arrow controls visit the same stops.
The following scenes return to the course notation and demonstrate
more complicated probability curves as hidden neurons are added.
The hidden features are sigmoids of affine input scores, and the output is a
sigmoid of an affine combination of those features. An ordinary logit model is
one sigmoid output directly on the inputs, with no hidden layer. One sigmoid
hidden unit followed by a sigmoid head is a composition, not generally an
ordinary logit model.

The one-input illustrations begin in the third classification scene and use
fixed hidden thresholds 0.2, 0.4, 0.6, and 0.8,
slope 30, output intercept −3, and successive output weights +6, −6, +6, −6.
They progress from a monotone curve to one high-probability interval to two.
Their weights are hand-chosen to illustrate capacity; they are not estimates
from the paper. The representation scene names the hidden activations as h
and identifies the affine score plus final sigmoid as the classification head.

Part 5 uses a new constructed binary dataset inspired by adjacent curved groups
in the course notes. `classification-data.js` contains 82 observations, the
frozen fitted parameters, provenance, and numerical checks. The data are two
vertically offset curves: x₂ = −sin²(x₁) + 0.9y, with 41 equally spaced x₁
values in [0, π] for each class y ∈ {0, 1}. The two class convex hulls intersect,
so a straight boundary in the original coordinates cannot separate them.

`scripts/generate-neural-classification.py` jointly fits the two ReLU features
and the sigmoid head with penalized binary cross-entropy, then exports the
frozen model. Its documented selection criteria choose a clear teaching fit
from multiple starts. The fitted head correctly classifies all 82 training
observations at a 0.5 cutoff; this establishes neither performance on unseen
data nor exact probabilities of zero or one. This is not the notes' original
three-class fitted network.

The sequence shows the fitted input-space boundary, traces one observation
through the two features, then moves every point into its own (h₁, h₂)
coordinates. Point identity, class, color, and shape persist. The movement
interpolates display positions between separately scaled coordinate systems;
it does not depict training. Intermediate axis ticks are omitted. The boundary
remains absent for this entire scene, including its completed and reduced-motion
states. The next scene reveals the same fitted head's straight zero-logit
boundary. The final comparison displays that line and its exact piecewise
linear preimage in the original coordinates. Both plots use the same weights.

## Design and validation

The page follows the public course guide: white backgrounds, regular serif
text, course blue for the model, rust for the comparison/intervention, open
axes, and direct labels. Sine is dashed and the network curve is solid, so
the distinction also works in grayscale. A labeled detail view deliberately
zooms into the approximation error. Wider networks omit numerical weight labels
where they would collide, retaining every neuron and connection.

The same scrolling story runs on narrow screens, with a full-width function
above a compact network. Reduced-motion settings use discrete completed
states for most transitions. The complete narrative remains readable without
JavaScript and in print.

Run the mathematical and scroll/render checks from the repository root:

```sh
node --test tests/neural-networks-state.test.cjs
node --test tests/neural-networks-scroll.test.cjs
node --test tests/neural-classification-model.test.cjs
node --test tests/neural-comparison-model.test.cjs
```

The model tests independently recompute hidden features, scores, training
accuracy, geometric margin, and the convex-hull witness. Integration tests
check all scenes, valid LaTeX in both layouts, point identity and coordinates,
the rendered separating line, reverse scrolling, and reveal order under
reduced motion. To refit and check the frozen export using the dependency
versions recorded in its provenance, run the offline generator with `--check`.

Browser review covers forward and backward scrolling at 1280×800, 390×650,
and 360×640, keyboard navigation, diagram labels, and the final return link.
Reduced motion is checked in a phone browser and by automated tests. These checks do not constitute an
accessibility-compliance claim. No build step or external dependency is
required. After merging, GitHub Pages serves this directory at
`https://alexanderthclark.github.io/pols4728/neural-networks/`.
