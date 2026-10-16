# Neural networks, one bend at a time

A scroll-driven introduction to one-dimensional ReLU networks. The narrative
contains sixteen scenes in three parts: shallow networks as sums of hinges,
width as the number of available joints, and constructive approximation of a
sine curve by widening the same hidden layer. Scrolling backward reverses the story.
There are no sliders or toggles. The arrow buttons and left/right arrow keys
provide optional navigation between the same scroll positions.

The title illustration depicts a round, torso-less blacksmith shaping a heated
rod on an anvil. Its registry entry is in `design/assets.json`. The WebP asset
is a lossless conversion of the generated transparent source. The requested
rust-orange heated rod is a purposeful exception to monochrome character art.
All teaching text and equations remain HTML.

## Scope to preserve

The user explicitly restricts this introduction to one scalar input, one hidden
ReLU layer, and a linear output. Part 2 is about **width**: add neurons alongside
one another in that single hidden layer. Preserve this scope in future edits.
The width sequence is one, two, three, and then six hidden neurons. Part 3 keeps
the same architecture while adding joints to approximate sine.

## Mathematics

`math.js` supplies the exact hinge networks and piecewise linear interpolation.
`state.js` maps each scene and its scroll progress to its network weights.
`views.js` draws the actual function and network, including a compact phone
layout. `story.js` connects those states to scrolling and keyboard navigation.

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

The final argument concerns uniform approximation of a continuous function on
a closed, bounded interval. It establishes representational capacity, not
whether training will find the displayed weights. Every displayed network
function remains continuous and piecewise linear.

## Design and validation

The page follows the public course guide: white backgrounds, regular serif
text, course blue for the model, rust for the comparison/intervention, open
axes, and direct labels. Sine is dashed and the network curve is solid, so
the distinction also works in grayscale. A labeled detail view deliberately
zooms into the approximation error. Dense network weights become crowded in
the final scene to demonstrate the size of the shallow representation.

The same scrolling story runs on narrow screens, with a full-width function
above a compact network. Reduced-motion settings use discrete completed
states for most transitions. The complete narrative remains readable without
JavaScript and in print.

Run the mathematical and scroll/render checks from the repository root:

```sh
node --test tests/neural-networks-state.test.cjs
node --test tests/neural-networks-scroll.test.cjs
```

The SVG figures were rendered and inspected at desktop and 350 px phone
sizes. Browser layout verification remains outstanding; no browser was
available in the authoring environment. The checks do not constitute an
accessibility-compliance claim. No build step or external dependency is
required. After merging, GitHub Pages serves this directory at
`https://alexanderthclark.github.io/pols4728/neural-networks/`.
