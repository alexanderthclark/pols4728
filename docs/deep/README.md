# What does another layer add?

An eight-frame scrollytelling construction for readers who know shallow ReLU
networks. One scalar input enters three fixed hinge features. A second hidden
layer creates new bends inside the first layer's linear intervals, then combines
three new features into one numeric output.

Hosted path: <https://alexanderthclark.github.io/pols4728/deep/>.
This is a static ES-module page with no build step or external runtime.

## Notation and model

The notation follows Simon J. D. Prince's *Understanding Deep Learning*, Chapter
4: **h** denotes hidden activations, **β** biases, **Ω** weights, and
`a[z] = max(0, z)` acts elementwise. The construction and chosen weights are an
original example. Each new unit receives the first-layer vector directly; there
is no completed scalar output between the hidden layers.

There is one input, two hidden layers of three units each, and an affine output:

```text
h1 = a[β0 + Ω0 x]             h2 = a[β1 + Ω1 h1]
y  = β2 + Ω2 h2

β0 = (0, -1, -2)             Ω0 = (1, 1, 1)ᵀ

β1 = (-0.5, -τ, -0.6)        Ω1 = [1  -2    2]
                                  [1  -3    4]
                                  [1  -1.5  1]

β2 = 0                      Ω2 = [1  0.3  0.3]
```

The first features are `(a[x], a[x-1], a[x-2])`, with fixed hinges at
`0, 1, 2`. The live threshold is `τ = 0.3` by default, with
`0.2 ≤ τ ≤ 0.8`; it changes only the second new unit's bias `-τ`.
The model option `threshold` is the positive magnitude `τ`.

The three second-layer preactivation roots are:

| Unit | Roots |
| --- | --- |
| First | `0.5, 1.5, 2.5` |
| Second | `τ, (3-τ)/2, (5+τ)/2` |
| Third | `0.6, 1.8, 2.2` |

Every root remains inside one of the open fixed intervals `(0,1)`, `(1,2)`,
and `(2,3.25)`. The viewing domain is `[-0.25,3.25]`.

The shallow readout and the first new unit's preactivation are the same sum:
`q1 = -0.5 + h11 - 2 h12 + 2 h13`. Its zero crossings do not change its
slope. Applying ReLU turns those crossings into new bends and hides old hinges
where the response is clipped to zero. The first new feature has bends at
`0.5, 1, 1.5, 2.5`; the old hinges at `0` and `2` disappear from its curve.

At the default threshold, the output's ten bends are
`0.3, 0.5, 0.6, 1, 1.35, 1.5, 1.8, 2.2, 2.5, 2.65`.
Their slope jumps are
`0.3, 1, 0.3, -3.35, 0.6, 1, 0.15, 0.15, 1, 0.6`; none cancels.
At thresholds `0.5` and `0.6`, two roots coincide and their jumps add, leaving
nine distinct bends. The construction has a zero interval but its right-hand
tail continues rising; the output is not globally compactly supported.

## Mathematical scope

Within each first-layer interval, all three fixed features are affine in
`x`. Any affine readout of them is therefore affine there. A second-layer
preactivation can cross zero inside that interval, and its ReLU introduces a
new hinge at the crossing.

More generally, a scalar-input shallow network with three ReLU hidden units
and an affine output has at most three bends, because each unit contributes at
most one hinge. This chosen deep output has more than three bends in the displayed
domain and cannot be matched exactly by that shallow width. Adding the layer
also adds units and parameters: this is not an equal-parameter comparison, a
general efficiency result, or evidence about training or test error. A wider
shallow network can reproduce the curve using one hinge per true bend.

The second-layer rows are distinct and have rank two, not three. Their patterns
are not constrained to a rank-one scalar bottleneck. No full-rank claim applies
to this construction.

## Files and curve API

- `index.html`, `style.css`, `story.js`, and `views.js` supply the narrative,
  controls, linked curve and network figures, and navigation.
- `model.mjs` supplies `networkParameters({threshold})`,
  `evaluateNetwork(x, {threshold})`, `surfaceValue(mode, x, {threshold})`, and
  `preactivationRoots({threshold})`. Evaluation returns the scalar input,
  both preactivation vectors, both hidden vectors, and the named responses,
  including `y` as an alias for `combined`.
- `geometry.mjs` supplies `curveGeometry(mode, {threshold, domain})`.
  Modes are `h11`, `h12`, `h13`, `shallow`, `preactivation`, `unit1`,
  `unit2`, `unit3`, and `combined`.
- `tests/deep-model.test.mjs` and `tests/deep-geometry.test.mjs` validate the
  equations, moving roots, true bends, and wider shallow reconstruction.

The geometry uses analytical affine pieces and roots with a floating-point
tolerance, rather than a dense sampled grid. Its return value contains:

```text
points                       [[x, response], ...]
bends                        [{x, y, leftSlope, rightSlope}, ...]
zeroCrossings                [x, ...]
segments                     [{left, right, slope, intercept}, ...]
firstLayerKnots              [0, 1, 2]
secondPreactivationRoots     [[unit1 roots], [unit2 roots], [unit3 roots]]
```

Segment `left` and `right` are scalar input coordinates. Points include the
view's endpoints, first-layer knots, and relevant roots of the selected response.
Only a genuine derivative change enters `bends`; redundant samples and hinges
hidden within flat zero regions do not. Despite its API name, `zeroCrossings`
means zero transitions: signed crossings for the shallow readout or
preactivation, and positive/zero boundaries for ReLU responses. Flat-zero
interior samples are excluded. The full first-layer knots and all three units'
preactivation roots are supplied separately, even when outside a custom view or
absent from the displayed curve's bends.

## Preview and checks

From the repository root:

```sh
python3 -m http.server 8874 --bind 127.0.0.1 --directory docs
```

Open <http://127.0.0.1:8874/deep/>. Run the focused checks with:

```sh
node --test tests/deep-model.test.mjs tests/deep-geometry.test.mjs
```

The complete repository command is `node --test tests/*.test.mjs`.
The 18 focused tests pass. They verify calculations and curve geometry; inspect
desktop and narrow mobile layouts, keyboard controls, and reduced motion
separately against `design/README.md`. These mathematical checks do not establish
accessibility compliance.

## Primary references

- [Prince's official book and companion materials](https://udlbook.github.io/udlbook/).
- [Official equations source](https://github.com/udlbook/udlbook/blob/main/UDL_Equations.tex),
  Chapter 4, especially general network notation in equations `dnn_la1` and
  `dnn_la2`.
- [Official Chapter 4 composition notebook](https://github.com/udlbook/udlbook/blob/main/Notebooks/Chap04/4_1_Composing_Networks.ipynb),
  as background for the book's composition viewpoint. This site's feature-vector
  construction uses a different example.
