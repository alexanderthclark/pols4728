# What does another layer add?

An eight-frame scrollytelling construction for readers who know shallow ReLU
networks. Three fixed first-layer ramps feed three new hidden units, which build
local responses and contribute to one scalar output. The linked surface, input
plane, and network diagram follow the same input point and parameters.

Hosted path: <https://alexanderthclark.github.io/pols4728/deep/>.
The page is a static ES-module site; it needs no build step or external runtime.

## Notation and construction

The notation follows Simon J. D. Prince's *Understanding Deep Learning*, Chapter
4: **h** is a vector of hidden activations, **β** denotes biases, **Ω** denotes
weights, and `a[z] = max(0, z)` acts elementwise. This particular two-input example
and its weights are an original illustration, rather than a reproduction of the
book's scalar-network composition example.

For input `x = (x1, x2)`, the model has exactly two hidden layers, with three units
in each, followed by an affine output:

```text
h1 = a[β0 + Ω0 x]             h2 = a[β1 + Ω1 h1]
y  = β2 + Ω2 h2

β0 = (0, 0, 0)               Ω0 = [ 1   0 ]
                                  [ 0   1 ]
                                  [-1  -1 ]

β1 = (1, 1, 1)               Ω1 = [-1  -1  -1 ]
                                  [-c  -1  -1 ]
                                  [-1  -2  -1 ]

β2 = 0                       Ω2 = [1  0.3  0.3]
```

The first features are `h1 = (a[x1], a[x2], a[-x1-x2])`. The default is `c = 2`.
The weight control changes only `Ω1[1][0] = -c`, with `0.5 ≤ c ≤ 3`; the model's
`pinch` option is the positive magnitude `c`, not the negative displayed weight.
At `c = 1`, the first two second-layer units coincide.

The shallow surface and the first new unit's preactivation are deliberately the
same calculation, `q1 = 1 - h11 - h12 - h13`. Before the added ReLU, the dashed
hexagon is a zero crossing; afterward it is an activation boundary. Writing
`S = h11 + h12 + h13`, we have
`S = max(|x1|, |x2|, |x1+x2|)` and `h21 = max(0, 1-S)`. Its closed support has
vertices `(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)` and a height-one peak at
the origin. At the default weights, all three second-layer responses, and thus
the output, vanish outside this hexagon.

The displayed domain is `[-1.6, 1.6]²`. When `c = 0.5`, the second unit's support
reaches `x1 = 2` and extends beyond the viewing window. A footprint clipped by the
window must not be described as the complete support.

## Mathematical scope

The comparison holds the first three features fixed. Because their biases are
zero, `h1(r x) = r h1(x)` for `r ≥ 0`. An affine readout of them is affine in `r`
along every ray. It cannot be nonzero near the origin and then remain identically
zero beyond a finite distance along every ray. The added ReLU provides the extra
change in slope needed here. The construction demonstrates a representational
mechanism; it makes no claim about parameter efficiency, arbitrary shallow
networks, training success, or test error.

## Files and geometry API

- `index.html`, `style.css`, and `story.js` supply the narrative, layout, controls,
  linked figures, and navigation.
- `model.mjs` supplies `networkParameters({pinch})`,
  `evaluateNetwork(x1, x2, {pinch})`, and `surfaceValue(mode, x1, x2, {pinch})`.
  Evaluation returns both preactivation vectors, both hidden vectors, and the
  named surfaces, including `y` as an alias for `combined`.
- `geometry.mjs` supplies `surfaceMesh(mode, {pinch, extent})`. The default extent
  is `1.6`. Modes are `h11`, `h12`, `h13`, `shallow`, `preactivation`, `unit1`,
  `unit2`, `unit3`, and `combined`.
- `tests/deep-model.test.mjs` and `tests/deep-geometry.test.mjs` check the stated
  formulas, bounded responses, weight changes, geometry, and gradients.

The mesh starts with the six first-layer cones, then splits them at the three
second-layer zero planes. It uses analytical line intersections with floating
point tolerance, rather than a sampled grid. Its return value contains:

```text
vertices      [[x1, x2, height], ...]
faces         [{indices, gradient:[dx,dy], intercept,
                activeFirst, activeSecond}, ...]
triangles     [[vertexIndex, vertexIndex, vertexIndex], ...]
creases       [[vertexIndex, vertexIndex], ...]
boundary      [[vertexIndex, vertexIndex], ...]
zeroContours  [[vertexIndex, vertexIndex], ...]
domainEdges   [[vertexIndex, vertexIndex], ...]
```

Faces are counterclockwise and share vertices without T-junctions. `creases`
contains only edges whose adjacent affine gradients differ. `boundary` marks
the positive footprint within the view, including positive segments on the
outer domain when the support is clipped. `zeroContours` contains actual zero
transitions, excluding internal edges of flat zero regions. `domainEdges` always
marks the viewing square. Fill triangles without drawing their edges; draw the
separate crease and contour lists to avoid artificial lines.

## Preview and checks

From the repository root:

```sh
python3 -m http.server 8874 --bind 127.0.0.1 --directory docs
```

Open <http://127.0.0.1:8874/deep/>. Run the focused mathematical checks with:

```sh
node --test tests/deep-model.test.mjs tests/deep-geometry.test.mjs
```

The complete repository test command is `node --test tests/*.test.mjs`.
The focused 13 tests pass. They verify the calculations and geometry, not visual
layout or accessibility. Review desktop and narrow mobile rendering, keyboard
controls, and reduced motion separately against `design/README.md`.

## Primary references

- [Prince's official book and companion materials](https://udlbook.github.io/udlbook/).
- [Official equations source](https://github.com/udlbook/udlbook/blob/main/UDL_Equations.tex),
  Chapter 4, especially the general network notation in equations `dnn_la1` and
  `dnn_la2`.
- [Official Chapter 4 composition notebook](https://github.com/udlbook/udlbook/blob/main/Notebooks/Chap04/4_1_Composing_Networks.ipynb),
  as background for the book's composition viewpoint. This site's vector-feature
  construction uses a different example.
