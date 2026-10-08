# Deep networks

A scrollytelling explanation of folding for readers who know shallow ReLU
networks. Three fixed ramps form a folded input coordinate. The next hidden layer
places three hinges on that coordinate, reusing the same downstream curve across
three input pieces. The public example fixes those hinges at `q = 0.2, 0.5, 0.8`.
Readers select an input and inspect the linked views. The chosen output has ten
joints with 22 dense parameters; an exactly matching shallow network needs at
least ten hidden units and 31 dense parameters.

Hosted path: <https://alexanderthclark.github.io/pols4728/deep/>.
The page is a static ES-module site with no build step or external runtime.

The opening frame uses `assets/origami-folding-character.png`: a gleeful round
character stands beside a compact table and lifts an origami strip overhead. Repeating
diagonal creases form triangular folded facets in the paper itself. Its monochrome
artwork is registered in `design/assets.json`, preserves its transparent
background and full silhouette, and appears only on the title frame. Like the
Shapley cover, the full-width white opening page centers its title above the
artwork and a Start link. It remains frame 1 of 8; Start opens the first
explanatory frame and Previous returns to the cover. The title and mathematical
explanation remain real HTML text; frames 2–8 retain the precise SVG figures.

## Shared fold and network

The notation follows Simon J. D. Prince's *Understanding Deep Learning*, Chapter
4: **h** denotes hidden activations, **β** biases, **Ω** weights, and
`a[z] = max(0, z)` acts elementwise. The site uses a chosen folding construction
to make the representation and parameter comparison explicit.

There is one input, two hidden layers with three ReLU units each, and an affine
output:

```text
h1 = a[β0 + Ω0 x]             h2 = a[β1 + Ω1 h1]
y  = β2 + Ω2 h2

β0 = (0, -1, -2)             Ω0 = (1, 1, 1)ᵀ

β1 = (-0.2, -0.5, -0.8)      Ω1 = [1  -2  2]
                                  [1  -2  2]
                                  [1  -2  2]

β2 = 0                      Ω2 = [1  -2  2]
```

The first features are `(a[x], a[x-1], a[x-2])`, with fixed hinges at
`0, 1, 2`. All three new units receive the same weighted coordinate:

```text
q(x) = h11 - 2 h12 + 2 h13

       0       for x ≤ 0
       x       for 0 ≤ x ≤ 1
       2 - x   for 1 ≤ x ≤ 2
       x - 2   for x ≥ 2
```

On `[0,3]`, the three unit intervals map forward, backward, then forward onto
`q ∈ [0,1]`. The middle interval reverses direction. For an interior folded
coordinate `0 < q < 1`, the three inputs `q, 2-q, 2+q` share the same output.
At the endpoints, branch locations coincide; `q = 0` also has the constant
branch `x ≤ 0`.

The next layer's features are `a[q-0.2]`, `a[q-0.5]`, and `a[q-0.8]`.
Their affine output is the downstream curve

```text
g(q) = a[q-0.2] - 2 a[q-0.5] + 2 a[q-0.8],    y(x) = g(q(x)).
```

The public thresholds remain fixed. A downstream hinge at `q = t` appears at
input locations `t, 2-t, 2+t`; the middle hinge therefore appears at
`x = 0.5, 1.5, 2.5`. These locations are linked by the fold and cannot be
positioned independently. Selecting different inputs shows that the same
downstream response is reused on each of the three pieces.

The fold coordinate is notation for the repeated weighted combination in
`Ω1`; it is not an extra hidden unit or an extra parameter block in the counted
dense network.

## Joints and parameter comparison

The displayed output has ten joints at
`0.2, 0.5, 0.8, 1, 1.2, 1.5, 1.8, 2.2, 2.5, 2.8`.
Their slope jumps are `1, -2, 2, -2, 2, -2, 1, 1, -2, 2`.
There are nine copied downstream hinges plus the original fold joint at `x=1`.
The old hinges at `0` and `2` lie within inactive flat regions.

For standard dense architectures, count every weight and bias slot, including
zeros and repeated numerical values:

| Architecture | Dense parameters | Joints |
| --- | ---: | --- |
| `1 → 3 → 3 → 1` | `(3+3)+(9+3)+(3+1) = 22` | 10 for this construction |
| `1 → 7 → 1` | `3×7+1 = 22` | At most 7 |
| `1 → 10 → 1` | `3×10+1 = 31` | Matches this ten-joint output exactly |

A scalar-input shallow network with an affine output and `D` ReLU hidden units
has at most `D` joints, because each unit contributes at most one hinge. Ten
nonzero slope jumps therefore require at least ten units, and at least 31 dense
parameters, in that shallow architecture. Ten suffice: if `b_j` are the joint
locations and `Δ_j` their slope jumps, the exact shallow expansion is
`y(x) = Σ_j Δ_j a[x-b_j]`. The leftmost response and slope are both zero.

Thus this target curve uses nine fewer dense parameter slots in the two-layer
architecture. A same-budget shallow width-seven network cannot match it exactly.
The reuse links the copied pieces and their joints; this does not establish that
an arbitrary ten-joint curve can be represented with these 22 parameters, or
that deeper networks always train or predict better. The displayed coefficients
are chosen; no training or prediction-error comparison is performed.

The input window `[-0.25,3.25]` retains global context. The left tail is constant,
and the right tail rises. The native downstream view `q ∈ [0,1]` describes the
three full core pieces; inputs above `3` have `q>1`. The downstream plots label
an off-view selected coordinate explicitly and retain the numeric readout.
The fixed public output is nonnegative; zero output intervals are not additional
joints unless the slope changes at their boundary.

## Files and exact geometry API

The public interaction changes the selected input and the inspected view, not
the network parameters. The mathematical modules retain a generalized middle
threshold for exact checks and reuse in code: `DEFAULT_THRESHOLD = 0.5` and
`THRESHOLD_RANGE = [0.35,0.65]`. Passing `threshold: τ` changes only the middle
second-layer bias to `-τ`. This internal API is not a public slider. Its three
copied joints move together and remain distinct, with ten nonzero output slope
jumps throughout that range. For `τ < 0.5`, the generalized affine output can
be negative and have zeros between activation hinges.

- `index.html`, `style.css`, `story.js`, and `views.js` supply the narrative,
  linked figures, controls, and navigation.
- `model.mjs` supplies `networkParameters({threshold})`,
  `evaluateNetwork(x, {threshold})`, `surfaceValue(mode, x, {threshold})`,
  `foldedValue(q, {threshold})`, `foldedPreimages(q)`, and
  `preactivationRoots({threshold})`.
- Evaluation exposes the scalar input, both preactivation vectors, both hidden
  vectors, `fold`/`q`, and named responses. `shallow` aliases the raw fold;
  `preactivation` is the middle new unit's `q-τ`; `y` aliases `combined`.
- `geometry.mjs` supplies `curveGeometry(mode, {threshold, domain})` over the
  input coordinate and `foldedCurveGeometry({threshold, domain})` over `q`.
  The latter defaults to `[0,1]`.
- Input modes are `h11`, `h12`, `h13`, `fold`, `shallow`,
  `preactivation`, `unit1`, `unit2`, `unit3`, and `combined`.
  A downstream renderer calls `foldedValue` and `foldedCurveGeometry` directly;
  there is no `downstream` mode in `surfaceValue`.
- `parameterCount(widths)` counts dense weight/bias slots from an array including
  input and output widths. `PARAMETER_COUNTS` records `deep:22`,
  `shallowMatch:31`, `shallowSameBudgetWidth:7`, and `shallowSameBudget:22`.

Both geometry functions use analytical affine pieces and roots with floating
point tolerance, rather than a dense sampled grid:

```text
points        [[coordinate, response], ...]
bends         [{x, y, leftSlope, rightSlope}, ...]
segments      [{left, right, slope, intercept}, ...]
zeroCrossings [coordinate, ...]
```

The `x` field denotes `q` in the downstream geometry; segment `left` and
`right` are scalar coordinates. Points include all required affine breakpoints
and output zeros. Only a genuine derivative change enters `bends`.
The compatibility name `zeroCrossings` includes signed crossings, isolated
zero contacts, and boundaries of flat zero regions; it excludes flat-zero
interiors. An output zero without a slope change is a sample, not a joint.

Input geometry also returns all `firstLayerKnots` and the three complete arrays
of `secondPreactivationRoots`, even when outside a custom view.
`foldedPreimages` supplies the three branch locations in `[0,3]` for
`0 ≤ q ≤ 1`; it does not enumerate the additional constant branch at zero.

## Preview and checks

From the repository root:

```sh
python3 -m http.server 8874 --bind 127.0.0.1 --directory docs
```

Open <http://127.0.0.1:8874/deep/>. Select inputs on the plot or with the input
control, replay the fold, and choose a view in the explorer. The three thresholds
stay at `0.2, 0.5, 0.8`; the public page has no threshold control.
Run the focused mathematical checks with:

```sh
node --test tests/deep-model.test.mjs tests/deep-geometry.test.mjs
```

The complete repository command is `node --test tests/*.test.mjs`.
The 19 focused tests pass. They cover the fixed public model and its generalized
internal threshold API: exact fold and output equations, copied crossings,
moving grouped joints, ten non-cancelling bends, exact input and downstream
geometry, signed output zeros, dense parameter counts, and exact width-ten
shallow reconstruction beyond the displayed domain.

These tests do not verify layout or accessibility. Review desktop and narrow
mobile rendering, keyboard controls, and reduced motion separately against
`design/README.md`.

## Primary references

- [Prince's official book and companion materials](https://udlbook.github.io/udlbook/),
  *Understanding Deep Learning*, Chapter 4, §§4.1–4.2 for composition and folding,
  and §4.5.2 for the parameter comparison.
- [Official equations source](https://github.com/udlbook/udlbook/blob/main/UDL_Equations.tex),
  especially general network notation in `dnn_la1` and `dnn_la2`.
- [Official Chapter 4 composition notebook](https://github.com/udlbook/udlbook/blob/main/Notebooks/Chap04/4_1_Composing_Networks.ipynb).
  The numerical joint locations and exact 22-versus-31 comparison here are
  derived for this site's chosen example.
