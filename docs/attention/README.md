# Attention: how tokens share information

A buildless POLS 4728 scrollytelling lesson. Serve `docs/` and open `/attention/`.
The course homepage links to the lesson. Local fonts and KaTeX require no CDN.

## Scope and convention

The opening follows a complete two-token, one-head socks-and-shoes model.
Twelve scroll-driven scenes show the order-sensitive target, identity pooling,
explicit position, Q/K matching, finite softmax, left multiplication, values,
prediction, reversal with fixed parameters, and ablations.

Seventeen further scenes explain larger token-row representations, equal and
hand-chosen mixtures, left multiplication by A, query/key projections, dot products,
scaled score matrices, exact score → exponential → softmax phases, values, an
optional output map, causal masking before softmax, renormalization, and the final
input-dependent attention operation. No general transformer architecture is taught.

X is T × d; Q = X WQ; K = X WK; A = row-softmax(Q Kᵀ / sqrt(dk) + M);
V = X WV; Y = A V. A's rows are destinations and columns are sources.
The diagonal is retained under causal masking. The four-token numeric example is invented,
explicitly labeled, and computed by `model.mjs`, not a fitted linguistic claim.
The hand-chosen mixture in the general calculation is separate from its query-derived example.

Primary reference: Elhage et al. (2021), A Mathematical Framework for Transformer
Circuits. Secondary: Prince, Understanding Deep Learning, chapter 12 §§12.2–12.3.
The page explains the transpose convention and includes the exact toy maps. A
broader notation harmonization with the course notes remains for a later pass.

## One-head task

`socks-model.mjs` stores each token as [socks indicator, shoes indicator, position].
The lesson introduces X=E+P: token embeddings and position vectors have the same
width. Learned positional embeddings do not require a reserved coordinate. We
choose fixed position vectors supported on the third coordinate for this example;
token embeddings have zero in that coordinate. Position is +1 in the first slot
and -1 in the second. Fixed projections read
q=shoes, k=log(3)*socks, and v=position. Thus shoes allocates 75% to socks and
25% to itself; socks allocates 50% to each. With Y=AV and readout
2*mean(Y)-0.5, the predictions are 0 for socks→shoes and -1 for shoes→socks.
No causal mask, feed-forward network, residual, or normalization is needed for
this constructed two-sequence task. The parameters are chosen, not trained.
Removing position or intervening with uniform attention yields -0.5 for both.

The matching preferences survive the swap because Q/K intentionally ignore
position. This is not a claim that token permutations always conjugate A when
positional features enter Q/K. Finite logits produce strictly positive weights;
no unattainable hard-routing weights are used.

The afterword identifies the real NeelNanda/Attn_Only_1L512W_C4_Code checkpoint
as a possible bridge to GPT-2: one attention-only layer with eight heads.
The current site does not run that checkpoint or claim measurements from it.

## Implementation and checking

`app.mjs` derives scene state directly from native scroll position. There are no
autoplay timers or required clicks. Reversing the scroll restores earlier states.
Mathematics uses self-hosted KaTeX with MathML output, including SVG labels.
The reading version contains all scene explanations and equations.

`prefers-reduced-motion: reduce` shows completed diagram states at scroll
thresholds. The softmax calculation retains its three discrete exact stages.
`?motion=reduce` selects that same rendering path for portable browser QA.
No browser-preference setting is required or changed by that test override.

Run:

```sh
node --check docs/attention/app.mjs
node --check docs/attention/model.mjs
node --test tests/attention-layer.test.mjs tests/attention-socks.test.mjs
```

Six tests independently check the known numeric projections and mixtures,
normalization, value/output shapes, stable softmax, correct causal masking, and
invariance of earlier causal outputs to future input changes. Four further tests
check the independent two-order arithmetic, interventions, shared projections,
and exact readout of the socks-and-shoes construction.

Manual review includes forward/backward native scrolling, narrow/mobile layout,
PageUp/PageDown navigation, the reading link, all lesson scenes, math rendering, and
the reduced-motion rendering path. Continuous intermediate states retain honest
numeric labels; softmax uses discrete exact calculations rather than interpolating
unnormalized values while calling them probabilities.
