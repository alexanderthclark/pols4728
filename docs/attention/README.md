# Attention: how tokens share information

A buildless POLS 4728 scrollytelling lesson. Serve `docs/` and open `/attention/`.
The course homepage links to the lesson. Local fonts and KaTeX require no CDN.

## Scope and convention

Twenty scroll-driven scenes follow a single two-token example. The first thirteen
establish the target, show why identity pooling loses order, add explicit position,
and motivate a chosen retrieval: shoes looks for socks and reads their position.
Fixed query/key/value maps implement that rule; softmax, left multiplication,
a task readout, order reversal, and ablations complete the calculation.

Seven short extension scenes keep the same example. They generalize scalar
query/key products to vector dot products, project an attention output back to
representation width, add that update to the original representation, then switch
to left-to-right prediction to motivate causal masking and renormalization. The
former seventeen-scene four-token restart has been removed.

X is T × d; Q = X WQ; K = X WK; A = row-softmax(Q Kᵀ / sqrt(dk) + M);
V = X WV; Y = A V. A's rows are destinations and columns are sources.
For a representation update, deltaX = Y WO and X′ = X + deltaX. This update
illustration does not replace the original task's Y-based readout. The later
causal example changes the information available, and does not retain the
original task's target-score claim. No general transformer architecture is taught.

Primary reference: Elhage et al. (2021), A Mathematical Framework for Transformer
Circuits. Secondary: Prince, Understanding Deep Learning, chapter 12 §§12.2–12.3.
The page explains the transpose convention and includes the exact toy maps. Grant Sanderson’s attention lesson is an additional visual reference. A
broader notation harmonization with the course notes remains for a later pass.

## One-head task

`socks-model.mjs` stores each token as [socks indicator, shoes indicator, position].
The lesson introduces X=E+P: token embeddings and position vectors have the same
width. The diagram displays both complete 2×3 operands (including zeros), a
literal plus sign, and the 2×3 sum, with matching entries highlighted as the
scroll works through identity and position coordinates. Learned positional embeddings do not require a reserved coordinate. We
choose fixed position vectors supported on the third coordinate for this example;
token embeddings have zero in that coordinate. Position is +1 in the first slot
and -1 in the second. Fixed projections read
q=shoes, k=log(3)*socks, and v=position. Thus shoes allocates 75% to socks and
25% to itself; socks allocates 50% to each. With Y=AV and readout
2*mean(Y)-0.5, the predictions are 0 for socks→shoes and -1 for shoes→socks.
No causal mask, feed-forward network, residual, or normalization is needed for
the initial constructed task. The parameters are chosen, not trained. The later
update illustration uses WO=[0,0,1], producing deltaX=[[0,0,0],[0,0,0.5]]
for socks first. The original third coordinate of shoes changes from -1 to -0.5;
the updated feature is no longer simply the original position indicator.
Removing position or intervening with uniform attention yields -0.5 for both.

The matching preferences survive the swap because Q/K intentionally ignore
position. This is not a claim that token permutations always conjugate A when
positional features enter Q/K. Finite logits produce strictly positive weights;
no unattainable hard-routing weights are used.

For the masking extension, reverse the sequence to [shoes,socks]. Its original
attention is [[0.25,0.75],[0.5,0.5]]. Blocking the first destination from reading
the later socks token gives [[1,0],[0.5,0.5]]. The diagonal stays available and
each allowed row is normalized. Uniform attention with a causal mask averages
only allowed positions. Masking changes the toy task predictions, so the
extension discusses next-token information constraints rather than those scores.

The afterword identifies the real NeelNanda/Attn_Only_1L512W_C4_Code checkpoint
as a possible bridge to GPT-2: one attention-only layer with eight heads.
The current site does not run that checkpoint or claim measurements from it.

## Implementation and checking

`app.mjs` combines `socks-scenes.mjs` and `extension-scenes.mjs` and derives scene state directly from native scroll position. There are no
autoplay timers or required clicks. Reversing the scroll restores earlier states.
Mathematics uses self-hosted KaTeX with MathML output, including SVG labels.
The reading version contains all scene explanations and equations.

`prefers-reduced-motion: reduce` shows completed diagram states at scroll
thresholds. The addition, softmax, and masking calculations retain discrete exact stages.
`?motion=reduce` selects that same rendering path for portable browser QA.
No browser-preference setting is required or changed by that test override.

Run:

```sh
node --check docs/attention/app.mjs
node --check docs/attention/socks-model.mjs
node --check docs/attention/socks-scenes.mjs
node --check docs/attention/extension-scenes.mjs
node --test tests/attention-layer.test.mjs tests/attention-socks.test.mjs
```

Six tests independently check the known numeric projections and mixtures,
normalization, value/output shapes, stable softmax, correct causal masking, and
invariance of earlier causal outputs to future input changes. Seven socks-model tests
check independent two-order arithmetic, interventions, shared projections, the
exact readout, causal masks, and the output projection/residual addition.

Manual review includes forward/backward native scrolling, narrow/mobile layout,
PageUp/PageDown navigation, the reading link, all lesson scenes, math rendering, and
the reduced-motion rendering path. Continuous intermediate states retain honest
numeric labels; softmax uses discrete exact calculations rather than interpolating
unnormalized values while calling them probabilities.
