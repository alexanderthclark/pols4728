# Attention: how tokens share information

A buildless POLS 4728 scrollytelling lesson. Serve `docs/` and open `/attention/`.
The course homepage links to the lesson. Local fonts and KaTeX require no CDN.

## Scope and convention

Seventeen scroll-driven scenes explain token-row representations, equal and
hand-chosen mixtures, left multiplication by A, query/key projections, dot products,
scaled score matrices, exact score → exponential → softmax phases, values, an
optional output map, causal masking before softmax, renormalization, and the final
input-dependent attention operation. No general transformer architecture is taught.

X is T × d; Q = X WQ; K = X WK; A = row-softmax(Q Kᵀ / sqrt(dk) + M);
V = X WV; Y = A V. A's rows are destinations and columns are sources.
The diagonal is retained under causal masking. The numeric example is invented,
explicitly labeled, and computed by `model.mjs`, not a fitted linguistic claim.
The hand-chosen introductory mixture is separate from the query-derived example.

Primary reference: Elhage et al. (2021), A Mathematical Framework for Transformer
Circuits. Secondary: Prince, Understanding Deep Learning, chapter 12 §§12.2–12.3.
The page explains the transpose convention and includes the exact toy maps. A
broader notation harmonization with the course notes remains for a later pass.

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
node --test tests/attention-layer.test.mjs
```

Six tests independently check the known numeric projections and mixtures,
normalization, value/output shapes, stable softmax, correct causal masking, and
invariance of earlier causal outputs to future input changes.

Manual review includes forward/backward native scrolling, narrow/mobile layout,
PageUp/PageDown navigation, the reading link, all 17 scenes, math rendering, and
the reduced-motion rendering path. Continuous intermediate states retain honest
numeric labels; softmax uses discrete exact calculations rather than interpolating
unnormalized values while calling them probabilities.
