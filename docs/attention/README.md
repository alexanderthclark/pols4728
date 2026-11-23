# Attention: how tokens share information

A buildless POLS 4728 scrollytelling lesson. Serve `docs/` and open `/attention/`.
The course homepage links to the lesson. Local fonts and KaTeX require no CDN.

## Scope and convention

The lesson opens with its full illustrated title page, then three intuition
scenes contrast the two user-supplied desert sentences. They distinguish a
shared token lookup vector from positional information and context-dependent
updates. These are conceptual diagrams, not measured sense disambiguation or
attention weights. Scrolling then enters the residual-stream overview and
continues through all worked examples. No teaching chapter is hidden behind
a disclosure panel.
The following eight-scene story starts with a sentiment encoder and the residual stream, using
these two reviews as the task:

- “Good and Infinite Jest don't belong in the same sentence”
- “Infinite Jest wasn't just good. It was spectacular.”

Their negative and positive labels are intended readings, not measured model
predictions. The opening traces token representations through residual attention
and MLP updates into a final sentence readout. Its uniform, empty matrix cells
identify the matrix's dimensions; they do not encode fitted weights. It then extracts the attention
matrix from one head, reads a source/destination entry, combines a row of value
vectors, and returns the projected output to the residual stream. The matrix is
symbolic; highlighted entries do not assert learned weight magnitudes. For
legibility the first sentence is shown as ten word-level units, with an explicit
note that an actual tokenizer may split them further.

The thirteen-scene numerical classifier follows in the main scroll sequence.
It starts from a complete three-by-three matrix, computes U=AV, mean-pools the
output rows into h, and applies a linear readout. This separate constructed
example omits residual paths and MLPs; h is not the final residual state in the
main architecture. Its interpretation comparisons and exact arithmetic remain.

The original twenty socks-and-shoes scenes are also part of the main scroll sequence.
The first thirteen
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
V = X WV; U = A V. A's rows are destinations and columns are sources. U contains mixed feature
rows, not outcome labels; actual targets and predictions retain y notation.
The value projection is WV and the output projection is WO, so V = X WV and
deltaX = A X WV WO. Attention biases and normalization are omitted in this
schematic. The maps are shared across token positions.
For a representation update, deltaX = U WO and X′ = X + deltaX. This update
illustration does not replace the original task's U-based readout. The later
causal example changes the information available, and does not retain the
original task's target-score claim. The new opening locates attention in a simplified encoder block; the separate residual-stream lesson treats the architecture in more detail.

Primary reference: Elhage et al. (2021), A Mathematical Framework for Transformer
Circuits. Secondary: Prince, Understanding Deep Learning, chapter 12 §§12.2–12.3.
The page explains the transpose convention and includes the exact toy maps. Grant Sanderson’s attention lesson is an additional visual reference. The opening defines the value and output projections before using the Q/K/V maps.

The Infinite Jest comparison progressively underlines the shared word, then the
negative and positive phrases. Both complete sentences and intended readings
remain visible. The measured-predictions scene centers its cards without a left
narration panel; the model description remains in the preceding introduction
and the full explanation stays in the reading transcript.

## Measured Infinite Jest results

After the eight schematic scenes, three additional scroll scenes show predictions
and attention exported from `distilbert/distilbert-base-uncased-finetuned-sst-2-english`,
pinned to revision `714eb0fa89d2f80546fda750413ed43d93601a13`. This SST-2
checkpoint has six blocks and twelve heads per block. Its classifier reads the
final `[CLS]` state through a linear pre-classifier, ReLU, and a final linear map;
it does not use the schematic overview's mean pooling. The model scores overall
sentiment, not sentiment toward an independently specified target.

The export includes both unchanged sentences, actual WordPiece tokens and special
tokens, token IDs and offsets, logits, softmax probabilities, and all 72 full
attention matrices for each sentence. No special-token columns are discarded or
renormalized. The initial comparison fixes layer 6/head 1 by position, rather
than selecting a head for an appealing pattern. A following explorer lets the
reader inspect every layer, head, sentence, and destination with an exact row
table. Colors use one linear 0–1 scale. Intended readings remain distinct from
model predictions; attention coefficients are not signed sentiment attributions.

`data/infinite-jest-attention.json` stores the computed results, checkpoint
revision, package versions, precision, evaluation settings, and row-sum checks.
`trained-scenes.mjs` and `trained-explorer.mjs` only display these results.
The page downloads the small results file, not the model weights.

Reproduce with Python 3.11:

```sh
python -m pip install -r scripts/requirements-attention-model.txt
python scripts/export_attention_model.py --output docs/attention/data/infinite-jest-attention.json
```

The `Export measured attention` GitHub Actions workflow runs the same export and
saves an artifact. It has read-only repository permissions and does not commit
generated results. Inference uses CPU float32, evaluation mode, eager attention,
and the pinned checkpoint's safetensors weights, without custom model code.

## Interpreting a classification

`interpretation-model.mjs` defines a hand-constructed self-attention classifier,
not a trained sentiment model or a reconstruction of LiDDA. The three tokens
[film, was, awful] label both the destination rows and source columns of
A=[[.80,.15,.05],[.60,.30,.10],[.40,.45,.15]]. Its rows sum to one; columns
need not, and the matrix is not symmetric. Q=I3 and
K=sqrt(3)*transpose(log(A)), with entrywise logarithms, give scaled scores
QKᵀ/sqrt(3)=log(A). Row-wise softmax realizes the displayed matrix exactly.

Values are V=[[0.1,4],[0,1],[-8,2]]. The three attention outputs are
U=AV=[[-.32,3.45],[-.74,2.90],[-1.16,2.35]]. Mean pooling gives the sentence
summary h=mean(U)=[-.74,2.90]. Equivalently, h=sum_j alpha_j*v_j, where
alpha_j=mean_i(a_ij): the effective source weights are the column means
[.60,.30,.10], computed across all destinations rather than selected from one row.
A column readout beta=[1,0]ᵀ and zero intercept give logit -.74 and
positive-class probability sigmoid(-.74), about .323.

The signed logit terms are +0.06, 0, -0.8. Replacing awful with great changes
only its first value coordinate to +8, keeping queries, keys, and the full
attention matrix fixed by explicit construction: logit .86, probability about
.703. Deleting film removes its destination row and source column; the retained
query, key, and value vectors preserve their original coordinates, and the
query/key dimension stays three. Recomputing softmax over the remaining sources
gives [.75,.25] in both surviving attention rows. Both output rows are [-2,1.25],
so mean pooling and the fixed readout give logit -2. The deletion effect is not
film's former +.06 term. Values may already contain information from other tokens
in a deeper transformer, and a nonlinear classifier does not share this toy's
simple additive logit decomposition.

Primary interpretation references: Kobayashi et al. (2020), Attention is Not
Only a Weight; Jain and Wallace (2019); Wiegreffe and Pinter (2019); Abnar and
Zuidema (2020); Sundararajan et al. (2017); and LiDDA v3 (27 May 2026).
The LiDDA section distinguishes normalized attention credit, prediction
sensitivity, and independently identified effects. It does not invent the
paper's incompletely specified query-axis reduction.

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
25% to itself; socks allocates 50% to each. With U=AV and readout
2*mean(U)-0.5, the predictions are 0 for socks→shoes and -1 for shoes→socks.
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

`app.mjs` initializes the intuition, three original pinned stories, and measured-model chapter. `intuition-scenes.mjs` introduces meaning in context; `sentence-scenes.mjs` is the default
route; `interpretation-scenes.mjs` supplies the numerical chapter; and the
original `socks-scenes.mjs` plus `extension-scenes.mjs` supply the full
worked-example chapter. Hash links offer shortcuts; scrolling reaches every scene. It derives
scene state directly from native scroll position. There are no
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
node --check docs/attention/sentence-scenes.mjs
node --check docs/attention/socks-model.mjs
node --check docs/attention/socks-scenes.mjs
node --check docs/attention/extension-scenes.mjs
node --check docs/attention/interpretation-model.mjs
node --check docs/attention/interpretation-scenes.mjs
node --test tests/attention-layer.test.mjs tests/attention-socks.test.mjs tests/attention-interpretation.test.mjs tests/attention-sentences.test.mjs
```

Six tests independently check the known numeric projections and mixtures,
normalization, value/output shapes, stable softmax, correct causal masking, and
invariance of earlier causal outputs to future input changes. Seven socks-model tests
check independent two-order arithmetic, interventions, shared projections, the
exact readout, causal masks, and the output projection/residual addition.
Six interpretation tests cover all three attention and output rows, row
normalization and asymmetry, mean pooling and effective source weights, signed
logit terms, fixed Q/K under value substitution, full row-and-column deletion,
and deterministic rendering of the thirteen scenes. These three suites contain
nineteen tests in total. Three more tests check the exact sentences and complete
word sequence, the residual/readout structure, and strict math plus reversible
rendering for the eight new scenes: twenty-two tests overall.

Manual review includes forward/backward native scrolling, narrow/mobile layout,
PageUp/PageDown navigation, the reading link, all lesson scenes, math rendering, and
the reduced-motion rendering path. Continuous intermediate states retain honest
numeric labels; softmax uses discrete exact calculations rather than interpolating
unnormalized values while calling them probabilities.

