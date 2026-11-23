# Attention: how tokens share information

A buildless POLS 4728 scrollytelling lesson. Serve `docs/` and open `/attention/`.
The course homepage links to the lesson. Local fonts and KaTeX require no CDN.

## Scope and convention

The lesson opens with its full illustrated title page, then six intuition
scenes precede the socks-and-shoes construction. A residual-stream overview first defines X_0=SW+P:
one-hot token rows S select learned embedding rows from W, and P adds position.
The next three scenes contrast the two user-supplied desert sentences. The
comparison emphasizes that initial rows do not yet incorporate surrounding words;
position can make the two rows different without contextualizing their meaning. They distinguish a
shared token lookup vector from positional information and context-dependent
updates. These are conceptual diagrams, not measured sense disambiguation or
attention weights. Two further scenes separate learned W_Q, W_K, W_V, W_O
from computed Q, K, V, and A(X), then follow repeated attention and MLP
updates to the running token vectors. Q/K determines this head’s attention
pattern; V/O transforms the mixed content. Parameters are fixed during
inference and generally differ across heads and layers; the attention weights
are computed from the current input representations. Refinement need not
mean monotonic improvement: an update can revise or cancel a feature.
Scrolling then enters the complete socks-and-shoes construction and its encoder
notebook bridge. All Infinite Jest material follows together: the seven-scene
schematic, measured predictions, exact Shapley contributions, and internal
interventions. No teaching chapter is hidden behind a disclosure panel.
The Infinite Jest schematic starts by comparing these two reviews:

- “Good and Infinite Jest don't belong in the same sentence”
- “Infinite Jest wasn't just good. It was spectacular.”

Their negative and positive labels are intended readings, not measured model
predictions. The next scene opens one attention head in the schematic encoder.
Its uniform, empty matrix cells identify dimensions; they do not encode fitted
weights. The story extracts this attention matrix, reads an entry, combines a
row of value vectors, then returns the projected output to the residual stream
and follows the MLP update into the sentence readout. The matrix is
symbolic; highlighted entries do not assert learned weight magnitudes. For
legibility the first sentence is shown as ten word-level units, with an explicit
note that an actual tokenizer may split them further.

The socks-and-shoes calculation and encoder notebook bridge precede the entire
Infinite Jest chapter, including its schematic introduction. The research
interpretation guide follows the measured chapter. The reading transcript follows
the same scene order. The former thirteen-scene film
classifier has been removed from the live route, navigation, and reading
version. Its source modules and arithmetic tests remain as an archived example.

Twenty-one socks-and-shoes scenes are part of the main scroll sequence.
The first twelve
establish the target, show why identity pooling loses order, add explicit position,
and motivate a chosen retrieval: shoes looks for socks and reads their position.
The query/key choice explicitly sets d_k=1: WQ and WK are 3×1, while Q and K
are 2×1. This width is an architecture choice; the map entries are parameters.
Fixed query/key/value maps implement that rule; softmax, left multiplication,
a task readout, order reversal, and ablations complete the calculation.

Two takeaway scenes then make the non-uniqueness explicit. For socks-first
A=[[a,1-a],[b,1-b]], position values [1,-1] give pooled h=a+b-1. Reversing
identities swaps both rows and columns of A, giving -h. Any a+b != 1 separates
the two orders; a shared recalibrated readout h_current/[2(a+b-1)]-1/2 recovers
0 and -1. The original readout works exactly when a+b=1.25. Equal column totals
(the doubly stochastic family a+b=1) erase the signal, including identity, swap,
uniform attention, and balanced selective mixing. Biased identical rows can
work, so rank and invertibility are not the criterion. The scroll shows three
successful patterns and four failures; the complete-text explanation derives
the condition and its realization by one-dimensional identity-only Q/K maps.

Seven short extension scenes keep the same example. They generalize scalar
query/key products to vector dot products, project an attention output back to
representation width, add that update to the original representation, then switch
to left-to-right prediction to motivate causal masking and renormalization. The
former seventeen-scene four-token restart has been removed.

The socks chapter ends with a bridge to the annotated encoder classifier in the
course repository, pinned to 5a29ace12e91b1aca3d26eea54b4cbceba92fb22. It connects
the hand-chosen two-token solution to learning A-before-B among distractors.
For each swapped pair, even the initial token-plus-position mean is identical;
the encoder must transform the representations so order survives mean pooling.
The notebook trains embeddings, two four-head encoder blocks, and a linear
classifier together. The link requires access to the course repository; the
notebook itself is not copied to this public site.

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

## Measured Infinite Jest investigation

The measured sequence uses `distilbert/distilbert-base-uncased-finetuned-sst-2-english`,
pinned to revision `714eb0fa89d2f80546fda750413ed43d93601a13`. Its six blocks
contain twelve heads each; the classifier reads final `[CLS]` through a linear
pre-classifier, ReLU, and a final linear map. It scores overall sentiment and
does not use the schematic overview's mean pooling.

The centered scroll sequence connects input contrasts, attention messages, and
prediction changes. Numbers come from saved inference results:

- Removing the final sentence from the positive review leaves P(positive) at
  99.93%; removing `just` lowers it to 0.03%. In layer 2/head 9, `t` reads `just`
  and `just` reads `t`. Removing those two messages without renormalization
  changes the short review from 99.93% to 10.02%. The same two-message removal
  on the full review changes 99.98% to 99.95%; neither experiment removes a
  whole head. The two reads use incoming states in parallel.
- In the negative review's layer 6/head 10, `[CLS]` gives the apostrophe 87.31%
  attention and `t` 4.29%. Their projected value norms differ enough that the
  resulting message norms are similar: 0.439 and 0.448. Removing only the
  apostrophe message changes P(positive) from 0.3348% to 0.3386%. Message norms
  describe vector magnitude, not signed sentiment contributions.
- The context extension contrasts `bad, not good` with `good, not bad`.
  Exchanging the contextual state at `not` after block 4 is one intervention.
  Replacing only its layer-5 value projections across all twelve heads is
  another: the recipient's layer-5 attention weights stay exactly fixed while
  its prediction reverses. Layer 6 and the remaining computation rerun.

The export retains all five tested title variants, a separately renormalized
message-removal check, and eight additional state-transfer pairs fixed before
validation (15 of 16 directional transfers reverse the predicted class).
These are limited construction-specific checks. The examples and routes were
selected after inspecting this checkpoint; they do not establish universal head
roles, additive token attributions, or population accuracy.

`data/infinite-jest-attention.json` retains both original reviews, their exact
WordPiece and special tokens, logits, probabilities, and all 72 full attention
matrices per review. The explorer keeps every layer, head, and destination,
with an exact row table and a shared linear 0–1 color scale. No special-token
columns are omitted or renormalized. `data/infinite-jest-investigation.json`
adds the selected matrices, intervention outputs, vector norms, validation
cases, checkpoint hash, package versions, and implementation checks.
`trained-scenes.mjs`, `trained-tour-scenes.mjs`, and `trained-explorer.mjs`
display the exports without downloading model weights. Detailed methods and
qualifications remain in the reading transcript; the pinned visual has no left
narration panel.

Reproduce from the repository root using the pinned dependencies:

```sh
python -m pip install -r scripts/requirements-attention-model.txt
python scripts/export_attention_model.py --output docs/attention/data/infinite-jest-attention.json
python scripts/export_attention_investigation.py --output docs/attention/data/infinite-jest-investigation.json
```

The investigation exporter downloads the pinned checkpoint unless `--model-path`
points to an existing local copy. `--reference` can select another copy of the
original attention export. It verifies the safetensors hash, reproduces the
original predictions and matrices, and checks the intervention mechanics.
It reruns the declared examples, selected interventions, and validation cases;
it does not repeat the exploratory head search. Inference uses CPU float32,
evaluation mode, eager attention, and no custom model code.

The `Export measured attention` GitHub Actions workflow reproduces the original
attention export and saves an artifact with read-only repository permissions;
it does not commit generated results.

## Exact Shapley explanation of the original reviews

Three scenes follow the original measured predictions: define a missing token,
then decompose the negative and positive review scores. The display uses the same
positive-minus-negative logit margin as the intervention chapter. For binary
softmax this equals log(P(positive)/P(negative)). It is an explanation of overall
sentiment for the complete input, not a target-conditioned sentiment model.

Each original input has thirteen non-special WordPiece tokens. Each token
position is one player. For every one of its 8,192 subsets, omitted tokens are
replaced by the checkpoint's [MASK] ID. [CLS], [SEP], input length, position IDs,
and the all-one attention mask remain fixed. The full classifier reruns for
every subset; attention weights are not held fixed. The ordinary Shapley formula
averages marginal score changes with weights |S|!(n-|S|-1)!/n!. This is exhaustive
coalition enumeration, not Partition/Owen attribution or permutation sampling.
The baseline is the score of the fully masked non-special input, not a neutral
score. This masking rule defines the game; a deletion or imputation rule defines
a different game.

`data/infinite-jest-shap.json` records the measured baseline, original score,
individual token contributions, original-prediction checks, and provenance.
The export script checks that baseline plus all attributions reconstructs the
original margin. The readable explanation retains the token details. The plot
sums subword contributions for readable word labels; it does not recompute a
separate word-player game. The signed waterfall is an allocation of the model's
prediction relative to this baseline, while the following interventions test
particular internal computations.

Reproduce with the attention-model dependencies and cached or downloaded pinned
checkpoint:

```sh
python scripts/export_attention_shap.py --output docs/attention/data/infinite-jest-shap.json
```

## Archived classification example

This film example is retained in source only; the live lesson does not import
its scenes or include its worked arithmetic.

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

`app.mjs` initializes four pinned stories in scroll order: the intuition primer,
socks-and-shoes calculation, Infinite Jest schematic, and measured-model chapter.
`intuition-scenes.mjs` introduces meaning in context; `sentence-scenes.mjs`
supplies the schematic; `trained-tour-scenes.mjs` displays measured results, `shap-scenes.mjs` inserts
the exact Shapley explanation after the original predictions; and
`socks-scenes.mjs` plus `extension-scenes.mjs` supply the full worked example.
`interpretation-scenes.mjs` is archived and is not imported by the live app. Hash links offer shortcuts; scrolling reaches every scene. It derives
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
node --check docs/attention/trained-tour-scenes.mjs
node --test tests/attention-*.test.mjs
```

Six tests independently check the known numeric projections and mixtures,
normalization, value/output shapes, stable softmax, correct causal masking, and
invariance of earlier causal outputs to future input changes. Twelve socks-model tests
check independent two-order arithmetic, interventions, shared projections, the
exact readout, causal masks, and the output projection/residual addition.
Six archived interpretation tests cover all three attention and output rows, row
normalization and asymmetry, mean pooling and effective source weights, signed
logit terms, fixed Q/K under value substitution, full row-and-column deletion,
and deterministic rendering of the thirteen scenes. These three suites contain
twenty-four tests in total. Three more tests check the exact sentences and complete
word sequence, the residual/readout structure, and strict math plus reversible
rendering for the seven sentence scenes: twenty-seven tests overall.

Manual review includes forward/backward native scrolling, narrow/mobile layout,
PageUp/PageDown navigation, the reading link, all lesson scenes, math rendering, and
the reduced-motion rendering path. Continuous intermediate states retain honest
numeric labels; softmax uses discrete exact calculations rather than interpolating
unnormalized values while calling them probabilities.

`tests/attention-preview.html`, served from the repository root, provides desktop
(1120 × 780), mobile (390 × 844), and small-mobile (360 × 640) iframe viewports
for reviewing the actual lesson. Its reduced-motion checkbox reloads the lesson
with the documented query override. These are review controls outside the
published `docs/` tree; they are not required by students or used in the story.

