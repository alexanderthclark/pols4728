# Word2vec: learning a geometry from context

A self-contained, 17-scene scrollytelling lesson for the machine-learning course.
The opening scene shows two round, torso-less explorers entering a vector cave,
with a lantern and the labels man, king, woman, and queen. The title and labels
remain real text. The lesson uses the course palette and bundled KaTeX for
mathematical notation.

## Viewing

Serve `docs/word2vec/` with any local static web server and open its `index.html` over
HTTP. No build step or external runtime request is needed. Scroll forward to
develop the explanation and backward to revisit earlier states. The page also
provides a complete reading version and links to its sources.

The scroll sequence covers the context-prediction task, training pairs, one-hot
identity, input-matrix columns, embedding lookup, output-matrix rows, dot-product
scores, softmax, loss, one exact gradient update, updates across a tiny corpus,
the two vectors for one word, input-vector similarity, analogy directions, and
the representation retained after training.

## Model scope

This version teaches skip-gram with full softmax and omits bias terms. CBOW is
mentioned as an alternative architecture in the endnote. The six-word vocabulary
is voters, support, reform, senate, backs, and proposal. With embedding dimension
two, the input matrix has 2 rows and 6 columns; the output matrix has 6 rows and
2 columns. The matrices are independent learned parameters.

The numerical example uses illustrative starting weights. Its first pair is
support → reform. One gradient step changes the selected input column and every
output row, raising the probability of the observed context word from 0.22497 to 0.26856.
The next scene uses directed adjacent-word pairs from four short sentences.
This demonstration is too small to produce a useful semantic model. The final
similarity and analogy diagrams are separate illustrative geometry, not fitted
results from this example or guaranteed identities.

`model.js` exports the vocabulary, initial weights, forward calculation,
gradient step, interpolation, and a separate single-pair training path.
`app.js` uses that module for its numerical scenes. Math and its fonts are
served locally from `vendor/katex/`. Source inspiration and citations appear
at the end of the lesson, including Chris Olah’s “Deep Learning, NLP, and
Representations” and Mikolov et al.’s original word2vec paper.

## Validation

From this directory, run:

```sh
node model.test.mjs
node render.test.mjs
```

The model checks cover matrix dimensions, forward values, numerical gradient
verification, unchanged unselected input columns, immutable updates, probability
normalization, stable softmax, and loss descent. The render harness runs all 17
scenes at four scroll phases in wide and narrow layouts with the bundled KaTeX
runtime. It checks missing math escapes, finite and in-bounds diagram coordinates,
meaningful geometry changes, and numerical integration with the model.

These are programmatic checks. They do not establish browser-level verification
of forward/backward scrolling, actual text extents, keyboard navigation, or
reduced-motion behavior. Those require review of the running page.

## Browser review

The first version was inspected in the in-app browser at its desktop viewport and a 390 × 844 phone viewport. Review covered the opening lantern artwork, input-column extraction, the two matrix layouts, dot products, softmax, parameter updates, geometry, forward and backward scrolling, native PageDown navigation, and the reduced-motion route. No browser console errors were observed. Mathematical and layout rendering checks supplement the manual review; this is not an accessibility certification.

Opening artwork is registered in design/assets.json. It was generated with the built-in image tool using the course's round torsoless character specimen; word labels and the title remain real text. The prompt is retained outside this site's repository.

## Archived word-vector exploration

The original title, 15-scene sequence, model calculations, styling, artwork,
endnotes, and sources are preserved. A requested clarity review adds a network
overview before the matrix walkthrough and clarifies two sentences about the
identity activation and retained embedding table. The ten exploration scenes formerly followed
the original afterword. They are now removed from the active lesson; their complete
HTML is retained in `../../archive/word2vec-exploration/section.html`, with
restoration instructions alongside it. All implementation files remain intact. The addition covers stored vectors, normalization, cosine similarity,
signed addition, nearest-neighbor ranking, multiplicative scoring, Euclidean
distance on unit vectors, and interpreting a t-SNE map.

The addition uses exact calculations on invented 2D word vectors; the neighbors
are not Google News results. Its separate t-SNE map is an actual fit to 18
synthetic 6D unit vectors, with recorded source and map neighbors.
`generate_projection.py` reproduces that authoring asset; no pretrained model
was downloaded. These parameters and software are authoring choices, not student
requirements.

The added code lives in `exploration.js`, with CSS scoped to the added
section in `exploration.css`. It does not override the original title or
story styles. `original-preservation.test.mjs` checks the original file hashes,
exact original title/story/afterword markup and unique IDs against
`preservation.json`, allowing only the explicit requested clarifications recorded
in `clarity-changes.json`. The original baseline is never replaced. The check also verifies the archived
fragment hash and that the active page no longer loads the exploration. Run it with the model and render checks above, plus
`node exploration-math.test.mjs` and `node exploration-render.test.mjs`.

## Neural-network overview

After the training pair, a neuron diagram shows six one-hot input units, two
linear hidden units, six scores, and a separate nonlinear softmax. Its two
connection bundles are the independent matrices W_in and W_out. Scrolling
follows the selected input through the embedding to probabilities; reverse
scroll restores the earlier calculation. The original matrix walkthrough
then develops those operations. The retained representation is the whole input
embedding table, with one fixed vector per vocabulary word. The network
derivation by Xin Rong is linked in the final references.

## Explicit training pairs

The training-pair scene shows two rows with center word x (input) and context
word y (output), retaining the word notation x = w_c and y = w_o. Scrolling duplicates the center-word input and
selects the support–reform example. The following network equation explicitly
identifies d_c as the one-hot encoding of x. The loss explanation uses the same
context-word terminology. Other walkthrough scenes remain unchanged, and the
homepage remains unlisted.

## Cosine pause at the analogy

The king − man + woman query ends near, rather than exactly at, queen.
A separate scroll step then compares their directions and rescales both arrows
to unit length while preserving their angle. Cosine similarity is the normalized
dot product; cosine distance is one minus that similarity. The comparison uses
the full embeddings in practice, not coordinates from a 2D projection.
`cosine.js` shares the illustrative coordinates across the two scenes. The
reading version includes the same explanation.

The cosine revision was reviewed at desktop width and in a 390 × 844 viewport,
including forward/backward scrolling, keyboard navigation, and the reduced-motion
route. The render checks verify that normalization preserves the cosine and ends
with equal arrow lengths. The archive-preservation check verifies that the removed
HTML is unchanged and its scripts are no longer loaded by the active page.
