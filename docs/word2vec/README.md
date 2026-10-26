# Word2vec: learning a geometry from context

A self-contained, 15-scene scrollytelling lesson for the machine-learning course.
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
output row, raising the observed target probability from 0.22497 to 0.26856.
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
normalization, stable softmax, and loss descent. The render harness runs all 15
scenes at four scroll phases in wide and narrow layouts with the bundled KaTeX
runtime. It checks missing math escapes, finite and in-bounds diagram coordinates,
meaningful geometry changes, and numerical integration with the model.

These are programmatic checks. They do not establish browser-level verification
of forward/backward scrolling, actual text extents, keyboard navigation, or
reduced-motion behavior. Those require review of the running page.

## Browser review

The first version was inspected in the in-app browser at its desktop viewport and a 390 × 844 phone viewport. Review covered the opening lantern artwork, input-column extraction, the two matrix layouts, dot products, softmax, parameter updates, geometry, forward and backward scrolling, native PageDown navigation, and the reduced-motion route. No browser console errors were observed. Mathematical and layout rendering checks supplement the manual review; this is not an accessibility certification.

Opening artwork is registered in design/assets.json. It was generated with the built-in image tool using the course's round torsoless character specimen; word labels and the title remain real text. The prompt is retained outside this site's repository.

## Appended word-vector exploration

The original title, all 15 scenes, model calculations, styling, artwork, endnotes,
and sources are preserved. Ten new scenes follow the original afterword on the
same page. The addition covers stored vectors, normalization, cosine similarity,
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
exact original title/story/afterword markup, append order, and unique IDs against
`preservation.json`. Run it with the model and render checks above, plus
`node exploration-math.test.mjs` and `node exploration-render.test.mjs`.
