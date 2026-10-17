# From words to vectors

A standalone scrollytelling lesson for POLS 4728, with three directly linkable
chapters: `#bag-of-words`, `#tf-idf`, and `#word2vec`.

The fixed scene stays pinned while scroll position gathers tokens, transforms
counts into TF–IDF weights, combines context vectors, and applies a CBOW gradient
step. Previous/next buttons provide a keyboard route through all stages. A
motion toggle uses completed threshold states; the same mode is selected by
`prefers-reduced-motion`. Continuous reading versions are included in HTML.

## Content and numerical conventions

- Four synthetic political documents; lowercase alphabetic tokens, no stemming
  or stop-word deletion.
- Raw-count TF × ln(N/df), without smoothing or vector normalization. Cosine
  similarity handles vector length at comparison time.
- A constructed six-word, two-dimensional CBOW example with distinct input and
  output tables, a symmetric radius-one context, and full softmax. Its parameters
  are illustrative rather than corpus-trained. The gradient step updates both
  tables simultaneously. Intermediate probabilities are recomputed from the
  corresponding fractional parameter step.
- Mosteller–Wallace's approximate reported “upon” rates are normalized per 1,000
  words in their comparison writings. The study combines many features; the
  chart does not purport to reproduce an authorship classifier.
- The Revelation–Gospel of John comparison summarizes Erwin–Oakes (2012), using
  Greek common-word profiles and explicitly distinguishing textual differences
  from an authorship conclusion. The diagram is schematic, not a replication.

All research references are linked in the lesson. Fonts are reused from the
course's public attention page with the GUST font license included. The visual
direction follows `design/README.md` and `design/tokens.json`; no new identity
artwork is introduced.

## Verification

Run `node --test tests/text-representations.test.mjs` from the repository root.
The tests check token/count matrices, independently calculated TF–IDF and cosine
values, normalized CBOW probabilities and loss, every gradient against numerical
finite differences, and the direction of a simultaneous update.

Manual browser review covered 1280×820, 390×844, and 375×667 viewports; forward
and backward scrolling; pinned diagrams; numerical endpoint states; term
selection; previous/next keyboard activation; and the motion-off threshold mode.
At 375×667, all six word2vec stages fit above their navigation controls. This
is not a claim of a complete assistive-technology audit.

No build step or external runtime dependencies are required. Serve `docs/` and
open `/text/` to preview.
