# Text as Data

Bag of Words and TF–IDF

A standalone scrollytelling lesson for POLS 4728: an opening question about
recognizable writing habits, followed by bag of words and TF–IDF.
Its 33 scenes comprise 3 on AI writing, 11 on bag of words, 6 on TF–IDF,
and 13 application scenes at the end: one on authorship, two on BERTopic,
and ten on the probability models behind LDA and STM.
Bag of words and TF–IDF share one pinned workspace, while retaining separate
chapter headings and anchors. The chapter anchors are `#ai-writing`, `#bag-of-words`, `#tf-idf`, `#authorship`,
`#bertopic`, and `#topic-models`. The core story ends with a comparison of representations and
the cycling–yoga explorer. Applications follow that conclusion and
exploration, before the sources. The closing comparison links to the separate
word2vec lesson; `#word2vec` remains a valid anchor to that paragraph.

## Narrative and presentation

The main example compares two complete, 30-minute Peloton class transcripts:
Ally Love’s Sundays with Love (confidence), November 15, 2020, and Chelsea
Jackson Roberts’s Morning Yoga Flow, May 13, 2021. Brief, attributed excerpts
introduce the language and demonstrate tokenization and filtering. A separate
step explicitly applies the same processing to the whole classes before their
full counts enter the shared matrix. The previews are not the counted documents.

The opening Gettysburg Address cartoon introduces counting and repetition.
The senators/voters example shows the loss of word order. The next three
sentences—“Plebiscites are unwise,” “Democracy is foolish,” and “Majority rule
rules”—show that disjoint word vectors can miss the first two sentences’ similar
stance. Their pairwise cosine similarities are all zero and cosine distances
are all one. Those primers lead directly into the class comparison.

The class data use the saved, audited corpus matrix: punctuation, symbols,
numbers, URLs and stage directions removed; hyphens split; words lowercased;
recorded stop words and fillers excluded; alphabetic terms of at least three
letters retained when present in at least three classes. No stemming or
lemmatization is applied. The shared vocabulary has 2,948 terms, and the two
classes contribute 1,490 and 2,154 retained occurrences. These are model tokens,
not a claim about total spoken words. Full class counts and corpus document
frequencies are exported from the existing model artifact, not approximated by
retokenizing the quotations in the browser.

Six columns stay fixed through counting and weighting: take, resistance,
confidence, inhale, earth, and tabletop. Their cycling counts are
`11, 42, 28, 2, 0, 0`; their yoga counts are `41, 0, 0, 105, 54, 30`.
The explorer provides 24 selected terms. Both cosine calculations use every
one of the 2,948 coordinates, including those outside the selected displays.

IDF uses all 127 nonempty transcripts, including the two featured classes.
The collection contains 106 cycling classes and three yoga classes; rarity
therefore describes this selected archive. Resistance appears in 110 classes
and confidence in 25: their cycling counts of 42 and 28 become weights of
6.04 and 45.51. In yoga, inhale appears 105 times but across 77 classes, while
earth appears 54 times but across only four classes. Their weights are 52.54
and 186.73. Take appears in every class and receives zero weight. The count
cosine is 0.2318325744; the TF–IDF cosine is 0.0215149277. This decrease is a
result for this pair, not a general property of TF–IDF.

The two class identities and matrix coordinates remain visible throughout.
Reverse scrolling retraces transformations; reduced motion shows completed
states. The title and all text remain selectable. Existing character artwork
is registered in `design/assets.json`.

At the end, “Application: Authorship” returns to the Federalist Papers and shows
why common words can reveal writing habits. Its fuller explanation preserves
the qualification that feature choice depends on the research question.
“Application: BERTopic” then uses a separate synthetic corpus to show how
word weights describe groups formed with embeddings. BERTopic pools texts within
each cluster and ranks their words with c-TF–IDF. The illustration supplies two
clusters of short political texts independently of the class transcripts. Its
word–cluster calculation labels “budget in cluster A” and distinguishes total
occurrence counts from document frequency.

The closing ten-scene sequence explains LDA and STM as probability models.
It begins with categorical probability vectors on a three-category simplex,
then distinguishes the simplex from a Dirichlet distribution over it. A density
mesh shows how symmetric Dirichlet parameters change concentration while
holding the mean at equal shares. A Dirichlet draw is a probability vector;
a categorical draw is one category.

The same three illustrative topics—Movement, Breath, and Reflection—carry
into LDA. The model draws shared topic–word distributions, then a separate
mixture for each document, then a topic and a word for each token. The diagram
makes the document mixture, topic-specific word probabilities, and latent
token assignment distinct objects. A worked “precious” example sums over
possible generating topics and then reverses the calculation to illustrate
probabilistic topic assignment. This conditional calculation assumes known
parameters; the explanation distinguishes it from inference of all unknown
quantities across the corpus.

STM then replaces the Dirichlet document-mixture model with latent Gaussian
log-ratios whose mean depends on metadata. A baseline log-ratio is fixed at
zero, and softmax maps the result back to the simplex. A final scene holds
the document mixture fixed while an optional content covariate changes the
word probabilities within a topic. The existing Peloton fit uses prevalence
covariates only. These examples explain model structure and are not fitted
or causal claims about instructors or class types.

The document example preserves the full requested Baby Cradle quotation from
Chelsea Jackson Roberts’s 10-minute Beginner Focused Flow, May 13, 2021.
It begins “So I love Seated Baby Cradle” and ends “every part of me as being
precious.” The reading version retains the separate Morning Yoga Flow passages
about Earth energy and gratitude, explicitly identifying them as another class.
Featured-quote provenance is source document
`05132021-chelsea_jackson_roberts-1200pm-drastic-1-ab56a7181664411383533b3348f0a2fa_200k`,
02:56.030–03:10.070. The separate Morning Yoga Flow excerpts come from
`05132021-chelsea_jackson_roberts-1030am-drastic-1-83e450847b574eea8e18a85204031751_200k`.
The class label Yoga is transcript-derived, not official Peloton metadata.
Only the requested excerpts are reproduced; the underlying corpus and fitted
models are not required to run the lesson.

Numerical conventions in the topic-model illustration:

- The three topic shares are `(0.5, 0.2, 0.3)`; topic names are explanatory labels,
  whereas fitted topics would be named only after inspecting their words and texts.
- Displayed columns are `cradle`, `inhale`, `precious`, and all remaining words.
  The shared probability rows are `(0.55, 0.10, 0.02, 0.33)`,
  `(0.05, 0.60, 0.01, 0.34)`, and `(0.05, 0.05, 0.20, 0.70)`.
- The document probability of “precious” is `0.5×0.02 + 0.2×0.01 + 0.3×0.20 = 0.072`.
  Its conditional topic probabilities are `(0.010, 0.002, 0.060) / 0.072`;
  Reflection therefore receives approximately `0.833` in this conditional example.
- The fully Bayesian LDA formulation shown uses a Dirichlet prior for each
  topic’s word distribution as well as for document topic mixtures. The word
  prior is denoted lambda to distinguish it from STM’s latent eta vector.
- STM content animation changes only the Reflection word probabilities from
  `(0.05, 0.05, 0.20, 0.70)` to `(0.05, 0.05, 0.35, 0.55)`; document shares stay fixed.
  The change is illustrative and does not estimate a covariate effect.

After the lesson, hovering over any count highlights its term and document and
updates the calculation below. Tapping a cell also selects it. Keyboard users
enter the table with Tab, move between counts with arrow keys, and leave with
Tab; Home/End select the first/last term in a row. The count matrix remains in
place while the calculation changes. Previous/next buttons and left/right arrow
keys provide keyboard navigation through the lesson.

Captions connect each step to the preceding result without adding transition
scenes. The visual transformations carry the numerical details, while the fuller
reading version retains definitions, explanations, and qualifications.

`lesson.mjs`, `peloton-lesson.mjs`, and `topic-model-lesson.mjs` are the sources for both concise scene captions and the fuller reading
version. A stage's `readingText` or `readingNote` preserves detail beyond the
on-screen caption; otherwise its normal text is used. Regenerate the static
reading version after changing lesson text:

```sh
node scripts/build-text-transcript.mjs
node scripts/build-text-transcript.mjs --check
```

The static reading version is a fallback for loading failures and printing.
It starts expanded directly after the title, with working chapter anchors.
Once the animated lesson initializes, `bootstrap.mjs` hides it; the normal page
ends with sources and further reading. If loading or initialization fails, the
expanded text remains in reading order. Print styles reveal that same text.
The animated module imports share one version suffix to avoid mixing cached
dependencies after an update.

Mathematics uses the course's existing self-hosted KaTeX runtime in `docs/word2vec/vendor/katex`.
`math.mjs` shares rendering between the diagrams and static reading version.
No package installation or site build is required; serve `docs/` and open `/text/`.

## Numerical conventions and sources

- In the three-sentence introduction, each count vector has three nonzero
  entries equal to 1, a norm of `sqrt(3)`, and zero dot product with either other
  vector. Cosine similarity is 0; cosine distance, defined as `1 - cosine`, is 1.
- Each complete class transcript is one document. The selected previews are
  used only to explain preprocessing. The separate LDA illustration continues
  to use Chelsea’s Seated Baby Cradle class, not the Morning Yoga Flow.
- The convention is `w_(t,d) = c_(t,d) ln(N/df_t)`, with raw counts, natural
  logarithms, no smoothing and `N = 127`. Cosine adjusts for vector length
  at comparison time. No class-balanced reweighting is applied.
- The matrix shows six columns, and the explorer shows 24. Both comparisons
  use the complete 2,948-term vectors.
- `peloton-data.mjs` contains aggregate counts, document frequencies, brief
  excerpts, exact source identifiers, excerpt times, and source hashes.
  It contains no complete transcripts. Regenerate from the privately held
  analysis with `Rscript scripts/export-peloton-text-data.R --source PATH`; the script
  verifies the selected vectors by reconstructing the original preprocessing
  and comparing it with the saved matrix. It is a maintenance tool, not a
  software requirement for students.
- Mosteller–Wallace's approximate “upon” rates are per 1,000 words in their
  comparison writings. The chart is not an authorship classifier by itself.
- Erwin–Oakes's Greek common-word comparison remains a further-reading citation,
  with no scene or recreated results in the lesson.
- The brief opening ChatGPT excerpt is attributed to Sam Kriss's essay. Its
  highlighted features motivate hypotheses; they do not validate an AI detector.

The lesson links the original research, the Gettysburg Address transcription,
and the supporting reading. Font licenses accompany the local font files.

## Verification

```sh
node --test tests/text-representations.test.mjs tests/speech-motion.test.mjs tests/text-rendering.test.mjs tests/text-loading.test.mjs tests/text-explorer.test.mjs tests/text-applications.test.mjs tests/short-sentences.test.mjs tests/quote-flow.test.mjs tests/text-continuity.test.mjs tests/topic-models.test.mjs tests/dirichlet.test.mjs tests/lda-mechanism.test.mjs
node scripts/build-text-transcript.mjs --check
```

These checks cover numerical values, token accounting, deterministic reversible
motion, diagram states, LaTeX validity, shared vocabulary counts, and explorer
selections. Explorer checks also cover persistent hover selection, stable table
nodes, focus preservation, keyboard movement, and listener cleanup. Application
checks cover pooled counts, c-TF–IDF weights, and reversible progression, including
completed states with reduced motion. Loading checks cover the default reading
route, failed module requests, partial initialization, anchor transfer, and
cleanup after a late initialization failure. They do not replace browser review.

Browser review should cover actual forward and backward scrolling on desktop
and narrow phones, including the transition into each application. Check that
the retained matrices, annotations, captions, and navigation remain separate,
and that keyboard navigation and reduced motion reach every teaching point.
The explorer's keyboard navigation should retain focus and reveal the selected
column without moving the page horizontally. These checks are not a complete
assistive-technology audit.

## Standalone speech cartoon

`cartoon.html` retains the separate animation player with pause, replay, and
keyboard-accessible seeking. It plays once, pauses when hidden, and starts in
a stationary completed state under reduced motion. Its ten Gettysburg Address
tokens preserve duplicates: “the” and “people” each occur three times.
`speech-motion.mjs` computes deterministic word paths and character poses;
`speech-cartoon.mjs` overlays live SVG words on the registered artwork.

The continuity update passes all 42 targeted checks. Browser review covered
1280×820 and 1280×650 desktops, plus 320×568 and 390×844 phones. The main
matrix retained identical coordinates across the bag-of-words/TF–IDF boundary;
all eleven quotation stages kept a fixed visual position on the small phone.
Forward and backward scrolling restored intermediate counts and weights.
Chapter anchors, keyboard controls, reduced motion, application placement,
and the route to sources were checked in the browser without console errors.

The expanded LDA/STM explanation passes all 54 targeted lesson checks. Browser
review covered a 1280×820 desktop and 320×568 and 390×843 phones. All ten stages
fit without horizontal overflow or captions crossing the controls. Actual
scrolling reached the sparse and uniform Dirichlet states and reversed to the
original state. Keyboard controls and all ten reduced-motion endpoints passed.
The probability tests cover normalization, the 0.072 word marginal, the 5/6
conditional Reflection probability, exact scene boundaries, and baseline softmax.
Density shades are rescaled within each plot; their locations, not intensity
across plots, show where the distribution concentrates.

The Peloton replacement was checked at 1280×820 and 1280×650 on desktop,
and 320×568 and 390×844 on phones. All eleven class stages fit without
horizontal overflow or collisions between the visual, caption, and controls.
Forward and backward scrolling restored the class and weighting stages;
keyboard navigation and all six reduced-motion TF–IDF endpoints passed.
Explorer arrow keys retained focus and displayed the selected term's calculation.
The source excerpts use the bundled font's glyph advances to prevent word
collisions. The browser reported no console warnings or errors.
