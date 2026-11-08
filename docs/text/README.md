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
the Schumpeter–Whitman explorer. Applications follow that conclusion and
exploration, before the sources. The closing comparison links to the separate
word2vec lesson; `#word2vec` remains a valid anchor to that paragraph.

## Narrative and presentation

The worked corpus contains the same two quotations used in the course slides:
Joseph Schumpeter on citizens entering the political field, from *Capitalism,
Socialism, and Democracy*, and Walt Whitman on democracy as a training school,
from *Democratic Vistas*. The wording is preserved in full. Scrolling follows
the quotations through tokenization, lemmatization, stop-word removal, a shared
vocabulary, count vectors, and TF–IDF. Reverse scrolling retraces these changes;
reduced motion uses completed states. The title and all text remain selectable,
and character artwork is registered in `design/assets.json`.

The opening Lincoln-like speaker and Gettysburg Address cartoon introduce the
bag metaphor. “Keep repetition” displays one speech row with six word columns;
its counts update as words arrive and finish at 1, 1, 3, 3, 1, 1.
The senators/voters example shows what counts lose: word order. The following
three-sentence comparison reverses the problem by showing how different words
can express similar ideas. These short primers lead directly to Schumpeter and
Whitman, the main teaching example. Both authors remain visible through
preprocessing, the count matrix, and all six TF–IDF scenes. No application or
separate feature-choice scene interrupts this walkthrough.

The short-sentence comparison uses three statements: “Plebiscites are unwise.”
“Democracy is foolish.” “Majority rule rules.” It moves from the sentences to their
nine-column count matrix, which stays visible during cosine comparison. The first two
share a negative stance toward democratic decision-making, but word overlap
cannot capture it: every pair has cosine similarity 0 and cosine distance 1.
This example lowercases and removes punctuation only; it retains stop words
and keeps “are”/“is” and “rule”/“rules” distinct. The later TF–IDF comparison
returns to the same cosine measure with weighted vectors.

Schumpeter's quotation splits
into 21 alphabetic tokens, with lowercasing and punctuation removal. Whitman's
has 24. Lemmatization maps forms such as “drops” to “drop,” “enters” to “enter,”
and “men” to “man.” The selected stop-word removal excludes 10 Schumpeter tokens
and 9 Whitman tokens, leaving these exact processed documents:

- Schumpeter: `typical citizen drop down lower level mental performance enter political field`
- Whitman: `political democracy exist practically work america threaten evil supply training school make grand young man`

These 11 and 15 retained occurrences form a 25-term vocabulary. Every retained
term occurs once within its document; “political” is the sole shared term. The
main matrix is an explicitly labeled six-column excerpt: `political`, `citizen`,
`mental`, `democracy`, `america`, and `young`. The complete explorer contains all
25 columns and all 50 word–document combinations.

The quotations become tokens, retained words become counts, and the same
matrix carries the walkthrough into TF–IDF. Its vocabulary stays fixed while
the annotations explain the calculation and its counts become weights.
The walkthrough follows shared “political” and then
Schumpeter-specific “citizen.” An explicit word–document formula compares
“citizen” across the two authors: its IDF is shared, but its scores are 0.693 and
0. “Political” receives zero in both documents. Their complete TF–IDF vectors
therefore have cosine similarity zero; the lesson makes clear that this does
not imply unrelated meanings. TF–IDF changes word importance; it does not
learn similarities between different vocabulary terms. This limitation connects
the ending to the separate lesson on embeddings.

At the end, “Application: Authorship” returns to the Federalist Papers and shows
why common words can reveal writing habits. Its fuller explanation preserves
the qualification that feature choice depends on the research question.
“Application: BERTopic” then uses a separate synthetic corpus to show how
word weights describe groups formed with embeddings. BERTopic pools texts within
each cluster and ranks their words with c-TF–IDF. The illustration supplies two
clusters of short political texts independently of the quotation corpus. Its
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

`lesson.mjs` is the source for both concise scene captions and the fuller reading
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
- Each quotation is one document. The processed corpus uses lowercasing,
  alphabetic tokenization, the stated lemma mappings, and the selected stop-word
  removal. Its 26 retained occurrences occupy 25 vocabulary coordinates.
- The slide convention is `w_(t,d) = c_(t,d) ln(N/df_t)`, where `c_(t,d)` is a raw
  count, `df_t` counts documents containing the term, and `N = 2`. The logarithm
  is natural. Weights are shown without smoothing or vector normalization.
  Cosine similarity adjusts vector length at comparison time.
- “Political” has document frequency 2 and IDF 0. Every other retained term has
  document frequency 1 and IDF `ln(2)`, approximately 0.693. A distinctive term
  has that weight in the quotation containing it and zero in the other.
- The six displayed columns are an excerpt. Calculations and the explorer use
  the complete 25-term vocabulary.
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
