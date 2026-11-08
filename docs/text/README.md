# Text as Data

Bag of Words and TF–IDF

A standalone scrollytelling lesson for POLS 4728: an opening question about
recognizable writing habits, followed by bag of words and TF–IDF.
Its 21 scenes comprise 3 on AI writing, 10 on bag of words, and 8 on TF–IDF.
The linked chapters are `#ai-writing`, `#bag-of-words`, and `#tf-idf`. A short
closing comparison links to the separate word2vec lesson; `#word2vec` remains
a valid anchor to that paragraph.

## Narrative and presentation

The worked corpus contains the same two quotations used in the course slides:
Joseph Schumpeter on citizens entering the political field, from *Capitalism,
Socialism, and Democracy*, and Walt Whitman on democracy as a training school,
from *Democratic Vistas*. The wording is preserved in full. Scrolling follows
the quotations through tokenization, lemmatization, stop-word removal, a shared
vocabulary, count vectors, and TF–IDF. Reverse scrolling retraces these changes;
reduced motion uses completed states. The title and all text remain selectable,
and character artwork is registered in `design/assets.json`.

The opening Lincoln-like speaker and Gettysburg Address cartoon remain a
separate introduction to the bag metaphor. Schumpeter's quotation then splits
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

The Federalist example explains why common words can reveal writing habits.
The transition back to Schumpeter and Whitman then asks which retained words
distinguish their quotations. TF–IDF keeps the processed vocabulary fixed while
changing its weights. The walkthrough follows shared “political” and then
Schumpeter-specific “citizen.” An explicit word–document formula compares
“citizen” across the two authors: its IDF is shared, but its scores are 0.693 and
0. “Political” receives zero in both documents. Their complete TF–IDF vectors
therefore have cosine similarity zero; the lesson makes clear that this does
not imply unrelated meanings.

Two closing application scenes use a separate synthetic corpus to show how
word weights describe groups formed with embeddings. BERTopic pools texts within
each cluster and ranks their words with c-TF–IDF. The illustration supplies two
clusters of short political texts independently of the quotation corpus. Its
word–cluster calculation labels “budget in cluster A” and distinguishes total
occurrence counts from document frequency.

After the lesson, hovering over any count highlights its term and document and
updates the calculation below. Tapping a cell also selects it. Keyboard users
enter the table with Tab, move between counts with arrow keys, and leave with
Tab; Home/End select the first/last term in a row. The count matrix remains in
place while the calculation changes. Previous/next buttons and left/right arrow
keys provide keyboard navigation through the lesson.

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
node --test tests/text-representations.test.mjs tests/speech-motion.test.mjs tests/text-rendering.test.mjs tests/text-loading.test.mjs tests/text-explorer.test.mjs tests/text-applications.test.mjs
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

The quotation update passed all 30 targeted checks. Browser review covered
1265×712 desktop and 390×844 / 320×568 phone viewports. Actual forward and
backward scrolling preserved the pinned scenes, and the 320×568 walkthrough
kept the revised quotation, preprocessing, matrix, and all eight TF–IDF scenes
clear of navigation without page-level horizontal overflow. Reduced-motion
weighting showed the completed matrix. The full explorer’s keyboard Home/End
navigation revealed the selected column within its horizontal scroller while
retaining page position. Static reading content and loading fallback behavior
were checked by the transcript and loading tests. This is not a complete
assistive-technology audit.

## Standalone speech cartoon

`cartoon.html` retains the separate animation player with pause, replay, and
keyboard-accessible seeking. It plays once, pauses when hidden, and starts in
a stationary completed state under reduced motion. Its ten Gettysburg Address
tokens preserve duplicates: “the” and “people” each occur three times.
`speech-motion.mjs` computes deterministic word paths and character poses;
`speech-cartoon.mjs` overlays live SVG words on the registered artwork.
