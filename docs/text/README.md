# From words to vectors

A standalone scrollytelling lesson for POLS 4728: an opening question about
recognizable writing habits, followed by bag of words, TF–IDF, and word2vec.
The linked chapters are `#ai-writing`, `#bag-of-words`, `#tf-idf`, and `#word2vec`.

## Narrative and presentation

The same four constructed political documents connect tokenization, the
shared vocabulary, count vectors, and TF–IDF. Scrolling sends individual speech
words into a bag, moves tokens into a count matrix, reweights its columns,
averages CBOW context vectors, and applies one calculated gradient step.
Reverse scrolling retraces these changes. Reduced motion uses completed states.
The title and all text remain selectable; character artwork is registered in
`design/assets.json`.

The core TF–IDF walkthrough follows “the” and then “budget” in D1. Its text and
highlighted cells always agree. Optional term/document controls appear after
the lesson, where all 28 combinations update the same count matrix and formula.
Previous/next buttons and left/right arrow keys provide keyboard navigation.
The prose and diagrams fit desktop, tablet, and narrow phone layouts.

`lesson.mjs` is the source for both concise scene captions and the fuller reading
version. A stage's `readingText` or `readingNote` preserves detail beyond the
on-screen caption; otherwise its normal text is used. Regenerate the static
reading version after changing lesson text:

```sh
node scripts/build-text-transcript.mjs
node scripts/build-text-transcript.mjs --check
```

The reading version remains available without JavaScript. Mathematics uses the
course's existing self-hosted KaTeX runtime in `docs/word2vec/vendor/katex`.
`math.mjs` shares rendering between the diagrams and static reading version.
No package installation or site build is required; serve `docs/` and open `/text/`.

## Numerical conventions and sources

- Four synthetic documents; lowercase alphabetic tokens, with punctuation
  removed. Common words and repeated tokens are retained; no stemming.
- Raw-count TF × ln(N/df), without smoothing or vector normalization. Cosine
  similarity adjusts vector length at comparison time.
- A six-word, two-coordinate CBOW example with distinct input and output tables,
  a symmetric radius-one context, and full softmax. Parameters are illustrative,
  not corpus-trained. Both tables update simultaneously; intermediate displayed
  probabilities are recomputed from the fractional parameter step.
- Analogy diagrams use constructed coordinates. The same displacement translates
  between man/king and woman/queen, then between country/capital pairs. These
  pictures illustrate the geometry, not an empirical fit.
- Mosteller–Wallace's approximate “upon” rates are per 1,000 words in their
  comparison writings. The chart is not an authorship classifier by itself.
- Erwin–Oakes's Greek common-word comparison uses 500-word samples and 75 common
  words. Distinct profiles do not isolate authorship from genre. The diagram
  shows their study design, not a replication.
- The brief opening ChatGPT excerpt is attributed to Sam Kriss's essay. Its
  highlighted features motivate hypotheses; they do not validate an AI detector.

The lesson links the original research, the Gettysburg Address transcription,
and the supporting reading. Font licenses accompany the local font files.

## Verification

```sh
node --test tests/text-representations.test.mjs tests/speech-motion.test.mjs tests/text-analogy.test.mjs tests/text-rendering.test.mjs
node scripts/build-text-transcript.mjs --check
```

The 24 tests check numerical values, every CBOW gradient against finite
differences, token accounting, deterministic reversible motion, 156 diagram
states, LaTeX validity, shared vocabulary counts, all 28 explorer selections,
and analogy translation. They do not replace browser review.

Browser review covered all 26 scenes at 1280×820, 768×1024, 390×844, 375×667,
and 320×568. Layout checks found no horizontal overflow or text/navigation
collisions. Manual checks covered actual forward/backward scrolling, pinned
intermediate states, keyboard navigation, completed reduced-motion states,
term/document selection, the reading version, and title composition. This is
not a claim of a complete assistive-technology audit.

## Standalone speech cartoon

`cartoon.html` retains the separate animation player with pause, replay, and
keyboard-accessible seeking. It plays once, pauses when hidden, and starts in
a stationary completed state under reduced motion. Its ten Gettysburg Address
tokens preserve duplicates: “the” and “people” each occur three times.
`speech-motion.mjs` computes deterministic word paths and character poses;
`speech-cartoon.mjs` overlays live SVG words on the registered artwork.
