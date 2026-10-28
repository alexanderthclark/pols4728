# Text as Data

Bag of Words and TF–IDF

A standalone scrollytelling lesson for POLS 4728: an opening question about
recognizable writing habits, followed by bag of words and TF–IDF.
Its 18 scenes comprise 3 on AI writing, 10 on bag of words, and 5 on TF–IDF.
The linked chapters are `#ai-writing`, `#bag-of-words`, and `#tf-idf`. A short
closing comparison links to the separate word2vec lesson; `#word2vec` remains
a valid anchor to that paragraph.

## Narrative and presentation

The same four constructed political documents connect tokenization, the
shared vocabulary, count vectors, and TF–IDF. Scrolling sends individual speech
words into a bag, moves tokens into a count matrix, and reweights its columns.
Reverse scrolling retraces these changes. Reduced motion uses completed states.
The title and all text remain selectable; character artwork is registered in
`design/assets.json`.

The preprocessing sequence distinguishes tokenization from lowercasing and
punctuation handling, then illustrates optional lemmatization and stop-word
removal. D1 separates into eight tokens; the lemmatization examples group
inflected forms; an illustrative removal of “the” leaves five tokens. The
worked document matrix remains unchanged: it retains common words and does not
lemmatize. The Federalist example explains why common words can reveal writing
habits, then the transition to TF–IDF asks which terms distinguish document topics.
The John/Revelation scene has been removed; its citation remains in further reading.

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

The complete reading version starts expanded directly after the title, with
working chapter anchors. Only after the animated lesson initializes does
`bootstrap.mjs` move that same reading version to the end and collapse its chapters.
If a module request or initialization fails, the expanded lesson remains in
reading order. The animated module imports share one version suffix to avoid
mixing cached dependencies after an update.

Mathematics uses the course's existing self-hosted KaTeX runtime in `docs/word2vec/vendor/katex`.
`math.mjs` shares rendering between the diagrams and static reading version.
No package installation or site build is required; serve `docs/` and open `/text/`.

## Numerical conventions and sources

- Four synthetic documents; lowercase alphabetic tokens, with punctuation
  removed. Common words and repeated tokens are retained; no stemming or
  lemmatization. The optional preprocessing illustrations do not alter these counts.
- Raw-count TF × ln(N/df), without smoothing or vector normalization. Cosine
  similarity adjusts vector length at comparison time.
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
node --test tests/text-representations.test.mjs tests/speech-motion.test.mjs tests/text-rendering.test.mjs tests/text-loading.test.mjs
node scripts/build-text-transcript.mjs --check
```

The 22 tests check numerical values, token accounting, deterministic reversible
motion, 108 diagram states, LaTeX validity, shared vocabulary counts, and all 28
explorer selections. Loading checks cover the default reading route, failed
module requests, partial initialization, anchor transfer, and cleanup after a
late initialization failure. They do not replace browser review.

Browser review covered the retained scenes at 1280×820, 768×1024, 390×844, 375×667,
and 320×568. The new preprocessing sequence and revised transition from the
Federalist example to TF–IDF were checked at the same five sizes. Layout checks
found no horizontal overflow or text/navigation collisions. Manual checks
covered actual forward/backward scrolling, pinned
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
