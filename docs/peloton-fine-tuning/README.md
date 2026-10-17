# Fine-tuning an encoder

A fourteen-scene, scroll-driven lesson about classifying Peloton instructor speech.
The opening paraphrases the instructor's Tara Isabella Burton motivation and
cites *Strange Rites* and Burton's Peloton essay; it is not presented as a direct
book quotation. The exact wording can be substituted after it is supplied.

The workflow follows the supplied `fine_tuning_distilbert.py`: earlier Gemini
Flash labels, topic-stratified 80/20 train/validation splitting, a DistilBERT
sequence classifier with trainable encoder and head, four epochs, and checkpoint
selection by validation loss. The three final probe sentences preserve the
script's wording. No original CSV, model weights, labels, or run results are
published in this lesson.

Coaching, Music, and Encouragement are illustrative categories. A tiny explicit
differentiable surrogate makes logits, softmax, cross-entropy, and a full gradient
update numerically inspectable; those values are not DistilBERT results. Its
gradients are checked independently by finite differences. The thirty sentence
markers and embedding map are schematic and retain stable identities on scroll.

`lesson.mjs` contains captions, full reading text, sources, and pure SVG rendering.
`model.mjs` contains the numerical example. The shared shell and stylesheet live
in `../fine-tuning-shared/`. Scroll position controls intermediate states directly;
there is no autonomous playback. Arrow keys, scene navigation, and a Steps button
provide completed states; system reduced motion uses that same route. Compact
diagrams and concise captions support narrow and short viewports.

Regenerate both static reading pages after prose edits with
`node scripts/build-fine-tuning-pages.mjs` from the repository root. Serve `docs/`
with an ordinary local static server to review. Tests run with the repository's
existing `node --test tests/*.test.mjs` command.

The course URL after merge and Pages publication is
<https://alexanderthclark.github.io/pols4728/peloton-fine-tuning/>.
