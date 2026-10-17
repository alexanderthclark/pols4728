# Fine-tuning for writing style

A twelve-scene scroll-driven lesson on adapting a text representation for a
specific similarity task. It is distinct from the Peloton classifier: three
passages pass through a shared encoder and a triplet objective compares distances
between their embeddings instead of predicting a topic label.

The four neutral passages vary budget/bridge content and plain/bureaucratic
style. Their letters and shape/color encodings remain fixed. All coordinates are
explicitly schematic, with equal pixels per coordinate unit in both viewports.
Squared distances and the margin-0.1 triplet hinge loss are calculated from those
same coordinates. The geometric interpolation illustrates changing a shared
encoding function; it is not an optimizer trace or an empirical model result.

Sources include the NAACL 2025 StyleDistance paper and checkpoint card. The lesson
correctly identifies pretrained RoBERTa with trained LoRA updates and distinguishes
the published checkpoint's synthetic/natural mixture from a synthetic-only
variant. The instructor's linked Colab has not been inspected, so no particular
training setup or outputs are attributed to it.

`lesson.mjs` supplies scenes, SVG rendering, and sources; `model.mjs` supplies the
schematic geometry and loss. The shared shell directly maps scroll position to
visual state and supports backward scrolling, arrow keys, native buttons, reduced
motion, and a complete static reading route. Diagrams and captions adapt to narrow
and short screens. No network model calls or training occur in the browser.

Regenerate both reading pages with `node scripts/build-fine-tuning-pages.mjs`.
The existing repository test command includes independent checks of geometry,
loss, shared transformations, and reversible rendering.

The course URL after merge and Pages publication is
<https://alexanderthclark.github.io/pols4728/style-distance/>.
