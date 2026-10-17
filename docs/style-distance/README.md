# Fine-tuning for writing style

A twelve-scene scroll-driven lesson on adapting a text representation for a
specific similarity task. It is distinct from the Peloton classifier: three
passages pass through a shared encoder and a triplet objective compares distances
between their embeddings instead of predicting a topic label.

The four passages concern the same subject but cross two opposing claims with
two writing styles. A is the Ecclesiastes fragment “the race is not to the swift.”
B is a shortened excerpt of George Orwell’s academic rendering in *Politics and
the English Language*: “…success or failure in competitive activities exhibits
no tendency to be commensurate with innate capacity…”. The ellipses identify
omitted text. C preserves the instructor’s counterexample, “the swift shall win
the race.” D is a new academic paraphrase of C: “Competitive success accrues to
the participant exhibiting the greatest velocity.” These are teaching comparisons,
not examples drawn from the StyleDistance dataset. The content labels
`uncertain` and `certain` identify the opposing claims; `biblical` and `academic`
identify style. They are schematic labels for these excerpts, not comprehensive
literary judgments.

The selected triplet is A as anchor, C as the style-positive example, and B as
the content-matched negative. A and C resemble one another in style while contradicting
one another; A and B convey related doubts about ability ensuring success while
changing style. They are not exact semantic equivalents. The initial map
places A–B and C–D close together. The adapted map instead brings A–C and B–D
closer. Passage D is absent from the selected triplet but also moves because the
schematic transformation is shared.

Letters and shape/color encodings remain fixed. All coordinates are explicitly
schematic, with equal pixels per coordinate unit in both viewports. Squared
distances and the margin-0.1 triplet hinge loss are calculated from those same
coordinates. The geometric interpolation illustrates changing a shared encoding
function; it is not an optimizer trace or an empirical model result.

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
loss, shared transformations, and reversible rendering. Tests assert the exact
A/C/B roles and rendered A–C/A–B edges as well as their numerical distances: the
old B/D/A roles would produce the same numbers on this symmetric map, so checking
the numerical loss alone would miss stale passage assignments.

The course URL after merge and Pages publication is
<https://alexanderthclark.github.io/pols4728/style-distance/>.
