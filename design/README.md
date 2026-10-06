> This guide is a snapshot of the Fall 2026 course design language, copied on
> October 6, 2026. The public repository uses it for new course pages, figures,
> and other teaching materials. The [visual reference](guide.pdf) and
> [design tokens](tokens.json) accompany the guide.
>
> The preserved semester guide below describes its original course repository.
> Its print dimensions, LaTeX helpers, build commands, and migration status apply
> to that source checkout; those tools and artwork are not all included here.
> Its manual document-build policy does not replace this public repository's
> GitHub Pages workflow. Public artwork is recorded in [assets.json](assets.json).

## Applying the guide to web pages

Carry the course's white backgrounds, restrained hierarchy, readable labels,
approved identity artwork, and meaningful palette roles into each new page.
Keep quantitative geometry and scales precise. Use words, shapes, line patterns,
or direct labels alongside color. Preserve connected reasoning and qualifications
as a scrolly story reveals its components.

The type sizes and Latin Modern specifications below describe printed documents.
For the web, choose readable responsive text and mathematical typography that
remain clear on narrow screens. Keep essential headings, labels, equations, and
code as real text. Give interactive controls descriptive labels and keyboard
access; support reduced motion and provide a readable route through the argument.

Use `tokens.json` for palette roles. Its dimensions use print units; adapt those
dimensions to the viewport rather than applying them directly as pixel values.
Inspect desktop and mobile layouts, contrast, and grayscale readability. Do not
claim accessibility compliance from visual inspection alone. Record purposeful
exceptions when a figure's domain or a teaching interaction requires them.

The existing Shapley story remains as published. This guide records the reference
for future work; importing it does not restyle that story. Refresh the guide, PDF,
and tokens together when the semester design language changes, copying the
requested files without importing private Git history or unrelated course files.

---

# POLS 4728 design language

The course has a playful visual identity and a serious scholarly text. Character
artwork welcomes readers on covers and at major part openings. Occasional
explanatory illustrations within chapters can clarify a statistical idea.
The notes develop concepts through explanation, mathematical reasoning, evidence,
and code.
Students are expected to read attentively and follow an argument across paragraphs.

The agreed plot direction is Open textbook, with Guided lecture annotations
when they make the learning point clearer.

## Editorial character

Write complete explanations that establish assumptions, develop the reasoning,
and connect claims to evidence. Preserve qualifications, derivations, and the
work required to understand them. Examples should advance the argument and sit
within the exposition. Playfulness in the artwork does not set the prose's tone.

Do not introduce whimsical named environments, decorative teaching boxes,
quirky pull quotes, motivational asides, or story sidebars. Definitions,
questions, examples, and reading references belong in the ordinary text with
conventional headings where needed. Draft-status notices identify unfinished
material; they are not a model for teaching components in the finished notes.

Do not use bold emphasis in body prose or bold lead-ins to create a second,
skimmable version of the argument. Conventional headings, mathematical notation,
and code syntax retain their functional typography. Bullets, numbered steps, and
reference lists are welcome when they make assumptions, vocabulary, examples,
alternatives, or procedures easier to follow. Keep useful existing lists and
the instructor's voice; do not turn them into paragraphs simply to make the
notes more prose-heavy. Use compact spacing and enough explanation within and
around each list to preserve the reasoning.

Use `\emph{}` selectively for new vocabulary at its introduction or definition
(for example, `\emph{training data}`) and for wording that needs particular
stress. Keep later mentions and routine list lead-ins in regular type; emphasis
should identify the term or precise phrase, not entire explanatory sentences.

Avoid LinkedIn-style writing: repeated one-sentence paragraphs, forced line
breaks for emphasis, punchy lead-ins, and an insistence on bullets for every
idea. A continuous argument belongs in connected paragraphs; a set of distinct
items can remain a list. Neither format should simplify away the substance.

Accessibility supports close reading through legible type, sound navigation,
readable figures, and usable source code. It must preserve the text's intellectual
substance. Clear writing explains difficult ideas without removing the difficulty
that belongs to the subject.

## Identity and artwork

The initial hierarchy preserves the repository's existing uses:

| Role | Existing artwork | Preferred use |
| --- | --- | --- |
| Primary course identity | `assets/ML_and_AI_logo.png` | Syllabus, course introduction, occasional section opener. Playful outlined letter characters. |
| Playful variant | `assets/ml_ai_yoyo.png` | Notes cover, using the complete original logo and its embedded subtitle as the visible course title. The yo-yo is a small visual joke, not a repeated bullet icon. |
| Bootcamp part cover | `assets/bootcamp-curtain-unlettered.png` | M and L at the front curtains, A and I marching mid-stage, and a blank sign at the rear. Typeset the title inside the sign. |
| Foundations part cover | `assets/foundations-arch-unlettered.png` | M and L support A and I, whose joined hands form an arch. Typeset the title within the open space between M and L. |
| The Canon part cover | `assets/the-canon-workshop-unlettered.png` | M, L, and A build a wooden frame while I jumps into a hammer swing. Typeset the title in the hanging sign, capitalizing both The and Canon. |
| Legacy identity | `assets/ML_logo_ss.pdf` | Retain for archival material; its wording omits AI. Do not select it for new course covers. |
| Special illustrated artwork | `assets/ml_ai_codex.png`, `assets/ml_ai_codex_white.png` | Optional digital illustration at a meaningful moment. Avoid in ordinary printed notes: even the white-background version contains a large, ink-heavy illustrated book. |

The registry in `assets.json` gives exact filenames, hashes, roles, and descriptions.
Keep image-generation prompts outside version control; the repository stores
artwork and its usage metadata.
The original lettered scenes remain in `assets/` as source artwork. Use the
unlettered derivatives for typeset part openings.
These part illustrations complement the primary course logo.
The user's additional cave-painting direction is welcome, but no distinct asset
with that identity was located in this repository. Register its actual source
before using or deriving a variant; do not relabel the illuminated-book artwork
as cave painting. The playful outlined family is the available reference for the
user's "fig-child" direction, pending any more specific source.

### Artwork rules

- Place a single identity mark on a white field. Keep the original proportions.
- Preserve clear space of at least one capital-letter height around the visible
  artwork. Existing transparent/white padding contributes to that clear space.
- Prefer about 80-145 mm wide for the full lockup in print. Below that, use a
  real-text course name rather than shrinking the embedded subtitle to illegibility.
- Never use a logo as a watermark behind teaching content, fill a page with a
  texture, stretch it, recolor it ad hoc, or combine multiple logo families on
  the same cover.
- On the notes cover, use the complete original `\CourseLogo{playful}`, placed
  high on the page, with no separate typeset course title. Preserve its embedded
  subtitle. Put a small Course Notes label below the logo and retain the author,
  course number, and semester as real text near the foot of the page.
  The full course title remains in the PDF metadata; this does not provide tagged
  alternative text for the logo. Part-opening titles remain selectable text.
- `\CourseLogo{primary}` and `\CourseLogo{playful}` select the registered assets.
  The course palette belongs to document structure and plots; logos retain their
  existing monochrome artwork.
- Cave-painting or new character illustrations should favor sparse contours,
  a white/transparent field, and a clear silhouette at printed size. Avoid faux
  parchment, muddy textures, tiny faces, and essential information inside an image.

### Part-opening illustrations

Use the registered Bootcamp, Foundations, and The Canon illustrations to mark the
opening of their corresponding major parts. Reserve one illustration for each part opener;
use illustrations within chapters only when they serve the explanation, following
the guidance below.

- Compose one real-text title within the unlettered scene. Set it in the notes'
  regular-weight Latin Modern Roman, approximately 29 pt at notes size. Keep the
  title selectable; do not bake it into the image or repeat it above the scene.
  Foundations uses a smaller second line, "of Machine Learning."
- Place a discreet small-cap part number above the composition. Balance the
  complete group slightly above the page's vertical center with more space below
  than above. Keep the folio quiet and omit running headers on the opener.
- Use about 90% of the text width, preserving the full scene and its proportions.
  Position titles optically within their signs or arch, using the shared placement
  data. Judge character scale and clear space across all three pages together.
  Do not crop characters or stretch the scene to fill the page.
- Use clean black line art on a white or transparent field so the page has no
  tinted rectangular image background. Keep draft status in the introductory
  prose, outside the display title.
- Let the part illustration carry the character artwork on that page. Use the
  primary and yo-yo logos on course-level covers and introductions.
- Keep the shared family recognizable: the same M, L, A, and I characters, sparse
  black outlines, a white background, and a simple action related to the part.
  New scenes need review and an entry in `assets.json` with a description,
  hash, and intended use.
- In a format that supports alternative text, describe the scene briefly if it
  adds meaning (for example, "M and L support A and I to form an arch"). If the
  illustration is purely decorative beside a complete heading, mark it as
  decorative. The registry description alone does not add PDF alternative text.

The shared compositions are shown in `design/guide.pdf`.
Use `\CoursePartCover{bootcamp}`, `\CoursePartCover{foundations}`, or
`\CoursePartCover{canon}` for a complete opening page. `tex/course-parts.sty`
holds each asset, navigation title, typesetting, and optical coordinates together.
The helper advances the normal part counter and creates the contents entry and
PDF bookmark without a duplicate visible heading. Both notes editions use it.
For specimens, `\CoursePartScene[width]{part-key}` reuses the same composition
without changing the part counter or adding navigation entries.

The part titled `The Canon` begins with the complete Regularization chapter,
including its shrinkage and OLS/collinearity warmup before ridge and lasso.
Foundations ends with validation. The full notes and standalone article use the
same chapter wrapper, keeping the warmup and methods together across both editions.

### Explanatory illustrations within chapters

Freedman, Pisani, and Purves's *Statistics* (4th ed., W. W. Norton, 2007) is a
reference for the relationship between exposition and illustration: direct
prose, concrete examples, economical diagrams, and occasional visual wit. Use
sparse black line drawings on white backgrounds, with simple compositions and
little shading. Within chapters, an illustration should clarify a mechanism,
reveal a misconception, or make a comparison memorable. Humor should arise from
the statistical idea.

Hand-drawn character is welcome in conceptual illustrations; quantitative
figures must retain precise geometry, readable labels, and faithful scales.
Preserve the course's established typography, palette, and characters, and
develop original illustrations suited to its subject matter.

Place each illustration near the passage it explains. Its caption and the
surrounding prose should state the learning point; essential labels and
mathematical notation remain real text. Use illustrations selectively when
they advance the argument. Keep the existing rules for scholarly prose,
ordinary headings, and restrained formatting.

Use a clear silhouette and legible details at the final printed size. Register
new character artwork in `assets.json` with a description, hash, and intended
use, and keep image-generation prompts outside version control. Review the
result in grayscale and describe meaningful visual content in alternative text
where the format supports it. Conceptual drawings must not imply numerical
relationships that the explanation does not establish.

## Typography and page rhythm

- Printed notes: 11 pt Latin Modern Roman with matching Latin Modern mathematics.
  This retains the Computer Modern character selected in the style study.
- Keep the established one-inch margins and generous line spacing. Use sentence
  case, descriptive headings, short captions, and regular-weight body prose.
- Plot labels and ticks: 10 pt at their final printed size; titles: 11 pt. A
  half-width plot uses the same text sizes, fewer ticks, and less annotation.
- Code uses selectable monospaced text. Keep examples short enough to fit without
  shrinking the type, and check copying separately from appearance; see
  Code readability and access below.
- Slide convention: the same palette and Latin Modern family, with larger text
  appropriate to a projector. Existing slides remain a separate migration item;
  do not apply print-size plot text to a full lecture-room slide.
- Use one caption to state what the reader should notice. Avoid repeating the
  same title in the figure, caption, and surrounding paragraph.

## Color and monochrome

`tokens.json` is the source of truth. Generated TeX and Matplotlib adapters must
not be edited by hand.

| Token | Value | Meaning |
| --- | --- | --- |
| Ink | `#242424` | Body text and essential labels |
| Blue | `#234E70` | Primary model, main series, links |
| Rust | `#A54F32` | Comparison, intervention, teaching emphasis |
| Teal | `#007F73` | Additional plotted categories |
| Purple | `#7B4F9D` | Additional plotted categories |
| Ochre | `#9B6A00` | Additional plotted categories |
| Sky | `#26799D` | Additional plotted categories |
| Rose | `#B64E7A` | Additional plotted categories |
| Muted | `#606060` | Secondary text and reference series |
| Rule | `#767676` | Thin axes and essential rules |
| Paper | `#FFFFFF` | Every default page and panel background |

Blue and rust are not automatic good/bad labels. Keep each model or concept's
identity stable across related figures. Blue and rust remain the document
accents, while plots have an eight-color categorical cycle: blue, rust, teal,
purple, ochre, sky, rose, and gray. Matplotlib and PGFPlots receive the same
cycle from the tokens. It repeats only after eight series. Use the named
`COLORS` roles when matching points to curves or preserving a concept's color
across panels; ordinary `ax.plot` calls use the cycle automatically.

The default solid, dashed, dotted, and dash-dot patterns repeat after four
series. Python's `apply_style(monochrome=True)` uses those four patterns in ink;
TeX print mode maps every categorical color to ink. Neither mode makes eight
series distinguishable by line pattern alone. Use direct labels or distinct
markers when patterns repeat, and inspect the result in grayscale. Explicitly
colored Python plots retain their chosen colors and need their own non-color
encodings. No categorical palette guarantees that every pair is distinguishable
for every reader.

Color may connect points with their curves, as in the precision–recall figure.
Keep that relationship and use direct curve labels for grayscale readability;
do not suppress useful colors merely to enforce a two-accent appearance. Use
small multiples when the actual figure becomes crowded, rather than imposing a
three-series limit. Shading may be a light secondary cue; its boundary and
caption must carry the information without the fill.

The automated contrast targets are 4.5:1 for text against white and 3:1 for
essential non-text marks. These are design checks informed by
[W3C contrast guidance](https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum),
not a claim that a whole PDF meets WCAG or PDF/UA.

## Exposition, examples, and annotations

Develop explanations in ordinary paragraphs, with displayed mathematics or code
where the argument requires them. A short reference list can distinguish terms,
assumptions, or examples; numbered steps can express a procedure. Questions and
further reading belong in the ordinary text. Avoid decorative containers and
catchphrase labels. Use descriptive headings to identify the subject.

Figure annotations should identify a concrete feature, such as "Upper tail" or
"Held-out error", and remain readable in monochrome. Captions explain the
comparison and its relevance to the surrounding argument. Links need descriptive
wording that also makes sense in print.

## Code readability and access

Use the existing LaTeX `listings` package for printed code, with the course's
shared syntax colors on white. Keep code as text rather than an image. Explain
what each example does in nearby prose, including its inputs and expected output;
syntax highlighting must not be the only way to understand it.

- Target at least 10 pt monospaced code at its final printed size (`\small` in
  the 11 pt notes). Shorten or split long examples at meaningful steps rather
  than switching to `\scriptsize` or scaling down the block.
- Preserve Python indentation and spaces. Prefer short source lines and explicit
  Python continuations inside parentheses to visual line wrapping; a printed wrap
  must not suggest a new statement or change the apparent nesting.
- Use bold keywords and italic comments alongside color. Check the listing in
  grayscale and keep comments readable. Describe results and warnings in words.
- Use line numbers when the explanation refers to particular lines; otherwise
  omit them. Keep numbers and surrounding prose out of copied code. Separate
  runnable code from console output and interpreter prompts.
- Copy a representative listing from the built PDF into a plain-text editor.
  Check indentation, line breaks, quotes, underscores, operators, and any line
  numbers that were captured. Compare it with the source and run a self-contained
  example where practical. Selectable PDF text alone does not guarantee accurate
  copying or a usable screen-reader reading order.

The notes' shared listing style uses 10 pt monospaced code, preserved spaces,
straight quotes, and no line numbers by default. The three Python programming
examples and two validation examples are included from maintained files in
`notes/python/`, with linked filename labels at the upper right. Source files and
representative PDF extractions are checked for equivalent Python syntax. The ATUS
example requires the external data described in `notes/python/README.md`. Full
PDF accessibility remains a separate build task.

### Downloadable Python sources

Follow the established Matplotlib for Storytellers interaction: a clickable
monospaced `.py` filename at the upper right of its listing opens that file's
GitHub source page. The reference implementation is `tex/mycommands.sty` in that
repository: `\pyfile{filename.py}` prints the local file with `\lstinputlisting`,
and `\gitlink` supplies the filename link. Retain this filename-based interaction
rather than introducing a separate download button or a raw-file link as the
primary action.

The five printed examples are published in the public
[`pols4728/examples/2026f/`](https://github.com/alexanderthclark/pols4728/tree/1181088cf09da70b3a8c08ad519079fe5b0fc19c/examples/2026f)
directory. The semester repository and its history stay private. The originals in
`notes/python/` remain the source of truth; publishing copies is a manual step.

1. Give substantial printed examples a maintained `.py` source with the imports,
   data instructions, and context needed to run it. The course's shared `\pyfile`
   helper prints that file and generates its filename link together.
2. Place the filename above the listing, aligned to its right edge within the
   text column. Keep it monospaced, at least 10 pt, and visible in print; avoid
   the reference's reduced-scale margin placement. The Python chapter explains
   that filenames open the corresponding source on GitHub.
3. Centralize the public repository URL, directory, and revision in
   `tex/course-macros.sty`. Link to the GitHub file view at a commit matching the
   PDF instead of a moving branch. When examples change, copy the revised files
   to the public repository, publish them, and update `\CourseCodeRevision` before
   rebuilding. Never merge private history into the public repository.
4. Verify anonymous access, downloading, and copying into an editor; compare the
   public files byte-for-byte with the local originals. Run self-contained examples
   and document any external data requirements. Remove machine-specific paths
   before publication and recheck links when releasing updated notes.

These source files provide a reliable copy/paste route independent of PDF
text extraction.

## Plot authoring

Python sources call `apply_style()` from `course_plot` before making figures. The
notebooks discover the repository from their working directory, so they work
from the root or `notes/notebooks` without a machine-specific path. Download the
repository together; a detached notebook does not contain its shared theme.

Use `new_figure(width="full")` or `new_figure(width="half")` for publication plots.
`save_figure(fig, path, description="...")` exports PDF/PGF through pdflatex with
the same font preamble as the notes and writes an adjacent description file.
Notebook previews use Matplotlib's bundled Computer Modern fonts without a TeX
installation. Preview and publication typography are closely related but not
identical; inspect the final PDF.

Native PGFPlots use `course axis` and the `course primary`, `course comparison`,
and `course reference` styles. Keep scales, tick positions, interpolation,
meaningful markers, and reference lines explicit in the figure source.

Default to open left/bottom axes, white background, and no grid. These are
defaults, not rules that override the figure's mathematical meaning. Retain a
border, grid, reference line, or other reading aid when it helps explain the
domain or comparison; document the reason in the figure source or guide.

ROC and precision–recall plots, including precision–recall isoquants, keep all
four spines. Their relevant domain is the unit box `[0,1] × [0,1]`; the upper and
right borders communicate real boundaries, rather than decoration. Set both
axes to `[0,1]` without padding outside that domain. Keep the intended axis
orientation, curve interpolation, baselines, and data unchanged. Boundary
markers may extend across the frame so they remain visible. Place labels inside
the frame or in the surrounding margin without extending the data limits.

In Python, call `unit_box(ax)` from `course_plot` after plotting (or
`unit_box(plt.gca())` in a pyplot example). In PGFPlots, use
`[course axis,course unit box,...]`. These conventions preserve the figure's
chosen proportions. A purposeful zoom can use tighter limits within the unit
box after applying the shared style; retain all four borders and explicitly
label it a detail view so it is not mistaken for the full domain.

Prefer direct labels when they fit; use an unboxed legend when labels would
overlap or curves coincide. Document purposeful exceptions to the shared defaults
and review their effect at the figure's final size.

Every figure needs a meaningful caption or description conveying the comparison,
axes/units, and learning point. Statistical definitions and data must not change
as part of a style migration.

## Building and manual review

```sh
make design-sync       # regenerate adapters after editing tokens
make design-guide      # visual specimen at design/guide.pdf
make figures           # regenerate all ten external vector figure inputs
make notes-print       # monochrome accents, same notes source
make                   # all existing course-document targets
make clean             # remove auxiliaries after reviewing the build
```

Compile affected documents locally before merging and review representative pages
at actual size and in grayscale. Check label placement, readable type, contrast,
distinguishable series, use of the shared theme, and registered artwork. Use the
guidance above to assess purposeful exceptions. Builds and design review are
manual; the repository has no continuous-integration workflows.

## Migration and accessibility status

- Shared notes/syllabus typography, links, code styling, and cover
  identity: migrated.
- Editorial policy: scholarly prose and restrained formatting documented in this
  guide and `AGENTS.md`. Decorative exercise and reading boxes removed from the
  shared notes style. Useful course-orientation lists, Python vocabulary,
  learning examples, ROC cases, bias--variance terms, and selection-optimism
  factors retain their list structure with restrained typography. Explanations
  remain in connected paragraphs; nested CV uses three numbered steps rather
  than deeply nested bullets. Mathematical notation, table headers, and source
  quotations retain their functional structure.
- Part covers: Bootcamp, Foundations, and The Canon use registered unlettered
  artwork with a single typeset title, restrained numbering, and balanced
  dedicated opening pages in both versions of the notes. Lettered originals
  remain available as source artwork.
- Code access: five active listings are included from local Python files, with
  readable type and filename links to matching public copies in `pols4728`. Links
  pin a public commit; the private semester repository remains the source of truth.
- Native classification plots, both notes notebooks, and the three regularization
  figure scripts: migrated. The regularization scripts retain their numerical
  checks and export their PDFs through the shared TeX font setup. Regenerating
  them requires NumPy, Matplotlib, and pdflatex; the collinearity script also uses
  scikit-learn and statwrap.
- All ten external notes figures now have maintained generators and use the
  shared theme. The six inherited figures and the ridge coefficient path are
  migrated. `notes/notebooks/README.md` records recovered notebook sources,
  preserved calculations, and explicit reconstructions. The legacy registry is
  empty. Regenerate vector PDFs with `make figures`; ordinary document builds
  use the tracked figure inputs.
- Existing slide theme: later migration. Homework sources, figures, and PDFs stay
  in ignored local directories; the shared styles remain available to local builds.
- Monochrome mode changes managed document accents. Embedded figure PDFs retain
  their colors and use line patterns, marker shapes, and direct labels that also
  work when printed in grayscale.
- PDF language metadata, real-text course identity, contrast, descriptive prose,
  and redundant visual encodings are supported in this pass.
- Current TeX Live 2023 outputs are untagged. Sidecar descriptions are not embedded
  alternative text. Do not label the results screen-reader accessible or PDF/UA
  compliant. Tagged structure, reading order, math accessibility, figure alternative
  text, and assistive-technology testing require a separate build migration using
  the [LaTeX tagging project's guidance](https://tagging-project.latex-project.org/).
