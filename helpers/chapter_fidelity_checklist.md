# Chapter Fidelity Checklist

Use this checklist before marking a migrated chapter as complete.

## Source and target

- Source chapter file is identified and locked.
- Target chapter path is identified and committed.
- Chapter title and TOC position match the source plan.

## Structural parity

- Subsection titles match source titles and order.
- Subsubsection titles match source titles and order.
- Paragraph-level titled blocks match source titles and order.

## Content fidelity

- Main prose is near-verbatim (no conceptual compression).
- Equations are preserved and labeled where labels exist.
- Figures are preserved with matching source assets and captions.
- Citation keys are preserved and resolve to bibliography entries.
- Footnotes and quote blocks are preserved.

## Audit gates

- Heading hierarchy parity passes.
- Equation count parity passes.
- Figure reference parity passes.
- Citation key parity passes.
- Normalized text-length ratio meets target threshold.

## Build quality

- Chapter renders in clean local build.
- No chapter-specific parse warnings or directive errors.
- Math, figures, citations, and footnotes render correctly.
- TOC navigation resolves to the chapter correctly.

## Deviation log

- Any unavoidable transformation is documented with:
  - source location,
  - target location,
  - reason (renderer or parser constraint),
  - exact text/math/figure impact.
