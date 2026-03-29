# Chapter Transformation Policy

This policy defines what is allowed when converting source material into book chapters.

## Allowed transformations

- Convert LaTeX sectioning to MyST headings without changing title text.
- Convert LaTeX equations to MyST math directives while preserving expressions and labels.
- Convert LaTeX figure blocks to MyST figure directives using the same underlying assets and caption meaning.
- Convert LaTeX emphasis and list syntax to Markdown equivalents.
- Convert LaTeX quote and reading-box environments to semantically equivalent MyST blocks.
- Apply minimal punctuation or spacing fixes required for parser stability.

## Disallowed transformations

- Conceptual summarization or omission of substantive arguments.
- Reordering major arguments or dropping sections/subsections.
- Replacing equations with prose descriptions.
- Dropping cited sources that appear in the source chapter.
- Replacing figures with different assets or removing figure context.

## Review rule

If a conversion cannot preserve source fidelity because of rendering constraints, document the exact deviation in the chapter's deviation log before merge.
