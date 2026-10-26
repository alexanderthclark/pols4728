# KaTeX presentation assets

Version: **0.16.22**, from the official `katex@0.16.22` npm package.

The lesson serves the KaTeX runtime locally to typeset changing equations.
`state.js` supplies LaTeX; `formula.js` uses `renderToString` with HTML and
MathML output, display mode, strict errors, and untrusted commands disabled.
Longer expressions have compact line breaks for phone layouts. The opening
also retains precompiled markup in the HTML. No remote requests are required.

## Included files

- `katex.min.js`: unmodified upstream runtime.
- `katex.min.css`: upstream rules with font declarations limited to the ten
  required faces and WOFF2 sources.
- `fonts/KaTeX_Main-Regular.woff2`: upright operators and numbers.
- `fonts/KaTeX_Math-Italic.woff2`: mathematical variables.
- `fonts/KaTeX_Main-Bold.woff2` and `KaTeX_Math-BoldItalic.woff2`: vectors and bold mathematical parameters.
- `fonts/KaTeX_Size1-Regular.woff2` and `KaTeX_Size2-Regular.woff2`: large operators.
- `fonts/KaTeX_Size3-Regular.woff2` and `KaTeX_Size4-Regular.woff2`: tall delimiters, including cases braces.
- `fonts/KaTeX_AMS-Regular.woff2`: blackboard-bold letters and mathematical symbols.
- `fonts/KaTeX_Caligraphic-Regular.woff2`: calligraphic mathematical letters.
- `LICENSE`: upstream MIT license for code and CSS.
- `FONT-LICENSE.txt`: SIL Open Font License 1.1 and the copyright notices from
  the corresponding upstream font metadata.

The fonts are copied unchanged; no subsetting or renaming has been performed.
