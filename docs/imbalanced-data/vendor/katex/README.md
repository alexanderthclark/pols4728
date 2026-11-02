# KaTeX 0.16.22

The scrolling formulas and the reading version's inline and display equations
are rendered from LaTeX at build time, with HTML plus MathML. The page needs no
KaTeX JavaScript or network calls. `logistic-math.js` contains the reusable
renderings for scenes and the optional sampling plot; `index.html` also contains
the complete static renderings for reading and printing without JavaScript.

From the project root, regenerate with an installed KaTeX 0.16.22 package:

```sh
node scripts/render-logistic-math.cjs
```

Edit the display and inline TeX dictionaries in that script. Static expressions
in `index.html` use `data-math="formulaKey"`; preserve their `<!-- /math -->`
marker so the generator can replace the complete rendering on subsequent runs.
Scene captions can select a rendering with `equationMath`. Wide chart sizing
must target only the direct SVG child of `.scene-canvas`: KaTeX's own nested
SVGs draw the tall parentheses and cancellation strokes.

The stylesheet retains six WOFF2 faces: Main Regular, Math Italic, and Size 1–4
Regular for scalable delimiters. Other font faces and formats are omitted.
KaTeX is MIT licensed; the unmodified fonts use the SIL Open Font License 1.1.
See `LICENSE` and `FONT-LICENSE.txt`.
