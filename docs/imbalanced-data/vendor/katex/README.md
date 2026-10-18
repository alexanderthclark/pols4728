# KaTeX 0.16.22

The opening formulas are rendered from LaTeX at build time into `logistic-math.js`
as HTML plus accessible MathML. The page needs no KaTeX JavaScript or network calls.

From the project root, regenerate with an installed KaTeX 0.16.22 package:

```sh
node scripts/render-logistic-math.cjs
```

The stylesheet retains six WOFF2 faces: Main Regular, Math Italic, and Size 1–4
Regular for scalable delimiters. Other font faces and formats are omitted.
KaTeX is MIT licensed; the unmodified fonts use the SIL Open Font License 1.1.
See `LICENSE` and `FONT-LICENSE.txt`.
