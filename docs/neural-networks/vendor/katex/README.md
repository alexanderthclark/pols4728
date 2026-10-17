# KaTeX presentation assets

Version: **0.16.22**, obtained from the official `katex@0.16.22` npm package.

The opening affine equation is compiled at build time using:

```js
katex.renderToString(String.raw`\hat{y}=wx+b=1\times x+0.5`, {
  output: 'htmlAndMathml', displayMode: true, throwOnError: true
});
```

`formula.js` contains the compiled result. The HTML supplies conventional
mathematical spacing, italic variables, and the positioned hat accent. Its
`katex-html` subtree is hidden from assistive technology; the parallel MathML
retains the exact TeX annotation and a spoken label. No KaTeX JavaScript runs
in the browser.

## Included files

- `katex.min.css`: upstream rendering rules, with font-face declarations
  reduced to the two faces used by this equation and WOFF2 sources only.
- `fonts/KaTeX_Main-Regular.woff2`: original, unmodified regular math font.
- `fonts/KaTeX_Math-Italic.woff2`: original, unmodified italic variable font.
- `LICENSE`: upstream MIT license for KaTeX code and CSS.
- `FONT-LICENSE.txt`: SIL Open Font License 1.1 and the font copyright notices
  embedded in the corresponding upstream TTF metadata.

The standard OFL text was obtained from
https://openfontlicense.org/documents/OFL.txt. The fonts are copied unchanged;
no font subsetting or renaming has been performed.
