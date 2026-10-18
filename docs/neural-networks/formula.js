/* Self-hosted KaTeX typesetting for every scroll-driven equation. */
(function (root, factory) {
  'use strict';
  const katex = typeof module === 'object' && module.exports
    ? require('./vendor/katex/katex.min.js')
    : root.katex || globalThis.katex;
  const api = factory(katex);
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.NNFormula = api;
})(typeof window !== 'undefined' ? window : globalThis, function (katex) {
  'use strict';

  const affineTex = String.raw`\hat{y}=wx+b=1\times x+0.5`;
  const cache = new Map();
  function typeset(tex) {
    if (cache.has(tex)) return cache.get(tex);
    let markup = katex.renderToString(tex, {
      output: 'htmlAndMathml', displayMode: true, throwOnError: true,
      strict: 'error', trust: false
    });
    if (tex === affineTex) {
      markup = markup.replace('<math ', '<math aria-label="y hat equals w times x plus b equals one times x plus zero point five" ');
    }
    if (cache.size >= 256) cache.delete(cache.keys().next().value);
    cache.set(tex, markup);
    return markup;
  }
  const affineMarkup = typeset(affineTex);

  function render(element, frame) {
    const tex = frame.compact && frame.equationCompact ? frame.equationCompact : frame.equation;
    element.innerHTML = typeset(tex);
    element.setAttribute('data-tex', tex);
  }

  return Object.freeze({ affineTex, affineMarkup, render });
});
