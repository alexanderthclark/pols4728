/* KaTeX HTML + accessible MathML generated from affineTex with KaTeX 0.16.22.
 * Build options: {output:'htmlAndMathml', displayMode:true, throwOnError:true}.
 * Runtime presentation uses local vendor/katex/katex.min.css and two fonts.
 * No KaTeX runtime JavaScript or remote requests are required.
 */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.NNFormula = api;
})(typeof window !== 'undefined' ? window : globalThis, function () {
  'use strict';
  const affineTex = String.raw`\hat{y}=wx+b=1\times x+0.5`;
  const affineMarkup = "<span class=\"katex-display\"><span class=\"katex\"><span class=\"katex-mathml\"><math aria-label=\"y hat equals w times x plus b equals one times x plus zero point five\" xmlns=\"http://www.w3.org/1998/Math/MathML\" display=\"block\"><semantics><mrow><mover accent=\"true\"><mi>y</mi><mo>^</mo></mover><mo>=</mo><mi>w</mi><mi>x</mi><mo>+</mo><mi>b</mi><mo>=</mo><mn>1</mn><mo>×</mo><mi>x</mi><mo>+</mo><mn>0.5</mn></mrow><annotation encoding=\"application/x-tex\">\\hat{y}=wx+b=1\\times x+0.5</annotation></semantics></math></span><span class=\"katex-html\" aria-hidden=\"true\"><span class=\"base\"><span class=\"strut\" style=\"height:0.8889em;vertical-align:-0.1944em;\"></span><span class=\"mord accent\"><span class=\"vlist-t vlist-t2\"><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.6944em;\"><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"mord mathnormal\" style=\"margin-right:0.03588em;\">y</span></span><span style=\"top:-3em;\"><span class=\"pstrut\" style=\"height:3em;\"></span><span class=\"accent-body\" style=\"left:-0.1944em;\"><span class=\"mord\">^</span></span></span></span><span class=\"vlist-s\">​</span></span><span class=\"vlist-r\"><span class=\"vlist\" style=\"height:0.1944em;\"><span></span></span></span></span></span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span><span class=\"mrel\">=</span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span></span><span class=\"base\"><span class=\"strut\" style=\"height:0.6667em;vertical-align:-0.0833em;\"></span><span class=\"mord mathnormal\" style=\"margin-right:0.02691em;\">w</span><span class=\"mord mathnormal\">x</span><span class=\"mspace\" style=\"margin-right:0.2222em;\"></span><span class=\"mbin\">+</span><span class=\"mspace\" style=\"margin-right:0.2222em;\"></span></span><span class=\"base\"><span class=\"strut\" style=\"height:0.6944em;\"></span><span class=\"mord mathnormal\">b</span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span><span class=\"mrel\">=</span><span class=\"mspace\" style=\"margin-right:0.2778em;\"></span></span><span class=\"base\"><span class=\"strut\" style=\"height:0.7278em;vertical-align:-0.0833em;\"></span><span class=\"mord\">1</span><span class=\"mspace\" style=\"margin-right:0.2222em;\"></span><span class=\"mbin\">×</span><span class=\"mspace\" style=\"margin-right:0.2222em;\"></span></span><span class=\"base\"><span class=\"strut\" style=\"height:0.6667em;vertical-align:-0.0833em;\"></span><span class=\"mord mathnormal\">x</span><span class=\"mspace\" style=\"margin-right:0.2222em;\"></span><span class=\"mbin\">+</span><span class=\"mspace\" style=\"margin-right:0.2222em;\"></span></span><span class=\"base\"><span class=\"strut\" style=\"height:0.6444em;\"></span><span class=\"mord\">0.5</span></span></span></span></span>";

  function render(element, frame) {
    if (frame.kind === 'affine') {
      element.innerHTML = affineMarkup;
      element.setAttribute('data-tex', affineTex);
    } else {
      element.textContent = frame.equation;
      element.removeAttribute('data-tex');
    }
  }

  return Object.freeze({ affineTex, affineMarkup, render });
});
