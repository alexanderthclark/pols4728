/* Native MathML generated from affineTex with KaTeX 0.16.22.
 * Build options: {output:'mathml', displayMode:true, throwOnError:true}.
 * No KaTeX JavaScript, CSS, or network request is required at runtime.
 */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.NNFormula = api;
})(typeof window !== 'undefined' ? window : globalThis, function () {
  'use strict';
  const affineTex = String.raw`\hat{y}=wx+b=1\times x+0.5`;
  const affineMarkup = "<math aria-label=\"y hat equals w times x plus b equals one times x plus zero point five\" xmlns=\"http://www.w3.org/1998/Math/MathML\" display=\"block\"><semantics><mrow><mover accent=\"true\"><mi>y</mi><mo>^</mo></mover><mo>=</mo><mi>w</mi><mi>x</mi><mo>+</mo><mi>b</mi><mo>=</mo><mn>1</mn><mo>×</mo><mi>x</mi><mo>+</mo><mn>0.5</mn></mrow><annotation encoding=\"application/x-tex\">\\hat{y}=wx+b=1\\times x+0.5</annotation></semantics></math>";

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
