import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import test from 'node:test';
const typography = createRequire(import.meta.url)('../docs/shared/svg-math.js');

test('plot notation uses complete script runs and upright script digits', () => {
  const label = typography.svg('Distance x₁ (km); F₁₂; hₘ²');
  assert.match(label, /font-style="italic">x<\/tspan>/);
  assert.match(label, /baseline-shift="sub"[^>]*font-size="75%"[^>]*font-style="normal">12<\/tspan>/);
  assert.match(label, /baseline-shift="sub"[^>]*>m<\/tspan><tspan baseline-shift="super"[^>]*>2<\/tspan>/);
  assert.doesNotMatch(label, /[₁₂ₘ²]/);
  assert.match(label, /^Distance /);
});

test('small numerical values keep their mantissa and use true exponents', () => {
  assert.equal(typography.html('−1.19 × 10^−7', false), '−1.19 × 10<sup>−7</sup>');
  assert.equal(typography.html('−1.19e−7', false), '−1.19 × 10<sup>−7</sup>');
  assert.match(typography.svg('1.42 × 10^−14', false), /1\.42 × 10<tspan baseline-shift="super"[^>]*>−14<\/tspan>/);
  assert.equal(typography.html('0', false), '0');
});

test('mathematical labels escape source text without italicizing ordinary words', () => {
  const label = typography.svg('Fit h₁ when x < 8 & F > 0');
  assert.match(label, /^Fit /);
  assert.match(typography.svg('Focus a delivery for y'), /^Focus a delivery for /);
  assert.match(label, /&lt; 8 &amp;/);
  assert.doesNotMatch(typography.svg('<script>alert(1)</script>'), /<script>/);
});

test('native formulas keep prose intact and replace only standalone mathematical identifiers', () => {
  const N = typography.notation;
  const source = 'Keep y fixed by every tree; stay at y1 and y_score. Then use y − F.';
  const parts = N.replaceParts(source, [['y − F', N.variable('y') + N.operator('−') + N.variable('F')], ['y', N.variable('y')]]);
  assert.deepEqual(parts.filter(part => typeof part !== 'string').map(part => part.label), ['y', 'y − F']);
  const rendered = N.parts(parts);
  assert.match(rendered, / fixed by every tree; stay at y1 and y_score\. Then use /);
  assert.match(rendered, /<math[^>]*><mrow><mi>y<\/mi><mo>−<\/mo><mi>F<\/mi>/);
  assert.equal(N.replaceParts('α is an αvalue; use α.', [['α', N.variable('α')]]).filter(part => typeof part !== 'string').length, 2);
});

test('native numerical formulas preserve precision while laying out fractions and scientific powers', () => {
  const N = typography.notation;
  assert.match(N.number('−1/3'), /^<mfrac><mrow><mn>−1<\/mn><\/mrow><mrow><mn>3<\/mn><\/mrow><\/mfrac>$/);
  assert.match(N.number('−1.19e−7'), /^<mn>−1\.19<\/mn><mo>×<\/mo><msup><mn>10<\/mn><mrow><mn>−7<\/mn>/);
  assert.match(N.number('1.42 × 10^−14'), /<mn>1\.42<\/mn><mo>×<\/mo><msup><mn>10<\/mn><mrow><mn>−14<\/mn>/);
  assert.equal(N.number('0.7031'), '<mn>0.7031</mn>');
  assert.match(N.apply(N.variable('Var'), N.variable('Xu')), /<mo stretchy="false">\(<\/mo>.*<mo stretchy="false">\)<\/mo>/);
  assert.equal(N.parts(['x < 2 & ', N.part(N.variable('x'), 'x < "2"')]).startsWith('x &lt; 2 &amp; '), true);
});
