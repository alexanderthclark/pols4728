const mathNamespace = 'http://www.w3.org/1998/Math/MathML';
export const featureLabels = ['ability', 'neighborhood', 'experience'];
const featureNames = ['ability', 'neighborhood opportunity', 'experience'];
const activeTerms = new Set(['all', 'result', 'sum', 'weight', 'after', 'before', 'difference', 'value', 'observation', 'coalition', 'joining']);

function setMarkup(mask, labels = featureLabels) {
  if (!Number.isInteger(mask) || mask < 0 || mask >= 2 ** labels.length) throw new RangeError('Use a mask within the feature universe.');
  const members = labels.filter((label, index) => mask & (1 << index));
  return members.length
    ? `<mrow><mo stretchy="false">{</mo>${members.map(label => `<mtext>${label}</mtext>`).join('<mo>,</mo>')}<mo stretchy="false">}</mo></mrow>`
    : '<mo lspace="0" rspace="0">∅</mo>';
}

function setLabel(mask, names = featureNames) {
  if (!mask) return 'the empty set';
  return `the set containing ${names.filter((name, index) => mask & (1 << index)).join(', ')}`;
}

function valueApplication(argument, valueFunction = '<msub><mi>v</mi><mi>x</mi></msub>') {
  return `<mrow>${valueFunction}<mo>⁡</mo><mrow><mo stretchy="false">(</mo>${argument}<mo stretchy="false">)</mo></mrow></mrow>`;
}

/** The full equation remains real MathML while each named term can be emphasized. */
export function shapleyFormulaMarkup(activeTerm = 'all') {
  if (!activeTerms.has(activeTerm)) throw new RangeError(`Unknown formula term: ${activeTerm}.`);
  const classes = term => `formula-term formula-${term}${activeTerm === 'all' || activeTerm === term ? ' formula-active' : ''}`;
  const symbol = (letter, term) => `<mi class="formula-symbol formula-symbol-${term}${activeTerm === 'all' || activeTerm === term ? ' formula-active' : ''}">${letter}</mi>`;
  const valueFunction = `<msub>${symbol('v', 'value')}${symbol('x', 'observation')}</msub>`;
  const size = `<mrow><mo form="prefix" stretchy="false">|</mo>${symbol('S', 'coalition')}<mo form="postfix" stretchy="false">|</mo></mrow>`;
  const addingFeature = `<mrow>${symbol('S', 'coalition')}<mo>∪</mo><mrow><mo stretchy="false">{</mo>${symbol('i', 'joining')}<mo stretchy="false">}</mo></mrow></mrow>`;
  return `<div class="formula-equation" data-focus="${activeTerm}" role="math" aria-label="The Shapley value of feature i in the prediction game v sub x equals the sum over all subsets S of the feature universe F excluding i, of the size of S factorial times m minus the size of S minus one factorial, divided by m factorial, multiplied by v sub x of S with i added minus v sub x of S.">
    <math class="formula-line" xmlns="${mathNamespace}" display="block" aria-hidden="true"><mrow>
      <mrow class="${classes('result')}"><msub>${symbol('φ', 'result')}${symbol('i', 'joining')}</msub><mo>⁡</mo><mrow><mo stretchy="false">(</mo>${valueFunction}<mo stretchy="false">)</mo></mrow></mrow><mo>=</mo>
      <munder class="${classes('sum')}"><mo>∑</mo><mrow>${symbol('S', 'coalition')}<mo>⊆</mo><mi>F</mi><mo>∖</mo><mrow><mo stretchy="false">{</mo>${symbol('i', 'joining')}<mo stretchy="false">}</mo></mrow></mrow></munder>
      <mfrac class="${classes('weight')}"><mrow><mrow>${size}<mo>!</mo></mrow><mrow><mo stretchy="false">(</mo><mi>m</mi><mo>−</mo>${size}<mo>−</mo><mn>1</mn><mo stretchy="false">)</mo><mo>!</mo></mrow></mrow><mrow><mi>m</mi><mo>!</mo></mrow></mfrac>
    </mrow></math>
    <math class="formula-line" xmlns="${mathNamespace}" display="block" aria-hidden="true"><mrow><mrow class="${classes('difference')}"><mo stretchy="false">[</mo><mrow class="${classes('after')}">${valueApplication(addingFeature, valueFunction)}</mrow><mo>−</mo><mrow class="${classes('before')}">${valueApplication(symbol('S', 'coalition'), valueFunction)}</mrow><mo stretchy="false">]</mo></mrow></mrow></math>
  </div>`;
}

/** A concrete coalition value, using the same bit masks as the row table. */
export function valueMarkup(mask, {labels = featureLabels, names = featureNames} = {}) {
  const argument = setMarkup(mask, labels);
  return `<math class="math-inline coalition-value-math" xmlns="${mathNamespace}" aria-label="v sub x of ${setLabel(mask, names)}">${valueApplication(argument)}</math>`;
}

/** Keep the two concrete coalition names readable above their prediction columns. */
export function joiningMarkup(beforeMask, feature, {labels = featureLabels, names = featureNames} = {}) {
  setMarkup(beforeMask, labels);
  if (!Number.isInteger(feature) || feature < 0 || feature >= labels.length) throw new RangeError('Use an index within the feature universe.');
  if (beforeMask & (1 << feature)) throw new RangeError('The joining feature must be excluded from the preceding group.');
  const afterMask = beforeMask | (1 << feature);
  return `<div class="coalition-comparison" role="group" aria-label="Compare predictions before and after ${names[feature]} joins"><div><span class="coalition-label">Before</span>${valueMarkup(beforeMask, {labels,names})}</div><div><span class="coalition-label">After</span>${valueMarkup(afterMask, {labels,names})}</div></div>`;
}

/** A prediction is an output of the fitted model, before any background average. */
export function predictionMarkup(argument = '<mi>x</mi>', label = 'y hat of x, the fitted model prediction') {
  return `<math class="math-inline" xmlns="${mathNamespace}" aria-label="${label}"><mrow><mover accent="true"><mi>y</mi><mo>^</mo></mover><mo stretchy="false">(</mo>${argument}<mo stretchy="false">)</mo></mrow></math>`;
}
