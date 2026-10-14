(function (root, factory) {
  'use strict';
  const model = factory();
  if (typeof module === 'object' && module.exports) module.exports = model;
  if (root) root.TreeBalance = model;
})(typeof window !== 'undefined' ? window : null, function () {
  'use strict';
  // A new Gini example, distinct from the preceding log-loss example.
  const splits = Object.freeze({
    A: Object.freeze({positiveLeft: .1, negativeLeft: 0}),
    B: Object.freeze({positiveLeft: .9, negativeLeft: .1})
  });
  const gini = p => 2 * p * (1 - p);
  function evaluate(name, p) {
    if (!splits[name]) throw new TypeError('Choose split A or B.');
    if (!Number.isFinite(p) || p <= 0 || p >= 1) throw new RangeError('Prevalence must be between zero and one.');
    const s = splits[name];
    const q = p * s.positiveLeft + (1 - p) * s.negativeLeft;
    const pLeft = p * s.positiveLeft / q;
    const pRight = p * (1 - s.positiveLeft) / (1 - q);
    const leftContribution = q * gini(pLeft);
    const rightContribution = (1 - q) * gini(pRight);
    const gain = gini(p) - leftContribution - rightContribution;
    return {name, p, q, pLeft, pRight, parent: gini(p), leftContribution,
      rightContribution, gain, relativeGain: gain / gini(p)};
  }
  function choose(p, cutoff = 0) {
    const a = evaluate('A', p), b = evaluate('B', p);
    const best = a.gain >= b.gain ? a : b;
    return best.gain + 1e-14 >= cutoff ? best.name : null;
  }
  return Object.freeze({splits, gini, evaluate, choose});
});
