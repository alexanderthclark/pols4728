(function (root, factory) {
  'use strict';
  const model = factory();
  if (typeof module === 'object' && module.exports) module.exports = model;
  if (root) root.TreeBalance = model;
})(typeof window !== 'undefined' ? window : null, function () {
  'use strict';
  // A new entropy example: A has a pure positive left child.
  const splits = Object.freeze({
    A: Object.freeze({positiveLeft: .5, negativeLeft: 0}),
    B: Object.freeze({positiveLeft: .9, negativeLeft: .1})
  });
  const entropy = p => {
    if (!Number.isFinite(p) || p < 0 || p > 1) throw new RangeError('A class probability must lie between zero and one.');
    if (p === 0 || p === 1) return 0;
    return -p * Math.log(p) - (1 - p) * Math.log1p(-p);
  };
  function evaluate(name, p) {
    if (!splits[name]) throw new TypeError('Choose split A or B.');
    if (!Number.isFinite(p) || p <= 0 || p >= 1) throw new RangeError('Prevalence must be between zero and one.');
    const s = splits[name];
    const q = p * s.positiveLeft + (1 - p) * s.negativeLeft;
    const pLeft = p * s.positiveLeft / q;
    const pRight = p * (1 - s.positiveLeft) / (1 - q);
    const leftContribution = q * entropy(pLeft);
    const rightContribution = (1 - q) * entropy(pRight);
    const gain = entropy(p) - leftContribution - rightContribution;
    return {name, p, q, pLeft, pRight, parent: entropy(p), leftContribution,
      rightContribution, gain, relativeGain: gain / entropy(p)};
  }
  function choose(p, cutoff = 0) {
    const a = evaluate('A', p), b = evaluate('B', p);
    const best = a.gain >= b.gain ? a : b;
    return best.gain + 1e-14 >= cutoff ? best.name : null;
  }
  return Object.freeze({splits, entropy, evaluate, choose});
});
