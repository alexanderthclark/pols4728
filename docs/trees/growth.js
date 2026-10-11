(function (root, factory) {
  'use strict';
  const partitions = root && root.TreePartitions || (typeof require === 'function' ? require('./model.js') : null);
  const api = factory(partitions);
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.GrowthExample = api;
})(typeof window !== 'undefined' ? window : null, function (partitions) {
  'use strict';
  if (!partitions) throw new Error('Load TreePartitions before the growth example.');

  // Synthetic deliveries on a full input grid, with variation that is not
  // exactly additive. Both fits use these observations and a four-split budget.
  const observations = [
    { id: 'A', x: [2, 0], y: 8 }, { id: 'B', x: [4, 0], y: 10 },
    { id: 'C', x: [8, 0], y: 13 }, { id: 'D', x: [10, 0], y: 16 },
    { id: 'E', x: [2, 1], y: 10 }, { id: 'F', x: [4, 1], y: 11 },
    { id: 'G', x: [8, 1], y: 15 }, { id: 'H', x: [10, 1], y: 16 },
    { id: 'I', x: [2, 3], y: 12 }, { id: 'J', x: [4, 3], y: 14 },
    { id: 'K', x: [8, 3], y: 18 }, { id: 'L', x: [10, 3], y: 21 },
    { id: 'M', x: [2, 4], y: 13 }, { id: 'N', x: [4, 4], y: 16 },
    { id: 'O', x: [8, 4], y: 20 }, { id: 'P', x: [10, 4], y: 24 }
  ];
  const bounds = [[0, 12], [-0.5, 4.5]];
  const indices = observations.map((_, index) => index);
  const targets = observations.map(observation => observation.y);
  const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
  const baseline = mean(targets);
  const baseTree = { id: 'base', value: baseline };
  const tolerance = 1e-12;

  function predict(trees, point) {
    return trees.reduce((sum, tree) => sum + partitions.evaluate(tree, point), 0);
  }

  function mse(predictions) {
    return predictions.reduce((sum, value, index) => sum + (targets[index] - value) ** 2, 0) / targets.length;
  }

  function summarize(trees) {
    const predictions = observations.map(observation => predict(trees, observation.x));
    return {
      trees, predictions, mse: mse(predictions),
      splitCount: trees.reduce((sum, tree) => sum + partitions.splitCount(tree), 0),
      regionCount: partitions.refinement(trees, bounds).length
    };
  }

  function groupStats(members, values) {
    const value = mean(members.map(index => values[index]));
    const sse = members.reduce((sum, index) => sum + (values[index] - value) ** 2, 0);
    return { value, sse };
  }

  // Enumerate every nonempty midpoint partition. Iteration order breaks
  // exact ties by feature first, then by the lower threshold.
  function bestSplit(members, values) {
    const parent = groupStats(members, values);
    let best = null;
    for (let feature = 0; feature < 2; feature++) {
      const inputs = [...new Set(members.map(index => observations[index].x[feature]))].sort((a, b) => a - b);
      for (let cut = 0; cut < inputs.length - 1; cut++) {
        const threshold = (inputs[cut] + inputs[cut + 1]) / 2;
        const leftMembers = members.filter(index => observations[index].x[feature] <= threshold);
        const rightMembers = members.filter(index => observations[index].x[feature] > threshold);
        const left = groupStats(leftMembers, values);
        const right = groupStats(rightMembers, values);
        const reduction = parent.sse - left.sse - right.sse;
        if (reduction > tolerance && (!best || reduction > best.reduction + tolerance)) {
          best = { feature, threshold, left, right, reduction };
        }
      }
    }
    return best;
  }

  function replaceLeaf(node, id, replacement) {
    if (node.id === id) return replacement;
    if ('value' in node) return { ...node };
    return { ...node, left: replaceLeaf(node.left, id, replacement), right: replaceLeaf(node.right, id, replacement) };
  }

  const treeStages = [];
  let tree = { id: 'tree-root', value: baseline };
  treeStages.push({ ...summarize([tree]), tree, split: null });
  while (treeStages.length <= 4 && treeStages[treeStages.length - 1].mse > tolerance) {
    let winner = null;
    // regions visits the left child first, so equal improvements select the
    // leftmost eligible leaf. Each cut stays inside that leaf's rectangle.
    for (const region of partitions.regions(tree, bounds)) {
      const members = indices.filter(index => partitions.leafAt(tree, observations[index].x).id === region.leafId);
      const candidate = bestSplit(members, targets);
      if (candidate && (!winner || candidate.reduction > winner.reduction + tolerance)) {
        winner = { ...candidate, parentId: region.leafId, bounds: region.bounds };
      }
    }
    if (!winner) break;
    const left = { id: `${winner.parentId}-L`, value: winner.left.value };
    const right = { id: `${winner.parentId}-R`, value: winner.right.value };
    const split = {
      feature: winner.feature, threshold: winner.threshold,
      bounds: winner.bounds.map(pair => pair.slice()), parentId: winner.parentId,
      left, right, reduction: winner.reduction
    };
    tree = replaceLeaf(tree, winner.parentId, {
      id: winner.parentId, feature: winner.feature, threshold: winner.threshold, left, right
    });
    treeStages.push({ ...summarize([tree]), tree, split });
  }

  const stumps = [];
  const boostStages = [{ ...summarize([baseTree]), stump: null, residuals: targets.map(value => value - baseline) }];
  // Each unit-rate residual fit is computed, not prescribed. Stop after four
  // split nodes, the same budget as the single tree, rather than at zero error.
  for (let round = 1; round <= 4; round++) {
    const previous = boostStages[boostStages.length - 1];
    const residuals = targets.map((value, index) => value - previous.predictions[index]);
    const fitted = bestSplit(indices, residuals);
    if (!fitted) break;
    const stump = {
      id: `h${round}-root`, feature: fitted.feature, threshold: fitted.threshold,
      left: { id: `h${round}-left`, value: fitted.left.value },
      right: { id: `h${round}-right`, value: fitted.right.value }
    };
    stumps.push(stump);
    boostStages.push({ ...summarize([baseTree, ...stumps]), stump, residuals });
  }

  function freeze(value) {
    if (value && typeof value === 'object' && !Object.isFrozen(value)) {
      Object.values(value).forEach(freeze);
      Object.freeze(value);
    }
    return value;
  }

  return freeze({
    observations, bounds, baseline, baseTree, stumps, treeStages, boostStages,
    evaluate: partitions.evaluate, predict, mse
  });
});
