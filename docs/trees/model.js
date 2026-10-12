(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.TreePartitions = api;
})(typeof window !== 'undefined' ? window : null, function () {
  'use strict';

  const defaultBounds = [[-2, 2], [-2, 2]];

  function isLeaf(node) {
    if (!node || typeof node !== 'object') throw new TypeError('A tree node is required.');
    if (Object.prototype.hasOwnProperty.call(node, 'value')) {
      if (!Number.isFinite(node.value)) throw new TypeError('Leaf values must be finite.');
      return true;
    }
    if (!Number.isInteger(node.feature) || node.feature < 0 || !Number.isFinite(node.threshold)) {
      throw new TypeError('Splits require a nonnegative feature index and a finite threshold.');
    }
    return false;
  }

  function copyBounds(bounds) {
    if (!Array.isArray(bounds) || bounds.length === 0 || !bounds.every(pair =>
      Array.isArray(pair) && pair.length === 2 && pair.every(Number.isFinite) && pair[0] < pair[1])) {
      throw new RangeError('Bounds must contain finite, increasing [minimum, maximum] pairs.');
    }
    return bounds.map(pair => pair.slice());
  }

  // Points on a threshold follow the left branch, without an epsilon offset.
  function pathNodes(tree, point) {
    if (!Array.isArray(point) || point.length === 0 || !point.every(Number.isFinite)) {
      throw new TypeError('A point must be a nonempty array of finite coordinates.');
    }
    const path = [];
    let node = tree;
    while (true) {
      path.push(node);
      if (isLeaf(node)) return path;
      if (node.feature >= point.length) throw new RangeError('The point is missing a split feature.');
      node = point[node.feature] <= node.threshold ? node.left : node.right;
    }
  }

  function leafAt(tree, point) {
    const path = pathNodes(tree, point);
    return path[path.length - 1];
  }

  function evaluate(tree, point) {
    return leafAt(tree, point).value;
  }

  function splitCount(tree) {
    return isLeaf(tree) ? 0 : 1 + splitCount(tree.left) + splitCount(tree.right);
  }

  // Bounds describe drawing geometry: adjacent regions share an edge. The
  // path/evaluate API supplies exact <= / > membership on that edge. Regions
  // of zero width are omitted, including a leaf touching only the plot edge.
  function regions(tree, bounds = defaultBounds) {
    const result = [];
    const initial = copyBounds(bounds);
    function visit(node, box, path) {
      const nextPath = path.concat(node.id);
      if (isLeaf(node)) {
        result.push({ bounds: box, value: node.value, leafId: node.id, path: nextPath });
        return;
      }
      if (node.feature >= box.length) throw new RangeError('Bounds are missing a split feature.');
      const [lo, hi] = box[node.feature];
      if (node.threshold > lo) {
        const leftBounds = box.map(pair => pair.slice());
        leftBounds[node.feature][1] = Math.min(hi, node.threshold);
        visit(node.left, leftBounds, nextPath);
      }
      if (node.threshold < hi) {
        const rightBounds = box.map(pair => pair.slice());
        rightBounds[node.feature][0] = Math.max(lo, node.threshold);
        visit(node.right, rightBounds, nextPath);
      }
    }
    visit(tree, initial, []);
    return result;
  }

  // Intersect actual leaf rectangles. Extending each split across the whole
  // domain would introduce artificial cells beyond a split's parent region.
  // Cells are retained after cancellation: common refinement describes the
  // component partitions, which may be finer than the sum's level sets.
  function refinement(trees, bounds = defaultBounds) {
    if (!Array.isArray(trees)) throw new TypeError('Trees must be an array.');
    let cells = [{ bounds: copyBounds(bounds), value: 0, leafId: '', leafIds: [], path: [], contributions: [] }];
    for (const tree of trees) {
      const leaves = regions(tree, bounds);
      const nextCells = [];
      for (const cell of cells) {
        for (const leaf of leaves) {
          const intersection = cell.bounds.map(([lo, hi], feature) => [
            Math.max(lo, leaf.bounds[feature][0]), Math.min(hi, leaf.bounds[feature][1])
          ]);
          if (intersection.some(([lo, hi]) => lo >= hi)) continue;
          const leafIds = cell.leafIds.concat(leaf.leafId);
          nextCells.push({
            bounds: intersection,
            value: cell.value + leaf.value,
            leafId: leafIds.join('|'),
            leafIds,
            path: cell.path.concat(leaf.path),
            contributions: cell.contributions.concat(leaf.value)
          });
        }
      }
      cells = nextCells;
    }
    return cells;
  }

  function segments1D(tree, min = -2, max = 2) {
    return regions(tree, [[min, max]]).map(region => ({
      lo: region.bounds[0][0], hi: region.bounds[0][1],
      value: region.value, leafId: region.leafId
    }));
  }

  function sumSegments1D(trees, min = -2, max = 2) {
    return refinement(trees, [[min, max]]).map(cell => ({
      lo: cell.bounds[0][0], hi: cell.bounds[0][1],
      value: cell.value, leafId: cell.leafId,
      leafIds: cell.leafIds, contributions: cell.contributions
    })).sort((a, b) => a.lo - b.lo);
  }

  const examples = {
    stump1D: {
      id: 's1-root', feature: 0, threshold: -0.5,
      left: { id: 's1-left', value: 0 }, right: { id: 's1-right', value: 1 }
    },
    grown1D: {
      id: 'g1-root', feature: 0, threshold: -0.5,
      left: { id: 'g1-left', value: 0 },
      right: {
        id: 'g1-right', feature: 0, threshold: 0.75,
        left: { id: 'g1-middle', value: 1 }, right: { id: 'g1-high', value: 3 }
      }
    },
    second1D: {
      id: 's2-root', feature: 0, threshold: 0.75,
      left: { id: 's2-left', value: 0 }, right: { id: 's2-right', value: 2 }
    },
    stumpX: {
      id: 'sx-root', feature: 0, threshold: 0,
      left: { id: 'sx-left', value: 0 }, right: { id: 'sx-right', value: 1 }
    },
    stumpY: {
      id: 'sy-root', feature: 1, threshold: 0,
      left: { id: 'sy-left', value: 0 }, right: { id: 'sy-right', value: 2 }
    },
    recursive2D: {
      id: 'r2-root', feature: 0, threshold: 0,
      left: { id: 'r2-left', value: 0 },
      right: {
        id: 'r2-right', feature: 1, threshold: 0,
        left: { id: 'r2-low', value: 1 }, right: { id: 'r2-high', value: 3 }
      }
    },
    matched2D: {
      id: 'm2-root', feature: 0, threshold: 0,
      left: {
        id: 'm2-left', feature: 1, threshold: 0,
        left: { id: 'm2-left-low', value: 0 }, right: { id: 'm2-left-high', value: 2 }
      },
      right: {
        id: 'm2-right', feature: 1, threshold: 0,
        left: { id: 'm2-right-low', value: 1 }, right: { id: 'm2-right-high', value: 3 }
      }
    },
    conditional: {
      id: 'c-root', feature: 0, threshold: 0,
      left: { id: 'c-left', value: 0 },
      right: {
        id: 'c-right', feature: 1, threshold: 0,
        left: { id: 'c-low', value: 0 }, right: { id: 'c-high', value: 2 }
      }
    }
  };

  return { evaluate, leafAt, pathNodes, splitCount, regions, refinement, segments1D, sumSegments1D, examples };
});
