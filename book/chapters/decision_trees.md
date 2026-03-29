(decision_trees)=
# Decision Trees

Decision trees learn prediction rules by recursively partitioning feature space. They are interpretable and interaction-friendly but high variance without regularization.

## Tree vocabulary

- Root: first split node.
- Internal node: rule-based split.
- Leaf: terminal region with predicted value/class.
- Depth: longest path from root to leaf.

A tree prediction is the average outcome (regression) or class probability/label (classification) within the terminal leaf.

## Split objective

For regression, CART chooses split `(j,s)` minimizing within-node squared error:

$$
L(j,s)=\sum_{i\in R_1(j,s)}(y_i-\bar y_1)^2 +
\sum_{i\in R_2(j,s)}(y_i-\bar y_2)^2.
$$

For classification, common impurity criteria are Gini and entropy.

## Decision stumps

A depth-1 tree (stump) uses a single split. It is easy to audit but often underfits.

Stumps are still useful:

- as weak learners for boosting,
- as explanatory first-pass models,
- for monotonic threshold baselines.

## Growing deeper trees

CART is greedy: each split is chosen myopically at the current node. This can yield strong predictive performance but also overfit if unconstrained.

Important hyperparameters:

- `max_depth`
- `max_leaf_nodes`
- `min_samples_split`
- `min_samples_leaf`
- `min_impurity_decrease`
- `ccp_alpha` (cost-complexity pruning)

Cost-complexity objective:

$$
C_\alpha(T)=\sum_{m=1}^{|T|} N_m\,\mathrm{Impurity}(m)+\alpha |T|.
$$

## Tuning strategy

1. Start with permissive tree.
2. Tune leaf-size and depth controls by cross-validation.
3. Add pruning (`ccp_alpha`) when trees remain unstable.
4. Evaluate on untouched test data.

Randomized or coarse-to-fine search is often more efficient than exhaustive grids.

## Practical caveats

- Small data perturbations can alter tree structure.
- Highly correlated predictors can substitute for each other, changing interpretation.
- Feature engineering still matters (missingness handling, monotonic transforms, rare categories).

Use single trees for interpretability and ensembles when prediction stability is the priority.
