(tree_methods)=
# Tree Methods

## Decision trees

Decision trees partition feature space recursively and assign a prediction in each terminal node.

- Classification trees minimize impurity (e.g., Gini or entropy).
- Regression trees minimize within-node squared error.

## Decision stumps and depth

A decision stump (depth 1) is a one-split model. Deeper trees capture interactions and
nonlinearity, but overfit quickly without constraints.

Core complexity controls:

- `max_depth`
- `min_samples_leaf`
- `min_samples_split`
- post-pruning / cost-complexity regularization

## Random forests

Random forests combine many decorrelated trees.

1. Bootstrap sample observations for each tree.
2. At each split, search only a random subset of features.
3. Aggregate by averaging (regression) or voting (classification).

This reduces variance relative to a single deep tree while preserving nonlinear flexibility.

## Feature importance caveats

Default feature-importance summaries can be misleading under:

- correlated predictors,
- high-cardinality categorical variables,
- distribution shift.

Use permutation-based checks and out-of-sample diagnostics.

## Boosting

Boosting builds an additive model stagewise, each step improving residual errors.

### AdaBoost intuition

- upweight hard examples,
- combine weak learners with weighted voting.

### Gradient boosting

- define loss,
- fit each new learner to negative gradients (pseudo-residuals),
- update predictions incrementally.

### XGBoost-style extensions

- explicit regularization,
- shrinkage,
- subsampling,
- optimized split search and systems engineering.

## Practical tuning checklist

1. Start with train/validation split and baseline tree.
2. Tune depth and leaf size before ensemble size.
3. Use cross-validation for final hyperparameter comparison.
4. Report both predictive performance and calibration behavior.
