(random_forests)=
# Random Forests

Random forests reduce decision-tree variance by averaging many decorrelated trees.

## Canon objectives

- connect bagging intuition to RF variance reduction,
- tune RF-specific hyperparameters (`max_features`, `n_estimators`) systematically,
- interpret feature-importance outputs with appropriate skepticism.

## Bagging intuition

For bootstrap samples `b=1,\dots,B`, fit trees `\hat f^{*b}` and average:

$$
\hat f_{\text{bag}}(x)=\frac{1}{B}\sum_{b=1}^B \hat f^{*b}(x)
$$

(for classification, average probabilities or use majority vote).

Bagging primarily reduces variance, not bias.

## Why random forests improve on plain bagging

Random forests also randomize feature choice at each split.

- At each node, sample `m` candidate predictors out of `p`.
- Search split only within that subset.
- Repeat independently at every node.

This decorrelates trees and improves ensemble variance reduction.

## Variance decomposition

If individual-tree variance is `\sigma^2` and pairwise correlation is `\rho`, ensemble variance is

$$
\mathrm{Var}(\bar T)=\sigma^2\left(\rho+\frac{1-\rho}{B}\right).
$$

As `B` increases, variance approaches `\rho\sigma^2`; lowering tree correlation is therefore critical.

## Core hyperparameters

- `n_estimators`: number of trees.
- `max_features`: features considered per split (most important RF-specific parameter).
- `min_samples_leaf`, `min_samples_split`, `max_depth`: tree complexity controls.
- `bootstrap` and `max_samples`: resampling behavior.

## Out-of-bag (OOB) evaluation

Each observation is excluded from roughly one-third of bootstrap samples. Predictions from trees that did not see observation `i` provide an OOB estimate, useful as a quick internal validation check.
OOB is useful for model development, but keep a final external test set for reported results.

## Feature importance caveats

Default impurity-based importances can be misleading when:

- predictors are correlated,
- cardinality differs sharply across variables,
- leakage features exist.

Use permutation importance and out-of-sample ablation as robustness checks.

## Practical workflow

1. Tune `max_features` and leaf-size controls first.
2. Increase `n_estimators` until OOB/validation error plateaus.
3. Compare against simpler baselines (regularized linear/logistic models).
4. Report both predictive metrics and uncertainty-relevant diagnostics.
