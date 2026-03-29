(boosting)=
# Boosting Methods

Boosting builds an additive predictor sequentially, where each new weak learner corrects errors left by prior learners.

## Canon objectives

- explain why sequential weak learners differ from bagged ensembles,
- tune learning rate, depth, and boosting rounds as a coupled system,
- control overfitting with validation-driven early stopping.

## Core idea

Initialize with a baseline model and iteratively add learners:

$$
\hat f_M(x)=\sum_{m=0}^M h_m(x).
$$

Unlike random forests, trees are not fit independently; order matters.

## Why boosting is powerful

- Captures nonlinearities and interactions with shallow trees.
- Converts weak learners into a strong predictor.
- Offers fine-grained bias-variance control through learning rate and number of iterations.

## AdaBoost (classification)

AdaBoost reweights observations each round:

1. Fit weak learner with current weights.
2. Increase weight on misclassified observations.
3. Add learner with performance-based coefficient.

Interpretation: the model focuses progressively on hard examples.

## Gradient boosting

Gradient boosting generalizes boosting to arbitrary differentiable loss.

At stage `m`, compute pseudo-residuals

$$
r_{im} = -\left[\frac{\partial \ell(y_i,f(x_i))}{\partial f(x_i)}\right]_{f=f_{m-1}},
$$

fit a tree to `r_{im}`, and update

$$
f_m(x)=f_{m-1}(x)+\eta\,h_m(x),
$$

where `\eta` is the learning rate.

For squared error loss, pseudo-residuals equal ordinary residuals.

## XGBoost-style enhancements

Modern boosted-tree systems add:

- explicit L1/L2 regularization on leaf weights,
- second-order optimization terms,
- subsampling for rows and columns,
- highly optimized split search and parallel computation.

These engineering choices are why boosted trees remain strong tabular-data baselines.

## Key hyperparameters

- `n_estimators` / number of boosting rounds,
- `learning_rate` (`\eta`),
- `max_depth` (or leaf complexity),
- row/column subsampling,
- regularization controls (`reg_alpha`, `reg_lambda`, `gamma` in XGBoost).

Lower learning rates usually require more rounds but improve generalization stability.

## Tuning workflow

1. Set conservative depth (often 2-6).
2. Tune learning rate with sufficient boosting rounds.
3. Add subsampling and regularization.
4. Use early stopping on validation loss.

Always benchmark against random forests and regularized linear models to verify complexity is actually buying out-of-sample performance.
