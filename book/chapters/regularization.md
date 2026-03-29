(regularization)=
# Regularization

Regularization constrains model flexibility so prediction error on new data does not explode when models are highly parameterized or predictors are collinear.

## Motivation

Flexible models can fit noise. Penalization provides a middle path between underfit and overfit.

```{figure} ../assets/images/Overfitted_Data.png
:width: 58%
:align: center

Illustration of overfitting with an overly flexible curve.
```

## Penalized objectives

For linear model `y \approx Xw`:

- OLS: `\min_w \|y-Xw\|_2^2`
- Ridge (L2): `\min_w \|y-Xw\|_2^2 + \lambda\|w\|_2^2`
- LASSO (L1): `\min_w \|y-Xw\|_2^2 + \lambda\|w\|_1`

`\lambda` is a hyperparameter selected with validation.

## Ridge regression

Ridge has closed-form solution:

$$
w_{\text{ridge}}=(X^\top X + \lambda I)^{-1}X^\top y.
$$

Equivalent constrained view:

$$
\min_w\ \|y-Xw\|_2^2\quad\text{s.t.}\quad\|w\|_2^2\le \tau.
$$

```{figure} ../assets/images/ridge_dual.png
:width: 68%
:align: center

Ridge dual geometry: circular L2 constraint intersects loss contours.
```

Key behavior:

- Shrinks all coefficients continuously toward zero.
- Usually keeps correlated predictors together.
- Strong baseline when signal is dense and multicollinearity is high.

## LASSO

LASSO solves

$$
\min_w\ \|y-Xw\|_2^2 + \lambda\|w\|_1.
$$

Equivalent constrained form uses an L1 ball, whose corners encourage exact zeros.

In orthonormal settings, LASSO is soft-thresholding:

$$
\hat w_j = \mathrm{sign}(z_j)\,(|z_j|-\kappa)_+,
$$

with `z_j` the OLS-style score and threshold `\kappa` tied to `\lambda`.

Key behavior:

- Performs variable selection through sparsity.
- Can be unstable with strongly correlated predictors.
- Often paired with post-selection OLS for less shrinkage bias.

## Choosing between ridge and lasso

- Use ridge when many predictors carry weak-to-moderate signal.
- Use lasso when sparse structure is plausible and selection is useful.
- Use elastic net when correlated groups should enter together.

## Tuning and implementation details

1. Standardize predictors before tuning (`X` columns on comparable scales).
2. Leave intercept unpenalized.
3. Choose `\lambda` with cross-validation.
4. Refit final model on full training data with chosen `\lambda`.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso

ridge = Pipeline([
    ("scale", StandardScaler()),
    ("model", Ridge(alpha=1.0)),
])

lasso = Pipeline([
    ("scale", StandardScaler()),
    ("model", Lasso(alpha=0.05)),
])
```

## Inference caveat

Regularization improves prediction stability but changes the sampling behavior of coefficients. If the objective is causal inference, use dedicated post-selection workflows (for example {ref}`post_double_lasso`).
