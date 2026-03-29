(linear_regression)=
# Linear Regression

Linear regression is the core reference model for this course: transparent, fast, and a useful baseline before adding model complexity.

## Canon objectives

- derive and interpret the OLS objective and solution,
- diagnose when coefficients are unstable versus when predictions are unstable,
- decide when linear specification quality is sufficient versus when to escalate model complexity.

## OLS objective

With outcome vector `y \in \mathbb{R}^n` and design matrix `X \in \mathbb{R}^{n\times p}`,

$$
\hat\beta = \arg\min_{\beta} \sum_{i=1}^n (y_i - x_i^\top\beta)^2
= \arg\min_{\beta}\,\|y-X\beta\|_2^2.
$$

When `X^\top X` is invertible,

$$
\hat\beta = (X^\top X)^{-1}X^\top y.
$$

```{figure} ../assets/images/linear_reg_loss_surface.pdf
:width: 95%
:align: center

Loss surface for simple linear regression; each point corresponds to one parameter vector.
```

## Geometric interpretation

OLS projects `y` onto the column space of `X`.

- `X\hat\beta` is the closest vector in that space.
- Residuals are orthogonal to every column of `X`.
- The normal equations summarize this: `X^\top(y-X\hat\beta)=0`.

This perspective explains why OLS is stable when predictors are well-conditioned and unstable when predictors are nearly redundant.

## Regression anatomy

A multivariate coefficient can be obtained via residual-on-residual regression (Frisch-Waugh-Lovell logic).

To recover the coefficient on `x_2` in `y = w_0 + w_1x_1 + w_2x_2`:

1. Regress `x_2` on `x_1`, save residuals `\tilde{x}_2`.
2. Regress `y` on `x_1`, save residuals `\tilde{y}`.
3. Regress `\tilde{y}` on `\tilde{x}_2`; slope equals `w_2`.

## Why OLS is special

Two reasons matter in practice:

- Gauss-Markov: under standard linear-model assumptions, OLS is BLUE.
- Likelihood: under Gaussian noise, OLS is the MLE for `\beta`.

These do not mean OLS is always best, but they make it the right baseline for diagnostics and comparisons.

For canonical workflow: always beat or tie linear regression out-of-sample before claiming a more complex model is justified.

## Multicollinearity as directional instability

When predictors are highly correlated, the loss surface is flat in certain directions: many coefficient vectors fit almost equally well.

```{figure} ../assets/images/loss_contours_r0.pdf
:width: 78%
:align: center

Low-correlation contours.
```

```{figure} ../assets/images/loss_contours_r94.pdf
:width: 78%
:align: center

High-correlation contours become elongated.
```

```{figure} ../assets/images/loss_contours_r100-1.pdf
:width: 78%
:align: center

Near-collinearity yields very flat optimization directions.
```

```{figure} ../assets/images/bootstrap_coef_cloud.pdf
:width: 68%
:align: center

Bootstrap coefficient cloud under multicollinearity.
```

Implication: coefficients can vary substantially across samples while prediction quality changes little.

## Feature engineering for linear models

OLS has no tuning hyperparameters once features are fixed, so specification quality depends on feature engineering.

- Standardization: comparable scales and better conditioning.
- Normalization/transforms: stabilize skewed predictors.
- One-hot encoding: represent categorical variables cleanly.
- Interactions and polynomials: capture structured nonlinearity.

## Evaluation focus

For predictive use, evaluate out-of-sample error:

$$
\mathrm{MSE}=\frac{1}{n}\sum_{i=1}^n (y_i-\hat y_i)^2.
$$

Report training and validation/test metrics separately, and inspect residual structure before claiming model adequacy.

Prediction and inference are related but distinct goals:

- for prediction, prioritize out-of-sample error and calibration;
- for inference, prioritize identification assumptions and uncertainty quantification.
