(linear_models)=
# Linear Models

## Ordinary least squares (OLS)

For outcome vector `y` and design matrix `X`, OLS solves

$$
\hat{\beta} = \arg\min_{\beta} \; \lVert y - X\beta \rVert_2^2.
$$

In scalar form, this is minimizing average squared residuals over training data.

```{figure} ../assets/images/linear_reg_loss_surface.pdf
:width: 95%
:align: center

OLS objective geometry over two coefficients.
```

## Regression anatomy

A linear predictor decomposes expected outcomes into additive components:

$$
\hat{y}_i = \hat{\beta}_0 + \sum_{j=1}^{p} \hat{\beta}_j x_{ij}.
$$

Interpretation depends on model specification, scaling, and collinearity structure.

## Multicollinearity as directional instability

When predictors are highly correlated, many nearby parameter vectors can fit similarly.
The result is unstable coefficients even when predictions remain stable.

```{figure} ../assets/images/loss_contours_r0.pdf
:width: 85%
:align: center

Low-correlation contour geometry.
```

```{figure} ../assets/images/loss_contours_r94.pdf
:width: 85%
:align: center

High-correlation contour geometry.
```

```{figure} ../assets/images/loss_contours_r100-1.pdf
:width: 85%
:align: center

Near-collinear contour geometry.
```

```{figure} ../assets/images/bootstrap_coef_cloud.pdf
:width: 70%
:align: center

Bootstrap coefficient cloud under multicollinearity.
```

## Feature engineering for linear models

- Standardization: improve numeric conditioning and coefficient comparability.
- Normalization/transforms: reduce skew and improve linear fit structure.
- One-hot encoding: represent categorical variables explicitly.
- Interactions/polynomials: represent nonlinear but structured effects.

## Regularization

Overfit-prone settings motivate penalized estimators.

```{figure} ../assets/images/Overfitted_Data.png
:width: 60%
:align: center

Illustration of overfitting risk.
```

### Ridge

$$
\hat{\beta}^{\text{ridge}} = \arg\min_{\beta}\;\lVert y-X\beta\rVert_2^2 + \lambda \lVert\beta\rVert_2^2.
$$

Ridge shrinks coefficients smoothly and is robust under collinearity.

```{figure} ../assets/images/ridge_dual.png
:width: 65%
:align: center

Ridge penalty geometry.
```

### LASSO

$$
\hat{\beta}^{\text{lasso}} = \arg\min_{\beta}\;\lVert y-X\beta\rVert_2^2 + \lambda \lVert\beta\rVert_1.
$$

LASSO encourages sparse solutions and can perform variable selection.

## Post-double LASSO (double selection)

For treatment-effect estimation with many controls:

1. LASSO outcome on controls.
2. LASSO treatment on controls.
3. Take union of selected controls.
4. Run final unpenalized regression with treatment + selected controls.

This workflow targets omitted-variable risk in high-dimensional settings.

## Logistic regression

For binary outcomes:

$$
\Pr(y_i=1\mid x_i) = \sigma(x_i^\top\beta), \quad \sigma(z)=\frac{1}{1+e^{-z}}.
$$

Use class-aware metrics under imbalance (precision, recall, PR-AUC), not only raw accuracy.
