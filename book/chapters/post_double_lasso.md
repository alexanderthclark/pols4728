(post_double_lasso)=
# Post-Double Lasso

Post-double lasso (double selection) is a high-dimensional inference workflow for estimating a target treatment effect while controlling many potential confounders.

## Problem setup

Partially linear model:

$$
y_i = \alpha_0 d_i + x_i^\top\beta_0 + \zeta_i,\qquad
d_i = x_i^\top\pi_0 + v_i.
$$

Goal: estimate `\alpha_0` when the control vector `x_i` can be large relative to sample size.

## Double-selection algorithm

1. Outcome lasso: regress `y` on `X`; keep selected controls `\hat S_y`.
2. Treatment lasso: regress `d` on `X`; keep selected controls `\hat S_d`.
3. Union set: `\hat S = \hat S_y \cup \hat S_d`.
4. Final OLS: regress `y` on `d` and `X_{\hat S}` (no penalty on `d`).

The coefficient on `d` from step 4 is the post-double-lasso estimate.

## Why this works

Single-stage lasso can drop controls important for treatment assignment but weak for outcome prediction (or vice versa). The union step protects against this omitted-variable pathway.

Orthogonal-score view:

$$
\psi_i(\alpha,\beta,\pi)=
(y_i-\alpha d_i-x_i^\top\beta)\,(d_i-x_i^\top\pi).
$$

Near-orthogonality reduces first-order sensitivity to nuisance-estimation errors.

## Practical conditions

- Approximate sparsity: only a subset of controls carry signal.
- Reasonable design geometry (restricted eigenvalue/compatibility).
- Penalty scaling aligned with `\sqrt{n\log p}` behavior.

These are substantive assumptions, not automatic consequences of running lasso.

## Implementation notes

- Standardize controls before lasso.
- Never penalize the treatment variable of interest.
- Use robust standard errors in the final OLS.
- If multiple target regressors exist (for example treatment interactions), run the treatment-selection step for each and union all selected controls.

```python
# Pseudocode sketch
S_y = lasso_select(X, y)
S_d = lasso_select(X, d)
S = sorted(set(S_y) | set(S_d))

X_final = X[:, S]
alpha_hat = ols_with_robust_se(y, d, X_final)
```

## Failure modes

- Weak or dense signal: selected sets become unstable.
- Strong collinearity: selection can vary across folds/samples.
- Overly aggressive tuning: omitted controls bias final treatment estimate.

Use sensitivity checks on the selected set and compare with domain-driven control specifications.
