(logistic_regression)=
# Logistic Regression

Logistic regression is the default linear classifier for binary outcomes when we want calibrated probabilities and interpretable log-odds effects.

## Canon objectives

- interpret logistic coefficients and odds ratios correctly,
- tune thresholds to match substantive costs rather than defaulting to `0.5`,
- evaluate rare-event settings without being misled by raw accuracy.

## Model

$$
p_i = \Pr(y_i=1\mid x_i) = \sigma(x_i^\top w),
\qquad
\sigma(z)=\frac{1}{1+e^{-z}}.
$$

Equivalent log-odds form:

$$
\log\frac{p_i}{1-p_i}=x_i^\top w.
$$

## Estimation objective

Parameters are estimated by maximizing Bernoulli likelihood (equivalently minimizing log loss):

$$
\mathcal{L}(w) = -\sum_{i=1}^n \left[y_i\log p_i + (1-y_i)\log(1-p_i)\right].
$$

Unlike OLS, logistic regression has no closed-form estimator; we solve by numerical optimization.

## Coefficient interpretation

A one-unit increase in feature `x_j` changes log-odds by `w_j` holding other features fixed.

Odds-ratio interpretation:

$$
\exp(w_j) = \frac{\text{odds}(y=1\mid x_j+1)}{\text{odds}(y=1\mid x_j)}.
$$

Interpretability remains sensitive to feature scaling and model specification.

## Thresholds and metrics

Classification requires a probability threshold `t`.

- Higher `t`: fewer positives, usually higher precision and lower recall.
- Lower `t`: more positives, usually higher recall and lower precision.

With class imbalance, raw accuracy can be misleading. Prioritize precision-recall tradeoffs and PR-AUC.
Treat threshold choice as a policy decision, not a fixed model property.

```{figure} ../assets/scatter_precision_recall_f1.svg
:width: 80%
:align: center

Precision, recall, and F1 tradeoffs for binary evaluation.
```

## Imbalance and sampling

For rare events:

- Keep test prevalence realistic.
- Use stratified splits for training and validation.
- Consider threshold tuning and class weighting.
- If training data are deliberately rebalanced, check calibration on natural-prevalence validation data.

## Regularized logistic regression

Penalty terms improve stability in high-dimensional settings.

- L2 penalty (ridge logistic): stable under collinearity.
- L1 penalty (lasso logistic): sparse selection.

Tune penalty strength via cross-validation in a pipeline that includes preprocessing.

## Production checklist

- Evaluate discrimination and calibration.
- Inspect confusion matrices at operational thresholds.
- Report prevalence-aware metrics.
- Validate probability reliability before deployment decisions.
- Compare against regularized linear and tree baselines before escalating complexity.
