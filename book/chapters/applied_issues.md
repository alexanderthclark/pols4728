(applied_issues)=
# Applied Issues

This lecture focuses on common failure modes in applied ML workflows: class imbalance, leakage, and misuse of interpretation tools.

## Reproducibility risk in applied prediction

High model complexity does not protect against simple workflow mistakes. Most catastrophic errors in published ML applications come from data handling, validation design, or metric mismatch.

## Imbalanced outcomes

With rare positive labels, naive accuracy can look excellent even when the model is useless.

Options for training under imbalance:

- downsample majority class,
- upsample minority class,
- synthetic sampling,
- class weighting,
- threshold optimization on validation data.

Critical rule: keep the test set at natural prevalence.

## CV with resampling: correct pattern

1. Split off test set first (natural class frequencies).
2. Inside training folds only, apply resampling/weighting.
3. Tune hyperparameters and threshold on validation folds.
4. Refit chosen pipeline on full training data.
5. Evaluate once on untouched test data.

Resampling before splitting leaks label prevalence information and invalidates performance claims.

## Logistic regression under imbalance

If the log-odds model is correctly specified and data are ample, logistic regression can still perform well on imbalanced data.

Practical focus:

- choose threshold for decision costs,
- inspect calibration,
- evaluate precision-recall behavior,
- use class weighting when needed for training stability.

## Tree ensembles under imbalance

Random forests and boosted trees may still ignore rare classes if objective/threshold setup is naive. For these models, tune both class handling and decision threshold rather than relying on defaults.

## Partial dependence plots (PDP) in practice

PDPs summarize average model response to feature changes:

$$
\widehat{\mathrm{PD}}_S(x_S)=\frac{1}{n}\sum_{i=1}^n \hat f(x_S, x_{-S}^{(i)}).
$$

Useful for global shape diagnostics, but easy to misread when interactions are strong or when counterfactual feature combinations are implausible.

## Production checklist

- Validate preprocessing and resampling inside CV only.
- Align metrics with substantive decision costs.
- Keep held-out test prevalence realistic.
- Pair global plots (PDP) with local diagnostics (ICE/SHAP).
- Document every modeling decision for replication.
