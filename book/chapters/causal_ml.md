(causal_ml)=
# Causal ML

Causal ML combines causal identification goals with flexible machine-learning estimators to study treatment-effect heterogeneity.

## Topic objectives

- distinguish prediction from causal effect estimation,
- understand meta-learners, DML, and causal forests at a practical level,
- report uncertainty and assumptions transparently in applied work.

## Motivation

Many social-science questions are counterfactual:

- how outcomes would change under treatment,
- which subgroups benefit more or less,
- whether average effects hide meaningful heterogeneity.

This builds directly on the assumptions in {ref}`causal_inference`.

## Meta-learners

Meta-learners wrap supervised models to estimate CATE `\tau(x)`.

- S-learner: one model with treatment indicator as a feature,
- T-learner: separate models for treated and control outcomes,
- X-learner: combines imputed effects and propensity weighting.

Choice depends on sample size balance and heterogeneity structure.

## Double machine learning (DML)

DML estimates treatment effects while using ML for nuisance components.
A core orthogonal-score form is

$$
\psi_i(\alpha,\eta)=\big(y_i-m(X_i)-\alpha(d_i-g(X_i))\big)(d_i-g(X_i)),
$$

where nuisance functions `m(\cdot)` and `g(\cdot)` are estimated via ML and cross-fitting.

Orthogonality reduces first-order sensitivity to nuisance estimation error.

## Causal trees and causal forests

Unlike predictive trees in {ref}`decision_trees`, causal trees split to expose treatment-effect heterogeneity.

Causal forests aggregate many such trees and use honesty/sample-splitting ideas to improve stability and inference.
They are a robust default when heterogeneous effects are expected and nonlinear interactions are plausible.

## Practical workflow

1. Define estimand (ATE, ATT, CATE) before modeling.
2. Check overlap and treatment-assignment diagnostics.
3. Use cross-fitting or honesty-aware estimators when possible.
4. Validate heterogeneity patterns with robustness checks.
5. Report assumptions, uncertainty intervals, and failure modes.

## Common failure modes

- weak overlap (extreme propensities),
- high-variance subgroup estimates,
- over-interpretation of noisy CATE surfaces,
- conflating predictive fit with causal validity.
