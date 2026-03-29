(interpretability)=
# Interpretability

## Why interpretability matters

Predictive performance alone is often insufficient in social science and policy applications.
Interpretability helps with:

- debugging models,
- communicating mechanisms,
- auditing fairness and reliability.

## Global vs local explanation

- Global explanations summarize average model behavior.
- Local explanations explain individual predictions.

Both are useful and can disagree when effects are heterogeneous.

## Partial dependence and ICE

For feature `x_j`, partial dependence averages model predictions over the empirical distribution
of other features:

$$
\text{PDP}_j(z)=\mathbb{E}_{X_{-j}}[\hat{f}(z, X_{-j})].
$$

ICE curves show per-observation trajectories rather than the average, revealing heterogeneity
hidden by PDP summaries.

## Shapley values

Shapley attribution distributes prediction contributions across features via coalition averages.
For feature `j`:

$$
\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F|-|S|-1)!}{|F|!}
\left[v(S \cup \{j\}) - v(S)\right].
$$

Key properties include efficiency, symmetry, and consistency under common definitions.

## SHAP in practice

SHAP is a practical approximation framework for Shapley-style attributions.

Use SHAP carefully:

- correlated features can redistribute attribution in unintuitive ways,
- explanations are conditional on model + background data,
- explanations are not causal effects.

## Ablation and SAGE

Ablation removes features (or groups) and measures performance change.
SAGE generalizes this idea with Shapley-style accounting of global loss contributions.

## Workflow

1. Evaluate predictive performance first.
2. Use PDP/ICE for shape diagnostics.
3. Use SHAP/SAGE for attribution summaries.
4. Validate interpretation claims with robustness checks and domain knowledge.
