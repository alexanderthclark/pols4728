(interpretability)=
# Interpretability

Interpretability asks why a model predicts what it predicts and which features matter for performance.

## Canon objectives

- distinguish global explanations from local explanations,
- use PDP/ICE/SHAP/ablation as complementary tools,
- avoid treating explanation outputs as causal effects.

## Global versus local explanation

- Global methods summarize average behavior across the dataset.
- Local methods explain one prediction at a time.

Both are necessary: global summaries can hide heterogeneity, while local explanations do not automatically generalize.

## Partial dependence

For feature set `S`, partial dependence is

$$
\mathrm{PD}_S(x_S)=\mathbb{E}_{X_{-S}}\big[\hat f(x_S,X_{-S})\big].
$$

Empirical estimator:

$$
\widehat{\mathrm{PD}}_S(x_S)=\frac{1}{n}\sum_{i=1}^n \hat f(x_S,x_{-S}^{(i)}).
$$

PDP is useful for shape diagnostics but can mislead when interactions are strong or feature combinations are unrealistic.

## ICE plots

Individual Conditional Expectation (ICE) plots keep observations separate:

$$
\mathrm{ICE}_{j,i}(v)=\hat f(v, x_{i,-j}).
$$

PDP is the average of ICE curves. Non-parallel ICE curves signal interaction effects.

## Shapley values

Shapley attribution distributes prediction contribution across features via coalition averages.

$$
\phi_j = \sum_{S\subseteq F\setminus\{j\}}
\frac{|S|!(|F|-|S|-1)!}{|F|!}
\left[v(S\cup\{j\})-v(S)\right].
$$

Desirable properties include efficiency and symmetry.

## SHAP

SHAP operationalizes Shapley-style explanations for ML models.

Local additive decomposition:

$$
\hat f(x)=\phi_0+\sum_{j=1}^p \phi_j(x),
$$

where `\phi_0` is a baseline prediction and `\phi_j(x)` is feature `j`'s contribution for observation `x`.

Important caveat: SHAP values depend on the background distribution and feature dependence assumptions.

## Ablation and SAGE

Ablation estimates global importance by retraining without a feature and measuring performance loss:

$$
\Delta_j = L(\hat f_{-j}) - L(\hat f).
$$

SAGE extends this with Shapley-style accounting over feature subsets for global loss contributions.

## Interpretation protocol

1. Validate predictive performance first.
2. Use PDP/ICE to inspect response shape and heterogeneity.
3. Use SHAP/ablation for contribution summaries.
4. Stress-test conclusions under correlated features and alternative preprocessing.

Interpretability methods explain model behavior, not causal mechanisms by themselves.
