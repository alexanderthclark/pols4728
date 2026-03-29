(causal_inference)=
# Causal Inference

Causal inference asks counterfactual questions: what would outcomes have been under different treatment assignments?

## Bootcamp objectives

- use potential-outcomes notation correctly,
- diagnose why naive treated-vs-control differences can be biased,
- understand IPW assumptions before moving to causal ML.

## Potential outcomes framework

For unit `i` with binary treatment `D_i \in \{0,1\}`:

- `Y_i(1)`: outcome under treatment,
- `Y_i(0)`: outcome under control,
- individual effect: `\tau_i = Y_i(1) - Y_i(0)`.

Observed outcome:

$$
Y_i = D_iY_i(1) + (1-D_i)Y_i(0).
$$

The core problem: we never observe both potential outcomes for the same unit.

## Why simple differences fail

A naive difference in observed means,

$$
\mathbb{E}[Y\mid D=1] - \mathbb{E}[Y\mid D=0],
$$

can differ from the ATE because of:

- selection bias in baseline outcomes,
- heterogeneous treatment effects across groups.

## Assignment mechanisms

- Randomized experiment: treatment assignment is independent of potential outcomes.
- Observational study: assignment typically depends on covariates and confounders.

For observational settings we often assume conditional ignorability:

$$
(Y(0),Y(1)) \perp\!\!\!\perp D \mid X,
$$

plus overlap:

$$
0 < e(X) = \Pr(D=1\mid X) < 1.
$$

## Inverse propensity weighting (ATE)

Under those assumptions, IPW estimates ATE via

$$
\widehat{\text{ATE}}_{\text{IPW}} =
\frac{1}{n}\sum_{i=1}^n
\left[
\frac{D_iY_i}{\hat e(X_i)} - \frac{(1-D_i)Y_i}{1-\hat e(X_i)}
\right].
$$

Intuition: reweight data to mimic a pseudo-population where treatment is as-if randomized.

## Practical cautions

- estimated propensities near 0 or 1 cause unstable, high-variance weights,
- bad propensity models produce bad causal estimates,
- diagnostics (covariate balance, weight distributions, sensitivity checks) are mandatory.

## Bridge to causal ML

Many methods in {ref}`causal_ml` target heterogeneous effects,

$$
\tau(x) = \mathbb{E}[Y(1)-Y(0)\mid X=x],
$$

rather than only a single average treatment effect.
