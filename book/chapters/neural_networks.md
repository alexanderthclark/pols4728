(neural_networks)=
# Neural Networks

This lecture covers feed-forward neural networks for tabular prediction: what functions they represent, how they are trained, and where they fit in practice.

## Architecture basics

A neural network composes affine maps and nonlinear activation functions.

For one hidden layer:

$$
\hat y = \phi_0 + \sum_{j=1}^H \phi_j\,a(\theta_{j0}+\theta_j^\top x),
$$

where `a(\cdot)` is an activation (for example ReLU).

```{figure} ../assets/diagrams/deep_network_architecture.svg
:width: 88%
:align: center

Shallow and deep feed-forward network structure.
```

## Shallow vs deep

- Shallow networks can approximate complex functions with enough width.
- Deep networks often represent similar functions more parameter-efficiently through composition.

Depth changes the inductive bias, not just capacity.

## Training with gradient methods

Parameters are learned by minimizing a loss with gradient-based updates:

$$
\theta \leftarrow \theta - \alpha\nabla_\theta L.
$$

In practice, stochastic mini-batch methods dominate:

- SGD: noisy gradients, useful implicit regularization.
- Adam: momentum + adaptive step sizes, robust default for many tasks.

## Regularization and stability

Important controls:

- early stopping,
- weight decay (`alpha` / L2 penalty),
- architecture size (depth/width),
- learning-rate schedule.

Networks can overfit quickly when capacity is large relative to sample size.

## Sklearn MLP usage

`MLPRegressor` / `MLPClassifier` provide a practical entry point for tabular experiments.

Key hyperparameters:

- `hidden_layer_sizes`,
- `alpha`,
- `learning_rate_init`,
- `max_iter`,
- `early_stopping`.

Tune these with cross-validation exactly as for other models.

## Double descent and overparameterization

In modern regimes, test error can show non-classical behavior (double descent): error rises near interpolation, then decreases again as capacity grows further.

Treat this as an empirical pattern to validate, not an excuse to skip model selection discipline.

## Tabular-data reality check

For medium-sized structured datasets, tree ensembles often outperform neural nets out-of-the-box. Neural networks can still win with careful tuning and richer representation learning goals.

Use them when:

- you need learned representations,
- feature interactions are very complex,
- dataset scale justifies tuning overhead.
