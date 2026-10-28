"""Compare SAGE with refitted ablation for two identical features."""

import numpy as np
import sage

# The population has two equally likely points: X1 = X2 = Y = -1 or +1.
background = np.array([[-1.0, -1.0], [1.0, 1.0]])
X = background.copy()
y = X[:, 0]
beta = np.linalg.lstsq(X, y, rcond=None)[0]  # Minimum-norm coefficients.


def predict(x):
    return x @ beta


def mse(prediction, target):
    return np.mean((prediction - target) ** 2)


# Conditional predictions for {}, {1}, {2}, and {1, 2}, respectively.
# Observing either duplicate reveals the other, so singleton loss is zero.
conditional_predictions = np.column_stack(
    [np.full_like(y, y.mean()), y, y, predict(X)]
)
r0, r1, r2, r12 = np.mean(
    (y[:, None] - conditional_predictions) ** 2, axis=0
)
conditional_sage = 0.5 * np.array(
    [r0 - r1 + r2 - r12, r0 - r2 + r1 - r12]
)

# This package imputer is MARGINAL: singleton loss is 0.25, not zero.
# It gives the same SAGE values here because the fitted model is symmetric.
imputer = sage.MarginalImputer(predict, background)
estimator = sage.PermutationEstimator(imputer, "mse")
result = estimator(
    X, y, batch_size=512, n_permutations=16384,
    detect_convergence=False, bar=False,
)

full_loss = mse(predict(X), y)
ablation = np.empty(2)
for j in range(2):
    reduced_X = np.delete(X, j, axis=1)
    reduced_beta = np.linalg.lstsq(reduced_X, y, rcond=None)[0]
    ablation[j] = mse(reduced_X @ reduced_beta, y) - full_loss

print("Minimum-norm coefficients:", beta)
print("Exact conditional SAGE:", conditional_sage)
print("Package marginal SAGE:", result.values)
print("Monte Carlo standard errors:", result.std)
print("Refitted drop-column ablation:", ablation)
