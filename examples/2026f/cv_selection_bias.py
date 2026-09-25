import numpy as np

np.random.seed(1)
N, K, repetitions = 100, 5, 10_000
cv_errors = np.empty((repetitions, 2))

for r in range(repetitions):
    x = np.random.normal(size=N)
    e = np.random.normal(size=N)
    y = 2 * x + e
    for j, slope in enumerate([3, 1]):
        squared_errors = (y - slope * x)**2
        folds = squared_errors.reshape(K, N // K)
        fold_errors = folds.mean(axis=1)
        cv_errors[r, j] = fold_errors.mean()

print(cv_errors.mean(axis=0))
print(cv_errors.min(axis=1).mean())
