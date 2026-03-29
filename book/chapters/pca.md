(pca)=
# Principal Components Analysis

Principal Components Analysis (PCA) creates orthogonal linear combinations of features that capture maximal variance.

## Why use PCA

- Reduce dimensionality before modeling.
- Mitigate multicollinearity.
- Build compact representations for visualization and downstream prediction.

PCA is unsupervised: it does not use the target variable directly.

## Construction

Given centered feature matrix `X`, PCA finds directions `v_1, v_2, ...` such that

$$
v_1 = \arg\max_{\|v\|=1} \mathrm{Var}(Xv),
$$

and each subsequent component maximizes remaining variance subject to orthogonality constraints.

Equivalent computational route: eigendecomposition of covariance matrix or SVD of `X`.

## Scores and explained variance

- Component loadings: directions in original feature space.
- Scores: transformed coordinates `Z = XV`.
- Explained variance ratio: variance share captured by each component.

Choose number of components by cumulative explained variance and validation performance in downstream tasks.

## Workflow

1. Standardize features when scales differ.
2. Fit PCA on training data only.
3. Transform validation/test with fitted PCA.
4. Evaluate downstream model with and without PCA.

## Example (sklearn)

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X = df[predictor_columns]
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
Z = pca.fit_transform(X_scaled)

print(pca.explained_variance_ratio_)
```

## Cautions

- High explained variance does not guarantee predictive usefulness.
- Components can be hard to interpret substantively.
- If interpretability is primary, compare PCA against transparent feature-engineering alternatives.
