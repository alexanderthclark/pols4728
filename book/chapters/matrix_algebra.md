(matrix_algebra)=
# Matrix Algebra

Matrix algebra is the minimal language needed for understanding optimization, regularization, embeddings, and modern ML models.

## Bootcamp objectives

- compute and interpret dot products and matrix multiplication,
- understand projection geometry behind least squares,
- connect eigendecompositions and SVD to dimensionality reduction.

## Dot products and orthogonality

For vectors `u, v \in \mathbb{R}^n`,

$$
u \cdot v = \sum_{i=1}^n u_i v_i.
$$

Two vectors are orthogonal when `u \cdot v = 0`.
For standardized data vectors, this is directly tied to zero correlation.

## Matrix-vector and matrix-matrix multiplication

If `A \in \mathbb{R}^{m\times n}` and `x \in \mathbb{R}^n`, then

$$
Ax = \begin{pmatrix}
 a_1^\top x \\
 \vdots \\
 a_m^\top x
\end{pmatrix},
$$

where each row of `A` takes a dot product with `x`.

For `A \in \mathbb{R}^{m\times n}` and `B \in \mathbb{R}^{n\times p}`,

$$
(AB)_{ij} = \sum_{k=1}^n A_{ik}B_{kj}.
$$

Shape compatibility is mandatory: inner dimensions must match.

## Projection geometry

In linear regression, fitted values are an orthogonal projection of `y` onto the column space of `X`.
The projection matrix is

$$
P_X = X(X^\top X)^{-1}X^\top,
$$

and the fitted vector is `\hat y = P_X y`.
Residuals are orthogonal to every column of `X`.

## Eigendecomposition (symmetric case)

For symmetric `A \in \mathbb{R}^{n\times n}`,

$$
A = Q\Lambda Q^\top,
$$

with orthonormal eigenvectors in `Q` and real eigenvalues in diagonal `\Lambda`.

If `A` is positive semidefinite (for example, a covariance matrix), all eigenvalues are nonnegative.
Large eigenvalues identify high-variance directions.

## Singular value decomposition (SVD)

Any `X \in \mathbb{R}^{n\times p}` admits

$$
X = U\Sigma V^\top,
$$

where `U` and `V` are orthonormal and `\Sigma` has nonnegative singular values.

Useful links:

- principal directions for PCA come from columns of `V`,
- low-rank approximation keeps top singular values,
- numerical stability in ML pipelines often depends on conditioning of `X`.

## Practical checklist

1. Always track matrix shapes before coding.
2. Standardize predictors when scales are very different.
3. Inspect rank/conditioning before trusting coefficient-level interpretation.
4. Use SVD-based routines for numerically unstable inverse problems.
