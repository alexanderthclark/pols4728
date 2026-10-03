# Research data and original regressions

These four datasets contain complete analysis samples with the original variables
already prepared. No merging, Stata files, or panel reshaping is required. The
accompanying Python and R scripts reproduce only the listed original regressions.

## Choose a dataset

| Dataset | One row represents | Outcome `y` | Exposure `d` |
| --- | --- | --- | --- |
| Plough | A country; 177 rows | Female labor-force participation in 2000, in percent | Ancestral plough-use share, 0–1 |
| Slavery | A county; 1,152 rows | Share of white survey respondents identifying as Democrats, including leaners | Enslaved population share in 1860, 0–1 |
| Floods | An electoral district; 299 rows | SPD vote-share change from 1998 to 2002, in percentage points | Flood-affected district, 0 or 1 |
| Fox News | A town; 9,256 rows | Republican two-party vote-share change from 1996 to 2000, as a proportion | Fox News available on local cable in 2000, 0 or 1 |

Plough has the simplest design: five controls and an intercept. Slavery and Fox
News also use supplied weights and indicator controls. Floods has already been
expressed as one change per district.

| Dataset | Data | Documentation | Original paper and data |
| --- | --- | --- | --- |
| Plough | [CSV](plough/data.csv) · [Download](https://raw.githubusercontent.com/alexanderthclark/pols4728/main/data/regression_choices/plough/data.csv) | [Codebook](plough/codebook.csv) · [Metadata](plough/metadata.json) | [Alesina, Giuliano, and Nunn (2013)](https://doi.org/10.1093/qje/qjt005) · [Author data page](https://nathannunn.arts.ubc.ca/data/) |
| Slavery | [CSV](slavery/data.csv) · [Download](https://raw.githubusercontent.com/alexanderthclark/pols4728/main/data/regression_choices/slavery/data.csv) | [Codebook](slavery/codebook.csv) · [Metadata](slavery/metadata.json) | [Acharya, Blackwell, and Sen (2016)](https://doi.org/10.1086/686631) · [Replication archive](https://doi.org/10.7910/DVN/CAEEG7) |
| Floods | [CSV](floods/data.csv) · [Download](https://raw.githubusercontent.com/alexanderthclark/pols4728/main/data/regression_choices/floods/data.csv) | [Codebook](floods/codebook.csv) · [Metadata](floods/metadata.json) | [Bechtel and Hainmueller (2011)](https://www.mit.edu/~jhainm/Paper/elbe.pdf) · [Replication archive](https://www.mit.edu/~jhainm/Replication/Elbe.zip) |
| Fox News | [CSV](fox_news/data.csv) · [Download](https://raw.githubusercontent.com/alexanderthclark/pols4728/main/data/regression_choices/fox_news/data.csv) | [Codebook](fox_news/codebook.csv) · [Metadata](fox_news/metadata.json) | [DellaVigna and Kaplan (2007)](https://eml.berkeley.edu/~sdellavi/wp/FoxVoteQJEAug07.pdf) · [Author data page](https://eml.berkeley.edu/~sdellavi/data/foxnewsdata.shtml) |

## Read the data

Each folder contains a CSV, a codebook describing every column, and metadata
recording the original specification, preparation, source URLs, and checksums.
All supplied regression variables are complete; no imputation was performed.

| Column name | Meaning |
| --- | --- |
| `y` | Outcome, on the original study's scale |
| `d` | Exposure whose coefficient is reported |
| `x_...` | Original substantive controls |
| `fixed_...` | Prepared state or congressional-district indicators; include an intercept with them |
| `weight` | Original regression weight, where used |
| `cluster_id` | Cable-system identifier used for Fox News standard errors |
| `id_...` | Observation identifiers, excluded from the regressions |
| `aux_...` | Additional flood covariate levels from the two source election rows, excluded from the original change regression |

The metadata's `controls` and `fixed_controls` lists identify the original
regressors. Do not use every CSV column as a predictor: weights and identifiers
have separate roles, and the auxiliary flood levels are not original regressors
in the change specification.

For example, load the Plough CSV from the repository root with:

```python
import pandas as pd
data = pd.read_csv("data/regression_choices/plough/data.csv")
```

```r
data <- read.csv("data/regression_choices/plough/data.csv")
```

The Slavery and Fox News outcomes are proportions: multiply a predicted outcome
change by 100 to express it in percentage points. Plough and Floods outcomes
already use percentage units. Slavery and Plough exposures are continuous shares.

## Reproduce the original regression

Download or clone the repository and retain its folder layout. Run from the
repository root. The scripts accept only a dataset name and use all of its
original controls, weights, and standard-error settings.

Python:

```sh
python -m pip install -r examples/regression_choices/requirements.txt
python examples/regression_choices/baseline.py plough
```

R requires `jsonlite` to read the metadata. Install it once, then run:

```sh
Rscript -e 'install.packages("jsonlite", repos="https://cloud.r-project.org")'
Rscript examples/regression_choices/baseline.R plough
```

Replace `plough` with `slavery`, `floods`, or `fox_news`. The scripts report the
coefficient, standard error, confidence interval, and p-value for `d`.

| Dataset | Original specification | Controls | Coefficient on `d` | Original standard error |
| --- | --- | ---: | ---: | ---: |
| Plough | Table III, column 1 | 5 | −14.894597 | 3.318042 |
| Slavery | Table 1, column 2 | 14, plus 13 state indicators | −0.126591 | 0.042504 |
| Floods | Table 1, column 3, expressed in changes | 10 | 6.912640 | 0.571000 |
| Fox News | Table IV, column 4 | 42, plus 234 district indicators | 0.004212 | 0.001541 |

Plough uses the country-level labor-force-participation specification with HC1
standard errors. Its separate individual-level World Values Survey analysis is
not included.

Slavery uses the authors' county weights, state indicators, and classical
weighted-OLS standard errors. The stored weights are sums of individual survey
weights, not rounded respondent counts.

Floods expresses the original two-election model as one change per district.
The script applies the authors' finite-sample variance correction and 298
inference degrees of freedom, preserving their coefficient and standard error.
Specifically, its HC0 covariance is multiplied by
`G/(G-1) * (2*G-1)/(2*G-q-1)`, where `G=299` districts and `q=12` counts the
change regression's intercept, exposure, and ten controls. The source panel has
598 rows; the supplied CSV has 299. A causal interpretation still requires an
appropriate comparison of trends in affected and unaffected districts.

Fox News uses the authors' 1996 vote weights, congressional-district indicators,
and standard errors clustered by 2,992 cable systems. Its outcome and all 42
substantive controls retain the original prepared values.

## Post-double-selection: R and Python guidance

Post-double-selection (PDS) uses two selection regressions and one unpenalized
regression. Given an outcome `y`, an exposure `d`, and eligible controls `X`:

1. Use lasso to select controls that predict `y` from `X`.
2. Separately use lasso to select controls that predict `d` from the same `X`.
3. Regress `y` on `d` and the **union** of the two selected control sets, with an
   intercept. Always retain `d`; report its coefficient from this final regression.

Selecting controls only from the outcome regression is not double selection.
The treatment-prediction step helps retain potential confounders that a procedure
focused only on predicting the outcome might omit. See
[Belloni, Chernozhukov, and Hansen (2014)](https://doi.org/10.1093/restud/rdt044).

### Which libraries to use

| Library | Appropriate use |
| --- | --- |
| R: [`hdm`](https://search.r-project.org/CRAN/refmans/hdm/html/rlassoEffects.html) | Recommended starting point for Belloni-style PDS. `rlassoEffect(..., method="double selection")` implements selection and effect inference with theoretically calibrated lasso penalties. |
| R: [`glmnet`](https://glmnet.stanford.edu/articles/glmnet.html) | An established lasso engine for implementing the selection steps yourself. Use `alpha=1` for lasso; `lambda` controls penalty strength. `cv.glmnet()` chooses a prediction-oriented penalty; it does not perform PDS or treatment-effect inference by itself. |
| Python: call R's `hdm` through `Rscript` | Recommended here for using the same Belloni-style PDS implementation from a Python workflow. Requires both Python and R; the synthetic example below shows the connection. |
| Python: [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html) | A reliable lasso solver, but not a complete implementation of Belloni-style PDS. It does not estimate the required penalty loadings or calibrate the penalty for you. Cross-validation and predictor standardization are not substitutes for those steps. |
| R/Python: [`DoubleML`](https://docs.doubleml.org/stable/guide/models.html) | Implements double/debiased machine learning. In particular, `DoubleMLPLR` with the partialling-out score uses nuisance predictions and cross-fitting; it is a related approach, not the union-and-refit PDS estimator above. |

Trusting a library's numerical implementation is different from establishing
valid inference for a particular design. PDS inference needs assumptions such as
approximate sparsity (a relatively small set of features adequately captures the
relevant relationships), adequate residual variation in `d`, and an appropriate
sampling model. A causal interpretation also needs the study's identification
assumptions; lasso cannot resolve reverse causality or unobserved confounding.

### R: a synthetic example with `hdm`

Install `hdm` once with `install.packages("hdm")`. This example generates independent,
unweighted observations with 40 candidate controls and a known effect of 1. It
does not read any of the course datasets.

```r
library(hdm)
set.seed(4728)
n <- 600
p <- 40
X <- matrix(rnorm(n * p), nrow = n, ncol = p)
colnames(X) <- paste0("x", seq_len(p))
d <- 0.7 * X[, 1] - 0.5 * X[, 2] + rnorm(n)
y <- 1.0 * d + 0.8 * X[, 1] + 0.6 * X[, 3] + rnorm(n)

# X contains candidate controls only: no d, y, or intercept column.
fit <- rlassoEffect(
  x = X, y = y, d = d,
  method = "double selection", post = TRUE
)
summary(fit)
confint(fit, level = 0.95)
```

The default [`rlasso` penalty](https://search.r-project.org/CRAN/refmans/hdm/html/rlasso.html)
allows heteroskedastic errors. `method="double selection"` requests PDS;
`post=TRUE` alone does not. No candidate control is forced into this toy model.
`hdm` provides its own standard errors and confidence intervals; they need not
match a separate OLS routine's finite-sample correction.

### Python: use the same `hdm` estimator from Python

For this exercise, use R's `hdm` for the PDS estimation even if you prepare your
variables in Python. Install R and `hdm` as above, ensure `Rscript` is available
on your system's command path, and install NumPy with `python -m pip install numpy`.
This example uses Python to generate synthetic data and calls `hdm` in R to fit
the model. It is a Python workflow with an R dependency, not a native Python PDS
package.

```python
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np

rscript = shutil.which("Rscript")
if rscript is None:
    raise RuntimeError("Install R and make Rscript available on your PATH.")

rng = np.random.default_rng(4728)
n, p = 600, 40
X = rng.normal(size=(n, p))
d = 0.7 * X[:, 0] - 0.5 * X[:, 1] + rng.normal(size=n)
y = 1.0 * d + 0.8 * X[:, 0] + 0.6 * X[:, 2] + rng.normal(size=n)

r_code = """
if (!requireNamespace("hdm", quietly = TRUE)) {
  stop('Install hdm in this R installation with install.packages("hdm").')
}
toy <- read.csv(commandArgs(trailingOnly = TRUE)[1])
X <- as.matrix(toy[, grep("^x", names(toy)), drop = FALSE])
fit <- hdm::rlassoEffect(
  x = X, y = toy$y, d = toy$d,
  method = "double selection", post = TRUE
)
print(summary(fit))
print(confint(fit, level = 0.95))
"""

with tempfile.TemporaryDirectory() as folder:
    csv_path = Path(folder) / "synthetic.csv"
    np.savetxt(
        csv_path, np.column_stack([y, d, X]), delimiter=",",
        header="y,d," + ",".join(f"x{j + 1}" for j in range(p)), comments="",
    )
    result = subprocess.run(
        [rscript, "-e", r_code, str(csv_path)],
        check=True, capture_output=True, text=True,
    )
    print(result.stdout)
```

Both examples delegate selection, penalty calibration, and effect inference to
`hdm`. R and NumPy generate different random samples, so their results need not
coincide. Both examples assume independent, unweighted observations without fixed
effects; the design qualifications below still apply.

**Why not just use `LassoCV`?** Belloni-style heteroskedastic lasso uses a separate
penalty loading for each candidate regressor, estimated using that regressor and
residuals from the relevant selection equation. These loadings are updated as
residual estimates improve, separately for the outcome and exposure equations.
[`hdm::rlasso`](https://search.r-project.org/CRAN/refmans/hdm/html/rlasso.html)
implements this procedure. Scaling each predictor to unit variance does not
account for how its values interact with the error variance, and cross-validation
chooses a prediction-oriented penalty rather than this calibration.

Scikit-learn's `Lasso` exposes a scalar `alpha`, not a vector of penalty loadings.
For given positive loadings, dividing each predictor by its loading can translate
the problem into ordinary lasso. However, the loading estimates, iterations,
penalty normalization, and design adjustments must still be implemented and
validated. That is additional estimator development, beyond using a lasso solver.
Two cross-validated selections followed by OLS demonstrate the union-and-refit
mechanics, but do not reproduce `hdm`'s calibration. Adding HC1 standard errors
does not supply the missing steps.

### Applying the idea to a research design

- **Define eligible controls before selecting.** Use the codebook and the paper's
  causal argument. Lasso cannot decide whether a variable is a confounder, a
  mediator, or an inappropriate control. Consider squares or interactions only
  for substantively appropriate variables, and define the candidate feature set
  before examining which specification gives a preferred result.
- **Distinguish selection from forced inclusion.** Original substantive controls
  can be candidates for selection. Keeping all of them fixed and selecting only
  extra features is a different specification and should be described that way.
  Design-required indicators should remain in the model and be accounted for
  during both selection steps; they are not interchangeable with optional controls.
- **Preserve the design throughout.** These toy examples assume independent,
  unweighted observations and no fixed effects. They cannot be applied unchanged
  to a weighted or clustered design. Weights and required indicators must be
  handled consistently in selection and refitting; dependence requires suitable
  penalty/tuning choices and clustered inference. Adding clustered standard
  errors only at the end does not by itself address the selection stage.
- **Keep comparisons interpretable.** Use the same observations, outcome scale,
  and exposure definition as the original regression. Record the candidate set,
  forced controls, penalty rule, seed, package versions, and both selected sets.
  Address convergence warnings and ensure the final regression has full column
  rank and enough residual degrees of freedom. Report coefficient changes and
  uncertainty, rather than treating a change in significance as the sole finding.

## Sources and scope

These are selected regressions from the papers, not every analysis in them.
Mistrust and Potato are omitted because their two-way clustering and multi-period
panel design require additional methods. Reproducing an original regression
does not establish its causal assumptions; consult each paper for its argument.

Source URLs, transformations, and file checksums are retained in the metadata.
Cite the original authors and replication materials. Rights remain with the
original rights holders; no new license over the source data is asserted here.
