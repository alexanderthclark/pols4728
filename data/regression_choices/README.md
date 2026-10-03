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

## Sources and scope

These are selected regressions from the papers, not every analysis in them.
Mistrust and Potato are omitted because their two-way clustering and multi-period
panel design require additional methods. Reproducing an original regression
does not establish its causal assumptions; consult each paper for its argument.

Source URLs, transformations, and file checksums are retained in the metadata.
Cite the original authors and replication materials. Rights remain with the
original rights holders; no new license over the source data is asserted here.
