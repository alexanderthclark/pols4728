# POLS4728 data and code

Public data and code archive for POLS4728. The CSV files remain at their original paths so existing download links continue to work.

## Course design

The [course design guide](design/README.md) explains the stylistic choices and
standards that give course materials a consistent design language. The
[visual reference](design/guide.pdf), [palette and typography
tokens](design/tokens.json), and [public artwork registry](design/assets.json)
document the Fall 2026 identity, typography, palette, and approved artwork.

## Python examples

The [Fall 2026 Python examples](examples/2026f/README.md) accompany the code listings in the lecture notes. The examples README explains dependencies, how to run each script, and the external data needed for the ATUS example.

## Interactive explanations

The [Shapley values story](docs/shapley/README.md) explains majority voting through a gradually built Hasse diagram. An optional six-path view shows where the Shapley weights come from. Its standalone website source is in `docs/shapley/`.

The [local SHAP story](docs/shap/README.md) begins with the Shapley formula and
connects its terms to one regression prediction by revealing a person's features in background rows, averaging the fixed model's
predictions, comparing those averages, and assembling the final waterfall.
An illustrative earnings equation makes ability and neighborhood opportunity
interact, including an adverse neighborhood with negative opportunity.
[Python reproduction files](examples/shap/README.md) include the fixed-model
calculation and checks against the SHAP package.

The [alien basketball story](docs/alien/README.md) follows one alien learning a cutoff from five games. It then keeps 1,000 independently trained predictions visible as they gather into stacks, locating expectation, bias, variance, and mean squared error at a fixed test input. A proportional error bar shows the decomposition, with zero outcome noise.

GitHub Pages publishes the `docs/` site automatically after the checks pass on `main`. See the [hosting notes](docs/shapley/README.md#hosting) for the website addresses and manual deployment instructions.

## Datasets

The [clean regression choices](data/regression_choices/README.md) provide four
ready-to-use research datasets:
ancestral plough use, slavery and political attitudes, flood response and voting,
and Fox News. Each includes a complete-case CSV, a codebook, and
[Python and R original-regression scripts](data/regression_choices/README.md#reproduce-the-original-regression). Students do not
need to merge files, construct panels, or read Stata data.

| File | Contents | Download |
| --- | --- | --- |
| [tufte_midterms.csv](data/tufte_midterms.csv) | Midterm-election data, including approval and economic variables | [Raw CSV](https://raw.githubusercontent.com/alexanderthclark/pols4728/main/data/tufte_midterms.csv) |
| [atus_prosocial.csv](data/atus_prosocial.csv) | ATUS activity variables and prosocial minutes | [Raw CSV](https://raw.githubusercontent.com/alexanderthclark/pols4728/main/data/atus_prosocial.csv) |
| [synthetic250.csv](data/synthetic250.csv) | Synthetic data with x and y columns | [Raw CSV](https://raw.githubusercontent.com/alexanderthclark/pols4728/main/data/synthetic250.csv) |
| [synthetic5000.csv](data/synthetic5000.csv) | Synthetic data with x and y columns | [Raw CSV](https://raw.githubusercontent.com/alexanderthclark/pols4728/main/data/synthetic5000.csv) |

The original [Tufte dataset notes](data/tufte_midterms_extended.md) are also retained. They use the older filename `tufte_midterms_master_full.csv`; the file in this repository is `tufte_midterms.csv`.

## Load a CSV in Python

```python
import pandas as pd

url = "https://raw.githubusercontent.com/alexanderthclark/pols4728/main/data/tufte_midterms.csv"
df = pd.read_csv(url)
```

## Former course materials

The former Jupyter Book, teaching materials, and build/deployment configuration were removed from `main`. The previous version is available in [Git history](https://github.com/alexanderthclark/pols4728/tree/e75b4061d92d80b8bb0ca3ed79aa430d85cbd5a9).
