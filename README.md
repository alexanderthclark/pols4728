# POLS4728 data and code

Public data and code archive for POLS4728. The CSV files remain at their original paths so existing download links continue to work.

## Python examples

The [Fall 2026 Python examples](examples/2026f/README.md) accompany the code listings in the lecture notes. The examples README explains dependencies, how to run each script, and the external data needed for the ATUS example.

## Datasets

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
