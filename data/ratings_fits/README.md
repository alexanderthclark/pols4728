# Clothing-sequence ratings

A synthetic regression dataset for predicting a score from an ordered clothing
sequence and a person's age. All observations are simulated.

| File | Role | Rows | NumPy seed | File size |
| --- | --- | ---: | ---: | ---: |
| [fit_data.parquet](fit_data.parquet) | Fitting | 1,200 | 11 | 7,592 bytes |
| [validation_fit_data.parquet](validation_fit_data.parquet) | Validation | 1,200 | 1234 | 7,552 bytes |

## Load in Python

Install pandas and PyArrow:

```bash
python -m pip install pandas pyarrow
```

```python
import pandas as pd

base_url = "https://raw.githubusercontent.com/alexanderthclark/pols4728/main/data/ratings_fits"
fit = pd.read_parquet(f"{base_url}/fit_data.parquet")
validation = pd.read_parquet(f"{base_url}/validation_fit_data.parquet")

# Convert each sequence to a Python list for list-based preprocessing.
# Depending on pandas settings, Parquet list cells can load as NumPy arrays.
fit["sequence"] = fit["sequence"].map(list)
validation["sequence"] = validation["sequence"].map(list)
```

## Columns

| Column | Parquet type | Meaning |
| --- | --- | --- |
| `sequence` | `list<string>` | Ordered clothing items; repeated items are retained. |
| `age` | `int64` | Integer age, from 18 through 69. |
| `target` | `float64` | Synthetic score to predict, from -5 through 3.5 in these samples. |

The vocabulary is `left_sock`, `right_sock`, `left_shoe`, `right_shoe`,
`pants`, and `ring`. Sequence lengths range from 0 through 10. Each file
contains one empty sequence; it is an empty list, not a missing value.

The files contain no saved DataFrame index, padding tokens, token IDs, or
precomputed features. Item order matters and should be preserved.

## Sampling and format

The validation observations are drawn independently using a different random
seed. This is not a grouped split: 73 distinct sequence/age combinations occur
in both files, and repeated observations are retained within each file.
Use these data to assess performance on new draws from the same distribution,
rather than to claim generalization to entirely unseen sequence/age combinations.

Parquet preserves the ordered list column without embedding JSON or Python
literal strings in CSV cells. The two files total 15,144 bytes, compared with
199,124 bytes for equivalent CSVs with JSON-encoded sequences and 74,784 bytes
for the original pickle outputs. Pickle is not used for public distribution.

Generated with NumPy 1.26.4, pandas 2.3.3, and PyArrow 22.0.0, using the supplied
generator's unchanged sampling and scoring logic. Both files were checked
row-for-row against its original outputs.
