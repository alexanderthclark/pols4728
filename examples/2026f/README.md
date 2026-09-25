# Fall 2026 Python examples

These scripts match the code listings in the POLS4728 lecture notes. Click a
filename below to view, copy, or download its source.

| Script | Example | Requirements |
| --- | --- | --- |
| [errors_procedural.py](errors_procedural.py) | Sum of squared prediction errors using a loop | Python standard library |
| [errors_functional.py](errors_functional.py) | The same calculation using a function and `map` | Python standard library |
| [errors_objects.py](errors_objects.py) | The same calculation using a class | Python standard library |
| [cv_selection_bias.py](cv_selection_bias.py) | Selection bias from choosing the lower cross-validation error | NumPy |
| [atus_selection_leakage.py](atus_selection_leakage.py) | Predictor selection using the full ATUS sample | NumPy, pandas, external ATUS data |

## Run the examples

Use Python 3. From this directory, install the optional dependencies for the
statistical examples:

```sh
python -m pip install -r requirements.txt
```

The three programming-style examples each print `14`:

```sh
python errors_procedural.py
python errors_functional.py
python errors_objects.py
```

Run the cross-validation simulation with:

```sh
python cv_selection_bias.py
```

Its fixed seed and 10,000 repetitions produce mean cross-validation errors near
`[1.9982, 1.9975]` for the two fixed rules. The mean of the selected lower error
is approximately `1.8402`.

## ATUS data and interpretation

`atus_selection_leakage.py` needs the 2024 American Time Use Survey summary file,
`atussum_2024.dat`, distributed by the U.S. Bureau of Labor Statistics. Obtain
and extract that file separately; it is not included here. The smaller
`data/atus_prosocial.csv` elsewhere in this repository is not a substitute.

Place `atussum_2024.dat` in this directory and run the script interactively:

```sh
python -i atus_selection_leakage.py
```

The script does not print a result automatically. At the Python prompt, inspect
the selected column and its correlation with weekly earnings:

```python
print(best_col, highest_correlation)
```

The example deliberately selects a predictor using the entire sample. It
illustrates data leakage, not a recommended model-selection procedure. If you
keep the data in another directory, run Python from that directory and provide
the path to the script; the data filename is resolved from the working directory.
