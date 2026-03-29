(python)=
# Python

Python is the default implementation language for most modern ML libraries and examples used in this course.

## Bootcamp objectives

- write readable, testable Python for data analysis,
- understand functions, arguments, and object methods,
- use NumPy and pandas for core ML data workflows.

## Pythonic baseline

The practical standard for this course:

- favor clarity over clever one-liners,
- make transformations explicit,
- isolate repeated logic in functions,
- keep data-cleaning and modeling steps reproducible.

## Functions and control flow

A function should do one clear job.

```python

def classify_margin(p, threshold=0.5):
    if p >= threshold:
        return 1
    return 0
```

Important distinctions:

- parameters are names in the function definition,
- arguments are values passed at call time,
- `return` exits the function and returns a value.

## Objects, methods, and modules

Python is object-oriented:

- functions: `sorted(values)`
- methods: `values.sort()`

Most ML work combines modules (`numpy`, `pandas`, `sklearn`) plus your own helper functions.

## NumPy essentials

Use arrays for vectorized numerical computation.

```python
import numpy as np

x = np.array([1.0, 2.0, 3.0])
z = (x - x.mean()) / x.std()
```

Core concepts to remember:

- shape and axis semantics,
- broadcasting rules,
- avoiding Python loops when vectorization is natural.

## pandas essentials

Use DataFrames for tabular data pipelines.

```python
import pandas as pd

df = pd.read_csv("data.csv")
X = df[["x1", "x2", "x3"]]
y = df["outcome"]
```

Common operations:

- column selection,
- row filtering,
- merge/join operations,
- handling missingness before model fitting.

## Reproducibility conventions

- pin package versions in environment files,
- set random seeds where applicable,
- separate raw data, processed data, and outputs,
- prefer scripts/notebooks that can run end-to-end without manual steps.

## Scope note

For advanced deep-learning and LLM workflows, Python tooling is currently more mature than R.
If you prefer R for classical models, that is fine; keep interfaces and outputs comparable.
