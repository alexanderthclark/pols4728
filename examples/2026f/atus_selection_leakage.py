import pandas as pd
import numpy as np

df = pd.read_csv("atussum_2024.dat")

# TRERNWA is a weekly earnings variable
df = df[df['TRERNWA'] > 0]
income = df['TRERNWA']

# time use columns happen to start with lower-case t
time_use_columns = [c for c in df.columns if c.startswith("t")]
# there are 372 time use columns. a lot!

# find most correlated column
highest_correlation = 0
best_col = None
for col in time_use_columns:
    r = np.corrcoef(income, df[col])[0][1]
    if np.abs(r) > np.abs(highest_correlation):
        best_col = col
        highest_correlation = r
# t030405 is the most correlated column:
# waiting associated with caring for household adults.
