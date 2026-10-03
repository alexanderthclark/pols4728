"""Reproduce one original regression: python baseline.py plough.

Other choices: slavery, floods, fox_news. Each run uses the complete original
control list, weights, indicator controls, and standard-error convention.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=("plough", "slavery", "floods", "fox_news"),
                        nargs="?", default="plough")
    args = parser.parse_args()
    folder = Path(__file__).resolve().parents[2] / "data" / "regression_choices" / args.study
    info = json.loads((folder / "metadata.json").read_text())
    data = pd.read_csv(folder / "data.csv", float_precision="round_trip")

    # The metadata lists exactly the regressors in the published specification.
    regressors = ["d"] + info["controls"] + info["fixed_controls"]
    design = sm.add_constant(data[regressors], has_constant="add")
    weights = (data[info["weight"]].to_numpy(dtype=float, copy=True)
               if info["weight"] else np.ones(len(data)))
    weights /= weights.mean()  # Numerically convenient; WLS is unchanged.
    model = sm.WLS(data["y"], design, weights=weights)
    if args.study == "fox_news":
        fit = model.fit(cov_type="cluster", cov_kwds={
            "groups": data[info["cluster"]], "use_correction": True,
            "df_correction": True}, use_t=True)
        df = data[info["cluster"]].nunique()-1
    elif args.study == "slavery":
        fit = model.fit()  # The authors use classical weighted-OLS uncertainty.
        df = fit.df_resid
    elif args.study == "floods":
        fit = model.fit(cov_type="HC0")
        df = len(data)-1
    else:
        fit = model.fit(cov_type="HC1", use_t=True)
        df = fit.df_resid

    coefficient = float(fit.params["d"])
    se = float(fit.bse["d"])
    if args.study == "floods":
        # One change per district is equivalent to the original two-period
        # regression. Apply its original panel CR1 correction to HC0 here.
        groups, rank = len(data), int(fit.model.rank)
        correction = groups/(groups-1) * (2*groups-1)/(2*groups-rank-1)
        se *= np.sqrt(correction)
    critical = stats.t.ppf(.975, df)
    print(info["title"])
    print(info["table"])
    print(f"N = {len(data)}; coefficient = {coefficient:.10f}; SE = {se:.10f}")
    print(f"95% interval: [{coefficient-critical*se:.10f}, {coefficient+critical*se:.10f}]")
    print(f"p-value = {2*stats.t.sf(abs(coefficient/se), df):.10g}")


if __name__ == "__main__":
    main()
