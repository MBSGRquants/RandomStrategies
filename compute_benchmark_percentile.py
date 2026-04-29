"""
Compute, for each (N, freq) combo, the percentile at which the benchmark
total return falls within the distribution of the 10 000 random strategies.

Run once:  poetry run python compute_benchmark_percentile.py
Output:    <OUTPUT_DIR>/benchmark_percentile.csv
"""

import pickle
import numpy as np
import pandas as pd

OUTPUT_DIR = r"U:\Gruppo Esperia Comune\Quant Team\Sviluppo\RandomStrategies\output"


def total_return(s: pd.Series) -> float:
    return s.iloc[-1] / s.iloc[0] - 1


def main() -> None:
    print("Loading all_navs.pkl …")
    with open(f"{OUTPUT_DIR}/all_navs.pkl", "rb") as f:
        all_navs: dict[tuple, pd.DataFrame] = pickle.load(f)

    mean_navs = pd.read_csv(
        f"{OUTPUT_DIR}/mean_navs.csv", index_col=0, parse_dates=True
    )
    bench_total_ret = total_return(mean_navs["bench_EW"])

    rows = []
    for key, df in all_navs.items():
        N, freq = key[0], key[1]
        weighting = key[2] if len(key) > 2 else "equal"
        sim_cols = [c for c in df.columns if c.startswith("sim_")]
        sim_returns = (df[sim_cols].iloc[-1] / df[sim_cols].iloc[0] - 1).values
        percentile = float(np.mean(sim_returns < bench_total_ret) * 100)
        rows.append({"N": N, "freq": freq, "weighting": weighting, "bench_percentile": percentile})
        print(f"  N={N} freq={freq} {weighting}: benchmark @ {percentile:.1f}th percentile")

    result = pd.DataFrame(rows)
    out_path = f"{OUTPUT_DIR}/benchmark_percentile.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSalvato in {out_path}")


if __name__ == "__main__":
    main()
