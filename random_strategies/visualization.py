import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mb_style import apply_mb_style

from random_strategies.simulation import SimulationResult

COLORS = apply_mb_style()
TRADING_DAYS = 252


def _sharpe(nav: pd.Series) -> float:
    ret = nav.pct_change().dropna()
    if ret.std() == 0:
        return np.nan
    return ret.mean() / ret.std() * np.sqrt(TRADING_DAYS)


def _total_return(nav: pd.Series) -> float:
    return nav.iloc[-1] / nav.iloc[0] - 1


def plot_fan_chart(result: SimulationResult, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))

    for nav in result.nav_list:
        ax.plot(nav.index, nav.values, color=COLORS["grey"], alpha=0.25, linewidth=0.6)

    ax.plot(result.benchmark_nav.index, result.benchmark_nav.values,
            color=COLORS["red"], linewidth=2, label="Benchmark EW")
    ax.plot(result.mean_nav.index, result.mean_nav.values,
            color=COLORS["dark_blue"], linewidth=2, label=f"Mean of {result.K} random")

    ax.set_title(f"Fan Chart — N={result.N}, freq={result.freq}, K={result.K}")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, f"{result.N}_{result.freq}_fan_chart.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path



def save_mean_navs(results: list[SimulationResult], output_dir: str) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    df = pd.concat(
        {"bench_EW": results[0].benchmark_nav} |
        {f"N{r.N}_{r.freq}": r.mean_nav for r in results},
        axis=1,
    )
    df.to_csv(os.path.join(output_dir, "mean_navs.csv"))
    return df


def save_all_navs(results: list[SimulationResult], output_dir: str) -> None:
    """Pickle: dict keyed by (N, freq) → DataFrame with columns sim_0..sim_K-1, mean, benchmark."""
    os.makedirs(output_dir, exist_ok=True)
    data = {}
    for r in results:
        df = pd.concat(r.nav_list, axis=1)
        df.columns = [f"sim_{k}" for k in range(len(r.nav_list))]
        df["mean"] = r.mean_nav
        df["benchmark"] = r.benchmark_nav
        data[(r.N, r.freq)] = df
    with open(os.path.join(output_dir, "all_navs.pkl"), "wb") as f:
        pickle.dump(data, f)


def build_summary_table(results: list[SimulationResult], output_dir: str) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for r in results:
        ret_list = [_total_return(nav) for nav in r.nav_list]
        sharpe_list = [_sharpe(nav) for nav in r.nav_list]
        rows.append({
            "N": r.N,
            "freq": r.freq,
            "mean_return": np.mean(ret_list),
            "median_return": np.median(ret_list),
            "p5_return": np.percentile(ret_list, 5),
            "p95_return": np.percentile(ret_list, 95),
            "mean_sharpe": np.nanmean(sharpe_list),
            "benchmark_return": _total_return(r.benchmark_nav),
            "benchmark_sharpe": _sharpe(r.benchmark_nav),
        })

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "summary_table.csv")
    df.to_csv(path, index=False)
    return df
