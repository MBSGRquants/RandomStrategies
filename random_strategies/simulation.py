import sys
sys.path.insert(0, r"C:\Users\BUSGR025\Desktop\local_code\portfolio_lib\portfolio_lib\src")

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List
from tqdm import tqdm

from portfolio_lib import NavEngine


FREQ_RESAMPLE = {
    "daily": None,
    "monthly": "ME",
    "quarterly": "QE",
    "semiannual": "6ME",
    "annual": "YE",
}


@dataclass
class SimulationResult:
    nav_list: List[pd.Series]
    benchmark_nav: pd.Series
    mean_nav: pd.Series
    N: int
    freq: str
    K: int
    seed: int
    weighting: str = "equal"


def _get_rebalancing_dates(date_index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    if freq == "daily":
        return date_index
    rule = FREQ_RESAMPLE[freq]
    dummy = pd.Series(1, index=date_index)
    period_ends = dummy.resample(rule).last().dropna().index
    # Map each period-end to the nearest preceding trading date
    indexer = date_index.get_indexer(period_ends, method="pad")
    dates = date_index[np.unique(indexer[indexer >= 0])]
    # Always include the first available date so the NAV starts from day 1
    if date_index[0] not in dates:
        dates = date_index[[0]].append(dates)
    return dates


def _build_random_weights(
    rebal_dates: pd.DatetimeIndex,
    tickers: list,
    N: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n_dates, n_tickers = len(rebal_dates), len(tickers)
    # Permute ticker indices independently for each date, take first N
    perms = rng.permuted(np.tile(np.arange(n_tickers), (n_dates, 1)), axis=1)
    selected = perms[:, :N]  # (n_dates, N)
    weights = np.zeros((n_dates, n_tickers))
    weights[np.repeat(np.arange(n_dates), N), selected.ravel()] = 1.0 / N
    return pd.DataFrame(weights, index=rebal_dates, columns=tickers)


def run_simulation(
    prices: pd.DataFrame,
    bench_weights: pd.DataFrame,
    gics: pd.Series,
    N: int,
    freq: str,
    K: int,
    seed: int,
) -> SimulationResult:
    rng = np.random.default_rng(seed)

    tickers = prices.columns.intersection(bench_weights.columns).tolist()
    prices = prices[tickers]

    date_index = prices.index
    rebal_dates = _get_rebalancing_dates(date_index, freq)

    # NavEngine: bind once to prices/fees, call K+1 times
    engine = NavEngine(prices=prices, fees=0.0, mgmt_fees=0.0)

    # Benchmark: EW on all tickers, rebalanced daily
    n_tickers = len(tickers)
    bench = pd.DataFrame(1.0 / n_tickers, index=date_index, columns=tickers)
    benchmark_nav = engine.calc(bench).daily

    # K random simulations
    nav_list = []
    for _ in tqdm(range(K), desc=f"N={N} freq={freq}", leave=False):
        weights = _build_random_weights(rebal_dates, tickers, N, rng)
        nav_list.append(engine.calc(weights).daily)

    mean_nav = pd.concat(nav_list, axis=1).mean(axis=1)

    return SimulationResult(
        nav_list=nav_list,
        benchmark_nav=benchmark_nav,
        mean_nav=mean_nav,
        N=N,
        freq=freq,
        K=K,
        seed=seed,
    )
