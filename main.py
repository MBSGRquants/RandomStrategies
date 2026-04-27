import sys
sys.path.insert(0, r"C:\Users\BUSGR025\Desktop\local_code\portfolio_lib\portfolio_lib\src")
sys.path.insert(0, r"C:\Users\BUSGR025\Desktop\local_code\DataLoader")

# ── Imports ───────────────────────────────────────────────────────────────────

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from loader import load_data
from portfolio_lib import NavEngine
from random_strategies.simulation import (
    SimulationResult, _get_rebalancing_dates, FREQ_RESAMPLE,
)
from random_strategies.visualization import plot_fan_chart, build_summary_table, save_mean_navs


# ── Parameters ────────────────────────────────────────────────────────────────

# N_LIST     = [40, 100]
# FREQ_LIST  = ["daily", "monthly", "quarterly", "semiannual", "annual"]
# K          = 1000

N_LIST     = [40, 100]
FREQ_LIST  = ["monthly", "quarterly", "semiannual", "annual"]
K          = 10000
SEED       = 42
OUTPUT_DIR = "output/"
START_YEAR = 2015
# ── Load data ─────────────────────────────────────────────────────────────────

_data          = load_data(category="stock", universe="B500")
prices         = _data["prices"]
bench_weights  = _data["weights"]
gics           = _data["gics"]
common_columns = _data["common_columns"]
common_dates   = _data["common_dates"]

prices        = prices[common_columns].reindex(common_dates)
bench_weights = bench_weights[common_columns].reindex(common_dates)
gics          = gics.reindex(common_columns)

prices        = prices[prices.index.year >= START_YEAR]
bench_weights = bench_weights[bench_weights.index.year >= START_YEAR]

# ── Simulate ──────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

tickers       = prices.columns.intersection(bench_weights.columns).tolist()
prices        = prices[tickers]
bench_weights = bench_weights[tickers]
date_index    = prices.index

engine = NavEngine(prices=prices, fees=0.0, mgmt_fees=0.0)

# Benchmark: EW on active constituents (bench_weights > 0) at each daily date
bench_weights_daily = bench_weights.reindex(date_index).ffill().fillna(0)
active_mask         = bench_weights_daily > 0
bench_ew            = active_mask.div(active_mask.sum(axis=1), axis=0).fillna(0)
benchmark_nav       = engine.calc(bench_ew).daily

results = []

for N in N_LIST:
    for freq in FREQ_LIST:
        rng         = np.random.default_rng(SEED)
        rebal_dates = _get_rebalancing_dates(date_index, freq)

        # Active universe per rebalancing date (bench_weights > 0 at that date)
        bench_at_rebal  = bench_weights.reindex(rebal_dates, method="pad").fillna(0)
        active_per_date = {
            d: bench_at_rebal.columns[bench_at_rebal.loc[d] > 0].tolist()
            for d in rebal_dates
        }

        # Pre-generate all K×n_rebal selections upfront (vectorized per date)
        ticker_to_idx  = {t: i for i, t in enumerate(tickers)}
        n_rebal        = len(rebal_dates)
        j_idx          = np.repeat(np.arange(n_rebal), N)  # row indices for fancy indexing

        # all_selections[j, k, :] = N global ticker indices for sim k at rebal date j
        all_selections = np.empty((n_rebal, K, N), dtype=np.intp)
        for j, d in enumerate(rebal_dates):
            active     = active_per_date[d]
            active_idx = np.array([ticker_to_idx[t] for t in active])
            perms      = rng.permuted(np.tile(np.arange(len(active)), (K, 1)), axis=1)
            all_selections[j] = active_idx[perms[:, :N]]  # (K, N) global indices

        nav_list = []
        for k in tqdm(range(K), desc=f"N={N} freq={freq}"):
            w = np.zeros((n_rebal, len(tickers)))
            w[j_idx, all_selections[:, k, :].ravel()] = 1.0 / N
            nav_list.append(engine.calc(pd.DataFrame(w, index=rebal_dates, columns=tickers)).daily)

        mean_nav = pd.concat(nav_list, axis=1).mean(axis=1)

        results.append(SimulationResult(
            nav_list=nav_list,
            benchmark_nav=benchmark_nav,
            mean_nav=mean_nav,
            N=N, freq=freq, K=K, seed=SEED,
        ))

# ── Generate outputs ──────────────────────────────────────────────────────────

for result in results:
    plot_fan_chart(result, OUTPUT_DIR)

mean_navs = save_mean_navs(results, OUTPUT_DIR)
summary   = build_summary_table(results, OUTPUT_DIR)
print(summary.to_string(index=False))
