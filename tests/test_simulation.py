import sys
sys.path.insert(0, r"C:\Users\BUSGR025\Desktop\local_code\portfolio_lib\portfolio_lib\src")

import numpy as np
import pandas as pd
import pytest

from random_strategies.simulation import run_simulation, SimulationResult


def _make_synthetic_data(n_tickers=20, n_days=500, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    returns = rng.normal(0.0005, 0.01, size=(n_days, n_tickers))
    prices = pd.DataFrame(
        100 * np.cumprod(1 + returns, axis=0),
        index=dates,
        columns=tickers,
    )
    bench_weights = pd.DataFrame(
        1.0 / n_tickers, index=dates, columns=tickers
    )
    gics = pd.Series("IT", index=tickers)
    return prices, bench_weights, gics


@pytest.fixture
def synthetic_data():
    return _make_synthetic_data()


def test_returns_k_nav_series(synthetic_data):
    prices, bench_weights, gics = synthetic_data
    result = run_simulation(prices, bench_weights, gics, N=5, freq="monthly", K=10, seed=42)
    assert isinstance(result, SimulationResult)
    assert len(result.nav_list) == 10


def test_nav_series_length(synthetic_data):
    prices, bench_weights, gics = synthetic_data
    result = run_simulation(prices, bench_weights, gics, N=5, freq="monthly", K=5, seed=42)
    expected_len = len(prices)
    for nav in result.nav_list:
        assert len(nav) == expected_len


def test_weights_sum_to_one(synthetic_data):
    prices, bench_weights, gics = synthetic_data
    from random_strategies.simulation import _build_random_weights, _get_rebalancing_dates
    rng = np.random.default_rng(0)
    tickers = prices.columns.tolist()
    rebal_dates = _get_rebalancing_dates(prices.index, "monthly")
    weights = _build_random_weights(rebal_dates, tickers, N=5, rng=rng)
    sums = weights.sum(axis=1)
    assert (sums - 1.0).abs().max() < 1e-10


def test_exactly_n_tickers_selected(synthetic_data):
    prices, bench_weights, gics = synthetic_data
    from random_strategies.simulation import _build_random_weights, _get_rebalancing_dates
    rng = np.random.default_rng(0)
    tickers = prices.columns.tolist()
    N = 5
    rebal_dates = _get_rebalancing_dates(prices.index, "monthly")
    weights = _build_random_weights(rebal_dates, tickers, N=N, rng=rng)
    for date in rebal_dates:
        nonzero = (weights.loc[date] > 0).sum()
        assert nonzero == N


def test_determinism(synthetic_data):
    prices, bench_weights, gics = synthetic_data
    r1 = run_simulation(prices, bench_weights, gics, N=5, freq="monthly", K=5, seed=99)
    r2 = run_simulation(prices, bench_weights, gics, N=5, freq="monthly", K=5, seed=99)
    for nav1, nav2 in zip(r1.nav_list, r2.nav_list):
        pd.testing.assert_series_equal(nav1, nav2)


def test_benchmark_nav_matches_ew(synthetic_data):
    prices, bench_weights, gics = synthetic_data
    from portfolio_lib import NavEngine
    tickers = prices.columns.tolist()
    n = len(tickers)
    ew = pd.DataFrame(1.0 / n, index=prices.index, columns=tickers)
    engine = NavEngine(prices=prices, fees=0.0, mgmt_fees=0.0)
    expected_nav = engine.calc(ew).daily
    result = run_simulation(prices, bench_weights, gics, N=5, freq="monthly", K=1, seed=0)
    pd.testing.assert_series_equal(result.benchmark_nav, expected_nav)


def test_mean_nav_shape(synthetic_data):
    prices, bench_weights, gics = synthetic_data
    result = run_simulation(prices, bench_weights, gics, N=5, freq="monthly", K=10, seed=42)
    assert len(result.mean_nav) == len(prices)
