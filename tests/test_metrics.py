import numpy as np
import pytest

from dqn_stock_trader.baselines import buy_and_hold_policy, random_policy, run_policy
from dqn_stock_trader.env import EnvConfig, TradingEnv
from dqn_stock_trader.metrics import max_drawdown, sharpe_ratio, total_return


def test_total_return() -> None:
    assert total_return(np.array([100.0, 110.0])) == pytest.approx(0.10)


def test_max_drawdown() -> None:
    assert max_drawdown(np.array([100.0, 120.0, 90.0, 130.0])) == pytest.approx(-0.25)


def test_sharpe_zero_vol() -> None:
    assert sharpe_ratio(np.array([100.0, 100.0, 100.0])) == 0.0


def test_sharpe_positive_for_steady_gains() -> None:
    equity = 100.0 * np.cumprod(np.r_[1.0, np.full(50, 1.001) + np.tile([0.0005, -0.0005], 25)])
    assert sharpe_ratio(equity) > 0


def test_buy_and_hold_tracks_price() -> None:
    prices = np.linspace(10_000.0, 12_000.0, 40)
    env = TradingEnv(prices, EnvConfig(window_size=5, initial_cash=10_000_000))
    result = run_policy(env, buy_and_hold_policy())
    assert result.equity[-1] > result.equity[0]
    assert result.n_trades >= 1
    assert len(result.equity) == len(prices)


def test_random_policy_is_reproducible() -> None:
    prices = np.linspace(10_000.0, 12_000.0, 40)
    first = run_policy(TradingEnv(prices, EnvConfig(window_size=5)), random_policy(seed=3))
    second = run_policy(TradingEnv(prices, EnvConfig(window_size=5)), random_policy(seed=3))
    assert np.array_equal(first.equity, second.equity)


def test_buy_and_hold_equity_invests_everything_on_day_one() -> None:
    from dqn_stock_trader.baselines import buy_and_hold_equity

    prices = np.array([10_000.0, 11_000.0, 12_000.0])
    config = EnvConfig(initial_cash=1_000_000, fee_rate=0.0, sell_tax_rate=0.0)
    equity = buy_and_hold_equity(prices, config)
    assert equity[0] == pytest.approx(1_000_000)
    assert equity[-1] == pytest.approx(1_200_000)
