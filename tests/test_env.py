import numpy as np
import pytest

from dqn_stock_trader.env import EnvConfig, TradingEnv

PRICES = np.array([10_000.0] * 12 + [11_000.0] * 3)


@pytest.fixture
def env() -> TradingEnv:
    return TradingEnv(PRICES, EnvConfig(window_size=10))


def test_state_in_unit_interval_and_shape(env: TradingEnv) -> None:
    state = env.reset()
    assert state.shape == (11,)
    assert env.state_size == 11
    assert env.n_actions == 5
    assert (state >= 0).all()
    assert (state <= 1).all()


def test_buy_charges_fee(env: TradingEnv) -> None:
    env.reset()
    result = env.step(1)
    assert result.trade == "buy"
    assert env.shares == 300
    assert env.cash == pytest.approx(100_000_000 - 300 * 10_000 * 1.0015)


def test_sell_charges_fee_and_tax_and_clears_position(env: TradingEnv) -> None:
    env.reset()
    env.step(1)
    cash_before = env.cash
    result = env.step(4)
    assert result.trade == "sell"
    assert env.shares == 0
    assert env.cash == pytest.approx(cash_before + 300 * 10_000 * (1 - 0.0015 - 0.001))


def test_mark_to_market_reward_equals_value_change(env: TradingEnv) -> None:
    env.reset()
    for _ in range(11):
        env.step(0)
    before = env.portfolio_value()
    result = env.step(3)
    assert result.reward == pytest.approx(env.portfolio_value() - before)
    assert result.reward == pytest.approx(900 * 1_000 - 900 * 10_000 * 0.0015)


def test_realized_reward_only_on_sell() -> None:
    env = TradingEnv(PRICES, EnvConfig(window_size=10, reward_mode="realized"))
    env.reset()
    for _ in range(11):
        env.step(0)
    assert env.step(1).reward == 0.0
    sell = env.step(4)
    assert sell.reward == pytest.approx(300 * 11_000 * (1 - 0.0015 - 0.001) - 300 * 10_000 * 1.0015)


def test_buy_without_enough_cash_is_noop() -> None:
    env = TradingEnv(PRICES, EnvConfig(window_size=10, initial_cash=1_000))
    env.reset()
    result = env.step(1)
    assert result.trade == "noop"
    assert env.cash == 1_000
    assert result.reward == 0.0


def test_sell_with_no_shares_is_noop(env: TradingEnv) -> None:
    env.reset()
    result = env.step(4)
    assert result.trade == "noop"
    assert env.cash == 100_000_000


def test_episode_ends_at_last_price(env: TradingEnv) -> None:
    env.reset()
    done = False
    steps = 0
    while not done:
        done = env.step(0).done
        steps += 1
    assert steps == len(PRICES) - 1
