"""Reference policies and a runner that records the equity curve."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from dqn_stock_trader.env import EnvConfig, TradingEnv

FloatArray = npt.NDArray[np.float64]
Policy = Callable[[FloatArray], int]


@dataclass(frozen=True)
class EpisodeResult:
    equity: FloatArray
    n_trades: int


def run_policy(env: TradingEnv, policy: Policy) -> EpisodeResult:
    state = env.reset()
    equity = [env.portfolio_value()]
    trades = 0
    done = False
    while not done:
        result = env.step(policy(state))
        state = result.state
        done = result.done
        trades += result.trade in ("buy", "sell")
        equity.append(env.portfolio_value())
    return EpisodeResult(equity=np.asarray(equity), n_trades=trades)


def buy_and_hold_policy() -> Policy:
    """Keep buying the largest lot; once cash runs short the buys become no-ops (holding)."""
    largest_lot_action = 3
    return lambda _state: largest_lot_action


def random_policy(seed: int, n_actions: int = 5) -> Policy:
    rng = np.random.default_rng(seed)
    return lambda _state: int(rng.integers(n_actions))


def buy_and_hold_equity(prices: FloatArray, config: EnvConfig, lot: int = 100) -> FloatArray:
    """Spend all cash on the first day in round lots, then hold; the market benchmark."""
    first = float(prices[0])
    shares = int(config.initial_cash / (first * (1 + config.fee_rate)) // lot * lot)
    cash = config.initial_cash - shares * first * (1 + config.fee_rate)
    return cash + shares * np.asarray(prices, dtype=np.float64)
