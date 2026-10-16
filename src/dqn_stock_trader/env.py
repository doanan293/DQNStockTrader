"""Single-stock trading environment with lot-sized buys, fees and mark-to-market rewards."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

Trade = Literal["hold", "buy", "sell", "noop"]
FloatArray = npt.NDArray[np.float64]

HOLD = 0
SELL_ALL = 4
MIN_PRICES = 2


@dataclass(frozen=True)
class EnvConfig:
    window_size: int = 10
    initial_cash: float = 100_000_000
    lot_sizes: tuple[int, ...] = (300, 600, 900)
    fee_rate: float = 0.0015
    sell_tax_rate: float = 0.001
    reward_mode: Literal["mark_to_market", "realized"] = "mark_to_market"


@dataclass(frozen=True)
class StepResult:
    state: FloatArray
    reward: float
    done: bool
    trade: Trade


def _sigmoid(x: FloatArray) -> FloatArray:
    return 1.0 / (1.0 + np.exp(-x))


class TradingEnv:
    """Action 0 holds, 1..3 buy one of `lot_sizes`, 4 sells the whole position."""

    def __init__(self, prices: FloatArray, config: EnvConfig) -> None:
        if len(prices) < MIN_PRICES:
            msg = "need at least two prices"
            raise ValueError(msg)
        self.prices = np.asarray(prices, dtype=np.float64)
        self.config = config
        self.cash = config.initial_cash
        self.shares = 0
        self.cost_basis = 0.0
        self.t = 0
        self._max_shares = config.initial_cash / float(self.prices[0])

    @property
    def n_actions(self) -> int:
        return 2 + len(self.config.lot_sizes)

    @property
    def state_size(self) -> int:
        return self.config.window_size + 1

    def reset(self) -> FloatArray:
        self.cash = self.config.initial_cash
        self.shares = 0
        self.cost_basis = 0.0
        self.t = 0
        return self._state()

    def portfolio_value(self) -> float:
        return self.cash + self.shares * float(self.prices[self.t])

    def step(self, action: int) -> StepResult:
        if not 0 <= action < self.n_actions:
            msg = f"invalid action {action}"
            raise ValueError(msg)
        value_before = self.portfolio_value()
        trade, realized = self._execute(action, float(self.prices[self.t]))
        self.t += 1
        done = self.t >= len(self.prices) - 1
        if self.config.reward_mode == "mark_to_market":
            reward = self.portfolio_value() - value_before
        else:
            reward = realized
        return StepResult(state=self._state(), reward=reward, done=done, trade=trade)

    def _execute(self, action: int, price: float) -> tuple[Trade, float]:
        if action == HOLD:
            return "hold", 0.0
        if action == SELL_ALL:
            if self.shares == 0:
                return "noop", 0.0
            proceeds = self.shares * price * (1 - self.config.fee_rate - self.config.sell_tax_rate)
            realized = proceeds - self.cost_basis
            self.cash += proceeds
            self.shares = 0
            self.cost_basis = 0.0
            return "sell", realized
        quantity = self.config.lot_sizes[action - 1]
        cost = quantity * price * (1 + self.config.fee_rate)
        if cost > self.cash:
            return "noop", 0.0
        self.cash -= cost
        self.shares += quantity
        self.cost_basis += cost
        return "buy", 0.0

    def _state(self) -> FloatArray:
        window = self.config.window_size
        start = self.t - window
        idx = np.clip(np.arange(start, self.t + 1), 0, None)
        block = self.prices[idx]
        changes = np.diff(block) / block[:-1] * 100.0
        position = min(self.shares / self._max_shares, 1.0)
        return np.append(_sigmoid(changes), position)
