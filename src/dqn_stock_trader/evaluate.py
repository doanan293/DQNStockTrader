"""Greedy evaluation of a saved agent on one data split."""

import dataclasses
from pathlib import Path
from typing import Literal

import numpy as np

from dqn_stock_trader.agent import DQNAgent, Variant
from dqn_stock_trader.baselines import EpisodeResult, run_policy
from dqn_stock_trader.config import ExperimentConfig
from dqn_stock_trader.data import load_ticker, split_by_date
from dqn_stock_trader.env import TradingEnv

Split = Literal["train", "val", "test"]


def split_prices(cfg: ExperimentConfig, ticker: str, split: Split) -> np.ndarray:
    splits = split_by_date(load_ticker(ticker, cfg.data_dir), cfg.train_end, cfg.val_end)
    frame = {"train": splits.train, "val": splits.val, "test": splits.test}[split]
    return frame["close"].to_numpy(dtype=np.float64)


def evaluate(
    cfg: ExperimentConfig,
    ticker: str,
    checkpoint: Path,
    variant: Variant = "double",
    split: Split = "test",
) -> EpisodeResult:
    env = TradingEnv(split_prices(cfg, ticker, split), cfg.env)
    agent = DQNAgent(
        env.state_size, env.n_actions, dataclasses.replace(cfg.agent, variant=variant), seed=0
    )
    agent.load(checkpoint)
    return run_policy(env, lambda s: agent.act(s, greedy=True))
