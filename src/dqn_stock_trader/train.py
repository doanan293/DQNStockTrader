"""Training loop with checkpoint selection on the validation period."""

import dataclasses
from pathlib import Path

import numpy as np

from dqn_stock_trader.agent import DQNAgent, Variant
from dqn_stock_trader.baselines import run_policy
from dqn_stock_trader.config import ExperimentConfig
from dqn_stock_trader.data import load_ticker, split_by_date
from dqn_stock_trader.env import TradingEnv
from dqn_stock_trader.metrics import total_return
from dqn_stock_trader.replay import Transition


def checkpoint_path(cfg: ExperimentConfig, ticker: str, variant: Variant, seed: int) -> Path:
    return cfg.checkpoint_dir / f"{ticker}_{variant}_s{seed}.pt"


def train(cfg: ExperimentConfig, ticker: str, variant: Variant, seed: int) -> Path:
    """Train one agent and keep the episode whose greedy policy earns most on validation."""
    splits = split_by_date(load_ticker(ticker, cfg.data_dir), cfg.train_end, cfg.val_end)
    train_prices = splits.train["close"].to_numpy(dtype=np.float64)
    val_prices = splits.val["close"].to_numpy(dtype=np.float64)
    env = TradingEnv(train_prices, cfg.env)
    agent = DQNAgent(
        env.state_size, env.n_actions, dataclasses.replace(cfg.agent, variant=variant), seed
    )
    # Rewards are in VND; scale to percent of starting cash so Q-values stay near unit range.
    reward_scale = 100.0 / cfg.env.initial_cash
    path = checkpoint_path(cfg, ticker, variant, seed)
    best = -np.inf
    for _ in range(cfg.episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state, greedy=False)
            result = env.step(action)
            agent.remember(
                Transition(state, action, result.reward * reward_scale, result.state, result.done)
            )
            agent.learn()
            state = result.state
            done = result.done
        val_env = TradingEnv(val_prices, cfg.env)
        score = total_return(run_policy(val_env, lambda s: agent.act(s, greedy=True)).equity)
        if score > best:
            best = score
            agent.save(path)
    return path
