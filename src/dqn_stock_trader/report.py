"""Run the experiment grid on the test period and write the results tables."""

import os
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from dqn_stock_trader.agent import Variant
from dqn_stock_trader.baselines import buy_and_hold_equity, random_policy, run_policy
from dqn_stock_trader.config import ExperimentConfig
from dqn_stock_trader.env import TradingEnv
from dqn_stock_trader.evaluate import evaluate, split_prices
from dqn_stock_trader.metrics import max_drawdown, sharpe_ratio, total_return
from dqn_stock_trader.train import checkpoint_path, train

FloatArray = npt.NDArray[np.float64]
COLUMNS = [
    "ticker",
    "policy",
    "variant",
    "seed_mean_return",
    "seed_std_return",
    "sharpe",
    "max_drawdown",
    "n_trades",
]
TradeKey = tuple[str, str, str, int]


@dataclass(frozen=True)
class RunRecord:
    ticker: str
    policy: str
    variant: str
    seed: int
    equity: FloatArray


def _sample_std(values: pd.Series) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def _percent(value: float) -> str:
    return f"{value:.1%}"


def _two_decimals(value: float) -> str:
    return f"{value:.2f}"


def _integer(value: float) -> str:
    return f"{value:.0f}"


def summarize(runs: list[RunRecord], trades: Mapping[TradeKey, int]) -> pd.DataFrame:
    """One row per (ticker, policy, variant), averaging metrics over seeds."""
    rows = [
        {
            "ticker": r.ticker,
            "policy": r.policy,
            "variant": r.variant,
            "return": total_return(r.equity),
            "sharpe": sharpe_ratio(r.equity),
            "max_drawdown": max_drawdown(r.equity),
            "n_trades": trades.get((r.ticker, r.policy, r.variant, r.seed), 0),
        }
        for r in runs
    ]
    grouped = pd.DataFrame(rows).groupby(["ticker", "policy", "variant"], sort=False)
    table = grouped.agg(
        seed_mean_return=("return", "mean"),
        seed_std_return=("return", _sample_std),
        sharpe=("sharpe", "mean"),
        max_drawdown=("max_drawdown", "mean"),
        n_trades=("n_trades", "mean"),
    ).reset_index()
    return table[COLUMNS]


def _train_job(job: tuple[ExperimentConfig, str, Variant, int]) -> Path:
    torch.set_num_threads(1)
    cfg, ticker, variant, seed = job
    return train(cfg, ticker, variant, seed)


def run_grid(cfg: ExperimentConfig, workers: int) -> None:
    """Train every missing checkpoint in parallel, one CPU thread per process."""
    jobs = [
        (cfg, ticker, variant, seed)
        for ticker in cfg.tickers
        for variant in cfg.variants_for(ticker)
        for seed in cfg.seeds
        if not checkpoint_path(cfg, ticker, variant, seed).exists()
    ]
    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_train_job, jobs))


def collect_runs(cfg: ExperimentConfig) -> tuple[list[RunRecord], dict[TradeKey, int]]:
    runs: list[RunRecord] = []
    trades: dict[TradeKey, int] = {}
    for ticker in cfg.tickers:
        prices = split_prices(cfg, ticker, "test")
        runs.append(RunRecord(ticker, "buy_and_hold", "-", 0, buy_and_hold_equity(prices, cfg.env)))
        trades[(ticker, "buy_and_hold", "-", 0)] = 1
        for seed in cfg.seeds:
            result = run_policy(TradingEnv(prices, cfg.env), random_policy(seed))
            runs.append(RunRecord(ticker, "random", "-", seed, result.equity))
            trades[(ticker, "random", "-", seed)] = result.n_trades
            for variant in cfg.variants_for(ticker):
                ckpt = checkpoint_path(cfg, ticker, variant, seed)
                result = evaluate(cfg, ticker, ckpt, variant=variant, split="test")
                runs.append(RunRecord(ticker, "dqn_agent", variant, seed, result.equity))
                trades[(ticker, "dqn_agent", variant, seed)] = result.n_trades
    return runs, trades


def _plot_equity(runs: list[RunRecord], ticker: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for run in runs:
        if run.ticker != ticker or run.seed != 0 or run.policy == "random":
            continue
        label = "buy and hold" if run.policy == "buy_and_hold" else f"agent ({run.variant})"
        ax.plot(run.equity / 1e6, label=label)
    ax.set_xlabel("trading day in test period")
    ax.set_ylabel("portfolio value (million VND)")
    ax.set_title(f"{ticker} test period, seed 0")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def build_report(cfg: ExperimentConfig, workers: int | None = None) -> pd.DataFrame:
    run_grid(cfg, workers or max(1, (os.cpu_count() or 2) - 2))
    runs, trades = collect_runs(cfg)
    table = summarize(runs, trades)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(cfg.output_dir / "summary.csv", index=False, float_format="%.4f")
    shown = table.copy()
    for column in ("seed_mean_return", "seed_std_return", "max_drawdown"):
        shown[column] = shown[column].map(_percent)
    shown["sharpe"] = shown["sharpe"].map(_two_decimals)
    shown["n_trades"] = shown["n_trades"].map(_integer)
    (cfg.output_dir / "summary.md").write_text(shown.to_markdown(index=False) + "\n")
    _plot_equity(runs, "FPT", cfg.output_dir / "fpt_equity.png")
    return table
