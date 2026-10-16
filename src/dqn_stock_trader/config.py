"""Experiment configuration loaded from YAML."""

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from dqn_stock_trader.agent import AgentConfig, Variant
from dqn_stock_trader.env import EnvConfig


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    data_dir: Path = Path("data/raw")
    checkpoint_dir: Path = Path("checkpoints")
    output_dir: Path = Path("results")
    tickers: list[str]
    seeds: list[int] = [0, 1, 2]
    episodes: int = 30
    train_end: date = date(2021, 12, 31)
    val_end: date = date(2022, 12, 31)
    test_end: date = date(2024, 3, 15)
    variants_on: dict[str, list[Variant]] = {"*": ["double"]}
    env: EnvConfig = EnvConfig()
    agent: AgentConfig = AgentConfig()

    def variants_for(self, ticker: str) -> list[Variant]:
        return self.variants_on.get(ticker, self.variants_on.get("*", ["double"]))


def load_config(path: Path) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text())
    raw["env"]["lot_sizes"] = tuple(raw["env"]["lot_sizes"])
    return ExperimentConfig.model_validate(raw)
