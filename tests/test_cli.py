from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from dqn_stock_trader.agent import DQNAgent
from dqn_stock_trader.cli import app
from dqn_stock_trader.config import ExperimentConfig, load_config
from dqn_stock_trader.evaluate import evaluate
from dqn_stock_trader.replay import Transition
from dqn_stock_trader.train import train

REPO = Path(__file__).parents[1]


@pytest.fixture
def small_cfg(tmp_path: Path) -> ExperimentConfig:
    frame = pd.read_csv(REPO / "data/raw/FPT.csv").iloc[:60]
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    frame.to_csv(data_dir / "FPT.csv", index=False)
    cfg = load_config(REPO / "configs/default.yaml")
    agent = cfg.agent.__class__(batch_size=8)
    return cfg.model_copy(
        update={
            "agent": agent,
            "episodes": 2,
            "data_dir": data_dir,
            "train_end": date(2014, 2, 28),
            "val_end": date(2014, 3, 14),
            "checkpoint_dir": tmp_path / "checkpoints",
        }
    )


def test_default_config_loads() -> None:
    cfg = load_config(REPO / "configs/default.yaml")
    assert len(cfg.tickers) == 10
    assert cfg.seeds == [0, 1, 2]
    assert cfg.env.fee_rate == 0.0015


def test_smoke_train_two_episodes(small_cfg: ExperimentConfig) -> None:
    checkpoint = train(small_cfg, "FPT", "double", seed=0)
    assert checkpoint.exists()


def test_evaluate_does_not_touch_replay(
    small_cfg: ExperimentConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = train(small_cfg, "FPT", "double", seed=0)
    calls: list[Transition] = []

    def spy(_self: DQNAgent, transition: Transition) -> None:
        calls.append(transition)

    monkeypatch.setattr(DQNAgent, "remember", spy)
    result = evaluate(small_cfg, "FPT", checkpoint, variant="double", split="test")
    assert calls == []
    assert len(result.equity) > 1


def test_cli_help() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "train" in result.output
    assert "evaluate" in result.output
    assert "report" in result.output
