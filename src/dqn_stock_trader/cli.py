"""Command line interface: dqn-trader train | evaluate | report."""

from pathlib import Path
from typing import Annotated

import typer

from dqn_stock_trader.agent import Variant
from dqn_stock_trader.config import load_config
from dqn_stock_trader.evaluate import evaluate as evaluate_agent
from dqn_stock_trader.metrics import max_drawdown, sharpe_ratio, total_return
from dqn_stock_trader.train import train as train_agent

app = typer.Typer(help="Deep Q-learning stock trading experiments.", no_args_is_help=True)
ConfigOption = Annotated[Path, typer.Option(help="YAML experiment config")]


@app.command()
def train(
    ticker: str,
    variant: Variant = "double",
    seed: int = 0,
    config: ConfigOption = Path("configs/default.yaml"),
) -> None:
    """Train one agent and save the best validation checkpoint."""
    path = train_agent(load_config(config), ticker, variant, seed)
    print(f"saved {path}")


@app.command()
def evaluate(
    ticker: str,
    checkpoint: Path,
    variant: Variant = "double",
    config: ConfigOption = Path("configs/default.yaml"),
) -> None:
    """Evaluate a checkpoint greedily on the test period."""
    result = evaluate_agent(load_config(config), ticker, checkpoint, variant=variant)
    print(
        f"return {total_return(result.equity):.2%}  sharpe {sharpe_ratio(result.equity):.2f}  "
        f"max drawdown {max_drawdown(result.equity):.2%}  trades {result.n_trades}"
    )
