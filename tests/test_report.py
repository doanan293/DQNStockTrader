import numpy as np
import pytest

from dqn_stock_trader.report import COLUMNS, RunRecord, summarize


def test_report_columns_and_rows() -> None:
    runs = [
        RunRecord("FPT", "dqn_agent", "double", 0, np.array([100.0, 110.0])),
        RunRecord("FPT", "dqn_agent", "double", 1, np.array([100.0, 90.0])),
        RunRecord("FPT", "buy_and_hold", "-", 0, np.array([100.0, 120.0])),
    ]
    table = summarize(runs, trades={("FPT", "dqn_agent", "double", 0): 4})
    assert list(table.columns) == COLUMNS
    assert len(table) == 2
    agent = table[table.policy == "dqn_agent"].iloc[0]
    assert agent.seed_mean_return == pytest.approx(0.0)
    assert agent.seed_std_return == pytest.approx(np.std([0.1, -0.1], ddof=1))
