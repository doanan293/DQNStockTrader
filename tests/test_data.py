from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from dqn_stock_trader.data import load_investing_csv, merge_ticker_files, split_by_date

FIXTURE = Path(__file__).parent / "fixtures" / "sample_investing.csv"


@pytest.fixture
def fixture_csv() -> Path:
    return FIXTURE


def test_parses_investing_number_formats(fixture_csv: Path) -> None:
    df = load_investing_csv(fixture_csv)
    assert df.loc[pd.Timestamp("2014-01-02"), "close"] == 3910.0
    assert df.loc[pd.Timestamp("2014-01-02"), "open"] == 3910.0
    assert df.loc[pd.Timestamp("2014-01-02"), "volume"] == pytest.approx(540_070.0)
    assert df.loc[pd.Timestamp("2014-01-06"), "volume"] == 1_200_000.0
    assert pd.isna(df.loc[pd.Timestamp("2014-01-07"), "volume"])
    assert df.index.is_monotonic_increasing
    assert list(df.columns) == ["close", "open", "high", "low", "volume"]


def test_merge_drops_duplicate_dates(tmp_path: Path) -> None:
    lines = FIXTURE.read_text().splitlines()
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    first.write_text("\n".join([lines[0], *lines[3:5]]) + "\n")
    second.write_text("\n".join([lines[0], *lines[1:4]]) + "\n")
    merged = merge_ticker_files([first, second])
    assert merged.index.is_unique
    assert len(merged) == 4


def test_split_has_no_overlap(fixture_csv: Path) -> None:
    df = load_investing_csv(fixture_csv)
    splits = split_by_date(df, train_end=date(2014, 1, 2), val_end=date(2014, 1, 3))
    assert splits.train.index.max() < splits.val.index.min()
    assert splits.val.index.max() < splits.test.index.min()
    assert len(splits.train) + len(splits.val) + len(splits.test) == len(df)
