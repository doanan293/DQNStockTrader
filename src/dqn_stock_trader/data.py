"""Load Investing.com price exports and split them chronologically."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["close", "open", "high", "low", "volume"]
_SOURCE_COLUMNS = {"Price": "close", "Open": "open", "High": "high", "Low": "low", "Vol.": "volume"}
_SUFFIXES = {"K": 1e3, "M": 1e6, "B": 1e9}


@dataclass(frozen=True)
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def _parse_number(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip().replace(",", "")
    if text in {"", "-", "nan"}:
        return float(np.nan)
    multiplier = _SUFFIXES.get(text[-1].upper(), 1.0)
    if multiplier != 1.0:
        text = text[:-1]
    return float(text) * multiplier


def load_investing_csv(path: Path) -> pd.DataFrame:
    """Read an Investing.com historical-data export into a clean, ascending frame."""
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    frame = pd.DataFrame(
        {target: raw[source].map(_parse_number) for source, target in _SOURCE_COLUMNS.items()}
    )
    frame.index = pd.DatetimeIndex(pd.to_datetime(raw["Date"], format="%m/%d/%Y"), name="date")
    return frame[COLUMNS].sort_index()


def merge_ticker_files(paths: Sequence[Path]) -> pd.DataFrame:
    """Concatenate several exports of one ticker, keeping the last value for repeated dates."""
    frames = [load_investing_csv(path) for path in paths]
    merged = pd.concat(frames)
    return merged[~merged.index.duplicated(keep="last")].sort_index()


def split_by_date(df: pd.DataFrame, train_end: date, val_end: date) -> Splits:
    """Split into train (<= train_end), validation (<= val_end) and test (the rest)."""
    train_cut = pd.Timestamp(train_end)
    val_cut = pd.Timestamp(val_end)
    index = pd.DatetimeIndex(df.index)
    return Splits(
        train=df[index <= train_cut],
        val=df[(index > train_cut) & (index <= val_cut)],
        test=df[index > val_cut],
    )


def load_ticker(ticker: str, data_dir: Path) -> pd.DataFrame:
    """Load a merged per-ticker file written by the migration (ISO dates, clean columns)."""
    frame = pd.read_csv(data_dir / f"{ticker}.csv", parse_dates=["date"], index_col="date")
    return frame[COLUMNS].sort_index()
