"""Portfolio performance metrics on an equity curve."""

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def total_return(equity: FloatArray) -> float:
    return float(equity[-1] / equity[0] - 1.0)


def sharpe_ratio(equity: FloatArray, periods_per_year: int = 252) -> float:
    """Annualized Sharpe of simple daily returns with a zero risk-free rate."""
    returns = np.diff(equity) / equity[:-1]
    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    if std == 0.0:
        return 0.0
    return float(np.mean(returns) / std * np.sqrt(periods_per_year))


def max_drawdown(equity: FloatArray) -> float:
    """Largest peak-to-trough fall, as a negative fraction."""
    peaks = np.maximum.accumulate(equity)
    return float(np.min(equity / peaks - 1.0))
