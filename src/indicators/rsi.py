"""Relative Strength Index indicator system."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


DEFAULT_RSI_BOUNDS = {
    "period_min": 5,
    "period_max": 30,
    "lower_min": 10.0,
    "lower_max": 40.0,
    "upper_min": 60.0,
    "upper_max": 90.0,
}


def _bounds(bounds: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = DEFAULT_RSI_BOUNDS.copy()
    if bounds:
        merged.update(bounds)
    return merged


def repair_rsi_params(params: Sequence[float], bounds: dict | None = None) -> list[float]:
    """Validate and repair RSI parameters.

    Parameter format:
    [period, lower_threshold, upper_threshold].
    """

    if len(params) != 3:
        raise ValueError(f"RSI params must have length 3, got {len(params)}")

    b = _bounds(bounds)
    period = int(np.clip(round(params[0]), b["period_min"], b["period_max"]))
    lower = float(np.clip(params[1], b["lower_min"], b["lower_max"]))
    upper = float(np.clip(params[2], b["upper_min"], b["upper_max"]))

    if lower >= upper:
        midpoint = (float(b["lower_max"]) + float(b["upper_min"])) / 2
        lower = min(float(b["lower_max"]), midpoint - 1.0)
        upper = max(float(b["upper_min"]), midpoint + 1.0)

    return [period, lower, upper]


def _validate_rsi_inputs(
    df: pd.DataFrame,
    period: int,
    lower_threshold: float,
    upper_threshold: float,
    close_col: str,
) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if close_col not in df.columns:
        raise KeyError(f"Missing close column: {close_col}")
    if period <= 1:
        raise ValueError("period must be greater than 1")
    if not 0 <= lower_threshold <= 100:
        raise ValueError("lower_threshold must be between 0 and 100")
    if not 0 <= upper_threshold <= 100:
        raise ValueError("upper_threshold must be between 0 and 100")
    if lower_threshold >= upper_threshold:
        raise ValueError("lower_threshold must be less than upper_threshold")


def calculate_rsi(
    df: pd.DataFrame,
    period: int,
    close_col: str = "close",
) -> pd.Series:
    """Calculate RSI using Wilder's smoothing method.

    Gains and losses are smoothed with an exponential moving average using
    alpha=1/period, which matches Wilder's RSI convention. The first valid RSI
    value appears after ``period`` observations.
    """

    _validate_rsi_inputs(df, int(period), 30.0, 70.0, close_col)

    close = pd.to_numeric(df[close_col], errors="coerce")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / int(period), min_periods=int(period), adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / int(period), min_periods=int(period), adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return rsi


def generate_rsi_signals(
    df: pd.DataFrame,
    period: int,
    lower_threshold: float,
    upper_threshold: float,
    close_col: str = "close",
) -> pd.DataFrame:
    """Generate RSI buy/sell signals from threshold crossings.

    Signal rule:
    buy=1 when RSI is below ``lower_threshold`` and sell=1 when RSI is above
    ``upper_threshold``. Rows where RSI is NaN receive zero signals.
    """

    _validate_rsi_inputs(
        df,
        int(period),
        float(lower_threshold),
        float(upper_threshold),
        close_col,
    )

    result = df.copy()
    rsi = calculate_rsi(result, int(period), close_col=close_col)
    valid_rsi = rsi.notna()
    result["rsi"] = rsi
    result["rsi_buy_signal"] = ((rsi < float(lower_threshold)) & valid_rsi).astype(int)
    result["rsi_sell_signal"] = ((rsi > float(upper_threshold)) & valid_rsi).astype(int)
    return result
