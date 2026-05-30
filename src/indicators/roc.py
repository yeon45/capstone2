"""Rate of Change indicator system."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


DEFAULT_ROC_BOUNDS = {
    "period_min": 3,
    "period_max": 30,
    "buy_threshold_min": 0.1,
    "buy_threshold_max": 10.0,
    "sell_threshold_min": -10.0,
    "sell_threshold_max": -0.1,
}


def _bounds(bounds: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = DEFAULT_ROC_BOUNDS.copy()
    if bounds:
        merged.update(bounds)
    return merged


def repair_roc_params(params: Sequence[float], bounds: dict | None = None) -> list[float]:
    """Validate and repair ROC parameters.

    Parameter format:
    [period, buy_threshold, sell_threshold].
    """

    if len(params) != 3:
        raise ValueError(f"ROC params must have length 3, got {len(params)}")

    b = _bounds(bounds)
    period = int(np.clip(round(params[0]), b["period_min"], b["period_max"]))
    buy_threshold = float(
        np.clip(params[1], b["buy_threshold_min"], b["buy_threshold_max"])
    )
    sell_threshold = float(
        np.clip(params[2], b["sell_threshold_min"], b["sell_threshold_max"])
    )

    if buy_threshold <= sell_threshold:
        buy_threshold = max(float(b["buy_threshold_min"]), sell_threshold + 0.1)
        if buy_threshold > float(b["buy_threshold_max"]):
            sell_threshold = min(float(b["sell_threshold_max"]), buy_threshold - 0.1)

    return [period, buy_threshold, sell_threshold]


def _validate_roc_inputs(
    df: pd.DataFrame,
    period: int,
    buy_threshold: float,
    sell_threshold: float,
    close_col: str,
) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if close_col not in df.columns:
        raise KeyError(f"Missing close column: {close_col}")
    if period <= 0:
        raise ValueError("period must be greater than 0")
    if buy_threshold <= sell_threshold:
        raise ValueError("buy_threshold must be greater than sell_threshold")


def calculate_roc(
    df: pd.DataFrame,
    period: int,
    close_col: str = "close",
) -> pd.Series:
    """Calculate ROC as ((close_t - close_t-period) / close_t-period) * 100.

    Prior close values equal to zero are replaced with NaN before division, so
    division-by-zero rows produce NaN ROC values instead of infinite values.
    """

    _validate_roc_inputs(df, int(period), 0.1, -0.1, close_col)

    close = pd.to_numeric(df[close_col], errors="coerce")
    prior_close = close.shift(int(period)).replace(0, np.nan)
    return ((close - prior_close) / prior_close) * 100


def generate_roc_signals(
    df: pd.DataFrame,
    period: int,
    buy_threshold: float,
    sell_threshold: float,
    close_col: str = "close",
) -> pd.DataFrame:
    """Generate ROC buy/sell signals from momentum thresholds.

    Signal rule:
    buy=1 when ROC is above ``buy_threshold`` and sell=1 when ROC is below
    ``sell_threshold``. Rows where ROC is NaN receive zero signals.
    """

    _validate_roc_inputs(
        df,
        int(period),
        float(buy_threshold),
        float(sell_threshold),
        close_col,
    )

    result = df.copy()
    roc = calculate_roc(result, int(period), close_col=close_col)
    valid_roc = roc.notna()
    result["roc"] = roc
    result["roc_buy_signal"] = ((roc > float(buy_threshold)) & valid_roc).astype(int)
    result["roc_sell_signal"] = ((roc < float(sell_threshold)) & valid_roc).astype(int)
    return result
