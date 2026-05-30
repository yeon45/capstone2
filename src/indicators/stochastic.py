"""Stochastic Oscillator indicator system."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


DEFAULT_STOCHASTIC_BOUNDS = {
    "k_period_min": 5,
    "k_period_max": 30,
    "d_period_min": 3,
    "d_period_max": 10,
    "lower_min": 10.0,
    "lower_max": 40.0,
    "upper_min": 60.0,
    "upper_max": 90.0,
}


def _bounds(bounds: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = DEFAULT_STOCHASTIC_BOUNDS.copy()
    if bounds:
        merged.update(bounds)
    return merged


def repair_stochastic_params(params: Sequence[float], bounds: dict | None = None) -> list[float]:
    """Validate and repair Stochastic parameters.

    Parameter format:
    [k_period, d_period, lower_threshold, upper_threshold].
    """

    if len(params) != 4:
        raise ValueError(f"Stochastic params must have length 4, got {len(params)}")

    b = _bounds(bounds)
    k_period = int(np.clip(round(params[0]), b["k_period_min"], b["k_period_max"]))
    d_period = int(np.clip(round(params[1]), b["d_period_min"], b["d_period_max"]))
    lower = float(np.clip(params[2], b["lower_min"], b["lower_max"]))
    upper = float(np.clip(params[3], b["upper_min"], b["upper_max"]))

    if lower >= upper:
        midpoint = (float(b["lower_max"]) + float(b["upper_min"])) / 2
        lower = min(float(b["lower_max"]), midpoint - 1.0)
        upper = max(float(b["upper_min"]), midpoint + 1.0)

    return [k_period, d_period, lower, upper]


def _validate_stochastic_inputs(
    df: pd.DataFrame,
    k_period: int,
    d_period: int,
    lower_threshold: float,
    upper_threshold: float,
    high_col: str,
    low_col: str,
    close_col: str,
) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    required = [high_col, low_col, close_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    if k_period <= 1:
        raise ValueError("k_period must be greater than 1")
    if d_period <= 0:
        raise ValueError("d_period must be greater than 0")
    if not 0 <= lower_threshold <= 100:
        raise ValueError("lower_threshold must be between 0 and 100")
    if not 0 <= upper_threshold <= 100:
        raise ValueError("upper_threshold must be between 0 and 100")
    if lower_threshold >= upper_threshold:
        raise ValueError("lower_threshold must be less than upper_threshold")


def calculate_stochastic(
    df: pd.DataFrame,
    k_period: int,
    d_period: int,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> tuple[pd.Series, pd.Series]:
    """Calculate Stochastic Oscillator %K and %D.

    %K is calculated as ((close - rolling_low) / (rolling_high - rolling_low)) * 100
    over ``k_period``. %D is a simple moving average of %K over ``d_period``.
    Rows where the high-low range is zero produce NaN instead of infinite values.
    """

    _validate_stochastic_inputs(
        df,
        int(k_period),
        int(d_period),
        20.0,
        80.0,
        high_col,
        low_col,
        close_col,
    )

    high = pd.to_numeric(df[high_col], errors="coerce")
    low = pd.to_numeric(df[low_col], errors="coerce")
    close = pd.to_numeric(df[close_col], errors="coerce")

    lowest_low = low.rolling(window=int(k_period), min_periods=int(k_period)).min()
    highest_high = high.rolling(window=int(k_period), min_periods=int(k_period)).max()
    denominator = (highest_high - lowest_low).replace(0, np.nan)
    stoch_k = ((close - lowest_low) / denominator) * 100
    stoch_d = stoch_k.rolling(window=int(d_period), min_periods=int(d_period)).mean()
    return stoch_k, stoch_d


def generate_stochastic_signals(
    df: pd.DataFrame,
    k_period: int,
    d_period: int,
    lower_threshold: float,
    upper_threshold: float,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    use_cross: bool = False,
) -> pd.DataFrame:
    """Generate Stochastic Oscillator buy/sell signals.

    By default, threshold signals are used: buy=1 when %K is below
    ``lower_threshold`` and sell=1 when %K is above ``upper_threshold``. When
    ``use_cross`` is True, buy/sell signals require %K/%D crossovers near the
    lower/upper threshold zones. Rows where %K or %D is NaN receive zero signals.
    """

    _validate_stochastic_inputs(
        df,
        int(k_period),
        int(d_period),
        float(lower_threshold),
        float(upper_threshold),
        high_col,
        low_col,
        close_col,
    )

    result = df.copy()
    stoch_k, stoch_d = calculate_stochastic(
        result,
        int(k_period),
        int(d_period),
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
    )
    result["stoch_k"] = stoch_k
    result["stoch_d"] = stoch_d

    if use_cross:
        prev_k = stoch_k.shift(1)
        prev_d = stoch_d.shift(1)
        valid = stoch_k.notna() & stoch_d.notna() & prev_k.notna() & prev_d.notna()
        buy_signal = (
            (prev_k <= prev_d)
            & (stoch_k > stoch_d)
            & ((stoch_k <= float(lower_threshold)) | (prev_k <= float(lower_threshold)))
            & valid
        )
        sell_signal = (
            (prev_k >= prev_d)
            & (stoch_k < stoch_d)
            & ((stoch_k >= float(upper_threshold)) | (prev_k >= float(upper_threshold)))
            & valid
        )
    else:
        valid = stoch_k.notna()
        buy_signal = (stoch_k < float(lower_threshold)) & valid
        sell_signal = (stoch_k > float(upper_threshold)) & valid

    result["stoch_buy_signal"] = buy_signal.astype(int)
    result["stoch_sell_signal"] = sell_signal.astype(int)
    return result
