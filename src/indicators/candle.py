"""Candlestick pattern signals from the ESN/GA trading paper."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


EPSILON = 1e-12

DEFAULT_CANDLE_BOUNDS = {
    "a_min": 1.0,
    "a_max": 5.0,
    "b_min": 0.0,
    "b_max": 1.0,
    "c_min": 0.0,
    "c_max": 1.0,
    "d_min": 0.0,
    "d_max": 0.5,
    "e_min": 0.0,
    "e_max": 0.5,
    "f_min": 0.0,
    "f_max": 0.5,
    "g_min": 0.0,
    "g_max": 0.5,
}

CANDLE_SIGNAL_COLUMNS = [
    "candle_hammer_hanging_man_signal",
    "candle_dark_cloud_cover_signal",
    "candle_piercing_line_signal",
    "candle_bullish_engulfing_signal",
    "candle_bearish_engulfing_signal",
]


def _bounds(bounds: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = DEFAULT_CANDLE_BOUNDS.copy()
    if bounds:
        merged.update(bounds)
    return merged


def repair_candle_params(params: Sequence[float], bounds: dict | None = None) -> list[float]:
    """Validate and repair candle pattern parameters.

    Parameter format:
    [a, b, c, d, e, f, g].
    """

    if len(params) != 7:
        raise ValueError(f"Candle params must have length 7, got {len(params)}")

    b = _bounds(bounds)
    return [
        float(np.clip(params[0], b["a_min"], b["a_max"])),
        float(np.clip(params[1], b["b_min"], b["b_max"])),
        float(np.clip(params[2], b["c_min"], b["c_max"])),
        float(np.clip(params[3], b["d_min"], b["d_max"])),
        float(np.clip(params[4], b["e_min"], b["e_max"])),
        float(np.clip(params[5], b["f_min"], b["f_max"])),
        float(np.clip(params[6], b["g_min"], b["g_max"])),
    ]


def validate_ohlc_columns(
    df: pd.DataFrame,
    required_cols: list[str] | None = None,
) -> None:
    """Validate that a DataFrame contains required OHLC columns."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    cols = required_cols or ["Open", "High", "Low", "Close"]
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}")


def generate_candle_signals(
    df: pd.DataFrame,
    params: Sequence[float],
    open_col: str = "Open",
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
) -> pd.DataFrame:
    """Generate independent candle pattern signal columns.

    Signals are signed per pattern: bullish=1, bearish=-1, no signal=0.
    No aggregate candle buy/sell columns are created.
    """

    required_cols = [open_col, high_col, low_col, close_col]
    validate_ohlc_columns(df, required_cols=required_cols)
    a, b, c, d, e, f, g = repair_candle_params(params)

    result = df.copy()
    open_ = pd.to_numeric(result[open_col], errors="coerce")
    high = pd.to_numeric(result[high_col], errors="coerce")
    low = pd.to_numeric(result[low_col], errors="coerce")
    close = pd.to_numeric(result[close_col], errors="coerce")

    body = (close - open_).abs()
    previous_body = (close.shift(1) - open_.shift(1)).abs()
    safe_body = body.mask(body == 0, EPSILON)
    safe_previous_body = previous_body.mask(previous_body == 0, EPSILON)

    upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower_wick = pd.concat([open_, close], axis=1).min(axis=1) - low
    total_wick = upper_wick + lower_wick
    is_white = close > open_
    is_black = close < open_
    prev_is_white = close.shift(1) > open_.shift(1)
    prev_is_black = close.shift(1) < open_.shift(1)

    valid_ohlc = open_.notna() & high.notna() & low.notna() & close.notna()
    valid_candle = (
        valid_ohlc
        & (high >= pd.concat([open_, close], axis=1).max(axis=1))
        & (low <= pd.concat([open_, close], axis=1).min(axis=1))
        & (body > 0)
    )
    valid_previous = valid_candle.shift(1).fillna(False)

    hammer_shape = (
        (lower_wick / safe_body >= a)
        & (lower_wick > upper_wick)
        & (lower_wick > total_wick / 2)
        & valid_candle
    )
    result["candle_hammer_hanging_man_signal"] = np.select(
        [hammer_shape & is_white, hammer_shape & is_black],
        [1, -1],
        default=0,
    ).astype(int)

    dark_cloud_cover = (
        prev_is_white
        & is_black
        & (open_ >= close.shift(1))
        & (close <= close.shift(1) - (b * safe_previous_body))
        & (close > open_.shift(1))
        & valid_candle
        & valid_previous
    )
    result["candle_dark_cloud_cover_signal"] = np.where(dark_cloud_cover, -1, 0).astype(int)

    piercing_line = (
        prev_is_black
        & is_white
        & (open_ <= close.shift(1))
        & (close >= close.shift(1) + (c * safe_previous_body))
        & (close < open_.shift(1))
        & valid_candle
        & valid_previous
    )
    result["candle_piercing_line_signal"] = np.where(piercing_line, 1, 0).astype(int)

    bullish_engulfing = (
        prev_is_black
        & is_white
        & (open_ <= close.shift(1) - (d * safe_previous_body))
        & (close >= open_.shift(1) + (e * safe_previous_body))
        & valid_candle
        & valid_previous
    )
    result["candle_bullish_engulfing_signal"] = np.where(bullish_engulfing, 1, 0).astype(int)

    bearish_engulfing = (
        prev_is_white
        & is_black
        & (open_ >= close.shift(1) + (f * safe_previous_body))
        & (close <= open_.shift(1) - (g * safe_previous_body))
        & valid_candle
        & valid_previous
    )
    result["candle_bearish_engulfing_signal"] = np.where(bearish_engulfing, -1, 0).astype(int)
    return result


CANDLE_PATTERN_COLUMNS = CANDLE_SIGNAL_COLUMNS
