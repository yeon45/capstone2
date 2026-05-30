"""Rule-based candlestick pattern indicator system."""

from __future__ import annotations

import numpy as np
import pandas as pd


CANDLE_PATTERN_COLUMNS = [
    "candle_hammer_hanging_man",
    "candle_dark_cloud_cover",
    "candle_piercing_line",
    "candle_bullish_engulfing",
    "candle_bearish_engulfing",
]

CANDLE_BUY_PATTERNS = [
    "candle_piercing_line",
    "candle_bullish_engulfing",
]

CANDLE_SELL_PATTERNS = [
    "candle_dark_cloud_cover",
    "candle_bearish_engulfing",
]


def validate_ohlc_columns(
    df: pd.DataFrame,
    required_cols: list[str] | None = None,
) -> None:
    """Validate that a DataFrame contains required OHLC columns."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    cols = required_cols or ["open", "high", "low", "close"]
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required OHLC columns: {missing}")


def add_candle_patterns(
    df: pd.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    hammer_ratio: float = 2.0,
    upper_shadow_ratio: float = 1.0,
    use_gap_condition: bool = False,
) -> pd.DataFrame:
    """Add paper-specified rule-based candlestick pattern columns.

    Common candle quantities are:
    body=abs(close-open), candle_range=high-low,
    upper_shadow=high-max(open, close), lower_shadow=min(open, close)-low,
    bullish=close>open, and bearish=close<open.

    Implemented default patterns match the paper scope: Hammer/Hanging Man,
    Dark Cloud Cover, Piercing Line, and Engulfing Pattern. Bullish and bearish
    engulfing are split to preserve signal direction. Rows with NaN OHLC values,
    zero candle range, or invalid previous candles receive zero signals.
    """

    required_cols = [open_col, high_col, low_col, close_col]
    validate_ohlc_columns(df, required_cols=required_cols)

    if hammer_ratio <= 0:
        raise ValueError("hammer_ratio must be greater than 0")
    if upper_shadow_ratio < 0:
        raise ValueError("upper_shadow_ratio must be non-negative")

    result = df.copy()
    open_ = pd.to_numeric(result[open_col], errors="coerce")
    high = pd.to_numeric(result[high_col], errors="coerce")
    low = pd.to_numeric(result[low_col], errors="coerce")
    close = pd.to_numeric(result[close_col], errors="coerce")

    valid_ohlc = open_.notna() & high.notna() & low.notna() & close.notna()
    candle_range = high - low
    body = (close - open_).abs()
    upper_shadow = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower_shadow = pd.concat([open_, close], axis=1).min(axis=1) - low
    bullish = close > open_
    bearish = close < open_
    valid_candle = valid_ohlc & (candle_range > 0)
    valid_body = valid_candle & (body > 0)

    # TODO: Trend context can be added later to distinguish hammer from hanging man.
    hammer_hanging_man = (
        ((lower_shadow / body) > float(hammer_ratio))
        & ((upper_shadow / body) < float(upper_shadow_ratio))
        & valid_body
    )
    result["candle_hammer_hanging_man"] = hammer_hanging_man.astype(int)

    prev_open = open_.shift(1)
    prev_close = close.shift(1)
    prev_bullish = bullish.shift(1).fillna(False)
    prev_bearish = bearish.shift(1).fillna(False)
    prev_valid = valid_candle.shift(1).fillna(False)
    prev_midpoint = (prev_open + prev_close) / 2

    dark_cloud_cover = (
        prev_bullish
        & bearish
        & (close < prev_midpoint)
        & (close > prev_open)
        & valid_candle
        & prev_valid
    )
    piercing_line = (
        prev_bearish
        & bullish
        & (close > prev_midpoint)
        & (close < prev_open)
        & valid_candle
        & prev_valid
    )

    if use_gap_condition:
        dark_cloud_cover = dark_cloud_cover & (open_ > prev_close)
        piercing_line = piercing_line & (open_ < prev_close)

    bullish_engulfing = (
        prev_bearish
        & bullish
        & (open_ <= prev_close)
        & (close >= prev_open)
        & valid_candle
        & prev_valid
    )
    bearish_engulfing = (
        prev_bullish
        & bearish
        & (open_ >= prev_close)
        & (close <= prev_open)
        & valid_candle
        & prev_valid
    )

    result["candle_dark_cloud_cover"] = dark_cloud_cover.astype(int)
    result["candle_piercing_line"] = piercing_line.astype(int)
    result["candle_bullish_engulfing"] = bullish_engulfing.astype(int)
    result["candle_bearish_engulfing"] = bearish_engulfing.astype(int)
    result["candle_buy_signal"] = result[CANDLE_BUY_PATTERNS].any(axis=1).astype(int)
    result["candle_sell_signal"] = result[CANDLE_SELL_PATTERNS].any(axis=1).astype(int)
    return result
