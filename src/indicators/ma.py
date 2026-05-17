"""Moving Average indicator system from the ESN/GA trading paper."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


DEFAULT_MA_BOUNDS = {
    "n_min": 5,
    "n_max": 60,
    "N_min": 20,
    "N_max": 200,
    "a_min": 1.2,
    "a_max": 5.0,
    "b_min": 0.5,
    "b_max": 3.0,
    "c_min": 0.001,
    "c_max": 0.10,
}


def _bounds(bounds: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = DEFAULT_MA_BOUNDS.copy()
    if bounds:
        merged.update(bounds)
    return merged


def repair_ma_params(params: Sequence[float], bounds: dict | None = None) -> list[float]:
    """Validate and repair MA parameters.

    Parameter format:
    [n, N, a_buy, b_buy, c_buy, a_sell, b_sell, c_sell].
    """

    if len(params) != 8:
        raise ValueError(f"MA params must have length 8, got {len(params)}")

    b = _bounds(bounds)
    n = int(np.clip(round(params[0]), b["n_min"], b["n_max"]))
    N = int(np.clip(round(params[1]), b["N_min"], b["N_max"]))

    if N <= n:
        N = min(int(b["N_max"]), max(int(b["N_min"]), n + 1))
        if N <= n:
            n = max(int(b["n_min"]), N - 1)

    a_buy = float(np.clip(params[2], b["a_min"], b["a_max"]))
    b_buy = float(np.clip(params[3], b["b_min"], b["b_max"]))
    c_buy = float(np.clip(params[4], b["c_min"], b["c_max"]))
    a_sell = float(np.clip(params[5], b["a_min"], b["a_max"]))
    b_sell = float(np.clip(params[6], b["b_min"], b["b_max"]))
    c_sell = float(np.clip(params[7], b["c_min"], b["c_max"]))

    a_buy = max(a_buy, 1.000001)
    a_sell = max(a_sell, 1.000001)
    b_buy = max(b_buy, 1e-12)
    c_buy = max(c_buy, 1e-12)
    b_sell = max(b_sell, 1e-12)
    c_sell = max(c_sell, 1e-12)

    return [n, N, a_buy, b_buy, c_buy, a_sell, b_sell, c_sell]


def calculate_ma_z(
    df: pd.DataFrame,
    params: Sequence[float],
    close_col: str = "Close",
    normalize: bool = True,
) -> pd.Series:
    """Calculate z_t = MA(N) - MA(n), normalized by close by default."""

    if close_col not in df.columns:
        raise ValueError(f"Missing close column: {close_col}")

    n, N, *_ = repair_ma_params(params)
    close = pd.to_numeric(df[close_col], errors="coerce")
    short_ma = close.rolling(window=int(n), min_periods=int(n)).mean()
    long_ma = close.rolling(window=int(N), min_periods=int(N)).mean()
    z = long_ma - short_ma

    if normalize:
        z = z / close.replace(0, np.nan)

    return z


def generate_ma_signals(
    df: pd.DataFrame,
    params: Sequence[float],
    close_col: str = "Close",
    normalize: bool = True,
) -> pd.DataFrame:
    """Generate MA buy/sell signals using improved Golden/Dead Cross logic."""

    repaired = repair_ma_params(params)
    _, _, a_buy, b_buy, c_buy, a_sell, b_sell, c_sell = repaired
    result = df.copy()
    z = calculate_ma_z(result, repaired, close_col=close_col, normalize=normalize)
    z_values = z.to_numpy(dtype=float)

    buy_signals = np.zeros(len(result), dtype=int)
    sell_signals = np.zeros(len(result), dtype=int)
    positive_cross_index: int | None = None
    negative_cross_index: int | None = None
    prev_z: float | None = None

    for i, current_z in enumerate(z_values):
        if np.isnan(current_z):
            prev_z = None
            positive_cross_index = None
            negative_cross_index = None
            continue

        if prev_z is not None and prev_z <= 0 < current_z:
            positive_cross_index = i
            negative_cross_index = None
        elif prev_z is not None and prev_z >= 0 > current_z:
            negative_cross_index = i
            positive_cross_index = None

        if current_z <= 0:
            positive_cross_index = None
        if current_z >= 0:
            negative_cross_index = None

        if positive_cross_index is not None and current_z > 0:
            section = z_values[positive_cross_index : i + 1]
            section = section[~np.isnan(section)]
            if len(section) > 0:
                mz = float(np.max(section))
                if mz > b_buy * c_buy and current_z < min(mz / a_buy, c_buy):
                    buy_signals[i] = 1
                    positive_cross_index = None

        if negative_cross_index is not None and current_z < 0:
            section = -z_values[negative_cross_index : i + 1]
            section = section[~np.isnan(section)]
            if len(section) > 0:
                mw = float(np.max(section))
                wt = -current_z
                if mw > b_sell * c_sell and wt < min(mw / a_sell, c_sell):
                    sell_signals[i] = 1
                    negative_cross_index = None

        prev_z = current_z

    result["ma_z"] = z
    result["ma_buy_signal"] = buy_signals
    result["ma_sell_signal"] = sell_signals
    return result
