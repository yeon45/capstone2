"""Fitness functions for indicator signals against CPM labels."""

from __future__ import annotations

import contextlib
import io
from typing import Iterable

import numpy as np

with contextlib.redirect_stderr(io.StringIO()):
    import pandas as pd


def _match_events_with_forward_window(
    true_indices: Iterable[int],
    pred_indices: Iterable[int],
    window: int,
) -> tuple[int, int, int]:
    """Match predictions in [true_idx, true_idx + window] to true events."""

    if window < 0:
        raise ValueError("window must be non-negative")

    true_list = sorted(int(i) for i in true_indices)
    pred_list = sorted(int(i) for i in pred_indices)
    used_pred_positions: set[int] = set()
    tp = 0
    fn = 0

    for true_idx in true_list:
        matched_position = None
        for pos, pred_idx in enumerate(pred_list):
            if pos in used_pred_positions:
                continue
            if pred_idx < true_idx:
                continue
            if pred_idx > true_idx + window:
                break
            matched_position = pos
            break
        if matched_position is None:
            fn += 1
        else:
            used_pred_positions.add(matched_position)
            tp += 1

    fp = len(pred_list) - len(used_pred_positions)
    return tp, fp, fn


def _safe_f1(tp: int, fp: int, fn: int) -> float:
    """Compute F1 score safely."""

    if tp == 0:
        return 0.0
    return (2 * tp) / ((2 * tp) + fp + fn)


def calculate_signal_fitness(
    df: pd.DataFrame,
    label_col: str = "turning_label",
    buy_signal_col: str = "ma_buy_signal",
    sell_signal_col: str = "ma_sell_signal",
    window: int = 5,
    buy_label: int = -1,
    sell_label: int = 1,
    false_signal_penalty: float = 0.0,
) -> tuple[float, dict]:
    """Calculate signal fitness against CPM labels.

    CPM label convention: bottom/buy=-1, top/sell=1.
    """

    required = [label_col, buy_signal_col, sell_signal_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for fitness: {missing}")

    true_buy = np.flatnonzero((df[label_col] == buy_label).to_numpy()).tolist()
    true_sell = np.flatnonzero((df[label_col] == sell_label).to_numpy()).tolist()
    pred_buy = np.flatnonzero((df[buy_signal_col] == 1).to_numpy()).tolist()
    pred_sell = np.flatnonzero((df[sell_signal_col] == 1).to_numpy()).tolist()

    buy_tp, buy_fp, buy_fn = _match_events_with_forward_window(true_buy, pred_buy, window)
    sell_tp, sell_fp, sell_fn = _match_events_with_forward_window(true_sell, pred_sell, window)
    buy_f1 = _safe_f1(buy_tp, buy_fp, buy_fn)
    sell_f1 = _safe_f1(sell_tp, sell_fp, sell_fn)
    fitness = (buy_f1 + sell_f1) / 2

    if false_signal_penalty > 0:
        fitness -= false_signal_penalty * (buy_fp + sell_fp)

    fitness = max(0.0, float(fitness))
    details = {
        "fitness": fitness,
        "buy_f1": buy_f1,
        "sell_f1": sell_f1,
        "buy_tp": buy_tp,
        "buy_fp": buy_fp,
        "buy_fn": buy_fn,
        "sell_tp": sell_tp,
        "sell_fp": sell_fp,
        "sell_fn": sell_fn,
        "num_buy_signals": len(pred_buy),
        "num_sell_signals": len(pred_sell),
    }
    return fitness, details


def _calculate_atr(
    df: pd.DataFrame,
    high_col: str,
    low_col: str,
    close_col: str,
    atr_window: int,
) -> np.ndarray:
    """Calculate a simple rolling ATR used for price-error normalization."""

    if atr_window <= 0:
        raise ValueError("atr_window must be positive")

    high = pd.to_numeric(df[high_col], errors="coerce")
    low = pd.to_numeric(df[low_col], errors="coerce")
    close = pd.to_numeric(df[close_col], errors="coerce")
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(atr_window, min_periods=1).mean()
    fallback = true_range.replace(0, np.nan).median()
    if pd.isna(fallback) or fallback <= 0:
        fallback = close.replace(0, np.nan).abs().median()
    if pd.isna(fallback) or fallback <= 0:
        fallback = 1.0
    return atr.replace(0, np.nan).fillna(float(fallback)).to_numpy(dtype=float)


def _normalization_values(
    df: pd.DataFrame,
    normalize: str,
    price_col: str,
    high_col: str,
    low_col: str,
    close_col: str,
    atr_window: int,
) -> np.ndarray:
    if normalize == "atr":
        return _calculate_atr(df, high_col, low_col, close_col, atr_window)
    if normalize in {"price", "close"}:
        values = pd.to_numeric(df[price_col], errors="coerce").abs()
        fallback = values.replace(0, np.nan).median()
        if pd.isna(fallback) or fallback <= 0:
            fallback = 1.0
        return values.replace(0, np.nan).fillna(float(fallback)).to_numpy(dtype=float)
    if normalize in {"none", None}:
        return np.ones(len(df), dtype=float)
    raise ValueError(f"Unsupported normalize value: {normalize}")


def _directional_price_error_matches(
    df: pd.DataFrame,
    signal_indices: list[int],
    tp_indices: list[int],
    direction: int,
    price_col: str,
    high_col: str,
    low_col: str,
    normalizers: np.ndarray,
    max_time_window: int,
    tau: float,
    max_price_error: float,
) -> dict:
    """Greedily match signals and turning points by lowest normalized price error."""

    signal_prices = pd.to_numeric(df[price_col], errors="coerce").to_numpy(dtype=float)
    tp_price_col = low_col if direction == 1 else high_col
    tp_prices = pd.to_numeric(df[tp_price_col], errors="coerce").to_numpy(dtype=float)
    candidates: list[tuple[float, int, int]] = []

    for signal_idx in signal_indices:
        start = signal_idx - max_time_window
        end = signal_idx + max_time_window
        for tp_idx in tp_indices:
            if tp_idx < start or tp_idx > end:
                continue
            normalizer = normalizers[signal_idx]
            if not np.isfinite(normalizer) or normalizer <= 0:
                normalizer = 1.0
            price_error = abs(signal_prices[signal_idx] - tp_prices[tp_idx]) / normalizer
            if not np.isfinite(price_error) or price_error > max_price_error:
                continue
            candidates.append((float(price_error), signal_idx, tp_idx))

    used_signals: set[int] = set()
    used_tps: set[int] = set()
    matches: list[dict[str, float | int]] = []
    reward = 0.0

    for price_error, signal_idx, tp_idx in sorted(candidates, key=lambda item: item[0]):
        if signal_idx in used_signals or tp_idx in used_tps:
            continue
        used_signals.add(signal_idx)
        used_tps.add(tp_idx)
        reward += float(np.exp(-price_error / tau))
        matches.append(
            {
                "signal_index": signal_idx,
                "turning_point_index": tp_idx,
                "time_error": int(signal_idx - tp_idx),
                "price_error": price_error,
            }
        )

    duplicate_signals = 0
    for signal_idx in signal_indices:
        if signal_idx in used_signals:
            continue
        if any(abs(signal_idx - tp_idx) <= max_time_window for tp_idx in used_tps):
            duplicate_signals += 1

    return {
        "matches": matches,
        "reward": reward,
        "matched_signals": len(used_signals),
        "matched_turning_points": len(used_tps),
        "false_signals": len(signal_indices) - len(used_signals),
        "missed_turning_points": len(tp_indices) - len(used_tps),
        "duplicate_signals": duplicate_signals,
        "num_signals": len(signal_indices),
        "num_turning_points": len(tp_indices),
    }


def calculate_price_error_signal_fitness(
    df: pd.DataFrame,
    signal_col: str,
    label_col: str = "turning_label",
    price_col: str = "Close",
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
    max_time_window: int = 20,
    atr_window: int = 14,
    tau: float = 1.0,
    max_price_error: float = 2.0,
    false_signal_penalty: float = 0.5, ## 0.5 -> 2.0 roc가 너무 많이 찍힘.
    missed_tp_penalty: float = 1.0,
    duplicate_signal_penalty: float = 0.25,
    normalize: str = "atr",
) -> tuple[float, dict]:
    """Evaluate signed signals against CPM turning points.

    Signed signal convention: buy=-1, sell=1, hold=0.
    """

    required = [signal_col, label_col, price_col, high_col, low_col, close_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for price-error fitness: {missing}")
    if max_time_window < 0:
        raise ValueError("max_time_window must be non-negative")
    if tau <= 0:
        raise ValueError("tau must be positive")
    if max_price_error <= 0:
        raise ValueError("max_price_error must be positive")

    labels = pd.to_numeric(df[label_col], errors="coerce").fillna(0).to_numpy(dtype=int)
    signals = pd.to_numeric(df[signal_col], errors="coerce").fillna(0).to_numpy(dtype=float)
    normalizers = _normalization_values(
        df,
        normalize=normalize,
        price_col=price_col,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        atr_window=atr_window,
    )

    buy_signals = np.flatnonzero(signals < 0).astype(int).tolist()
    sell_signals = np.flatnonzero(signals > 0).astype(int).tolist()
    bottom_tps = np.flatnonzero(labels == -1).astype(int).tolist()
    top_tps = np.flatnonzero(labels == 1).astype(int).tolist()

    buy = _directional_price_error_matches(
        df,
        signal_indices=buy_signals,
        tp_indices=bottom_tps,
        direction=1,
        price_col=price_col,
        high_col=high_col,
        low_col=low_col,
        normalizers=normalizers,
        max_time_window=max_time_window,
        tau=tau,
        max_price_error=max_price_error,
    )
    sell = _directional_price_error_matches(
        df,
        signal_indices=sell_signals,
        tp_indices=top_tps,
        direction=-1,
        price_col=price_col,
        high_col=high_col,
        low_col=low_col,
        normalizers=normalizers,
        max_time_window=max_time_window,
        tau=tau,
        max_price_error=max_price_error,
    )

    def direction_score(stats: dict) -> float:
        denominator = (
            stats["matched_signals"]
            + false_signal_penalty * stats["false_signals"]
            + missed_tp_penalty * stats["missed_turning_points"]
            + duplicate_signal_penalty * stats["duplicate_signals"]
        )
        if denominator <= 0:
            return 0.0
        return float(stats["reward"] / denominator)

    buy_score = direction_score(buy)
    sell_score = direction_score(sell)
    fitness = float((buy_score + sell_score) / 2.0)
    details = {
        "fitness": fitness,
        "fitness_method": "price_error_signal_fitness",
        "buy_score": buy_score,
        "sell_score": sell_score,
        "num_buy_signals": buy["num_signals"],
        "num_sell_signals": sell["num_signals"],
        "num_bottom_turning_points": buy["num_turning_points"],
        "num_top_turning_points": sell["num_turning_points"],
        "buy_matches": buy["matched_signals"],
        "sell_matches": sell["matched_signals"],
        "buy_false_signals": buy["false_signals"],
        "sell_false_signals": sell["false_signals"],
        "buy_missed_turning_points": buy["missed_turning_points"],
        "sell_missed_turning_points": sell["missed_turning_points"],
        "buy_duplicate_signals": buy["duplicate_signals"],
        "sell_duplicate_signals": sell["duplicate_signals"],
        "match_params": {
            "max_time_window": max_time_window,
            "atr_window": atr_window,
            "tau": tau,
            "max_price_error": max_price_error,
            "false_signal_penalty": false_signal_penalty,
            "missed_tp_penalty": missed_tp_penalty,
            "duplicate_signal_penalty": duplicate_signal_penalty,
            "normalize": normalize,
        },
    }
    return fitness, details


def price_error_signal_fitness(
    df: pd.DataFrame,
    signal_col: str,
    label_col: str = "turning_label",
    price_col: str = "Close",
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
    max_time_window: int = 20,
    atr_window: int = 14,
    tau: float = 1.0,
    max_price_error: float = 2.0,
    false_signal_penalty: float = 0.5,
    missed_tp_penalty: float = 1.0,
    duplicate_signal_penalty: float = 0.25,
    normalize: str = "atr",
) -> float:
    """Return price-error signal fitness as a scalar for optimization."""

    fitness, _ = calculate_price_error_signal_fitness(
        df,
        signal_col=signal_col,
        label_col=label_col,
        price_col=price_col,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        max_time_window=max_time_window,
        atr_window=atr_window,
        tau=tau,
        max_price_error=max_price_error,
        false_signal_penalty=false_signal_penalty,
        missed_tp_penalty=missed_tp_penalty,
        duplicate_signal_penalty=duplicate_signal_penalty,
        normalize=normalize,
    )
    return fitness


def calculate_price_error_buy_sell_fitness(
    df: pd.DataFrame,
    buy_signal_col: str,
    sell_signal_col: str,
    label_col: str = "turning_label",
    price_col: str = "Close",
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
    max_time_window: int = 20,
    atr_window: int = 14,
    tau: float = 1.0,
    max_price_error: float = 2.0,
    false_signal_penalty: float = 0.5,
    missed_tp_penalty: float = 1.0,
    duplicate_signal_penalty: float = 0.25,
    normalize: str = "atr",
) -> tuple[float, dict]:
    """Evaluate separate buy/sell signal columns with price-error fitness."""

    required = [buy_signal_col, sell_signal_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required buy/sell signal columns: {missing}")

    signed_col = "__price_error_signed_signal__"
    temp_df = df.copy()
    buy = pd.to_numeric(temp_df[buy_signal_col], errors="coerce").fillna(0)
    sell = pd.to_numeric(temp_df[sell_signal_col], errors="coerce").fillna(0)
    temp_df[signed_col] = np.sign(sell - buy).astype(int)
    return calculate_price_error_signal_fitness(
        temp_df,
        signal_col=signed_col,
        label_col=label_col,
        price_col=price_col,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        max_time_window=max_time_window,
        atr_window=atr_window,
        tau=tau,
        max_price_error=max_price_error,
        false_signal_penalty=false_signal_penalty,
        missed_tp_penalty=missed_tp_penalty,
        duplicate_signal_penalty=duplicate_signal_penalty,
        normalize=normalize,
    )
