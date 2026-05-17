"""Fitness functions for indicator signals against CPM labels."""

from __future__ import annotations

from typing import Iterable

import numpy as np
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
    buy_label: int = 1,
    sell_label: int = -1,
    false_signal_penalty: float = 0.0,
) -> tuple[float, dict]:
    """Calculate signal fitness against fixed CPM turning labels."""

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
