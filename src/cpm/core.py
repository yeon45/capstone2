"""Minimal local Critical Point Model implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _append_or_replace_point(
    points: list[dict],
    point: dict,
    T: int,
) -> None:
    """Append a point while enforcing alternating type and minimum gap."""

    if not points:
        points.append(point)
        return

    last = points[-1]
    if point["type"] == last["type"]:
        is_more_extreme = (
            point["type"] == "bottom" and point["price"] < last["price"]
        ) or (
            point["type"] == "top" and point["price"] > last["price"]
        )
        if is_more_extreme:
            points[-1] = point
        return

    if point["index"] - last["index"] < T:
        return

    points.append(point)


def detect_turning_points(prices: pd.Series, P: float, T: int) -> list[dict]:
    """Detect alternating CPM top and bottom points from Close prices."""

    if P <= 0:
        raise ValueError("P must be positive")
    if T < 0:
        raise ValueError("T must be non-negative")

    values = pd.to_numeric(prices, errors="coerce").to_numpy(dtype=float)
    valid_indices = np.flatnonzero(~np.isnan(values))
    if len(valid_indices) < 2:
        return []

    first_idx = int(valid_indices[0])
    first_price = float(values[first_idx])
    candidate_low_idx = first_idx
    candidate_low = first_price
    candidate_high_idx = first_idx
    candidate_high = first_price
    mode: str | None = None
    points: list[dict] = []

    for raw_idx in valid_indices[1:]:
        idx = int(raw_idx)
        price = float(values[idx])

        if price < candidate_low:
            candidate_low = price
            candidate_low_idx = idx
        if price > candidate_high:
            candidate_high = price
            candidate_high_idx = idx

        if mode is None:
            rise_from_low = price >= candidate_low * (1.0 + P)
            fall_from_high = price <= candidate_high * (1.0 - P)
            if rise_from_low:
                point = {"index": candidate_low_idx, "type": "bottom", "price": candidate_low}
                _append_or_replace_point(points, point, T)
                mode = "seeking_top"
                candidate_high = price
                candidate_high_idx = idx
            elif fall_from_high:
                point = {"index": candidate_high_idx, "type": "top", "price": candidate_high}
                _append_or_replace_point(points, point, T)
                mode = "seeking_bottom"
                candidate_low = price
                candidate_low_idx = idx
            continue

        if mode == "seeking_top":
            if price > candidate_high:
                candidate_high = price
                candidate_high_idx = idx
            elif price <= candidate_high * (1.0 - P):
                point = {"index": candidate_high_idx, "type": "top", "price": candidate_high}
                _append_or_replace_point(points, point, T)
                mode = "seeking_bottom"
                candidate_low = price
                candidate_low_idx = idx

        elif mode == "seeking_bottom":
            if price < candidate_low:
                candidate_low = price
                candidate_low_idx = idx
            elif price >= candidate_low * (1.0 + P):
                point = {"index": candidate_low_idx, "type": "bottom", "price": candidate_low}
                _append_or_replace_point(points, point, T)
                mode = "seeking_top"
                candidate_high = price
                candidate_high_idx = idx

    return points


def turning_points_to_label_array(n_rows: int, turning_points: list[dict]) -> np.ndarray:
    """Convert CPM turning point dictionaries to a label array."""

    labels = np.zeros(n_rows, dtype=int)
    for point in turning_points:
        idx = int(point["index"])
        if idx < 0 or idx >= n_rows:
            continue
        if point["type"] == "bottom":
            labels[idx] = 1
        elif point["type"] == "top":
            labels[idx] = -1
    return labels
