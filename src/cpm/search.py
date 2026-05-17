"""CPM parameter grid search."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.cpm.core import detect_turning_points


def evaluate_cpm_params(prices: pd.Series, P: float, T: int) -> dict[str, float | int]:
    """Evaluate one CPM P,T combination with heuristic and movement metrics."""

    turning_points = detect_turning_points(prices, P=P, T=T)
    num_points = len(turning_points)
    num_buy = sum(point["type"] == "bottom" for point in turning_points)
    num_sell = sum(point["type"] == "top" for point in turning_points)
    indices = [int(point["index"]) for point in turning_points]
    point_prices = np.array([float(point["price"]) for point in turning_points], dtype=float)

    if len(indices) >= 2:
        gaps = np.diff(indices)
        mean_gap = float(np.mean(gaps))
    else:
        mean_gap = 0.0

    if len(point_prices) >= 2:
        previous_prices = point_prices[:-1]
        next_prices = point_prices[1:]
        valid = previous_prices != 0
        abs_moves = np.abs((next_prices[valid] - previous_prices[valid]) / previous_prices[valid])
        total_abs_move = float(np.sum(abs_moves))
        avg_abs_move = float(np.mean(abs_moves)) if len(abs_moves) > 0 else 0.0
    else:
        total_abs_move = 0.0
        avg_abs_move = 0.0

    efficiency = total_abs_move / max(num_points, 1)

    target_points = len(prices) / 60
    point_score = float(np.exp(-abs(num_points - target_points) / max(target_points, 1)))
    balance_score = 1.0 - abs(num_buy - num_sell) / max(num_points, 1)
    gap_score = min(mean_gap / max(T, 1), 1.0) if mean_gap > 0 else 0.0
    score = 0.5 * point_score + 0.3 * balance_score + 0.2 * gap_score
    coverage = num_points / max(len(prices), 1)

    return {
        "P": float(P),
        "T": int(T),
        "num_points": int(num_points),
        "num_buy": int(num_buy),
        "num_sell": int(num_sell),
        "mean_gap": mean_gap,
        "coverage": float(coverage),
        "total_abs_move": total_abs_move,
        "avg_abs_move": avg_abs_move,
        "efficiency": float(efficiency),
        "score": float(score),
    }


def grid_search_cpm(
    prices: pd.Series,
    p_values: list[float],
    t_values: list[int],
) -> pd.DataFrame:
    """Evaluate every P,T pair and return results sorted by score."""

    rows = [evaluate_cpm_params(prices, P, T) for P in p_values for T in t_values]
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def select_pareto_front(search_df: pd.DataFrame) -> pd.DataFrame:
    """Return non-dominated CPM rows using total_abs_move and num_points.

    Objectives:
    - maximize total_abs_move
    - minimize num_points
    """

    required = ["total_abs_move", "num_points"]
    missing = [col for col in required if col not in search_df.columns]
    if missing:
        raise ValueError(f"CPM search results missing required Pareto columns: {missing}")

    rows = search_df.reset_index(drop=True).copy()
    non_dominated_indices: list[int] = []

    for i, current in rows.iterrows():
        dominated = False
        for j, other in rows.iterrows():
            if i == j:
                continue

            at_least_as_good = (
                other["total_abs_move"] >= current["total_abs_move"]
                and other["num_points"] <= current["num_points"]
            )
            strictly_better = (
                other["total_abs_move"] > current["total_abs_move"]
                or other["num_points"] < current["num_points"]
            )
            if at_least_as_good and strictly_better:
                dominated = True
                break

        if not dominated:
            non_dominated_indices.append(i)

    return (
        rows.loc[non_dominated_indices]
        .sort_values(["num_points", "total_abs_move"], ascending=[True, False])
        .reset_index(drop=True)
    )


def _normalize(values: pd.Series) -> np.ndarray:
    """Normalize values to [0, 1], returning zeros when the range is flat."""

    array = values.to_numpy(dtype=float)
    value_range = float(np.max(array) - np.min(array)) if len(array) > 0 else 0.0
    if value_range == 0:
        return np.zeros(len(array), dtype=float)
    return (array - float(np.min(array))) / value_range


def select_pareto_knee(search_df: pd.DataFrame) -> dict:
    """Select the Pareto-front knee balancing move capture and point count."""

    front = select_pareto_front(search_df)
    if front.empty:
        raise ValueError("Pareto front is empty")
    if len(front) == 1:
        return front.iloc[0].to_dict()

    front = front.sort_values("num_points", ascending=True).reset_index(drop=True)
    x = _normalize(front["num_points"])
    y = _normalize(front["total_abs_move"])
    points = np.column_stack([x, y])
    start = points[0]
    end = points[-1]
    line = end - start
    line_norm = float(np.linalg.norm(line))

    if line_norm == 0:
        knee_idx = int(front["score"].to_numpy(dtype=float).argmax()) if "score" in front else 0
    else:
        deltas = points - start
        distances = np.abs((line[0] * deltas[:, 1]) - (line[1] * deltas[:, 0])) / line_norm
        knee_idx = int(np.argmax(distances))

    return front.iloc[knee_idx].to_dict()


def select_best_cpm_params(search_df: pd.DataFrame, method: str = "pareto_knee") -> dict:
    """Select the best CPM parameter row from search results."""

    if search_df.empty:
        raise ValueError("CPM search results are empty")

    if method == "score":
        return search_df.sort_values("score", ascending=False).iloc[0].to_dict()
    if method in {"pareto_knee", "knee", "knee_log"}:
        return select_pareto_knee(search_df)

    supported_future = {"saturation", "constrained", "curvature"}
    if method in supported_future:
        raise NotImplementedError(
            f"CPM selection method '{method}' is not implemented; supported methods are "
            "'score', 'pareto_knee', 'knee', and 'knee_log'."
        )

    raise ValueError(f"Unsupported CPM selection method: {method}")
