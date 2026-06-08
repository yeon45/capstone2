"""CPM label generation."""

from __future__ import annotations

import pandas as pd

from src.cpm.core import detect_turning_points, turning_points_to_label_array


def add_turning_labels(
    df: pd.DataFrame,
    P: float,
    T: int,
    close_col: str = "Close",
    label_col: str = "turning_label",
) -> pd.DataFrame:
    """Add CPM turning labels and point types.

    Label convention: bottom/buy=-1, top/sell=1, no turning point=0.
    """

    if close_col not in df.columns:
        raise ValueError(f"Missing close column: {close_col}")

    result = df.copy()
    turning_points = detect_turning_points(result[close_col], P=P, T=T)
    labels = turning_points_to_label_array(len(result), turning_points)
    point_types = [""] * len(result)
    for point in turning_points:
        idx = int(point["index"])
        if 0 <= idx < len(result):
            point_types[idx] = str(point["type"])

    result[label_col] = labels
    result["cpm_point_type"] = point_types
    return result
