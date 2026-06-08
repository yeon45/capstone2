"""CPM turning point plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.paths import ensure_parent_dir


def plot_cpm_turning_points(
    df: pd.DataFrame,
    output_path: str | Path,
    date_col: str = "Date",
    close_col: str = "Close",
    label_col: str = "turning_label",
) -> None:
    """Plot Close price with CPM bottom and top labels."""

    required = [date_col, close_col, label_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for CPM plot: {missing}")

    dates = pd.to_datetime(df[date_col])
    close = pd.to_numeric(df[close_col], errors="coerce")
    bottoms = df[label_col] == -1
    tops = df[label_col] == 1

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, close, label="Close", linewidth=1.0)
    ax.scatter(dates[bottoms], close[bottoms], marker="^", color="green", label="CPM bottom", s=35)
    ax.scatter(dates[tops], close[tops], marker="v", color="red", label="CPM top", s=35)
    ax.set_title("CPM Turning Points")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = Path(output_path)
    ensure_parent_dir(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
