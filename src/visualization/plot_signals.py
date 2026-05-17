"""Indicator signal plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.paths import ensure_parent_dir


def plot_ma_signals(
    df: pd.DataFrame,
    output_path: str | Path,
    date_col: str = "Date",
    close_col: str = "Close",
    buy_signal_col: str = "ma_buy_signal",
    sell_signal_col: str = "ma_sell_signal",
) -> None:
    """Plot Close price with MA buy and sell signals."""

    required = [date_col, close_col, buy_signal_col, sell_signal_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for MA signal plot: {missing}")

    dates = pd.to_datetime(df[date_col])
    close = pd.to_numeric(df[close_col], errors="coerce")
    buys = df[buy_signal_col] == 1
    sells = df[sell_signal_col] == 1

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, close, label="Close", linewidth=1.0)
    ax.scatter(dates[buys], close[buys], marker="^", color="green", label="MA buy", s=35)
    ax.scatter(dates[sells], close[sells], marker="v", color="red", label="MA sell", s=35)
    ax.set_title("MA Signals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = Path(output_path)
    ensure_parent_dir(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
