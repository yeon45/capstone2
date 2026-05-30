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


def plot_rsi_signals(
    df: pd.DataFrame,
    output_path: str | Path,
    date_col: str = "Date",
    close_col: str = "Close",
    rsi_col: str = "rsi",
    buy_signal_col: str = "rsi_buy_signal",
    sell_signal_col: str = "rsi_sell_signal",
    lower_threshold: float = 30.0,
    upper_threshold: float = 70.0,
) -> None:
    """Plot Close price with RSI buy/sell signals and an RSI subplot."""

    required = [date_col, close_col, rsi_col, buy_signal_col, sell_signal_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for RSI signal plot: {missing}")

    dates = pd.to_datetime(df[date_col])
    close = pd.to_numeric(df[close_col], errors="coerce")
    rsi = pd.to_numeric(df[rsi_col], errors="coerce")
    buys = df[buy_signal_col] == 1
    sells = df[sell_signal_col] == 1

    fig, (price_ax, rsi_ax) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    price_ax.plot(dates, close, label="Close", linewidth=1.0)
    price_ax.scatter(dates[buys], close[buys], marker="^", color="green", label="RSI buy", s=35)
    price_ax.scatter(dates[sells], close[sells], marker="v", color="red", label="RSI sell", s=35)
    price_ax.set_title("RSI Signals")
    price_ax.set_ylabel("Close")
    price_ax.legend()
    price_ax.grid(True, alpha=0.3)

    rsi_ax.plot(dates, rsi, label="RSI", color="purple", linewidth=1.0)
    rsi_ax.axhline(lower_threshold, color="green", linestyle="--", linewidth=0.8, alpha=0.7)
    rsi_ax.axhline(upper_threshold, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    rsi_ax.set_xlabel("Date")
    rsi_ax.set_ylabel("RSI")
    rsi_ax.set_ylim(0, 100)
    rsi_ax.legend()
    rsi_ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = Path(output_path)
    ensure_parent_dir(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_signals(
    df: pd.DataFrame,
    output_path: str | Path,
    date_col: str = "Date",
    close_col: str = "Close",
    roc_col: str = "roc",
    buy_signal_col: str = "roc_buy_signal",
    sell_signal_col: str = "roc_sell_signal",
    buy_threshold: float = 1.0,
    sell_threshold: float = -1.0,
) -> None:
    """Plot Close price with ROC buy/sell signals and a ROC subplot."""

    required = [date_col, close_col, roc_col, buy_signal_col, sell_signal_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for ROC signal plot: {missing}")

    dates = pd.to_datetime(df[date_col])
    close = pd.to_numeric(df[close_col], errors="coerce")
    roc = pd.to_numeric(df[roc_col], errors="coerce")
    buys = df[buy_signal_col] == 1
    sells = df[sell_signal_col] == 1

    fig, (price_ax, roc_ax) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    price_ax.plot(dates, close, label="Close", linewidth=1.0)
    price_ax.scatter(dates[buys], close[buys], marker="^", color="green", label="ROC buy", s=35)
    price_ax.scatter(dates[sells], close[sells], marker="v", color="red", label="ROC sell", s=35)
    price_ax.set_title("ROC Signals")
    price_ax.set_ylabel("Close")
    price_ax.legend()
    price_ax.grid(True, alpha=0.3)

    roc_ax.plot(dates, roc, label="ROC", color="darkorange", linewidth=1.0)
    roc_ax.axhline(0, color="black", linestyle="-", linewidth=0.7, alpha=0.5)
    roc_ax.axhline(buy_threshold, color="green", linestyle="--", linewidth=0.8, alpha=0.7)
    roc_ax.axhline(sell_threshold, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    roc_ax.set_xlabel("Date")
    roc_ax.set_ylabel("ROC")
    roc_ax.legend()
    roc_ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = Path(output_path)
    ensure_parent_dir(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_stochastic_signals(
    df: pd.DataFrame,
    output_path: str | Path,
    date_col: str = "Date",
    close_col: str = "Close",
    k_col: str = "stoch_k",
    d_col: str = "stoch_d",
    buy_signal_col: str = "stoch_buy_signal",
    sell_signal_col: str = "stoch_sell_signal",
    lower_threshold: float = 20.0,
    upper_threshold: float = 80.0,
) -> None:
    """Plot Close price with Stochastic signals and %K/%D subplot."""

    required = [date_col, close_col, k_col, d_col, buy_signal_col, sell_signal_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for Stochastic signal plot: {missing}")

    dates = pd.to_datetime(df[date_col])
    close = pd.to_numeric(df[close_col], errors="coerce")
    stoch_k = pd.to_numeric(df[k_col], errors="coerce")
    stoch_d = pd.to_numeric(df[d_col], errors="coerce")
    buys = df[buy_signal_col] == 1
    sells = df[sell_signal_col] == 1

    fig, (price_ax, stoch_ax) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    price_ax.plot(dates, close, label="Close", linewidth=1.0)
    price_ax.scatter(
        dates[buys],
        close[buys],
        marker="^",
        color="green",
        label="Stochastic buy",
        s=35,
    )
    price_ax.scatter(
        dates[sells],
        close[sells],
        marker="v",
        color="red",
        label="Stochastic sell",
        s=35,
    )
    price_ax.set_title("Stochastic Signals")
    price_ax.set_ylabel("Close")
    price_ax.legend()
    price_ax.grid(True, alpha=0.3)

    stoch_ax.plot(dates, stoch_k, label="%K", color="tab:blue", linewidth=1.0)
    stoch_ax.plot(dates, stoch_d, label="%D", color="darkorange", linewidth=1.0)
    stoch_ax.axhline(lower_threshold, color="green", linestyle="--", linewidth=0.8, alpha=0.7)
    stoch_ax.axhline(upper_threshold, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    stoch_ax.set_xlabel("Date")
    stoch_ax.set_ylabel("Stochastic")
    stoch_ax.set_ylim(0, 100)
    stoch_ax.legend()
    stoch_ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = Path(output_path)
    ensure_parent_dir(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_candle_patterns(
    df: pd.DataFrame,
    output_path: str | Path,
    date_col: str = "Date",
    close_col: str = "Close",
    hammer_col: str = "candle_hammer_hanging_man",
    buy_signal_col: str = "candle_buy_signal",
    sell_signal_col: str = "candle_sell_signal",
) -> None:
    """Plot Close price with aggregate candle buy and sell signals."""

    required = [date_col, close_col, hammer_col, buy_signal_col, sell_signal_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for candle pattern plot: {missing}")

    dates = pd.to_datetime(df[date_col])
    close = pd.to_numeric(df[close_col], errors="coerce")
    hammers = df[hammer_col] == 1
    buys = df[buy_signal_col] == 1
    sells = df[sell_signal_col] == 1

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, close, label="Close", linewidth=1.0)
    ax.scatter(
        dates[hammers],
        close[hammers],
        marker="o",
        color="royalblue",
        label="Hammer/Hanging Man",
        s=28,
        alpha=0.7,
    )
    ax.scatter(dates[buys], close[buys], marker="^", color="green", label="Candle buy", s=35)
    ax.scatter(dates[sells], close[sells], marker="v", color="red", label="Candle sell", s=35)
    ax.set_title("Candle Pattern Signals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = Path(output_path)
    ensure_parent_dir(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_candle_pattern_counts(
    df: pd.DataFrame,
    output_path: str | Path,
    pattern_cols: list[str],
) -> None:
    """Plot occurrence counts for individual candle pattern columns."""

    missing = [col for col in pattern_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for candle pattern counts: {missing}")

    counts = df[pattern_cols].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    counts.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Candle Pattern Counts")
    ax.set_xlabel("Pattern")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    path = Path(output_path)
    ensure_parent_dir(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
