"""Simple long-only backtest for ESN trading-score signals."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


PRICE_COLUMNS = ["Close", "close", "Adj Close", "adj_close"]
DATE_COLUMNS = ["date", "Date", "datetime", "Datetime"]


def infer_price_col(df: pd.DataFrame, price_col: str | None = None) -> str:
    """Return an available close-price column."""

    candidates = [price_col] if price_col else []
    candidates.extend(col for col in PRICE_COLUMNS if col not in candidates)
    for col in candidates:
        if col and col in df.columns:
            return col
    raise ValueError(f"No price column found. Checked: {candidates}")


def infer_date_values(df: pd.DataFrame) -> pd.Series:
    """Return date values from a common date column or from the index."""

    for col in DATE_COLUMNS:
        if col in df.columns:
            return df[col].reset_index(drop=True)
    return pd.Series(df.index, name="date").reset_index(drop=True)


def normalize_signal(value: Any) -> int:
    """Normalize numeric or string trade signals to {-1, 0, 1}."""

    if pd.isna(value):
        return 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"buy", "-1"}:
            return -1
        if lowered in {"sell", "1"}:
            return 1
        if lowered in {"hold", "0", ""}:
            return 0
    numeric = float(value)
    if numeric < 0:
        return -1
    if numeric > 0:
        return 1
    return 0


def run_long_only_backtest(
    df: pd.DataFrame,
    price_col: str | None = None,
    signal_col: str = "signal",
    initial_cash: float = 1.0,
    fee_rate: float = 0.0,
    allow_fractional: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run a long-only all-in/all-out backtest from ESN signals."""

    if signal_col not in df.columns:
        raise ValueError(f"Signal column not found: {signal_col}")
    if initial_cash <= 0.0:
        raise ValueError(f"initial_cash must be positive, got {initial_cash}.")
    if fee_rate < 0.0:
        raise ValueError(f"fee_rate must be non-negative, got {fee_rate}.")

    resolved_price_col = infer_price_col(df, price_col)
    working = df.copy().reset_index(drop=True)
    dates = infer_date_values(working)
    close = working[resolved_price_col].astype(float).reset_index(drop=True)
    signals = working[signal_col].map(normalize_signal).astype(int).reset_index(drop=True)

    cash = float(initial_cash)
    shares = 0.0
    rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []

    for i, price in enumerate(close):
        signal = int(signals.iloc[i])
        action = "hold"

        if signal == -1 and shares <= 0.0 and cash > 0.0:
            if allow_fractional:
                buy_shares = cash / (float(price) * (1.0 + fee_rate))
            else:
                buy_shares = math.floor(cash / (float(price) * (1.0 + fee_rate)))
            if buy_shares > 0.0:
                gross_cost = buy_shares * float(price)
                fee = gross_cost * fee_rate
                cash -= gross_cost + fee
                shares += buy_shares
                action = "buy"
        elif signal == 1 and shares > 0.0:
            sold_shares = shares
            gross_value = shares * float(price)
            fee = gross_value * fee_rate
            cash += gross_value - fee
            shares = 0.0
            action = "sell"

        portfolio_value = cash + shares * float(price)
        position = 1 if shares > 0.0 else 0
        if action in {"buy", "sell"}:
            trades.append(
                {
                    "date": dates.iloc[i],
                    "action": action,
                    "price": float(price),
                    "shares": float(shares if action == "buy" else sold_shares),
                    "cash_after": float(cash),
                    "portfolio_value_after": float(portfolio_value),
                }
            )
        rows.append(
            {
                "date": dates.iloc[i],
                "close": float(price),
                "signal": signal,
                "position": position,
                "cash": float(cash),
                "shares": float(shares),
                "portfolio_value": float(portfolio_value),
            }
        )

    equity_curve = pd.DataFrame(rows)
    if equity_curve.empty:
        equity_curve = _empty_equity_curve()
    equity_curve["daily_return"] = equity_curve["portfolio_value"].pct_change().fillna(0.0)
    equity_curve["cumulative_return"] = equity_curve["portfolio_value"] / float(initial_cash) - 1.0
    trades_df = pd.DataFrame(
        trades,
        columns=[
            "date",
            "action",
            "price",
            "shares",
            "cash_after",
            "portfolio_value_after",
        ],
    )
    summary = _strategy_summary(equity_curve, trades_df, initial_cash)
    summary["buy_and_hold"] = buy_and_hold_summary(
        df=working,
        price_col=resolved_price_col,
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        allow_fractional=allow_fractional,
    )
    return equity_curve, trades_df, summary


def buy_and_hold_summary(
    df: pd.DataFrame,
    price_col: str | None = None,
    initial_cash: float = 1.0,
    fee_rate: float = 0.0,
    allow_fractional: bool = True,
) -> dict[str, float]:
    """Return buy-and-hold metrics over the same period."""

    resolved_price_col = infer_price_col(df, price_col)
    close = df[resolved_price_col].astype(float).reset_index(drop=True)
    if close.empty:
        return _metric_summary(pd.Series(dtype=float), initial_cash)

    first_price = float(close.iloc[0])
    if allow_fractional:
        shares = initial_cash / (first_price * (1.0 + fee_rate))
    else:
        shares = math.floor(initial_cash / (first_price * (1.0 + fee_rate)))
    cash = initial_cash - shares * first_price * (1.0 + fee_rate)
    equity = cash + shares * close
    return _metric_summary(equity.astype(float), initial_cash)


def _strategy_summary(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    initial_cash: float,
) -> dict[str, Any]:
    metrics = _metric_summary(equity_curve["portfolio_value"], initial_cash)
    num_buy = int((trades["action"] == "buy").sum()) if not trades.empty else 0
    num_sell = int((trades["action"] == "sell").sum()) if not trades.empty else 0
    return {
        "initial_cash": float(initial_cash),
        **metrics,
        "num_trades": int(len(trades)),
        "num_buy": num_buy,
        "num_sell": num_sell,
        "exposure_ratio": float(equity_curve["position"].mean()) if len(equity_curve) else 0.0,
        "win_rate": _win_rate(trades),
    }


def _metric_summary(equity: pd.Series, initial_cash: float) -> dict[str, float]:
    if equity.empty:
        return {
            "final_value": float(initial_cash),
            "total_return": 0.0,
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
        }

    final_value = float(equity.iloc[-1])
    total_return = final_value / float(initial_cash) - 1.0
    n_days = len(equity)
    annualized_return = (final_value / float(initial_cash)) ** (252.0 / n_days) - 1.0
    cumulative_max = equity.cummax()
    drawdown = equity / cumulative_max - 1.0
    daily_return = equity.pct_change().fillna(0.0)
    std = float(daily_return.std(ddof=1))
    sharpe = float(np.sqrt(252.0) * daily_return.mean() / std) if std > 0.0 else 0.0
    return {
        "final_value": final_value,
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "max_drawdown": float(drawdown.min()),
        "sharpe_ratio": sharpe,
    }


def _win_rate(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    buys = trades[trades["action"] == "buy"].reset_index(drop=True)
    sells = trades[trades["action"] == "sell"].reset_index(drop=True)
    n_pairs = min(len(buys), len(sells))
    if n_pairs == 0:
        return 0.0
    wins = 0
    for i in range(n_pairs):
        if float(sells.loc[i, "portfolio_value_after"]) > float(buys.loc[i, "portfolio_value_after"]):
            wins += 1
    return float(wins / n_pairs)


def _empty_equity_curve() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "close",
            "signal",
            "position",
            "cash",
            "shares",
            "portfolio_value",
        ]
    )
