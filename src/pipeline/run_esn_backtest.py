"""Run a long-only backtest from ESN trading-score outputs."""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path
from typing import Any

import numpy as np

with contextlib.redirect_stderr(io.StringIO()):
    import pandas as pd

from src.backtest.simple_backtest import infer_price_col, run_long_only_backtest
from src.esn.dataset import build_triangle_target, load_esn_dataset
from src.esn.metrics import score_to_signal
from src.esn.model import EchoStateNetwork
from src.pipeline.run_esn import _load_esn_config
from src.utils.config import load_config
from src.utils.io import load_dataframe, save_dataframe, save_json


BACKTEST_REQUIRED_COLUMNS = ["date", "close", "triangle_target", "trading_score", "signal"]
EXTRA_COLUMNS = [
    "ma_signal",
    "rsi_signal",
    "roc_signal",
    "stoch_signal",
    "candle_hammer_hanging_man_signal",
    "candle_dark_cloud_cover_signal",
    "candle_piercing_line_signal",
    "candle_bullish_engulfing_signal",
    "candle_bearish_engulfing_signal",
    "split",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/spy_1d_esn_best_backtest.yaml")
    parser.add_argument("--predictions", default="")
    parser.add_argument("--output-dir", default="results/esn_backtest")
    parser.add_argument("--initial-cash", type=float, default=10000.0)
    parser.add_argument("--fee-rate", type=float, default=0.0)
    parser.add_argument("--no-fractional", action="store_true")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_df = _load_or_build_predictions(args.config, args.predictions, output_dir)
    backtest_df = _select_split(prediction_df, args.split)
    if backtest_df.empty:
        raise ValueError(f"No rows available for split={args.split}.")

    equity_curve, trades, summary = run_long_only_backtest(
        backtest_df,
        price_col="close",
        signal_col="signal",
        initial_cash=args.initial_cash,
        fee_rate=args.fee_rate,
        allow_fractional=not args.no_fractional,
    )
    summary = {
        "split": args.split,
        "rows": int(len(backtest_df)),
        "strategy": summary,
    }

    save_dataframe(equity_curve, output_dir / "equity_curve.csv")
    save_dataframe(trades, output_dir / "trades.csv")
    save_json(summary, output_dir / "summary.json")

    print(f"Backtest split: {args.split}")
    print(f"Rows: {len(backtest_df)}")
    print(f"Equity curve: {output_dir / 'equity_curve.csv'}")
    print(f"Trades: {output_dir / 'trades.csv'}")
    print(f"Summary: {output_dir / 'summary.json'}")
    print(summary)


def _load_or_build_predictions(
    config_path: str,
    predictions_path: str,
    output_dir: Path,
) -> pd.DataFrame:
    if predictions_path:
        path = Path(predictions_path)
        if path.exists():
            df = load_dataframe(path)
            if _has_backtest_columns(df):
                return _canonicalize_prediction_columns(df)

    output_path = output_dir / "esn_predictions_for_backtest.csv"
    df = _build_predictions_for_backtest(config_path)
    save_dataframe(df, output_path)
    return df


def _has_backtest_columns(df: pd.DataFrame) -> bool:
    lower_to_original = {str(col).lower(): col for col in df.columns}
    return all(col in lower_to_original for col in BACKTEST_REQUIRED_COLUMNS)


def _canonicalize_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        lowered = str(col).lower()
        if lowered in {"date", "close", "triangle_target", "trading_score", "signal", "split"}:
            rename[col] = lowered
    return df.rename(columns=rename)


def _select_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    if "split" in df.columns:
        return df[df["split"].astype(str).str.lower() == split].reset_index(drop=True)
    if split != "test":
        raise ValueError("Prediction file has no split column; only split=test fallback is supported.")
    return df.reset_index(drop=True)


def _build_predictions_for_backtest(config_path: str) -> pd.DataFrame:
    config = load_config(config_path)
    esn_config = _load_esn_config(config)
    dataset = load_esn_dataset(config_path)
    model = EchoStateNetwork(
        input_dim=len(dataset.feature_cols),
        reservoir_size=int(esn_config["reservoir_size"]),
        spectral_radius=float(esn_config["spectral_radius"]),
        sparsity=float(esn_config["sparsity"]),
        input_scale=float(esn_config["input_scale"]),
        leaking_rate=float(esn_config["leaking_rate"]),
        ridge_alpha=float(esn_config["ridge_alpha"]),
        washout=int(esn_config["washout"]),
        random_state=int(esn_config["random_state"]),
    )
    model.fit(dataset.train.X, dataset.train.y)

    train_pred = np.asarray(model.predict(dataset.train.X), dtype=float)
    val_pred = np.asarray(model.predict(dataset.val.X), dtype=float)
    test_pred = np.asarray(model.predict(dataset.test.X), dtype=float)
    threshold = float(esn_config.get("threshold", 0.1))

    aligned_df = _aligned_input_frame(config, dataset.feature_cols)
    train_end = int(dataset.split_indices["train_end"])
    val_end = int(dataset.split_indices["val_end"])

    return pd.concat(
        [
            _prediction_frame(aligned_df.iloc[:train_end], train_pred, threshold, "train"),
            _prediction_frame(aligned_df.iloc[train_end:val_end], val_pred, threshold, "val"),
            _prediction_frame(aligned_df.iloc[val_end:], test_pred, threshold, "test"),
        ],
        ignore_index=True,
    )


def _aligned_input_frame(config: dict[str, Any], feature_cols: list[str]) -> pd.DataFrame:
    df = load_dataframe(config["input_path"]) if "input_path" in config else load_dataframe(
        Path("data") / "processed" / str(config["ticker"]) / str(config["interval"]) / f"{config['ticker']}_all_signals.csv"
    )
    esn_config = config.get("esn", {})
    target_col = str(esn_config.get("target_col", "triangle_target"))
    target_shift = int(esn_config.get("target_shift", 0))
    if target_col == "triangle_target" and target_col not in df.columns:
        df[target_col] = build_triangle_target(df, _turning_points_from_df(df))

    price_col = infer_price_col(df)
    date_col = _date_column(df)
    working = df.copy()
    working["_esn_target"] = working[target_col].shift(target_shift)
    subset_cols = [*feature_cols, "_esn_target", price_col]
    if date_col:
        subset_cols.append(date_col)
    working = working.dropna(subset=subset_cols).reset_index(drop=True)
    if date_col:
        working["date"] = working[date_col]
    else:
        working["date"] = working.index
    working["close"] = working[price_col].astype(float)
    working["triangle_target"] = working["_esn_target"].astype(float)
    return working


def _prediction_frame(
    frame: pd.DataFrame,
    y_pred: np.ndarray,
    threshold: float,
    split: str,
) -> pd.DataFrame:
    output_cols = ["date", "close", "triangle_target", *EXTRA_COLUMNS]
    available_cols = [col for col in output_cols if col in frame.columns and col != "split"]
    out = frame[available_cols].copy().reset_index(drop=True)
    out["trading_score"] = y_pred.astype(float)
    out["signal"] = np.asarray(score_to_signal(y_pred, threshold), dtype=int)
    out["split"] = split
    ordered = [col for col in [*BACKTEST_REQUIRED_COLUMNS, *EXTRA_COLUMNS] if col in out.columns]
    return out[ordered]


def _turning_points_from_df(df: pd.DataFrame) -> list[dict[str, Any]]:
    if "cpm_point_type" in df.columns:
        points = []
        for idx, point_type in df["cpm_point_type"].fillna("").items():
            if str(point_type).strip():
                points.append({"index": idx, "type": str(point_type)})
        if points:
            return points
    if "turning_label" in df.columns:
        points = []
        for idx, label in df["turning_label"].items():
            if int(label) == -1:
                points.append({"index": idx, "type": "bottom"})
            elif int(label) == 1:
                points.append({"index": idx, "type": "top"})
        if points:
            return points
    raise ValueError("Cannot build triangle_target; no CPM turning point columns found.")


def _date_column(df: pd.DataFrame) -> str | None:
    for col in ["Date", "date", "Datetime", "datetime"]:
        if col in df.columns:
            return col
    return None


if __name__ == "__main__":
    main()
