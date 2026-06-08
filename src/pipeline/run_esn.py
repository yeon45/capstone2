"""Train, validate, and test an ESN on the final signal dataset."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from typing import Any

import numpy as np

with contextlib.redirect_stderr(io.StringIO()):
    import pandas as pd

from src.esn.dataset import ESNDataset, ESNSplit, load_esn_dataset
from src.esn.metrics import (
    classification_like_metrics,
    regression_metrics,
    score_to_signal,
)
from src.esn.model import EchoStateNetwork
from src.utils.config import load_config
from src.utils.io import save_dataframe, save_json
from src.utils.paths import (
    get_esn_config_used_path,
    get_esn_metrics_path,
    get_esn_predictions_path,
)


DEFAULT_ESN_CONFIG: dict[str, Any] = {
    "target_col": "triangle_target",
    "target_shift": 0,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "reservoir_size": 300,
    "spectral_radius": 0.9,
    "sparsity": 0.1,
    "input_scale": 0.5,
    "leaking_rate": 0.3,
    "ridge_alpha": 1.0,
    "washout": 30,
    "random_state": 42,
    "threshold": 0.3,
    "threshold_values": [0.2, 0.3, 0.4, 0.5],
    "buy_threshold_values": [0.2, 0.3, 0.4, 0.5],
    "sell_threshold_values": [-0.2, -0.3, -0.4, -0.5],
}


def main() -> None:
    """Run the single chronological split ESN pipeline."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/spy_1d.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ticker = str(config["ticker"])
    interval = str(config["interval"])
    esn_config = _load_esn_config(config)
    dataset = load_esn_dataset(args.config)
    washout = int(esn_config["washout"])

    _print_dataset_summary(dataset)
    if len(dataset.train.X) <= washout:
        raise ValueError(
            f"Train rows ({len(dataset.train.X)}) must be larger than washout ({washout})."
        )

    model = EchoStateNetwork(
        input_dim=len(dataset.feature_cols),
        reservoir_size=int(esn_config["reservoir_size"]),
        spectral_radius=float(esn_config["spectral_radius"]),
        sparsity=float(esn_config["sparsity"]),
        input_scale=float(esn_config["input_scale"]),
        leaking_rate=float(esn_config["leaking_rate"]),
        ridge_alpha=float(esn_config["ridge_alpha"]),
        washout=washout,
        random_state=int(esn_config["random_state"]),
    )
    model.fit(dataset.train.X, dataset.train.y)

    train_pred = np.asarray(model.predict(dataset.train.X), dtype=float)
    val_pred = np.asarray(model.predict(dataset.val.X), dtype=float)
    test_pred = np.asarray(model.predict(dataset.test.X), dtype=float)
    selected = _select_thresholds(
        dataset.val.y,
        val_pred,
        _threshold_values(esn_config),
    )

    train_signal = np.asarray(score_to_signal(train_pred, selected["threshold"]), dtype=int)
    val_signal = np.asarray(score_to_signal(val_pred, selected["threshold"]), dtype=int)
    test_signal = np.asarray(score_to_signal(test_pred, selected["threshold"]), dtype=int)

    metrics = _build_metrics(
        ticker=ticker,
        interval=interval,
        config_path=args.config,
        esn_config=esn_config,
        dataset=dataset,
        selected_thresholds=selected,
        train_pred=train_pred,
        val_pred=val_pred,
        test_pred=test_pred,
        train_signal=train_signal,
        val_signal=val_signal,
        test_signal=test_signal,
        washout=washout,
    )
    prediction_df = pd.concat(
        [
            _prediction_frame(dataset.train, train_pred, train_signal, "train"),
            _prediction_frame(dataset.val, val_pred, val_signal, "val"),
            _prediction_frame(dataset.test, test_pred, test_signal, "test"),
        ],
        ignore_index=True,
    )

    metrics_path = get_esn_metrics_path(ticker, interval)
    predictions_path = get_esn_predictions_path(ticker, interval)
    config_used_path = get_esn_config_used_path(ticker, interval)
    save_json(metrics, metrics_path)
    save_dataframe(prediction_df, predictions_path)
    save_json(metrics["config_used"], config_used_path)

    print("Selected thresholds:")
    print(json.dumps(selected, indent=2))
    print("Test metrics:")
    print(json.dumps(metrics["test"], indent=2))
    print(f"ESN metrics: {metrics_path}")
    print(f"ESN predictions: {predictions_path}")
    print(f"ESN config used: {config_used_path}")


def _load_esn_config(config: dict[str, Any]) -> dict[str, Any]:
    esn_config = DEFAULT_ESN_CONFIG.copy()
    esn_config.update(config.get("esn", {}))
    return esn_config


def _select_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold_values: list[float],
) -> dict[str, float]:
    best: dict[str, float] | None = None
    y_true_signal = np.asarray(score_to_signal(y_true, 0.0), dtype=int)
    for threshold in threshold_values:
        y_signal = np.asarray(score_to_signal(y_pred, threshold), dtype=int)
        metrics = classification_like_metrics(y_true_signal, y_signal)
        score = float(metrics["accuracy"])
        if best is None or score > best["validation_accuracy"]:
            best = {
                "threshold": float(threshold),
                "validation_accuracy": score,
            }
    if best is None:
        raise ValueError("No threshold combinations were provided.")
    return best


def _threshold_values(esn_config: dict[str, Any]) -> list[float]:
    if "threshold_values" in esn_config:
        values = [float(value) for value in esn_config["threshold_values"]]
    elif "threshold" in esn_config:
        values = [float(esn_config["threshold"])]
    else:
        values = sorted(
            {
                abs(float(value))
                for value in [
                    *esn_config.get("buy_threshold_values", []),
                    *esn_config.get("sell_threshold_values", []),
                ]
            }
        )
    if not values:
        raise ValueError("esn.threshold_values must contain at least one threshold.")
    if any(value < 0.0 for value in values):
        raise ValueError(f"Threshold values must be non-negative: {values}")
    return values


def _build_metrics(
    ticker: str,
    interval: str,
    config_path: str,
    esn_config: dict[str, Any],
    dataset: ESNDataset,
    selected_thresholds: dict[str, float],
    train_pred: np.ndarray,
    val_pred: np.ndarray,
    test_pred: np.ndarray,
    train_signal: np.ndarray,
    val_signal: np.ndarray,
    test_signal: np.ndarray,
    washout: int,
) -> dict[str, Any]:
    train_eval = slice(washout, None)
    return {
        "ticker": ticker,
        "interval": interval,
        "input_path": str(dataset.input_path),
        "feature_cols": dataset.feature_cols,
        "target_col": dataset.target_col,
        "target_shift": dataset.target_shift,
        "target_definition": "CPM turning points converted to triangle-wave trading score",
        "split_indices": dataset.split_indices,
        "selected_thresholds": selected_thresholds,
        "train": _split_metrics(
            dataset.train.y[train_eval],
            train_pred[train_eval],
            train_signal[train_eval],
        ),
        "validation": _split_metrics(dataset.val.y, val_pred, val_signal),
        "test": _split_metrics(dataset.test.y, test_pred, test_signal),
        "config_used": {
            "config_path": config_path,
            "ticker": ticker,
            "interval": interval,
            "esn": esn_config,
        },
    }


def _split_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_signal: np.ndarray,
) -> dict[str, Any]:
    y_true_signal = np.asarray(score_to_signal(y_true, 0.0), dtype=int)
    return {
        "regression": regression_metrics(y_true, y_pred),
        "classification_like": classification_like_metrics(y_true_signal, y_signal),
    }


def _prediction_frame(
    split: ESNSplit,
    y_pred: np.ndarray,
    y_signal: np.ndarray,
    split_name: str,
) -> pd.DataFrame:
    data: dict[str, Any] = {
        "y_true": split.y.astype(float),
        "trading_score": y_pred.astype(float),
        "signal": y_signal.astype(int),
        "split": split_name,
    }
    if split.dates is not None:
        data = {"date": split.dates, **data}
    return pd.DataFrame(data)


def _print_dataset_summary(dataset: ESNDataset) -> None:
    print(f"ESN input file: {dataset.input_path}")
    print(f"Feature shape: train={dataset.train.X.shape}, val={dataset.val.X.shape}, test={dataset.test.X.shape}")
    print(f"Target shape: train={dataset.train.y.shape}, val={dataset.val.y.shape}, test={dataset.test.y.shape}")
    _print_period("Train", dataset.train.dates)
    _print_period("Validation", dataset.val.dates)
    _print_period("Test", dataset.test.dates)


def _print_period(name: str, dates: np.ndarray | None) -> None:
    if dates is None or len(dates) == 0:
        return
    print(f"{name} period: {dates[0]} -> {dates[-1]}")


if __name__ == "__main__":
    main()
