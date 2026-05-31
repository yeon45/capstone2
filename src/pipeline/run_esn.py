"""Train an ESN classifier and save next-step turning-label predictions."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from typing import Any

import numpy as np

try:
    with contextlib.redirect_stderr(io.StringIO()):
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
        )
except Exception:  # pragma: no cover - depends on local binary package state
    accuracy_score = None
    classification_report = None
    confusion_matrix = None
    f1_score = None

from src.models.esn import ESNClassifier
from src.pipeline.build_esn_dataset import ESN_INPUT_SIGNAL_COLUMNS
from src.utils.config import load_config
from src.utils.io import load_dataframe, save_dataframe, save_json
from src.utils.paths import (
    get_all_indicator_signals_path,
    get_esn_metrics_path,
    get_esn_predictions_path,
)


DEFAULT_ESN_CONFIG = {
    "reservoir_size": 200,
    "spectral_radius": 0.9,
    "sparsity": 0.1,
    "leaking_rate": 0.3,
    "ridge_alpha": 1.0,
    "random_state": 42,
    "washout": 30,
    "class_weight": "balanced",
    "train_ratio": 0.7,
}

OUTPUT_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "turning_label",
    "target",
    "esn_pred",
    "split",
    *ESN_INPUT_SIGNAL_COLUMNS,
]


def _load_esn_config(config: dict[str, Any]) -> dict[str, Any]:
    esn_config = DEFAULT_ESN_CONFIG.copy()
    esn_config.update(config.get("esn", {}))
    return esn_config


def _validate_input_columns(df) -> None:
    required_columns = [
        "turning_label",
        *ESN_INPUT_SIGNAL_COLUMNS,
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"All-indicator signal data missing required columns: {missing}")


def _build_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = [-1, 0, 1]
    if classification_report is None:
        return _build_metrics_numpy(y_true, y_pred, labels)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def _build_metrics_numpy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = _confusion_matrix_numpy(y_true, y_pred, labels)
    report: dict[str, Any] = {}
    supports = cm.sum(axis=1)
    f1_values = []
    weighted_f1_sum = 0.0

    for idx, label in enumerate(labels):
        tp = float(cm[idx, idx])
        fp = float(cm[:, idx].sum() - cm[idx, idx])
        fn = float(cm[idx, :].sum() - cm[idx, idx])
        support = int(supports[idx])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_value = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        report[str(label)] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1_value,
            "support": support,
        }
        f1_values.append(f1_value)
        weighted_f1_sum += f1_value * support

    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    total_support = int(supports.sum())
    macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0
    weighted_f1_value = weighted_f1_sum / total_support if total_support else 0.0
    report["accuracy"] = accuracy
    report["macro avg"] = {
        "precision": float(np.mean([report[str(label)]["precision"] for label in labels])),
        "recall": float(np.mean([report[str(label)]["recall"] for label in labels])),
        "f1-score": macro_f1,
        "support": total_support,
    }
    report["weighted avg"] = {
        "precision": _weighted_average(report, labels, supports, "precision"),
        "recall": _weighted_average(report, labels, supports, "recall"),
        "f1-score": weighted_f1_value,
        "support": total_support,
    }
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": float(weighted_f1_value),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }


def _confusion_matrix_numpy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
) -> np.ndarray:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label in label_to_idx and pred_label in label_to_idx:
            cm[label_to_idx[true_label], label_to_idx[pred_label]] += 1
    return cm


def _weighted_average(
    report: dict[str, Any],
    labels: list[int],
    supports: np.ndarray,
    metric: str,
) -> float:
    total_support = int(supports.sum())
    if total_support == 0:
        return 0.0
    weighted_sum = sum(
        report[str(label)][metric] * int(support)
        for label, support in zip(labels, supports)
    )
    return float(weighted_sum / total_support)


def _print_report(title: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print(title)
    labels = [-1, 0, 1]
    if classification_report is not None:
        print(
            classification_report(
                y_true,
                y_pred,
                labels=labels,
                zero_division=0,
            )
        )
        return

    metrics = _build_metrics_numpy(y_true, y_pred, labels)
    for label in labels:
        row = metrics["classification_report"][str(label)]
        print(
            f"{label:>3} precision={row['precision']:.4f} "
            f"recall={row['recall']:.4f} f1={row['f1-score']:.4f} "
            f"support={row['support']}"
        )
    print(
        f"accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} "
        f"weighted_f1={metrics['weighted_f1']:.4f}"
    )


def main() -> None:
    """Run ESN training, prediction, and metric export."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/spy_1d.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ticker = config["ticker"]
    interval = config["interval"]
    esn_config = _load_esn_config(config)

    input_path = get_all_indicator_signals_path(ticker, interval)
    if not input_path.exists():
        raise FileNotFoundError(
            f"All-indicator signal data not found: {input_path}. "
            "Run build_esn_dataset first."
        )

    df = load_dataframe(input_path)
    _validate_input_columns(df)

    df = df.copy()
    df["target"] = df["turning_label"].shift(-1)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)

    train_ratio = float(esn_config["train_ratio"])
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}.")

    X = df[ESN_INPUT_SIGNAL_COLUMNS].fillna(0).astype(float).to_numpy()
    y = df["target"].to_numpy(dtype=int)

    split_idx = int(len(df) * train_ratio)
    washout = int(esn_config["washout"])
    if split_idx <= washout:
        raise ValueError(
            f"Train split ({split_idx} rows) must be larger than washout ({washout})."
        )
    if split_idx >= len(df):
        raise ValueError("Train split leaves no test rows.")

    X_train = X[:split_idx]
    y_train = y[:split_idx]

    model = ESNClassifier(
        input_dim=len(ESN_INPUT_SIGNAL_COLUMNS),
        reservoir_size=int(esn_config["reservoir_size"]),
        spectral_radius=float(esn_config["spectral_radius"]),
        sparsity=float(esn_config["sparsity"]),
        leaking_rate=float(esn_config["leaking_rate"]),
        ridge_alpha=float(esn_config["ridge_alpha"]),
        random_state=int(esn_config["random_state"]),
        washout=washout,
        class_weight=esn_config.get("class_weight", "balanced"),
    )
    model.fit(X_train, y_train)

    all_pred = model.predict(X)
    train_pred = all_pred[:split_idx]
    test_pred = all_pred[split_idx:]
    y_test = y[split_idx:]

    result_df = df.copy()
    result_df["esn_pred"] = all_pred.astype(int)
    result_df["split"] = "test"
    result_df.loc[: split_idx - 1, "split"] = "train"

    output_columns = [col for col in OUTPUT_COLUMNS if col in result_df.columns]
    predictions_path = get_esn_predictions_path(ticker, interval)
    save_dataframe(result_df[output_columns], predictions_path)

    train_eval_start = washout
    train_metrics = _build_metrics(y_train[train_eval_start:], train_pred[train_eval_start:])
    test_metrics = _build_metrics(y_test, test_pred)
    metrics = {
        "ticker": ticker,
        "interval": interval,
        "input_path": str(input_path),
        "target": "turning_label.shift(-1)",
        "input_signal_columns": ESN_INPUT_SIGNAL_COLUMNS,
        "train_ratio": train_ratio,
        "split_index": split_idx,
        "train_rows": split_idx,
        "test_rows": len(df) - split_idx,
        "esn_config": esn_config,
        **test_metrics,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }

    metrics_path = get_esn_metrics_path(ticker, interval)
    save_json(metrics, metrics_path)

    _print_report("Train classification report (after washout):", y_train[train_eval_start:], train_pred[train_eval_start:])
    _print_report("Test classification report:", y_test, test_pred)
    print(f"ESN predictions: {predictions_path}")
    print(f"ESN metrics: {metrics_path}")
    print(json.dumps({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
