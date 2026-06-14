"""Run an ESN hyperparameter sweep from the project YAML config."""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

with contextlib.redirect_stderr(io.StringIO()):
    import pandas as pd

from src.esn.dataset import ESNDataset, load_esn_dataset
from src.esn.metrics import classification_like_metrics, regression_metrics, score_to_signal
from src.esn.model import EchoStateNetwork
from src.pipeline.run_esn import _load_esn_config, _threshold_values
from src.utils.config import load_config
from src.utils.io import save_dataframe, save_json


SWEEP_PARAMETER_KEYS = [
    "reservoir_size",
    "spectral_radius",
    "sparsity",
    "input_scale",
    "leaking_rate",
    "ridge_alpha",
    "washout",
    "random_state",
]

DEFAULT_ESN_SWEEP_CONFIG: dict[str, Any] = {
    "enabled": False,
    "selection_metric": "val_directional_accuracy",
    "selection_mode": "max",
    "output_dir": "results/esn_sweep",
    "save_predictions": False,
    "save_best_config": True,
}


def main() -> None:
    """Run the ESN sweep pipeline."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/spy_1d.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    sweep_config = _load_sweep_config(config)
    if not sweep_config["enabled"]:
        raise ValueError("esn_sweep.enabled must be true to run the ESN sweep.")

    ticker = str(config["ticker"])
    interval = str(config["interval"])
    output_dir = Path(str(sweep_config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    if sweep_config["save_predictions"]:
        predictions_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_esn_dataset(args.config)
    base_esn_config = _load_esn_config(config)
    parameter_sets = _parameter_grid(sweep_config)

    print(f"ESN input file: {dataset.input_path}")
    print(f"Total ESN sweep runs: {len(parameter_sets)}")
    print(f"Sweep output dir: {output_dir}")

    results: list[dict[str, Any]] = []
    best_payload: dict[str, Any] | None = None
    for run_index, params in enumerate(parameter_sets, start=1):
        esn_config = copy.deepcopy(base_esn_config)
        esn_config.update(params)
        if "threshold_values" in sweep_config:
            esn_config["threshold_values"] = sweep_config["threshold_values"]

        print(f"[{run_index}/{len(parameter_sets)}] {params}")
        payload = _run_one(
            run_id=f"run_{run_index:04d}",
            ticker=ticker,
            interval=interval,
            config_path=args.config,
            dataset=dataset,
            esn_config=esn_config,
            save_predictions=bool(sweep_config["save_predictions"]),
            predictions_dir=predictions_dir,
        )
        results.append(payload["summary"])
        if best_payload is None or _is_better(
            payload["summary"],
            best_payload["summary"],
            sweep_config["selection_metric"],
            sweep_config["selection_mode"],
        ):
            best_payload = payload

    if best_payload is None:
        raise ValueError("ESN sweep produced no results.")

    results_df = pd.DataFrame(results)
    results_csv_path = output_dir / "sweep_results.csv"
    results_json_path = output_dir / "sweep_results.json"
    best_metrics_path = output_dir / "best_metrics.json"
    save_dataframe(results_df, results_csv_path)
    save_json(results, results_json_path)
    save_json(best_payload["metrics"], best_metrics_path)

    if sweep_config["save_best_config"]:
        best_config = _build_best_config(config, best_payload["metrics"]["config_used"]["esn"])
        _save_yaml(best_config, output_dir / "best_esn_config.yaml")
        save_json(best_config, output_dir / "best_esn_config.json")

    best_summary = best_payload["summary"]
    print("Best ESN sweep result:")
    print(json.dumps(best_summary, indent=2))
    print(f"Sweep CSV: {results_csv_path}")
    print(f"Sweep JSON: {results_json_path}")
    print(f"Best metrics: {best_metrics_path}")


def _load_sweep_config(config: dict[str, Any]) -> dict[str, Any]:
    sweep_config = copy.deepcopy(DEFAULT_ESN_SWEEP_CONFIG)
    sweep_config.update(config.get("esn_sweep", {}))
    return sweep_config


def _parameter_grid(sweep_config: dict[str, Any]) -> list[dict[str, Any]]:
    values_by_key: list[tuple[str, list[Any]]] = []
    for key in SWEEP_PARAMETER_KEYS:
        if key not in sweep_config:
            continue
        values = sweep_config[key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"esn_sweep.{key} must be a non-empty list.")
        values_by_key.append((key, values))

    if not values_by_key:
        raise ValueError("esn_sweep must define at least one hyperparameter list.")

    parameter_sets = []
    keys = [key for key, _ in values_by_key]
    for combination in itertools.product(*(values for _, values in values_by_key)):
        parameter_sets.append(dict(zip(keys, combination)))
    return parameter_sets


def _run_one(
    run_id: str,
    ticker: str,
    interval: str,
    config_path: str,
    dataset: ESNDataset,
    esn_config: dict[str, Any],
    save_predictions: bool,
    predictions_dir: Path,
) -> dict[str, Any]:
    washout = int(esn_config["washout"])
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
    selected = _select_thresholds(dataset.val.y, val_pred, _threshold_values(esn_config))

    train_signal = np.asarray(score_to_signal(train_pred, selected["threshold"]), dtype=int)
    val_signal = np.asarray(score_to_signal(val_pred, selected["threshold"]), dtype=int)
    test_signal = np.asarray(score_to_signal(test_pred, selected["threshold"]), dtype=int)

    metrics = _build_sweep_metrics(
        run_id=run_id,
        ticker=ticker,
        interval=interval,
        config_path=config_path,
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
    prediction_path = ""
    if save_predictions:
        prediction_df = pd.concat(
            [
                _prediction_frame(dataset.train, train_pred, train_signal, "train"),
                _prediction_frame(dataset.val, val_pred, val_signal, "val"),
                _prediction_frame(dataset.test, test_pred, test_signal, "test"),
            ],
            ignore_index=True,
        )
        prediction_path = str(predictions_dir / f"{run_id}_predictions.csv")
        save_dataframe(prediction_df, prediction_path)

    summary = _flatten_metrics(metrics)
    summary["prediction_path"] = prediction_path
    return {"metrics": metrics, "summary": summary}


def _select_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold_values: list[float],
) -> dict[str, float]:
    best: dict[str, float] | None = None
    y_true_signal = np.asarray(score_to_signal(y_true, 0.0), dtype=int)
    for threshold in threshold_values:
        y_signal = np.asarray(score_to_signal(y_pred, threshold), dtype=int)
        class_metrics = classification_like_metrics(y_true_signal, y_signal)
        reg_metrics = regression_metrics(y_true, y_pred)
        score = float(class_metrics["accuracy"])
        if best is None or score > best["validation_accuracy"]:
            best = {
                "threshold": float(threshold),
                "validation_accuracy": score,
                "validation_directional_accuracy": float(reg_metrics["directional_accuracy"]),
            }
    if best is None:
        raise ValueError("No threshold combinations were provided.")
    return best


def _build_sweep_metrics(
    run_id: str,
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
        "run_id": run_id,
        "ticker": ticker,
        "interval": interval,
        "input_path": str(dataset.input_path),
        "feature_cols": dataset.feature_cols,
        "target_col": dataset.target_col,
        "target_shift": dataset.target_shift,
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


def _prediction_frame(split: Any, y_pred: np.ndarray, y_signal: np.ndarray, split_name: str) -> pd.DataFrame:
    data: dict[str, Any] = {
        "y_true": split.y.astype(float),
        "trading_score": y_pred.astype(float),
        "signal": y_signal.astype(int),
        "split": split_name,
    }
    if split.dates is not None:
        data = {"date": split.dates, **data}
    return pd.DataFrame(data)


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    esn_config = metrics["config_used"]["esn"]
    row: dict[str, Any] = {
        "run_id": metrics["run_id"],
        "ticker": metrics["ticker"],
        "interval": metrics["interval"],
        "threshold": metrics["selected_thresholds"]["threshold"],
    }
    for key in SWEEP_PARAMETER_KEYS:
        row[key] = esn_config[key]
    for split_name, prefix in [("train", "train"), ("validation", "val"), ("test", "test")]:
        split_metrics = metrics[split_name]
        row[f"{prefix}_mse"] = split_metrics["regression"]["mse"]
        row[f"{prefix}_mae"] = split_metrics["regression"]["mae"]
        row[f"{prefix}_directional_accuracy"] = split_metrics["regression"][
            "directional_accuracy"
        ]
        row[f"{prefix}_accuracy"] = split_metrics["classification_like"]["accuracy"]
        row[f"{prefix}_buy_precision"] = split_metrics["classification_like"]["buy_precision"]
        row[f"{prefix}_buy_recall"] = split_metrics["classification_like"]["buy_recall"]
        row[f"{prefix}_sell_precision"] = split_metrics["classification_like"][
            "sell_precision"
        ]
        row[f"{prefix}_sell_recall"] = split_metrics["classification_like"]["sell_recall"]
    return row


def _is_better(
    candidate: dict[str, Any],
    incumbent: dict[str, Any],
    selection_metric: str,
    selection_mode: str,
) -> bool:
    if selection_metric not in candidate:
        raise ValueError(f"Unknown selection_metric: {selection_metric}")
    if selection_mode not in {"max", "min"}:
        raise ValueError(f"selection_mode must be 'max' or 'min', got {selection_mode}.")

    candidate_value = float(candidate[selection_metric])
    incumbent_value = float(incumbent[selection_metric])
    if selection_mode == "max":
        return candidate_value > incumbent_value
    return candidate_value < incumbent_value


def _build_best_config(config: dict[str, Any], best_esn_config: dict[str, Any]) -> dict[str, Any]:
    best_config = copy.deepcopy(config)
    best_config["esn"] = copy.deepcopy(best_esn_config)
    if "esn_sweep" in best_config:
        best_config["esn_sweep"] = copy.deepcopy(best_config["esn_sweep"])
        best_config["esn_sweep"]["enabled"] = False
    return best_config


def _save_yaml(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


if __name__ == "__main__":
    main()
