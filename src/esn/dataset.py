"""Dataset loading and chronological split helpers for ESN training."""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

with contextlib.redirect_stderr(io.StringIO()):
    import pandas as pd

try:
    with contextlib.redirect_stderr(io.StringIO()):
        from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - depends on local binary package state
    StandardScaler = None

from src.pipeline.build_esn_dataset import ESN_INPUT_SIGNAL_COLUMNS
from src.utils.config import load_config
from src.utils.io import load_dataframe
from src.utils.paths import get_all_indicator_signals_path, get_all_signals_data_path


@dataclass(frozen=True)
class ESNSplit:
    X: np.ndarray
    y: np.ndarray
    dates: np.ndarray | None


@dataclass(frozen=True)
class ESNDataset:
    train: ESNSplit
    val: ESNSplit
    test: ESNSplit
    feature_cols: list[str]
    target_col: str
    target_shift: int
    input_path: Path
    scaler: Any
    split_indices: dict[str, int]
    dates: np.ndarray | None
    target_values: np.ndarray


def load_esn_dataset(config_path: str | Path) -> ESNDataset:
    """Load final signal data and return scaled chronological train/val/test arrays."""

    config = load_config(config_path)
    ticker = str(config["ticker"])
    interval = str(config["interval"])
    esn_config = config.get("esn", {})
    feature_cols = list(esn_config.get("feature_cols", ESN_INPUT_SIGNAL_COLUMNS))
    target_col = str(esn_config.get("target_col", "triangle_target"))
    target_shift = int(esn_config.get("target_shift", 0))
    train_ratio = float(esn_config.get("train_ratio", 0.7))
    val_ratio = float(esn_config.get("val_ratio", 0.15))
    test_ratio = float(esn_config.get("test_ratio", 0.15))

    input_path = _resolve_input_path(ticker, interval)
    df = load_dataframe(input_path)
    if target_col == "triangle_target" and target_col not in df.columns:
        df[target_col] = build_triangle_target(df, _turning_points_from_df(df))
    X, y, dates = build_esn_arrays(df, feature_cols, target_col, target_shift)
    return split_esn_arrays(
        X=X,
        y=y,
        dates=dates,
        feature_cols=feature_cols,
        target_col=target_col,
        target_shift=target_shift,
        input_path=input_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )


def build_esn_arrays(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    target_shift: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Build X/y arrays from a signal DataFrame with shifted target alignment."""

    required_cols = [*feature_cols, target_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"ESN input data missing required columns: {missing}")

    working_df = df.copy()
    working_df["_esn_target"] = working_df[target_col].shift(target_shift)
    subset_cols = [*feature_cols, "_esn_target"]
    date_col = _date_column(working_df)
    if date_col:
        subset_cols.append(date_col)
    working_df = working_df.dropna(subset=subset_cols).reset_index(drop=True)

    X = working_df[feature_cols].astype(float).to_numpy()
    y = working_df["_esn_target"].astype(float).to_numpy()
    dates = working_df[date_col].to_numpy() if date_col else None
    return X, y, dates


def build_triangle_target(
    df: pd.DataFrame,
    turning_points: list[dict[str, Any]] | pd.DataFrame,
) -> pd.Series:
    """Build a CPM turning-point triangle-wave teacher target in [-1, 1].

    Bottom/buy points are mapped to -1, top/sell points are mapped to 1, and
    values between adjacent turning points are linearly interpolated.
    This target is an offline supervised-learning label derived from CPM; it is
    not directly observable at a live trading decision point.
    """

    points = _normalize_turning_points(df, turning_points)
    if len(points) < 2:
        raise ValueError(
            "At least two CPM turning points are required to build triangle_target."
        )

    target = pd.Series(index=df.index, dtype=float, name="triangle_target")
    first_pos, first_value = points[0]
    last_pos, last_value = points[-1]
    target.iloc[: first_pos + 1] = first_value
    target.iloc[last_pos:] = last_value

    for (start_pos, start_value), (end_pos, end_value) in zip(points, points[1:]):
        if end_pos <= start_pos:
            continue
        target.iloc[start_pos : end_pos + 1] = np.linspace(
            start_value,
            end_value,
            end_pos - start_pos + 1,
        )
    return target.clip(-1.0, 1.0)


def make_esn_dataset(
    df: pd.DataFrame,
    signal_cols: list[str],
    target_col: str | None = None,
    target: pd.Series | None = None,
    dropna: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return indicator-signal X and triangle-wave y with exactly aligned indices."""

    if target is None and target_col is None:
        raise ValueError("Either target_col or target Series must be provided.")
    missing = [col for col in signal_cols if col not in df.columns]
    if missing:
        raise ValueError(f"ESN signal columns missing from input data: {missing}")

    X = df[signal_cols].copy()
    if target is None:
        if target_col not in df.columns:
            raise ValueError(f"ESN target column missing from input data: {target_col}")
        y = df[target_col].copy()
    else:
        y = target.copy()

    if not X.index.equals(y.index):
        y = y.reindex(X.index)
    combined = X.copy()
    combined["_triangle_target"] = y
    if dropna:
        combined = combined.dropna(subset=[*signal_cols, "_triangle_target"])
    elif combined[[*signal_cols, "_triangle_target"]].isna().any().any():
        raise ValueError("ESN dataset contains NaN values and dropna=False.")

    return combined[signal_cols], combined["_triangle_target"].rename("triangle_target")


def split_esn_arrays(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray | None,
    feature_cols: list[str],
    target_col: str,
    target_shift: int,
    input_path: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> ESNDataset:
    """Chronologically split arrays and fit the feature scaler on train only."""

    _validate_ratios(train_ratio, val_ratio, test_ratio)
    n_rows = len(X)
    train_end = int(n_rows * train_ratio)
    val_end = train_end + int(n_rows * val_ratio)
    if train_end <= 0 or val_end <= train_end or val_end >= n_rows:
        raise ValueError(
            "Invalid chronological split; each of train/val/test must contain rows. "
            f"Got n_rows={n_rows}, train_end={train_end}, val_end={val_end}."
        )

    scaler = StandardScaler() if StandardScaler is not None else _NumpyStandardScaler()
    X_train = scaler.fit_transform(X[:train_end])
    X_val = scaler.transform(X[train_end:val_end])
    X_test = scaler.transform(X[val_end:])

    return ESNDataset(
        train=ESNSplit(X_train, y[:train_end], _slice_dates(dates, 0, train_end)),
        val=ESNSplit(X_val, y[train_end:val_end], _slice_dates(dates, train_end, val_end)),
        test=ESNSplit(X_test, y[val_end:], _slice_dates(dates, val_end, n_rows)),
        feature_cols=feature_cols,
        target_col=target_col,
        target_shift=target_shift,
        input_path=input_path,
        scaler=scaler,
        split_indices={"train_end": train_end, "val_end": val_end, "n_rows": n_rows},
        dates=dates,
        target_values=y,
    )


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
    raise ValueError(
        "Cannot build triangle_target; input data needs cpm_point_type or turning_label."
    )


def _normalize_turning_points(
    df: pd.DataFrame,
    turning_points: list[dict[str, Any]] | pd.DataFrame,
) -> list[tuple[int, float]]:
    records = (
        turning_points.to_dict("records")
        if isinstance(turning_points, pd.DataFrame)
        else list(turning_points)
    )
    normalized: list[tuple[int, float]] = []
    for point in records:
        pos = _point_position(df, point)
        value = _point_value(point)
        if 0 <= pos < len(df):
            normalized.append((pos, value))
    normalized = sorted(set(normalized), key=lambda item: item[0])
    if len(normalized) < 2:
        raise ValueError(
            "Fewer than two valid turning points fall inside the ESN DataFrame index."
        )
    return normalized


def _point_position(df: pd.DataFrame, point: dict[str, Any]) -> int:
    if "index" in point:
        index_value = point["index"]
        if index_value in df.index:
            return int(df.index.get_loc(index_value))
        return int(index_value)
    for key in ["date", "Date", "datetime", "Datetime"]:
        if key in point:
            date_col = _date_column(df)
            if date_col is None:
                raise ValueError(f"Turning point has {key}, but df has no date column.")
            matches = np.flatnonzero(df[date_col].astype(str).to_numpy() == str(point[key]))
            if len(matches) == 0:
                raise ValueError(f"Turning point date not found in df[{date_col}]: {point[key]}")
            return int(matches[0])
    raise ValueError(f"Turning point missing index/date information: {point}")


def _point_value(point: dict[str, Any]) -> float:
    raw_type = str(point.get("type", point.get("point_type", point.get("signal", "")))).lower()
    if raw_type in {"bottom", "buy"}:
        return -1.0
    if raw_type in {"top", "sell"}:
        return 1.0
    raise ValueError(
        "Turning point type must be bottom/top or buy/sell; "
        f"got {point.get('type', point)}."
    )


def _resolve_input_path(ticker: str, interval: str) -> Path:
    primary_path = get_all_signals_data_path(ticker, interval)
    if primary_path.exists():
        return primary_path
    fallback_path = get_all_indicator_signals_path(ticker, interval)
    if fallback_path.exists():
        return fallback_path
    raise FileNotFoundError(
        "ESN input file not found. Checked: "
        f"{primary_path} and {fallback_path}. Run build_esn_dataset first."
    )


def _date_column(df: pd.DataFrame) -> str | None:
    for col in ["Date", "date", "Datetime", "datetime"]:
        if col in df.columns:
            return col
    return None


def _slice_dates(dates: np.ndarray | None, start: int, end: int) -> np.ndarray | None:
    if dates is None:
        return None
    return dates[start:end]


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }
    invalid = {name: value for name, value in ratios.items() if value <= 0.0}
    if invalid:
        raise ValueError(f"Split ratios must be positive: {invalid}")
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}.")


class _NumpyStandardScaler:
    """Minimal StandardScaler fallback when sklearn cannot be imported."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0
        return self.transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler must be fitted before transform().")
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_
