"""Small IO helpers for JSON and CSV files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.paths import ensure_parent_dir


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary to JSON using UTF-8 and indent=2."""

    output_path = Path(path)
    ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file as a dictionary."""

    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"JSON file not found: {input_path}")
    return json.loads(input_path.read_text(encoding="utf-8"))


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame to CSV, creating parent directories."""

    output_path = Path(path)
    ensure_parent_dir(output_path)
    df.to_csv(output_path, index=False)


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a CSV file as a DataFrame with a clear missing-file error."""

    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"CSV file not found: {input_path}")
    return pd.read_csv(input_path)
