"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and validate required top-level keys."""

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")

    required_keys = ["ticker", "interval", "start", "cpm", "ma_ga"]
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    return config
