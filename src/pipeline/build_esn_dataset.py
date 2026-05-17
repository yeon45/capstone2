"""Placeholder pipeline for building the future ESN dataset."""

from __future__ import annotations

import argparse

from src.utils.config import load_config
from src.utils.io import load_dataframe, save_dataframe
from src.utils.paths import (
    get_all_indicator_signals_path,
    get_all_signals_data_path,
    get_ma_signal_data_path,
)


def main() -> None:
    """Copy MA signals into the all-signals dataset placeholder."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/spy_1d.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ticker = config["ticker"]
    interval = config["interval"]
    ma_path = get_ma_signal_data_path(ticker, interval)
    if not ma_path.exists():
        raise FileNotFoundError(f"MA signal data not found: {ma_path}. Run run_ma_ga first.")

    df = load_dataframe(ma_path)
    processed_path = get_all_signals_data_path(ticker, interval)
    output_path = get_all_indicator_signals_path(ticker, interval)
    save_dataframe(df, processed_path)
    save_dataframe(df, output_path)

    print("Only MA signals are currently included. ESN training is not implemented yet.")
    print(f"processed all-signals data: {processed_path}")
    print(f"output all-indicator signals: {output_path}")


if __name__ == "__main__":
    main()
