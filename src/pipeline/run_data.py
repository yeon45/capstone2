"""Download and preprocess OHLCV data."""

from __future__ import annotations

import argparse

from src.data.download import download_ohlcv
from src.data.preprocess import preprocess_ohlcv
from src.utils.config import load_config
from src.utils.io import save_dataframe
from src.utils.paths import get_clean_data_path, get_raw_data_path


def main() -> None:
    """Run the data download and preprocessing pipeline."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/spy_1d.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ticker = config["ticker"]
    interval = config["interval"]
    start = config["start"]
    end = config.get("end")

    raw_path = get_raw_data_path(ticker, interval)
    clean_path = get_clean_data_path(ticker, interval)
    raw_df = download_ohlcv(ticker, interval=interval, start=start, end=end, output_path=raw_path)
    clean_df = preprocess_ohlcv(raw_df)
    save_dataframe(clean_df, clean_path)

    print(f"ticker: {ticker}")
    print(f"interval: {interval}")
    print(f"date range: {clean_df['Date'].iloc[0]} to {clean_df['Date'].iloc[-1]}")
    print(f"rows: {len(clean_df)}")
    print(f"raw output: {raw_path}")
    print(f"clean output: {clean_path}")


if __name__ == "__main__":
    main()
