"""Run rule-based candle pattern detection against CPM-labeled data."""

from __future__ import annotations

import argparse
import json

from src.indicators.candle import CANDLE_PATTERN_COLUMNS, add_candle_patterns
from src.utils.config import load_config
from src.utils.io import load_dataframe, save_dataframe
from src.utils.paths import (
    get_candle_pattern_counts_figure_path,
    get_candle_patterns_data_path,
    get_candle_patterns_figure_path,
    get_candle_patterns_output_path,
    get_tp_data_path,
)
from src.visualization.plot_signals import (
    plot_candle_pattern_counts,
    plot_candle_patterns,
)


def main() -> None:
    """Generate candle pattern columns and save signal outputs."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/spy_1d.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ticker = config["ticker"]
    interval = config["interval"]
    tp_path = get_tp_data_path(ticker, interval)
    if not tp_path.exists():
        raise FileNotFoundError(f"TP data not found: {tp_path}. Run run_cpm first.")

    df = load_dataframe(tp_path)
    missing = [col for col in ["Open", "High", "Low", "Close", "turning_label"] if col not in df.columns]
    if missing:
        raise ValueError(f"TP data missing required columns: {missing}")

    candle_config = config.get("candle", {})
    signal_df = add_candle_patterns(
        df,
        open_col="Open",
        high_col="High",
        low_col="Low",
        close_col="Close",
        hammer_ratio=float(candle_config.get("hammer_ratio", 2.0)),
        upper_shadow_ratio=float(candle_config.get("upper_shadow_ratio", 1.0)),
        use_gap_condition=bool(candle_config.get("use_gap_condition", False)),
    )

    processed_signal_path = get_candle_patterns_data_path(ticker, interval)
    output_signal_path = get_candle_patterns_output_path(ticker, interval)
    figure_path = get_candle_patterns_figure_path(ticker, interval)
    counts_figure_path = get_candle_pattern_counts_figure_path(ticker, interval)

    save_dataframe(signal_df, processed_signal_path)
    save_dataframe(signal_df, output_signal_path)
    plot_candle_patterns(signal_df, figure_path)
    plot_candle_pattern_counts(signal_df, counts_figure_path, CANDLE_PATTERN_COLUMNS)

    pattern_counts = {col: int(signal_df[col].sum()) for col in CANDLE_PATTERN_COLUMNS}
    print("candle pattern counts:")
    print(json.dumps(pattern_counts, indent=2))
    print(f"Candle buy signals: {int(signal_df['candle_buy_signal'].sum())}")
    print(f"Candle sell signals: {int(signal_df['candle_sell_signal'].sum())}")
    print(f"processed candle patterns: {processed_signal_path}")
    print(f"output candle patterns: {output_signal_path}")
    print(f"candle figure: {figure_path}")
    print(f"candle count figure: {counts_figure_path}")


if __name__ == "__main__":
    main()
