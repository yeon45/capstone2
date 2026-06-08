"""Build the ESN input dataset from optimized indicator signal columns."""

from __future__ import annotations

import argparse

from src.utils.config import load_config
from src.utils.io import load_dataframe, save_dataframe
from src.utils.paths import (
    get_all_indicator_signals_path,
    get_all_signals_data_path,
    get_candle_patterns_data_path,
    get_ma_signal_data_path,
    get_roc_signal_data_path,
    get_rsi_signal_data_path,
    get_stochastic_signal_data_path,
)


ESN_INPUT_SIGNAL_COLUMNS = [
    "ma_signal",
    "rsi_signal",
    "roc_signal",
    "stoch_signal",
    "candle_hammer_hanging_man_signal",
    "candle_dark_cloud_cover_signal",
    "candle_piercing_line_signal",
    "candle_bullish_engulfing_signal",
    "candle_bearish_engulfing_signal",
]


def _merge_indicator_columns(df, indicator_df, columns: list[str]):
    """Merge indicator columns by Date when available, otherwise by row position."""

    missing_cols = [col for col in columns if col not in indicator_df.columns]
    if missing_cols:
        raise ValueError(f"Indicator signal data missing required columns: {missing_cols}")

    if "Date" in df.columns and "Date" in indicator_df.columns:
        return df.merge(indicator_df[["Date", *columns]], on="Date", how="left")
    if len(df) == len(indicator_df):
        result = df.copy()
        for col in columns:
            result[col] = indicator_df[col].to_numpy()
        return result
    raise ValueError("Cannot merge indicator signals: missing Date columns and row counts differ.")


def _add_signed_signal(
    df,
    signal_col: str,
    buy_col: str,
    sell_col: str,
) -> None:
    """Add one signed ESN signal column with buy=-1, sell=1, hold=0."""

    missing = [col for col in [buy_col, sell_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Cannot build {signal_col}; missing columns: {missing}")
    df[signal_col] = df[sell_col].fillna(0).astype(int) - df[buy_col].fillna(0).astype(int)


def _normalize_cpm_label_convention(df) -> None:
    """Normalize turning_label to bottom/buy=-1, top/sell=1 when point types exist."""

    if "turning_label" not in df.columns or "cpm_point_type" not in df.columns:
        return
    point_type = df["cpm_point_type"].fillna("").astype(str).str.lower()
    df.loc[point_type == "bottom", "turning_label"] = -1
    df.loc[point_type == "top", "turning_label"] = 1


def main() -> None:
    """Combine indicator outputs and materialize the 9 ESN input signals."""

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
    _normalize_cpm_label_convention(df)
    _add_signed_signal(df, "ma_signal", "ma_buy_signal", "ma_sell_signal")
    included_indicators = ["MA"]

    rsi_path = get_rsi_signal_data_path(ticker, interval)
    rsi_cols = ["rsi", "rsi_buy_signal", "rsi_sell_signal"]
    if rsi_path.exists():
        rsi_df = load_dataframe(rsi_path)
        df = _merge_indicator_columns(df, rsi_df, rsi_cols)
        _add_signed_signal(df, "rsi_signal", "rsi_buy_signal", "rsi_sell_signal")
        included_indicators.append("RSI")

    roc_path = get_roc_signal_data_path(ticker, interval)
    roc_cols = ["roc", "roc_buy_signal", "roc_sell_signal"]
    if roc_path.exists():
        roc_df = load_dataframe(roc_path)
        df = _merge_indicator_columns(df, roc_df, roc_cols)
        _add_signed_signal(df, "roc_signal", "roc_buy_signal", "roc_sell_signal")
        included_indicators.append("ROC")

    stochastic_path = get_stochastic_signal_data_path(ticker, interval)
    stochastic_cols = ["stoch_k", "stoch_d", "stoch_buy_signal", "stoch_sell_signal"]
    if stochastic_path.exists():
        stochastic_df = load_dataframe(stochastic_path)
        df = _merge_indicator_columns(df, stochastic_df, stochastic_cols)
        _add_signed_signal(df, "stoch_signal", "stoch_buy_signal", "stoch_sell_signal")
        included_indicators.append("Stochastic")

    candle_path = get_candle_patterns_data_path(ticker, interval)
    candle_cols = [
        "candle_hammer_hanging_man_signal",
        "candle_dark_cloud_cover_signal",
        "candle_piercing_line_signal",
        "candle_bullish_engulfing_signal",
        "candle_bearish_engulfing_signal",
    ]
    if candle_path.exists():
        candle_df = load_dataframe(candle_path)
        df = _merge_indicator_columns(df, candle_df, candle_cols)
        included_indicators.append("Candle")

    missing_esn_cols = [col for col in ESN_INPUT_SIGNAL_COLUMNS if col not in df.columns]
    if missing_esn_cols:
        raise ValueError(
            "Cannot build ESN dataset; missing required input signal columns: "
            f"{missing_esn_cols}"
        )

    processed_path = get_all_signals_data_path(ticker, interval)
    output_path = get_all_indicator_signals_path(ticker, interval)
    save_dataframe(df, processed_path)
    save_dataframe(df, output_path)

    print(f"Included indicator signals: {', '.join(included_indicators)}")
    print(f"ESN input signal columns ({len(ESN_INPUT_SIGNAL_COLUMNS)}):")
    for col in ESN_INPUT_SIGNAL_COLUMNS:
        print(f"- {col}")
    print(f"processed all-signals data: {processed_path}")
    print(f"output all-indicator signals: {output_path}")


if __name__ == "__main__":
    main()
