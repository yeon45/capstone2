"""Project path helpers."""

from __future__ import annotations

from pathlib import Path


def get_raw_data_path(ticker: str, interval: str) -> Path:
    """Return raw OHLCV CSV path."""

    return Path("data") / "raw" / ticker / interval / f"{ticker}.csv"


def get_processed_dir(ticker: str, interval: str) -> Path:
    """Return processed data directory."""

    return Path("data") / "processed" / ticker / interval


def get_clean_data_path(ticker: str, interval: str) -> Path:
    """Return clean OHLCV CSV path."""

    return get_processed_dir(ticker, interval) / f"{ticker}_clean.csv"


def get_tp_data_path(ticker: str, interval: str) -> Path:
    """Return CPM-labeled data CSV path."""

    return get_processed_dir(ticker, interval) / f"{ticker}_with_tp.csv"


def get_ma_signal_data_path(ticker: str, interval: str) -> Path:
    """Return processed MA signal CSV path."""

    return get_processed_dir(ticker, interval) / f"{ticker}_with_ma_signals.csv"


def get_rsi_signal_data_path(ticker: str, interval: str) -> Path:
    """Return processed RSI signal CSV path."""

    return get_processed_dir(ticker, interval) / f"{ticker}_with_rsi_signals.csv"


def get_roc_signal_data_path(ticker: str, interval: str) -> Path:
    """Return processed ROC signal CSV path."""

    return get_processed_dir(ticker, interval) / f"{ticker}_with_roc_signals.csv"


def get_stochastic_signal_data_path(ticker: str, interval: str) -> Path:
    """Return processed Stochastic signal CSV path."""

    return get_processed_dir(ticker, interval) / f"{ticker}_with_stochastic_signals.csv"


def get_candle_patterns_data_path(ticker: str, interval: str) -> Path:
    """Return processed candle pattern CSV path."""

    return get_processed_dir(ticker, interval) / f"{ticker}_with_candle_patterns.csv"


def get_all_signals_data_path(ticker: str, interval: str) -> Path:
    """Return processed all-signals CSV path."""

    return get_processed_dir(ticker, interval) / f"{ticker}_all_signals.csv"


def get_cpm_output_dir(ticker: str, interval: str) -> Path:
    """Return CPM output directory."""

    return Path("outputs") / "cpm" / ticker / interval


def get_cpm_search_results_path(ticker: str, interval: str) -> Path:
    """Return CPM grid search output path."""

    return get_cpm_output_dir(ticker, interval) / "cpm_search_results.csv"


def get_best_cpm_params_path(ticker: str, interval: str) -> Path:
    """Return best CPM params JSON path."""

    return get_cpm_output_dir(ticker, interval) / "best_cpm_params.json"


def get_indicator_params_dir(ticker: str, interval: str) -> Path:
    """Return indicator parameter output directory."""

    return Path("outputs") / "indicator_params" / ticker / interval


def get_indicator_param_path(ticker: str, interval: str, indicator: str) -> Path:
    """Return indicator parameter JSON path."""

    return get_indicator_params_dir(ticker, interval) / f"{indicator}_params.json"


def get_signals_output_dir(ticker: str, interval: str) -> Path:
    """Return signal output directory."""

    return Path("outputs") / "signals" / ticker / interval


def get_ma_signal_output_path(ticker: str, interval: str) -> Path:
    """Return output MA signal CSV path."""

    return get_signals_output_dir(ticker, interval) / f"{ticker}_with_ma_signals.csv"


def get_rsi_signal_output_path(ticker: str, interval: str) -> Path:
    """Return output RSI signal CSV path."""

    return get_signals_output_dir(ticker, interval) / f"{ticker}_with_rsi_signals.csv"


def get_roc_signal_output_path(ticker: str, interval: str) -> Path:
    """Return output ROC signal CSV path."""

    return get_signals_output_dir(ticker, interval) / f"{ticker}_with_roc_signals.csv"


def get_stochastic_signal_output_path(ticker: str, interval: str) -> Path:
    """Return output Stochastic signal CSV path."""

    return get_signals_output_dir(ticker, interval) / f"{ticker}_with_stochastic_signals.csv"


def get_candle_patterns_output_path(ticker: str, interval: str) -> Path:
    """Return output candle pattern CSV path."""

    return get_signals_output_dir(ticker, interval) / f"{ticker}_with_candle_patterns.csv"


def get_all_indicator_signals_path(ticker: str, interval: str) -> Path:
    """Return output all-indicator signal CSV path."""

    return get_signals_output_dir(ticker, interval) / f"{ticker}_all_indicator_signals.csv"


def get_predictions_output_dir(ticker: str, interval: str) -> Path:
    """Return legacy prediction output directory."""

    return Path("outputs") / "predictions" / ticker / interval


def get_esn_output_dir(ticker: str, interval: str) -> Path:
    """Return ESN output directory."""

    return Path("outputs") / "esn" / ticker / interval


def get_esn_predictions_path(ticker: str, interval: str) -> Path:
    """Return ESN prediction CSV path."""

    return get_esn_output_dir(ticker, interval) / "esn_predictions.csv"


def get_esn_metrics_path(ticker: str, interval: str) -> Path:
    """Return ESN metrics JSON path."""

    return get_esn_output_dir(ticker, interval) / "esn_metrics.json"


def get_esn_config_used_path(ticker: str, interval: str) -> Path:
    """Return ESN used-config JSON path."""

    return get_esn_output_dir(ticker, interval) / "esn_config_used.json"


def get_figures_dir(ticker: str, interval: str) -> Path:
    """Return figure output directory."""

    return Path("outputs") / "figures" / ticker / interval


def get_cpm_figure_path(ticker: str, interval: str) -> Path:
    """Return CPM turning point figure path."""

    return get_figures_dir(ticker, interval) / "cpm_turning_points.png"


def get_ma_figure_path(ticker: str, interval: str) -> Path:
    """Return MA signal figure path."""

    return get_figures_dir(ticker, interval) / "ma_signals.png"


def get_rsi_figure_path(ticker: str, interval: str) -> Path:
    """Return RSI signal figure path."""

    return get_figures_dir(ticker, interval) / "rsi_signals.png"


def get_roc_figure_path(ticker: str, interval: str) -> Path:
    """Return ROC signal figure path."""

    return get_figures_dir(ticker, interval) / "roc_signals.png"


def get_stochastic_figure_path(ticker: str, interval: str) -> Path:
    """Return Stochastic signal figure path."""

    return get_figures_dir(ticker, interval) / "stochastic_signals.png"


def get_candle_patterns_figure_path(ticker: str, interval: str) -> Path:
    """Return candle pattern signal figure path."""

    return get_figures_dir(ticker, interval) / "candle_patterns.png"


def get_candle_pattern_figures_dir(ticker: str, interval: str) -> Path:
    """Return directory for individual candle pattern figures."""

    return get_figures_dir(ticker, interval) / "candle_patterns"


def get_candle_pattern_counts_figure_path(ticker: str, interval: str) -> Path:
    """Return candle pattern counts figure path."""

    return get_figures_dir(ticker, interval) / "candle_pattern_counts.png"


def ensure_parent_dir(path: Path) -> None:
    """Create the parent directory for a path if needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
