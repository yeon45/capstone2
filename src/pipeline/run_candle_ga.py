"""Run candle pattern GA optimization against fixed CPM labels."""

from __future__ import annotations

import argparse
import json

from src.ga.candle_optimizer import calculate_candle_fitness, run_candle_ga
from src.indicators.candle import CANDLE_SIGNAL_COLUMNS, generate_candle_signals
from src.utils.config import load_config
from src.utils.io import load_dataframe, load_json, save_dataframe, save_json
from src.utils.paths import (
    get_best_cpm_params_path,
    get_candle_pattern_figures_dir,
    get_candle_pattern_counts_figure_path,
    get_candle_patterns_data_path,
    get_candle_patterns_figure_path,
    get_candle_patterns_output_path,
    get_indicator_param_path,
    get_tp_data_path,
)
from src.visualization.plot_signals import (
    plot_individual_candle_patterns,
    plot_candle_pattern_counts,
    plot_candle_patterns,
)


def _param_dict(params: list[float]) -> dict:
    return {
        "a": params[0],
        "b": params[1],
        "c": params[2],
        "d": params[3],
        "e": params[4],
        "f": params[5],
        "g": params[6],
    }


def _fitness_config(config: dict, ga_config: dict) -> tuple[int, dict]:
    fitness_config = dict(config.get("fitness", {}))
    match_window = int(fitness_config.pop("max_time_window", ga_config["match_window"]))
    return match_window, fitness_config


def _empty_candle_signal_df(df):
    signal_df = df.copy()
    for col in CANDLE_SIGNAL_COLUMNS:
        signal_df[col] = 0
    return signal_df


def main() -> None:
    """Run candle parameter optimization and save signal outputs."""

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
    missing = [
        col
        for col in ["Open", "High", "Low", "Close", "turning_label"]
        if col not in df.columns
    ]
    if missing:
        raise ValueError(f"TP data missing required columns: {missing}")

    cpm_params_path = get_best_cpm_params_path(ticker, interval)
    cpm_params = load_json(cpm_params_path)
    ga_config = config["candle_ga"]
    bounds = config.get("candle_bounds")
    match_window, fitness_config = _fitness_config(config, ga_config)

    signal_df = _empty_candle_signal_df(df)
    pattern_results = {}
    for offset, signal_col in enumerate(CANDLE_SIGNAL_COLUMNS):
        print(f"optimizing candle pattern: {signal_col}")
        best_params, best_fitness, logbook = run_candle_ga(
            df,
            label_col="turning_label",
            open_col="Open",
            high_col="High",
            low_col="Low",
            close_col="Close",
            window=match_window,
            signal_cols=[signal_col],
            population_size=int(ga_config["population_size"]),
            generations=int(ga_config["generations"]),
            cx_prob=float(ga_config["cx_prob"]),
            mut_prob=float(ga_config["mut_prob"]),
            seed=int(ga_config["seed"]) + offset,
            bounds=bounds,
            fitness_config=fitness_config,
        )
        pattern_signal_df = generate_candle_signals(
            df,
            params=best_params,
            open_col="Open",
            high_col="High",
            low_col="Low",
            close_col="Close",
        )
        signal_df[signal_col] = pattern_signal_df[signal_col]
        _, pattern_fitness_details = calculate_candle_fitness(
            pattern_signal_df,
            label_col="turning_label",
            signal_cols=[signal_col],
            window=match_window,
            high_col="High",
            low_col="Low",
            close_col="Close",
            fitness_config=fitness_config,
        )
        pattern_results[signal_col] = {
            "best_params": _param_dict(best_params),
            "best_fitness": best_fitness,
            "fitness": pattern_fitness_details,
            "ga_logbook": logbook,
        }

    _, fitness_details = calculate_candle_fitness(
        signal_df,
        label_col="turning_label",
        window=match_window,
        high_col="High",
        low_col="Low",
        close_col="Close",
        fitness_config=fitness_config,
    )

    payload = {
        "ticker": ticker,
        "interval": interval,
        "indicator": "Candle",
        "target_label_source": "CPM",
        "cpm_params": {"P": cpm_params["P"], "T": cpm_params["T"]},
        "best_params": {
            pattern: result["best_params"] for pattern, result in pattern_results.items()
        },
        "best_fitness": {
            pattern: result["best_fitness"] for pattern, result in pattern_results.items()
        },
        "fitness": fitness_details,
        "pattern_results": pattern_results,
        "ga_config": ga_config,
        "fitness_config": {"max_time_window": match_window, **fitness_config},
    }

    params_path = get_indicator_param_path(ticker, interval, "candle")
    processed_signal_path = get_candle_patterns_data_path(ticker, interval)
    output_signal_path = get_candle_patterns_output_path(ticker, interval)
    figure_path = get_candle_patterns_figure_path(ticker, interval)
    individual_figure_dir = get_candle_pattern_figures_dir(ticker, interval)
    counts_figure_path = get_candle_pattern_counts_figure_path(ticker, interval)

    save_json(payload, params_path)
    save_dataframe(signal_df, processed_signal_path)
    save_dataframe(signal_df, output_signal_path)
    plot_candle_patterns(signal_df, figure_path)
    plot_individual_candle_patterns(signal_df, individual_figure_dir)
    plot_candle_pattern_counts(signal_df, counts_figure_path, CANDLE_SIGNAL_COLUMNS)

    pattern_counts = {col: int((signal_df[col] != 0).sum()) for col in CANDLE_SIGNAL_COLUMNS}
    print("best candle params by pattern:")
    print(json.dumps(payload["best_params"], indent=2))
    print("combined final fitness details:")
    print(json.dumps(fitness_details, indent=2))
    print("candle pattern counts:")
    print(json.dumps(pattern_counts, indent=2))
    print(f"Candle params: {params_path}")
    print(f"processed candle signals: {processed_signal_path}")
    print(f"output candle signals: {output_signal_path}")
    print(f"candle figure: {figure_path}")
    print(f"individual candle figures: {individual_figure_dir}")
    print(f"candle count figure: {counts_figure_path}")


if __name__ == "__main__":
    main()
