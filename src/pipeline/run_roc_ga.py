"""Run ROC GA optimization against fixed CPM labels."""

from __future__ import annotations

import argparse
import json

from src.ga.fitness import calculate_price_error_buy_sell_fitness
from src.ga.roc_optimizer import run_roc_ga
from src.indicators.roc import generate_roc_signals
from src.utils.config import load_config
from src.utils.io import load_dataframe, load_json, save_dataframe, save_json
from src.utils.paths import (
    get_best_cpm_params_path,
    get_indicator_param_path,
    get_roc_figure_path,
    get_roc_signal_data_path,
    get_roc_signal_output_path,
    get_tp_data_path,
)
from src.visualization.plot_signals import plot_roc_signals


def _param_dict(params: list[float]) -> dict:
    return {
        "period": params[0],
        "buy_threshold": params[1],
        "sell_threshold": params[2],
    }


def _fitness_config(config: dict, ga_config: dict) -> tuple[int, dict]:
    fitness_config = dict(config.get("fitness", {}))
    match_window = int(fitness_config.pop("max_time_window", ga_config["match_window"]))
    return match_window, fitness_config


def main() -> None:
    """Run ROC parameter optimization and save signal outputs."""

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
    missing = [col for col in ["High", "Low", "Close", "turning_label"] if col not in df.columns]
    if missing:
        raise ValueError(f"TP data missing required columns: {missing}")

    cpm_params_path = get_best_cpm_params_path(ticker, interval)
    cpm_params = load_json(cpm_params_path)
    ga_config = config["roc_ga"]
    bounds = config.get("roc_bounds")
    match_window, fitness_config = _fitness_config(config, ga_config)

    best_params, best_fitness, logbook = run_roc_ga(
        df,
        label_col="turning_label",
        close_col="Close",
        window=match_window,
        population_size=int(ga_config["population_size"]),
        generations=int(ga_config["generations"]),
        cx_prob=float(ga_config["cx_prob"]),
        mut_prob=float(ga_config["mut_prob"]),
        seed=int(ga_config["seed"]),
        bounds=bounds,
        fitness_config=fitness_config,
    )

    signal_df = generate_roc_signals(
        df,
        period=int(best_params[0]),
        buy_threshold=float(best_params[1]),
        sell_threshold=float(best_params[2]),
        close_col="Close",
    )
    _, fitness_details = calculate_price_error_buy_sell_fitness(
        signal_df,
        label_col="turning_label",
        buy_signal_col="roc_buy_signal",
        sell_signal_col="roc_sell_signal",
        price_col="Close",
        high_col="High",
        low_col="Low",
        close_col="Close",
        max_time_window=match_window,
        **fitness_config,
    )

    payload = {
        "ticker": ticker,
        "interval": interval,
        "indicator": "ROC",
        "target_label_source": "CPM",
        "cpm_params": {"P": cpm_params["P"], "T": cpm_params["T"]},
        "best_params": _param_dict(best_params),
        "best_fitness": best_fitness,
        "fitness": fitness_details,
        "ga_config": ga_config,
        "fitness_config": {"max_time_window": match_window, **fitness_config},
        "ga_logbook": logbook,
    }

    params_path = get_indicator_param_path(ticker, interval, "roc")
    processed_signal_path = get_roc_signal_data_path(ticker, interval)
    output_signal_path = get_roc_signal_output_path(ticker, interval)
    figure_path = get_roc_figure_path(ticker, interval)

    save_json(payload, params_path)
    save_dataframe(signal_df, processed_signal_path)
    save_dataframe(signal_df, output_signal_path)
    plot_roc_signals(
        signal_df,
        figure_path,
        buy_threshold=float(best_params[1]),
        sell_threshold=float(best_params[2]),
    )

    print("best ROC params:")
    print(json.dumps(payload["best_params"], indent=2))
    print("final fitness details:")
    print(json.dumps(fitness_details, indent=2))
    print(f"ROC buy signals: {int(signal_df['roc_buy_signal'].sum())}")
    print(f"ROC sell signals: {int(signal_df['roc_sell_signal'].sum())}")
    print(f"ROC params: {params_path}")
    print(f"processed ROC signals: {processed_signal_path}")
    print(f"output ROC signals: {output_signal_path}")
    print(f"ROC figure: {figure_path}")


if __name__ == "__main__":
    main()
