"""Run MA GA optimization against fixed CPM labels."""

from __future__ import annotations

import argparse
import json

from src.ga.fitness import calculate_price_error_buy_sell_fitness
from src.ga.ma_optimizer import run_ma_ga
from src.indicators.ma import generate_ma_signals
from src.utils.config import load_config
from src.utils.io import load_dataframe, load_json, save_dataframe, save_json
from src.utils.paths import (
    get_best_cpm_params_path,
    get_indicator_param_path,
    get_ma_figure_path,
    get_ma_signal_data_path,
    get_ma_signal_output_path,
    get_tp_data_path,
)
from src.visualization.plot_signals import plot_ma_signals


def _param_dict(params: list[float]) -> dict:
    return {
        "n": params[0],
        "N": params[1],
        "a_buy": params[2],
        "b_buy": params[3],
        "c_buy": params[4],
        "a_sell": params[5],
        "b_sell": params[6],
        "c_sell": params[7],
    }


def main() -> None:
    """Run MA parameter optimization and save signal outputs."""

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
    ga_config = config["ma_ga"]
    bounds = config.get("ma_bounds")

    best_params, best_fitness, logbook = run_ma_ga(
        df,
        label_col="turning_label",
        close_col="Close",
        window=int(ga_config["match_window"]),
        population_size=int(ga_config["population_size"]),
        generations=int(ga_config["generations"]),
        cx_prob=float(ga_config["cx_prob"]),
        mut_prob=float(ga_config["mut_prob"]),
        seed=int(ga_config["seed"]),
        bounds=bounds,
    )

    signal_df = generate_ma_signals(df, best_params, close_col="Close", normalize=True)
    _, fitness_details = calculate_price_error_buy_sell_fitness(
        signal_df,
        label_col="turning_label",
        buy_signal_col="ma_buy_signal",
        sell_signal_col="ma_sell_signal",
        price_col="Close",
        high_col="High",
        low_col="Low",
        close_col="Close",
        max_time_window=int(ga_config["match_window"]),
    )

    payload = {
        "ticker": ticker,
        "interval": interval,
        "indicator": "MA",
        "target_label_source": "CPM",
        "cpm_params": {"P": cpm_params["P"], "T": cpm_params["T"]},
        "best_params": _param_dict(best_params),
        "best_fitness": best_fitness,
        "fitness": fitness_details,
        "ga_config": ga_config,
        "ga_logbook": logbook,
    }

    params_path = get_indicator_param_path(ticker, interval, "ma")
    processed_signal_path = get_ma_signal_data_path(ticker, interval)
    output_signal_path = get_ma_signal_output_path(ticker, interval)
    figure_path = get_ma_figure_path(ticker, interval)

    save_json(payload, params_path)
    save_dataframe(signal_df, processed_signal_path)
    save_dataframe(signal_df, output_signal_path)
    plot_ma_signals(signal_df, figure_path)

    print("best MA params:")
    print(json.dumps(payload["best_params"], indent=2))
    print("final fitness details:")
    print(json.dumps(fitness_details, indent=2))
    print(f"MA buy signals: {int(signal_df['ma_buy_signal'].sum())}")
    print(f"MA sell signals: {int(signal_df['ma_sell_signal'].sum())}")
    print(f"MA params: {params_path}")
    print(f"processed MA signals: {processed_signal_path}")
    print(f"output MA signals: {output_signal_path}")
    print(f"MA figure: {figure_path}")


if __name__ == "__main__":
    main()
