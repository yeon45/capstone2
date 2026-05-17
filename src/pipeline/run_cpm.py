"""Run CPM parameter search and turning point labeling."""

from __future__ import annotations

import argparse

from src.cpm.label import add_turning_labels
from src.cpm.search import grid_search_cpm, select_best_cpm_params
from src.utils.config import load_config
from src.utils.io import load_dataframe, save_dataframe, save_json
from src.utils.paths import (
    get_best_cpm_params_path,
    get_clean_data_path,
    get_cpm_figure_path,
    get_cpm_search_results_path,
    get_tp_data_path,
)
from src.visualization.plot_cpm import plot_cpm_turning_points


def _best_cpm_payload(ticker: str, interval: str, selection_method: str, best: dict) -> dict:
    """Build a JSON-serializable payload for selected CPM parameters."""

    return {
        "ticker": ticker,
        "interval": interval,
        "selection_method": selection_method,
        "P": float(best["P"]),
        "T": int(best["T"]),
        "num_points": int(best["num_points"]),
        "num_buy": int(best["num_buy"]),
        "num_sell": int(best["num_sell"]),
        "mean_gap": float(best["mean_gap"]),
        "coverage": float(best["coverage"]),
        "total_abs_move": float(best["total_abs_move"]),
        "avg_abs_move": float(best["avg_abs_move"]),
        "efficiency": float(best["efficiency"]),
        "score": float(best["score"]),
    }


def main() -> None:
    """Run CPM search, label generation, and plotting."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/spy_1d.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ticker = config["ticker"]
    interval = config["interval"]
    clean_path = get_clean_data_path(ticker, interval)
    if not clean_path.exists():
        raise FileNotFoundError(f"Clean data not found: {clean_path}. Run run_data first.")

    df = load_dataframe(clean_path)
    if "Close" not in df.columns:
        raise ValueError("Clean data must contain Close column")

    cpm_config = config["cpm"]
    search_df = grid_search_cpm(df["Close"], cpm_config["p_values"], cpm_config["t_values"])
    search_path = get_cpm_search_results_path(ticker, interval)
    save_dataframe(search_df, search_path)

    selection_method = cpm_config.get("selection_method", "pareto_knee")
    best = select_best_cpm_params(search_df, selection_method)
    P = float(best["P"])
    T = int(best["T"])
    params_path = get_best_cpm_params_path(ticker, interval)
    save_json(_best_cpm_payload(ticker, interval, selection_method, best), params_path)

    labeled_df = add_turning_labels(df, P=P, T=T)
    tp_path = get_tp_data_path(ticker, interval)
    save_dataframe(labeled_df, tp_path)

    figure_path = get_cpm_figure_path(ticker, interval)
    plot_cpm_turning_points(labeled_df, figure_path)

    print(f"selected P: {P}")
    print(f"selected T: {T}")
    print(f"bottom/buy labels: {int((labeled_df['turning_label'] == 1).sum())}")
    print(f"top/sell labels: {int((labeled_df['turning_label'] == -1).sum())}")
    print(f"search results: {search_path}")
    print(f"best CPM params: {params_path}")
    print(f"labeled data: {tp_path}")
    print(f"CPM figure: {figure_path}")


if __name__ == "__main__":
    main()
