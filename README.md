# Stock Trading Research Project

This project is a clean Python implementation scaffold based on the paper
`Stock trading system based on echo state network and genetic algorithm`.

Current scope:

- CPM creates fixed turning point labels from OHLCV close prices.
- The Moving Average system creates buy/sell technical indicator signals.
- GA optimizes MA parameters against fixed CPM labels.
- ESN training will be added later.
- RSI, ROC, Stochastic, and Candle modules are placeholders for later work.

## Setup

```bash
pip install -r requirements.txt
```

## Run Order

```bash
python -m src.pipeline.run_data --config configs/spy_1d.yaml
python -m src.pipeline.run_cpm --config configs/spy_1d.yaml
python -m src.pipeline.run_ma_ga --config configs/spy_1d.yaml
python -m src.pipeline.build_esn_dataset --config configs/spy_1d.yaml
```

## Main Outputs

- `data/raw/SPY/1d/SPY.csv`
- `data/processed/SPY/1d/SPY_clean.csv`
- `data/processed/SPY/1d/SPY_with_tp.csv`
- `data/processed/SPY/1d/SPY_with_ma_signals.csv`
- `outputs/cpm/SPY/1d/best_cpm_params.json`
- `outputs/cpm/SPY/1d/cpm_search_results.csv`
- `outputs/indicator_params/SPY/1d/ma_params.json`
- `outputs/signals/SPY/1d/SPY_with_ma_signals.csv`
- `outputs/figures/SPY/1d/cpm_turning_points.png`
- `outputs/figures/SPY/1d/ma_signals.png`

## Quick Inspection

```python
import pandas as pd

df = pd.read_csv("data/processed/SPY/1d/SPY_with_ma_signals.csv")
print(df["turning_label"].value_counts())
print(df["ma_buy_signal"].value_counts())
print(df["ma_sell_signal"].value_counts())
```

## Notes

The CPM implementation is local and minimal. It does not import from external CPM
repositories. CPM parameter search writes the full P,T grid to
`outputs/cpm/SPY/1d/cpm_search_results.csv`.

CPM supports two selection styles:

- `score`: a simple heuristic score using point count, buy/sell balance, and gap quality.
- `pareto_knee`: the recommended default. It builds a Pareto front that maximizes
  total absolute move captured while minimizing the number of turning points, then
  selects the knee of that tradeoff curve.

`pareto_knee` is intended to choose a balanced CPM parameter pair between too many
noisy turning points from small P/T values and too few turning points from large
P/T values. Once selected, the CPM labels are fixed for MA GA optimization.
