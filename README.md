# Stock Trading Research Project

This project is a clean Python implementation scaffold based on the paper
`Stock trading system based on echo state network and genetic algorithm`.

Current scope:

- CPM creates fixed turning point labels from OHLCV close prices.
- The Moving Average system creates buy/sell technical indicator signals.
- GA optimizes MA parameters against fixed CPM labels.
- RSI is now implemented as a GA-optimized indicator.
- ROC is now implemented as a GA-optimized indicator.
- Stochastic Oscillator is now implemented as a GA-optimized indicator.
- Candle patterns are implemented using only the patterns described in the paper:
  Hammer / Hanging Man, Dark Cloud Cover, Piercing Line, and Engulfing Pattern.
- Each candle pattern is exported as an independent binary feature.
- Bullish and bearish engulfing are split to preserve signal direction.
- Non-paper patterns such as doji, morning star, evening star, inverted hammer,
  and shooting star are not included in the default pipeline.
- Aggregate `candle_buy_signal` and `candle_sell_signal` are also exported.
- ESN training remains future work.

## Setup

```bash
pip install -r requirements.txt
```

## Run Order

```bash
python -m src.pipeline.run_data --config configs/spy_1d.yaml
python -m src.pipeline.run_cpm --config configs/spy_1d.yaml
python -m src.pipeline.run_ma_ga --config configs/spy_1d.yaml
python -m src.pipeline.run_rsi_ga --config configs/spy_1d.yaml
python -m src.pipeline.run_roc_ga --config configs/spy_1d.yaml
python -m src.pipeline.run_stochastic_ga --config configs/spy_1d.yaml
python -m src.pipeline.run_candle_patterns --config configs/spy_1d.yaml
python -m src.pipeline.build_esn_dataset --config configs/spy_1d.yaml
```

## Main Outputs

- `data/raw/SPY/1d/SPY.csv`
- `data/processed/SPY/1d/SPY_clean.csv`
- `data/processed/SPY/1d/SPY_with_tp.csv`
- `data/processed/SPY/1d/SPY_with_ma_signals.csv`
- `data/processed/SPY/1d/SPY_with_rsi_signals.csv`
- `data/processed/SPY/1d/SPY_with_roc_signals.csv`
- `data/processed/SPY/1d/SPY_with_stochastic_signals.csv`
- `data/processed/SPY/1d/SPY_with_candle_patterns.csv`
- `outputs/cpm/SPY/1d/best_cpm_params.json`
- `outputs/cpm/SPY/1d/cpm_search_results.csv`
- `outputs/indicator_params/SPY/1d/ma_params.json`
- `outputs/indicator_params/SPY/1d/rsi_params.json`
- `outputs/indicator_params/SPY/1d/roc_params.json`
- `outputs/indicator_params/SPY/1d/stochastic_params.json`
- `outputs/signals/SPY/1d/SPY_with_ma_signals.csv`
- `outputs/signals/SPY/1d/SPY_with_rsi_signals.csv`
- `outputs/signals/SPY/1d/SPY_with_roc_signals.csv`
- `outputs/signals/SPY/1d/SPY_with_stochastic_signals.csv`
- `outputs/signals/SPY/1d/SPY_with_candle_patterns.csv`
- `outputs/figures/SPY/1d/cpm_turning_points.png`
- `outputs/figures/SPY/1d/ma_signals.png`
- `outputs/figures/SPY/1d/rsi_signals.png`
- `outputs/figures/SPY/1d/roc_signals.png`
- `outputs/figures/SPY/1d/stochastic_signals.png`
- `outputs/figures/SPY/1d/candle_patterns.png`
- `outputs/figures/SPY/1d/candle_pattern_counts.png`

## Quick Inspection

```python
import pandas as pd

df = pd.read_csv("data/processed/SPY/1d/SPY_with_ma_signals.csv")
print(df["turning_label"].value_counts())
print(df["ma_buy_signal"].value_counts())
print(df["ma_sell_signal"].value_counts())

rsi_df = pd.read_csv("data/processed/SPY/1d/SPY_with_rsi_signals.csv")
print(rsi_df["rsi_buy_signal"].value_counts())
print(rsi_df["rsi_sell_signal"].value_counts())

roc_df = pd.read_csv("data/processed/SPY/1d/SPY_with_roc_signals.csv")
print(roc_df["roc_buy_signal"].value_counts())
print(roc_df["roc_sell_signal"].value_counts())

stoch_df = pd.read_csv("data/processed/SPY/1d/SPY_with_stochastic_signals.csv")
print(stoch_df["stoch_buy_signal"].value_counts())
print(stoch_df["stoch_sell_signal"].value_counts())

candle_df = pd.read_csv("data/processed/SPY/1d/SPY_with_candle_patterns.csv")
print(candle_df["candle_buy_signal"].value_counts())
print(candle_df["candle_sell_signal"].value_counts())
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
P/T values. Once selected, the CPM labels are fixed for MA, RSI, ROC, and
Stochastic GA optimization.

`build_esn_dataset` currently combines implemented indicator features, including
RSI, ROC, Stochastic, paper-defined candle pattern columns, and aggregate candle
signals when those indicator pipelines have been run. Aggregate candle signals
are rule-derived from same-row OHLC pattern detections and are included as
candidate ESN features, not as ESN targets. Hammer / Hanging Man is exported as
a shape feature but is not included in the aggregate buy/sell signals because
its direction depends on trend context.
