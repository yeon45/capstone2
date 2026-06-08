# Stock Trading Research Project

## Current Scope

- Download and preprocess stock OHLCV data with `yfinance`
- Generate CPM turning point labels
- Optimize technical-indicator signals with GA
- Build the final 9-column signal dataset for ESN input
- Convert CPM turning points into a triangle-wave target
- Train, validate, threshold, and test an ESN trading-score model with a single chronological split

## Project Structure

```text
.
|-- configs/
|   `-- spy_1d.yaml
|-- data/
|   |-- raw/
|   `-- processed/
|-- outputs/
|   |-- cpm/
|   |-- esn/
|   |-- figures/
|   |-- indicator_params/
|   `-- signals/
|-- src/
|   |-- cpm/
|   |-- data/
|   |-- esn/
|   |-- ga/
|   |-- indicators/
|   |-- pipeline/
|   |-- utils/
|   `-- visualization/
|-- requirements.txt
`-- README.md
```

## Install

```bash
pip install -r requirements.txt
```

## Run Pipeline

```bash
python -m src.pipeline.run_data --config configs/spy_1d.yaml
python -m src.pipeline.run_cpm --config configs/spy_1d.yaml
python -m src.pipeline.run_ma_ga --config configs/spy_1d.yaml
python -m src.pipeline.run_rsi_ga --config configs/spy_1d.yaml
python -m src.pipeline.run_roc_ga --config configs/spy_1d.yaml
python -m src.pipeline.run_stochastic_ga --config configs/spy_1d.yaml
python -m src.pipeline.run_candle_ga --config configs/spy_1d.yaml
python -m src.pipeline.build_esn_dataset --config configs/spy_1d.yaml
python -m src.pipeline.run_esn --config configs/spy_1d.yaml
```

## ESN Input Columns

- `ma_signal`
- `rsi_signal`
- `roc_signal`
- `stoch_signal`
- `candle_hammer_hanging_man_signal`
- `candle_dark_cloud_cover_signal`
- `candle_piercing_line_signal`
- `candle_bullish_engulfing_signal`
- `candle_bearish_engulfing_signal`

The default ESN input is `data/processed/{ticker}/{interval}/{ticker}_all_signals.csv`.
If that file is missing, the ESN pipeline falls back to
`outputs/signals/{ticker}/{interval}/{ticker}_all_indicator_signals.csv`.

The ESN does not predict price. CPM turning points are converted into
`triangle_target` in the range `[-1, 1]`, and the ESN learns this CPM-based
trading score from indicator signal columns. Scores below `-threshold` are buy
signals, scores above `threshold` are sell signals, and the rest are hold.

Important leakage note: `triangle_target` is an offline teacher label derived
from CPM turning points, so it can include future turning-point information. Test
targets are used only as evaluation truth. A live trading decision should use
the ESN output plus an online candidate-reversal confirmation/gating rule. This
repository does not implement backtest or online candidate-reversal gating yet.

## Main Outputs

- `data/raw/SPY/1d/SPY.csv`
- `data/processed/SPY/1d/SPY_clean.csv`
- `data/processed/SPY/1d/SPY_with_tp.csv`
- `data/processed/SPY/1d/SPY_all_signals.csv`
- `outputs/signals/SPY/1d/SPY_all_indicator_signals.csv`
- `outputs/esn/SPY/1d/esn_metrics.json`
- `outputs/esn/SPY/1d/esn_predictions.csv`
- `outputs/esn/SPY/1d/esn_config_used.json`

## Quick Check

```python
import pandas as pd

df = pd.read_csv("data/processed/SPY/1d/SPY_all_signals.csv")

print(df["turning_label"].value_counts())
print(df[[
    "ma_signal",
    "rsi_signal",
    "roc_signal",
    "stoch_signal",
    "candle_hammer_hanging_man_signal",
    "candle_dark_cloud_cover_signal",
    "candle_piercing_line_signal",
    "candle_bullish_engulfing_signal",
    "candle_bearish_engulfing_signal",
]].describe())
```

## TODO

- ESN output based trading rule improvement
- Backtest and performance evaluation
- Walk-forward validation extension
- Full pipeline batch execution script
- Tests
