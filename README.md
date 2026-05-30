# Stock Trading Research Project

## 현재 구현 범위

- `yfinance`로 OHLCV 데이터 다운로드 및 전처리
- CPM으로 고정된 매수/매도 변곡점 라벨 생성
- CPM 라벨을 기준으로 GA가 다음 지표의 파라미터 최적화
  - Moving Average
  - RSI
  - ROC
  - Stochastic Oscillator
  - Hammer / Hanging Man
  - Dark Cloud Cover
  - Piercing Line
  - Bullish Engulfing
  - Bearish Engulfing
- 최종 ESN 입력용 9개 signed signal 컬럼 생성
- ESN 학습/예측은 미구현

## 프로젝트 구조

```text
.
|-- configs/
|   `-- spy_1d.yaml              # 종목, 기간, CPM/GA 설정
|-- data/
|   |-- raw/                     # 다운로드한 원본 OHLCV 데이터
|   `-- processed/               # 전처리 데이터, CPM 라벨, 지표 신호
|-- outputs/
|   |-- cpm/                     # CPM 탐색 결과, 선택 파라미터
|   |-- figures/                 # CPM/지표 시각화 이미지
|   |-- indicator_params/        # GA 최적 파라미터 JSON
|   `-- signals/                 # 지표별 신호, 최종 ESN 입력 신호
|-- src/
|   |-- cpm/                     # CPM 탐색, 변곡점 라벨링
|   |-- data/                    # 데이터 다운로드, 전처리
|   |-- ga/                      # GA 최적화, fitness 계산
|   |-- indicators/              # 지표별 signal 생성
|   |-- pipeline/                # 단계별 실행 스크립트
|   |-- utils/                   # config, IO, path helper
|   `-- visualization/           # 결과 시각화
|-- requirements.txt
`-- README.md
```

## 설치

```bash
pip install -r requirements.txt
```

## 설정

- 기본 설정 파일: `configs/spy_1d.yaml`
- 기본 종목: `SPY`
- 기본 주기: `1d`
- 기본 시작일: `2010-01-01`
- `end: null`이면 실행 시점 기준 최신 데이터까지 다운로드
- 다른 실험을 하려면 설정 파일을 복사한 뒤 `ticker`, `interval`, `start`, `end`, CPM/GA 설정 수정

## 실행 순서

```bash
python -m src.pipeline.run_data --config configs/spy_1d.yaml
python -m src.pipeline.run_cpm --config configs/spy_1d.yaml
python -m src.pipeline.run_ma_ga --config configs/spy_1d.yaml
python -m src.pipeline.run_rsi_ga --config configs/spy_1d.yaml
python -m src.pipeline.run_roc_ga --config configs/spy_1d.yaml
python -m src.pipeline.run_stochastic_ga --config configs/spy_1d.yaml
python -m src.pipeline.run_candle_ga --config configs/spy_1d.yaml
python -m src.pipeline.build_esn_dataset --config configs/spy_1d.yaml
```

## 파이프라인 역할

| 순서 | 모듈 | 역할 |
| --- | --- | --- |
| 1 | `run_data` | OHLCV 다운로드, 전처리 |
| 2 | `run_cpm` | CPM 파라미터 탐색, 변곡점 라벨 생성 |
| 3 | `run_ma_ga` | MA 파라미터 최적화, MA signal 생성 |
| 4 | `run_rsi_ga` | RSI 파라미터 최적화, RSI signal 생성 |
| 5 | `run_roc_ga` | ROC 파라미터 최적화, ROC signal 생성 |
| 6 | `run_stochastic_ga` | Stochastic 파라미터 최적화, Stochastic signal 생성 |
| 7 | `run_candle_ga` | Candle 파라미터 최적화, Candle signal 생성 |
| 8 | `build_esn_dataset` | 지표 signal 병합, ESN 입력 데이터 생성 |

## 주요 산출물

- `data/raw/SPY/1d/SPY.csv`
- `data/processed/SPY/1d/SPY_clean.csv`
- `data/processed/SPY/1d/SPY_with_tp.csv`
- `data/processed/SPY/1d/SPY_with_ma_signals.csv`
- `data/processed/SPY/1d/SPY_with_rsi_signals.csv`
- `data/processed/SPY/1d/SPY_with_roc_signals.csv`
- `data/processed/SPY/1d/SPY_with_stochastic_signals.csv`
- `data/processed/SPY/1d/SPY_with_candle_patterns.csv`
- `data/processed/SPY/1d/SPY_all_signals.csv`
- `outputs/cpm/SPY/1d/best_cpm_params.json`
- `outputs/cpm/SPY/1d/cpm_search_results.csv`
- `outputs/indicator_params/SPY/1d/ma_params.json`
- `outputs/indicator_params/SPY/1d/rsi_params.json`
- `outputs/indicator_params/SPY/1d/roc_params.json`
- `outputs/indicator_params/SPY/1d/stochastic_params.json`
- `outputs/indicator_params/SPY/1d/candle_params.json`
- `outputs/signals/SPY/1d/SPY_all_indicator_signals.csv`
- `outputs/figures/SPY/1d/*.png`

## ESN 입력 컬럼

- `ma_signal`
- `rsi_signal`
- `roc_signal`
- `stoch_signal`
- `candle_hammer_hanging_man_signal`
- `candle_dark_cloud_cover_signal`
- `candle_piercing_line_signal`
- `candle_bullish_engulfing_signal`
- `candle_bearish_engulfing_signal`

## 빠른 확인

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

## CPM 선택 방식

- 설정 위치: `configs/spy_1d.yaml`
- 탐색 범위: `cpm.p_values`, `cpm.t_values`
- 탐색 결과: `outputs/cpm/SPY/1d/cpm_search_results.csv`
- 선택 파라미터: `outputs/cpm/SPY/1d/best_cpm_params.json`
- 기본 선택 방식: `pareto_knee`
- 추가 선택 방식: `score`

## 실행 TODO

- ESN 학습/검증/예측 파이프라인 구현
- ESN 출력 기반 trading rule 구현
- backtest 및 성능 평가 구현
- 전체 파이프라인 일괄 실행 스크립트 추가
- CPM, indicator, GA fitness, ESN dataset merge 테스트 추가
- ticker/interval별 config 예시 추가
