# HAR-RV Boosted

> **Extending the HAR-RV Model with ElasticNet for Realized Volatility Forecasting**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview

This project extends the classic HAR-RV model (Corsi, 2009) with **ElasticNet regularization** (L1+L2) and an enriched feature set of **20 predictors** across 6 categories. The model predicts 5-day-ahead realized volatility for S&P 500 large-cap stocks using 1-minute intraday data.

The full methodology, analysis and results are documented in:
- **[ElasticNet Notebook](notebook_elasticnet.ipynb)** --- from Ridge to ElasticNet: feature selection, walk-forward backtesting, and anti-leakage validation

## Key Results

Evaluated on 5 stocks (AAPL, MSFT, NVDA, JPM, META) with walk-forward backtesting, purging and adaptive training window:

| Model | Features | Spearman IC |
|-------|:--------:|:-----------:|
| Ridge (3 features HAR) | RV_w, RV_m, RV_q | 0.451 |
| Ridge (20 features) | All 20 | 0.452 |
| Lasso (20 features) | All 20 | 0.473 |
| ElasticNet (20 features) | All 20 | 0.474 |
| **ElasticNet (12 features selected)** | **Greedy selection** | **0.494** |

ElasticNet with feature selection achieves **+9.3%** improvement over Ridge baseline.

## Features

20 predictors organized in 6 categories:

| Category | Features | Reference |
|----------|----------|-----------|
| **HAR-RV core** | RV_w, RV_m, RV_q, RV_neg_w, J_w, VIX | Corsi (2009) |
| **Multi-timeframe** | RV_overnight, Parkinson, Vol_ratio | Parkinson (1980) |
| **Intraday 1-min** | RV_1min, Vol_ratio_1m5m, Vol_AM_PM, Autocorr | Microstructure |
| **Higher moments** | RSkew, RKurt, Jump_ratio | Amaya et al. (2015) |
| **Leverage** | Ret_w, Leverage_22d | Black (1976) |
| **Cross-sectional** | RV_w_zscore, RV_w_rank_delta | Relative positioning |

## Project Structure

```
HAR-RV-boosted/
├── har_rv_model.py           # Core model (features, target, backtest, ElasticNet)
├── notebook_elasticnet.ipynb  # Main notebook (analysis + results)
├── data/                      # Stock data (1-min intraday, OHLCV)
├── requirements.txt           # Python dependencies
├── .env.example               # API keys template (Alpaca)
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the model

```python
from har_rv_model import HARRVModel

model = HARRVModel(horizon=5, train_window=252)
vix = model.get_vix()
result = model.backtest('AAPL', vix)

print(f"IC: {result['ic']:.3f}")
print(f"Hit Rate: {result['hit_rate']:.1%}")
```

## Methodology

- **Target**: $\log(RV_{t+5} / RV_m)$ --- log-ratio of future 5-day vol vs current monthly vol
- **Walk-forward**: expanding window with purge gap of 4 days (avoids target overlap)
- **Adaptive window**: training window interpolated between 189--504 days based on VIX level (short window in stress, long in calm)
- **Regularization**: ElasticNet ($\alpha=0.01$, $l_1=0.5$) --- L1 zeros out noisy features, L2 stabilizes correlated groups
- **Feature selection**: greedy backward elimination on 1st temporal half, validated OOS on 2nd half (no look-ahead bias)
- **Evaluation**: Spearman IC (rank correlation between predictions and realized values)

## References

- Corsi, F. (2009). *A Simple Approximate Long-Memory Model of Realized Volatility.* Journal of Financial Econometrics, 7(2), 174--196.
- Zou, H. & Hastie, T. (2005). *Regularization and variable selection via the elastic net.* JRSS-B, 67(2), 301--320.
- Andersen, T. G., Bollerslev, T., & Diebold, F. X. (2003). *Modeling and Forecasting Realized Volatility.* Econometrica, 71(2), 579--625.
- Barndorff-Nielsen, O. E., & Shephard, N. (2004). *Power and bipower variation.* Econometrica, 72(3), 885--925.
- Black, F. (1976). *Studies of stock price volatility changes.* Proceedings of the ASA.
- Amaya, D., et al. (2015). *Does realized skewness and kurtosis predict the cross-section of equity returns?* Journal of Financial Economics, 118(1), 135--167.
- Parkinson, M. (1980). *The extreme value method for estimating the variance of the rate of return.* Journal of Business, 53(1), 61--65.

## Author

**Paul MONTIER**
M2 AQTC --- 2025

## License

MIT License
