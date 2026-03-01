# HAR-RV Boosted

> **Extending the HAR-RV Model with XGBoost for Realized Volatility Forecasting**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview

This project extends the classic HAR-RV model (Corsi, 2009) by replacing the linear regression with **XGBoost** and enriching the feature set from 5 to **20 predictors** across 5 categories. The model predicts 5-day-ahead realized volatility for S&P 500 large-cap stocks using 1-minute intraday data.

The full methodology and results are documented in:
- **[Research Notebook](notebook_research.ipynb)** --- step-by-step walkthrough of the research process
- **[Academic Paper](har_rv_model_paper.pdf)** --- formal write-up (French, working paper format)

## Key Results

Evaluated on 5 stocks (AAPL, MSFT, NVDA, JPM, META) with walk-forward backtesting and purging:

| Configuration | Spearman IC | vs. Baseline |
|---------------|:-----------:|:------------:|
| Baseline (HAR-RV features, corrections applied) | 0.291 | --- |
| + Higher moments (RSkew, RKurt, Jump ratio) | 0.310 | +6.5% |
| + VIX-adaptive training window | 0.331 | +13.7% |
| + Leverage features (Ret_w, Leverage_22d) | **0.394** | **+35.4%** |

## Features

20 predictors organized in 5 categories:

| Category | Features | Reference |
|----------|----------|-----------|
| **HAR-RV core** | RV_d, RV_w, RV_m, RV_q | Corsi (2009) |
| **Asymmetry** | RV_neg_w, RV_pos_w, RV_asym_w | Barndorff-Nielsen & Shephard (2004) |
| **Jumps & moments** | J_w, J_ratio_w, RSkew_w, RKurt_w | Andersen et al. (2007), Amaya et al. (2015) |
| **Market regime** | VIX, VIX_ret_5d, VIX_level | CBOE |
| **Leverage & ratios** | Ret_w, Leverage_22d, RV_ratio_wm, RV_ratio_wq, RV_trend_5d, RV_change_1d | Black (1976) |

## Project Structure

```
HAR-RV-boosted/
├── har_rv_model.py           # Main model (features, target, backtest, XGBoost)
├── notebook_research.ipynb   # Research notebook (complementary to the paper)
├── har_rv_model_paper.pdf    # Academic paper (compiled PDF)
├── requirements.txt          # Python dependencies
├── .env.example              # API keys template
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

Or run the full backtest:

```bash
python har_rv_model.py
```

## Methodology Highlights

- **Target**: log(RV_{t+5} / RV_m) --- relative volatility change, decoupled from features
- **Walk-forward**: expanding window with purge gap of 4 days (avoids target overlap)
- **Adaptive window**: training window size interpolated between 189--504 days based on VIX regime
- **Evaluation**: Spearman Information Coefficient (rank correlation between predictions and realized values)

## References

- Corsi, F. (2009). *A Simple Approximate Long-Memory Model of Realized Volatility.* Journal of Financial Econometrics, 7(2), 174--196.
- Andersen, T. G., Bollerslev, T., & Diebold, F. X. (2007). *Roughing it up.* Review of Economics and Statistics, 89(4), 701--720.
- Barndorff-Nielsen, O. E., & Shephard, N. (2004). *Power and bipower variation.* Econometrica, 72(3), 885--925.
- Black, F. (1976). *Studies of stock price volatility changes.* Proceedings of the ASA.
- Amaya, D., et al. (2015). *Does realized skewness and kurtosis predict the cross-section of equity returns?* Journal of Financial Economics, 118(1), 135--167.

## Author

**Paul MONTIER**
M2 AQTC --- 2025

## License

MIT License
