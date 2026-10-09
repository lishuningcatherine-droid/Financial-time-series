# Financial Time Series — EDA

Exploratory data analysis of SPY daily data, with 9 related stocks/ETFs as exogenous features. Analysis and visualization only — no models are trained.

- `data/yfinance_prices.csv`: raw data (yfinance, 2018-01-02 to 2026-09-01)
- `eda.ipynb`: sectioned analysis notebook (executed, with outputs)
- `figures/`: all figures
- `eda_report.md`: key findings and feature-selection recommendations for modeling

Reproduce:

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace eda.ipynb
```
