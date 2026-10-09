# SPY Daily Feature EDA Report

> This stage is exploratory only — **no models were trained**. All findings describe correlations and **do not imply causation**.
> Full code and per-figure observations are in [`eda.ipynb`](eda.ipynb); figures are in [`figures/`](figures/).

## 1. Data and setup

| Item | Detail |
|---|---|
| Data file | `data/yfinance_prices.csv` (yfinance, long format: `date, ticker, open, high, low, close, adj_close, volume`) |
| Coverage | 2018-01-02 to 2026-09-01, daily, 10 tickers, 21,778 rows |
| Primary target | **SPY**; **same-day** log returns of AAPL, AMZN, GOOGL, JPM, META, MSFT, NVDA, QQQ, XOM as exogenous features |
| Price | Returns and labels use `adj_close` (split- and dividend-adjusted) |
| Labels | `y = 1 if adj_close(t+1) > adj_close(t) else 0`; `next_ret = ln(P_{t+1}/P_t)` (observation only) |
| Usable sample | 1,976 trading days (2018-10-16 to 2026-08-27, after removing the 200-day warm-up, the last day and the missing days) |
| Share of next-day up moves | 55.3% |

**Data quality**: no missing values, duplicate dates, OHLC inconsistencies, non-positive prices or zero volume. JPM and XOM each miss 2026-08-28; as agreed, it is kept as NaN with no filling.
Returns are computed on the aligned wide table with `pct_change(fill_method=None)` / `log(P/P.shift)`, so that day and the following day are both NaN and no spurious two-day return is created.
`close` is already split-adjusted. The 17 days with |daily return| > 15% all correspond to real events (March 2020 COVID crash, earnings days, the 2025-01-27 DeepSeek shock, the 2025-04-09 tariff pause) and are all kept.

**No leakage**: features use only data up to day t (all rolling windows look backward). The notebook includes a self-check: after truncating the data at 2020-03-16, 2023-06-30 and 2026-01-02 and recomputing, features are identical to the full-sample values (max difference 0).
The volatility-regime threshold uses an **expanding-window median**, not the full-sample median.

**30 features**: `ret_1d, ret_5d, ret_20d, ret_lag1–5, vol_20d, vol_60d, ma_dev_20/50/200, rsi_14, macd, macd_signal, macd_hist` (all three MACD terms normalized by price)`, vol_chg, vol_ratio_20, hl_range, gap, r_AAPL … r_XOM`.

## 2. Univariate features

- **Fat tails and skew are pervasive** (Figs. 1–2): SPY's daily return has excess kurtosis ≈ 13.6 and skew -0.54; the Jarque-Bera test strongly rejects normality. `gap` has kurtosis ≈ 28. Volatility features are strongly right-skewed; trend features are left-skewed.
- **Stationarity** (Fig. 4): only the price levels are non-stationary (ADF p ≈ 0.99); all returns and ratio-type features are stationary. `ma_dev_200`, `vol_60d` and `vol_20d` are closest to the boundary — highly persistent.
- **Time series** (Fig. 3): SPY rose ~3.2x; volume spikes during sell-offs and high-volatility periods.

## 3. Relationships between features (focus)

### 3.1 Correlation structure (Figs. 5–8)
Hierarchical clustering splits the features into three blocks:

1. **Same-day return block**: SPY vs QQQ (ρ ≈ 0.94); MSFT, AAPL, GOOGL, JPM same-day returns (0.70–0.77); XOM is the weakest (≈ 0.48).
2. **Trend/momentum block**: `ma_dev_20/50/200, macd, macd_signal, macd_hist, rsi_14, ret_5d, ret_20d`, within-block correlations 0.7–0.95.
3. **Volatility block**: `vol_20d, vol_60d, hl_range`, negatively correlated with the trend block (declines come with higher volatility).

`vol_chg` and `vol_ratio_20` form a separate group, weakly negative with returns. `ret_lag1–5` are nearly uncorrelated with each other.

29 feature pairs have |r| > 0.7, and **most are mechanical correlations caused by overlapping definitions** — e.g. MACD is the difference of two moving averages, and both `ret_20d` and `ma_dev_20` measure the past month's move.

### 3.2 Highly collinear groups (|r| > 0.9)
| Group | Members | Recommendation |
|---|---|---|
| 1 | `ma_dev_50`, `macd`, `macd_signal` | Keep one (e.g. `ma_dev_50`) |
| 2 | `ret_1d`, `r_QQQ` | Move almost in lockstep; keep one |

In addition, `ret_20d`, `ma_dev_20` and `rsi_14` correlate 0.8–0.9 with group 1 (near-collinear).

### 3.3 Unstable relationships (Figs. 9–10)
| Pair | Full-sample r | 60-day rolling range | Stability |
|---|---|---|---|
| SPY ~ QQQ | 0.94 | 0.73 to 0.99 | **Stable** |
| ma_dev_50 ~ macd | 0.95 | 0.47 to 0.98 | Fairly stable (dips at trend reversals) |
| ma_dev_50 ~ macd_signal | 0.89 | -0.01 to 0.96 | Unstable |
| SPY ~ JPM | 0.70 | 0.04 to 0.94 | Unstable |
| SPY ~ XOM | 0.48 | -0.56 to 0.90 | **Least stable**: ~0.8 in 2019–2020, near 0 from 2024, ~-0.5 in 2026 |
| SPY ~ volume change | -0.24 | -0.76 to 0.27 | Unstable, occasionally flips sign |
| vol_20d ~ vol_ratio_20 | -0.06 | -0.69 to 0.67 | Full-sample value is not representative |

**Regime dependence**: in high-volatility periods (`vol_20d` above its historical expanding median), cross-stock correlations are broadly higher. For example, AAPL~NVDA rises from 0.31 to 0.66 and AAPL~MSFT from 0.43 to 0.73 — correlations converge in a crisis. XOM's correlation with tech rises from ~0 to 0.2–0.3. The negative link between `ma_dev_200` and volatility deepens from about -0.25 to about -0.65.

### 3.4 Lead-lag (Fig. 11)
- **Past declines lead future increases in volatility**: past values of `ret_1d` are negatively correlated with current `hl_range` and `vol_20d` (about -0.1 to -0.2, consistent with the leverage effect).
- **Volume falls back after a big-move day**: corr(|ret|_{t-1}, vol_chg_t) ≈ -0.18; the same-day volume spike correlation is about +0.22.
- The ±0.1 spikes at non-zero lags for SPY vs QQQ and XOM mirror SPY's own ACF pattern, so they are **more likely spurious lead-lag transmitted through autocorrelation**. There is no evidence that any single stock leads SPY.

## 4. Features vs. next-day direction

- **Direction is essentially not identifiable from any single feature** (Fig. 12): for all 30 features the up and down groups overlap heavily; the smallest Mann-Whitney p-value is 0.13 and none is significant.
- **Linear and rank correlations with the next-day return are all weak** (Fig. 13): the largest |Spearman ρ| is 0.050 (`rsi_14`), just outside the ±0.044 band. The top features are `rsi_14` (-), `hl_range` (+), `r_AAPL` (-), `vol_60d` (+) and `ret_5d` (-), together pointing to "weak mean reversion + slightly higher returns after high volatility".
  For `ret_1d`, `r_AAPL` and `r_GOOGL`, Pearson (about -0.09 to -0.14) is much stronger than Spearman (about -0.03 to -0.04), showing that this correlation is **driven mainly by a handful of extreme reversal days in March 2020**.
- **Mutual information** (Fig. 14): MI with direction `y` sits near the permutation noise level for all features. 16 features have MI with `next_ret` above the noise level, but their |ρ| with `|next_ret|` is 0.26–0.38 while their ρ with `y` is ≈ 0.
  In other words, **their non-linear information is mainly about the size of the next day's move, not its direction**.
- **Autocorrelation** (Fig. 15): the full-sample lag-1 ACF of returns is about -0.14 and Ljung-Box is significant, but **excluding 2020 it is only -0.03**, and the rank ACF is also -0.03 — not a stable pattern.
  The ACF of |returns| is significant at all 30 lags (lag 1 ≈ 0.34): strong volatility clustering.
- **Stability over time** (Fig. 16): `rsi_14` and `ma_dev_50` are negatively correlated with the next-day return in all 9 years — the most stable weak signals — but the yearly magnitude is only -0.03 to -0.18. The negative correlations of `r_AAPL` and `r_GOOGL` come almost entirely from 2020, and AAPL even flips positive in 2024. The sign of `ret_lag3` alternates across years.

## 5. Key conclusions

1. **Strongly inter-correlated features**: same-day cross-asset returns (especially SPY~QQQ), the trend/momentum family (MA deviation + MACD + RSI + multi-day returns) and the volatility family (`vol_20d`/`vol_60d`/`hl_range`); each family is highly redundant internally.
2. **Features most related to direction**: no single feature has a reliable relationship with next-day direction. The relatively most stable are `rsi_14` and `ma_dev_50` (weakly negative, same sign every year) and `hl_range` and `vol_60d` (weakly positive), but the effect sizes are tiny (|ρ| ≤ 0.05).
   **The relationship with the next day's move size is much stronger** (|ρ| ≈ 0.3).
3. **Unstable relationships**: SPY's correlation with XOM and JPM, the volume–return relationship, and most feature–next-day-return relationships (driven by 2020 or flipping sign by year). The correlation structure also changes markedly with the volatility regime.

## 6. Feature selection recommendations for modeling

> These are EDA-based suggestions only; final choices must be validated with strict time-series cross-validation (walk-forward, with a gap between training and test sets).

1. **Remove redundancy**: keep one representative per highly collinear group, for example
   - Trend: `ma_dev_50` (or `macd`) + `ma_dev_200` (long-term) + `rsi_14`; drop `macd_signal`; choose one of `ret_20d` / `ma_dev_20`.
   - Volatility: `vol_20d` + `hl_range`, with `vol_60d` as an optional slow-moving variable.
   - Cross-asset: drop `r_QQQ` (collinear with `ret_1d`). The tech returns are mutually correlated and can be replaced by a single factor (e.g. the QQQ-minus-SPY relative return) or compressed with PCA. `r_XOM` and `r_JPM` carry sector-dispersion information and can be kept, but their relationship with SPY is unstable.
2. **Reconsider the prediction target**: direction (`y`) is very weakly predictable, and with a 55.3% up base rate a model is likely to learn only "always predict up". Volatility (`|next_ret|`, future realized volatility) is clearly more predictable and is worth adding as a parallel target, or using for position sizing and risk control.
3. **Feature transforms**: log-transform right-skewed features (`vol_*`, `hl_range`, `vol_ratio_20`); winsorize fat-tailed returns (with quantiles estimated on the training set only) or use rank transforms to limit the influence of extreme days on linear models.
4. **Regime awareness**: correlation structure and signals change with the volatility regime. Consider adding a regime variable (e.g. `vol_20d` relative to its historical quantile), or evaluating models by regime, and check the influence of 2020 separately (a sensitivity analysis excluding 2020 is recommended).
5. **Keep preventing leakage**: every standardization, winsorization, PCA and threshold must be fit only within the training window. Note that `ret_lag*` mechanically overlaps with `ret_5d`/`ret_20d`; do not mistake their shared information for an additional signal.

## 7. Limitations

- Only one index ETF and 9 large-cap stocks are analyzed. The universe consists of today's surviving large companies, so survivorship bias is present.
- Significance bands use a white-noise approximation; volatility clustering makes the true bands wider, so borderline-significant findings should be treated conservatively.
- Exogenous features use same-day closing returns, assuming they are available at the close of day t. If orders are placed before the close, they should be switched to t-1 data.
