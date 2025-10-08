# ML-on-CAPM
This repository now includes a standalone analysis script, `capm_extreme_analysis.py`,
that:

* downloads NVIDIA and S&P 500 price histories with `yfinance`
* computes excess returns relative to a constant risk-free rate
* fits a baseline CAPM regression and flags extreme residual events
* augments those extremes with synthetic, decaying follow-up observations
* trains OLS, Random Forest, SVR, and Gradient Boosting models on both the
  synthetic-augmented and non-extreme subsets
* reports extended regression diagnostics and produces visualisations for the
  extreme region as well as classical OLS diagnostic plots

To run the analysis (after installing the dependencies listed at the top of the
script), execute:

```bash
python capm_extreme_analysis.py
```
