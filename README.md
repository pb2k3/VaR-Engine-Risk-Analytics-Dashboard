# 📊 VaR Engine — Equity Risk Analytics Dashboard

A Python + Streamlit dashboard that estimates equity tail risk using Monte Carlo simulation, comparing **Standard Deviation** vs **Parkinson's Volatility** as estimators. Type any ticker symbol — get instant Value at Risk, Expected Shortfall, and full distribution analysis.

Built as an FRM (Financial Risk Manager) portfolio project.

---

## 🎯 What It Does

- 📈 Fetches real historical OHLC (Open-High-Low-Close) data from Yahoo Finance for any global stock
- 📐 Computes log returns, mean drift, and **two volatility estimators**: close-to-close Standard Deviation and Parkinson's (1980) high-low range estimator
- 🎲 Runs **Monte Carlo simulation** with up to 25,000 scenarios per method
- 📉 Calculates **Value at Risk (VaR)** and **Expected Shortfall (ES)** at 90%, 95%, and 99% confidence levels
- 🎨 Visualizes price history, simulated return distributions, and theoretical normal curves
- 🧠 Auto-generates **interpretation text** based on whether intraday or overnight risk dominates for the loaded stock

## 💡 The Project's Core Finding

For **Tesla (Jan 2024 – May 2026)**, Parkinson's volatility came in **24% lower** than close-to-close StdDev — the opposite of the textbook expectation. The dashboard reveals this is because Tesla's risk is dominated by overnight gaps (earnings reactions, post-market news) that Parkinson's estimator ignores by design. This is a known limitation in the academic literature (addressed by later estimators like Yang-Zhang 2000).

Different stocks produce different results — try Apple, Reliance, NVIDIA, or BTC-USD to see how methodology choice changes risk numbers.

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.9 or higher** ([download from python.org](https://www.python.org/downloads/))
- Mac, Windows, or Linux

### Installation (One-Time, ~5 minutes)

**Option A — Clone with Git (recommended):**
```bash
git clone https://github.com/YOUR_USERNAME/var-engine.git
cd var-engine
pip3 install -r requirements.txt
```

**Option B — Download as ZIP (no Git needed):**
1. Click the green **"Code"** button at the top of this repo
2. Click **"Download ZIP"**
3. Unzip the folder somewhere convenient (Desktop is fine)
4. Open Terminal, navigate to that folder:
   ```bash
   cd ~/Desktop/var-engine-main
   pip3 install -r requirements.txt
   ```

### Running the Dashboard

```bash
streamlit run var_engine.py
```

Your browser opens automatically to `http://localhost:8501`. Done.

**To stop the dashboard:** press `Ctrl + C` in the terminal (yes, Ctrl, even on Mac).

**To run it again later:** open Terminal and use the same two commands:
```bash
cd ~/Desktop/var-engine-main
streamlit run var_engine.py
```

---

## 📝 How to Use

### Sidebar Inputs

| Input | Description |
|---|---|
| **Ticker** | Stock symbol — e.g., `TSLA`, `AAPL`, `NVDA`, `RELIANCE.NS`, `BTC-USD` |
| **Quick Picks** | One-click buttons for Tesla, Apple, NVIDIA, Reliance |
| **Date Range** | Start/end dates for historical data (default: last 2 years) |
| **Investment Amount** | Position size in USD ($1,000 – $10M) — drives all dollar metrics |
| **Confidence Level** | 90%, 95%, or 99% — for VaR and ES calculation |
| **Simulations** | 1,000 to 25,000 Monte Carlo scenarios |
| **Re-run Simulation** | Generate fresh random scenarios |

### Ticker Format Guide

| Market | Format | Example |
|---|---|---|
| US stocks (NYSE/NASDAQ) | Plain ticker | `TSLA`, `AAPL`, `MSFT` |
| Indian stocks (NSE) | Append `.NS` | `RELIANCE.NS`, `ICICIBANK.NS`, `HDFCBANK.NS` |
| Indian stocks (BSE) | Append `.BO` | `RELIANCE.BO` |
| Crypto | Append `-USD` | `BTC-USD`, `ETH-USD` |
| ETFs | Plain ticker | `SPY`, `QQQ`, `VOO` |
| UK stocks | Append `.L` | `BARC.L`, `HSBA.L` |

### Dashboard Sections

1. **Key Metrics** — KPI tiles showing latest close, position size, both volatility estimators, VaR, ES
2. **Price History** — interactive chart with closing price + intraday range band
3. **Volatility Comparison** — side-by-side cards for Std Dev vs Parkinson
4. **Monte Carlo Distribution** — histograms with theoretical normal overlay and VaR/ES lines
5. **Risk Metrics Table** — all confidence levels for both methods
6. **Auto-Interpretation** — rules-based text explaining the result for your specific stock
7. **Methodology** — formulas, assumptions, limitations
8. **Raw Data Explorer** — full historical dataset view

---

## 📚 Methodology

### Log Returns
$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

Used instead of simple returns because they're additive over time and approximately normally distributed.

### Standard Deviation (Close-to-Close)
Sample standard deviation of daily log returns. Captures overnight gaps; misses intraday swings.

### Parkinson's Volatility (1980)
$$\sigma_P = \sqrt{\frac{1}{4 \ln 2} \cdot \frac{1}{n}\sum_{i=1}^{n} \left[\ln\left(\frac{H_i}{L_i}\right)\right]^2}$$

Uses only daily High-Low range. ~5× more statistically efficient than StdDev under continuous Brownian motion. Major limitation: ignores overnight gaps and drift.

### Monte Carlo Simulation
Generates N returns from N(μ, σ²) using the inverse-CDF method:
$$r_{sim} = \mu + \sigma \cdot \Phi^{-1}(U), \quad U \sim \text{Uniform}(0,1)$$

Simulated prices follow geometric Brownian motion: $P_{sim} = P_0 \cdot e^{r_{sim}}$.

### Value at Risk (VaR)
$$\text{VaR}_\alpha = \text{percentile}(\text{PnL},\ 1-\alpha)$$

A 95% VaR is the 5th percentile worst outcome — the loss exceeded only 5% of the time.

### Expected Shortfall (ES / CVaR)
$$\text{ES}_\alpha = \mathbb{E}[\text{PnL} \mid \text{PnL} \le \text{VaR}_\alpha]$$

The average loss conditional on losing more than VaR — captures tail severity, not just frequency.

---

## ⚠️ Limitations

This is an educational tool. Real-world risk modeling would address:

- **Normal distribution assumption** underestimates fat tails (real returns show kurtosis > 3)
- **Static volatility** ignores volatility clustering (GARCH effects)
- **Single-asset only** — no portfolio correlation modeling
- **Parkinson ignores overnight gaps** — addressed by Garman-Klass (1980), Rogers-Satchell (1991), and Yang-Zhang (2000) estimators

## 🔭 Possible Extensions

- Student's t-distribution for fatter-tailed returns
- GARCH(1,1) model for time-varying volatility
- Multi-asset portfolio with Cholesky decomposition for correlated returns
- VaR backtesting against actual realized losses (Kupiec test, Christoffersen test)
- Additional volatility estimators (Yang-Zhang, Garman-Klass, Rogers-Satchell)

---

## 🔧 Tech Stack

- **[Streamlit](https://streamlit.io/)** — web app framework
- **[yfinance](https://github.com/ranaroussi/yfinance)** — Yahoo Finance data fetcher
- **[Plotly](https://plotly.com/python/)** — interactive charts
- **NumPy / SciPy** — statistical computation
- **pandas** — data manipulation

## 🐛 Troubleshooting

**`pip3: command not found`** — Try `python3 -m pip install -r requirements.txt`

**`streamlit: command not found`** — Try `python3 -m streamlit run var_engine.py`

**`externally-managed-environment` error on Mac** — Either add `--break-system-packages` to the pip command, or use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Yahoo Finance returns no data** — Yahoo occasionally rate-limits. Wait 60 seconds and try again, or use a different ticker.

**Indian stocks not loading** — Make sure to add the exchange suffix: `RELIANCE.NS` for NSE, `RELIANCE.BO` for BSE.

---

## ⚠️ Data Usage Disclaimer

This dashboard uses the `yfinance` library to fetch historical market data from Yahoo Finance for **educational and research purposes only**. The data is displayed for in-session analysis and is not stored, redistributed, or made available for download. Users should comply with Yahoo Finance's terms of service. For commercial or production use, consider a licensed data provider such as Alpha Vantage, Polygon.io, or a paid Yahoo Finance subscription.

---

## 📜 License

MIT License — free to use, modify, and share.

## 🙏 Acknowledgments

Built on the foundations of financial risk management literature:
- Parkinson, M. (1980). *The Extreme Value Method for Estimating the Variance of the Rate of Return*
- Garman, M. B., & Klass, M. J. (1980). *On the Estimation of Security Price Volatilities from Historical Data*
- Rogers, L. C. G., & Satchell, S. E. (1991). *Estimating Variance From High, Low and Closing Prices*
- Yang, D., & Zhang, Q. (2000). *Drift-Independent Volatility Estimation Based on High, Low, Open, and Close Prices*

---

*This dashboard is for educational purposes only and does not constitute financial advice.*
