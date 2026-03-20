# Full-Depth Portfolio Optimisation Dashboard

An interactive, end-to-end portfolio construction and risk analytics tool built in Python for JupyterLab. Combines four expected return models, a four-engine Monte Carlo simulation framework, a complete drawdown analytics suite, and an interactive Efficient Frontier Picker — all within a single-cell dashboard driven by a fully interactive UI.

Built as part of MSc Financial Technology research at Bristol Business School, UWE (2024–2025).

---

## Overview

Portfolio optimisation in practice requires more than textbook mean-variance analysis. This tool was built to address three limitations common in academic implementations:

1. **Single-model dependency** — relying on historical means alone produces unstable, unreliable expected return estimates. This tool blends four distinct models (MPT, CAPM, APT, ML) to produce more robust inputs.
2. **Point-estimate thinking** — deterministic portfolio projections ignore the full distribution of outcomes. This tool uses Monte Carlo simulation with four distinct engines to assess outcome distributions, not just expected values.
3. **Incomplete risk analysis** — Sharpe ratio alone is an insufficient risk summary. This tool implements a full drawdown analytics suite covering stress testing, density heatmaps, breach probabilities, time-under-water distributions, and max drawdown histograms.

---

## Key Features

### Expected Return Models
- **MPT** — historical mean returns with LedoitWolf shrinkage covariance estimation (reduces estimation error in high-dimensional portfolios)
- **CAPM** — Ridge regression beta estimation against a user-selected benchmark; alpha-adjusted expected returns
- **APT** — PCA-based statistical factor model; user-configurable number of factors with optional benchmark inclusion
- **ML Ensemble** — Ridge regression with TimeSeriesSplit cross-validation; predicts next-month returns from lagged features
- **Blend** — Weighted ensemble of all four models with user-configurable ML blend weight

### Portfolio Optimisation
- Sharpe ratio maximisation or Mean-Variance Utility optimisation
- Long-only or long/short with gross exposure constraint
- Per-asset weight caps with optional Sharpe-weighted dynamic bounds
- Beta-neutral penalty (minimises portfolio market beta)
- Volatility targeting (scales risky/risk-free split to hit a target portfolio volatility)
- Subset selection: exhaustive search for small universes; random search with configurable budget for larger ones

### Monte Carlo Simulation
Four simulation engines, all with contribution schedules, rebalancing, and dividend handling:
- **Normal** — multivariate Gaussian returns
- **Student-t** — fat-tailed returns with configurable degrees of freedom
- **Regime-switching Student-t** — two-state Markov chain (low/high volatility regimes) with configurable transition probabilities and volatility multiplier
- **Historical Bootstrap** — resamples actual monthly returns; fully non-parametric

### Drawdown Analytics Suite
- **Stress test chart** — worst X% average/median/single path vs benchmarks, with P10–P90 band
- **Density heatmap** — distribution of drawdown depths across time for the stressed subset
- **Breach probability curves** — probability of breaching configurable drawdown thresholds over time
- **Time-under-water distribution** — histogram of under-water episode lengths across stressed paths
- **Max drawdown histogram** — distribution of maximum drawdowns across all runs vs stressed subset

### Interactive Efficient Frontier Picker
- Click any point on the efficient frontier to open a Frontier Picker submenu
- Choose any of the four return models at the clicked target return
- Preview: full Summary, Portfolio Set-up, full Monte Carlo, and full Drawdown suite for that point
- Apply: replace the active portfolio with the selected frontier point and rebuild the dashboard

### Additional Analytics
- Rolling correlation (portfolio vs benchmark, configurable window)
- Correlation heatmap (post-cleaning daily returns)
- Full diagnostics: per-model portfolio summaries with Sortino ratios, CAPM beta/alpha per asset, APT factor exposures and premia, ML R² scores, covariance and correlation matrices
- Benchmark overlays: S&P 500, FTSE 100, Nasdaq throughout all charts
- Dividend handling: reinvest (compounds back into portfolio) or income mode (tracks cash income separately)
- Full CSV export of all outputs

### Data Pipeline
- Automatic data download via yfinance with dividend adjustment
- Robust cleaning pipeline with auto-tune and fallback ladder (progressively relaxed thresholds)
- Business day alignment, forward-fill with configurable limit, minimum observation and coverage filters
- Handles mixed universes (US equities, UK equities, mixed exchanges)

---

## Dashboard Structure

The UI is built entirely in ipywidgets within JupyterLab. Configuration is via an 8-section Accordion:

| Section | Controls |
|---|---|
| 1. Data | Tickers, start date, risk-free rate, inflation, benchmarks |
| 2. Cleaning | Auto-tune toggle, business day alignment, ffill limit, coverage thresholds |
| 3. Portfolio | Initial capital, contributions, frequency, years, risk aversion, strategy, model |
| 4. Constraints | Long/short, weight caps, gross exposure, beta-neutral penalty |
| 5. Selection | Subset selection on/off, subset size, exhaustive limit, random budget |
| 6. Risk Controls | Vol targeting, Sharpe-weighted caps |
| 7. Analytics | Monte Carlo settings, chart toggles |
| 8. Drawdowns | Method, stress mode, density heatmap, breach levels, TUW, max-DD histogram |

Results are displayed across 11 tabs:

`Summary` | `Portfolio Set-up` | `Weights & Holdings` | `Growth` | `Monte Carlo` | `Annual Growth` | `Drawdowns` | `Efficient Frontier` | `Correlation Heatmap` | `Rolling Corr` | `Diagnostics`

---

## Example Output

### Portfolio Summary (example run)
| Metric | Value |
|---|---|
| Model | Blend (MPT + CAPM + APT + ML) |
| Annualised Return (risky sleeve) | 27.32% |
| Volatility (risky sleeve) | 15.87% |
| Sharpe Ratio | 1.46 |
| Risky Weight | 94.5% |
| Risk-Free Weight | 5.5% |
| Monte Carlo runs | 300 |
| Simulation engine | Student-t (fat tails, df=6) |

*(Screenshots below)*

---

## Screenshots

### Full Dashboard — Configuration UI

![Overall Dashboard View](screenshots/Overall%20Dashboard%20View.png)

![Full Dashboard with Results](screenshots/Overall%20View%20of%20Portfolio%20Analysis%20Dashboard%20.png)

---

### Configuration Sections

**Section 1 — Data:** Ticker input, start date, risk-free rate, inflation, benchmark selection
![Data Configuration](screenshots/Dashboard%20Display%201.png)

**Section 2 — Cleaning:** Auto-tune, business day alignment, ffill limit, coverage thresholds
![Cleaning Configuration](screenshots/Dashboard%20Display%202.png)

**Section 3 — Portfolio:** Capital, contributions, years, risk aversion, strategy, model blend
![Portfolio Configuration](screenshots/Dashboard%20Display%203.png)

**Section 4 — Constraints:** Long/short, weight caps, gross exposure, beta-neutral penalty
![Constraints Configuration](screenshots/Dashboard%20Display%204.png)

**Section 5 — Selection:** Subset size, exhaustive limit, random search budget
![Selection Configuration](screenshots/Dashboard%20Display%205.png)

**Section 6 — Risk Controls:** Volatility targeting, Sharpe-weighted caps
![Risk Controls Configuration](screenshots/Dashboard%20Display%206.png)

**Section 7 — Analytics:** Monte Carlo settings, simulation engine, regime parameters
![Analytics Configuration](screenshots/Dashboard%20Display%207.png)

**Section 8 — Drawdowns:** Stress mode, density heatmap, breach thresholds, time-under-water
![Drawdowns Configuration](screenshots/Dashboard%20Display%208.png)

---

### Portfolio Summary & Results
Full summary output: Blend model (MPT + CAPM + APT + ML), 30 tickers, random subset selection. Annualised return 40.62%, Sharpe 1.84, Sortino 1.64, final portfolio value £401,524 on £121,537 total invested — profit £279,988.

![Portfolio Performance Summary](screenshots/Portfolio%20performance%20and%20summary%20.png)

---

### Portfolio Set-up & Weights
Target allocation table with per-asset weights and initial investment amounts.

![Portfolio Set-up](screenshots/Portfolio%20Set%20Up.png)

Per-asset weights, volatility contributions, risk contribution percentages, and CAPM betas alongside full holdings table.

![Weights & Holdings](screenshots/Stock%20Risk%20contributions%20.png)

---

### Portfolio Growth
Portfolio growth with Monte Carlo P10–P90 band versus S&P 500, FTSE 100, and Nasdaq benchmarks.

![Portfolio Growth with Benchmarks](screenshots/Stock%20performance%20analysis.png)

Year-by-year growth percentage across the simulation horizon.

![Year-by-Year Growth](screenshots/Yearly%20Performance%20.png)

---

### Monte Carlo Simulation
Distribution of final portfolio values across 3,000 simulations (Student-t fat-tail engine) alongside downsampled spaghetti paths.

![Monte Carlo Distribution and Spaghetti](screenshots/Monte%20Carlo%20simulations%20.png)

---

### Drawdown Analytics Suite

**Drawdown Stress Test** — Portfolio stress line versus S&P 500, FTSE 100, and Nasdaq with P10–P90 worst-10% band. Portfolio max drawdown -8.03% versus S&P 500 -24.77% and Nasdaq -33.10%.

![Drawdown Stress Test](screenshots/Drawdowns%20with%20Benchmark%20comparisons%20.png)

**Breach Probability Curves** — Probability of breaching -20% and -30% drawdown thresholds over time across the worst 10% of simulation runs.

![Breach Probabilities](screenshots/Breach%20probabilities%20.png)

**Time Under Water Distribution** — Histogram of under-water episode lengths. The majority of episodes resolve within 5 months.

![Time Under Water](screenshots/Tim%20Under%20Water%20analysis%20.png)

**Max Drawdown Distribution** — Worst-10% subset versus all simulation runs.

![Max Drawdown Distribution](screenshots/Max%20drawdown%20Distribution%20diagram.png)

---

### Correlation Heatmap
Post-cleaning daily return correlations across the selected asset universe.

![Correlation Heatmap](screenshots/Correlation%20Heatmap.png)

---

### Interactive Efficient Frontier
Clicking any point opens the Frontier Picker — choose model, preview full Monte Carlo and drawdown suite, then apply to replace the active portfolio.

![Efficient Frontier](screenshots/Interactive%20Efficient%20Frontier%20Graph%20Image%201.png)

![Frontier Hover](screenshots/Interactive%20Efficient%20Frontier%20Graph%20Image%203.png)

![Frontier Selected Point](screenshots/Interactive%20Efficient%20Frontier%20Graph%20Image%202.png)
```

---

Commit message:
```
Fix all screenshot paths with correct filenames and %20 encoding
---

## Technical Stack

| Component | Library |
|---|---|
| Data download | yfinance |
| Numerical computation | NumPy, SciPy |
| Data manipulation | Pandas |
| Covariance estimation | scikit-learn (LedoitWolf) |
| ML models | scikit-learn (Ridge, PCA, TimeSeriesSplit) |
| Visualisation | Plotly |
| Interactive UI | ipywidgets |
| Optimisation | SciPy (SLSQP) |

---

## Installation & Usage

### Requirements
```
numpy
pandas
yfinance
plotly
ipywidgets
scipy
scikit-learn
```

Install all dependencies:
```bash
pip install numpy pandas yfinance plotly ipywidgets scipy scikit-learn
```

### Running the Dashboard
1. Clone or download this repository
2. Open `portfolio_dashboard.ipynb` in JupyterLab
3. Run the single cell (Shift+Enter)
4. Configure inputs using the Accordion UI
5. Click **"Build Portfolio"**

> **Note:** JupyterLab is required (not classic Jupyter Notebook). The interactive widgets and Plotly FigureWidget require JupyterLab's rendering environment.

---

## Methodology Notes

### Why LedoitWolf Shrinkage?
Sample covariance matrices are notoriously unstable for portfolios with many assets — small sample sizes relative to the number of parameters lead to extreme, unreliable estimates. LedoitWolf shrinkage pulls the covariance matrix toward a structured target, significantly improving out-of-sample stability. This is standard practice in institutional portfolio construction.

### Why Four Monte Carlo Engines?
Different market environments require different distributional assumptions. The Normal engine provides a baseline; Student-t captures the fat tails observed in real asset returns; the Regime-switching engine models the distinct low-volatility and crisis regimes visible in historical data; Historical Bootstrap makes no parametric assumptions at all. Comparing results across engines provides a more robust view of outcome uncertainty.

### Why Subset Selection?
Optimising over a large universe of assets often produces concentrated, fragile portfolios. Searching for the optimal subset of a given size forces diversification across the selection step itself, and often produces more robust out-of-sample allocations than full-universe optimisation.

### Limitations
- **Data source:** yfinance provides adjusted prices suitable for research. Licensed data sources (Bloomberg, Refinitiv) are required for production or commercial use.
- **Look-ahead bias:** All models are trained on the full historical window. Walk-forward validation is not currently implemented.
- **Transaction costs:** No transaction cost modelling is included. Real-world implementation costs would reduce net returns.
- **Parameter sensitivity:** Results are sensitive to the choice of start date, risk-free rate, and model blend weights. The scenario analysis and Monte Carlo suite are designed to surface this sensitivity rather than obscure it.

---

## Academic Context

This tool was developed as part of MSc Financial Technology research at Bristol Business School, University of the West of England (2024–2025). It informed the quantitative methodology of the dissertation *"Index Fund Allocation Optimisation"*, which applies Modern Portfolio Theory to FTSE 100 index constituents and evaluates the practical implementability of efficient frontier allocations.

---

## Disclaimer

This tool is an academic research project. All outputs are for educational and illustrative purposes only. Nothing in this repository constitutes financial advice. Past performance of any portfolio shown does not guarantee future results.

---

*Thomas Oxley | MSc Financial Technology | Bristol Business School, UWE*
*linkedin.com/in/thomas-oxley-868047174*
