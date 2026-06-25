# Portfolio Risk & Return Analyzer

## Overview

Portfolio Risk & Return Analyzer is a Python application that evaluates the historical performance and risk profile of an equity portfolio using publicly available market data. The project downloads historical prices from Yahoo Finance, computes commonly used financial metrics, benchmarks portfolio performance against the Nifty 50 index, and performs Monte Carlo simulation to analyze portfolio allocations.

The project was developed to apply concepts from portfolio theory and quantitative finance through data analysis, simulation, and visualization.

---

## Features

- Downloads historical price data for NSE-listed stocks using Yahoo Finance
- Calculates daily and annualized portfolio returns
- Computes annualized volatility
- Evaluates Sharpe Ratio and Sortino Ratio
- Calculates Maximum Drawdown
- Estimates portfolio Beta relative to the Nifty 50 benchmark
- Computes Value at Risk (95% confidence level)
- Performs Monte Carlo simulation to generate an Efficient Frontier
- Generates visualizations for portfolio performance and risk analysis
- Exports stock-level metrics as a CSV file

---

## Technologies Used

- Python
- Pandas
- NumPy
- SciPy
- Matplotlib
- Seaborn
- yfinance

---

## Project Structure

```text
Portfolio-risk-Analyzer/
│
├── portfolio_analyzer.py      # Main application
├── requirements.txt           # Project dependencies
├── README.md
└── output/
    ├── 1_cumulative_returns.png
    ├── 2_efficient_frontier.png
    ├── 3_correlation_heatmap.png
    ├── 4_risk_dashboard.png
    ├── 5_drawdown.png
    └── stock_metrics.csv
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/muditar12/Portfolio-risk-Analyzer.git
cd Portfolio-risk-Analyzer
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

macOS / Linux

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run the application with:

```bash
python portfolio_analyzer.py
```

The program will:

- Download historical stock and benchmark data
- Calculate portfolio performance and risk metrics
- Run Monte Carlo portfolio optimization
- Generate portfolio visualizations
- Save all charts and reports in the `output` directory

---

## Output

The application generates the following outputs:

- Cumulative returns comparison against the Nifty 50
- Efficient Frontier from Monte Carlo simulation
- Correlation heatmap of portfolio assets
- Portfolio risk dashboard
- Drawdown comparison
- CSV file containing stock-level performance metrics

---

## Portfolio Metrics

The analyzer computes the following metrics:

- Annualized Return
- Annualized Volatility
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Beta
- Value at Risk (95%)
- Efficient Frontier

---

## Customization

The portfolio can be customized by modifying the stock ticker dictionary in `portfolio_analyzer.py`.

Other configurable parameters include:

- Analysis period
- Benchmark index
- Risk-free rate
- Number of Monte Carlo simulations

---

## Future Improvements

Possible extensions include:

- Interactive dashboard using Streamlit
- Portfolio optimization with user-defined constraints
- Additional risk measures such as Conditional Value at Risk (CVaR)
- Portfolio rebalancing recommendations
- Support for international markets
- Real-time market data integration

---
