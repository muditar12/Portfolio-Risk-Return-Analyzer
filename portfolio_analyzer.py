"""
Portfolio Risk & Return Analyzer
=================================
A Python tool to analyze NSE equity portfolio performance,
compute risk metrics, run Monte Carlo simulation, and benchmark
against Nifty 50.

Author: Your Name
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
from scipy.stats import norm
import os

# ─── Configuration ────────────────────────────────────────────────────────────

TICKERS = {
    "Reliance":   "RELIANCE.NS",
    "TCS":        "TCS.NS",
    "Infosys":    "INFY.NS",
    "HDFC Bank":  "HDFCBANK.NS",
    "Wipro":      "WIPRO.NS",
}

BENCHMARK     = "^NSEI"          # Nifty 50
START_DATE    = "2022-01-01"
END_DATE      = "2024-12-31"
RISK_FREE     = 0.065            # ~6.5% Indian T-bill rate (annualized)
N_SIMULATIONS = 10_000
TRADING_DAYS  = 252
OUTPUT_DIR    = "output"

# ─── Data Fetching ─────────────────────────────────────────────────────────────

def fetch_data(tickers: dict, benchmark: str, start: str, end: str) -> pd.DataFrame:
    all_tickers = list(tickers.values()) + [benchmark]
    print(f"[1/5] Downloading price data for {len(all_tickers)} instruments...")
    raw = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"]
    prices.columns = [col for col in prices.columns]
    rename_map = {v: k for k, v in tickers.items()}
    rename_map[benchmark] = "Nifty50"
    prices = prices.rename(columns=rename_map)
    prices = prices.dropna()
    print(f"    Loaded {len(prices)} trading days of data.")
    return prices


# ─── Return Calculations ───────────────────────────────────────────────────────

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def annualized_return(daily_returns: pd.Series) -> float:
    return (1 + daily_returns.mean()) ** TRADING_DAYS - 1


def annualized_volatility(daily_returns: pd.Series) -> float:
    return daily_returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(ret: float, vol: float, rf: float = RISK_FREE) -> float:
    return (ret - rf) / vol if vol != 0 else 0


def sortino_ratio(daily_returns: pd.Series, rf: float = RISK_FREE) -> float:
    ann_ret = annualized_return(daily_returns)
    downside = daily_returns[daily_returns < 0].std() * np.sqrt(TRADING_DAYS)
    return (ann_ret - rf) / downside if downside != 0 else 0


def max_drawdown(daily_returns: pd.Series) -> float:
    cum = (1 + daily_returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()


def beta(asset_returns: pd.Series, bench_returns: pd.Series) -> float:
    cov = np.cov(asset_returns, bench_returns)[0, 1]
    var = np.var(bench_returns)
    return cov / var if var != 0 else 0


def value_at_risk(daily_returns: pd.Series, confidence: float = 0.95) -> float:
    return np.percentile(daily_returns, (1 - confidence) * 100)


# ─── Individual Stock Metrics ─────────────────────────────────────────────────

def stock_metrics_table(returns: pd.DataFrame, benchmark_col: str = "Nifty50") -> pd.DataFrame:
    print("[2/5] Computing individual stock metrics...")
    stocks = [c for c in returns.columns if c != benchmark_col]
    rows = []
    for s in stocks:
        r = returns[s]
        b = returns[benchmark_col]
        ann_r = annualized_return(r)
        ann_v = annualized_volatility(r)
        rows.append({
            "Stock":            s,
            "Ann. Return (%)":  round(ann_r * 100, 2),
            "Volatility (%)":   round(ann_v * 100, 2),
            "Sharpe Ratio":     round(sharpe_ratio(ann_r, ann_v), 3),
            "Sortino Ratio":    round(sortino_ratio(r), 3),
            "Max Drawdown (%)": round(max_drawdown(r) * 100, 2),
            "Beta":             round(beta(r, b), 3),
            "VaR 95% (%)":      round(value_at_risk(r) * 100, 2),
        })
    return pd.DataFrame(rows).set_index("Stock")


# ─── Portfolio Weights ────────────────────────────────────────────────────────

def equal_weights(n: int) -> np.ndarray:
    return np.ones(n) / n


def portfolio_performance(weights: np.ndarray, returns: pd.DataFrame,
                           benchmark_col: str = "Nifty50") -> dict:
    stocks = [c for c in returns.columns if c != benchmark_col]
    r = returns[stocks]
    port_daily = r.dot(weights)
    ann_r = annualized_return(port_daily)
    ann_v = annualized_volatility(port_daily)
    b = returns[benchmark_col]
    return {
        "ann_return":   ann_r,
        "ann_vol":      ann_v,
        "sharpe":       sharpe_ratio(ann_r, ann_v),
        "sortino":      sortino_ratio(port_daily),
        "max_dd":       max_drawdown(port_daily),
        "beta":         beta(port_daily, b),
        "var_95":       value_at_risk(port_daily),
        "daily_returns": port_daily,
    }


# ─── Monte Carlo Simulation ───────────────────────────────────────────────────

def monte_carlo(returns: pd.DataFrame, benchmark_col: str = "Nifty50",
                n: int = N_SIMULATIONS) -> pd.DataFrame:
    print(f"[3/5] Running {n:,} Monte Carlo simulations...")
    stocks = [c for c in returns.columns if c != benchmark_col]
    n_stocks = len(stocks)
    r = returns[stocks]
    cov_matrix = r.cov() * TRADING_DAYS
    mean_returns = r.mean() * TRADING_DAYS

    results = []
    for _ in range(n):
        w = np.random.dirichlet(np.ones(n_stocks))
        ret = np.dot(w, mean_returns)
        vol = np.sqrt(w @ cov_matrix @ w)
        sh  = sharpe_ratio(ret, vol)
        results.append([ret, vol, sh, *w])

    cols = ["Return", "Volatility", "Sharpe"] + stocks
    df = pd.DataFrame(results, columns=cols)
    return df


# ─── Plotting ─────────────────────────────────────────────────────────────────

def set_style():
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "#F9F9F9",
        "axes.edgecolor":    "#DDDDDD",
        "axes.grid":         True,
        "grid.color":        "#EEEEEE",
        "grid.linestyle":    "-",
        "font.family":       "DejaVu Sans",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def plot_cumulative_returns(prices: pd.DataFrame, outdir: str):
    fig, ax = plt.subplots(figsize=(12, 6))
    norm_prices = prices / prices.iloc[0] * 100

    colors = ["#4F8EF7", "#F7934C", "#6BCB77", "#FF6B6B", "#A78BFA", "#374151"]
    for i, col in enumerate(norm_prices.columns):
        lw = 2.5 if col == "Nifty50" else 1.5
        ls = "--" if col == "Nifty50" else "-"
        ax.plot(norm_prices.index, norm_prices[col],
                label=col, color=colors[i % len(colors)], lw=lw, ls=ls)

    ax.axhline(100, color="#BBBBBB", lw=1, ls=":")
    ax.set_title("Cumulative Returns vs Nifty 50 (Base = 100)", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Indexed Price (Base 100)")
    ax.set_xlabel("")
    ax.legend(loc="upper left", framealpha=0.8, fontsize=9)
    plt.tight_layout()
    path = os.path.join(outdir, "1_cumulative_returns.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def plot_efficient_frontier(mc_df: pd.DataFrame, portfolio_perf: dict, outdir: str):
    fig, ax = plt.subplots(figsize=(10, 7))

    sc = ax.scatter(
        mc_df["Volatility"] * 100,
        mc_df["Return"] * 100,
        c=mc_df["Sharpe"],
        cmap="RdYlGn", alpha=0.4, s=6, linewidths=0
    )
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")

    # Max Sharpe portfolio
    best = mc_df.loc[mc_df["Sharpe"].idxmax()]
    ax.scatter(best["Volatility"] * 100, best["Return"] * 100,
               color="#F59E0B", s=180, zorder=5, marker="*", label="Max Sharpe Portfolio")

    # Equal-weight portfolio
    ax.scatter(portfolio_perf["ann_vol"] * 100, portfolio_perf["ann_return"] * 100,
               color="#3B82F6", s=100, zorder=5, marker="D", label="Equal-Weight Portfolio")

    ax.set_title("Efficient Frontier (Monte Carlo)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Annualized Volatility (%)")
    ax.set_ylabel("Annualized Return (%)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(outdir, "2_efficient_frontier.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def plot_correlation_heatmap(returns: pd.DataFrame, outdir: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = returns.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=-1, vmax=1, ax=ax, linewidths=0.5,
                annot_kws={"size": 9})
    ax.set_title("Return Correlation Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    path = os.path.join(outdir, "3_correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def plot_metrics_dashboard(metrics_df: pd.DataFrame, port_perf: dict,
                            bench_returns: pd.Series, outdir: str):
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # --- Sharpe Ratio bar ---
    ax1 = fig.add_subplot(gs[0, 0])
    colors = ["#4F8EF7" if v >= 0 else "#EF4444" for v in metrics_df["Sharpe Ratio"]]
    bars = ax1.bar(metrics_df.index, metrics_df["Sharpe Ratio"], color=colors, width=0.5)
    ax1.axhline(0, color="#555", lw=0.8)
    ax1.set_title("Sharpe Ratio by Stock", fontweight="bold")
    ax1.set_ylabel("Sharpe Ratio")
    ax1.tick_params(axis="x", rotation=30)
    for bar, v in zip(bars, metrics_df["Sharpe Ratio"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    # --- Volatility bar ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(metrics_df.index, metrics_df["Volatility (%)"], color="#A78BFA", width=0.5)
    ax2.set_title("Annualized Volatility (%)", fontweight="bold")
    ax2.set_ylabel("Volatility (%)")
    ax2.tick_params(axis="x", rotation=30)

    # --- Max Drawdown bar ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(metrics_df.index, metrics_df["Max Drawdown (%)"], color="#F87171", width=0.5)
    ax3.set_title("Maximum Drawdown (%)", fontweight="bold")
    ax3.set_ylabel("Drawdown (%)")
    ax3.tick_params(axis="x", rotation=30)

    # --- Beta vs Nifty ---
    ax4 = fig.add_subplot(gs[1, 1])
    beta_colors = ["#F59E0B" if v > 1 else "#6BCB77" for v in metrics_df["Beta"]]
    ax4.bar(metrics_df.index, metrics_df["Beta"], color=beta_colors, width=0.5)
    ax4.axhline(1, color="#555", lw=0.8, ls="--", label="Beta = 1 (Market)")
    ax4.set_title("Beta vs Nifty 50", fontweight="bold")
    ax4.set_ylabel("Beta")
    ax4.tick_params(axis="x", rotation=30)
    ax4.legend(fontsize=8)

    fig.suptitle("Portfolio Risk Dashboard", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(outdir, "4_risk_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def plot_drawdown(returns: pd.DataFrame, outdir: str):
    benchmark_col = "Nifty50"
    stocks = [c for c in returns.columns if c != benchmark_col]
    weights = equal_weights(len(stocks))
    port_daily = returns[stocks].dot(weights)

    cum = (1 + port_daily).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak * 100

    cum_b = (1 + returns[benchmark_col]).cumprod()
    peak_b = cum_b.cummax()
    dd_b = (cum_b - peak_b) / peak_b * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(dd.index, dd, 0, alpha=0.4, color="#EF4444", label="Portfolio Drawdown")
    ax.fill_between(dd_b.index, dd_b, 0, alpha=0.3, color="#6B7280", label="Nifty 50 Drawdown")
    ax.set_title("Drawdown Over Time", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Drawdown (%)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(outdir, "5_drawdown.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


# ─── Report ───────────────────────────────────────────────────────────────────

def print_report(metrics_df: pd.DataFrame, port_perf: dict,
                 mc_df: pd.DataFrame, weights: np.ndarray,
                 stocks: list):
    best = mc_df.loc[mc_df["Sharpe"].idxmax()]
    print("\n" + "=" * 60)
    print("  PORTFOLIO RISK & RETURN REPORT")
    print("=" * 60)

    print("\n📊 INDIVIDUAL STOCK METRICS")
    print(metrics_df.to_string())

    print("\n\n⚖️  EQUAL-WEIGHT PORTFOLIO SUMMARY")
    print(f"  Stocks         : {', '.join(stocks)}")
    print(f"  Weights        : {[round(w, 3) for w in weights]}")
    print(f"  Ann. Return    : {port_perf['ann_return']*100:.2f}%")
    print(f"  Volatility     : {port_perf['ann_vol']*100:.2f}%")
    print(f"  Sharpe Ratio   : {port_perf['sharpe']:.3f}")
    print(f"  Sortino Ratio  : {port_perf['sortino']:.3f}")
    print(f"  Max Drawdown   : {port_perf['max_dd']*100:.2f}%")
    print(f"  Beta (Nifty50) : {port_perf['beta']:.3f}")
    print(f"  VaR 95%/day    : {port_perf['var_95']*100:.2f}%")

    print("\n\n🎯 OPTIMAL PORTFOLIO (Max Sharpe from Monte Carlo)")
    stock_cols = [c for c in mc_df.columns if c not in ["Return", "Volatility", "Sharpe"]]
    print(f"  Ann. Return    : {best['Return']*100:.2f}%")
    print(f"  Volatility     : {best['Volatility']*100:.2f}%")
    print(f"  Sharpe Ratio   : {best['Sharpe']:.3f}")
    print("  Weights:")
    for s in stock_cols:
        print(f"    {s:<12} : {best[s]*100:.1f}%")
    print("\n" + "=" * 60)


def save_csv_report(metrics_df: pd.DataFrame, outdir: str):
    path = os.path.join(outdir, "stock_metrics.csv")
    metrics_df.to_csv(path)
    print(f"    Saved: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_style()

    # 1. Fetch data
    prices = fetch_data(TICKERS, BENCHMARK, START_DATE, END_DATE)
    returns = compute_returns(prices)

    # 2. Stock metrics
    metrics_df = stock_metrics_table(returns)
    save_csv_report(metrics_df, OUTPUT_DIR)

    # 3. Equal-weight portfolio
    stocks = list(TICKERS.keys())
    weights = equal_weights(len(stocks))
    port_perf = portfolio_performance(weights, returns)

    # 4. Monte Carlo
    mc_df = monte_carlo(returns)

    # 5. Plots
    print("[4/5] Generating charts...")
    plot_cumulative_returns(prices, OUTPUT_DIR)
    plot_efficient_frontier(mc_df, port_perf, OUTPUT_DIR)
    plot_correlation_heatmap(returns, OUTPUT_DIR)
    plot_metrics_dashboard(metrics_df, port_perf, returns["Nifty50"], OUTPUT_DIR)
    plot_drawdown(returns, OUTPUT_DIR)

    # 6. Report
    print("[5/5] Printing summary report...")
    print_report(metrics_df, port_perf, mc_df, weights, stocks)

    print(f"\n✅ All outputs saved to the '{OUTPUT_DIR}/' folder.")


if __name__ == "__main__":
    main()
