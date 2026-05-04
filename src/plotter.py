# ============================================================
# src/plotter.py — All visualization logic lives here
# Each function produces and saves one plot
# ============================================================

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from scipy.stats import norm
import os
from config import PLOTS_DIR


def _save(filename):
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_price_and_volatility(prices, rolling_vol):
    """
    Plot 1: Price time series with rolling volatility on second axis.
    Shows the raw data and how volatility changes over time.
    """
    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(prices.index, prices.values, color='steelblue', linewidth=1.2, label='Price')
    ax1.set_ylabel('Price (USD)', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_xlabel('Date')

    ax2 = ax1.twinx()
    ax2.plot(rolling_vol.index, rolling_vol.values, color='tomato',
             linewidth=1.0, alpha=0.8, label='Rolling Volatility')
    ax2.set_ylabel('Rolling Volatility (σ)', color='tomato')
    ax2.tick_params(axis='y', labelcolor='tomato')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title('Price and Rolling Volatility Over Time')
    plt.tight_layout()
    _save("01_price_and_volatility.png")


def plot_return_distribution(log_returns):
    """
    Plot 2: Histogram of log returns with fitted normal curve.
    Shows whether returns are approximately normal — key statistical claim.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(log_returns, bins=40, density=True, color='steelblue',
            alpha=0.6, label='Log Returns')

    # Fit and overlay a normal distribution
    mu, std = float(log_returns.mean()), float(log_returns.std())
    x = np.linspace(log_returns.min(), log_returns.max(), 200)
    ax.plot(x, norm.pdf(x, mu, std), color='tomato', linewidth=2,
            label=f'Normal fit (μ={mu:.5f}, σ={std:.5f})')

    ax.set_xlabel('Log Return')
    ax.set_ylabel('Density')
    ax.set_title('Log Return Distribution')
    ax.legend()
    plt.tight_layout()
    _save("02_return_distribution.png")


def plot_pnl_comparison(as_df, baseline_df):
    """
    Plot 3: PnL over time — AS strategy vs naive baseline.
    The core comparison plot of the project.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(as_df.index, as_df['pnl'], color='steelblue',
            linewidth=1.3, label='AS Strategy (adaptive)')
    ax.plot(baseline_df.index, baseline_df['pnl'], color='tomato',
            linewidth=1.3, linestyle='--', label='Baseline (fixed spread)')

    ax.axhline(y=as_df['pnl'].iloc[0], color='gray', linewidth=0.8,
               linestyle=':', label='Starting PnL')

    ax.set_xlabel('Date')
    ax.set_ylabel('PnL (USD)')
    ax.set_title('PnL Over Time: AS Strategy vs Baseline')
    ax.legend()
    plt.tight_layout()
    _save("03_pnl_comparison.png")


def plot_inventory(as_df, baseline_df):
    """
    Plot 4: Inventory over time for both strategies.
    Shows how well each strategy controls inventory accumulation.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(as_df.index, as_df['inventory'], color='steelblue',
            linewidth=1.2, label='AS Strategy')
    ax.plot(baseline_df.index, baseline_df['inventory'], color='tomato',
            linewidth=1.2, linestyle='--', label='Baseline')

    ax.axhline(y=0, color='gray', linewidth=0.8, linestyle=':')
    ax.set_xlabel('Date')
    ax.set_ylabel('Inventory (shares)')
    ax.set_title('Inventory Over Time')
    ax.legend()
    plt.tight_layout()
    _save("04_inventory.png")


def plot_spread_over_time(as_df):
    """
    Plot 5: The AS full spread under the paper approximation.
    In the current approximation the total spread is constant; inventory
    changes skew bid/ask around the reservation price instead.
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(as_df.index, as_df['spread'], color='purple',
            linewidth=1.0, alpha=0.8)
    ax.set_xlabel('Date')
    ax.set_ylabel('Spread (USD)')
    ax.set_title('AS Strategy: Full Spread Over Time (paper approximation)')
    plt.tight_layout()
    _save("05_spread_over_time.png")


def plot_monte_carlo_final_pnl(as_runs, baseline_runs):
    """
    Plot Monte Carlo total PnL distributions for both strategies.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(as_runs["Total PnL"], bins=30, alpha=0.6, color="steelblue", label="AS Strategy")
    ax.hist(
        baseline_runs["Total PnL"],
        bins=30,
        alpha=0.55,
        color="tomato",
        label="Baseline",
    )

    ax.set_xlabel("Total PnL (USD)")
    ax.set_ylabel("Frequency")
    ax.set_title("Monte Carlo: Total PnL Distribution")
    ax.legend()
    plt.tight_layout()
    _save("06_monte_carlo_final_pnl.png")


def plot_monte_carlo_drawdown(as_runs, baseline_runs):
    """
    Plot Monte Carlo max drawdown distributions for both strategies.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(as_runs["Max Drawdown"], bins=30, alpha=0.6, color="steelblue", label="AS Strategy")
    ax.hist(
        baseline_runs["Max Drawdown"],
        bins=30,
        alpha=0.55,
        color="tomato",
        label="Baseline",
    )

    ax.set_xlabel("Max Drawdown (USD)")
    ax.set_ylabel("Frequency")
    ax.set_title("Monte Carlo: Max Drawdown Distribution")
    ax.legend()
    plt.tight_layout()
    _save("07_monte_carlo_drawdown.png")
