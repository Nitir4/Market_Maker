# ============================================================
# src/statistics.py — Computes all statistical inputs
# Log returns, rolling volatility, summary stats
# ============================================================

import numpy as np
import pandas as pd
from config import ROLLING_WINDOW


def compute_log_returns(prices):
    """
    Compute log returns from price series.
    r(t) = ln(P(t) / P(t-1))
    """
    # Ensure prices is a Series
    if isinstance(prices, pd.DataFrame):
        prices = prices.squeeze()

    log_returns = np.log(prices / prices.shift(1)).dropna()

    # Ensure output is Series
    if isinstance(log_returns, pd.DataFrame):
        log_returns = log_returns.squeeze()

    return log_returns


def compute_rolling_volatility(log_returns):
    """
    Compute rolling standard deviation of log returns.
    """
    # Ensure input is Series
    if isinstance(log_returns, pd.DataFrame):
        log_returns = log_returns.squeeze()

    rolling_vol = log_returns.rolling(window=ROLLING_WINDOW).std().dropna()
    return rolling_vol


def compute_summary_stats(log_returns):
    """
    Print and return basic summary statistics.
    """

    # Ensure input is Series
    if isinstance(log_returns, pd.DataFrame):
        log_returns = log_returns.squeeze()

    stats = {
        "Mean daily return":     round(float(log_returns.mean()), 6),
        "Std dev (overall)":     round(float(log_returns.std()), 6),
        "Skewness":              round(float(log_returns.skew()), 4),
        "Kurtosis":              round(float(log_returns.kurtosis()), 4),
        "Min return":            round(float(log_returns.min()), 6),
        "Max return":            round(float(log_returns.max()), 6),
    }

    print("\n--- Statistical Summary ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("----------------------------\n")

    return stats