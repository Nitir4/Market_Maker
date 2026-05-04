# ============================================================
# src/metrics.py — Evaluation metrics for simulation results
# Both strategies run through the same function
# ============================================================

import numpy as np
import pandas as pd
from config import INITIAL_CASH, OBSERVATION_DT, TRADING_DAYS_PER_YEAR


def compute_metrics(sim_df, strategy_name="Strategy", initial_cash=None, periods_per_year=None):
    """
    Compute all evaluation metrics from a simulation result DataFrame.

    Returns a dictionary of metrics. Both AS and baseline run
    through this same function so results are directly comparable.
    """

    initial_cash = sim_df.attrs.get('initial_cash', INITIAL_CASH) if initial_cash is None else initial_cash
    if not np.isfinite(initial_cash) or initial_cash <= 0 or sim_df.empty:
        raise ValueError('Use a nonempty result and finite positive initial cash.')
    pnl_series = sim_df['pnl'].astype(float)
    if not np.isfinite(pnl_series).all() or (pnl_series <= 0).any():
        raise ValueError('Equity must be finite and positive for return-based metrics.')
    inventory_series = sim_df['inventory']
    spread_series = sim_df['spread']
    # Include the first recorded post-trade change from initial pre-trade cash.
    previous = pnl_series.shift(1).fillna(initial_cash)
    returns = pnl_series/previous-1
    total_pnl = round(float(pnl_series.iloc[-1]-initial_cash), 2)
    std = returns.std(ddof=1)
    periods_per_year = sim_df.attrs.get('periods_per_year', TRADING_DAYS_PER_YEAR/OBSERVATION_DT) if periods_per_year is None else periods_per_year
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError('Annualization periods must be finite and positive.')
    sharpe = round(float(returns.mean()/std*np.sqrt(periods_per_year)), 4) if np.isfinite(std) and std > 0 else 0.0
    rolling_max = pnl_series.cummax().clip(lower=initial_cash)
    max_drawdown = round(float((pnl_series-rolling_max).min()), 2)

    # Average spread
    avg_spread = round(spread_series.mean(), 6)

    # Inventory std dev (measures risk exposure over time)
    inventory_std = round(float(inventory_series.std(ddof=0)), 4)

    # Total trades executed
    total_buys  = int(sim_df['bid_hit'].sum())
    total_sells = int(sim_df['ask_hit'].sum())

    metrics = {
        "Strategy":         strategy_name,
        "Initial Cash":     float(initial_cash),
        "Final Equity":     float(pnl_series.iloc[-1]),
        "Total PnL":        total_pnl,
        "Sharpe Ratio":     sharpe,
        "Periods Per Year": float(periods_per_year),
        "Max Drawdown":     max_drawdown,
        "Avg Spread":       avg_spread,
        "Inventory Std":    inventory_std,
        "Total Buys":       total_buys,
        "Total Sells":      total_sells,
    }

    return metrics


def print_metrics_table(as_metrics, baseline_metrics):
    """
    Print both strategies side by side as a clean comparison table.
    """
    print("\n" + "=" * 55)
    print(f"{'Metric':<22} {'AS Strategy':>15} {'Baseline':>15}")
    print("=" * 55)

    keys = ["Total PnL", "Sharpe Ratio", "Max Drawdown",
            "Avg Spread", "Inventory Std", "Total Buys", "Total Sells"]

    for key in keys:
        as_val  = str(as_metrics.get(key, "-"))
        bl_val  = str(baseline_metrics.get(key, "-"))
        print(f"  {key:<20} {as_val:>15} {bl_val:>15}")

    print("=" * 55 + "\n")


def save_metrics_table(as_metrics, baseline_metrics, path):
    """
    Save the comparison table to a CSV file.
    """
    df = pd.DataFrame([as_metrics, baseline_metrics])
    df.to_csv(path, index=False)
    print(f"Metrics saved to {path}")
