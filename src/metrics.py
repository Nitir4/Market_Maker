# ============================================================
# src/metrics.py — Evaluation metrics for simulation results
# Both strategies run through the same function
# ============================================================

import numpy as np
import pandas as pd


def compute_metrics(sim_df, strategy_name="Strategy"):
    """
    Compute all evaluation metrics from a simulation result DataFrame.

    Returns a dictionary of metrics. Both AS and baseline run
    through this same function so results are directly comparable.
    """

    pnl_series = sim_df['pnl']
    inventory_series = sim_df['inventory']
    spread_series = sim_df['spread']

    # Daily PnL changes
    daily_pnl_change = pnl_series.diff().dropna()

    # Total PnL (final value minus starting cash is accounted in PnL formula)
    total_pnl = round(pnl_series.iloc[-1] - pnl_series.iloc[0], 2)

    # Sharpe Ratio (annualized)
    mean_daily = daily_pnl_change.mean()
    std_daily = daily_pnl_change.std()
    sharpe = round((mean_daily / std_daily) * np.sqrt(252), 4) if std_daily > 0.01 else 0.0

    # Max Drawdown
    rolling_max = pnl_series.cummax()
    drawdown = pnl_series - rolling_max
    max_drawdown = round(drawdown.min(), 2)

    # Average spread
    avg_spread = round(spread_series.mean(), 6)

    # Inventory std dev (measures risk exposure over time)
    inventory_std = round(inventory_series.std(), 4)

    # Total trades executed
    total_buys  = int(sim_df['bid_hit'].sum())
    total_sells = int(sim_df['ask_hit'].sum())

    metrics = {
        "Strategy":         strategy_name,
        "Total PnL":        total_pnl,
        "Sharpe Ratio":     sharpe,
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
