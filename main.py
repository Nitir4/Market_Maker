# ============================================================
# main.py — Run this file to execute the entire project
# Reads top to bottom like a story of what the project does
# ============================================================

import os
import sys

# Make sure src imports work regardless of where you run from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PLOTS_DIR, RESULTS_DIR
from src.data_loader import load_data
from src.statistics import compute_log_returns, compute_rolling_volatility, compute_summary_stats
from src.simulation import run_as_simulation
from src.baseline import run_baseline_simulation
from src.metrics import compute_metrics, print_metrics_table, save_metrics_table
from src.monte_carlo import (
    print_monte_carlo_validation,
    print_monte_carlo_summary,
    run_monte_carlo,
    save_monte_carlo_results,
    save_monte_carlo_validation,
)
from src.plotter import (
    plot_price_and_volatility,
    plot_return_distribution,
    plot_pnl_comparison,
    plot_inventory,
    plot_spread_over_time,
    plot_monte_carlo_drawdown,
    plot_monte_carlo_final_pnl,
)


def main():

    # --- Step 0: Make sure output folders exist ---
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Output folders ready.\n")

    # --- Step 1: Load Data ---
    print("=" * 50)
    print("STEP 1: Loading Data")
    print("=" * 50)
    prices = load_data()

    # --- Step 2: Statistical Analysis ---
    print("\n" + "=" * 50)
    print("STEP 2: Statistical Analysis")
    print("=" * 50)
    log_returns   = compute_log_returns(prices)
    rolling_vol   = compute_rolling_volatility(log_returns)
    summary_stats = compute_summary_stats(log_returns)

    # --- Step 3: Generate Data Plots ---
    print("=" * 50)
    print("STEP 3: Generating Data Plots")
    print("=" * 50)
    plot_price_and_volatility(prices, rolling_vol)
    plot_return_distribution(log_returns)

    # --- Step 4: Run AS Simulation ---
    print("\n" + "=" * 50)
    print("STEP 4: Running AS Strategy Simulation")
    print("=" * 50)
    from config import MC_RANDOM_SEED
    import numpy as np
    as_results = run_as_simulation(prices, rolling_vol, rng=np.random.default_rng(MC_RANDOM_SEED))
    print(f"Simulation complete. {len(as_results)} timesteps.")

    # --- Step 5: Run Baseline Simulation ---
    print("\n" + "=" * 50)
    print("STEP 5: Running Baseline Simulation")
    print("=" * 50)
    baseline_results = run_baseline_simulation(prices, rolling_vol, rng=np.random.default_rng(MC_RANDOM_SEED))
    print(f"Baseline complete. {len(baseline_results)} timesteps.")

    # --- Step 6: Compute Metrics ---
    print("\n" + "=" * 50)
    print("STEP 6: Computing Metrics")
    print("=" * 50)
    as_metrics       = compute_metrics(as_results, strategy_name="AS Strategy")
    baseline_metrics = compute_metrics(baseline_results, strategy_name="Baseline")
    print_metrics_table(as_metrics, baseline_metrics)

    # Save metrics to CSV
    metrics_path = os.path.join(RESULTS_DIR, "metrics_comparison.csv")
    save_metrics_table(as_metrics, baseline_metrics, metrics_path)

    # --- Step 7: Generate Simulation Plots ---
    print("\n" + "=" * 50)
    print("STEP 7: Generating Simulation Plots")
    print("=" * 50)
    plot_pnl_comparison(as_results, baseline_results)
    plot_inventory(as_results, baseline_results)
    plot_spread_over_time(as_results)

    # --- Step 8: Monte Carlo Evaluation ---
    print("\n" + "=" * 50)
    print("STEP 8: Monte Carlo Evaluation")
    print("=" * 50)
    monte_carlo_results = run_monte_carlo(prices)
    print_monte_carlo_summary(
        monte_carlo_results["as_summary"],
        monte_carlo_results["baseline_summary"],
        monte_carlo_results["as_runs"],
        monte_carlo_results["baseline_runs"],
    )
    print_monte_carlo_validation(monte_carlo_results["validation"])
    save_monte_carlo_results(
        monte_carlo_results["as_runs"],
        monte_carlo_results["baseline_runs"],
    )
    save_monte_carlo_validation(monte_carlo_results["validation"])
    plot_monte_carlo_final_pnl(
        monte_carlo_results["as_runs"],
        monte_carlo_results["baseline_runs"],
    )
    plot_monte_carlo_drawdown(
        monte_carlo_results["as_runs"],
        monte_carlo_results["baseline_runs"],
    )

    # --- Done ---
    print("\n" + "=" * 50)
    print("ALL DONE")
    print(f"Plots saved to:   {PLOTS_DIR}/")
    print(f"Metrics saved to: {RESULTS_DIR}/")
    print("=" * 50)


if __name__ == "__main__":
    main()
