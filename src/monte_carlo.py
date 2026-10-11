# ============================================================
# src/monte_carlo.py — Arithmetic Brownian Monte Carlo wrapper
# Separate from the historical backtest pipeline.
# ============================================================

import os

import numpy as np
import pandas as pd

from config import MC_MIN_PRICE, MC_RANDOM_SEED, N_SIMULATIONS, RESULTS_DIR
from src.baseline import run_baseline_simulation
from src.metrics import compute_metrics
from src.simulation import run_as_simulation
from src.statistics import compute_log_returns, compute_rolling_volatility


def estimate_abm_parameters(prices):
    """
    Estimate arithmetic Brownian motion parameters from historical prices.

    Model:
      S_{t+1} = S_t + mu + sigma * Z_t
    """
    if not isinstance(prices, pd.Series) or len(prices) < 3 or not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError('ABM calibration needs at least three finite positive prices.')
    price_changes = prices.diff().dropna()
    return float(price_changes.mean()), float(price_changes.std())


def simulate_abm_price_path(prices, rng, min_price=MC_MIN_PRICE):
    """
    Generate one arithmetic Brownian price path with the same length/index as
    the historical series. A small floor keeps simulated prices positive so the
    downstream log-return calculations remain valid.
    """
    if not np.isfinite(min_price) or min_price <= 0:
        raise ValueError('Price floor must be finite and positive.')
    mu_step, sigma_step = estimate_abm_parameters(prices)
    simulated = np.empty(len(prices), dtype=float)
    simulated[0] = float(prices.iloc[0])

    shocks = rng.normal(loc=mu_step, scale=sigma_step, size=len(prices) - 1)
    for i, shock in enumerate(shocks, start=1):
        simulated[i] = max(min_price, simulated[i - 1] + shock)

    return pd.Series(simulated, index=prices.index, name="Close")


def validate_abm_monte_carlo(prices, n_paths=25, seed=MC_RANDOM_SEED):
    """
    Compare historical step-change moments against a sample of simulated
    arithmetic Brownian paths to sanity-check the Monte Carlo generator.
    """
    hist_changes = prices.diff().dropna()
    hist_mu = float(hist_changes.mean())
    hist_sigma = float(hist_changes.std())

    simulated_means = []
    simulated_stds = []

    for i in range(n_paths):
        rng = np.random.default_rng(seed + 50_000 + i)
        sim_prices = simulate_abm_price_path(prices, rng)
        sim_changes = sim_prices.diff().dropna()
        simulated_means.append(float(sim_changes.mean()))
        simulated_stds.append(float(sim_changes.std()))

    return {
        "Historical Step Mean": round(hist_mu, 6),
        "Historical Step Std": round(hist_sigma, 6),
        "Simulated Mean Step Mean": round(float(np.mean(simulated_means)), 6),
        "Simulated Mean Step Std": round(float(np.std(simulated_means)), 6),
        "Simulated Std Mean": round(float(np.mean(simulated_stds)), 6),
        "Simulated Std Std": round(float(np.std(simulated_stds)), 6),
        "Validation Paths": int(n_paths),
    }


def _simulation_metrics_row(sim_df, run_id, strategy_name):
    metrics = compute_metrics(sim_df, strategy_name=strategy_name)
    metrics["Run"] = run_id
    metrics["Final Equity"] = float(sim_df["pnl"].iloc[-1])
    metrics["Final PnL"] = metrics["Total PnL"]
    metrics["Final Inventory"] = int(sim_df["inventory"].iloc[-1])
    return metrics


def _summary_table(metrics_df, strategy_name):
    summary = {
        "Strategy": strategy_name,
        "Mean Total PnL": round(float(metrics_df["Total PnL"].mean()), 2),
        "Std Total PnL": round(float(metrics_df["Total PnL"].std()), 2),
        "Median Total PnL": round(float(metrics_df["Total PnL"].median()), 2),
        "P05 Total PnL": round(float(metrics_df["Total PnL"].quantile(0.05)), 2),
        "P95 Total PnL": round(float(metrics_df["Total PnL"].quantile(0.95)), 2),
        "Mean Sharpe Ratio": round(float(metrics_df["Sharpe Ratio"].mean()), 4),
        "Mean Max Drawdown": round(float(metrics_df["Max Drawdown"].mean()), 2),
        "Mean Inventory Std": round(float(metrics_df["Inventory Std"].mean()), 4),
        "Mean Final Inventory": round(float(metrics_df["Final Inventory"].mean()), 4),
    }
    return summary


def print_monte_carlo_summary(as_summary, baseline_summary, as_metrics_df, baseline_metrics_df):
    """
    Print compact summary stats for the Monte Carlo comparison.
    """
    win_rate = float((as_metrics_df["Total PnL"] > baseline_metrics_df["Total PnL"]).mean())

    print("\n" + "=" * 70)
    print("MONTE CARLO SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<24} {'AS Strategy':>18} {'Baseline':>18}")
    print("=" * 70)

    keys = [
        "Mean Total PnL",
        "Std Total PnL",
        "Median Total PnL",
        "P05 Total PnL",
        "P95 Total PnL",
        "Mean Sharpe Ratio",
        "Mean Max Drawdown",
        "Mean Inventory Std",
        "Mean Final Inventory",
    ]

    for key in keys:
        print(f"  {key:<22} {str(as_summary[key]):>18} {str(baseline_summary[key]):>18}")

    print("=" * 70)
    print(f"  {'AS Outperformance Prob':<22} {win_rate:>18.4%}")
    print("=" * 70 + "\n")


def print_monte_carlo_validation(validation_stats):
    """
    Print a compact sanity check for the arithmetic Brownian path generator.
    """
    print("=" * 70)
    print("MONTE CARLO PATH VALIDATION")
    print("=" * 70)
    for key, value in validation_stats.items():
        print(f"  {key:<26} {value}")
    print("=" * 70 + "\n")


def save_monte_carlo_results(as_metrics_df, baseline_metrics_df):
    """
    Save per-run and aggregate Monte Carlo outputs under outputs/results.
    """
    as_path = os.path.join(RESULTS_DIR, "monte_carlo_as_runs.csv")
    baseline_path = os.path.join(RESULTS_DIR, "monte_carlo_baseline_runs.csv")
    summary_path = os.path.join(RESULTS_DIR, "monte_carlo_summary.csv")

    as_metrics_df.to_csv(as_path, index=False)
    baseline_metrics_df.to_csv(baseline_path, index=False)

    summary_df = pd.DataFrame(
        [
            _summary_table(as_metrics_df, "AS Strategy"),
            _summary_table(baseline_metrics_df, "Baseline"),
        ]
    )
    summary_df["AS Outperformance Prob"] = [round(float((as_metrics_df["Total PnL"] > baseline_metrics_df["Total PnL"]).mean()), 6), ""]
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved: {as_path}")
    print(f"Saved: {baseline_path}")
    print(f"Saved: {summary_path}")

    return summary_df


def save_monte_carlo_validation(validation_stats):
    """
    Save Monte Carlo path-validation stats under outputs/results.
    """
    validation_path = os.path.join(RESULTS_DIR, "monte_carlo_path_validation.csv")
    pd.DataFrame([validation_stats]).to_csv(validation_path, index=False)
    print(f"Saved: {validation_path}")


def run_monte_carlo(prices, n_simulations=N_SIMULATIONS, seed=MC_RANDOM_SEED, baseline_fixed_spread=None):
    """
    Run arithmetic-Brownian Monte Carlo on top of the existing AS and baseline
    strategies. Historical plots and metrics are untouched; this produces
    separate distributions for "what could have happened."
    """
    if isinstance(n_simulations, bool) or not isinstance(n_simulations, (int, np.integer)) or n_simulations < 1:
        raise ValueError('Simulation count must be a positive integer.')
    as_rows = []
    baseline_rows = []

    if baseline_fixed_spread is None:
        from config import FIXED_SPREAD
        baseline_fixed_spread = FIXED_SPREAD

    for run_id in range(1, n_simulations + 1):
        # independent RNG for price path generation but a shared RNG for
        # fill Bernoulli draws so both strategies face the same randomness
        # stream (paired comparison). This reduces sampling noise and
        # isolates differences to quoting logic.
        path_rng = np.random.default_rng(seed + run_id)
        fills_rng = np.random.default_rng(seed + 30_000 + run_id)

        simulated_prices = simulate_abm_price_path(prices, path_rng)
        simulated_returns = compute_log_returns(simulated_prices)
        simulated_rolling_vol = compute_rolling_volatility(simulated_returns)

        as_df = run_as_simulation(simulated_prices, simulated_rolling_vol, rng=fills_rng)
        # reset the fills RNG to the same seed for the baseline so the
        # sequence of uniform/binary draws is identical (paired)
        fills_rng = np.random.default_rng(seed + 30_000 + run_id)
        baseline_df = run_baseline_simulation(
            simulated_prices,
            simulated_rolling_vol,
            rng=fills_rng,
            fixed_spread=baseline_fixed_spread,
        )

        as_rows.append(_simulation_metrics_row(as_df, run_id, "AS Strategy"))
        baseline_rows.append(_simulation_metrics_row(baseline_df, run_id, "Baseline"))

    as_metrics_df = pd.DataFrame(as_rows)
    baseline_metrics_df = pd.DataFrame(baseline_rows)

    as_summary = _summary_table(as_metrics_df, "AS Strategy")
    baseline_summary = _summary_table(baseline_metrics_df, "Baseline")

    return {
        "as_runs": as_metrics_df,
        "baseline_runs": baseline_metrics_df,
        "as_summary": as_summary,
        "baseline_summary": baseline_summary,
        "validation": validate_abm_monte_carlo(prices, seed=seed),
    }
