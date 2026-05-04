# ============================================================
# src/simulation.py — Avellaneda-Stoikov simulation (paper approximation)
# Main strategy: reservation adapts to inventory/volatility; spread is fixed by gamma,kappa
# ============================================================

import numpy as np
import pandas as pd
from config import INITIAL_CASH, T
from src.model import (
    calibration_time_step,
    fill_probability,
    optimal_spread,
    reservation_price,
)


def run_as_simulation(prices, rolling_vol, rng=None):
    """
    Run the AS market making simulation over the price series.

    At each timestep:
    1. Compute reservation price (inventory-adjusted fair value)
    2. Compute the paper's approximate bid/ask distances
    3. Post bid and ask from the mid-price using those distances
    4. Simulate fills using Bernoulli draws
    5. Update inventory, cash, and PnL

    Returns a DataFrame with full simulation history.
    """

    if rng is None:
        rng = np.random

    # Align prices and volatility on the same index
    common_index = rolling_vol.index
    prices = prices.loc[common_index]

    n_steps = len(prices)
    price_values = prices.values
    vol_values = rolling_vol.values
    dt = calibration_time_step(n_steps, T)

    # Initialize state
    cash = INITIAL_CASH
    inventory = 0

    # Storage for results
    records = []

    for i in range(n_steps):
        mid = price_values[i]
        sigma = vol_values[i]

        # Time remaining (normalized from 1 down to 0)
        time_remaining = T * (1 - i / n_steps)

        # Step 1: Compute reservation price
        r = reservation_price(mid, inventory, sigma, time_remaining)

        # Step 2: Compute paper-approximate bid/ask distances
        delta_b, delta_a = optimal_spread(inventory, sigma, time_remaining)

        # Step 3: Set bid and ask from the mid-price
        bid = mid - delta_b
        ask = mid + delta_a

        # Step 4: Compute fill probabilities
        bid_distance = abs(mid - bid)
        ask_distance = abs(ask - mid)

        p_bid = fill_probability(bid_distance, dt)
        p_ask = fill_probability(ask_distance, dt)

        # Step 5: Bernoulli draws — did the trade happen?
        bid_hit = rng.binomial(1, p_bid)
        ask_hit = rng.binomial(1, p_ask)

        # Step 6: Execute trades and update state
        if bid_hit:
            inventory += 1
            cash -= bid       # We bought at bid price

        if ask_hit:
            inventory -= 1
            cash += ask       # We sold at ask price

        # Step 7: Mark-to-market PnL
        pnl = cash + inventory * mid

        # Log this timestep
        records.append({
            "date":        prices.index[i],
            "mid_price":   round(mid, 4),
            "reservation": round(r, 4),
            "bid":         round(bid, 4),
            "ask":         round(ask, 4),
            "spread":      round(delta_b + delta_a, 4),
            "p_bid":       round(p_bid, 4),
            "p_ask":       round(p_ask, 4),
            "bid_hit":     bid_hit,
            "ask_hit":     ask_hit,
            "inventory":   inventory,
            "cash":        round(cash, 2),
            "pnl":         round(pnl, 2),
        })

    df = pd.DataFrame(records).set_index("date")
    return df
