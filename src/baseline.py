# ============================================================
# src/baseline.py — Naive fixed-spread baseline strategy
# Same execution model, no inventory adjustment, constant spread
# ============================================================

import numpy as np
import pandas as pd
from config import FIXED_SPREAD, INITIAL_CASH, T
from src.model import calibration_time_step, fill_probability


def run_baseline_simulation(prices, rolling_vol, rng=None, fixed_spread=None):
    """
    Run the naive fixed-spread market making simulation.

    Same structure as AS simulation but:
    - Spread is always FIXED_SPREAD (constant, from config)
    - No reservation price adjustment
    - Quotes always centered on mid-price

    This is the comparison baseline. Same data, same execution
    model — only the quoting strategy differs.
    """

    if rng is None:
        rng = np.random

    # Align on same index as AS simulation
    common_index = rolling_vol.index
    prices = prices.loc[common_index]

    n_steps = len(prices)
    price_values = prices.values
    dt = calibration_time_step(n_steps, T)

    # Initialize state
    cash = INITIAL_CASH
    inventory = 0

    records = []

    for i in range(n_steps):
        mid = price_values[i]

        # Determine spread to use (allow overriding to match AS mean spread)
        if fixed_spread is None:
            fixed_spread = FIXED_SPREAD

        # Fixed spread — no adaptation to volatility or inventory
        half_spread = fixed_spread / 2
        bid = mid - half_spread
        ask = mid + half_spread

        # Same fill probability model as AS strategy (fair comparison)
        bid_distance = abs(mid - bid)
        ask_distance = abs(ask - mid)

        p_bid = fill_probability(bid_distance, dt)
        p_ask = fill_probability(ask_distance, dt)

        # Bernoulli draws
        bid_hit = rng.binomial(1, p_bid)
        ask_hit = rng.binomial(1, p_ask)

        # Execute trades
        if bid_hit:
            inventory += 1
            cash -= bid

        if ask_hit:
            inventory -= 1
            cash += ask

        # Mark-to-market PnL
        pnl = cash + inventory * mid

        records.append({
            "date":      prices.index[i],
            "mid_price": round(mid, 4),
            "bid":       round(bid, 4),
            "ask":       round(ask, 4),
            "spread":    round(fixed_spread, 4),
            "p_bid":     round(p_bid, 4),
            "p_ask":     round(p_ask, 4),
            "bid_hit":   bid_hit,
            "ask_hit":   ask_hit,
            "inventory": inventory,
            "cash":      round(cash, 2),
            "pnl":       round(pnl, 2),
        })

    df = pd.DataFrame(records).set_index("date")
    return df
