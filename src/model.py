# ============================================================
# src/model.py — Avellaneda-Stoikov math functions
# Pure math only. No simulation, no data loading here.
# ============================================================

import numpy as np
from config import GAMMA, KAPPA, MARKET_SPREAD


def _validate_model_params():
  if GAMMA <= 0:
    raise ValueError(f"GAMMA must be > 0 for AS model; got {GAMMA}")
  if KAPPA <= 0:
    raise ValueError(f"KAPPA must be > 0 for AS model; got {KAPPA}")


def reservation_price(mid_price, inventory, volatility, time_remaining):
    """
    Adjust the agent's fair value based on inventory held.

    Formula (paper approximation, Sec. 3.2):
    r = s - q * gamma * sigma^2 * (T - t)

    - If inventory is positive (long), reservation price drops
      → agent wants to sell, so quotes shift down
    - If inventory is negative (short), reservation price rises
      → agent wants to buy, so quotes shift up

    This is the inventory risk control mechanism.
    """
    r = mid_price - inventory * GAMMA * (volatility ** 2) * time_remaining
    return r


def optimal_spread(inventory, volatility, time_remaining):
    """
    Compute the paper's approximate optimal bid/ask distances (Sec. 3.2).

    Approximation used in the paper:
      delta_b = gamma*q*sigma^2*(T-t) + (1/gamma)*ln(1 + gamma/kappa)
      delta_a = -gamma*q*sigma^2*(T-t) + (1/gamma)*ln(1 + gamma/kappa)

    where:
      delta_b = s - bid
      delta_a = ask - s

    The inventory term skews the two sides asymmetrically, while the
    logarithmic term is the constant spread component.
    """
    _validate_model_params()
    spread_component = (1 / GAMMA) * np.log(1 + GAMMA / KAPPA)
    inventory_component = inventory * GAMMA * (volatility ** 2) * time_remaining

    delta_b = inventory_component + spread_component
    delta_a = -inventory_component + spread_component
    return delta_b, delta_a


def calibration_time_step(n_steps, horizon):
    """
    Paper-style simulation time step.

    If the total trading horizon is T and we simulate n_steps decisions,
    then dt = T / n_steps.
    """
    return horizon / n_steps


def arrival_scale(dt, market_spread=MARKET_SPREAD):
    """
    Calibrate A from the paper's Eq. (3.9):

      lambda(-M/2) * dt = A * exp(kappa * M / 2) * dt = 1

    Solving for A gives:
      A = exp(-kappa * M / 2) / dt

    Here M is the market-spread proxy used by the arrival calibration. In this
    project it is supplied from configuration and can be replaced by a
    separately calibrated SPY microstructure estimate.
    """
    _validate_model_params()
    return np.exp(-KAPPA * market_spread / 2) / dt


def fill_probability(spread_distance, dt, market_spread=MARKET_SPREAD):
    """
    Probability that a quote gets hit over one simulation step.

    Paper approximation:
      lambda(delta) = A * exp(-kappa * delta)
      p_fill ~= lambda(delta) * dt

    with A calibrated from Eq. (3.9).

    We clip to [0, 1] because this simulator uses one Bernoulli trial per step.
    """
    lam = arrival_scale(dt, market_spread) * np.exp(-KAPPA * spread_distance)
    prob = lam * dt
    return float(np.clip(prob, 0.0, 1.0))
