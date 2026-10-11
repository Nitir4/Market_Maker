"""Avellaneda–Stoikov approximation with explicit price/time units."""
import math
import numbers
import numpy as np
from config import GAMMA, KAPPA, MARKET_SPREAD, FILL_REFERENCE_DT


def _validate_model_params():
    if not all(math.isfinite(v) and v > 0 for v in (GAMMA, KAPPA)):
        raise ValueError('GAMMA and KAPPA must be finite and positive.')


def _risk_inputs(inventory, volatility, time_remaining):
    _validate_model_params()
    if not all(math.isfinite(v) for v in (inventory, volatility, time_remaining)) or volatility < 0 or time_remaining < 0:
        raise ValueError('Use finite inventory, nonnegative price volatility and remaining time.')


def reservation_price(mid_price, inventory, volatility, time_remaining):
    """r = s - q gamma sigma² tau; sigma is price / sqrt(model time)."""
    _risk_inputs(inventory, volatility, time_remaining)
    if not math.isfinite(mid_price) or mid_price <= 0:
        raise ValueError('Mid-price must be finite and positive.')
    return mid_price - inventory * GAMMA * volatility**2 * time_remaining


def optimal_spread(inventory, volatility, time_remaining):
    """Signed bid/ask distances from mid, including the variance spread term.

    Paper equations (29)-(30): width = gamma sigma² tau + 2 log1p(gamma/kappa)/gamma.
    """
    _risk_inputs(inventory, volatility, time_remaining)
    risk = GAMMA * volatility**2 * time_remaining
    half_width = .5 * risk + np.log1p(GAMMA/KAPPA)/GAMMA
    skew = inventory * risk
    return skew+half_width, -skew+half_width


def calibration_time_step(n_steps, horizon):
    if isinstance(n_steps, bool) or not isinstance(n_steps, numbers.Integral) or n_steps < 1:
        raise ValueError('Use at least one simulation step.')
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError('Horizon must be finite and positive.')
    return horizon/n_steps


def arrival_scale(dt, market_spread=MARKET_SPREAD):
    """Stylized A = exp(-kappa M/2)/reference_dt, not daily-data calibration."""
    _validate_model_params()
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(market_spread) or market_spread < 0:
        raise ValueError('Use positive dt and a finite nonnegative market spread.')
    return np.exp(-KAPPA*market_spread/2)/dt


def fill_probability(spread_distance, dt, market_spread=MARKET_SPREAD, reference_dt=None):
    """Poisson probability of at least one hit; one unit maximum per side/step.

    reference_dt fixes A if callers subdivide an interval. Signed distances
    must not be replaced with abs: a more aggressive quote has higher intensity.
    """
    reference_dt = FILL_REFERENCE_DT if reference_dt is None else reference_dt
    arrival_scale(reference_dt, market_spread)
    if not math.isfinite(spread_distance) or not math.isfinite(dt) or dt <= 0:
        raise ValueError('Use a finite distance and positive dt.')
    log_mean = -KAPPA*(market_spread/2+spread_distance)+math.log(dt/reference_dt)
    return float(-np.expm1(-np.exp(min(log_mean, 50.))))
