"""Fixed-spread quotes with the same execution/accounting as AS."""
import math
from config import FIXED_SPREAD
from src.simulation_common import run_passive_simulation


def run_baseline_simulation(prices, rolling_vol, rng=None, fixed_spread=None):
    spread = FIXED_SPREAD if fixed_spread is None else fixed_spread
    if not math.isfinite(spread) or spread <= 0:
        raise ValueError('Fixed spread must be finite and positive.')
    return run_passive_simulation(prices,rolling_vol,
                                 lambda mid,q,sigma,tau:(mid-spread/2,mid+spread/2,mid),rng)
