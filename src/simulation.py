"""AS quotes with the shared passive execution/accounting kernel."""
from src.model import optimal_spread, reservation_price
from src.simulation_common import run_passive_simulation


def run_as_simulation(prices, rolling_vol, rng=None):
    def quotes(mid, inventory, sigma, remaining):
        delta_b, delta_a = optimal_spread(inventory,sigma,remaining)
        return mid-delta_b, mid+delta_a, reservation_price(mid,inventory,sigma,remaining)
    return run_passive_simulation(prices,rolling_vol,quotes,rng)
