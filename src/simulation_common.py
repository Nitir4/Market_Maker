"""Shared one-unit passive fill/accounting kernel for both strategies."""
import numpy as np
import pandas as pd
from config import INITIAL_CASH, MARKET_SPREAD, OBSERVATION_DT, FILL_REFERENCE_DT, TRADING_DAYS_PER_YEAR
from src.model import fill_probability, arrival_scale


def prepare_inputs(prices, rolling_vol):
    if not isinstance(prices, pd.Series) or not isinstance(rolling_vol, pd.Series):
        raise ValueError('Prices and daily return volatility must be Series.')
    for values in (prices, rolling_vol):
        if values.empty or not values.index.is_unique or not values.index.is_monotonic_increasing:
            raise ValueError('Inputs must be nonempty with unique chronological indices.')
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError('Input observations must be finite.')
    if (prices <= 0).any() or (rolling_vol < 0).any():
        raise ValueError('Prices must be positive and return volatility nonnegative.')
    if not rolling_vol.index.isin(prices.index).all():
        raise ValueError('Every volatility observation needs a matching price.')
    if not np.isfinite(INITIAL_CASH) or INITIAL_CASH <= 0 or not np.isfinite(MARKET_SPREAD) or MARKET_SPREAD <= 0:
        raise ValueError('Starting cash and market spread must be finite and positive.')
    aligned = prices.loc[rolling_vol.index].astype(float)
    if not np.isfinite(OBSERVATION_DT) or OBSERVATION_DT <= 0 or not np.isfinite(TRADING_DAYS_PER_YEAR) or TRADING_DAYS_PER_YEAR <= 0:
        raise ValueError('Observation interval and trading days per year must be positive.')
    arrival_scale(FILL_REFERENCE_DT)
    return aligned, rolling_vol.astype(float), float(OBSERVATION_DT)


def run_passive_simulation(prices, rolling_vol, quote_function, rng=None):
    prices, rolling_vol, dt = prepare_inputs(prices, rolling_vol)
    rng = np.random.default_rng() if rng is None else rng
    cash, inventory = float(INITIAL_CASH), 0
    records=[]
    for i, (date, mid) in enumerate(prices.items()):
        # Volatility is per observation; model time is measured in trading days.
        # Extending a dataset extends its terminal horizon, not its time units.
        sigma = mid*float(rolling_vol.iloc[i])/np.sqrt(dt)
        remaining = (len(prices)-i)*dt
        desired_bid, desired_ask, reservation = quote_function(mid, inventory, sigma, remaining)
        if not np.isfinite([desired_bid,desired_ask]).all() or desired_bid >= desired_ask:
            raise ValueError('Quote function must produce finite ordered prices.')
        # This kernel models passive quotes only. Cap aggressive inventory-skewed
        # quotes at their own-side touch instead of awarding fictitious crossed fills.
        touch_half = min(MARKET_SPREAD/2, mid/2)
        floor = min(.01, mid/2)
        bid = max(floor, min(desired_bid, mid-touch_half))
        ask = max(desired_ask, mid+touch_half)
        p_bid = fill_probability(mid-bid, dt, reference_dt=FILL_REFERENCE_DT)
        p_ask = fill_probability(ask-mid, dt, reference_dt=FILL_REFERENCE_DT)
        # Inverse-CDF draws consume one uniform per side regardless of p. Resetting
        # the seed truly pairs the two strategies even at probabilities zero/one.
        bid_hit = int(rng.random() < p_bid)
        ask_hit = int(rng.random() < p_ask)
        inventory += bid_hit-ask_hit
        cash += ask_hit*ask-bid_hit*bid
        equity = cash+inventory*mid
        records.append(dict(date=date,time_start=i*dt,time_end=(i+1)*dt,time_remaining=remaining,
                            mid_price=mid,reservation=reservation,bid=bid,ask=ask,
                            desired_bid=desired_bid,desired_ask=desired_ask,spread=ask-bid,
                            price_volatility=sigma,p_bid=p_bid,p_ask=p_ask,bid_hit=bid_hit,
                            ask_hit=ask_hit,inventory=inventory,cash=cash,pnl=equity,
                            profit=equity-INITIAL_CASH))
    result=pd.DataFrame(records).set_index('date')
    result.attrs.update(initial_cash=float(INITIAL_CASH),dt=dt,fill_model='passive one-unit Poisson-hit',
                        reference_dt=float(FILL_REFERENCE_DT),horizon=len(prices)*dt,
                        periods_per_year=TRADING_DAYS_PER_YEAR/dt,time_units='trading days',
                        volatility_units='price/sqrt(trading day)')
    return result
