# Market Maker simulation

A stylized Avellaneda–Stoikov market-making experiment with a fixed-spread baseline and arithmetic-Brownian Monte Carlo. Run `pip install -r requirements.txt` and `python main.py` from this folder. The runner may download daily price data through yfinance if no usable cache exists.

## Equations and units

For inventory q, mid-price s, price volatility sigma and remaining model time tau:

```
r = s - q*gamma*sigma²*tau
width = gamma*sigma²*tau + 2*log1p(gamma/kappa)/gamma
bid_distance = q*gamma*sigma²*tau + width/2
ask_distance = -q*gamma*sigma²*tau + width/2
```

These use equations (29)-(30) in [Avellaneda and Stoikov (2008)](https://math.nyu.edu/inmemoriam/avellaneda/HighFrequencyTrading.pdf). The previous implementation omitted the variance contribution to width.

`rolling_vol` is daily log-return standard deviation. Each retained observation spans fixed `OBSERVATION_DT` trading days (default one), independent of dataset length. Terminal horizon is `n*dt`, so extending data legitimately extends inventory risk until that endpoint. The kernel converts it to `sigma = mid*rolling_vol/sqrt(dt)` in price units per square root of trading day. Gamma therefore acts in inverse-price units. Changing the time convention or price scale requires consistent calibration; earlier outputs using raw return volatility are historical, not corrected results.

## Execution and accounting

Both strategies share the same passive execution kernel. Desired aggressive quotes are capped at their own-side touch using the configured spread proxy; deep bids retain a positive price floor. These constraints are implementation choices, so this is not an exact reproduction of the paper's unconstrained quotes.

The arrival-scale proxy is `A=exp(-kappa*M/2)/reference_dt`. The probability of at least one arrival is `1-exp(-A*exp(-kappa*distance)*dt)`. The simulator executes at most one unit per side and interval. `FILL_REFERENCE_DT` (default one trading day) holds A fixed even when subdividing intervals without an explicit override. The proxy is illustrative, not a recalibrated daily arrival estimate. This daily observation model does not resolve intraday queue position or the number of multiple fills. The stored microstructure fit's one-second scale is not silently treated as a daily arrival rate.

Each side consumes exactly one uniform draw, enabling paired seeded comparisons despite different probabilities. Cash and inventory update at full precision, with equity `cash+inventory*mid`. CSV `pnl` retains the historical equity column name; `profit` and metric `Total PnL` subtract initial cash. Monte Carlo now distinguishes `Final Equity` from `Final PnL`. Drawdown includes initial cash, and Sharpe uses equity returns including the first recorded change.

The default fixed spread matches `MARKET_SPREAD`; it is not automatically the paper's average-AS-spread benchmark. Both strategies use the same fill and accounting rules. Parameters are illustrative proxies. Monte Carlo estimates its price-process moments from the input history, and its positive floor modifies the unrestricted ABM distribution.

## Checks and limitations

```bash
pip install pytest
python -m pytest tests -q
```

The checks cover equations, interval-consistent Poisson probabilities, volatility units, exact cash/inventory identities, initial PnL/drawdown, seeded pairing, malformed inputs and a reproducible two-path synthetic Monte Carlo run. No historical data download, independent SPY recalibration or profitability claim was made during the correction pass. Existing outputs and calibration records are preserved as historical evidence and should be regenerated after deliberately reviewing parameter units.

## Observation protocol

Each row anchors a piecewise-frozen mid and causal volatility estimate. Quotes are held for one interval, with at most one fill per side and no replenishment; fills and marked equity use that frozen mid. Price changes between anchors revalue carried inventory. This is a stylized historical-price replay, not observed daily exchange executions. Result attributes record dt, reference interval, horizon and annualization; CSV users should preserve these separately. Sharpe uses `TRADING_DAYS_PER_YEAR/dt`. Existing generated outputs precede the fixed-clock correction and are preserved as historical artifacts.
