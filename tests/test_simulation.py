import numpy as np
import pandas as pd
import pytest

from config import GAMMA,KAPPA,INITIAL_CASH,OBSERVATION_DT,MARKET_SPREAD,FIXED_SPREAD
from src.model import reservation_price,optimal_spread,fill_probability,calibration_time_step
from src.simulation import run_as_simulation
from src.baseline import run_baseline_simulation
from src.metrics import compute_metrics
from src.monte_carlo import run_monte_carlo,_simulation_metrics_row,simulate_abm_price_path


def inputs(n=8):
    index=pd.date_range('2020-01-01',periods=n,freq='B')
    return pd.Series(np.linspace(99,101,n),index=index),pd.Series(.001,index=index)


def test_as_equations_include_variance_width_and_center_on_reservation():
    mid,q,sigma,tau=100,2,3,.4
    db,da=optimal_spread(q,sigma,tau)
    r=reservation_price(mid,q,sigma,tau)
    assert (mid-db+mid+da)/2==pytest.approx(r)
    assert db+da==pytest.approx(GAMMA*sigma**2*tau+2*np.log1p(GAMMA/KAPPA)/GAMMA)
    assert r==pytest.approx(mid-q*GAMMA*sigma**2*tau)


def test_poisson_fill_probability_signed_distances_and_interval_consistency():
    aggressive=fill_probability(-.1,1.)
    passive=fill_probability(.1,1.)
    assert 0<=passive<aggressive<=1
    p=fill_probability(.01,1.,reference_dt=1.)
    split=fill_probability(.01,.1,reference_dt=1.)
    assert 1-(1-split)**10==pytest.approx(p)
    assert fill_probability(-1e5,1.)==1
    assert fill_probability(1e5,1.)==0


@pytest.mark.parametrize('steps,horizon',[(0,1),(1,0),(1,np.nan),(1.5,1)])
def test_invalid_clocks_rejected(steps,horizon):
    with pytest.raises(ValueError):calibration_time_step(steps,horizon)


@pytest.mark.parametrize('runner',[run_as_simulation,run_baseline_simulation])
def test_cash_inventory_equity_and_exact_precision_agree(runner):
    prices,vol=inputs()
    a=runner(prices,vol,rng=np.random.default_rng(505))
    b=runner(prices,vol,rng=np.random.default_rng(505))
    pd.testing.assert_frame_equal(a,b)
    expected_cash=INITIAL_CASH+(a.ask_hit*a.ask-a.bid_hit*a.bid).cumsum()
    expected_inventory=(a.bid_hit-a.ask_hit).cumsum()
    np.testing.assert_allclose(a.cash,expected_cash,rtol=0,atol=1e-10)
    np.testing.assert_array_equal(a.inventory,expected_inventory)
    np.testing.assert_allclose(a.pnl,a.cash+a.inventory*a.mid_price,rtol=0,atol=1e-10)
    np.testing.assert_allclose(a.profit,a.pnl-INITIAL_CASH)
    assert ((a.bid>0)&(a.bid<a.mid_price)&(a.ask>a.mid_price)).all()
    assert a.attrs['initial_cash']==INITIAL_CASH


def test_return_volatility_is_converted_to_price_and_model_time_units():
    prices,vol=inputs()
    a=run_as_simulation(prices,vol,np.random.default_rng(505))
    np.testing.assert_allclose(a.price_volatility,prices*vol/np.sqrt(OBSERVATION_DT))


def test_dataset_length_does_not_recalibrate_clock_or_baseline_hazard():
    prices,vol=inputs(12)
    full=run_baseline_simulation(prices,vol,np.random.default_rng(505))
    prefix=run_baseline_simulation(prices.iloc[:5],vol.iloc[:5],np.random.default_rng(505))
    pd.testing.assert_frame_equal(full.iloc[:5].drop(columns="time_remaining"),prefix.drop(columns="time_remaining"))
    assert full.attrs['dt']==prefix.attrs['dt']==OBSERVATION_DT
    assert full.attrs['reference_dt']==prefix.attrs['reference_dt']
    assert full.attrs['horizon']==12*OBSERVATION_DT
    assert prefix.attrs['horizon']==5*OBSERVATION_DT


def test_default_probability_reference_is_fixed_when_subdividing_time():
    p=fill_probability(.01,1.)
    sub=fill_probability(.01,.1)
    assert 1-(1-sub)**10==pytest.approx(p)


def test_metric_annualization_uses_observation_clock():
    prices,vol=inputs(12)
    frame=run_baseline_simulation(prices,vol,np.random.default_rng(505))
    a=compute_metrics(frame,periods_per_year=252.)
    b=compute_metrics(frame,periods_per_year=1008.)
    assert b['Sharpe Ratio']==pytest.approx(2*a['Sharpe Ratio'],abs=.00015)
    with pytest.raises(ValueError,match='Annualization'):compute_metrics(frame,periods_per_year=0)


@pytest.mark.parametrize('runner',[run_as_simulation,run_baseline_simulation])
@pytest.mark.parametrize('kind',['empty','negative','nan','duplicate','missing-vol-date'])
def test_invalid_inputs_fail_explicitly(runner,kind):
    prices,vol=inputs()
    if kind=='empty':vol=vol.iloc[:0]
    elif kind=='negative':prices.iloc[1]=-1
    elif kind=='nan':vol.iloc[1]=np.nan
    elif kind=='duplicate':prices.index=[prices.index[0]]*len(prices)
    else:vol.index=vol.index+pd.Timedelta(days=50)
    with pytest.raises(ValueError):runner(prices,vol)


def test_metrics_include_initial_change_and_drawdown():
    frame=pd.DataFrame({'pnl':[100001.,100003.],'inventory':[1,2],'spread':[.1,.1],'bid_hit':[1,1],'ask_hit':[0,0]})
    assert compute_metrics(frame)['Total PnL']==3
    frame.pnl=[99999.,99998.]
    assert compute_metrics(frame)['Max Drawdown']==-2
    frame=frame.iloc[:1]
    assert compute_metrics(frame)['Inventory Std']==0
    assert compute_metrics(frame)['Sharpe Ratio']==0
    assert _simulation_metrics_row(frame,1,'AS')['Final PnL']==-1


def test_paired_uniform_draws_identical_for_same_quote_function():
    # Zero return volatility removes inventory skew; compare the effective same quotes.
    prices,vol=inputs();vol[:]=0
    spread=2*np.log1p(GAMMA/KAPPA)/GAMMA
    a=run_as_simulation(prices,vol,np.random.default_rng(5))
    b=run_baseline_simulation(prices,vol,np.random.default_rng(5),fixed_spread=spread)
    pd.testing.assert_frame_equal(a[['bid','ask','p_bid','p_ask','bid_hit','ask_hit','cash','pnl']],b[['bid','ask','p_bid','p_ask','bid_hit','ask_hit','cash','pnl']])
    assert FIXED_SPREAD==MARKET_SPREAD


def test_short_synthetic_monte_carlo_reproducible_and_reports_profit_not_equity():
    rng=np.random.default_rng(2);prices,vol=inputs(50)
    prices+=np.cumsum(rng.normal(0,.1,len(prices)))
    a=run_monte_carlo(prices,n_simulations=2,seed=505)
    b=run_monte_carlo(prices,n_simulations=2,seed=505)
    pd.testing.assert_frame_equal(a['as_runs'],b['as_runs'])
    for name in ['as_runs','baseline_runs']:
        np.testing.assert_allclose(a[name]['Final PnL'],a[name]['Total PnL'])
        np.testing.assert_allclose(a[name]['Final Equity']-INITIAL_CASH,a[name]['Final PnL'],atol=.005)
    with pytest.raises(ValueError):run_monte_carlo(prices,n_simulations=0)
    with pytest.raises(ValueError):simulate_abm_price_path(prices.iloc[:2],rng)
