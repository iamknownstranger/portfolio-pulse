"""Adversarial EGM cases and independent accounting checks for the research engine."""

from dataclasses import replace
import hashlib
import io

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from common.backtesting import (BacktestConfig, decision_dates, rolling_windows, run_backtest,
                                run_manifest, simulate_targets, walk_forward_allocation)


def history(observations=140, seed=3):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(100 * np.exp(rng.normal(.0008, .035, (observations, 3)).cumsum(axis=0)),
                        columns=['A', 'B', 'C'], index=pd.bdate_range('2023-11-20', periods=observations))


@pytest.mark.parametrize('observations', [63, 64, 65, 126, 127, 128])
def test_every_rolling_return_is_accounted_for_including_one_observation_tail(observations):
    equity = pd.Series([100.] * observations + [50.],
                       index=pd.bdate_range('2024-01-01', periods=observations + 1))
    windows = rolling_windows(equity)
    assert windows['Observations'].sum() == observations
    assert (1 + windows['Return']).prod() == pytest.approx(.5)
    assert windows.iloc[-1]['Return'] == pytest.approx(-.5)
    assert windows.iloc[-1]['Max drawdown'] == pytest.approx(-.5)


@pytest.mark.parametrize('index', [pd.date_range('2024-01-01', periods=4, tz='UTC'),
                                   pd.date_range('2024-01-01 16:00', periods=4, tz='America/New_York')])
def test_prices_and_targets_share_temporal_normalization(index):
    prices = pd.DataFrame({'A': [100., 100., 110., 121.]}, index=index)
    targets = pd.DataFrame({'A': [1., np.nan, np.nan, np.nan]}, index=index)
    result = simulate_targets(prices, targets, BacktestConfig(commission_bps=0, slippage_bps=0))
    assert result.equity.iloc[-1] == pytest.approx(12100.)
    assert len(result.ledger) == 1
    assert result.ledger.iloc[0]['Execution date'] == pd.Timestamp('2024-01-02')


def test_missing_price_timestamp_is_rejected_before_cash_accrual():
    prices = history(3)
    prices.index = pd.to_datetime(['2024-01-01', '2024-01-02', None])
    with pytest.raises(ValueError, match='date'):
        run_backtest(prices, config=BacktestConfig(cash_rate=.1))


def test_missing_target_timestamp_is_not_silently_discarded():
    prices = history(3)
    targets = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    targets.iloc[0] = [1., 0., 0.]
    targets.index = pd.to_datetime([None, prices.index[1], prices.index[2]])
    with pytest.raises(ValueError, match='date'):
        simulate_targets(prices, targets, BacktestConfig())


def test_tiny_nonzero_positions_remain_reconstructible_from_execution_ledger():
    prices = pd.DataFrame({'A': [100., 100., 101.], 'B': [100., 100., 102.]},
                          index=pd.bdate_range('2024-01-01', periods=3))
    result = run_backtest(prices, {'A': 1 - 1e-13, 'B': 1e-13},
                          BacktestConfig(initial_capital=1e12, strategy='Buy & hold'))
    assert set(result.ledger['Asset']) == {'A', 'B'}
    for component in ['Commission', 'Slippage']:
        assert result.ledger[component].sum() == pytest.approx(result.costs[component].sum(), rel=1e-14, abs=1e-10)
    units = result.ledger.groupby('Asset')['Units'].sum().reindex(prices.columns, fill_value=0.)
    marked = units * prices.iloc[-1]
    actual = result.allocations.iloc[-1][prices.columns] * result.equity.iloc[-1]
    np.testing.assert_allclose(marked, actual, rtol=1e-14, atol=1e-8)


@pytest.mark.parametrize('strategy', ['Buy & hold', 'Periodic rebalance', 'Trend filter', 'Momentum rotation'])
@pytest.mark.parametrize('seed', [1, 17, 109])
def test_replaying_execution_ledger_reconstructs_every_account_snapshot(strategy, seed):
    prices = history(seed=seed)
    # Include a multi-day gap; interest must use calendar days, not an observation count.
    prices = prices.drop(prices.index[35:39])
    settings = BacktestConfig(strategy=strategy, frequency='Weekly', lookback=10, top_n=2,
                              commission_bps=7., slippage_bps=19., cash_rate=.043)
    result = run_backtest(prices, {'A': .5, 'B': .3, 'C': .2}, settings)
    units = pd.Series(0., index=prices.columns)
    cash = settings.initial_capital
    previous = prices.index[0]
    for timestamp, row in prices.iterrows():
        cash *= (1 + settings.cash_rate) ** ((timestamp - previous).days / 365.25)
        executions = result.ledger[result.ledger['Execution date'] == timestamp]
        for trade in executions.to_dict('records'):
            assert trade['Signal date'] < trade['Execution date']
            assert trade['Notional'] == pytest.approx(abs(trade['Units'] * trade['Reference price']))
            assert trade['Reference price'] == row[trade['Asset']]
            cash -= trade['Units'] * trade['Reference price'] + trade['Commission'] + trade['Slippage']
            units[trade['Asset']] += trade['Units']
        marked = units * row
        expected_equity = cash + marked.sum()
        assert cash >= -1e-7
        assert result.equity.loc[timestamp] == pytest.approx(expected_equity, rel=1e-11, abs=1e-7)
        expected = pd.concat([marked, pd.Series({'Cash': cash})]) / expected_equity
        np.testing.assert_allclose(result.allocations.loc[timestamp], expected, rtol=1e-9, atol=1e-11)
        previous = timestamp
    assert np.isfinite(result.equity).all()
    assert np.allclose(result.allocations.sum(axis=1), 1)
    assert (result.allocations >= -1e-12).all().all()


@pytest.mark.parametrize('strategy', ['Buy & hold', 'Periodic rebalance', 'Trend filter', 'Momentum rotation'])
def test_linear_cost_model_is_invariant_to_capital_and_share_price_scale(strategy):
    prices = history()
    settings = BacktestConfig(strategy=strategy, lookback=10, frequency='Weekly', cash_rate=.03)
    result = run_backtest(prices, config=settings)
    capital_scaled = run_backtest(prices, config=replace(settings, initial_capital=123456.))
    share_scaled = run_backtest(prices.mul(pd.Series({'A': .01, 'B': 10., 'C': 1000.})), config=settings)
    np.testing.assert_allclose(result.equity / settings.initial_capital,
                               capital_scaled.equity / 123456., rtol=1e-11)
    np.testing.assert_allclose(result.equity, share_scaled.equity, rtol=1e-11)
    np.testing.assert_allclose(result.allocations, share_scaled.allocations, atol=1e-11)


def test_partial_investment_and_cash_yield_fund_buys_sells_and_costs():
    prices = pd.DataFrame(100., columns=['A', 'B'], index=pd.date_range('2024-01-01', periods=5))
    targets = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    targets.iloc[0] = [.25, .25]
    targets.iloc[2] = [.1, .5]
    settings = BacktestConfig(commission_bps=10, slippage_bps=90, cash_rate=.1)
    result = simulate_targets(prices, targets, settings)
    prior_cash = 10000 * 1.1 ** (1 / 365.25)
    net = prior_cash / 1.005  # half of post-cost equity invested; 1% per side.
    assert result.equity.iloc[1] == pytest.approx(net)
    assert result.allocations.iloc[1].to_dict() == pytest.approx({'A': .25, 'B': .25, 'Cash': .5})
    assert result.allocations.iloc[3].to_dict() == pytest.approx({'A': .1, 'B': .5, 'Cash': .4})
    rebalance = result.ledger[result.ledger['Execution date'] == prices.index[3]]
    assert set(rebalance['Side']) == {'Buy', 'Sell'}


def test_quarter_and_year_decisions_include_first_observed_close():
    index = pd.to_datetime(['2023-12-29', '2024-01-02', '2024-03-29', '2024-04-01', '2024-04-02'])
    assert decision_dates(index, 'Quarterly').tolist() == [True, True, False, True, False]
    middle_of_month = pd.to_datetime(['2024-01-15', '2024-01-16', '2024-02-02'])
    assert decision_dates(middle_of_month, 'Monthly').tolist() == [True, False, True]


def test_momentum_waits_for_next_schedule_after_warming_up_and_uses_all_tickers():
    prices = pd.DataFrame({'A': [100, 101, 102, 103, 104, 105],
                           'B': [100, 99, 98, 97, 96, 95]},
                          index=pd.to_datetime(['2024-01-25', '2024-01-26', '2024-01-29',
                                                '2024-01-30', '2024-02-01', '2024-02-02']))
    result = run_backtest(prices, {'A': 0., 'B': 1.},
                          BacktestConfig(strategy='Momentum rotation', lookback=2, top_n=2))
    assert (result.allocations.iloc[:-1]['Cash'] == 1.).all()
    assert result.ledger['Asset'].tolist() == ['A']
    assert result.ledger.iloc[0]['Execution date'] == prices.index[-1]
    assert result.targets.iloc[-2].tolist() == [1., 0.]


def test_trend_at_moving_average_stays_cash_and_short_history_cannot_enter():
    prices = pd.DataFrame(100., columns=['A'], index=pd.bdate_range('2024-01-01', periods=8))
    for lookback in [2, 20]:
        result = run_backtest(prices, config=BacktestConfig(strategy='Trend filter', lookback=lookback))
        assert result.ledger.empty
        assert (result.allocations['Cash'] == 1.).all()


def test_recipe_and_input_csv_round_trip_preserve_run_provenance():
    prices = history()
    prices['__benchmark__'] = prices.mean(axis=1)
    weights = pd.Series({'A': .5, 'B': .3, 'C': .2})
    settings = BacktestConfig(strategy='Trend filter', lookback=21)
    first = run_backtest(prices[weights.index], weights, settings)
    manifest = run_manifest(first, prices, weights, 'SPY')
    exported = prices.to_csv(index_label='Date').encode()
    assert hashlib.sha256(exported).hexdigest() == manifest['price_sha256']
    restored = pd.read_csv(io.BytesIO(exported), index_col='Date', parse_dates=True, float_precision='round_trip')
    replay = run_backtest(restored[weights.index], weights, BacktestConfig(**manifest['config']))
    np.testing.assert_allclose(first.equity, replay.equity, rtol=1e-12)
    assert run_manifest(replay, restored, weights, 'SPY')['run_id'] == manifest['run_id']


def test_single_observation_walk_forward_folds_record_previous_holdings_return(monkeypatch):
    from types import SimpleNamespace
    from common import optimization
    prices = pd.DataFrame({'A': [100.] * 21 + [100., 120., 120.],
                           'B': [100.] * 21 + [100., 100., 100.]},
                          index=pd.bdate_range('2024-01-01', periods=24))
    chosen = iter([{'A': 1., 'B': 0.}, {'A': 0., 'B': 1.}, {'A': 1., 'B': 0.}])
    monkeypatch.setattr(optimization, 'optimize_allocation', lambda *a, **k:
                        SimpleNamespace(weights=pd.Series(next(chosen)), method='Test allocation', notice=None))
    result = walk_forward_allocation(prices, training_observations=20, test_observations=1,
                                     config=BacktestConfig(commission_bps=0, slippage_bps=0))
    assert result.folds['Test observations'].tolist() == [1, 1, 1]
    assert result.folds['Return'].tolist() == pytest.approx([0., .2, 0.])
    assert result.result.equity.iloc[-1] == pytest.approx(12000.)
    assert result.result.ledger['Execution date'].tolist() == [prices.index[21], prices.index[22], prices.index[22], prices.index[23], prices.index[23]]


def test_rebalance_between_existing_holdings_has_independent_two_sided_fee_oracle():
    prices = pd.DataFrame(100., columns=['A', 'B'], index=pd.bdate_range('2024-01-01', periods=4))
    targets = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    targets.iloc[0] = [1., 0.]
    targets.iloc[1] = [0., 1.]
    result = simulate_targets(prices, targets, BacktestConfig(commission_bps=25, slippage_bps=75))
    after_entry = 10000 / 1.01
    after_rotation = after_entry * .99 / 1.01
    assert result.equity.iloc[1] == pytest.approx(after_entry)
    assert result.equity.iloc[-1] == pytest.approx(after_rotation)
    rotation = result.ledger[result.ledger['Execution date'] == prices.index[2]]
    assert rotation['Notional'].sum() == pytest.approx(after_entry + after_rotation)
    assert result.costs.sum().sum() == pytest.approx(10000 - after_rotation)


def test_walk_forward_equal_weight_comparator_has_identical_nonzero_costs_and_timing():
    result = walk_forward_allocation(history(100), method='Equal weight', max_weight=.4,
        training_observations=30, test_observations=21,
        config=BacktestConfig(commission_bps=17, slippage_bps=43, cash_rate=.07))
    assert_series_equal(result.result.equity, result.equal_weight.equity)
    assert_frame_equal(result.result.ledger, result.equal_weight.ledger)
    assert_frame_equal(result.result.costs, result.equal_weight.costs)
    assert result.result.costs.sum().sum() > 0
    assert result.result.ledger.iloc[0]['Signal date'] == result.folds.iloc[0]['Decision / train end']
    assert result.result.ledger.iloc[0]['Execution date'] == result.folds.iloc[0]['First execution']


def test_daily_reset_at_unchanged_target_does_not_fabricate_roundoff_executions():
    prices = pd.DataFrame(100., columns=['A'], index=pd.bdate_range('2024-01-01', periods=10))
    result = run_backtest(prices, config=BacktestConfig(frequency='Daily'))
    assert len(result.ledger) == 1
    assert (result.costs.iloc[2:] == 0).all().all()


def test_duplicate_unsorted_date_inputs_follow_the_same_last_observation_rule():
    dates = pd.to_datetime(['2024-01-03', '2024-01-01', '2024-01-02', '2024-01-02', '2024-01-04'])
    prices = pd.DataFrame({'A': [120., 100., 999., 100., 132.]}, index=dates)
    targets = pd.DataFrame({'A': [np.nan, 1., 0., np.nan, np.nan]}, index=dates)
    result = simulate_targets(prices, targets, BacktestConfig(commission_bps=0, slippage_bps=0))
    assert result.equity.iloc[-1] == pytest.approx(13200.)
    assert result.equity.index.is_unique and result.equity.index.is_monotonic_increasing
    assert len(result.ledger) == 1


def test_roundoff_tolerance_cannot_create_unfunded_target_equity():
    prices = pd.DataFrame(100., columns=['A'], index=pd.date_range('2024-01-01', periods=3))
    targets = pd.DataFrame({'A': [1 + 5e-11, np.nan, np.nan]}, index=prices.index)
    result = simulate_targets(prices, targets,
                              BacktestConfig(initial_capital=1e12, commission_bps=0, slippage_bps=0))
    assert result.equity.iloc[-1] == result.equity.iloc[0]
    assert result.allocations.iloc[-1]['A'] <= 1


@pytest.mark.parametrize('columns', [['A'], ['A', 'B', 'Unknown'], ['A', 'A']])
def test_target_assets_cannot_be_silently_dropped_or_duplicated(columns):
    prices = pd.DataFrame(100., columns=['A', 'B'], index=pd.date_range('2024-01-01', periods=3))
    targets = pd.DataFrame(0., columns=columns, index=prices.index)
    targets.iloc[0, 0] = 1
    with pytest.raises(ValueError, match='asset'):
        simulate_targets(prices, targets, BacktestConfig())
