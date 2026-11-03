"""Exercise actual Streamlit routes with deterministic data and no remote calls."""

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from streamlit.util import calc_hash

from common import data
from common.portfolio import DEFAULT_SYMBOLS, analyze_portfolio


@pytest.fixture
def market(monkeypatch):
    rng = np.random.default_rng(4)
    dates = pd.bdate_range(end=date.today(), periods=260)
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(.0005, .012, (260, 5)), axis=0)),
                          index=dates, columns=DEFAULT_SYMBOLS)

    def fetch(symbols, *args, **kwargs):
        return prices.reindex(columns=symbols)

    monkeypatch.setattr(data, 'get_price_data', fetch)
    monkeypatch.setattr(data, 'get_benchmark_data', lambda *a: prices.mean(axis=1))
    return prices


def navigate(app, route):
    # AppTest.switch_page hashes filenames, while st.Page uses explicit URL
    # paths. Match the router's URL hash until AppTest supports those paths.
    app._page_hash = calc_hash(route)
    return app.run()


@pytest.mark.parametrize('route,title', [
    ('portfolio', 'Portfolio Pulse'), ('risk-wall', 'Risk Wall'), ('performance', 'Performance Analytics'),
    ('rebalancing', 'Rebalancing & Optimization'), ('holdings', 'Holdings & Exposure'), ('index-insights', 'Index Insights'),
    ('backtest', 'Backtest'),
])
def test_equity_routes_render_without_exceptions(market, route, title):
    app = AppTest.from_file('streamlit_app.py', default_timeout=20)
    navigate(app, route)
    assert not app.exception
    assert app.title[0].value == title
    assert len(app.metric) >= 3


def test_applied_filters_persist_across_pages_and_invalid_input_never_fetches(market, monkeypatch):
    app = AppTest.from_file('streamlit_app.py', default_timeout=20).run()
    app.text_area(key='_pf_symbols').input('msft, aapl, msft')
    app.selectbox(key='_pf_benchmark').select('NASDAQ')
    app.number_input(key='_pf_rf').set_value(4.0)
    app.button(key='FormSubmitter:portfolio_form-Update analysis').click().run()
    assert app.session_state['portfolio_settings']['symbols'] == ['MSFT', 'AAPL']
    navigate(app, 'risk-wall')
    assert app.text_area(key='_pf_symbols').value.lower().startswith('msft')
    assert app.selectbox(key='_pf_benchmark').value == 'NASDAQ'
    assert app.number_input(key='_pf_rf').value == 4.0
    assert app.session_state['portfolio_settings']['risk_free_rate'] == .04
    monkeypatch.setattr(data, 'get_price_data', lambda *a, **k: pytest.fail('invalid selection fetched data'))
    app.text_area(key='_pf_symbols').input('')
    app.button(key='FormSubmitter:portfolio_form-Update analysis').click().run()
    assert app.error and not app.exception
    assert app.session_state['portfolio_settings']['symbols'] == ['MSFT', 'AAPL']


def test_custom_allocations_drive_the_same_metrics_on_each_dashboard(market):
    weights = dict(zip(DEFAULT_SYMBOLS, [.6, .1, .1, .1, .1]))
    expected = f"{analyze_portfolio(market, weights).stats()['total_return']:.2%}"
    app = AppTest.from_file('streamlit_app.py', default_timeout=20)
    app.session_state['portfolio_weights'] = weights
    for route in ['portfolio', 'risk-wall', 'performance']:
        navigate(app, route)
        assert not app.exception
        assert next(m.value for m in app.metric if m.label == 'Total return') == expected


def test_missing_benchmark_does_not_block_portfolio_analysis(market, monkeypatch):
    monkeypatch.setattr(data, 'get_benchmark_data', lambda *a: pd.Series(dtype=float))
    app = AppTest.from_file('streamlit_app.py', default_timeout=20).run()
    assert not app.exception and app.metric
    assert any('Benchmark prices are unavailable' in i.value for i in app.info)


def test_missing_asset_and_short_windows_are_actionable(market, monkeypatch):
    monkeypatch.setattr(data, 'get_price_data', lambda *a, **k: market.drop(columns=['NVDA']))
    app = AppTest.from_file('streamlit_app.py', default_timeout=20).run()
    assert not app.exception
    assert any('NVDA' in w.value for w in app.warning)
    monkeypatch.setattr(data, 'get_price_data', lambda *a, **k: market.iloc[:2])
    app.run()
    assert not app.exception
    assert any('three' in i.value for i in app.info)


def test_allocation_apply_and_reset(market):
    app = AppTest.from_file('streamlit_app.py', default_timeout=20)
    navigate(app, 'holdings')
    app.button(key='FormSubmitter:allocation_form-Apply allocations').click().run()
    assert not app.exception
    assert sum(app.session_state['portfolio_weights'].values()) == pytest.approx(1)
    app.session_state['portfolio_weights'] = dict(zip(DEFAULT_SYMBOLS, [.6, .1, .1, .1, .1]))
    app.run()
    app.button(key='FormSubmitter:allocation_form-Reset to equal weights').click().run()
    assert not app.exception
    assert next(m.value for m in app.metric if m.label == 'Largest allocation') == '20.00%'


def test_yahoo_option_fallback_keeps_expiry_metadata_and_renders(market, monkeypatch):
    import requests
    import yfinance as yf

    def unavailable(*a, **k):
        raise requests.ConnectionError('NSE unavailable')

    monkeypatch.setattr(requests.Session, 'get', unavailable)
    expiry = (date.today() + timedelta(days=30)).isoformat()
    calls = pd.DataFrame({'strike': [90, 100, 110], 'lastPrice': [12, 5, 1],
                          'impliedVolatility': [.2, .2, .2], 'openInterest': [100, 100, 100], 'volume': [50, 60, 70]})
    ticker = SimpleNamespace(options=[expiry], history=lambda **k: pd.DataFrame({'Close': [100.]}),
                             option_chain=lambda e: SimpleNamespace(calls=calls, puts=calls))
    monkeypatch.setattr(yf, 'Ticker', lambda *a: ticker)
    app = AppTest.from_file('streamlit_app.py', default_timeout=20)
    navigate(app, 'option-chain')
    assert not app.exception
    assert any('fallback' in str(m.value).lower() for m in app.markdown)


def test_backtest_settings_save_and_navigation_preserve_run(market):
    app = AppTest.from_file('streamlit_app.py', default_timeout=30)
    navigate(app, 'backtest')
    assert not app.exception
    initial = app.session_state['backtest_active']['manifest']['run_id']
    app.selectbox(key='_bt_strategy').select('Trend filter')
    app.number_input(key='_bt_lookback').set_value(21)
    app.button(key='FormSubmitter:backtest_form-Run backtest').click().run()
    assert not app.exception
    active = app.session_state['backtest_active']
    assert active['result'].config.strategy == 'Trend filter'
    assert active['manifest']['run_id'] != initial
    next(b for b in app.button if b.label == 'Save run to session').click().run()
    assert not app.exception
    assert len(app.session_state['backtest_saved']) == 1
    run_id = app.session_state['backtest_active']['manifest']['run_id']
    navigate(app, 'performance')
    navigate(app, 'backtest')
    assert not app.exception
    assert app.session_state['backtest_active']['manifest']['run_id'] == run_id
    assert app.selectbox(key='_bt_strategy').value == 'Trend filter'


def test_optimizer_target_handoff_copies_weights_without_changing_model(market):
    app = AppTest.from_file('streamlit_app.py', default_timeout=30)
    navigate(app, 'rebalancing')
    next(b for b in app.button if b.label == 'Backtest this target').click().run()
    assert not app.exception
    assert app.title[0].value == 'Backtest'
    assert app.selectbox(key='_bt_source').value == 'Optimizer target'
    weights = app.session_state['backtest_candidate']['weights']
    assert app.session_state['backtest_active']['manifest']['weights'] == weights
    assert app.session_state['backtest_active']['manifest']['context']['source_estimation_end'] == market.index[-1].date().isoformat()
    assert 'portfolio_weights' not in app.session_state
    assert any('in-sample' in i.value for i in app.info)


def test_backtest_handles_missing_benchmark_and_short_signal_history(market, monkeypatch):
    monkeypatch.setattr(data, 'get_benchmark_data', lambda *a: pd.Series(dtype=float))
    monkeypatch.setattr(data, 'get_price_data', lambda *a, **k: market.iloc[:12])
    app = AppTest.from_file('streamlit_app.py', default_timeout=30)
    navigate(app, 'backtest')
    assert not app.exception
    assert app.session_state['backtest_active']['reference'] is None
    assert any('Benchmark prices are unavailable' in i.value for i in app.info)
    assert any('21 price observations' in i.value for i in app.info)
    assert any('20 complete signal' in i.value for i in app.info)


def test_changing_portfolio_refreshes_backtest_but_keeps_saved_snapshot(market):
    app = AppTest.from_file('streamlit_app.py', default_timeout=30)
    navigate(app, 'backtest')
    next(b for b in app.button if b.label == 'Save run to session').click().run()
    previous = app.session_state['backtest_active']['manifest']['run_id']
    app.text_area(key='_pf_symbols').input('AAPL MSFT')
    app.button(key='FormSubmitter:portfolio_form-Update analysis').click().run()
    assert not app.exception
    assert app.session_state['backtest_active']['manifest']['run_id'] != previous
    assert set(app.session_state['backtest_active']['manifest']['weights']) == {'AAPL', 'MSFT'}
    assert previous in app.session_state['backtest_saved']


def test_optimizer_walk_forward_renders_folds_and_excludes_training_window(market):
    app = AppTest.from_file('streamlit_app.py', default_timeout=30)
    navigate(app, 'rebalancing')
    next(b for b in app.button if b.label == 'Run walk-forward validation').click().run()
    assert not app.exception
    validation = app.session_state['walkforward_active']['validation']
    assert validation.result.equity.index[0] == market.index[126]
    assert len(validation.folds) == 3
    assert any(m.label == 'Forward net return' for m in app.metric)


def test_optimizer_source_and_run_persist_when_navigating_away_and_back(market):
    app = AppTest.from_file('streamlit_app.py', default_timeout=30)
    weights = dict(zip(DEFAULT_SYMBOLS, [.4, .3, .2, .1, 0.]))
    app.session_state['backtest_candidate'] = {'weights': weights, 'label': 'Minimum volatility',
        'estimated_start': market.index[0].date().isoformat(), 'estimated_end': market.index[-1].date().isoformat()}
    app.session_state['backtest_requested_source'] = 'Optimizer target'
    navigate(app, 'backtest')
    assert not app.exception
    previous = app.session_state['backtest_active']['manifest']
    navigate(app, 'performance')
    navigate(app, 'backtest')
    assert not app.exception
    assert app.selectbox(key='_bt_source').value == 'Optimizer target'
    assert app.session_state['backtest_active']['manifest'] == previous
    assert app.session_state['backtest_active']['manifest']['weights'] == weights


def test_backtest_normalizes_benchmark_dates_before_reference_simulation(market, monkeypatch):
    benchmark = market.mean(axis=1)
    benchmark.index = benchmark.index.tz_localize('UTC') + pd.Timedelta(hours=16)
    monkeypatch.setattr(data, 'get_benchmark_data', lambda *a: benchmark)
    app = AppTest.from_file('streamlit_app.py', default_timeout=30)
    navigate(app, 'backtest')
    assert not app.exception
    assert not app.error
    active = app.session_state['backtest_active']
    assert active['reference'] is not None
    assert active['reference'].equity.index.equals(active['result'].equity.index)


def test_saved_run_inputs_and_outputs_survive_provider_revisions(market, monkeypatch):
    import hashlib
    from pandas.testing import assert_frame_equal, assert_series_equal
    app = AppTest.from_file('streamlit_app.py', default_timeout=30)
    navigate(app, 'backtest')
    next(b for b in app.button if b.label == 'Save run to session').click().run()
    original_id = app.session_state['backtest_active']['manifest']['run_id']
    original = app.session_state['backtest_saved'][original_id]
    old_prices, old_equity = original['prices'].copy(), original['equity'].copy()
    revised = market.copy()
    revised.iloc[-10:] *= 1.1
    monkeypatch.setattr(data, 'get_price_data', lambda symbols, *a, **k: revised.reindex(columns=symbols))
    app.run()
    assert not app.exception
    assert app.session_state['backtest_active']['manifest']['run_id'] != original_id
    stored = app.session_state['backtest_saved'][original_id]
    assert_frame_equal(stored['prices'], old_prices)
    assert_series_equal(stored['equity'], old_equity)
    exported = stored['prices'].to_csv(index_label='Date').encode()
    assert hashlib.sha256(exported).hexdigest() == stored['manifest']['price_sha256']
    assert list(app.selectbox(key='_bt_export_saved').options)
    assert stored['equity_export']['Strategy'].equals(old_equity)


def test_saved_run_limit_allows_replacing_same_id_and_clear_resets_collection(market):
    import copy
    app = AppTest.from_file('streamlit_app.py', default_timeout=30)
    navigate(app, 'backtest')
    next(b for b in app.button if b.label == 'Save run to session').click().run()
    run_id = app.session_state['backtest_active']['manifest']['run_id']
    saved = app.session_state['backtest_saved'][run_id]
    # Seed four additional complete snapshots; exercise the limit without five unrelated simulations.
    collection = {run_id: saved, **{f'prior-{i}': copy.deepcopy(saved) for i in range(4)}}
    app.session_state['backtest_saved'] = collection
    next(b for b in app.button if b.label == 'Save run to session').click().run()
    assert not app.exception
    assert len(app.session_state['backtest_saved']) == 5
    app.selectbox(key='_bt_strategy').select('Buy & hold')
    app.button(key='FormSubmitter:backtest_form-Run backtest').click().run()
    next(b for b in app.button if b.label == 'Save run to session').click().run()
    assert any('Five runs are already saved' in warning.value for warning in app.warning)
    assert len(app.session_state['backtest_saved']) == 5
    next(b for b in app.button if b.label == 'Clear saved comparison').click().run()
    assert not app.exception
    assert 'backtest_saved' not in app.session_state


def test_benchmark_enters_second_common_close_while_trend_is_still_warming_up(market):
    app = AppTest.from_file('streamlit_app.py', default_timeout=30)
    navigate(app, 'backtest')
    app.selectbox(key='_bt_strategy').select('Trend filter')
    app.number_input(key='_bt_lookback').set_value(63)
    app.button(key='FormSubmitter:backtest_form-Run backtest').click().run()
    assert not app.exception
    active = app.session_state['backtest_active']
    assert active['reference'].ledger.iloc[0]['Execution date'] == market.index[1]
    assert (active['result'].allocations.iloc[:63]['Cash'] == 1.).all()


def test_changing_universe_invalidates_optimizer_candidate_and_clamps_momentum_holdings(market):
    app = AppTest.from_file('streamlit_app.py', default_timeout=30)
    app.session_state['backtest_candidate'] = {'weights': dict.fromkeys(DEFAULT_SYMBOLS, .2),
                                              'label': 'Equal weight'}
    app.session_state['backtest_requested_source'] = 'Optimizer target'
    navigate(app, 'backtest')
    app.text_area(key='_pf_symbols').input('AAPL')
    app.button(key='FormSubmitter:portfolio_form-Update analysis').click().run()
    assert not app.exception
    assert app.selectbox(key='_bt_source').value == 'Current model'
    assert app.number_input(key='_bt_top_n').value == 1
    assert app.session_state['backtest_active']['manifest']['weights'] == {'AAPL': 1.}


def test_invalid_benchmark_dates_fall_back_to_available_strategy_history(market, monkeypatch):
    benchmark = market.mean(axis=1)
    benchmark.index = pd.DatetimeIndex([*benchmark.index[:-1], pd.NaT])
    monkeypatch.setattr(data, 'get_benchmark_data', lambda *a: benchmark)
    app = AppTest.from_file('streamlit_app.py', default_timeout=30)
    navigate(app, 'backtest')
    assert not app.exception and not app.error
    assert app.session_state['backtest_active']['reference'] is None
    assert any('overlapping benchmark history' in info.value for info in app.info)


def test_nonoverlapping_benchmark_does_not_block_strategy(market, monkeypatch):
    benchmark = market.mean(axis=1)
    benchmark.index = benchmark.index - pd.DateOffset(years=5)
    monkeypatch.setattr(data, 'get_benchmark_data', lambda *a: benchmark)
    app = AppTest.from_file('streamlit_app.py', default_timeout=30)
    navigate(app, 'backtest')
    assert not app.exception and not app.error
    assert app.session_state['backtest_active']['reference'] is None
    assert app.session_state['backtest_active']['result'].equity.index.equals(market.index)


def test_sparse_benchmark_preserves_common_baseline_and_discloses_removed_dates(market, monkeypatch):
    benchmark = market.mean(axis=1).iloc[::3]
    monkeypatch.setattr(data, 'get_benchmark_data', lambda *a: benchmark)
    app = AppTest.from_file('streamlit_app.py', default_timeout=30)
    navigate(app, 'backtest')
    assert not app.exception and not app.error
    active = app.session_state['backtest_active']
    assert active['result'].equity.index.equals(benchmark.index)
    assert active['reference'].equity.index.equals(benchmark.index)
    assert active['result'].equity.iloc[0] == active['reference'].equity.iloc[0] == 10000.
    assert any('Benchmark alignment excluded' in info.value for info in app.info)


def test_walk_forward_output_is_hidden_when_objective_or_cost_assumption_changes(market):
    app = AppTest.from_file('streamlit_app.py', default_timeout=30)
    navigate(app, 'rebalancing')
    next(b for b in app.button if b.label == 'Run walk-forward validation').click().run()
    assert any(metric.label == 'Forward net return' for metric in app.metric)
    app.number_input(key='rebalance_cost_bps').set_value(25.).run()
    assert not app.exception
    assert not any(metric.label == 'Forward net return' for metric in app.metric)
    assert any('Run validation to evaluate' in info.value for info in app.info)


@pytest.fixture
def cached_portfolio_before_date_export(monkeypatch):
    """Model a running worker whose portfolio module predates the date-helper export."""
    import importlib
    import sys
    from types import ModuleType
    import common

    portfolio_module = importlib.import_module('common.portfolio')
    cached = ModuleType('common.portfolio')
    cached.__dict__.update(portfolio_module.__dict__)
    cached.__dict__.pop('normalize_observation_dates', None)
    # Python can retain this namespace when updated page source first imports
    # backtesting. Preserve existing imports and restore them after each case.
    monkeypatch.setitem(sys.modules, 'common.portfolio', cached)
    monkeypatch.setattr(common, 'portfolio', cached)
    monkeypatch.setattr(common, 'backtesting', None, raising=False)
    monkeypatch.delitem(sys.modules, 'common.backtesting', raising=False)
    assert not hasattr(cached, 'normalize_observation_dates')
    return cached


def test_backtesting_imports_with_pre_update_portfolio_still_cached(cached_portfolio_before_date_export):
    import importlib
    backtesting = importlib.import_module('common.backtesting')
    prices = pd.DataFrame({'A': [100., 100., 110., 121.]},
                          index=pd.bdate_range('2024-01-01', periods=4))
    result = backtesting.run_backtest(prices, config=backtesting.BacktestConfig(
        strategy='Buy & hold', commission_bps=0, slippage_bps=0))
    assert result.equity.iloc[-1] == pytest.approx(12100.)
    invalid = prices.copy()
    invalid.index = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', None])
    with pytest.raises(ValueError, match='date'):
        backtesting.run_backtest(invalid)


@pytest.mark.parametrize('entrypoint', ['streamlit_app.py', '1_💼_Portfolio_Pulse.py'])
@pytest.mark.parametrize('route,title', [('rebalancing', 'Rebalancing & Optimization'), ('backtest', 'Backtest')])
def test_cloud_entrypoints_render_with_pre_update_portfolio_cached(
        market, cached_portfolio_before_date_export, entrypoint, route, title):
    app = AppTest.from_file(entrypoint, default_timeout=30)
    navigate(app, route)
    assert not app.exception
    assert app.title[0].value == title
    assert len(app.metric) >= 3
