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
