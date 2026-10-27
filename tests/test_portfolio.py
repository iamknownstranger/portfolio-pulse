from datetime import date

import numpy as np
import pandas as pd
import pytest

from common.portfolio import (
    analyze_portfolio, compare_with_benchmark, drawdowns, normalize_symbols,
    normalized_weights, performance_stats, rebalance_plan, relative_stats,
)


def prices(values, columns=None):
    return pd.DataFrame(values, index=pd.bdate_range('2025-01-02', periods=len(values)), columns=columns or ['A', 'B'])


def test_symbols_are_stable_and_support_etfs_class_shares_and_exchange_suffixes():
    assert normalize_symbols(' aapl, MSFT; aapl\n BRK.B SPY RELIANCE.NS ^GSPC BTC-USD ') == [
        'AAPL', 'MSFT', 'BRK-B', 'SPY', 'RELIANCE.NS', '^GSPC', 'BTC-USD']


@pytest.mark.parametrize('value', ['', 'AAPL <script>', 'AAPL /tmp/file', ','.join(f'T{i}' for i in range(31))])
def test_invalid_symbol_selections_are_rejected(value):
    with pytest.raises(ValueError):
        normalize_symbols(value)


@pytest.mark.parametrize('weights', [{'A': .5}, {'A': -.2, 'B': 1.2}, {'A': np.nan, 'B': .5}, {'A': .5, 'B': .4}])
def test_invalid_allocations_do_not_silently_change_the_model(weights):
    with pytest.raises(ValueError):
        normalized_weights(['A', 'B'], weights)


def test_daily_rebalancing_and_linked_contributions_match_the_actual_wealth_path():
    portfolio = analyze_portfolio(prices([[100, 100], [200, 100], [100, 200]]))
    assert portfolio.wealth.tolist() == pytest.approx([1, 1.5, 1.875])
    assert portfolio.contributions().sum() == pytest.approx(.875)
    assert portfolio.contributions()['A'] == pytest.approx(.125)
    assert portfolio.contributions()['B'] == pytest.approx(.75)


def test_price_scale_does_not_determine_equal_weight_allocation():
    frame = prices([[10, 500], [11, 450], [12, 400]])
    original = analyze_portfolio(frame)
    frame['B'] *= 100
    pd.testing.assert_series_equal(original.wealth, analyze_portfolio(frame).wealth)


def test_first_loss_is_included_in_drawdown_and_total_return():
    portfolio = analyze_portfolio(prices([[100], [80], [90]], ['A']))
    assert portfolio.stats()['max_drawdown'] == pytest.approx(-.2)
    assert drawdowns(portfolio.wealth).iloc[1] == pytest.approx(-.2)
    assert portfolio.stats()['total_return'] == pytest.approx(-.1)


def test_common_calendar_does_not_fill_missing_prices_or_change_asset_weights():
    portfolio = analyze_portfolio(prices([[100, 100], [110, np.nan], [120, 80], [130, 90]]), {'A': .8, 'B': .2})
    assert portfolio.excluded_rows == 1
    assert portfolio.returns.iloc[0] == pytest.approx(.8 * .2 + .2 * -.2)
    assert portfolio.weights.to_dict() == {'A': .8, 'B': .2}


@pytest.mark.parametrize('frame', [pd.DataFrame(), prices([[100, 100]]), prices([[100, None], [101, None], [102, None]])])
def test_insufficient_history_is_actionable(frame):
    with pytest.raises(ValueError):
        analyze_portfolio(frame)


def test_benchmark_alignment_happens_before_return_calculation():
    frame = prices([[100], [200], [220], [242]], ['A'])
    frame.index = frame.index.date  # mixed index types used to yield empty joins
    bench = pd.Series([100, 110, 121], index=pd.to_datetime(frame.index[1:]))
    portfolio, benchmark_returns, benchmark_wealth = compare_with_benchmark(frame, bench)
    assert portfolio.wealth.iloc[0] == benchmark_wealth.iloc[0] == 1
    assert portfolio.wealth.iloc[-1] == pytest.approx(1.21)
    assert benchmark_returns.tolist() == pytest.approx([.1, .1])


def test_zero_volatility_ratios_are_unavailable_instead_of_infinite():
    stats = performance_stats(pd.Series([0., 0., 0.]))
    assert np.isnan(stats['sharpe'])
    assert np.isnan(stats['sortino'])
    assert stats['max_drawdown'] == 0


def test_var_horizons_and_sortino_denominator():
    returns = pd.Series([-.02, .01, .01, .03])
    stats = performance_stats(returns)
    daily_sigma = np.sqrt(np.average(returns ** 2, weights=.94 ** np.arange(3, -1, -1)))
    assert stats['ewma_var_95'] == pytest.approx(1.6448536269514722 * daily_sigma)
    downside = np.sqrt(.02 ** 2 / 4) * np.sqrt(252)
    assert stats['sortino'] == pytest.approx(returns.mean() * 252 / downside)
    assert performance_stats(pd.Series([.01, .02]))['var_95'] == 0


def test_matching_benchmark_has_beta_one_and_alpha_zero():
    returns = pd.Series([.01, -.02, .04, -.01])
    metrics = relative_stats(returns, returns, .04)
    assert metrics['beta'] == pytest.approx(1)
    assert metrics['alpha'] == pytest.approx(0)
    assert metrics['tracking_error'] == 0
    assert np.isnan(metrics['information_ratio'])


def test_rebalance_plan_is_self_financing_and_preserves_totals():
    plan = rebalance_plan({'A': .8, 'B': .2}, {'A': .5, 'B': .5}, 10000)
    assert plan['Trade value ($)'].sum() == pytest.approx(0)
    assert plan.loc['A', 'Trade value ($)'] == pytest.approx(-3000)
    assert plan.loc['B', 'Action'] == 'Buy'
    assert plan['Target weight'].sum() == pytest.approx(1)


@pytest.mark.parametrize('value', [0, -1, np.nan])
def test_rebalance_requires_positive_value(value):
    with pytest.raises(ValueError):
        rebalance_plan({'A': 1}, {'A': 1}, value)
