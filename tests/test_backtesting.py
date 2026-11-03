"""Independent accounting, timing and diagnostic contracts for daily backtests."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from common.backtesting import (BacktestConfig, decision_dates, drawdown_episodes, rolling_windows,
                                run_backtest, run_manifest, signal_deciles, signal_observations,
                                simulate_targets, validation_windows)


@pytest.fixture
def prices():
    return pd.DataFrame({"A": [100., 100., 120., 90., 108.], "B": [100., 100., 80., 120., 120.]},
                        index=pd.bdate_range("2024-01-02", periods=5))


def config(**kwargs):
    return replace(BacktestConfig(commission_bps=0, slippage_bps=0), **kwargs)


def test_buy_hold_drifts_and_initial_entry_has_one_close_delay(prices):
    result = run_backtest(prices, {"A": .75, "B": .25}, config(strategy="Buy & hold"))
    expected = pd.Series([10000., 10000., 11000., 9750., 11100.], index=prices.index, name="Strategy")
    assert_series_equal(result.equity, expected)
    assert len(result.ledger) == 2
    assert (result.ledger["Signal date"] == prices.index[0]).all()
    assert (result.ledger["Execution date"] == prices.index[1]).all()
    assert result.allocations.iloc[2]["A"] == pytest.approx(9000 / 11000)


def test_future_price_jump_is_not_earned_before_initial_fill(prices):
    prices.iloc[1] *= 10
    result = run_backtest(prices, config=config(strategy="Buy & hold"))
    assert result.equity.iloc[1] == 10000
    assert result.equity.iloc[2] == pytest.approx(1000)
    assert result.stats()["max_drawdown"] <= -.9


def test_daily_rebalancing_matches_hand_compounded_drift_returns(prices):
    result = run_backtest(prices, config=config(frequency="Daily"))
    # First interval is cash; thereafter a 50/50 reset at the preceding execution close.
    expected = 10000 * (1 + prices.pct_change().iloc[2:].mean(axis=1)).prod()
    assert result.equity.iloc[-1] == pytest.approx(expected)
    assert np.allclose(result.allocations.iloc[1:][["A", "B"]], .5)


def test_entry_and_exit_costs_are_self_funded_on_both_sides(prices):
    prices[:] = 100.
    targets = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    targets.iloc[0] = [.5, .5]
    targets.iloc[1] = [0., 0.]
    result = simulate_targets(prices, targets, config(commission_bps=10, slippage_bps=90))
    invested = 10000 / 1.01
    final = invested * .99
    assert result.equity.iloc[1] == pytest.approx(invested)
    assert result.equity.iloc[-1] == pytest.approx(final)
    assert result.costs.sum().sum() == pytest.approx(10000 - final)
    assert result.ledger["Commission"].sum() == pytest.approx(result.costs["Commission"].sum())
    assert result.ledger["Slippage"].sum() == pytest.approx(result.costs["Slippage"].sum())
    assert set(result.ledger["Side"]) == {"Buy", "Sell"}
    assert (result.allocations >= -1e-12).all().all()
    assert np.allclose(result.allocations.sum(axis=1), 1)


def test_cash_yield_uses_elapsed_calendar_days_and_last_signal_cannot_fill(prices):
    targets = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    targets.iloc[-1] = [.5, .5]
    result = simulate_targets(prices, targets, config(cash_rate=.1))
    elapsed = (prices.index[-1] - prices.index[0]).days
    assert result.equity.iloc[-1] == pytest.approx(10000 * 1.1 ** (elapsed / 365.25))
    assert result.ledger.empty
    assert (result.allocations["Cash"] == 1).all()


def test_calendar_schedule_uses_first_observation_of_new_period():
    index = pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02", "2024-02-05"])
    assert decision_dates(index, "Monthly").tolist() == [True, False, True, False, False]
    assert decision_dates(index, "Weekly").tolist() == [True, False, False, False, True]


@pytest.mark.parametrize("strategy", ["Trend filter", "Momentum rotation"])
def test_future_mutation_cannot_change_past_signals_or_execution(strategy):
    rng = np.random.default_rng(7)
    history = pd.DataFrame(100 * np.exp(rng.normal(.002, .02, (90, 3)).cumsum(axis=0)),
                           index=pd.bdate_range("2024-01-01", periods=90), columns=list("ABC"))
    settings = config(strategy=strategy, lookback=5, frequency="Daily", top_n=2)
    first = run_backtest(history, config=settings)
    history.iloc[50:] *= np.arange(1, 41)[:, None] * 10
    changed = run_backtest(history, config=settings)
    assert_series_equal(first.equity.iloc[:50], changed.equity.iloc[:50])
    assert_frame_equal(first.targets.iloc[:50], changed.targets.iloc[:50])
    assert_frame_equal(first.signals.iloc[:50], changed.signals.iloc[:50])
    assert (first.allocations.iloc[:5]["Cash"] == 1).all()


def test_positive_momentum_selects_stably_and_never_borrows():
    history = pd.DataFrame({"A": [100, 110, 121, 133.1], "B": [100, 110, 121, 133.1], "C": [100, 90, 81, 72.9]},
                           index=pd.bdate_range("2024-01-01", periods=4))
    result = run_backtest(history, config=config(strategy="Momentum rotation", lookback=2, top_n=1, frequency="Daily"))
    assert result.targets.iloc[2].tolist() == [1., 0., 0.]
    assert result.ledger["Asset"].tolist() == ["A"]
    assert result.ledger.iloc[0]["Execution date"] == history.index[3]


def test_missing_common_dates_are_removed_without_filling(prices):
    prices.iloc[2, 1] = np.nan
    result = run_backtest(prices, config=config(strategy="Buy & hold"))
    assert result.excluded_rows == 1
    assert prices.index[2] not in result.equity.index
    assert result.equity.iloc[-1] == pytest.approx(11400.)


@pytest.mark.parametrize("bad", [{"initial_capital": -1}, {"commission_bps": np.inf}, {"slippage_bps": -1},
                                  {"cash_rate": -1}, {"lookback": 1}, {"strategy": "unknown"}])
def test_invalid_configuration_is_rejected(prices, bad):
    with pytest.raises(ValueError):
        run_backtest(prices, config=config(**bad))


@pytest.mark.parametrize("target", [[.8, .8], [-.1, .5], [np.nan, .5], [np.inf, 0.]])
def test_invalid_target_allocations_are_rejected(prices, target):
    targets = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    targets.iloc[0] = target
    with pytest.raises(ValueError):
        simulate_targets(prices, targets, config())


def test_signal_outcomes_begin_after_execution_and_drop_unknown_future():
    history = pd.DataFrame({"A": [100., 110., 121., 133.1, 159.72, 143.748, 172.4976]},
                           index=pd.bdate_range("2024-01-01", periods=7))
    observations = signal_observations(history, "A", lookback=2, horizon=1)
    first = observations.iloc[0]
    assert observations.index[0] == history.index[2]
    assert first["Signal"] == pytest.approx(.21)
    assert first["Forward return"] == pytest.approx(.20)  # index 3 -> 4, not decision 2 -> 3
    assert observations.index[-1] == history.index[-3]
    assert signal_deciles(observations).empty


def test_tied_and_constant_signals_have_honest_buckets():
    observations = pd.DataFrame({"Signal": [1.] * 20 + [2.] * 20, "Forward return": np.arange(40) / 1000})
    buckets = signal_deciles(observations)
    assert buckets["Observations"].sum() == 40
    assert len(buckets) <= 10
    observations["Signal"] = 1
    assert signal_deciles(observations).empty


def test_validation_splits_returns_without_overlap_or_missing_initial_losses():
    equity = pd.Series(np.linspace(100, 50, 101), index=pd.bdate_range("2024-01-01", periods=101))
    table = validation_windows(equity)
    assert table["Observations"].sum() == 100
    assert table.iloc[0]["End"] == table.iloc[1]["Start"]
    assert (1 + table["Return"]).prod() == pytest.approx(.5)
    assert (table["Max drawdown"] < 0).all()
    windows = rolling_windows(equity, window=30)
    assert windows["Observations"].sum() == 100
    assert (1 + windows["Return"]).prod() == pytest.approx(.5)


def test_drawdown_episodes_include_unrecovered_initial_loss_and_recovery():
    wealth = pd.Series([1, .8, .9, 1, 1.2, .6, .9], index=pd.date_range("2024-01-01", periods=7))
    episodes = drawdown_episodes(wealth)
    assert len(episodes) == 2
    assert episodes.iloc[0]["Drawdown"] == pytest.approx(-.5)
    assert pd.isna(episodes.iloc[0]["Recovery"])
    assert episodes.iloc[1]["Peak"] == wealth.index[0]
    assert episodes.iloc[1]["Recovery"] == wealth.index[3]
    assert episodes.iloc[1]["Days underwater"] == 3


def test_manifest_fingerprints_data_and_execution_assumptions(prices):
    import hashlib
    result = run_backtest(prices, config=config())
    weights = pd.Series({"A": .5, "B": .5})
    manifest = run_manifest(result, prices, weights, "SPY")
    assert manifest['price_sha256'] == hashlib.sha256(prices.to_csv(index_label='Date').encode()).hexdigest()
    assert manifest == run_manifest(result, prices, weights, "SPY")
    revised = prices.copy()
    revised.iloc[-1, 0] += 1
    assert run_manifest(result, revised, weights, "SPY")["run_id"] != manifest["run_id"]
    assert run_manifest(result, prices, weights, "QQQ")["run_id"] != manifest["run_id"]


def test_walk_forward_refits_only_on_past_data_and_compounds_disjoint_folds(monkeypatch):
    from common.backtesting import walk_forward_allocation
    from common import optimization

    rng = np.random.default_rng(17)
    history = pd.DataFrame(100 * np.exp(rng.normal(.001, .01, (100, 2)).cumsum(axis=0)),
                           index=pd.bdate_range('2024-01-01', periods=100), columns=['A', 'B'])
    actual = optimization.optimize_allocation
    training_ranges = []

    def record(prices, *args, **kwargs):
        training_ranges.append((prices.index[0], prices.index[-1]))
        return actual(prices, *args, **kwargs)

    monkeypatch.setattr(optimization, 'optimize_allocation', record)
    result = walk_forward_allocation(history, 'Minimum volatility', .7, training_observations=30,
                                     test_observations=20, config=config())
    assert len(training_ranges) == 4
    assert result.result.equity.index[0] == history.index[30]
    for (start, end), row in zip(training_ranges, result.folds.to_dict('records')):
        assert start == row['Train start']
        assert end == row['Decision / train end'] < row['First execution']
    assert result.folds['Test observations'].sum() == 69
    assert (1 + result.folds['Return']).prod() == pytest.approx(result.result.equity.iloc[-1] / 10000)
    assert (result.weights <= .7 + 1e-6).all().all()
    history.iloc[51:] *= 10
    changed = walk_forward_allocation(history, 'Minimum volatility', .7, training_observations=30,
                                      test_observations=20, config=config())
    assert_frame_equal(result.weights.iloc[:2], changed.weights.iloc[:2])
    assert_series_equal(result.result.equity.loc[:history.index[50]], changed.result.equity.loc[:history.index[50]])
