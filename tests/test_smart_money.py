import numpy as np
import pandas as pd
import pytest

from common.smart_money import (
    STAGE_ACCUMULATION,
    STAGE_CROWDED,
    STAGE_DORMANT,
    STAGE_EARLY,
    STAGE_MOMENTUM,
    breadth_above_ma,
    build_theme_signals,
    chaikin_money_flow,
    classify_stage,
    constituent_table,
    cross_sectional_z,
    diff_stages,
    dollar_volume_thrust,
    equal_weight_index,
    pct_from_high,
    relative_strength,
    rotation_matrix,
    rs_acceleration,
    squash,
    truncate_frames,
)

BARS = 400


def _calendar(n=BARS):
    return pd.bdate_range("2024-01-01", periods=n)


def _frames(drift_per_bar, n=BARS, tickers=("AAA", "BBB", "CCC"),
            volume=1_000_000, volume_multiplier=1.0, range_pct=0.02,
            close_position=0.5):
    """Synthetic OHLCV frames with a controllable trend and bar placement.

    close_position 1.0 puts the close at the bar high (accumulation),
    0.0 at the low (distribution).
    """
    index = _calendar(n)
    close = pd.DataFrame(
        {t: 100 * (1 + drift_per_bar) ** np.arange(n) for t in tickers}, index=index
    )
    span = close * range_pct
    low = close - span * close_position
    high = low + span
    volumes = pd.DataFrame({t: float(volume) for t in tickers}, index=index)
    if volume_multiplier != 1.0:
        volumes.iloc[-20:] *= volume_multiplier
    return {"Close": close, "High": high, "Low": low, "Volume": volumes}


def _benchmark(drift_per_bar=0.0, n=BARS):
    return pd.Series(
        100 * (1 + drift_per_bar) ** np.arange(n), index=_calendar(n), dtype=float
    )


class TestEqualWeightIndex:
    def test_tracks_a_uniform_basket(self):
        close = _frames(0.001)["Close"]
        index = equal_weight_index(close)
        assert index.iloc[-1] == pytest.approx((1.001) ** (BARS - 1), rel=1e-6)

    def test_averages_divergent_constituents(self):
        index = pd.bdate_range("2024-01-01", periods=3)
        close = pd.DataFrame({"UP": [100.0, 110.0, 121.0], "FLAT": [50.0, 50.0, 50.0]}, index=index)
        # Daily returns average (10% + 0%)/2 = 5% per bar.
        assert equal_weight_index(close).iloc[-1] == pytest.approx(1.05 ** 2)

    def test_late_listing_does_not_step_the_index(self):
        index = pd.bdate_range("2024-01-01", periods=4)
        close = pd.DataFrame(
            {"OLD": [100.0, 101.0, 102.0, 103.0], "NEW": [np.nan, np.nan, 500.0, 505.0]},
            index=index,
        )
        values = equal_weight_index(close)
        # The new listing joins via its return, so no jump when it appears.
        assert values.is_monotonic_increasing
        assert values.max() < 1.1

    def test_empty_input_returns_empty(self):
        assert equal_weight_index(pd.DataFrame()).empty


class TestChaikinMoneyFlow:
    def test_close_at_high_is_full_accumulation(self):
        frames = _frames(0.0, close_position=1.0)
        cmf = chaikin_money_flow(frames["High"], frames["Low"], frames["Close"], frames["Volume"])
        assert cmf.iloc[-1].mean() == pytest.approx(1.0)

    def test_close_at_low_is_full_distribution(self):
        frames = _frames(0.0, close_position=0.0)
        cmf = chaikin_money_flow(frames["High"], frames["Low"], frames["Close"], frames["Volume"])
        assert cmf.iloc[-1].mean() == pytest.approx(-1.0)

    def test_midrange_close_is_neutral(self):
        frames = _frames(0.0, close_position=0.5)
        cmf = chaikin_money_flow(frames["High"], frames["Low"], frames["Close"], frames["Volume"])
        assert cmf.iloc[-1].mean() == pytest.approx(0.0, abs=1e-9)

    def test_zero_range_bars_do_not_divide_by_zero(self):
        index = pd.bdate_range("2024-01-01", periods=30)
        flat = pd.DataFrame({"A": [100.0] * 30}, index=index)
        volume = pd.DataFrame({"A": [1000.0] * 30}, index=index)
        cmf = chaikin_money_flow(flat, flat, flat, volume)
        assert not np.isinf(cmf.to_numpy(dtype=float)).any()
        assert cmf["A"].dropna().empty

    def test_empty_input_returns_empty(self):
        empty = pd.DataFrame()
        assert chaikin_money_flow(empty, empty, empty, empty).empty


class TestDollarVolumeThrust:
    def test_doubled_recent_volume_reads_as_plus_one(self):
        frames = _frames(0.0, volume_multiplier=2.0)
        assert dollar_volume_thrust(frames["Close"], frames["Volume"]) == pytest.approx(1.0)

    def test_steady_volume_reads_as_zero(self):
        frames = _frames(0.0)
        assert dollar_volume_thrust(frames["Close"], frames["Volume"]) == pytest.approx(0.0)

    def test_short_history_returns_nan(self):
        frames = _frames(0.0, n=15)
        assert np.isnan(dollar_volume_thrust(frames["Close"], frames["Volume"]))


class TestRelativeStrength:
    def test_outperformance_is_positive(self):
        theme, bench = equal_weight_index(_frames(0.002)["Close"]), _benchmark(0.0)
        assert relative_strength(theme, bench, 63) > 0

    def test_matching_the_benchmark_is_zero(self):
        theme, bench = equal_weight_index(_frames(0.001)["Close"]), _benchmark(0.001)
        assert relative_strength(theme, bench, 63) == pytest.approx(0.0, abs=1e-9)

    def test_window_longer_than_history_returns_nan(self):
        theme, bench = equal_weight_index(_frames(0.001, n=30)["Close"]), _benchmark(0.0, 30)
        assert np.isnan(relative_strength(theme, bench, 126))

    def test_acceleration_positive_when_recent_move_is_faster(self):
        close = _frames(0.0)["Close"]
        # Flat for most of the window, then a sharp late advance.
        close.iloc[-21:] *= np.linspace(1.0, 1.3, 21)[:, None]
        assert rs_acceleration(equal_weight_index(close), _benchmark(0.0)) > 0

    def test_acceleration_negative_when_trend_is_stalling(self):
        close = _frames(0.003)["Close"]
        close.iloc[-21:] = close.iloc[-22]  # advance stops dead
        assert rs_acceleration(equal_weight_index(close), _benchmark(0.0)) < 0


class TestBreadth:
    def test_rising_basket_is_fully_above_its_average(self):
        breadth = breadth_above_ma(_frames(0.002)["Close"])
        assert breadth.iloc[-1] == pytest.approx(1.0)

    def test_falling_basket_has_no_members_above_average(self):
        breadth = breadth_above_ma(_frames(-0.002)["Close"])
        assert breadth.iloc[-1] == pytest.approx(0.0)

    def test_mixed_basket_is_partial(self):
        index = _calendar()
        close = pd.DataFrame({
            "UP": 100 * 1.002 ** np.arange(BARS),
            "DOWN": 100 * 0.998 ** np.arange(BARS),
        }, index=index)
        assert breadth_above_ma(close).iloc[-1] == pytest.approx(0.5)


class TestPctFromHigh:
    def test_at_the_high_is_zero(self):
        assert pct_from_high(equal_weight_index(_frames(0.001)["Close"])) == pytest.approx(0.0)

    def test_below_the_high_is_negative(self):
        close = _frames(0.001)["Close"]
        close.iloc[-1] *= 0.8
        assert pct_from_high(equal_weight_index(close)) < 0

    def test_empty_series_returns_nan(self):
        assert np.isnan(pct_from_high(pd.Series(dtype=float)))


class TestCrossSectionalZ:
    def test_constant_input_returns_zeros_not_nan(self):
        z = cross_sectional_z(pd.Series([5.0, 5.0, 5.0]))
        assert (z == 0).all()

    def test_all_nan_input_returns_zeros(self):
        assert (cross_sectional_z(pd.Series([np.nan, np.nan])) == 0).all()

    def test_ranking_is_preserved(self):
        z = cross_sectional_z(pd.Series([1.0, 2.0, 3.0]))
        assert z.iloc[0] < z.iloc[1] < z.iloc[2]

    def test_outliers_are_clipped(self):
        z = cross_sectional_z(pd.Series([0.0, 0.0, 0.0, 0.0, 1e9]), clip=3.0)
        assert z.max() <= 3.0

    def test_nan_members_score_neutral(self):
        z = cross_sectional_z(pd.Series([1.0, 2.0, 3.0, np.nan]))
        assert z.iloc[3] == 0.0


class TestSquash:
    def test_zero_maps_to_midpoint(self):
        assert squash(0.0) == pytest.approx(50.0)

    def test_output_stays_within_bounds(self):
        assert 0 <= squash(-50) < 1
        assert 99 < squash(50) <= 100

    def test_is_monotonic(self):
        assert squash(-1) < squash(0) < squash(1)


class TestClassifyStage:
    def test_flow_without_price_move_is_accumulation(self):
        assert classify_stage({
            "accum_score": 75, "rs_3m": 0.0, "rs_accel": -0.0001,
            "breadth": 0.5, "from_52w_high": -0.20, "run_12m": 0.05,
        }) == STAGE_ACCUMULATION

    def test_flow_plus_inflection_is_early_trend(self):
        assert classify_stage({
            "accum_score": 70, "rs_3m": 0.03, "rs_accel": 0.001,
            "breadth": 0.55, "from_52w_high": -0.10, "run_12m": 0.20,
        }) == STAGE_EARLY

    def test_broad_confirmed_advance_is_momentum(self):
        assert classify_stage({
            "accum_score": 55, "rs_3m": 0.25, "rs_accel": 0.002,
            "breadth": 0.85, "from_52w_high": -0.02, "run_12m": 0.40,
        }) == STAGE_MOMENTUM

    def test_extended_run_with_stalling_acceleration_is_crowded(self):
        assert classify_stage({
            "accum_score": 65, "rs_3m": 0.30, "rs_accel": -0.001,
            "breadth": 0.8, "from_52w_high": -0.01, "run_12m": 1.50,
        }) == STAGE_CROWDED

    def test_extension_outranks_momentum(self):
        # Strong trailing numbers must not relabel a narrowing, extended
        # theme as momentum — that is the failure mode the page exists to avoid.
        assert classify_stage({
            "accum_score": 80, "rs_3m": 0.40, "rs_accel": 0.003,
            "breadth": 0.30, "from_52w_high": -0.01, "run_12m": 2.0,
        }) == STAGE_CROWDED

    def test_nothing_firing_is_dormant(self):
        assert classify_stage({
            "accum_score": 30, "rs_3m": -0.10, "rs_accel": -0.001,
            "breadth": 0.2, "from_52w_high": -0.40, "run_12m": -0.30,
        }) == STAGE_DORMANT

    def test_all_nan_row_is_dormant(self):
        assert classify_stage({k: np.nan for k in (
            "accum_score", "rs_3m", "rs_accel", "breadth", "from_52w_high", "run_12m",
        )}) == STAGE_DORMANT


class TestBuildThemeSignals:
    @pytest.fixture
    def signals(self):
        bench = _benchmark(0.0002)
        themes = {
            "leader": _frames(0.0020, volume_multiplier=2.5, close_position=0.9),
            "laggard": _frames(-0.0015, close_position=0.1),
            "flat": _frames(0.0002),
        }
        return build_theme_signals(themes, bench)

    def test_returns_a_row_per_theme(self, signals):
        assert set(signals.index) == {"leader", "laggard", "flat"}

    def test_sorted_by_score_descending(self, signals):
        assert signals["smart_money_score"].is_monotonic_decreasing

    def test_accumulating_theme_outscores_distributing_one(self, signals):
        assert signals.loc["leader", "accum_score"] > signals.loc["laggard", "accum_score"]

    def test_every_theme_receives_a_stage(self, signals):
        assert signals["stage"].notna().all()

    def test_scores_stay_on_the_zero_to_hundred_scale(self, signals):
        assert signals["smart_money_score"].between(0, 100).all()
        assert signals["accum_score"].between(0, 100).all()

    def test_series_are_attached_for_charting(self, signals):
        assert set(signals.attrs["series"]) == {"leader", "laggard", "flat"}
        assert not signals.attrs["series"]["leader"]["theme_index"].empty

    def test_short_history_theme_is_flagged_unrankable(self):
        bench = _benchmark(0.0002)
        signals = build_theme_signals(
            {"ok": _frames(0.001), "new_listing": _frames(0.001, n=40)}, bench,
        )
        assert bool(signals.loc["ok", "rankable"])
        assert not bool(signals.loc["new_listing", "rankable"])

    def test_short_history_theme_does_not_crash_or_rank(self):
        bench = _benchmark(0.0002)
        signals = build_theme_signals({"new_listing": _frames(0.001, n=40)}, bench)
        assert signals.loc["new_listing", "stage"] == STAGE_DORMANT

    def test_theme_with_too_few_constituents_is_unrankable(self):
        bench = _benchmark(0.0002)
        signals = build_theme_signals(
            {"thin": _frames(0.001, tickers=("AAA",))}, bench, min_constituents=3,
        )
        assert not bool(signals.loc["thin", "rankable"])

    def test_empty_benchmark_returns_empty(self):
        assert build_theme_signals({"x": _frames(0.001)}, pd.Series(dtype=float)).empty

    def test_no_themes_returns_empty(self):
        assert build_theme_signals({}, _benchmark(0.0)).empty


class TestTruncateFrames:
    def test_drops_the_requested_number_of_bars(self):
        truncated = truncate_frames({"t": _frames(0.001)}, 30)
        assert len(truncated["t"]["Close"]) == BARS - 30

    def test_zero_bars_is_a_no_op(self):
        frames = {"t": _frames(0.001)}
        assert len(truncate_frames(frames, 0)["t"]["Close"]) == BARS

    def test_frames_shorter_than_the_cut_are_dropped(self):
        truncated = truncate_frames({"t": _frames(0.001, n=10)}, 30)
        assert truncated["t"] == {}

    def test_as_of_recomputation_sees_no_future_data(self):
        # A theme that only rallies in the final 30 bars must not look strong
        # in the truncated snapshot.
        close = _frames(0.0)["Close"]
        close.iloc[-30:] *= 2.0
        frames = {"late_mover": {**_frames(0.0), "Close": close}}
        bench = _benchmark(0.0)
        now = build_theme_signals(frames, bench)
        before = build_theme_signals(truncate_frames(frames, 30), bench.iloc[:-30])
        assert now.loc["late_mover", "rs_3m"] > before.loc["late_mover", "rs_3m"]


class TestDiffStages:
    def _snapshot(self, stages):
        return pd.DataFrame({"stage": stages})

    def test_move_into_accumulation_is_entering(self):
        diff = diff_stages(
            self._snapshot({"a": STAGE_ACCUMULATION}),
            self._snapshot({"a": STAGE_DORMANT}),
        )
        assert diff.loc["a", "direction"] == "entering"

    def test_early_stage_confirmed_by_price_is_maturing(self):
        # A flagged theme graduating into momentum is the thesis paying off,
        # and must not disappear into an unlabelled bucket.
        diff = diff_stages(
            self._snapshot({"a": STAGE_MOMENTUM}),
            self._snapshot({"a": STAGE_ACCUMULATION}),
        )
        assert diff.loc["a", "direction"] == "maturing"

    def test_every_transition_from_an_early_stage_is_labelled(self):
        for after in (STAGE_MOMENTUM, STAGE_CROWDED, STAGE_DORMANT):
            diff = diff_stages(
                self._snapshot({"a": after}), self._snapshot({"a": STAGE_EARLY}),
            )
            assert diff.loc["a", "direction"] in {"maturing", "cooling", "entering"}

    def test_rollover_into_crowded_is_cooling(self):
        diff = diff_stages(
            self._snapshot({"a": STAGE_CROWDED}),
            self._snapshot({"a": STAGE_MOMENTUM}),
        )
        assert diff.loc["a", "direction"] == "cooling"

    def test_unchanged_themes_are_omitted(self):
        diff = diff_stages(
            self._snapshot({"a": STAGE_MOMENTUM}),
            self._snapshot({"a": STAGE_MOMENTUM}),
        )
        assert diff.empty

    def test_themes_missing_from_one_side_are_skipped(self):
        diff = diff_stages(
            self._snapshot({"a": STAGE_EARLY, "b": STAGE_EARLY}),
            self._snapshot({"a": STAGE_DORMANT}),
        )
        assert list(diff.index) == ["a"]

    def test_empty_inputs_return_empty_frame(self):
        assert diff_stages(pd.DataFrame(), pd.DataFrame()).empty


class TestRotationMatrix:
    def test_one_row_per_theme(self):
        matrix = rotation_matrix(
            {
                "up": equal_weight_index(_frames(0.002)["Close"]),
                "down": equal_weight_index(_frames(-0.002)["Close"]),
            },
            _benchmark(0.0),
        )
        assert set(matrix.index) == {"up", "down"}

    def test_outperformer_scores_above_underperformer(self):
        matrix = rotation_matrix(
            {
                "up": equal_weight_index(_frames(0.002)["Close"]),
                "down": equal_weight_index(_frames(-0.002)["Close"]),
            },
            _benchmark(0.0),
        )
        assert matrix.loc["up"].mean() > matrix.loc["down"].mean()

    def test_empty_benchmark_returns_empty(self):
        assert rotation_matrix({"a": pd.Series([1.0])}, pd.Series(dtype=float)).empty


class TestConstituentTable:
    def test_lists_every_constituent(self):
        table = constituent_table(_frames(0.001), _benchmark(0.0))
        assert list(table.index) == ["AAA", "BBB", "CCC"]

    def test_carries_the_expected_columns(self):
        table = constituent_table(_frames(0.001), _benchmark(0.0))
        for column in ["Last", "RS vs Benchmark (%)", "Money Flow (CMF)", "From 52w High (%)"]:
            assert column in table.columns

    def test_empty_frames_return_empty(self):
        assert constituent_table({"Close": pd.DataFrame()}, _benchmark(0.0)).empty
