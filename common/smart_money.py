"""Thematic accumulation signals for the Smart Money Radar.

Pure math with no Streamlit imports so the whole scoring pipeline stays
unit-testable (same convention as ``common.dates`` and ``common.options``).

The design goal is to surface themes *before* they boom. Ranking by trailing
return only ever surfaces what already worked, so the composite weights
accumulation evidence — money flow, dollar-volume expansion, broadening
participation — far above realized return, and the stage classifier
separates "capital arriving while price is still based" from "already run".
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252
RS_WINDOWS = {"1M": 21, "3M": 63, "6M": 126}

CMF_WINDOW = 21
BREADTH_MA = 50
BREADTH_LOOKBACK = 21
THRUST_RECENT = 20
THRUST_BASELINE = 100

# Minimum bars a theme needs before its signals are trustworthy enough to rank.
MIN_BARS = 130

# Composite weights. Trailing relative strength is deliberately the smallest
# term — the page exists to find themes before the return shows up.
SCORE_WEIGHTS = {
    "cmf": 0.30,
    "dv_thrust": 0.25,
    "rs_accel": 0.20,
    "breadth_chg": 0.15,
    "rs_3m": 0.10,
}

# Accumulation score excludes trailing return entirely: pure "is money
# arriving right now" evidence, which is what the quadrant chart plots.
ACCUM_WEIGHTS = {
    "cmf": 0.40,
    "dv_thrust": 0.35,
    "breadth_chg": 0.25,
}

STAGE_CROWDED = "⚠️ Crowded / Extended"
STAGE_MOMENTUM = "🔥 Momentum"
STAGE_EARLY = "🚀 Early Trend"
STAGE_ACCUMULATION = "🌱 Accumulation"
STAGE_DORMANT = "💤 Dormant"

# Best-to-worst for display ordering, not a ranking of desirability.
STAGE_ORDER = [
    STAGE_ACCUMULATION,
    STAGE_EARLY,
    STAGE_MOMENTUM,
    STAGE_CROWDED,
    STAGE_DORMANT,
]

# The stages the watchlist panel treats as "worth knowing about early".
EARLY_STAGES = frozenset({STAGE_ACCUMULATION, STAGE_EARLY})


# --- Basket construction -------------------------------------------------

def equal_weight_index(close):
    """Equal-weighted basket index (start = 1.0) from constituent closes.

    Averages daily returns cross-sectionally rather than averaging prices, so
    a constituent that lists late or has gaps joins the basket cleanly instead
    of creating a step change in the index level.
    """
    if close is None or close.empty:
        return pd.Series(dtype=float)
    returns = close.pct_change()
    basket = returns.mean(axis=1, skipna=True)
    if basket.dropna().empty:
        return pd.Series(0.0, index=close.index, dtype=float) + 1.0
    return (1 + basket.fillna(0)).cumprod()


def reindex_to_calendar(frame, calendar, ffill=False):
    """Align a frame/series to a reference trading calendar.

    Crypto trades every day while equities do not. Putting every theme on the
    benchmark's calendar makes window lengths (21/63/126 bars) mean the same
    elapsed time everywhere. Prices are forward-filled; volumes are not,
    because carrying a volume figure forward would double-count it.
    """
    if frame is None or len(frame) == 0:
        return frame
    aligned = frame.reindex(calendar)
    return aligned.ffill() if ffill else aligned


# --- Individual signals --------------------------------------------------

def chaikin_money_flow(high, low, close, volume, window=CMF_WINDOW):
    """Chaikin Money Flow per constituent — accumulation vs distribution.

    Money-flow multiplier weights each bar's volume by where the close landed
    inside the bar's range: closing near the high on heavy volume reads as
    accumulation. Bars with no range (high == low) contribute nothing rather
    than dividing by zero.
    """
    if close is None or close.empty:
        return pd.DataFrame()
    span = high - low
    multiplier = ((close - low) - (high - close)) / span.where(span != 0)
    flow_volume = (multiplier * volume).where(span != 0)
    flow_sum = flow_volume.rolling(window, min_periods=max(2, window // 2)).sum()
    volume_sum = volume.where(span != 0).rolling(
        window, min_periods=max(2, window // 2)
    ).sum()
    return flow_sum / volume_sum.where(volume_sum != 0)


def dollar_volume_thrust(close, volume, recent=THRUST_RECENT, baseline=THRUST_BASELINE):
    """Recent basket dollar volume versus its prior baseline, as a ratio - 1.

    Sustained expansion in traded value is the cleanest free footprint of
    institutional accumulation: size has to print somewhere.
    """
    if close is None or close.empty or volume is None or volume.empty:
        return np.nan
    dollar_volume = (close * volume).sum(axis=1, skipna=True).dropna()
    if len(dollar_volume) < recent + 10:
        return np.nan
    recent_mean = dollar_volume.iloc[-recent:].mean()
    base_slice = dollar_volume.iloc[-(recent + baseline):-recent]
    if base_slice.empty:
        return np.nan
    base_mean = base_slice.mean()
    if not np.isfinite(base_mean) or base_mean <= 0:
        return np.nan
    return recent_mean / base_mean - 1


def _window_return(series, window):
    """Simple return over the trailing ``window`` bars, NaN if too short."""
    clean = series.dropna()
    if len(clean) <= window:
        return np.nan
    start, end = clean.iloc[-1 - window], clean.iloc[-1]
    if not np.isfinite(start) or start <= 0:
        return np.nan
    return end / start - 1


def relative_strength(theme_index, benchmark, window):
    """Theme return minus benchmark return over ``window`` bars."""
    theme_ret = _window_return(theme_index, window)
    bench_ret = _window_return(benchmark, window)
    if not np.isfinite(theme_ret) or not np.isfinite(bench_ret):
        return np.nan
    return theme_ret - bench_ret


def rs_acceleration(theme_index, benchmark, short=RS_WINDOWS["1M"], long=RS_WINDOWS["6M"]):
    """Per-bar short-horizon RS minus per-bar long-horizon RS.

    Positive means outperformance is building faster now than it has over the
    longer window — the inflection that precedes a crowded trend.
    """
    short_rs = relative_strength(theme_index, benchmark, short)
    long_rs = relative_strength(theme_index, benchmark, long)
    if not np.isfinite(short_rs) or not np.isfinite(long_rs):
        return np.nan
    return short_rs / short - long_rs / long


def breadth_above_ma(close, window=BREADTH_MA):
    """Fraction of constituents trading above their ``window``-bar average."""
    if close is None or close.empty:
        return pd.Series(dtype=float)
    moving_average = close.rolling(window, min_periods=max(5, window // 2)).mean()
    above = close > moving_average
    valid = close.notna() & moving_average.notna()
    counts = valid.sum(axis=1)
    return (above & valid).sum(axis=1).where(counts > 0) / counts.where(counts > 0)


def pct_from_high(series, window=TRADING_DAYS):
    """Distance below the trailing high, as a non-positive fraction."""
    clean = series.dropna()
    if clean.empty:
        return np.nan
    peak = clean.iloc[-window:].max()
    if not np.isfinite(peak) or peak <= 0:
        return np.nan
    return clean.iloc[-1] / peak - 1


# --- Cross-sectional scoring --------------------------------------------

def cross_sectional_z(values, clip=3.0):
    """Z-score a signal across themes at a single point in time.

    Cross-sectional (not time-series) so no future information can leak in.
    A degenerate spread returns zeros instead of dividing by zero, and the
    result is clipped so one runaway theme cannot dominate the composite.
    """
    series = pd.Series(values, dtype=float)
    if series.dropna().empty:
        return pd.Series(0.0, index=series.index)
    spread = series.std(ddof=0)
    if not np.isfinite(spread) or spread == 0:
        return pd.Series(0.0, index=series.index)
    return ((series - series.mean()) / spread).clip(-clip, clip).fillna(0.0)


def squash(raw):
    """Map an unbounded weighted z-score onto a readable 0-100 scale."""
    return 50 + 50 * np.tanh(raw)


def classify_stage(row):
    """Label a theme's position in the accumulation -> crowding lifecycle.

    Order matters: extension is checked first so a theme that has already
    tripled cannot be sold back to the user as "momentum" once its breadth
    or acceleration has started to roll over.
    """
    accum = row.get("accum_score", np.nan)
    rs_3m = row.get("rs_3m", np.nan)
    accel = row.get("rs_accel", np.nan)
    breadth = row.get("breadth", np.nan)
    from_high = row.get("from_52w_high", np.nan)
    run_12m = row.get("run_12m", np.nan)

    extended = (
        np.isfinite(from_high) and np.isfinite(run_12m)
        and from_high > -0.05 and run_12m > 0.60
    )
    if extended and (
        (np.isfinite(accel) and accel <= 0) or (np.isfinite(breadth) and breadth < 0.5)
    ):
        return STAGE_CROWDED
    if (
        np.isfinite(rs_3m) and rs_3m > 0.05
        and np.isfinite(accel) and accel > 0
        and np.isfinite(breadth) and breadth >= 0.6
    ):
        return STAGE_MOMENTUM
    if (
        np.isfinite(accum) and accum >= 60
        and np.isfinite(accel) and accel > 0
        and np.isfinite(rs_3m) and rs_3m > -0.02
        and not extended
    ):
        return STAGE_EARLY
    if np.isfinite(accum) and accum >= 60 and np.isfinite(rs_3m) and rs_3m <= 0.05:
        return STAGE_ACCUMULATION
    return STAGE_DORMANT


# --- Pipeline ------------------------------------------------------------

def theme_raw_signals(frames, benchmark):
    """Raw (un-scored) signals for one theme from its OHLCV frames.

    ``frames`` is a dict of field name -> DataFrame (Close/High/Low/Volume),
    each indexed by date with one column per constituent.
    """
    close = frames.get("Close")
    if close is None or close.empty:
        return None

    calendar = benchmark.index
    # Count genuine observations before forward-filling: a theme that only
    # traded for part of the calendar would otherwise have its last price
    # carried across every remaining bar and look like a full history.
    observed = int(reindex_to_calendar(close, calendar).notna().any(axis=1).sum())
    close = reindex_to_calendar(close, calendar, ffill=True).dropna(how="all")
    if close.empty:
        return None
    high = reindex_to_calendar(frames.get("High"), calendar, ffill=True)
    low = reindex_to_calendar(frames.get("Low"), calendar, ffill=True)
    volume = reindex_to_calendar(frames.get("Volume"), calendar)

    index = equal_weight_index(close)
    breadth_series = breadth_above_ma(close)
    breadth_now = breadth_series.dropna().iloc[-1] if not breadth_series.dropna().empty else np.nan
    breadth_prior = (
        breadth_series.dropna().iloc[-1 - BREADTH_LOOKBACK]
        if len(breadth_series.dropna()) > BREADTH_LOOKBACK
        else np.nan
    )

    cmf_now = np.nan
    if high is not None and low is not None and volume is not None:
        cmf = chaikin_money_flow(high, low, close, volume)
        if not cmf.empty:
            latest = cmf.dropna(how="all")
            if not latest.empty:
                cmf_now = latest.iloc[-1].mean(skipna=True)

    return {
        "constituents": int(close.notna().any().sum()),
        "bars": observed,
        "theme_index": index,
        "breadth_series": breadth_series,
        "rs_1m": relative_strength(index, benchmark, RS_WINDOWS["1M"]),
        "rs_3m": relative_strength(index, benchmark, RS_WINDOWS["3M"]),
        "rs_6m": relative_strength(index, benchmark, RS_WINDOWS["6M"]),
        "rs_accel": rs_acceleration(index, benchmark),
        "dv_thrust": dollar_volume_thrust(close, volume),
        "cmf": cmf_now,
        "breadth": breadth_now,
        "breadth_chg": breadth_now - breadth_prior
        if np.isfinite(breadth_now) and np.isfinite(breadth_prior) else np.nan,
        "from_52w_high": pct_from_high(index),
        "run_12m": _window_return(index, TRADING_DAYS),
    }


def build_theme_signals(theme_frames, benchmark, min_constituents=3, min_bars=MIN_BARS):
    """Score every theme and label its lifecycle stage.

    ``theme_frames`` maps theme slug -> {field: DataFrame}. Nothing is fetched
    here, which keeps the whole pipeline testable against synthetic frames.

    Themes with too little history or too few surviving constituents are kept
    in the output (so the page can report them) but excluded from the
    cross-sectional statistics, where they would otherwise distort the mean.
    """
    benchmark = benchmark.dropna() if benchmark is not None else pd.Series(dtype=float)
    if benchmark.empty or not theme_frames:
        return pd.DataFrame()

    rows, series = {}, {}
    for slug, frames in theme_frames.items():
        raw = theme_raw_signals(frames, benchmark)
        if raw is None:
            continue
        series[slug] = {
            "theme_index": raw.pop("theme_index"),
            "breadth_series": raw.pop("breadth_series"),
        }
        rows[slug] = raw

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).T
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["rankable"] = (df["constituents"] >= min_constituents) & (df["bars"] >= min_bars)

    rankable = df.index[df["rankable"].astype(bool)]
    scored = df.loc[rankable] if len(rankable) else df.iloc[0:0]

    accum_raw = pd.Series(0.0, index=scored.index)
    for signal, weight in ACCUM_WEIGHTS.items():
        accum_raw += weight * cross_sectional_z(scored[signal])
    df["accum_score"] = squash(accum_raw).reindex(df.index)

    score_raw = pd.Series(0.0, index=scored.index)
    for signal, weight in SCORE_WEIGHTS.items():
        score_raw += weight * cross_sectional_z(scored[signal])
    df["smart_money_score"] = squash(score_raw).reindex(df.index)

    df["stage"] = [
        classify_stage(row) if row["rankable"] else STAGE_DORMANT
        for _, row in df.iterrows()
    ]
    df.attrs["series"] = series
    return df.sort_values("smart_money_score", ascending=False)


def truncate_frames(theme_frames, bars):
    """Drop the last ``bars`` rows from every frame, for as-of recomputation.

    The watchlist panel re-runs the whole pipeline on truncated data rather
    than reusing today's z-scores, so the "30 days ago" snapshot cannot see
    anything that had not happened yet.
    """
    if bars <= 0:
        return theme_frames
    return {
        slug: {field: frame.iloc[:-bars] for field, frame in frames.items()
               if frame is not None and len(frame) > bars}
        for slug, frames in theme_frames.items()
    }


def diff_stages(now, before):
    """Themes whose lifecycle stage changed between two snapshots."""
    if now is None or now.empty or before is None or before.empty:
        return pd.DataFrame(columns=["stage_before", "stage_now", "direction"])

    shared = now.index.intersection(before.index)
    records = []
    for slug in shared:
        was, is_now = before.loc[slug, "stage"], now.loc[slug, "stage"]
        if was == is_now:
            continue
        if is_now in EARLY_STAGES and was not in EARLY_STAGES:
            direction = "entering"
        elif was in EARLY_STAGES and is_now == STAGE_MOMENTUM:
            # The thesis played out: an early flag has been confirmed by price.
            direction = "maturing"
        elif was in EARLY_STAGES | {STAGE_MOMENTUM} and is_now in (STAGE_CROWDED, STAGE_DORMANT):
            direction = "cooling"
        else:
            direction = "other"
        records.append({
            "theme": slug,
            "stage_before": was,
            "stage_now": is_now,
            "direction": direction,
        })
    if not records:
        return pd.DataFrame(columns=["stage_before", "stage_now", "direction"])
    return pd.DataFrame(records).set_index("theme")


def rotation_matrix(theme_indices, benchmark, freq="ME"):
    """Per-period relative strength for every theme — the rotation heatmap.

    Each cell is the theme's return minus the benchmark's return over that
    calendar period, so reading across a row shows a narrative gaining and
    losing sponsorship over time.
    """
    benchmark = benchmark.dropna() if benchmark is not None else pd.Series(dtype=float)
    if benchmark.empty or not theme_indices:
        return pd.DataFrame()

    bench_periods = benchmark.resample(freq).last().pct_change().dropna()
    if bench_periods.empty:
        return pd.DataFrame()

    rows = {}
    for slug, index in theme_indices.items():
        clean = index.dropna()
        if clean.empty:
            continue
        theme_periods = clean.resample(freq).last().pct_change().dropna()
        aligned = theme_periods.reindex(bench_periods.index)
        rows[slug] = (aligned - bench_periods) * 100
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).T.dropna(axis=1, how="all")


def constituent_table(frames, benchmark, window=RS_WINDOWS["3M"]):
    """Per-name breakdown for the drill-down view of a single theme."""
    close = frames.get("Close")
    if close is None or close.empty:
        return pd.DataFrame()

    calendar = benchmark.index
    close = reindex_to_calendar(close, calendar, ffill=True)
    high = reindex_to_calendar(frames.get("High"), calendar, ffill=True)
    low = reindex_to_calendar(frames.get("Low"), calendar, ffill=True)
    volume = reindex_to_calendar(frames.get("Volume"), calendar)

    cmf = (
        chaikin_money_flow(high, low, close, volume)
        if high is not None and low is not None and volume is not None
        else pd.DataFrame()
    )

    rows = []
    for ticker in close.columns:
        prices = close[ticker].dropna()
        if prices.empty:
            continue
        thrust = np.nan
        if volume is not None and ticker in volume.columns:
            thrust = dollar_volume_thrust(
                close[[ticker]], volume[[ticker]]
            )
        latest_cmf = np.nan
        if not cmf.empty and ticker in cmf.columns:
            valid = cmf[ticker].dropna()
            latest_cmf = valid.iloc[-1] if not valid.empty else np.nan
        rows.append({
            "Ticker": ticker,
            "Last": prices.iloc[-1],
            "Return 3M (%)": (_window_return(prices, window) or np.nan) * 100,
            "RS vs Benchmark (%)": relative_strength(prices, benchmark, window) * 100,
            "Dollar Vol Thrust (%)": thrust * 100 if np.isfinite(thrust) else np.nan,
            "Money Flow (CMF)": latest_cmf,
            "From 52w High (%)": pct_from_high(prices) * 100,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("Ticker").sort_values(
        "RS vs Benchmark (%)", ascending=False
    )
