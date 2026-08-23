from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from common.data import get_benchmark_data, get_ohlcv_data
from common.sec_edgar import (
    CURATED_MANAGERS,
    build_issuer_ticker_map,
    diff_positions,
    get_insider_buys,
    get_manager_positions,
    get_ticker_cik_map,
    map_positions_to_tickers,
)
from common.smart_money import (
    EARLY_STAGES,
    STAGE_ACCUMULATION,
    STAGE_CROWDED,
    STAGE_EARLY,
    build_theme_signals,
    constituent_table,
    diff_stages,
    rotation_matrix,
    truncate_frames,
)
from common.themes import THEMES, all_tickers, asset_classes, theme_label

AS_OF_BARS = 30
STAGE_COLORS = {
    "🌱 Accumulation": "#2E8B57",
    "🚀 Early Trend": "#1F77B4",
    "🔥 Momentum": "#FF7F0E",
    "⚠️ Crowded / Extended": "#D62728",
    "💤 Dormant": "#9E9E9E",
}

st.title("🧭 Smart Money Radar")
st.caption(
    "Ranks thematic baskets by evidence of **accumulation** rather than by trailing return, "
    "so a narrative can surface while it is still based. Price and volume signals are a "
    "*proxy* for institutional flow, not actual flow data. The SEC panels below are real "
    "filings: Form 4 insider purchases are timely, 13F holdings are filed up to 45 days "
    "after quarter end and therefore confirm a theme rather than front-run it."
)

# --- Controls ---
col_bench, col_class, col_hist = st.columns([2, 2, 2])
BENCHMARKS = {"S&P 500": "^GSPC", "NASDAQ": "^IXIC", "Russell 2000": "^RUT", "Dow Jones": "^DJI"}
benchmark_name = col_bench.selectbox("Benchmark", list(BENCHMARKS), index=0)
benchmark_symbol = BENCHMARKS[benchmark_name]
selected_class = col_class.selectbox("Asset class", ["All", *asset_classes()], index=0)
years = col_hist.slider(
    "Years of history", 2, 5, 3,
    help="Signals need a full year of bars for 52-week extension plus a 30-day "
         "look-back for the change alerts.",
)

end_date = date.today()
start_date = end_date - timedelta(days=int(years * 365.25))

with st.spinner("Loading theme constituents…"):
    ohlcv = get_ohlcv_data(all_tickers(), start_date.isoformat(), end_date.isoformat())
    benchmark = get_benchmark_data(benchmark_symbol, start_date.isoformat(), end_date.isoformat())

if not ohlcv or benchmark.empty:
    st.warning(
        "Market data is unavailable right now (Yahoo Finance may be rate limiting). "
        "Please try again in a few minutes."
    )
    st.stop()

available = set(ohlcv["Close"].columns)
theme_frames, missing = {}, {}
for slug, theme in THEMES.items():
    if selected_class != "All" and theme["asset_class"] != selected_class:
        continue
    present = [t for t in theme["tickers"] if t in available]
    if not present:
        continue
    theme_frames[slug] = {
        field: frame.reindex(columns=present) for field, frame in ohlcv.items()
    }
    absent = [t for t in theme["tickers"] if t not in available]
    if absent:
        missing[slug] = absent

if not theme_frames:
    st.warning("No theme returned usable price data for this selection.")
    st.stop()

signals = build_theme_signals(theme_frames, benchmark)
if signals.empty:
    st.warning("Not enough overlapping history to score the themes.")
    st.stop()

series = signals.attrs.get("series", {})
signals = signals.assign(theme=[theme_label(slug) for slug in signals.index])

# As-of snapshot: the whole pipeline is recomputed on truncated data, so the
# cross-sectional z-scores cannot see anything that had not happened yet.
before = build_theme_signals(
    truncate_frames(theme_frames, AS_OF_BARS), benchmark.iloc[:-AS_OF_BARS],
)
alerts = diff_stages(signals, before)

ranked = signals[signals["rankable"].astype(bool)]

# --- What changed ---
st.subheader("What Changed in the Last 30 Days")
PANELS = [
    ("entering", "🟢 Newly entering accumulation",
     "No theme crossed into accumulation or early trend this month."),
    ("maturing", "🔵 Confirmed by price",
     "No early-stage theme graduated into momentum this month."),
    ("cooling", "🟠 Cooling or newly crowded",
     "No theme rolled over from a trending stage this month."),
]
for column, (direction, heading, empty_message) in zip(st.columns(3), PANELS):
    with column.container(border=True):
        st.markdown(f"**{heading}**")
        moves = alerts[alerts["direction"] == direction] if not alerts.empty else alerts
        if moves.empty:
            st.caption(empty_message)
        else:
            for slug, row in moves.iterrows():
                st.markdown(
                    f"- {theme_label(slug)} — {row['stage_before']} → **{row['stage_now']}**"
                )

def _leader(frame, column):
    """Index label of the highest finite value, or None if the column is empty."""
    values = frame[column].dropna()
    return values.idxmax() if not values.empty else None


def _name(slug):
    """Theme label without its leading icon, for compact metric tiles."""
    return theme_label(slug).split(" ", 1)[-1]


# --- Headline metrics ---
if not ranked.empty:
    st.subheader("Where Capital Is Moving")
    early = ranked[ranked["stage"].isin(EARLY_STAGES)]
    tiles = st.columns(4)

    top_accum = _leader(ranked, "accum_score")
    if top_accum is None:
        tiles[0].metric("Strongest accumulation", "—")
    else:
        tiles[0].metric(
            "Strongest accumulation", _name(top_accum),
            f"{ranked.loc[top_accum, 'accum_score']:.0f}/100 flow score",
        )

    top_accel = _leader(ranked, "rs_accel")
    if top_accel is None:
        tiles[1].metric("Fastest inflection", "—")
    else:
        tiles[1].metric(
            "Fastest inflection", _name(top_accel),
            f"{ranked.loc[top_accel, 'rs_accel'] * 10000:+.1f} bps/day vs {benchmark_name}",
        )

    crowded = ranked[ranked["stage"] == STAGE_CROWDED]
    worst = _leader(crowded, "run_12m") if not crowded.empty else None
    if worst is None:
        tiles[2].metric("Most crowded", "None flagged", "no theme extended and narrowing")
    else:
        tiles[2].metric(
            "Most crowded", _name(worst),
            f"{crowded.loc[worst, 'run_12m'] * 100:+.0f}% over 12m", delta_color="inverse",
        )
    tiles[3].metric(
        "Themes still early", f"{len(early)} of {len(ranked)}",
        "accumulation or early trend",
    )

# --- Leaderboard ---
st.subheader("Theme Leaderboard")
st.write(
    "Ranked by a composite that weights money flow, dollar-volume expansion and broadening "
    "participation far above trailing return — ranking on return alone would only ever "
    "surface what has already worked."
)

DISPLAY_COLUMNS = {
    "theme": "Theme",
    "stage": "Stage",
    "smart_money_score": "Score",
    "accum_score": "Accumulation",
    "cmf": "Money Flow",
    "dv_thrust": "Vol Thrust (%)",
    "rs_accel": "RS Accel (bps/d)",
    "rs_3m": "RS 3M (%)",
    "rs_6m": "RS 6M (%)",
    "breadth": "Breadth (%)",
    "from_52w_high": "From 52w High (%)",
    "run_12m": "12M Run (%)",
    "constituents": "Names",
}
board = signals.reindex(columns=list(DISPLAY_COLUMNS)).rename(columns=DISPLAY_COLUMNS)
# The chart series ride along in .attrs; Streamlit cannot serialize them and
# they are not needed for the table.
board.attrs = {}
for column in ["Vol Thrust (%)", "RS 3M (%)", "RS 6M (%)", "Breadth (%)",
               "From 52w High (%)", "12M Run (%)"]:
    board[column] = board[column] * 100
board["RS Accel (bps/d)"] = board["RS Accel (bps/d)"] * 10000

st.dataframe(
    board, use_container_width=True, hide_index=True,
    column_config={
        "Score": st.column_config.ProgressColumn(
            "Score", min_value=0, max_value=100, format="%.0f",
            help="Composite: money flow, volume expansion and broadening participation, "
                 "weighted above trailing return.",
        ),
        "Accumulation": st.column_config.ProgressColumn(
            "Accumulation", min_value=0, max_value=100, format="%.0f",
            help="Flow evidence only — excludes trailing return entirely.",
        ),
        "Money Flow": st.column_config.NumberColumn(format="%+.3f"),
        "Vol Thrust (%)": st.column_config.NumberColumn(format="%+.1f%%"),
        "RS Accel (bps/d)": st.column_config.NumberColumn(format="%+.1f"),
        "RS 3M (%)": st.column_config.NumberColumn(format="%+.1f%%"),
        "RS 6M (%)": st.column_config.NumberColumn(format="%+.1f%%"),
        "Breadth (%)": st.column_config.NumberColumn(format="%.0f%%"),
        "From 52w High (%)": st.column_config.NumberColumn(format="%.1f%%"),
        "12M Run (%)": st.column_config.NumberColumn(format="%+.1f%%"),
        "Names": st.column_config.NumberColumn(format="%d"),
    },
)
st.download_button(
    "Download signals as CSV", board.to_csv(index=False).encode("utf-8"),
    file_name=f"smart_money_radar_{end_date.isoformat()}.csv", mime="text/csv",
)

unrankable = signals[~signals["rankable"].astype(bool)]
if not unrankable.empty:
    st.caption(
        "Excluded from the cross-sectional scoring for insufficient history or too few "
        "surviving constituents: "
        + ", ".join(theme_label(slug) for slug in unrankable.index) + "."
    )
if missing:
    st.caption(
        "Symbols with no data returned: "
        + "; ".join(f"{theme_label(s)} ({', '.join(t)})" for s, t in missing.items()) + "."
    )

# --- Quadrant ---
if not ranked.empty:
    st.subheader("Early vs Already-Run")
    st.write(
        "The horizontal axis is how much a theme has **already** outperformed over six "
        "months; the vertical axis is how hard money is going in **now**. The top-left "
        "quadrant — heavy accumulation, price still flat — is what this page exists to find."
    )
    quadrant = ranked.assign(
        rs_6m_pct=ranked["rs_6m"] * 100,
        thrust_size=ranked["dv_thrust"].abs().fillna(0) * 100 + 10,
    )
    fig = px.scatter(
        quadrant, x="rs_6m_pct", y="accum_score", color="stage", text="theme",
        size="thrust_size", size_max=38, color_discrete_map=STAGE_COLORS,
        labels={"rs_6m_pct": f"6-month relative strength vs {benchmark_name} (%)",
                "accum_score": "Accumulation score (flow going in now)",
                "stage": "Lifecycle stage"},
        hover_data={"thrust_size": False, "theme": False},
    )
    fig.update_traces(textposition="top center", textfont_size=10)
    fig.add_hline(y=60, line_dash="dot", line_color="gray")
    fig.add_vline(x=0, line_dash="dot", line_color="gray")
    x_min, x_max = quadrant["rs_6m_pct"].min(), quadrant["rs_6m_pct"].max()
    span = max(abs(x_min), abs(x_max), 5) * 0.75
    for x_pos, y_pos, label in [
        (-span, 92, "Under the radar — accumulating, hasn't run"),
        (span, 92, "Confirmed leaders — flow and price both in"),
        (-span, 12, "Falling knives — no flow, no trend"),
        (span, 12, "Distribution risk — price ran, flow fading"),
    ]:
        fig.add_annotation(x=x_pos, y=y_pos, text=label, showarrow=False,
                           font=dict(size=10, color="gray"))
    fig.update_layout(height=560)
    st.plotly_chart(fig, use_container_width=True)

# --- Rotation ---
st.subheader("Theme Rotation Over Time")
st.write(
    "Monthly relative strength versus the benchmark. Reading across a row shows a narrative "
    "gaining and losing sponsorship; reading down a column shows what capital rotated into "
    "that month."
)
rotation = rotation_matrix(
    {slug: payload["theme_index"] for slug, payload in series.items()}, benchmark,
)
if rotation.empty:
    st.info("Not enough history to build the rotation heatmap.")
else:
    rotation = rotation.iloc[:, -18:]
    rotation.index = [theme_label(slug) for slug in rotation.index]
    rotation.columns = [pd.Timestamp(c).strftime("%b %Y") for c in rotation.columns]
    heat = px.imshow(
        rotation, text_auto=".0f", aspect="auto", color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        labels={"color": "Relative strength (%)", "x": "Month", "y": ""},
    )
    heat.update_layout(height=max(360, 26 * len(rotation)))
    st.plotly_chart(heat, use_container_width=True)

# --- Drill-down ---
st.subheader("Theme Detail")
choice = st.selectbox(
    "Theme", list(signals.index), format_func=theme_label,
    index=0,
)
detail = signals.loc[choice]
theme_meta = THEMES[choice]
st.markdown(f"**{theme_meta['thesis']}**")

info = st.columns(4)
info[0].metric("Stage", detail["stage"])
info[1].metric("Smart money score", f"{detail['smart_money_score']:.0f}/100"
               if np.isfinite(detail["smart_money_score"]) else "—")
info[2].metric("Money flow (CMF)", f"{detail['cmf']:+.3f}"
               if np.isfinite(detail["cmf"]) else "—")
info[3].metric("Dollar-volume thrust", f"{detail['dv_thrust'] * 100:+.0f}%"
               if np.isfinite(detail["dv_thrust"]) else "—")

trend_tab, names_tab, insider_tab = st.tabs(
    ["Trend & breadth", "Constituents", "Insider buying (SEC Form 4)"]
)

with trend_tab:
    payload = series.get(choice, {})
    theme_index = payload.get("theme_index", pd.Series(dtype=float))
    if theme_index.empty:
        st.info("No index series available for this theme.")
    else:
        bench_norm = benchmark.reindex(theme_index.index).ffill()
        bench_norm = bench_norm / bench_norm.dropna().iloc[0] * 100
        growth = pd.DataFrame({
            theme_meta["label"]: theme_index / theme_index.iloc[0] * 100,
            benchmark_name: bench_norm,
        }).dropna(how="all")
        st.plotly_chart(
            px.line(growth, labels={"value": "Growth of 100", "index": "Date",
                                    "variable": ""},
                    title=f"{theme_meta['label']} vs {benchmark_name}"),
            use_container_width=True,
        )
        breadth = payload.get("breadth_series", pd.Series(dtype=float)).dropna() * 100
        if not breadth.empty:
            st.plotly_chart(
                px.area(breadth, labels={"value": "% above 50-day average", "index": "Date"},
                        title="Participation: share of constituents in an uptrend")
                .update_layout(showlegend=False, yaxis_range=[0, 100]),
                use_container_width=True,
            )
            st.caption(
                "Breadth rising alongside price means the move is broadening. Price making "
                "new highs while breadth falls means fewer names are carrying it."
            )

with names_tab:
    table = constituent_table(theme_frames[choice], benchmark)
    if table.empty:
        st.info("No constituent data available.")
    else:
        st.dataframe(
            table, use_container_width=True,
            column_config={
                "Last": st.column_config.NumberColumn(format="%.2f"),
                "Return 3M (%)": st.column_config.NumberColumn(format="%+.1f%%"),
                "RS vs Benchmark (%)": st.column_config.NumberColumn(format="%+.1f%%"),
                "Dollar Vol Thrust (%)": st.column_config.NumberColumn(format="%+.1f%%"),
                "Money Flow (CMF)": st.column_config.NumberColumn(format="%+.3f"),
                "From 52w High (%)": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )

with insider_tab:
    st.write(
        "Open-market purchases (Form 4 transaction code **P**) by officers and directors of "
        "this theme's constituents. Grants, option exercises and tax-withholding sales are "
        "excluded — they are scheduled compensation events, not conviction. Genuine "
        "open-market buys are rare, so an empty result is a normal outcome."
    )
    lookback_days = st.select_slider(
        "Look back", options=[30, 90, 180, 365], value=180, key="insider_days",
    )
    if st.button("Load insider filings from SEC EDGAR", key="load_insiders"):
        with st.spinner("Fetching Form 4 filings…"):
            try:
                buys = get_insider_buys(tuple(theme_meta["tickers"]), days=lookback_days)
            except Exception as e:  # EDGAR outage must not break the page
                st.error(f"Could not reach SEC EDGAR: {e}")
                buys = pd.DataFrame()
        if buys.empty:
            st.info(
                f"No open-market insider purchases filed for these names in the last "
                f"{lookback_days} days."
            )
        else:
            st.metric("Total insider buying", f"${buys['value'].sum():,.0f}")
            st.dataframe(
                buys.style.format({
                    "shares": "{:,.0f}", "price": "{:,.2f}", "value": "${:,.0f}",
                    "transaction_date": lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"),
                }, na_rep="—"),
                use_container_width=True, hide_index=True,
            )

# --- 13F ---
st.subheader("Institutional Positioning (SEC 13F)")
with st.expander("Quarter-over-quarter changes across well-known managers"):
    st.write(
        "Position changes between the two most recent 13F filings of "
        f"{len(CURATED_MANAGERS)} managers, aggregated onto the themes above. "
        "**13F filings are due 45 days after quarter end**, so this confirms which "
        "narratives institutions were building into — it does not front-run them. "
        "Only long US equity positions are reportable; shorts, cash and non-US "
        "holdings are invisible here."
    )
    if st.button("Load 13F filings from SEC EDGAR", key="load_13f"):
        ticker_to_theme = {
            ticker: slug for slug, theme in THEMES.items() for ticker in theme["tickers"]
        }
        progress = st.progress(0.0, text="Fetching 13F filings…")
        rows, quarters = [], set()
        try:
            cik_map = get_ticker_cik_map()
            issuer_map = build_issuer_ticker_map({
                ticker: cik_map[ticker]["title"]
                for ticker in ticker_to_theme if ticker in cik_map
            })
            for i, (cik, name) in enumerate(CURATED_MANAGERS.items(), start=1):
                progress.progress(i / len(CURATED_MANAGERS), text=f"Fetching {name}…")
                current, prior, current_date, prior_date = get_manager_positions(cik)
                if current.empty:
                    continue
                quarters.add((str(prior_date), str(current_date)))
                changes = diff_positions(
                    map_positions_to_tickers(current, issuer_map),
                    map_positions_to_tickers(prior, issuer_map),
                )
                changes = changes.dropna(subset=["ticker"])
                if changes.empty:
                    continue
                changes["manager"] = name
                changes["theme"] = changes["ticker"].map(ticker_to_theme)
                rows.append(changes)
        except Exception as e:
            st.error(f"Could not reach SEC EDGAR: {e}")
            rows = []
        finally:
            progress.empty()

        if not rows:
            st.info("No mapped 13F positions were returned. Try again shortly.")
        else:
            holdings = pd.concat(rows, ignore_index=True).dropna(subset=["theme"])
            if quarters:
                prior_q, current_q = sorted(quarters)[-1]
                st.caption(f"Comparing quarter ending {prior_q} → {current_q}.")

            by_theme = (
                holdings.groupby("theme")
                .agg(**{
                    "Net $ change": ("value_change", "sum"),
                    "Position value": ("value", "sum"),
                    "New stakes": ("action", lambda a: (a == "New").sum()),
                    "Adds": ("action", lambda a: (a == "Added").sum()),
                    "Trims": ("action", lambda a: (a == "Trimmed").sum()),
                })
                .sort_values("Net $ change", ascending=False)
            )
            by_theme.index = [theme_label(slug) for slug in by_theme.index]
            st.plotly_chart(
                px.bar(by_theme.reset_index(), x="Net $ change", y="index",
                       orientation="h", color="Net $ change",
                       color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                       labels={"index": "", "Net $ change": "Net change in reported value ($)"},
                       title="Net institutional buying by theme, last quarter")
                .update_layout(height=max(320, 30 * len(by_theme)), showlegend=False),
                use_container_width=True,
            )
            st.dataframe(
                by_theme.style.format({
                    "Net $ change": "${:,.0f}", "Position value": "${:,.0f}",
                    "New stakes": "{:.0f}", "Adds": "{:.0f}", "Trims": "{:.0f}",
                }),
                use_container_width=True,
            )

            st.markdown("**Largest single position changes**")
            movers = holdings.reindex(
                columns=["manager", "ticker", "issuer", "action", "value", "value_change"]
            ).sort_values("value_change", key=abs, ascending=False).head(25)
            movers["theme"] = movers["ticker"].map(
                lambda t: theme_label(ticker_to_theme.get(t, ""))
            )
            st.dataframe(
                movers.style.format({"value": "${:,.0f}", "value_change": "${:+,.0f}"}),
                use_container_width=True, hide_index=True,
            )
