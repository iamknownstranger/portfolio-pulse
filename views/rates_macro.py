from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common.data import get_price_data

st.title("🏦 Fixed Income — Rates & Macro")
st.caption("US Treasury yields (CBOE indices via Yahoo Finance, quoted in %) and broad macro gauges.")

# Yahoo/CBOE yield indices and their maturities in years
YIELD_TICKERS = {
    "^IRX": ("13-Week T-Bill", 0.25),
    "^FVX": ("5-Year", 5.0),
    "^TNX": ("10-Year", 10.0),
    "^TYX": ("30-Year", 30.0),
}
MACRO_TICKERS = {
    "DX-Y.NYB": "US Dollar Index (DXY)",
    "^VIX": "CBOE Volatility Index (VIX)",
}

col_from, col_to = st.columns(2)
start_date = col_from.date_input("Start date", value=date.today() - timedelta(days=3 * 365))
end_date = col_to.date_input("End date", value=date.today())

yields = get_price_data(list(YIELD_TICKERS), start_date.isoformat(), end_date.isoformat())

if yields.empty:
    st.warning("No Treasury yield data available right now. Please try again later.")
    st.stop()

latest = yields.dropna(how="all").iloc[-1]
previous = yields.dropna(how="all").iloc[-2] if len(yields.dropna(how="all")) > 1 else latest

# --- Latest yields ---
st.subheader("Latest Yields")
cols = st.columns(len(YIELD_TICKERS))
for col, (ticker, (label, _)) in zip(cols, YIELD_TICKERS.items()):
    if ticker in latest.index and pd.notnull(latest[ticker]):
        delta = latest[ticker] - previous.get(ticker, latest[ticker])
        col.metric(label, f"{latest[ticker]:.2f}%", f"{delta:+.2f} pp")
    else:
        col.metric(label, "N/A")

# --- Yield curve snapshot ---
st.subheader("Yield Curve")
curve_points = [
    (maturity, latest[ticker], label)
    for ticker, (label, maturity) in YIELD_TICKERS.items()
    if ticker in latest.index and pd.notnull(latest[ticker])
]
if len(curve_points) >= 2:
    curve_points.sort()
    curve_fig = go.Figure(go.Scatter(
        x=[p[0] for p in curve_points],
        y=[p[1] for p in curve_points],
        mode="lines+markers+text",
        text=[p[2] for p in curve_points],
        textposition="top center",
    ))
    curve_fig.update_layout(xaxis_title="Maturity (years)", yaxis_title="Yield (%)",
                            title=f"US Treasury Yield Curve — {yields.dropna(how='all').index[-1]}")
    st.plotly_chart(curve_fig, use_container_width=True)

    # Curve slope: long minus short end, a classic recession watch signal
    short_y, long_y = curve_points[0][1], curve_points[-1][1]
    slope = long_y - short_y
    slope_cols = st.columns(2)
    slope_cols[0].metric(f"Curve Slope ({curve_points[-1][2]} − {curve_points[0][2]})", f"{slope:+.2f} pp")
    slope_cols[1].metric("Curve Shape", "Inverted ⚠️" if slope < 0 else "Normal")
else:
    st.info("Not enough curve points available to draw the yield curve.")

# --- Yield history ---
st.subheader("Yield History")
history = yields.rename(columns={t: label for t, (label, _) in YIELD_TICKERS.items()})
hist_fig = px.line(history, x=history.index, y=history.columns,
                   labels={"x": "Date", "value": "Yield (%)"}, title="Treasury Yields Over Time")
st.plotly_chart(hist_fig, use_container_width=True)

# --- Macro gauges ---
st.subheader("Macro Gauges")
macro = get_price_data(list(MACRO_TICKERS), start_date.isoformat(), end_date.isoformat())
if macro.empty:
    st.info("Macro gauge data (DXY, VIX) is unavailable right now.")
else:
    macro_latest = macro.dropna(how="all").iloc[-1]
    macro_prev = macro.dropna(how="all").iloc[-2] if len(macro.dropna(how="all")) > 1 else macro_latest
    mcols = st.columns(len(MACRO_TICKERS))
    for col, (ticker, label) in zip(mcols, MACRO_TICKERS.items()):
        if ticker in macro_latest.index and pd.notnull(macro_latest[ticker]):
            change = macro_latest[ticker] - macro_prev.get(ticker, macro_latest[ticker])
            col.metric(label, f"{macro_latest[ticker]:,.2f}", f"{change:+.2f}")
        else:
            col.metric(label, "N/A")
    macro_named = macro.rename(columns=MACRO_TICKERS)
    macro_fig = px.line(macro_named, x=macro_named.index, y=macro_named.columns,
                        labels={"x": "Date"}, title="Dollar Index & VIX")
    st.plotly_chart(macro_fig, use_container_width=True)
