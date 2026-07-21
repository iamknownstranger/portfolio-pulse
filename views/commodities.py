from datetime import date, timedelta

import numpy as np
import plotly.express as px
import streamlit as st

from common.data import get_price_data

st.title("🛢️ Commodities")
st.caption("Front-month futures via Yahoo Finance.")

COMMODITY_TICKERS = {
    "GC=F": "Gold",
    "SI=F": "Silver",
    "HG=F": "Copper",
    "CL=F": "WTI Crude",
    "BZ=F": "Brent Crude",
    "NG=F": "Natural Gas",
    "ZC=F": "Corn",
    "ZW=F": "Wheat",
}

col_from, col_to, col_sel = st.columns([1, 1, 2])
start_date = col_from.date_input("Start date", value=date.today() - timedelta(days=365))
end_date = col_to.date_input("End date", value=date.today())
selected_names = col_sel.multiselect(
    "Commodities", list(COMMODITY_TICKERS.values()), default=list(COMMODITY_TICKERS.values())
)
tickers = [t for t, name in COMMODITY_TICKERS.items() if name in selected_names]

if not tickers:
    st.info("Select at least one commodity to analyze.")
    st.stop()

prices = get_price_data(tickers, start_date.isoformat(), end_date.isoformat())

if prices.empty:
    st.warning("No commodity price data available for the selection. Please try again later.")
    st.stop()

prices = prices.rename(columns=COMMODITY_TICKERS)
returns = prices.pct_change().dropna(how="all")

# --- Latest prices and period performance ---
st.subheader("Snapshot")
valid = prices.dropna(how="all")
latest, first = valid.iloc[-1], valid.iloc[0]
cols = st.columns(min(4, len(prices.columns)))
for i, name in enumerate(prices.columns):
    if not (np.isfinite(latest.get(name, np.nan)) and np.isfinite(first.get(name, np.nan))):
        continue
    period_return = (latest[name] / first[name] - 1) * 100
    cols[i % len(cols)].metric(name, f"{latest[name]:,.2f}", f"{period_return:+.2f}% over period")

# --- Normalized performance ---
st.subheader("Relative Performance")
normalized = valid / valid.iloc[0] * 100
perf_fig = px.line(normalized, x=normalized.index, y=normalized.columns,
                   labels={"x": "Date", "value": "Indexed (start = 100)"},
                   title="Growth of 100 by Commodity")
st.plotly_chart(perf_fig, use_container_width=True)

# --- Volatility ---
st.subheader("30-Day Rolling Volatility (annualized)")
rolling_vol = returns.rolling(30).std() * np.sqrt(252) * 100
vol_fig = px.line(rolling_vol, x=rolling_vol.index, y=rolling_vol.columns,
                  labels={"x": "Date", "value": "Volatility (%)"})
st.plotly_chart(vol_fig, use_container_width=True)

# --- Correlations ---
if len(prices.columns) > 1:
    st.subheader("Return Correlations")
    st.write("Cross-commodity correlations help spot diversification within the asset class "
             "(e.g. precious metals vs energy vs agriculture).")
    corr_fig = px.imshow(returns.corr(method="pearson"), text_auto=".2f",
                         title="Daily Return Correlation Matrix")
    st.plotly_chart(corr_fig, use_container_width=True)
