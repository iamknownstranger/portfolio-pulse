from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from common.data import get_price_data

st.title("🪙 Crypto Metrics")
st.caption("Spot prices via Yahoo Finance. Crypto trades 24/7, so risk metrics "
           "are annualized over 365 days rather than the 252 used for equities.")

CRYPTO_TICKERS = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "SOL-USD": "Solana",
    "BNB-USD": "BNB",
    "XRP-USD": "XRP",
    "ADA-USD": "Cardano",
    "AVAX-USD": "Avalanche",
    "DOGE-USD": "Dogecoin",
}
CRYPTO_TRADING_DAYS = 365

col_from, col_to, col_sel = st.columns([1, 1, 2])
start_date = col_from.date_input("Start date", value=date.today() - timedelta(days=365))
end_date = col_to.date_input("End date", value=date.today())
selected_names = col_sel.multiselect(
    "Assets", list(CRYPTO_TICKERS.values()),
    default=["Bitcoin", "Ethereum", "Solana", "XRP"],
)
tickers = [t for t, name in CRYPTO_TICKERS.items() if name in selected_names]

if not tickers:
    st.info("Select at least one asset to analyze.")
    st.stop()

prices = get_price_data(tickers, start_date.isoformat(), end_date.isoformat())

if prices.empty:
    st.warning("No crypto price data available for the selection. Please try again later.")
    st.stop()

prices = prices.rename(columns=CRYPTO_TICKERS).dropna(how="all")
returns = prices.pct_change().dropna(how="all")

# --- Snapshot ---
st.subheader("Snapshot")
latest, first = prices.iloc[-1], prices.iloc[0]
cols = st.columns(min(4, len(prices.columns)))
for i, name in enumerate(prices.columns):
    if not (np.isfinite(latest.get(name, np.nan)) and np.isfinite(first.get(name, np.nan))):
        continue
    period_return = (latest[name] / first[name] - 1) * 100
    cols[i % len(cols)].metric(name, f"${latest[name]:,.2f}", f"{period_return:+.2f}% over period")

# --- Relative performance ---
st.subheader("Relative Performance")
normalized = prices / prices.iloc[0] * 100
perf_fig = px.line(normalized, x=normalized.index, y=normalized.columns,
                   labels={"x": "Date", "value": "Indexed (start = 100)"},
                   title="Growth of 100 by Asset")
st.plotly_chart(perf_fig, use_container_width=True)

# --- Risk metrics table ---
st.subheader("Risk & Return Metrics")
st.write("Crypto's defining risk features: volatility multiples of equities and deep, "
         "prolonged drawdowns. VaR is the 95% one-day historical value at risk.")
metrics = {}
for name in returns.columns:
    r = returns[name].dropna()
    if r.empty:
        continue
    cum = (1 + r).cumprod()
    drawdown = (cum - cum.cummax()) / cum.cummax()
    vol = r.std() * np.sqrt(CRYPTO_TRADING_DAYS)
    downside = r[r < 0].std() * np.sqrt(CRYPTO_TRADING_DAYS)
    annual_ret = r.mean() * CRYPTO_TRADING_DAYS
    metrics[name] = {
        "Annualized Return (%)": annual_ret * 100,
        "Annualized Volatility (%)": vol * 100,
        "Sharpe": annual_ret / vol if vol else np.nan,
        "Sortino": annual_ret / downside if downside else np.nan,
        "Max Drawdown (%)": drawdown.min() * 100,
        "VaR 95% (1d, %)": -np.percentile(r, 5) * 100,
    }
if metrics:
    metrics_df = pd.DataFrame(metrics).T
    st.dataframe(metrics_df.style.format("{:.2f}"), use_container_width=True)

# --- Drawdown timeline ---
st.subheader("Drawdown Timeline")
cum_all = (1 + returns.fillna(0)).cumprod()
drawdowns = (cum_all - cum_all.cummax()) / cum_all.cummax() * 100
dd_fig = px.line(drawdowns, x=drawdowns.index, y=drawdowns.columns,
                 labels={"x": "Date", "value": "Drawdown (%)"},
                 title="Drawdown from Running Peak")
st.plotly_chart(dd_fig, use_container_width=True)

# --- ETH/BTC relative strength ---
if {"Bitcoin", "Ethereum"}.issubset(prices.columns):
    st.subheader("ETH/BTC Relative Strength")
    st.write("A rising ratio means Ethereum outperforming Bitcoin — a common gauge of "
             "risk appetite within crypto (\"alt season\" when alts outpace BTC).")
    ratio = (prices["Ethereum"] / prices["Bitcoin"]).dropna()
    ratio_fig = px.line(x=ratio.index, y=ratio, labels={"x": "Date", "y": "ETH/BTC"},
                        title="ETH/BTC Price Ratio")
    st.plotly_chart(ratio_fig, use_container_width=True)

# --- Rolling volatility ---
st.subheader("30-Day Rolling Volatility (annualized)")
rolling_vol = returns.rolling(30).std() * np.sqrt(CRYPTO_TRADING_DAYS) * 100
vol_fig = px.line(rolling_vol, x=rolling_vol.index, y=rolling_vol.columns,
                  labels={"x": "Date", "value": "Volatility (%)"})
st.plotly_chart(vol_fig, use_container_width=True)

# --- Correlations within crypto ---
if len(returns.columns) > 1:
    st.subheader("Correlations Within Crypto")
    corr_fig = px.imshow(returns.corr(method="pearson"), text_auto=".2f",
                         title="Daily Return Correlation Matrix")
    st.plotly_chart(corr_fig, use_container_width=True)

# --- Cross-asset correlation: the diversifier question ---
st.subheader("Correlation vs Traditional Assets")
st.write("Rolling correlation with equities and gold, on overlapping trading days. "
         "Near zero supports crypto as a diversifier; persistently high correlation "
         "with equities means it trades like a risk asset.")
benchmarks = get_price_data(["^GSPC", "GC=F"], start_date.isoformat(), end_date.isoformat())
if benchmarks.empty:
    st.info("Benchmark data (S&P 500, gold) is unavailable right now.")
else:
    benchmarks = benchmarks.rename(columns={"^GSPC": "S&P 500", "GC=F": "Gold"})
    bench_returns = benchmarks.pct_change().dropna(how="all")
    rows = []
    for name in returns.columns:
        row = {"Asset": name}
        for bench in bench_returns.columns:
            joined = pd.concat([returns[name], bench_returns[bench]], axis=1, join="inner").dropna()
            row[f"Corr vs {bench}"] = (
                joined.iloc[:, 0].corr(joined.iloc[:, 1]) if len(joined) > 2 else np.nan
            )
        rows.append(row)
    corr_table = pd.DataFrame(rows).set_index("Asset")
    st.dataframe(corr_table.style.format("{:.2f}"), use_container_width=True)
