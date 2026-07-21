import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from common.data import get_benchmark_data, get_price_data, load_companies_csv
from common.sidebar import render_sidebar

st.title("📊 Performance Analytics")

symbols, start_date, end_date, period, benchmark_symbol, benchmark_name = render_sidebar()

df = get_price_data(symbols, start_date, end_date)
benchmark_series = get_benchmark_data(benchmark_symbol, start_date, end_date)

# Load actual data
df_companies = load_companies_csv()

# FILTER by symbols
if "Symbol" in df_companies.columns and symbols:
    df_companies = df_companies[df_companies["Symbol"].isin(symbols)]

# Market Cap Distribution Histogram (using actual data)
st.subheader("Market Cap Distribution")
fig_hist = px.histogram(df_companies, x="marketcap", nbins=50,
                        title="Distribution of Market Caps",
                        labels={"marketcap": "Market Cap"})
st.plotly_chart(fig_hist, use_container_width=True)

# Top 5 Companies Contribution by Market Cap
st.subheader("Top 5 Contributors by Market Cap")
top5 = df_companies.sort_values("marketcap", ascending=False).head(5)
fig_top5 = px.bar(top5, x="Name", y="marketcap", title="Top 5 Companies Contribution",
                  labels={"marketcap": "Market Cap"})
st.plotly_chart(fig_top5, use_container_width=True)

if df.empty or benchmark_series.empty:
    st.warning("No price data available for the selected stocks, benchmark, and date range. "
               "Please adjust your selection and try again.")
    st.stop()

# --- Time-Series Charts (real portfolio and benchmark data) ---
st.subheader(f"Cumulative Returns vs {benchmark_name}")
stock_daily = df.pct_change().dropna()
portfolio_daily = stock_daily.mean(axis=1)
benchmark_daily = benchmark_series.pct_change().dropna()

portfolio_cum = (1 + portfolio_daily).cumprod()
benchmark_cum = (1 + benchmark_daily).cumprod()
portfolio_cum.index = pd.to_datetime(portfolio_cum.index)

df_returns = pd.concat([
    portfolio_cum.rename("Portfolio"),
    benchmark_cum.rename(benchmark_name),
], axis=1, join="inner").dropna()

fig_returns = px.line(df_returns, x=df_returns.index, y=df_returns.columns,
                      title="Cumulative Returns", labels={"x": "Date", "value": "Growth of 1"})
st.plotly_chart(fig_returns, use_container_width=True)

# --- Performance Metrics Calculation ---
aligned = pd.concat([
    portfolio_daily.rename("portfolio"),
    benchmark_daily.rename("benchmark"),
], axis=1, join="inner").dropna()

annual_return = aligned["portfolio"].mean() * 252 * 100
annual_vol = aligned["portfolio"].std() * np.sqrt(252) * 100
benchmark_annual_return = aligned["benchmark"].mean() * 252 * 100

benchmark_var = aligned["benchmark"].var()
beta = aligned["portfolio"].cov(aligned["benchmark"]) / benchmark_var if benchmark_var else np.nan
alpha = annual_return - beta * benchmark_annual_return if pd.notnull(beta) else np.nan
downside_std = aligned["portfolio"][aligned["portfolio"] < 0].std() * np.sqrt(252)
sortino = (annual_return / 100) / downside_std if downside_std else np.nan
treynor = (annual_return / 100) / beta if beta else np.nan

# --- Key Performance Metrics ---
st.subheader("Key Performance Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Alpha", f"{alpha:.2f}", help="Annualized excess return over the benchmark (CAPM), in %.")
col2.metric("Beta", f"{beta:.2f}", help="Sensitivity of the portfolio to benchmark moves.")
col3.metric("Sortino Ratio", f"{sortino:.2f}", help="Return per unit of downside volatility.")
col4.metric("Treynor Ratio", f"{treynor:.2f}", help="Return per unit of market (beta) risk.")

# --- Contribution Analysis (per-stock contribution to portfolio return) ---
st.subheader("Top Contributors / Detractors")
# Equal-weighted portfolio: each stock contributes its cumulative return / N
stock_cum_returns = ((1 + stock_daily).cumprod().iloc[-1] - 1) * 100
contribution = (stock_cum_returns / len(stock_daily.columns)).sort_values(ascending=False)
df_contrib = contribution.reset_index()
df_contrib.columns = ["Asset", "Contribution"]
fig_contrib = px.bar(df_contrib, x="Asset", y="Contribution",
                     title="Contribution to Portfolio Return (%, equal-weighted)")
st.plotly_chart(fig_contrib, use_container_width=True)
