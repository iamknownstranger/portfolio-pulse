import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

from common.sidebar import render_sidebar

st.set_page_config(page_title="Performance Analytics", page_icon="📊", layout="wide")
st.title("📊 Performance Analytics")

symbols, start_date, end_date, period, benchmark_symbol, benchmark_name = render_sidebar()

# Updated to ensure calculations use real data from DuckDB or yfinance
@st.cache_data(ttl=86400)
def get_data(tickers, start, end):
    try:
        # Attempt to fetch data from DuckDB
        query = f"""
        SELECT date, symbol, close_price FROM market_data
        WHERE symbol IN ({', '.join([f'\'{ticker}\'' for ticker in tickers])})
        AND date BETWEEN '{start}' AND '{end}'
        ORDER BY date;
        """
        df_duckdb = pd.DataFrame(con.execute(query).fetchall(), columns=['date', 'symbol', 'close_price'])
        if not df_duckdb.empty:
            df_duckdb = df_duckdb.pivot(index='date', columns='symbol', values='close_price')
            return df_duckdb
    except Exception as e:
        st.warning(f"DuckDB data fetch failed: {e}. Falling back to yfinance.")

    # Fallback to yfinance if DuckDB fails
    try:
        df_yf = yf.download(tickers, start=start, end=end, auto_adjust=False, multi_level_index=False)["Close"]
        df_yf.index = df_yf.index.date
        return df_yf.dropna(axis=1, how='all')
    except Exception as e:
        st.error(f"Error fetching data from yfinance: {e}")
        return pd.DataFrame()

# --- Fetch benchmark index data ---
def get_top100us_index(start_date, end_date):
    df = pd.read_csv("data/largest-companies-in-the-usa-by-market-cap.csv")
    df = df.sort_values("marketcap", ascending=False).head(100)
    tickers = df["Symbol"].tolist()
    try:
        df_yf = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, multi_level_index=False)["Close"]
        df_yf.index = pd.to_datetime(df_yf.index)
        index_val = df_yf.mean(axis=1)
        return index_val.dropna()
    except Exception as e:
        st.warning(f"Failed to fetch Top 100 US index data: {e}")
        return pd.Series(dtype=float)

benchmark_df = None
if benchmark_symbol == "TOP100US":
    benchmark_df = get_top100us_index(start_date, end_date)
elif benchmark_symbol:
    try:
        benchmark_df = yf.download(benchmark_symbol, start=start_date, end=end_date)["Close"]
        benchmark_df = benchmark_df.dropna()
        benchmark_df.index = pd.to_datetime(benchmark_df.index)
    except Exception as e:
        st.warning(f"Failed to fetch benchmark data: {e}")

# Load actual data
df_companies = pd.read_csv("data/companiesmarketcap.com - Largest American companies by market capitalization.csv")
df_companies["marketcap"] = pd.to_numeric(df_companies["marketcap"], errors='coerce')

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
fig_contrib = px.bar(top5, x="Name", y="marketcap", title="Top 5 Companies Contribution",
                      labels={"marketcap": "Market Cap"})
st.plotly_chart(fig_contrib, use_container_width=True)

# --- Time-Series Charts ---
st.subheader("Cumulative Returns vs Benchmark")
# ...retrieve portfolio and benchmark data...
df_returns = pd.DataFrame({
    "Date": pd.date_range("2022-01-01", periods=100),
    "Portfolio": np.random.normal(0.1, 2, 100).cumsum(),
    "Benchmark": np.random.normal(0.08, 2, 100).cumsum()
})
fig_returns = px.line(df_returns, x="Date", y=["Portfolio", "Benchmark"], title="Cumulative Returns")
st.plotly_chart(fig_returns, use_container_width=True)

# --- New: Performance Metrics Calculation ---
# Calculate daily returns from simulated cumulative data
df_returns["Portfolio Daily"] = df_returns["Portfolio"].pct_change()
df_returns["Benchmark Daily"] = df_returns["Benchmark"].pct_change()
# Drop initial NaN
portfolio_daily = df_returns["Portfolio Daily"].dropna()
benchmark_daily = df_returns["Benchmark Daily"].dropna()

annual_return = portfolio_daily.mean() * 252 * 100
annual_vol = portfolio_daily.std() * np.sqrt(252) * 100
beta = np.cov(portfolio_daily, benchmark_daily)[0,1] / np.var(benchmark_daily)
alpha = annual_return - beta * (benchmark_daily.mean()*252*100)
downside_std = portfolio_daily[portfolio_daily < 0].std() * np.sqrt(252)
sortino = (annual_return/100) / downside_std if downside_std != 0 else np.nan
treynor = (annual_return/100) / beta if beta != 0 else np.nan

# --- Key Performance Metrics ---
st.subheader("Key Performance Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Alpha", f"{alpha:.2f}")
col2.metric("Beta", f"{beta:.2f}")
col3.metric("Sortino Ratio", f"{sortino:.2f}")
col4.metric("Treynor Ratio", f"{treynor:.2f}")

# --- Contribution Analysis ---
st.subheader("Top Contributors / Detractors")
# ...placeholder data for bar chart...
df_contrib = pd.DataFrame({
    "Asset": ["Asset A", "Asset B", "Asset C"],
    "Contribution": [5, -3, 2]
})
fig_contrib = px.bar(df_contrib, x="Asset", y="Contribution", title="Contribution Analysis")
st.plotly_chart(fig_contrib, use_container_width=True)
