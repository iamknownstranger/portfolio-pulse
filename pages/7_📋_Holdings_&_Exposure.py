import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf
from common.sidebar import render_sidebar


st.set_page_config(page_title="Holdings & Exposure", page_icon="📋", layout="wide")
st.title("📋 Holdings & Exposure")

symbols, start_date, end_date, period, benchmark_symbol, benchmark_name = render_sidebar()


# Load actual holdings data from CSV
df_holdings = pd.read_csv("data/companiesmarketcap.com - Largest American companies by market capitalization.csv")
df_holdings["marketcap"] = pd.to_numeric(df_holdings["marketcap"], errors='coerce')

# FILTER by symbols if available
if "Symbol" in df_holdings.columns and symbols:
    df_holdings = df_holdings[df_holdings["Symbol"].isin(symbols)]

# --- New: Key Holdings Metrics ---
total_market_cap = df_holdings["marketcap"].sum()
avg_market_cap = df_holdings["marketcap"].mean()
st.subheader("Key Holdings Metrics")
col1, col2 = st.columns(2)
col1.metric("Total Market Cap", f"${total_market_cap:,.0f}")
col2.metric("Avg Market Cap", f"${avg_market_cap:,.0f}")

# Display Holdings Table (show top 10 for brevity)
st.subheader("Your Holdings")
st.dataframe(df_holdings.head(10), hide_index=True)

# Geographical Exposure based on the 'country' column
st.subheader("Geographical Exposure")
geo = df_holdings["country"].value_counts().reset_index()
geo.columns = ["Country", "Count"]
fig_geo = px.pie(geo, names="Country", values="Count", title="Geographical Exposure")
st.plotly_chart(fig_geo, use_container_width=True)

# Market Cap Distribution by Category
st.subheader("Market Cap Distribution")
# Define categories from quantiles of marketcap
quantiles = df_holdings["marketcap"].quantile([0.33, 0.66]).values
def cap_category(x):
    if x < quantiles[0]:
        return "Small Cap"
    elif x < quantiles[1]:
        return "Mid Cap"
    else:
        return "Large Cap"
df_holdings["Category"] = df_holdings["marketcap"].apply(cap_category)
cap_dist = df_holdings["Category"].value_counts().reset_index()
cap_dist.columns = ["Category", "Count"]
fig_caps = px.bar(cap_dist, x="Category", y="Count", title="Market Cap Distribution")
st.plotly_chart(fig_caps, use_container_width=True)

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

